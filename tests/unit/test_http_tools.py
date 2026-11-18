import asyncio
import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from predictive_agent.config import Settings
from predictive_agent.tools.http_tools import (
    HTTPToolClient,
    ToolBindingError,
    ToolClientClosedError,
    ToolConnectionError,
    ToolHTTPStatusError,
    ToolHTTPTimeoutError,
    ToolNotFoundError,
    ToolResponseError,
)
from predictive_agent.tools.registry import (
    CostClass,
    ExecutionContext,
    ToolArgumentsError,
    ToolAuthorizationError,
    ToolDeadlineExceeded,
    ToolRegistry,
    ToolSpec,
)

SCOPES = frozenset(
    {
        "orders:read",
        "shipments:read",
        "customers:read",
        "policies:read",
        "inventory:read",
    }
)
CONTEXT = ExecutionContext(SCOPES, subject="fixture-user", tenant_id="fixture-tenant")


def _settings() -> Settings:
    return Settings(tool_api_base_url="http://tools.test", http_timeout_s=2)


def _payload(path: str) -> dict[str, Any]:
    if path.startswith("/oms/orders/"):
        return {
            "order_id": "ORD-0042",
            "customer_id": "CUST-073",
            "status": "shipped",
            "ordered_at": "2026-01-02",
            "currency": "USD",
            "total": 12.5,
            "items": [{"sku": "SKU-0007", "quantity": 1, "unit_price": 12.5}],
        }
    if path.startswith("/shipping/shipments/"):
        return {
            "shipment_id": "SHP-0042",
            "order_id": "ORD-0042",
            "carrier": "DHL",
            "tracking_number": "TRK42",
            "status": "in_transit",
            "estimated_delivery": "2026-01-05",
        }
    if path.startswith("/crm/customers/"):
        return {
            "customer_id": "CUST-073",
            "name": "Ada Kaya",
            "email": "ada@example.test",
            "tier": "gold",
            "country": "TR",
        }
    if path.startswith("/policies/"):
        return {
            "policy_name": "returns",
            "version": "2026-01",
            "description": "Returns.",
            "rules": ["Within 30 days."],
        }
    return {
        "sku": "SKU-0007",
        "name": "Product 007",
        "available": 4,
        "warehouse": "IST-1",
        "unit_price": 12.5,
    }


@pytest.mark.parametrize(
    ("name", "arguments", "expected_path"),
    [
        ("get_order", {"order_id": "ORD-0042"}, "/oms/orders/ORD-0042"),
        ("get_shipment", {"order_id": "ORD-0042"}, "/shipping/shipments/ORD-0042"),
        ("get_customer", {"customer_id": "CUST-073"}, "/crm/customers/CUST-073"),
        ("get_policy", {"policy_name": "returns"}, "/policies/returns"),
        ("get_inventory", {"sku": "SKU-0007"}, "/inventory/products/SKU-0007"),
    ],
)
async def test_all_tools_use_trusted_get_endpoints(
    name: str, arguments: dict[str, str], expected_path: str
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_payload(request.url.path))

    injected = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://tools.test"
    )
    client = HTTPToolClient(_settings(), http_client=injected)
    result = await client.execute(
        name, arguments, context=CONTEXT, deadline=client.deadline_after()
    )
    await client.close()

    assert result
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == expected_path
    assert not injected.is_closed
    await injected.aclose()


async def test_invalid_and_unauthorized_calls_make_no_request() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(500)

    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://tools.test"
        ) as injected,
        HTTPToolClient(_settings(), http_client=injected) as client,
    ):
        deadline = client.deadline_after()
        with pytest.raises(ToolArgumentsError):
            await client.execute(
                "get_order", {"order_id": "../secret"}, context=CONTEXT, deadline=deadline
            )
        with pytest.raises(ToolAuthorizationError):
            await client.execute(
                "get_order",
                {"order_id": "ORD-0042"},
                context=ExecutionContext(frozenset()),
                deadline=deadline,
            )
    assert requests == 0


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(404, json={"detail": "missing"}), ToolNotFoundError),
        (httpx.Response(503, json={"detail": "down"}), ToolHTTPStatusError),
        (httpx.Response(200, content=b"not-json"), ToolResponseError),
        (httpx.Response(200, json={"order_id": "ORD-0042"}), ToolResponseError),
    ],
)
async def test_external_failures_are_typed(
    response: httpx.Response, error: type[Exception]
) -> None:
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: response), base_url="http://tools.test"
        ) as injected,
        HTTPToolClient(_settings(), http_client=injected) as client,
    ):
        with pytest.raises(error):
            await client.execute(
                "get_order",
                {"order_id": "ORD-0042"},
                context=CONTEXT,
                deadline=client.deadline_after(),
            )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("coerced_total", "12.5"),
        ("extra", True),
        ("mismatched_identity", "ORD-0043"),
        ("zero_quantity", 0),
        ("negative_total", -1.0),
    ],
)
async def test_response_contract_rejects_invalid_values(mutation: str, value: object) -> None:
    payload = _payload("/oms/orders/ORD-0042")
    if mutation == "coerced_total":
        payload["total"] = value
    elif mutation == "extra":
        payload["unexpected"] = value
    elif mutation == "mismatched_identity":
        payload["order_id"] = value
    elif mutation == "zero_quantity":
        payload["items"][0]["quantity"] = value
    else:
        payload["total"] = value

    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
            base_url="http://tools.test",
        ) as injected,
        HTTPToolClient(_settings(), http_client=injected) as client,
    ):
        with pytest.raises(ToolResponseError):
            await client.execute(
                "get_order",
                {"order_id": "ORD-0042"},
                context=CONTEXT,
                deadline=client.deadline_after(),
            )


@pytest.mark.parametrize(
    "content",
    [
        b"{not-json",
        b'\xff{"order_id":"ORD-0042"}',
        json.dumps(_payload("/oms/orders/ORD-0042")).replace("12.5", "NaN", 1).encode(),
    ],
)
async def test_response_contract_rejects_malformed_encoding_and_non_finite_json(
    content: bytes,
) -> None:
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content)),
            base_url="http://tools.test",
        ) as injected,
        HTTPToolClient(_settings(), http_client=injected) as client,
    ):
        with pytest.raises(ToolResponseError):
            await client.execute(
                "get_order",
                {"order_id": "ORD-0042"},
                context=CONTEXT,
                deadline=client.deadline_after(),
            )


async def test_response_contract_accepts_legitimate_zero_boundaries() -> None:
    order = _payload("/oms/orders/ORD-0042")
    order["total"] = 0.0
    order["items"][0]["unit_price"] = 0.0
    inventory = _payload("/inventory/products/SKU-0007")
    inventory["available"] = 0
    inventory["unit_price"] = 0.0

    def handler(request: httpx.Request) -> httpx.Response:
        payload = order if request.url.path.startswith("/oms/") else inventory
        return httpx.Response(200, json=payload)

    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://tools.test",
        ) as injected,
        HTTPToolClient(_settings(), http_client=injected) as client,
    ):
        order_result = await client.execute(
            "get_order",
            {"order_id": "ORD-0042"},
            context=CONTEXT,
            deadline=client.deadline_after(),
        )
        inventory_result = await client.execute(
            "get_inventory",
            {"sku": "SKU-0007"},
            context=CONTEXT,
            deadline=client.deadline_after(),
        )
    assert order_result["total"] == 0.0
    assert inventory_result["available"] == 0
    assert inventory_result["unit_price"] == 0.0


@pytest.mark.parametrize("status", [302, 503])
async def test_redirects_and_errors_are_not_followed_or_retried(status: int) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(status, headers={"location": "http://untrusted.test/escape"})

    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://tools.test"
        ) as injected,
        HTTPToolClient(_settings(), http_client=injected) as client,
    ):
        with pytest.raises(ToolHTTPStatusError) as caught:
            await client.execute(
                "get_order",
                {"order_id": "ORD-0042"},
                context=CONTEXT,
                deadline=client.deadline_after(),
            )
    assert caught.value.status_code == status
    assert requests == 1


async def test_connection_failure_is_typed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private detail", request=request)

    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://tools.test"
        ) as injected,
        HTTPToolClient(_settings(), http_client=injected) as client,
    ):
        with pytest.raises(ToolConnectionError, match="connection failed"):
            await client.execute(
                "get_order",
                {"order_id": "ORD-0042"},
                context=CONTEXT,
                deadline=client.deadline_after(),
            )


async def test_independent_http_timeout_remains_distinct_from_registry_deadline() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("independent timeout", request=request)

    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://tools.test"
        ) as injected,
        HTTPToolClient(_settings(), http_client=injected) as client,
    ):
        with pytest.raises(ToolHTTPTimeoutError):
            await client.execute(
                "get_order",
                {"order_id": "ORD-0042"},
                context=CONTEXT,
                deadline=client.deadline_after(),
            )


async def test_immediate_and_in_flight_deadlines_are_not_reset() -> None:
    started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(10)
        return httpx.Response(200, json=_payload(request.url.path))

    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://tools.test"
        ) as injected,
        HTTPToolClient(_settings(), http_client=injected) as client,
    ):
        with pytest.raises(ToolDeadlineExceeded):
            await client.execute("get_order", {"order_id": "ORD-0042"}, context=CONTEXT, deadline=0)
        parent_deadline = asyncio.get_running_loop().time() + 0.02
        with pytest.raises(ToolDeadlineExceeded):
            await client.execute(
                "get_order",
                {"order_id": "ORD-0042"},
                context=CONTEXT,
                deadline=parent_deadline,
            )
        assert started.is_set()


async def test_cancellation_propagates() -> None:
    started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        return httpx.Response(500)

    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://tools.test"
        ) as injected,
        HTTPToolClient(_settings(), http_client=injected) as client,
    ):
        task = asyncio.create_task(
            client.execute(
                "get_order",
                {"order_id": "ORD-0042"},
                context=CONTEXT,
                deadline=client.deadline_after(),
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_internal_client_is_closed_with_owner() -> None:
    client = HTTPToolClient(_settings())
    owned = client._client  # pyright: ignore[reportPrivateUsage]
    await client.close()
    await client.close()
    assert client.closed
    assert owned.is_closed


async def test_all_bound_execution_paths_fail_after_wrapper_closure() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=_payload(request.url.path))

    injected = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://tools.test"
    )
    client = HTTPToolClient(_settings(), http_client=injected)
    deadline = asyncio.get_running_loop().time() + 10
    bound_registry = client.registry
    binding = bound_registry.langgraph_bindings(context=CONTEXT, deadline=deadline)
    order_binding = next(tool for tool in binding if tool.name == "get_order")
    await client.close()
    await client.close()

    with pytest.raises(ToolClientClosedError):
        await client.execute(
            "get_order", {"order_id": "ORD-0042"}, context=CONTEXT, deadline=deadline
        )
    with pytest.raises(ToolClientClosedError):
        await bound_registry.execute(
            "get_order", {"order_id": "ORD-0042"}, context=CONTEXT, deadline=deadline
        )
    with pytest.raises(ToolClientClosedError):
        await order_binding.ainvoke({"order_id": "ORD-0042"})

    assert requests == 0
    assert not injected.is_closed
    await injected.aclose()


def test_unsupported_custom_registry_binding_fails_during_construction() -> None:
    class CustomArgs(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        value: str

    registry = ToolRegistry(
        (
            ToolSpec(
                name="custom_tool",
                description="Unsupported custom HTTP tool.",
                args_model=CustomArgs,
                read_only=True,
                idempotent=True,
                freshness_ttl_s=0.0,
                auth_scope="custom:read",
                cost_class=CostClass.LOW,
            ),
        )
    )
    with pytest.raises(ToolBindingError, match="custom_tool"):
        HTTPToolClient(_settings(), registry=registry)
