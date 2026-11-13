from collections.abc import AsyncIterator
from datetime import datetime

import httpx
import pytest
from mocks.apis.app import app
from mocks.apis.data import build_data


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as api_client:
        yield api_client


@pytest.mark.parametrize(
    ("path", "expected_fields"),
    [
        (
            "/oms/orders/ORD-0001",
            {"order_id", "customer_id", "status", "ordered_at", "currency", "total", "items"},
        ),
        (
            "/shipping/shipments/ORD-0001",
            {
                "shipment_id",
                "order_id",
                "carrier",
                "tracking_number",
                "status",
                "estimated_delivery",
            },
        ),
        ("/crm/customers/CUST-001", {"customer_id", "name", "email", "tier", "country"}),
        ("/policies/returns", {"policy_name", "version", "description", "rules"}),
        (
            "/inventory/products/SKU-0001",
            {"sku", "name", "available", "warehouse", "unit_price"},
        ),
    ],
)
async def test_endpoint_schema_and_success(
    client: httpx.AsyncClient, path: str, expected_fields: set[str]
) -> None:
    response = await client.get(path)

    assert response.status_code == 200
    assert set(response.json()) == expected_fields
    assert datetime.fromisoformat(response.headers["X-Mock-Served-At"]).tzinfo is not None


@pytest.mark.parametrize(
    "path",
    [
        "/oms/orders/unknown",
        "/shipping/shipments/unknown",
        "/crm/customers/unknown",
        "/policies/unknown",
        "/inventory/products/unknown",
    ],
)
async def test_unknown_ids_return_404(client: httpx.AsyncClient, path: str) -> None:
    response = await client.get(path)

    assert response.status_code == 404
    assert response.headers["X-Mock-Served-At"]


def test_fixed_seed_builds_identical_data() -> None:
    first = build_data(42)
    second = build_data(42)

    assert first == second
    assert len(first.orders) == 200
    assert len(first.customers) == 100
