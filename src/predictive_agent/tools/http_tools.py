"""Authorized asynchronous HTTP execution for registered tools."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import date
from typing import Any, Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from predictive_agent.config import Settings
from predictive_agent.tools.definitions import build_tool_registry
from predictive_agent.tools.registry import (
    ExecutionContext,
    ToolDeadlineExceeded,
    ToolExecutor,
    ToolRegistry,
)


class ToolHTTPError(Exception):
    """Base class for sanitized external API failures."""


class ToolConnectionError(ToolHTTPError):
    """The configured external API could not be reached."""


class ToolNotFoundError(ToolHTTPError):
    """The requested fixture resource does not exist."""


class ToolHTTPStatusError(ToolHTTPError):
    """The external API returned an unexpected status."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"external tool API returned HTTP {status_code}")
        self.status_code = status_code


class ToolResponseError(ToolHTTPError):
    """The external API response was not valid JSON in the expected shape."""


class ToolHTTPTimeoutError(ToolHTTPError, TimeoutError):
    """HTTPX timed out independently before the absolute registry deadline."""


class ToolClientClosedError(ToolHTTPError, RuntimeError):
    """Execution was attempted through a closed HTTP tool client."""


class ToolBindingError(ToolHTTPError, ValueError):
    """A registered tool has no supported HTTP binding."""


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class OrderItem(_ResponseModel):
    sku: str = Field(pattern=r"^SKU-\d{4}$")
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0, allow_inf_nan=False)


class OrderResult(_ResponseModel):
    order_id: str = Field(pattern=r"^ORD-\d{4}$")
    customer_id: str = Field(pattern=r"^CUST-\d{3}$")
    status: str
    ordered_at: date
    currency: str
    total: float = Field(ge=0, allow_inf_nan=False)
    items: list[OrderItem]


class ShipmentResult(_ResponseModel):
    shipment_id: str = Field(pattern=r"^SHP-\d{4}$")
    order_id: str = Field(pattern=r"^ORD-\d{4}$")
    carrier: str
    tracking_number: str
    status: str
    estimated_delivery: date


class CustomerResult(_ResponseModel):
    customer_id: str = Field(pattern=r"^CUST-\d{3}$")
    name: str
    email: str
    tier: str
    country: str


class PolicyResult(_ResponseModel):
    policy_name: Literal["returns", "shipping", "cancellations"]
    version: str
    description: str
    rules: list[str]


class InventoryResult(_ResponseModel):
    sku: str = Field(pattern=r"^SKU-\d{4}$")
    name: str
    available: int = Field(ge=0)
    warehouse: str
    unit_price: float = Field(ge=0, allow_inf_nan=False)


_ENDPOINTS: dict[str, tuple[str, str]] = {
    "get_order": ("/oms/orders/{value}", "order_id"),
    "get_shipment": ("/shipping/shipments/{value}", "order_id"),
    "get_customer": ("/crm/customers/{value}", "customer_id"),
    "get_policy": ("/policies/{value}", "policy_name"),
    "get_inventory": ("/inventory/products/{value}", "sku"),
}
_RESULT_MODELS: dict[str, type[_ResponseModel]] = {
    "get_order": OrderResult,
    "get_shipment": ShipmentResult,
    "get_customer": CustomerResult,
    "get_policy": PolicyResult,
    "get_inventory": InventoryResult,
}


class HTTPToolClient:
    """Own one pooled HTTP client and expose the registry's guarded execution path."""

    def __init__(
        self,
        settings: Settings,
        *,
        registry: ToolRegistry | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        base_registry = registry or build_tool_registry()
        unsupported = set(base_registry.names()).difference(_ENDPOINTS)
        if unsupported:
            raise ToolBindingError(f"no HTTP binding for registered tools: {sorted(unsupported)!r}")
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=settings.tool_api_base_url,
            follow_redirects=False,
            timeout=settings.http_timeout_s,
        )
        executors: dict[str, ToolExecutor] = {
            name: self._executor_for(name) for name in base_registry.names()
        }
        self.registry = base_registry.with_executors(executors)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> HTTPToolClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close an internally created client; injected clients retain caller ownership."""
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    def deadline_after(self, timeout_s: float | None = None) -> float:
        timeout = self.settings.http_timeout_s if timeout_s is None else timeout_s
        if timeout <= 0:
            raise ValueError("timeout_s must be positive")
        return asyncio.get_running_loop().time() + timeout

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        context: ExecutionContext | None,
        deadline: float,
    ) -> dict[str, Any]:
        self._ensure_open()
        return await self.registry.execute(name, arguments, context=context, deadline=deadline)

    def _executor_for(self, name: str) -> ToolExecutor:
        async def execute(
            arguments: BaseModel, context: ExecutionContext, deadline: float
        ) -> dict[str, Any]:
            self._ensure_open()
            del context  # Authorization is completed by ToolRegistry before this boundary.
            return await self._request(name, arguments, deadline)

        return execute

    async def _request(self, name: str, arguments: BaseModel, deadline: float) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise ToolDeadlineExceeded(f"deadline expired before executing {name!r}")
        template, argument_name = _ENDPOINTS[name]
        value = getattr(arguments, argument_name)
        path = template.format(value=quote(value, safe=""))
        try:
            response = await self._client.get(path, timeout=remaining, follow_redirects=False)
        except httpx.TimeoutException as error:
            raise ToolHTTPTimeoutError(f"HTTP timeout while executing {name!r}") from error
        except httpx.RequestError as error:
            raise ToolConnectionError("external tool API connection failed") from error

        if response.status_code == 404:
            raise ToolNotFoundError(f"resource for {name!r} was not found")
        if response.status_code != 200:
            raise ToolHTTPStatusError(response.status_code)
        try:
            result = _RESULT_MODELS[name].model_validate_json(response.content, strict=True)
        except ValidationError as error:
            raise ToolResponseError(
                "external tool API returned invalid JSON or response data"
            ) from error
        if getattr(result, argument_name) != value:
            raise ToolResponseError("external tool API returned a mismatched resource")
        return result.model_dump(mode="json")

    def _ensure_open(self) -> None:
        if self._closed:
            raise ToolClientClosedError("HTTP tool client is closed")
