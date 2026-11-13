from collections.abc import AsyncIterator
from datetime import datetime

import httpx
import pytest
from mocks.apis.app import app
from mocks.apis.data import (
    DATA,
    ORDER_TO_SHIPMENT_STATUS,
    Customer,
    Order,
    Policy,
    Product,
    Shipment,
    build_data,
)
from pydantic import BaseModel


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as api_client:
        yield api_client


@pytest.mark.parametrize(
    ("path", "schema", "expected_fields"),
    [
        (
            "/oms/orders/ORD-0001",
            Order,
            {"order_id", "customer_id", "status", "ordered_at", "currency", "total", "items"},
        ),
        (
            "/shipping/shipments/ORD-0001",
            Shipment,
            {
                "shipment_id",
                "order_id",
                "carrier",
                "tracking_number",
                "status",
                "estimated_delivery",
            },
        ),
        (
            "/crm/customers/CUST-001",
            Customer,
            {"customer_id", "name", "email", "tier", "country"},
        ),
        ("/policies/returns", Policy, {"policy_name", "version", "description", "rules"}),
        (
            "/inventory/products/SKU-0001",
            Product,
            {"sku", "name", "available", "warehouse", "unit_price"},
        ),
    ],
)
async def test_endpoint_schema_and_success(
    client: httpx.AsyncClient,
    path: str,
    schema: type[BaseModel],
    expected_fields: set[str],
) -> None:
    response = await client.get(path)

    assert response.status_code == 200
    assert set(response.json()) == expected_fields
    schema.model_validate(response.json())
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


def test_default_dataset_has_expected_record_counts() -> None:
    assert len(DATA.orders) == 200
    assert len(DATA.shipments) == 200
    assert len(DATA.customers) == 100
    assert len(DATA.products) == 100
    assert len(DATA.policies) == 3


def test_every_order_and_shipment_satisfies_lifecycle_invariants() -> None:
    assert DATA.orders.keys() == DATA.shipments.keys()

    for order_id, order in DATA.orders.items():
        shipment = DATA.shipments[order_id]

        assert shipment.order_id == order_id
        assert shipment.status == ORDER_TO_SHIPMENT_STATUS[order.status]
        assert not (order.status in {"processing", "paid"} and shipment.status == "delivered")
        if order.status == "delivered":
            assert shipment.status == "delivered"

    for shipment in DATA.shipments.values():
        assert shipment.order_id in DATA.orders
