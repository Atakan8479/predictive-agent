"""FastAPI application exposing deterministic mock external services."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request, Response

from mocks.apis.data import DATA, Customer, Order, Policy, Product, Shipment

app = FastAPI(title="Predictive Agent Mock APIs", version="1.0.0")


@app.middleware("http")
async def add_served_at_header(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    response = await call_next(request)
    response.headers["X-Mock-Served-At"] = datetime.now(UTC).isoformat()
    return response


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/oms/orders/{order_id}", response_model=Order)
def get_order(order_id: str) -> Order:
    if (order := DATA.orders.get(order_id)) is None:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@app.get("/shipping/shipments/{order_id}", response_model=Shipment)
def get_shipment(order_id: str) -> Shipment:
    if (shipment := DATA.shipments.get(order_id)) is None:
        raise HTTPException(status_code=404, detail="Shipment not found")
    return shipment


@app.get("/crm/customers/{customer_id}", response_model=Customer)
def get_customer(customer_id: str) -> Customer:
    if (customer := DATA.customers.get(customer_id)) is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@app.get("/policies/{policy_name}", response_model=Policy)
def get_policy(policy_name: str) -> Policy:
    if (policy := DATA.policies.get(policy_name)) is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy


@app.get("/inventory/products/{sku}", response_model=Product)
def get_product(sku: str) -> Product:
    if (product := DATA.products.get(sku)) is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product
