"""Fixed-seed datasets for the external API mocks."""

from dataclasses import dataclass
from datetime import date, timedelta
from random import Random

from pydantic import BaseModel

SEED = 20261113


class OrderItem(BaseModel):
    sku: str
    quantity: int
    unit_price: float


class Order(BaseModel):
    order_id: str
    customer_id: str
    status: str
    ordered_at: date
    currency: str
    total: float
    items: list[OrderItem]


class Shipment(BaseModel):
    shipment_id: str
    order_id: str
    carrier: str
    tracking_number: str
    status: str
    estimated_delivery: date


class Customer(BaseModel):
    customer_id: str
    name: str
    email: str
    tier: str
    country: str


class Policy(BaseModel):
    policy_name: str
    version: str
    description: str
    rules: list[str]


class Product(BaseModel):
    sku: str
    name: str
    available: int
    warehouse: str
    unit_price: float


@dataclass(frozen=True)
class MockData:
    orders: dict[str, Order]
    shipments: dict[str, Shipment]
    customers: dict[str, Customer]
    policies: dict[str, Policy]
    products: dict[str, Product]


def build_data(seed: int = SEED) -> MockData:
    rng = Random(seed)
    first_names = ("Ada", "Ece", "Mert", "Deniz", "Selin", "Can", "Elif", "Arda")
    last_names = ("Yilmaz", "Kaya", "Demir", "Celik", "Sahin", "Aydin")
    countries = ("TR", "DE", "GB", "NL", "US")

    customers = {
        customer_id: Customer(
            customer_id=customer_id,
            name=f"{rng.choice(first_names)} {rng.choice(last_names)}",
            email=f"customer{index:03d}@example.test",
            tier=rng.choice(("standard", "silver", "gold")),
            country=rng.choice(countries),
        )
        for index in range(1, 101)
        if (customer_id := f"CUST-{index:03d}")
    }

    products = {
        sku: Product(
            sku=sku,
            name=f"Product {index:03d}",
            available=rng.randint(0, 250),
            warehouse=rng.choice(("IST-1", "ANK-1", "IZM-1")),
            unit_price=round(rng.uniform(5, 500), 2),
        )
        for index in range(1, 101)
        if (sku := f"SKU-{index:04d}")
    }

    orders: dict[str, Order] = {}
    shipments: dict[str, Shipment] = {}
    product_ids = tuple(products)
    customer_ids = tuple(customers)
    base_date = date(2026, 1, 1)
    for index in range(1, 201):
        order_id = f"ORD-{index:04d}"
        ordered_at = base_date + timedelta(days=rng.randint(0, 89))
        items = [
            OrderItem(
                sku=sku,
                quantity=rng.randint(1, 4),
                unit_price=products[sku].unit_price,
            )
            for sku in rng.sample(product_ids, rng.randint(1, 4))
        ]
        orders[order_id] = Order(
            order_id=order_id,
            customer_id=rng.choice(customer_ids),
            status=rng.choice(("processing", "paid", "shipped", "delivered")),
            ordered_at=ordered_at,
            currency="USD",
            total=round(sum(item.quantity * item.unit_price for item in items), 2),
            items=items,
        )
        shipments[order_id] = Shipment(
            shipment_id=f"SHP-{index:04d}",
            order_id=order_id,
            carrier=rng.choice(("DHL", "FedEx", "UPS")),
            tracking_number=f"TRK{seed}{index:06d}",
            status=rng.choice(("label_created", "in_transit", "delivered")),
            estimated_delivery=ordered_at + timedelta(days=rng.randint(2, 8)),
        )

    policies = {
        policy.policy_name: policy
        for policy in (
            Policy(
                policy_name="returns",
                version="2026-01",
                description="Standard return policy.",
                rules=["Returns accepted within 30 days.", "Items must be unused."],
            ),
            Policy(
                policy_name="cancellations",
                version="2026-01",
                description="Order cancellation policy.",
                rules=[
                    "Processing orders may be cancelled.",
                    "Shipped orders cannot be cancelled.",
                ],
            ),
            Policy(
                policy_name="shipping",
                version="2026-01",
                description="Standard shipping policy.",
                rules=[
                    "Tracking is provided after dispatch.",
                    "Delivery estimates are not guarantees.",
                ],
            ),
        )
    }
    return MockData(orders, shipments, customers, policies, products)


DATA = build_data()
