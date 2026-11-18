"""The five controlled benchmark tool definitions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from predictive_agent.tools.registry import CostClass, ToolDependency, ToolRegistry, ToolSpec


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GetOrderArgs(ToolArguments):
    order_id: str = Field(pattern=r"^ORD-\d{4}$")


class GetShipmentArgs(ToolArguments):
    order_id: str = Field(pattern=r"^ORD-\d{4}$")


class GetCustomerArgs(ToolArguments):
    customer_id: str = Field(pattern=r"^CUST-\d{3}$")


class GetPolicyArgs(ToolArguments):
    policy_name: Literal["returns", "shipping", "cancellations"]


class GetInventoryArgs(ToolArguments):
    sku: str = Field(pattern=r"^SKU-\d{4}$")


TOOL_SPECS = (
    ToolSpec(
        name="get_order",
        description="Retrieve an order by its ORD-#### identifier.",
        args_model=GetOrderArgs,
        read_only=True,
        idempotent=True,
        freshness_ttl_s=30.0,
        auth_scope="orders:read",
        cost_class=CostClass.MEDIUM,
    ),
    ToolSpec(
        name="get_shipment",
        description="Retrieve shipment tracking for an order by its ORD-#### identifier.",
        args_model=GetShipmentArgs,
        read_only=True,
        idempotent=True,
        freshness_ttl_s=10.0,
        auth_scope="shipments:read",
        depends_on=(ToolDependency(tool_name="get_order", provides_arguments=("order_id",)),),
        cost_class=CostClass.MEDIUM,
    ),
    ToolSpec(
        name="get_customer",
        description="Retrieve a customer by its CUST-### identifier.",
        args_model=GetCustomerArgs,
        read_only=True,
        idempotent=True,
        freshness_ttl_s=300.0,
        auth_scope="customers:read",
        cost_class=CostClass.LOW,
    ),
    ToolSpec(
        name="get_policy",
        description="Retrieve the returns, shipping, or cancellations policy.",
        args_model=GetPolicyArgs,
        read_only=True,
        idempotent=True,
        freshness_ttl_s=3600.0,
        auth_scope="policies:read",
        cost_class=CostClass.LOW,
    ),
    ToolSpec(
        name="get_inventory",
        description="Retrieve inventory for a product by its SKU-#### identifier.",
        args_model=GetInventoryArgs,
        read_only=True,
        idempotent=True,
        freshness_ttl_s=10.0,
        auth_scope="inventory:read",
        cost_class=CostClass.MEDIUM,
    ),
)

TOOL_REGISTRY = ToolRegistry(TOOL_SPECS)


def build_tool_registry() -> ToolRegistry:
    """Return a fresh registry with the authoritative immutable specifications."""
    return ToolRegistry(TOOL_SPECS)
