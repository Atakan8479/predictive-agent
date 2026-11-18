import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from predictive_agent.tools.definitions import (
    TOOL_REGISTRY,
    GetInventoryArgs,
    GetOrderArgs,
    build_tool_registry,
)
from predictive_agent.tools.registry import (
    CostClass,
    DuplicateToolError,
    ExecutionContext,
    ToolArgumentsError,
    ToolAuthorizationError,
    ToolDeadlineExceeded,
    ToolDependency,
    ToolDependencyError,
    ToolNotBoundError,
    ToolRegistry,
    ToolSpec,
    UnknownToolError,
)


def _spec(
    name: str,
    *,
    read_only: bool = True,
    idempotent: bool = True,
    depends_on: tuple[ToolDependency, ...] = (),
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Run {name}.",
        args_model=GetOrderArgs,
        read_only=read_only,
        idempotent=idempotent,
        freshness_ttl_s=0,
        auth_scope="test:read",
        depends_on=depends_on,
        cost_class=CostClass.LOW,
    )


def test_authoritative_registry_contains_exactly_five_tools_deterministically() -> None:
    assert TOOL_REGISTRY.names() == (
        "get_customer",
        "get_inventory",
        "get_order",
        "get_policy",
        "get_shipment",
    )
    assert TOOL_REGISTRY.names() == build_tool_registry().names()
    assert all(spec.read_only and spec.idempotent for spec in TOOL_REGISTRY.specs())


def test_duplicate_and_unknown_tools_are_explicit() -> None:
    with pytest.raises(DuplicateToolError):
        ToolRegistry((_spec("lookup"), _spec("lookup")))
    with pytest.raises(UnknownToolError):
        TOOL_REGISTRY.get("missing")
    with pytest.raises(UnknownToolError):
        TOOL_REGISTRY.is_prefetch_eligible("missing")


def test_specs_are_frozen_and_invalid_metadata_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TOOL_REGISTRY.get("get_order").freshness_ttl_s = 2  # type: ignore[misc]
    with pytest.raises(ValidationError):
        _spec("Invalid Name")
    with pytest.raises(ValidationError):
        ToolSpec(
            name="lookup",
            description="Run lookup.",
            args_model=GetOrderArgs,
            read_only=True,
            idempotent=True,
            freshness_ttl_s=float("nan"),
            auth_scope="bad scope",
            cost_class=CostClass.LOW,
        )


@pytest.mark.parametrize(
    ("read_only", "idempotent", "eligible"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_prefetch_gate_requires_read_only_and_idempotent(
    read_only: bool, idempotent: bool, eligible: bool
) -> None:
    registry = ToolRegistry((_spec("lookup", read_only=read_only, idempotent=idempotent),))
    assert registry.is_prefetch_eligible("lookup") is eligible


def test_dependency_contract_is_inspectable_and_does_not_force_execution() -> None:
    shipment = TOOL_REGISTRY.get("get_shipment")
    assert shipment.depends_on == (
        ToolDependency(tool_name="get_order", provides_arguments=("order_id",)),
    )
    assert TOOL_REGISTRY.missing_dependency_arguments("get_shipment", ()) == ("order_id",)
    assert TOOL_REGISTRY.missing_dependency_arguments("get_shipment", ("order_id",)) == ()


def test_unknown_self_and_cyclic_dependencies_are_rejected() -> None:
    with pytest.raises(ToolDependencyError, match="unknown"):
        ToolRegistry(
            (
                _spec(
                    "first",
                    depends_on=(
                        ToolDependency(tool_name="missing", provides_arguments=("order_id",)),
                    ),
                ),
            )
        )
    with pytest.raises(ToolDependencyError, match="itself"):
        ToolRegistry(
            (
                _spec(
                    "first",
                    depends_on=(
                        ToolDependency(tool_name="first", provides_arguments=("order_id",)),
                    ),
                ),
            )
        )
    first = _spec(
        "first", depends_on=(ToolDependency(tool_name="second", provides_arguments=("order_id",)),)
    )
    second = _spec(
        "second", depends_on=(ToolDependency(tool_name="first", provides_arguments=("order_id",)),)
    )
    with pytest.raises(ToolDependencyError, match="cyclic"):
        ToolRegistry((first, second))


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_order", {"order_id": "ORD-0042"}),
        ("get_shipment", {"order_id": "ORD-0042"}),
        ("get_customer", {"customer_id": "CUST-073"}),
        ("get_policy", {"policy_name": "returns"}),
        ("get_inventory", {"sku": "SKU-0007"}),
    ],
)
def test_argument_contracts_accept_fixture_identifiers(
    name: str, arguments: dict[str, str]
) -> None:
    assert TOOL_REGISTRY.validate_arguments(name, arguments).model_dump() == arguments


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"order_id": "ORD-0042", "extra": "no"},
        {"order_id": 42},
        {"order_id": "ord-0042"},
        {"order_id": " ORD-0042"},
        {"order_id": "ORD-0042/../x"},
    ],
)
def test_argument_contract_rejects_missing_extra_coerced_or_unsafe_values(
    arguments: dict[str, Any],
) -> None:
    with pytest.raises(ToolArgumentsError):
        TOOL_REGISTRY.validate_arguments("get_order", arguments)


def test_canonical_arguments_are_sorted_unicode_preserving_and_validated() -> None:
    class NestedArgs(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        z: dict[str, str]
        a: list[int]
        label: str = "café"

    registry = ToolRegistry(
        (
            ToolSpec(
                name="nested",
                description="Validate nested canonical JSON.",
                args_model=NestedArgs,
                read_only=True,
                idempotent=True,
                freshness_ttl_s=0,
                auth_scope="test:read",
                cost_class=CostClass.LOW,
            ),
        )
    )
    left = {"z": {"b": "2", "a": "1"}, "a": [2, 1]}
    right = {"a": [2, 1], "z": {"a": "1", "b": "2"}}
    expected = '{"a":[2,1],"label":"café","z":{"a":"1","b":"2"}}'
    assert registry.canonical_arguments("nested", left) == expected
    assert registry.canonical_arguments("nested", right) == expected
    assert registry.canonical_arguments("nested", {**right, "label": "other"}) != expected
    with pytest.raises(ToolArgumentsError):
        registry.canonical_arguments("nested", {**right, "unknown": True})


def test_canonicalization_rejects_non_finite_values() -> None:
    class FloatArgs(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        value: float = Field(allow_inf_nan=True)

    registry = ToolRegistry((_spec("lookup").model_copy(update={"args_model": FloatArgs}),))
    with pytest.raises(ValueError, match="JSON compliant"):
        registry.canonical_arguments("lookup", {"value": float("nan")})


def test_openai_schemas_derive_from_runtime_contracts_and_are_deterministic() -> None:
    schemas = TOOL_REGISTRY.openai_schemas()
    assert schemas == build_tool_registry().openai_schemas()
    assert [schema["function"]["name"] for schema in schemas] == list(TOOL_REGISTRY.names())
    for schema in schemas:
        name = schema["function"]["name"]
        parameters = schema["function"]["parameters"]
        spec = TOOL_REGISTRY.get(name)
        assert schema["function"]["description"] == spec.description
        assert parameters["additionalProperties"] is False
        assert '"title"' not in json.dumps(parameters)
        assert parameters["required"] == list(spec.args_model.model_fields)
        json.dumps(schema, allow_nan=False)
    inventory = next(item for item in schemas if item["function"]["name"] == "get_inventory")
    assert inventory["function"]["parameters"]["properties"]["sku"]["pattern"] == r"^SKU-\d{4}$"


async def test_execution_enforces_validation_auth_binding_and_deadline() -> None:
    calls: list[BaseModel] = []

    async def executor(
        arguments: BaseModel, context: ExecutionContext, deadline: float
    ) -> dict[str, Any]:
        calls.append(arguments)
        return {"ok": True}

    registry = ToolRegistry((_spec("lookup"),)).with_executors({"lookup": executor})
    deadline = asyncio.get_running_loop().time() + 1
    with pytest.raises(ToolArgumentsError):
        await registry.execute(
            "lookup",
            {"order_id": "bad"},
            context=ExecutionContext(frozenset({"test:read"})),
            deadline=deadline,
        )
    with pytest.raises(ToolAuthorizationError):
        await registry.execute("lookup", {"order_id": "ORD-0042"}, context=None, deadline=deadline)
    with pytest.raises(ToolArgumentsError):
        await registry.execute(
            "lookup",
            {"order_id": "ORD-0042", "auth_scope": "test:read"},
            context=None,
            deadline=deadline,
        )
    with pytest.raises(ToolDeadlineExceeded):
        await registry.execute(
            "lookup",
            {"order_id": "ORD-0042"},
            context=ExecutionContext(frozenset({"test:read"})),
            deadline=0,
        )
    assert not calls
    assert await registry.execute(
        "lookup",
        {"order_id": "ORD-0042"},
        context=ExecutionContext(frozenset({"test:read"})),
        deadline=deadline,
    ) == {"ok": True}
    with pytest.raises(ToolNotBoundError):
        await ToolRegistry((_spec("lookup"),)).execute(
            "lookup",
            {"order_id": "ORD-0042"},
            context=ExecutionContext(frozenset({"test:read"})),
            deadline=deadline,
        )


async def test_langgraph_bindings_preserve_contract_and_guarded_execution() -> None:
    async def executor(
        arguments: BaseModel, context: ExecutionContext, deadline: float
    ) -> dict[str, Any]:
        return arguments.model_dump()

    registry = ToolRegistry((_spec("lookup"),)).with_executors({"lookup": executor})
    context = ExecutionContext(frozenset({"test:read"}))
    tools = registry.langgraph_bindings(
        context=context, deadline=asyncio.get_running_loop().time() + 1
    )
    assert len(tools) == 1
    assert tools[0].name == "lookup"
    assert tools[0].description == registry.get("lookup").description
    assert tools[0].args_schema is GetOrderArgs
    assert await tools[0].ainvoke({"order_id": "ORD-0042"}) == {"order_id": "ORD-0042"}
    with pytest.raises(ValidationError):
        await tools[0].ainvoke({"order_id": "bad"})
    blocked = registry.langgraph_bindings(
        context=ExecutionContext(frozenset()),
        deadline=asyncio.get_running_loop().time() + 1,
    )
    with pytest.raises(ToolAuthorizationError):
        await blocked[0].ainvoke({"order_id": "ORD-0042"})


def test_inventory_argument_contract_is_canonical_not_historical_alias() -> None:
    assert TOOL_REGISTRY.get("get_inventory").args_model is GetInventoryArgs
    assert "get_product" not in TOOL_REGISTRY.names()
