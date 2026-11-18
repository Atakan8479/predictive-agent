import asyncio
import json
from typing import Any, Literal

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
        freshness_ttl_s=0.0,
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


def test_registered_metadata_matches_documented_policy() -> None:
    actual = {
        spec.name: (
            spec.freshness_ttl_s,
            spec.auth_scope,
            spec.cost_class,
            tuple(dependency.tool_name for dependency in spec.depends_on),
        )
        for spec in TOOL_REGISTRY.specs()
    }
    assert actual == {
        "get_order": (30.0, "orders:read", CostClass.MEDIUM, ()),
        "get_shipment": (10.0, "shipments:read", CostClass.MEDIUM, ("get_order",)),
        "get_customer": (300.0, "customers:read", CostClass.LOW, ()),
        "get_policy": (3600.0, "policies:read", CostClass.LOW, ()),
        "get_inventory": (10.0, "inventory:read", CostClass.MEDIUM, ()),
    }


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
    ("field", "value"),
    [("read_only", 1), ("idempotent", 0), ("freshness_ttl_s", "1")],
)
def test_safety_metadata_rejects_permissive_coercion(field: str, value: object) -> None:
    data = _spec("lookup").model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        ToolSpec.model_validate(data)


def test_argument_models_must_enable_strict_validation_and_forbid_extras() -> None:
    class PermissiveArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        order_id: str

    with pytest.raises(ValidationError, match="strict object model"):
        ToolRegistry((_spec("lookup").model_copy(update={"args_model": PermissiveArgs}),))


def test_execution_context_copies_mutable_scope_inputs() -> None:
    supplied = {"test:read"}
    context = ExecutionContext(supplied)
    supplied.clear()
    assert context.granted_scopes == frozenset({"test:read"})
    assert isinstance(context.granted_scopes, frozenset)
    with pytest.raises(TypeError, match="iterable"):
        ExecutionContext("test:read")


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
                freshness_ttl_s=0.0,
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


def test_schema_projection_preserves_title_properties_definitions_and_data() -> None:
    class TitleDefinition(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        title: str

    class SchemaArgs(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        title: str
        nested: TitleDefinition
        metadata: dict[str, str] = {"title": "preserved"}
        mode: Literal["safe"] = "safe"

    registry = ToolRegistry((_spec("schema").model_copy(update={"args_model": SchemaArgs}),))
    parameters = registry.openai_schemas()[0]["function"]["parameters"]
    assert "title" in parameters["properties"]
    assert parameters["properties"]["nested"] == {"$ref": "#/$defs/TitleDefinition"}
    assert "TitleDefinition" in parameters["$defs"]
    assert "title" in parameters["$defs"]["TitleDefinition"]["properties"]
    assert parameters["properties"]["metadata"]["default"] == {"title": "preserved"}
    assert parameters["properties"]["mode"]["const"] == "safe"
    assert parameters["required"] == ["title", "nested"]


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


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), float("-inf")])
async def test_non_finite_deadlines_fail_before_executor(deadline: float) -> None:
    called = False

    async def executor(
        arguments: BaseModel, context: ExecutionContext, absolute_deadline: float
    ) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    registry = ToolRegistry((_spec("lookup"),)).with_executors({"lookup": executor})
    with pytest.raises(ToolDeadlineExceeded):
        await registry.execute(
            "lookup",
            {"order_id": "ORD-0042"},
            context=ExecutionContext({"test:read"}),
            deadline=deadline,
        )
    assert not called


async def test_registry_enforces_in_flight_deadline_for_generic_executor() -> None:
    started = asyncio.Event()

    async def executor(
        arguments: BaseModel, context: ExecutionContext, deadline: float
    ) -> dict[str, Any]:
        started.set()
        await asyncio.Event().wait()
        return {"late": True}

    registry = ToolRegistry((_spec("lookup"),)).with_executors({"lookup": executor})
    with pytest.raises(ToolDeadlineExceeded):
        await registry.execute(
            "lookup",
            {"order_id": "ORD-0042"},
            context=ExecutionContext({"test:read"}),
            deadline=asyncio.get_running_loop().time() + 0.02,
        )
    assert started.is_set()


async def test_registry_rejects_result_from_non_preemptible_executor_after_deadline() -> None:
    async def executor(
        arguments: BaseModel, context: ExecutionContext, deadline: float
    ) -> dict[str, Any]:
        while asyncio.get_running_loop().time() <= deadline:
            pass
        return {"late": True}

    registry = ToolRegistry((_spec("lookup"),)).with_executors({"lookup": executor})
    with pytest.raises(ToolDeadlineExceeded):
        await registry.execute(
            "lookup",
            {"order_id": "ORD-0042"},
            context=ExecutionContext({"test:read"}),
            deadline=asyncio.get_running_loop().time() + 0.01,
        )


async def test_executor_originated_timeout_is_not_reclassified() -> None:
    async def executor(
        arguments: BaseModel, context: ExecutionContext, deadline: float
    ) -> dict[str, Any]:
        raise TimeoutError("executor timeout")

    registry = ToolRegistry((_spec("lookup"),)).with_executors({"lookup": executor})
    with pytest.raises(TimeoutError, match="executor timeout") as caught:
        await registry.execute(
            "lookup",
            {"order_id": "ORD-0042"},
            context=ExecutionContext({"test:read"}),
            deadline=asyncio.get_running_loop().time() + 1,
        )
    assert type(caught.value) is TimeoutError


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
