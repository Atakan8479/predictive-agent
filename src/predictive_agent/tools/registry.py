"""Authoritative tool contracts, validation, schemas, and execution gates."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, field_validator

_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_AUTH_SCOPE = re.compile(r"^[a-z][a-z0-9_-]*:[a-z][a-z0-9_-]*$")


class ToolRegistryError(Exception):
    """Base class for registry-owned failures."""


class DuplicateToolError(ToolRegistryError):
    """A tool name is already registered."""


class UnknownToolError(ToolRegistryError, KeyError):
    """A requested tool is not registered."""


class ToolArgumentsError(ToolRegistryError, ValueError):
    """Tool arguments do not satisfy the authoritative Pydantic contract."""


class ToolAuthorizationError(ToolRegistryError, PermissionError):
    """Trusted execution context is missing a required scope."""


class ToolDependencyError(ToolRegistryError, ValueError):
    """Dependency metadata is invalid or required input is unavailable."""


class ToolNotBoundError(ToolRegistryError):
    """No execution implementation is associated with the tool."""


class ToolDeadlineExceeded(ToolRegistryError, TimeoutError):
    """The caller's absolute monotonic deadline expired."""


class CostClass(StrEnum):
    """Proposed relative execution cost, not a latency guarantee."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolDependency(BaseModel):
    """Data another tool can provide; it does not mandate execution order."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    tool_name: str = Field(pattern=_TOOL_NAME.pattern)
    provides_arguments: tuple[str, ...] = Field(min_length=1)

    @field_validator("provides_arguments")
    @classmethod
    def validate_arguments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(not _TOOL_NAME.fullmatch(item) for item in value):
            raise ValueError("dependency arguments must be unique snake_case names")
        return value


class ToolSpec(BaseModel):
    """Immutable authoritative contract and safety metadata for one tool."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True,
    )

    name: str = Field(pattern=_TOOL_NAME.pattern)
    description: str = Field(min_length=1, max_length=512)
    args_model: type[BaseModel]
    read_only: StrictBool
    idempotent: StrictBool
    freshness_ttl_s: float = Field(ge=0, allow_inf_nan=False, strict=True)
    auth_scope: str = Field(pattern=_AUTH_SCOPE.pattern)
    depends_on: tuple[ToolDependency, ...] = ()
    cost_class: CostClass

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("description must not have surrounding whitespace")
        return value

    @field_validator("args_model")
    @classmethod
    def validate_args_model(cls, value: type[BaseModel]) -> type[BaseModel]:
        schema = value.model_json_schema()
        if (
            schema.get("type") != "object"
            or schema.get("additionalProperties") is not False
            or value.model_config.get("extra") != "forbid"
            or value.model_config.get("strict") is not True
        ):
            raise ValueError("args_model must be a strict object model with extra='forbid'")
        return value

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, value: tuple[ToolDependency, ...]) -> tuple[ToolDependency, ...]:
        names = [dependency.tool_name for dependency in value]
        if len(names) != len(set(names)):
            raise ValueError("dependency tool names must be unique")
        return value

    @property
    def prefetch_eligible(self) -> bool:
        """Basic fail-closed safety gate; authorization and policy still apply."""
        return self.read_only and self.idempotent


@dataclass(frozen=True, slots=True, init=False)
class ExecutionContext:
    """Trusted application context; never sourced from model tool arguments."""

    granted_scopes: frozenset[str]
    subject: str | None = None
    tenant_id: str | None = None

    def __init__(
        self,
        granted_scopes: Iterable[str],
        subject: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        if isinstance(granted_scopes, str):
            raise TypeError("granted_scopes must be an iterable of complete scope strings")
        object.__setattr__(self, "granted_scopes", frozenset(granted_scopes))
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "tenant_id", tenant_id)


ToolExecutor = Callable[[BaseModel, ExecutionContext, float], Awaitable[dict[str, Any]]]


class ToolRegistry:
    """Single deterministic source for tool contracts and bound execution."""

    def __init__(
        self,
        specs: Iterable[ToolSpec] = (),
        *,
        executors: Mapping[str, ToolExecutor] | None = None,
    ) -> None:
        registered: dict[str, ToolSpec] = {}
        for supplied_spec in specs:
            spec = ToolSpec.model_validate(supplied_spec.model_dump(round_trip=True))
            if spec.name in registered:
                raise DuplicateToolError(f"tool {spec.name!r} is already registered")
            registered[spec.name] = spec
        self._validate_dependencies(registered)
        bound = dict(executors or {})
        unknown_bindings = set(bound).difference(registered)
        if unknown_bindings:
            raise UnknownToolError(f"cannot bind unknown tools: {sorted(unknown_bindings)!r}")
        self._specs = MappingProxyType(registered)
        self._executors = MappingProxyType(bound)

    def register(self, spec: ToolSpec) -> ToolRegistry:
        """Return a new registry containing the added immutable specification."""
        if spec.name in self._specs:
            raise DuplicateToolError(f"tool {spec.name!r} is already registered")
        return ToolRegistry((*self.specs(), spec), executors=self._executors)

    def with_executors(self, executors: Mapping[str, ToolExecutor]) -> ToolRegistry:
        """Return a registry view associated with execution implementations."""
        return ToolRegistry(self.specs(), executors=executors)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise UnknownToolError(f"unknown tool {name!r}") from error

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))

    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs())

    def validate_arguments(self, name: str, arguments: Mapping[str, Any]) -> BaseModel:
        spec = self.get(name)
        try:
            return spec.args_model.model_validate(dict(arguments), strict=True)
        except ValidationError as error:
            raise ToolArgumentsError(f"invalid arguments for {name!r}") from error

    def canonical_arguments(self, name: str, arguments: Mapping[str, Any]) -> str:
        validated = self.validate_arguments(name, arguments)
        value = validated.model_dump(mode="json", exclude_none=False, exclude_unset=False)
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def openai_schemas(self) -> tuple[dict[str, Any], ...]:
        schemas: list[dict[str, Any]] = []
        for spec in self.specs():
            parameters = _sorted_json_object(
                _without_schema_titles(spec.args_model.model_json_schema())
            )
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": parameters,
                    },
                }
            )
        return tuple(schemas)

    def is_prefetch_eligible(self, name: str) -> bool:
        return self.get(name).prefetch_eligible

    def missing_dependency_arguments(
        self, name: str, available_arguments: Iterable[str]
    ) -> tuple[str, ...]:
        spec = self.get(name)
        available = frozenset(available_arguments)
        provided = {
            argument for dependency in spec.depends_on for argument in dependency.provides_arguments
        }
        required = set(spec.args_model.model_json_schema().get("required", []))
        return tuple(sorted(required.intersection(provided).difference(available)))

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        context: ExecutionContext | None,
        deadline: float,
    ) -> dict[str, Any]:
        spec = self.get(name)
        validated = self.validate_arguments(name, arguments)
        if context is None or spec.auth_scope not in context.granted_scopes:
            raise ToolAuthorizationError(f"tool {name!r} requires scope {spec.auth_scope!r}")
        if not math.isfinite(deadline) or deadline <= asyncio.get_running_loop().time():
            raise ToolDeadlineExceeded(f"deadline expired before executing {name!r}")
        executor = self._executors.get(name)
        if executor is None:
            raise ToolNotBoundError(f"tool {name!r} has no execution implementation")
        timeout = asyncio.timeout_at(deadline)
        try:
            async with timeout:
                result = await executor(validated, context, deadline)
        except TimeoutError as error:
            if timeout.expired() or asyncio.get_running_loop().time() >= deadline:
                raise ToolDeadlineExceeded(f"deadline expired while executing {name!r}") from error
            raise
        if asyncio.get_running_loop().time() >= deadline:
            raise ToolDeadlineExceeded(f"deadline expired while executing {name!r}")
        return result

    def langgraph_bindings(
        self,
        *,
        context: ExecutionContext,
        deadline: float,
    ) -> tuple[BaseTool, ...]:
        """Build async LangGraph-compatible tools with trusted context already bound."""

        def build(spec: ToolSpec) -> BaseTool:
            async def invoke(**arguments: Any) -> dict[str, Any]:
                return await self.execute(spec.name, arguments, context=context, deadline=deadline)

            return StructuredTool.from_function(
                coroutine=invoke,
                name=spec.name,
                description=spec.description,
                args_schema=spec.args_model,
            )

        return tuple(build(spec) for spec in self.specs())

    @staticmethod
    def _validate_dependencies(specs: Mapping[str, ToolSpec]) -> None:
        graph: dict[str, tuple[str, ...]] = {}
        for name, spec in specs.items():
            dependencies = tuple(dependency.tool_name for dependency in spec.depends_on)
            if name in dependencies:
                raise ToolDependencyError(f"tool {name!r} cannot depend on itself")
            unknown = set(dependencies).difference(specs)
            if unknown:
                raise ToolDependencyError(
                    f"tool {name!r} has unknown dependencies: {sorted(unknown)!r}"
                )
            argument_names = set(spec.args_model.model_fields)
            provided = {
                argument
                for dependency in spec.depends_on
                for argument in dependency.provides_arguments
            }
            if invalid := provided.difference(argument_names):
                raise ToolDependencyError(
                    f"tool {name!r} dependencies provide unknown arguments: {sorted(invalid)!r}"
                )
            graph[name] = dependencies

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ToolDependencyError("cyclic tool dependency graph")
            if name in visited:
                return
            visiting.add(name)
            for dependency in graph[name]:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in sorted(graph):
            visit(name)


def _sorted_json_object(value: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return cast(dict[str, Any], json.loads(encoded))


_SCHEMA_MAP_KEYWORDS = frozenset(
    {"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"}
)
_SCHEMA_LIST_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_VALUE_KEYWORDS = frozenset(
    {
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)


def _without_schema_titles(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove schema annotation titles without traversing data-valued keywords."""
    projected: dict[str, Any] = {}
    for keyword, value in schema.items():
        if keyword == "title":
            continue
        if keyword in _SCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            mapping = cast(dict[str, Any], value)
            projected[keyword] = {
                name: _without_schema_titles(cast(dict[str, Any], child))
                if isinstance(child, dict)
                else child
                for name, child in mapping.items()
            }
        elif keyword in _SCHEMA_LIST_KEYWORDS and isinstance(value, list):
            items = cast(list[Any], value)
            projected[keyword] = [
                _without_schema_titles(cast(dict[str, Any], item))
                if isinstance(item, dict)
                else item
                for item in items
            ]
        elif keyword in _SCHEMA_VALUE_KEYWORDS and isinstance(value, dict):
            projected[keyword] = _without_schema_titles(cast(dict[str, Any], value))
        else:
            projected[keyword] = value
    return projected
