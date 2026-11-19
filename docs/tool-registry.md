# Tool registry

`predictive_agent.tools` is the single source for tool identity, validation, schemas, safety
metadata, authorization, and execution binding. Registry construction and schema generation do not
perform network I/O.

## Contracts and benchmark tools

`ToolSpec` is a frozen Pydantic model. Its fields have these semantics:

- `name`, `description`, and `args_model` define the public model-facing contract.
- `read_only` means intended execution does not modify external state. `idempotent` separately
  means repeating an equivalent request is safe with respect to intended effects.
- `freshness_ttl_s` is a proposed future reuse window in seconds. Zero means do not reuse. It is
  neither an HTTP timeout nor a latency measurement, and never bypasses authorization.
- `auth_scope` is the synthetic scope required from trusted application context.
- `depends_on` identifies data another registered tool can provide. It does not trigger execution
  or impose ordering when the required argument is already available.
- `cost_class` is a relative planning label, not measured latency or a performance guarantee.

| Tool | Arguments | Scope | TTL | Cost | Dependency |
|---|---|---|---:|---|---|
| `get_order` | `order_id: ^ORD-\\d{4}$` | `orders:read` | 30 s | medium | — |
| `get_shipment` | `order_id: ^ORD-\\d{4}$` | `shipments:read` | 10 s | medium | `get_order` may provide `order_id` |
| `get_customer` | `customer_id: ^CUST-\\d{3}$` | `customers:read` | 300 s | low | — |
| `get_policy` | `returns`, `shipping`, or `cancellations` | `policies:read` | 3600 s | low | — |
| `get_inventory` | `sku: ^SKU-\\d{4}$` | `inventory:read` | 10 s | medium | — |

All five are read-only and idempotent. The TTL and cost values are conservative proposed policy
for deterministic fixtures; they are not experimental measurements and may differ from live-system
policy.

Argument models are strict, frozen objects with unknown fields forbidden. Identifiers are
case-sensitive and whitespace is not normalized because the fixture API keys are exact. Invalid
types, malformed IDs, unsafe path strings, and model-supplied authorization fields fail before I/O.

## Canonical JSON and schemas

`ToolRegistry.canonical_arguments()` first validates with the registered Pydantic model, emits its
JSON-compatible representation including defaults, then serializes with sorted keys, UTF-8/Unicode
preservation, compact separators, and non-finite-number rejection. Array order is preserved.
Canonical arguments are not a complete authorization-safe cache key; later cache identity also
needs tool, tenant, authorization, and data-version dimensions.

`openai_schemas()` derives each function-tool definition from the same Pydantic model. Names and
descriptions come only from `ToolSpec`; required fields, patterns, enums, and
`additionalProperties: false` match runtime validation.

The LLM-facing compatibility projection removes only Pydantic's descriptive `title` annotations,
which the strict Mock LLM schema subset does not support. It does not remove validation constraints;
the complete Pydantic model remains the authoritative runtime contract. The projection is
schema-aware: property and definition names called `title`, plus `default`, `enum`, `const`, and
other data-valued keywords, remain unchanged.

`langgraph_bindings(context=..., deadline=...)` returns async LangChain `BaseTool`/
`StructuredTool` instances derived from those same specs. Trusted context and the absolute deadline
are bound by application code and hidden from the model schema. Bindings delegate through
`ToolRegistry.execute()`, so they cannot bypass validation or authorization.

## Safety, authorization, and dependencies

Basic future-prefetch eligibility is exactly:

```python
spec.read_only and spec.idempotent
```

Unknown tools fail closed. Eligibility is not permission to speculate: a future prefetch layer must
also check authorization, dependencies, freshness, request context, and policy.

`ExecutionContext` carries trusted granted scopes plus optional subject and tenant identifiers.
Missing context or scope fails before HTTP I/O. The scopes above are synthetic M1.7 fixtures, not
the future M1.9 JWT claim contract. A generic `customers:read` scope does not prove resource-level
access in a real multi-tenant deployment.

`ExecutionContext` copies granted scopes into an immutable `frozenset`, so later mutation of a
caller-owned collection cannot change an authorization decision.

Shipment dependency metadata says `get_order` can provide `order_id`. The registry exposes missing
dependency input inspection but never performs dependency resolution, chaining, or scheduling. If
an authorized valid order ID is already known, shipment execution does not force an order lookup.

## HTTP execution and lifecycle

`HTTPToolClient.execute(name, arguments, context=..., deadline=...)` validates arguments and scope,
then issues one asynchronous GET to the fixed registered route. Tool input cannot alter the base
URL, method, or endpoint. Redirect following and automatic retries are disabled. Typed failures
distinguish registry lookup, argument validation, authorization, deadline expiry, connection error,
404, other HTTP status, invalid JSON, and unexpected response shape; cancellation propagates.

One `httpx.AsyncClient` is reused for connection pooling. `async with` or `close()` releases an
internally created client, while an injected client remains caller-owned. The lifecycle guard is
inside every bound executor, so direct calls, retained registry references, and previously created
LangGraph bindings all fail consistently after closure without issuing HTTP requests.

A parent monotonic deadline from `asyncio.get_running_loop().time()` is enforced by the registry
with `asyncio.timeout_at()` and also bounds the HTTP request timeout; it is never refreshed.
Synchronous response decoding and validation are checked against the deadline before a successful
result is returned, but this is not a hard real-time preemption guarantee for synchronous Python
code. This component adds no artificial delay.

`PA_TOOL_API_BASE_URL` defaults to `http://localhost:18100`, the Toxiproxy route for representative
latency tests. Set `http://localhost:8080` explicitly only for direct correctness or infrastructure
checks. With no retries or local delays, experimental network effects remain attributable to
Toxiproxy.

## Inventory compatibility

`get_inventory` is the only public registry name. The M1.5 mock keeps `get_product` as an isolated
historical compatibility name and also recognizes `get_inventory`, returning the exact name in the
active advertised contract. Advertising both aliases together fails closed as ambiguous. Historical
calibration artifacts and timing coefficients remain unchanged; the new name is not claimed as an
independent recalibration.

## Extending and validating

Adding a tool requires a strict Pydantic argument model, one `ToolSpec`, and one trusted HTTP
endpoint/result binding. The core registry, provider schema lists, and orchestration modes do not
need separate edits.

Ordinary validation requires no services:

```bash
uv sync --locked
uv run pytest -m "not integration"
```

With Compose running, `PA_RUN_INTEGRATION=1 uv run pytest tests/integration -s` checks all five
proxied routes and a bounded registry-schema → Mock LLM → validation/auth → HTTP flow.

For M1.8, create one request deadline and one trusted `ExecutionContext`, pass
`TOOL_REGISTRY.openai_schemas()` to `LLMClient`, fully validate returned calls through the registry,
reuse one lifespan-owned `HTTPToolClient`, and build the graph from
`client.registry.langgraph_bindings(...)` with request-specific trusted context and the original
deadline. Execute the graph asynchronously because the bindings are async-only.
`LLMToolCall.executable` remains only an LLM-client readiness check; the deadline must not be reset
between LLM and tool operations, and bindings must never be reused after their HTTP client closes.

Caching, speculative execution, automatic dependencies, the agent graph, JWT validation, and
resource-level tenant authorization remain outside M1.7.
