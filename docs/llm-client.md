# Asynchronous LLM client

`predictive_agent.llm.LLMClient` is the shared communication layer for OpenAI-compatible
providers. It owns one reusable `openai.AsyncOpenAI` instance and supports regular completions,
streaming text, fragmented tool calls, usage metadata, bounded retries, and OpenTelemetry API
spans. Provider switching is a `Settings` change; higher-level agent code uses the same methods.

## Configuration

All values use the `PA_` prefix and are loaded by `Settings`.

| Setting | Purpose | Default |
|---|---|---|
| `PA_LLM_PROVIDER` | `ollama`, `mock`, `groq`, or `openai-compatible` | `ollama` |
| `PA_LLM_BASE_URL` | Absolute OpenAI-compatible `/v1` URL | `http://localhost:11434/v1` |
| `PA_LLM_API_KEY` | Provider credential or local placeholder | `ollama` |
| `PA_LLM_MODEL` | Provider model identifier | `qwen3.5:9b-q4_K_M` |
| `PA_LLM_THINK` | Ollama reasoning mode | `false` |
| `PA_OLLAMA_KEEP_ALIVE` | Ollama model residency request | `30m` |
| `PA_HTTP_TIMEOUT_S` | Default total logical-operation budget | `120` |
| `PA_LLM_MAX_RETRIES` | Additional attempts after the first attempt | `2` |
| `PA_LLM_RETRY_BACKOFF_INITIAL_S` | Initial deterministic backoff | `0.25` |
| `PA_LLM_RETRY_BACKOFF_MAX_S` | Backoff and `Retry-After` cap | `2` |

Examples:

```dotenv
# Ollama
PA_LLM_PROVIDER=ollama
PA_LLM_BASE_URL=http://localhost:11434/v1
PA_LLM_API_KEY=ollama
PA_LLM_MODEL=qwen3.5:9b-q4_K_M
PA_LLM_THINK=false

# Docker Compose mock LLM
PA_LLM_PROVIDER=mock
PA_LLM_BASE_URL=http://localhost:8090/v1
PA_LLM_API_KEY=mock
PA_LLM_MODEL=mock-calibrated

# Groq (supply credentials outside source control)
PA_LLM_PROVIDER=groq
PA_LLM_BASE_URL=https://api.groq.com/openai/v1
PA_LLM_API_KEY=<provided-api-key>
PA_LLM_MODEL=<supported-model>
```

Only Ollama receives `extra_body={"think": ..., "keep_alive": ...}`. Those fields are never
sent to the strict mock endpoint, Groq, or generic OpenAI-compatible providers. The SDK's own
retry policy is disabled so there is one observable retry policy.

## Interface and lifecycle

```python
from contextlib import aclosing

from predictive_agent.config import Settings
from predictive_agent.llm import LLMClient


async def answer() -> str | None:
    async with LLMClient(Settings()) as client:
        result = await client.complete(
            [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "What is the current order state?"},
            ],
            max_completion_tokens=128,
        )
        return result.content
```

Construct a client at an application lifecycle boundary and reuse it to retain HTTP connection
pooling. `async with` or `await client.close()` releases an internally created SDK client. An
injected SDK client remains owned by its caller.

`complete()` returns an `LLMResult`. `stream()` is an async iterator of `LLMStreamEvent` values:

- `text_delta` exposes non-empty text.
- `tool_call_delta` exposes indexed, possibly incomplete tool-call state.
- `tool_call_ready` appears only after the provider finalizes tool-call generation and the call has
  an ID, names a submitted function, parses to a JSON object, and contains every top-level
  `required` property declared by that function.
- `completed` contains the final `LLMResult`, usage, finish reason, and metrics.

`LLMToolCall.executable` means the call passes that same client-level readiness gate. It does not
mean the complete JSON Schema has been validated: types, formats, nested constraints,
`additionalProperties`, and business rules remain the tool layer's responsibility. Non-ready final
calls remain available on `LLMResult.tool_calls` with `readiness_error` and
`missing_required_arguments` details.

The client only assembles and gates tool calls. Agent code remains responsible for full schema
validation, tool execution, and appending assistant tool-call and tool-result messages before the
next completion.

Once the provider stream closes successfully, the client constructs the immutable result, freezes
the successful metrics, and ends `agent.llm_turn` before yielding any `tool_call_ready` or
`completed` event. Closing after readiness therefore preserves success; closing while provider I/O
is still active records `abandoned`.

Use `contextlib.aclosing()` when a consumer may stop before the terminal event so the provider
stream and logical tracing span close immediately:

```python
async with aclosing(client.stream(messages)) as events:
    async for event in events:
        if should_stop(event):
            break
```

## Deadlines, cancellation, and retries

Each logical operation uses one absolute event-loop deadline. Pass a parent deadline created from
the same loop, or use `client.deadline_after(seconds)`. When omitted, the deadline is operation
start plus `PA_HTTP_TIMEOUT_S`. It covers connection setup, every request attempt, response
headers, provider-chunk waits, backoff, and the logical stream lifetime. Retries never reset it,
and an expired deadline raises `LLMDeadlineExceeded`.

Streaming installs the timeout only while opening the provider stream or awaiting its next chunk.
No LLM-owned timeout remains active while consumer code handles a yielded event, so unrelated
consumer awaits are not cancelled by the LLM client. Wall-clock time still advances against the
same absolute deadline; if consumer work runs past it, the next provider operation fails before
reading another chunk. Total duration includes client-observed connection, retry, backoff, provider
waiting, and stream consumption through provider completion. It is frozen when the provider stream
closes, so consumer work after a readiness event cannot inflate successful LLM duration. An
unbuffered stream still cannot remove consumer backpressure before provider completion, and this
measurement is not provider-internal inference time.

Cancellation propagates as `asyncio.CancelledError`; open streams are closed by their context
manager. Only connection failures and HTTP 429 responses are retried. `Retry-After` and
`retry-after-ms` are honored up to the configured backoff cap and only when they fit within the
remaining deadline. Authentication, invalid requests, arbitrary HTTP errors, protocol errors,
timeouts, and cancellations are not retried. A stream failure after any text or tool-call delta
raises `LLMPartialStreamError` without retrying, preventing duplicate content or tool execution.

## Measurements and tracing

`LLMMetrics` reports provider, requested and returned models, first transport event, first
meaningful text or tool delta, first text, first tool-call delta, first executable tool-call
availability, total client-observed duration, attempts, retries, 429 count, outcome, and provider
usage when present. Missing token usage stays `None`. Regular completions do not fabricate
time-to-first-token.

Each request creates an `agent.llm_turn` span through the OpenTelemetry API. Attributes contain
models, provider, timing, usage, attempts, 429 count, and outcome. Prompts, tool results, API keys,
and authorization data are not recorded. A future tracing milestone may install an SDK/exporter;
this component does not configure one.

For streaming, the logical span is created with the caller's active span as its parent but is made
current only around provider-owned awaits. It is not current while an event is yielded, so
consumer-side spans retain the caller's context instead of becoming accidental children of
`agent.llm_turn`. Success ends the span before post-completion readiness or terminal events are
exposed; timeout, error, cancellation, and explicit early closure end it once with the corresponding
outcome. Closure before provider completion is recorded as `abandoned`; closure after successful
provider completion keeps the already-finalized `success` outcome even when terminal events remain
unread.

These timings include client and network work and must not be described as provider-internal GPU
prefill or decode measurements.

## Validation

The ordinary suite uses deterministic SDK test doubles and no provider access:

```bash
uv sync --locked
uv run pytest -m "not integration"
```

Run the Docker mock integration with the Compose stack:

```bash
docker compose up -d --wait
PA_RUN_INTEGRATION=1 uv run pytest tests/integration/test_llm_client_integration.py -s
docker compose down
```

Ollama checks additionally require `PA_RUN_OLLAMA_INTEGRATION=1`; the representative tool-call
check requires `PA_RUN_OLLAMA_TOOL_INTEGRATION=1` and a tool-capable configured model. Groq is
never required by ordinary CI. Run it only with `PA_RUN_GROQ_INTEGRATION=1`, a valid
`PA_LLM_API_KEY`, and a supported `PA_LLM_MODEL`.

## Known limits and handoff

- Capability discovery is explicit configuration; the client does not probe provider features.
- The readiness gate checks recognition, JSON-object syntax, and top-level required-property
  presence only; full JSON Schema validation remains a tool-layer concern.
- There is no automatic recovery after exposed partial stream output.
- Consumers that stop early must explicitly close the async generator; `aclosing()` is the
  recommended deterministic pattern.
- OpenTelemetry export, agent graphs, tool execution, prefetching, and benchmark comparisons belong
  to later milestones.
- M1.5 timing coefficients remain unchanged. Mock calibration limitations continue to be tracked
  separately and are not resolved by successful provider communication.
