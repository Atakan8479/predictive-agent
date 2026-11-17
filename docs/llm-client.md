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
- `tool_call_ready` appears once an ID, function name, and JSON-object arguments are complete.
- `completed` contains the final `LLMResult`, usage, finish reason, and metrics.

The client only assembles tool calls. Agent code remains responsible for tool execution and for
appending assistant tool-call and tool-result messages before the next completion.

## Deadlines, cancellation, and retries

Each logical operation uses one absolute event-loop deadline. Pass a parent deadline created from
the same loop, or use `client.deadline_after(seconds)`. When omitted, the deadline is operation
start plus `PA_HTTP_TIMEOUT_S`. It covers connection setup, every request attempt, response
headers, stream consumption, backoff, and cleanup. Retries never reset it, and an expired deadline
raises `LLMDeadlineExceeded`.

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
- Tool schemas and message histories are passed through to the SDK and remain agent concerns.
- There is no automatic recovery after exposed partial stream output.
- OpenTelemetry export, agent graphs, tool execution, prefetching, and benchmark comparisons belong
  to later milestones.
- M1.5 timing coefficients remain unchanged. Mock calibration limitations continue to be tracked
  separately and are not resolved by successful provider communication.
