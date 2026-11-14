# Mock LLM calibration and paired runtime validation

The current M1.5 acceptance evidence is the direct paired run
[`mock-llm-paired-runtime-2026-11-15.json`](mock-llm-paired-runtime-2026-11-15.json).
It compares Ollama 0.40.16 running `qwen3.5:9b-q4_K_M` with the actual Docker mock
service built from commit `3061623c202a7becdb1e83065260685c286910c5`.

The earlier
[`mock-llm-calibration-2026-11-14.json`](mock-llm-calibration-2026-11-14.json) is
preserved unchanged as historical evidence. Its cases have since been inspected and are
development data, not untouched final validation. Its “end-to-end” values were equations
conditioned on Ollama-generated output; they were not direct mock-runtime measurements.

## Root causes corrected

The old prompt regression assigned ASCII letters a zero coefficient, so long alphabetic
inputs were not length-sensitive. The revised empirical feature model uses total character
count as a positive base feature and digit, punctuation, and non-ASCII adjustments. Four
alphabetic, alphanumeric, and repeated-text workloads were added to training.

The reported policy discrepancy was caused by cache isolation, not a policy-specific timing
requirement. A nonce was prepended to a system message, and Ollama used that nonce as
`policy_name`; the mock correctly used `returns`. Nonces appended to system messages are now
explicitly labelled as benchmark metadata that must not become tool arguments. Fresh
development and final policy requests emitted one equivalent schema-valid action without a
special delay.

The actual runtime is calibrated from paired development wall time. Each case includes a
zero-delay mock control to measure HTTP/SSE overhead. Text cases fit base and prompt-prefill
time; equivalent tool cases fit the remaining executable-action decode time. The selected
runtime configuration was frozen before the new final cases were created:

| Parameter | Runtime value |
|---|---:|
| Base latency | 77.282 ms |
| Effective prefill throughput | 3,089.067 tokens/s |
| Decode throughput | 64.340 tokens/s |

## Token estimator

The fitted prompt estimate is:

```text
tokens = 18.662082
       + 0.220064 * total_message_characters
       + 0.814815 * ASCII_digits
       + 0.000000 * ASCII_whitespace
       + 0.160370 * ASCII_punctuation
       + 0.365650 * non_ASCII_characters
       + 20.817746 * additional_messages
       + 0.941413 * serialized_tool_schema_characters
```

The positive total-character term makes alphabetic, minimal-whitespace, alphanumeric, and
repeated inputs length-sensitive. Unit coverage also includes ordinary prose, code, JSON,
Turkish, conversation length, and tool-schema size.

On the new final paired set, prompt-token error was 10.01% median and 61.18% maximum.
The maximum belongs to structured conversation history; tool-schema cases reached roughly
29–33%. Direct runtime timing nevertheless passed because the runtime parameters were fitted
against actual paired wall time. The estimator is a lightweight benchmark approximation, not
a tokenizer replacement; structural prompt-token generalization remains the largest known
accuracy limitation.

## Paired methodology

Ollama uses streaming `/api/chat`; the mock uses streaming OpenAI-compatible
`/v1/chat/completions`. The runner records both input payloads, full generated content and tool
calls, canonical output, token counts, cache counts, first meaningful output, first executable
tool call, terminal duration, zero-delay controls, and per-sample errors.

A tool call is executable only when all arguments are complete JSON, the named supplied tool
exists, and required arguments satisfy the supported schema. Tool-ready timing is comparable
only when both endpoints produce exactly one identical executable action. Total duration also
requires identical canonical output and completion-token lengths within ±15%. Text outputs
are intentionally different, so text TTFT is comparable but text total duration is not.

Training was used for token features, the inspected November 14 cases were used for development,
and the 12 final cases were introduced only after timing, equivalence rules, and coefficients were
frozen. The final set covers English, Turkish, code, JSON, conversation history, all five tools,
a post-tool response, and a multi-turn tool change. Two repetitions used unique nonces,
`cache_prompt=false`, and excluded recorded warm-ups. Every final cached-token count was zero.

## Independent final results

Run `b7c7a09d07b09ac0` executed on 2026-11-15 under WSL2 with Python 3.12.3,
HTTPX 0.28.1, an NVIDIA GeForce RTX 5060 Ti (8,151 MiB), driver 610.88, and 15 GiB
visible memory.

| Direct case-median metric | Median error | p95 | Maximum | Result |
|---|---:|---:|---:|---:|
| First meaningful output | 5.58% | 11.25% | 11.25% | 12/12 pass |
| Executable tool ready | 5.64% | 11.25% | 11.25% | 6/6 pass |
| Length-equivalent total | 5.20% | 8.76% | 8.76% | 6/6 pass |

All individual measurements also passed: 24/24 first meaningful, 12/12 tool ready, and
12/12 comparable totals. Their maximum errors were 11.68%, 11.68%, and 9.06% respectively.

| Final executable action | Ollama / mock tool-ready | Error | Output tokens | Total error |
|---|---:|---:|---:|---:|
| `get_order` | 0.674 / 0.675 s | 1.51% | 32 / 32 | 2.30% |
| `get_shipment` | 0.690 / 0.758 s | 9.77% | 33 / 33 | 6.79% |
| `get_customer` | 0.675 / 0.676 s | 0.30% | 32 / 32 | 2.01% |
| `get_policy` | 0.586 / 0.581 s | 1.00% | 27 / 26 | 3.62% |
| `get_product` | 0.654 / 0.726 s | 10.88% | 31 / 31 | 8.41% |
| Multi-turn `get_product` | 0.688 / 0.766 s | 11.25% | 31 / 31 | 8.76% |

## Reproduction

Training/development calibration writes a new non-overwritable artifact:

```bash
uv run python -m scripts.calibrate_mock_llm \
  --development --repetitions 2 \
  --output docs/benchmarks/mock-llm-calibration-development-YYYY-MM-DD.json
```

With Ollama and the fresh Compose mock running, paired development validation is:

```bash
uv run python -m scripts.validate_mock_llm_runtime \
  --repetitions 2 --output docs/benchmarks/mock-llm-paired-development-YYYY-MM-DD.json
```

`--final` selects the frozen final dataset. A final artifact must use a new path and should
not be used to tune the implementation that produced it.

## Supported contract and limits

Automatic selection is isolated to the five M1.3 tools and requires matching current-turn
intent; an available incidental identifier cannot establish intent. Named choices remain
explicit. Duplicate tool names, malformed histories, and unsupported schemas fail clearly.

The supported schema subset is object `properties`, `required`, boolean
`additionalProperties`; primitive `string`, `integer`, `number`, `boolean`, `null`; and
`enum`, `const`, `default`, `pattern`, `minLength`, `maxLength`, `minimum`, `maximum`.
Patterns use JSON Schema search semantics. Null constraint values, non-finite numbers, and
boolean/numeric enum mismatches are rejected where applicable.

Results are specific to this model, quantization, Ollama version, prompt template, mock
implementation, Docker/WSL path, and hardware. Recalibrate after changing them. The evidence
does not claim arbitrary-model, long-form, reasoning-mode, concurrent-load, or complete JSON
Schema accuracy.
