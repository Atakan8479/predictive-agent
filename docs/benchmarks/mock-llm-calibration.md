# Mock LLM calibration and paired runtime validation

The current M1.5 acceptance evidence is the direct paired run
[`mock-llm-paired-runtime-2026-11-15.json`](mock-llm-paired-runtime-2026-11-15.json).
It compares Ollama 0.40.16 running `qwen3.5:9b-q4_K_M` with the actual Docker mock
service built from commit `3061623c202a7becdb1e83065260685c286910c5`.

The supplementary controlled-text run
[`mock-llm-text-duration-2026-11-16.json`](mock-llm-text-duration-2026-11-16.json)
preserves the remaining text total-duration investigation. It did not establish a comparable
text total-duration sample, so that M1.5 acceptance criterion remains unresolved.

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

## Supplementary controlled-text result

Run `ef48dcafd77dd275` used the frozen timing configuration and source commit
`ed990367ef973c7974e6cb23802ce9ba7b81476a`. Five cases with three repetitions covered short
and long English prompts, three completion-length bands, Turkish text, structured JSON, and
representative post-tool responses. Warm-up was recorded and excluded, every request used a
unique nonce with `cache_prompt=false`, and every cached-token count was zero.

Before execution, a controlled text sample was defined as semantically comparable only if both
providers reproduced the declared expected text exactly. Total duration additionally required
completion-token counts within ±15%. The validator did not refit or change the frozen timing
parameters.

| Case | Exact expected output | Token error | Raw total-error median | Acceptance |
|---|---:|---:|---:|---|
| English short | 1/3 | 20–40% | 25.40% | Non-comparable |
| English long | 3/3 | 20% | 33.51% | Non-comparable |
| Post-tool short JSON | 0/3 | 0% | 5.70% | Non-comparable |
| Post-tool Turkish | 0/3 | 0% | 3.96% | Non-comparable |
| Post-tool structured JSON | 0/3 | 3.33% | 9.34% | Non-comparable |

Ollama sometimes retained the `<answer>` tags and, for post-tool cases, omitted the mock's
`Mock final answer:` prefix. The English cases reproduced the expected content in four
measurements, but their Ollama/mock completion counts were 5/6 tokens, a 20% difference above
the threshold. Consequently, comparable text totals were 0/5 cases and 0/15 measurements;
median, p95, and maximum comparable errors are not defined. Across all 15 raw but
non-comparable totals, the diagnostic median was 6.71%, p95 was 37.76%, maximum was 37.76%, and
10/15 happened to fall within ±15%. These diagnostic values are not acceptance evidence.

The experiment was retained without changing prompts, timing constants, or acceptance rules.
Controlled text total-duration accuracy therefore remains explicitly unresolved, and M1.5 is
not fully closed by the available evidence.

First meaningful output was within ±15% for 7/15 supplementary measurements. Its median
error was 20.02% and maximum error was 34.85%. This does not invalidate the November 15
result, where all 24 measurements passed with an 11.68% maximum, but it means TTFT accuracy
has not been established across the broader controlled-text workload.

## M1.5 status and experimental limitations

| Component | Status |
|---|---|
| Mock LLM software implementation | Complete |
| OpenAI-compatible response handling | Pass |
| Deterministic tool-call loops | Pass |
| Buffered executable tool timing | Pass |
| Comparable tool total-duration validation | Pass |
| Empirical token estimation | Implemented; limitations documented |
| Original paired-runtime TTFT validation | Pass |
| Broader controlled-text TTFT generalization | Partial pass |
| Controlled text total-duration validation | Unresolved |
| Docker lifecycle and automated tests | Pass |
| Reproducible experimental evidence | Pass |
| Overall M1.5 experimental validation | Partially complete |

The remaining limits are:

1. **Controlled text duration:** no supplementary sample met both the predeclared exact-output
   and token-length equivalence rules, so text total-duration accuracy remains unverified.
2. **TTFT generalization:** the November 15 workload passed 24/24 measurements, while the
   broader November 16 controlled-text workload passed 7/15.
3. **Prompt-token estimation:** final paired error was 10.01% median and 61.18% maximum. The
   estimator is a lightweight benchmark approximation, not tokenizer-equivalent.
4. **Output equivalence:** future work must declare semantic, canonical, completion-length,
   executable-action, and computational-workload equivalence before collecting final data.
   The unsuccessful November 16 run is not retroactively reclassified.
5. **Environment dependence:** calibration applies to the recorded model, quantization,
   hardware, software, prompt template, and workload scope; unrestricted generalization is not
   claimed.

Follow-up Issue [#17](https://github.com/Atakan8479/predictive-agent/issues/17) is deliberately
scheduled after the representative M1.8–M1.10 synchronous path exists and before M1.11 final
benchmark collection or later M2/M4 comparative conclusions.

## M1.6 handoff

M1.6 can proceed because it depends on the implemented provider contract, not unrestricted
timing generalization. The available interfaces and configuration are:

- Ollama and the mock are both exposed through OpenAI-compatible base URLs selected by
  `PA_LLM_BASE_URL`, `PA_LLM_MODEL`, and `PA_LLM_API_KEY`.
- Ollama-specific `PA_LLM_THINK`, `PA_OLLAMA_URL`, and `PA_OLLAMA_KEEP_ALIVE` are centralized in
  `Settings`; `PA_HTTP_TIMEOUT_S` supplies the current network timeout default.
- The mock serves regular and SSE `POST /v1/chat/completions` plus `GET /healthz` at the Compose
  service published on port 8090 by default.
- Mock timing remains frozen at 77.282 ms base latency, 3,089.067 prompt tokens/s, and 64.340
  completion tokens/s, with existing `PA_MOCK_LLM_*` settings and request-header overrides.
- Regular responses, streaming, deterministic calls for all five tools, post-tool responses,
  usage accounting, and explicit validation failures are covered by unit and integration tests.

Issue #6 should add the provider-independent asynchronous client, absolute deadline, bounded
connection/429 retry behavior, and `agent.llm_turn` instrumentation without changing the mock
runtime. The unresolved text-timing limits matter for later comparative performance claims, not
for provider switching or client correctness.

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

The frozen supplementary controlled-text dataset is selected separately:

```bash
uv run python -m scripts.validate_mock_llm_runtime \
  --text-duration --repetitions 3 \
  --output docs/benchmarks/mock-llm-text-duration-YYYY-MM-DD.json
```

`--final` and `--text-duration` are mutually exclusive.

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
