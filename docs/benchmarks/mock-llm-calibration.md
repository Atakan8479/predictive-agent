# Mock LLM calibration

The mock was recalibrated on 2026-11-14 against Ollama 0.40.16 and
`qwen3.5:9b-q4_K_M` (Q4_K_M). The final run used Python 3.12.3 under WSL2 on an
NVIDIA GeForce RTX 5060 Ti (8,151 MiB, driver 610.88) with 15 GiB of memory visible
to WSL. Run `423b33f257531f35` is preserved in
[`mock-llm-calibration-2026-11-14.json`](mock-llm-calibration-2026-11-14.json).

Recalibrate with a new output path; evidence files are never overwritten:

```bash
uv run python -m scripts.calibrate_mock_llm \
  --ollama-url http://127.0.0.1:11434 \
  --model qwen3.5:9b-q4_K_M \
  --repetitions 2 \
  --output docs/benchmarks/mock-llm-calibration-YYYY-MM-DD.json
```

## Method

The fit and final validation datasets are separate. Training contains 35 cases and 70
measurements: short, medium, and long English, Turkish, code, and JSON prompts; varied
conversation histories and tool definitions; all five tools; post-tool answers; and 16,
32, and 64-token output budgets. A previously used ten-case set is development-only.
The final ten cases were introduced after development and measured twice only after the
model and acceptance threshold were frozen.

Every request uses streaming `/api/chat`, reasoning disabled, temperature zero, seed 42,
and `cache_prompt=false`. A unique run and repetition prefix prevents earlier or same-run
prompt reuse. The maximum reported cached prompt-token count was zero. Warm-up is excluded
from fitting and evaluation.

The script records headers, first event, first meaningful text, executable tool-call
arrival, final event, total wall time, Ollama counters, generated output, features,
predictions, and errors. Case results use the median of two repetitions; raw repetitions
remain in the JSON evidence.

## Runtime model

Prompt and completion token counts use shared canonical serialization and feature
extraction in `mocks/llm/calibration.py`. Prompt fitting separates character features
from message-count and serialized-tool-schema effects. Text and tool-call completions use
separate non-negative regressions.

| Timing parameter | Fitted value |
|---|---:|
| Base latency | 75.460 ms |
| Effective prefill throughput | 3,503.696 tokens/s |
| Decode throughput | 68.048 tokens/s |

For text, first meaningful output is `base + prompt_tokens / prefill_tps`; subsequent
tokens arrive at the decode rate. Ollama buffers tool JSON, so an executable tool call is
released after `base + prompt_tokens / prefill_tps + completion_tokens / decode_tps`.
Regular and SSE responses use the same total-duration equation, and decoding is counted
once.

Request headers override container environment values, which override `Settings`
defaults. Base latency must be finite and non-negative; throughputs must be finite and
positive.

## Independent final validation

The original M1.5 gate is ±15% for timing. All final case medians and all 20 individual
measurements passed.

| Metric | Median error | p95 | Maximum | Passed |
|---|---:|---:|---:|---:|
| First meaningful output | 6.32% | 11.72% | 11.72% | 10/10 cases |
| Executable tool ready | 5.86% | 11.72% | 11.72% | 5/5 tool cases |
| Conditional total, observed output tokens | 6.38% | 8.85% | 8.85% | reported, not independent |
| End-to-end total, estimated output tokens | 6.47% | 14.14% | 14.14% | 10/10 cases |

The raw-measurement maxima were 12.42% for first meaningful/tool-ready latency and
14.99% for independent end-to-end duration. The prior tool-call TTFT error was 74.49%;
the buffered model reduces the final tool-ready maximum to 11.72%.

Prompt-token error was 8.00% median and 30.16% maximum. Completion-token error was 5.08%
median and 25.00% maximum. These are reported separately because timing can still pass
when token-estimation errors offset fitted throughput. The conditional total metric uses
observed Ollama output-token counts and is explicitly not presented as independent runtime
accuracy; the end-to-end metric uses estimated completion tokens.

## Supported deterministic schema subset

Tool parameters support object schemas with `properties`, `required`, and boolean
`additionalProperties`; primitive `string`, `integer`, `number`, `boolean`, and `null`
types; and `enum`, `const`, `default`, `pattern`, `minLength`, `maxLength`, `minimum`, and
`maximum`. Unsupported or malformed constraints fail explicitly. Defaults, constants, and
single-value enums must satisfy every applicable constraint before use.

## Limits

The fit is specific to this model, quantization, Ollama version, prompt template, and host.
Recalibrate after changing any of them. Final validation covers deterministic 16–64 token
outputs and the five M1.3 tools; it does not claim accuracy for arbitrary models, long-form
generation, reasoning mode, concurrent GPU saturation, or other JSON Schema keywords.
