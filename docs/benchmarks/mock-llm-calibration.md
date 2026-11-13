# Mock LLM calibration

The mock was recalibrated on 2026-11-13 UTC against Ollama 0.40.16 and
`qwen3.5:9b-q4_K_M` (Q4_K_M) from Python 3.12.3 under WSL2. The measured run ID was
`a309df1d288f5fdf`. The script prints the full structured evidence as JSON:

```bash
uv run python scripts/calibrate_mock_llm.py \
  --ollama-url http://127.0.0.1:11434 \
  --model qwen3.5:9b-q4_K_M
```

## Method

The fitting and validation datasets are separate. The fit uses 29 samples: 24 English, Turkish,
code, and JSON prompts from 64 to 2,048 characters, plus five conversation/tool workloads. The
timing fit uses the 26 prompt-only samples. Native Ollama tool calls are excluded from the prefill
fit because Ollama buffers their JSON until the complete tool call is available; they remain in the
held-out validation and are not hidden from the reported errors.

The held-out set contains ten different prompts: English, Turkish, code, JSON, and one agent-loop
prompt with four messages and an actual tool definition. None is used to fit or tune coefficients.
A warm-up request is also excluded from fitting.

All requests use streaming `/api/chat`, reasoning disabled, temperature zero, seed 42, and a fixed
64-token output budget. TTFT is wall time from request start to the first non-empty content or tool
call event. Time to response headers, the first non-empty NDJSON event, the final event, total wall
time, and Ollama duration counters are recorded separately. Empty metadata events are never counted
as generated output.

Prompt caching is disabled with `cache_prompt=false`. A unique run prefix and a distinct prefix for
every sample prevent reuse from earlier or same-run requests. Both the training and held-out sets
reported a maximum of zero cached prompt tokens; no measurement was excluded based on cache state.

## Fitted model

```text
prompt_tokens = 36.757288
              + 0.183034 * plain_ascii_characters
              + 0.688019 * ascii_punctuation_characters
              + 0.751300 * non_ascii_characters
              + 16.340530 * additional_messages
              + 111.804631 * tools
```

| Timing parameter | Fitted value |
|---|---:|
| Base latency | 94.775 ms |
| Effective prefill throughput | 3,146.301 tokens/s |
| Decode throughput | 65.790 tokens/s |

The mock preserves these equations:

```text
TTFT = base_latency + estimated_prompt_tokens / prefill_tps
Total_duration = TTFT + generated_tokens / decode_tps
```

## Independent validation

| Held-out sample | Tokens actual / predicted | TTFT actual / predicted | TTFT error | Total actual / predicted | Total error |
|---|---:|---:|---:|---:|---:|
| English 96 | 62 / 61 | 0.111 / 0.114 s | 3.25% | 1.201 / 1.087 s | 9.49% |
| English 768 | 182 / 198 | 0.146 / 0.158 s | 8.04% | 1.241 / 1.131 s | 8.88% |
| English 1,536 | 275 / 334 | 0.184 / 0.201 s | 9.06% | 1.271 / 1.174 s | 7.67% |
| Turkish 192 | 88 / 91 | 0.127 / 0.124 s | 2.69% | 1.211 / 1.096 s | 9.48% |
| Turkish 1,024 | 380 / 288 | 0.191 / 0.186 s | 2.52% | 1.280 / 1.159 s | 9.47% |
| Code 384 | 140 / 139 | 0.143 / 0.139 s | 3.05% | 1.229 / 1.112 s | 9.53% |
| Code 1,280 | 399 / 373 | 0.232 / 0.213 s | 7.99% | 1.318 / 1.186 s | 10.01% |
| JSON 320 | 145 / 157 | 0.141 / 0.145 s | 2.69% | 1.227 / 1.117 s | 8.92% |
| JSON 1,408 | 462 / 547 | 0.246 / 0.269 s | 9.34% | 1.349 / 1.241 s | 7.96% |
| Tools and history | 346 / 312 | 0.760 / 0.194 s | 74.49% | 0.776 / 0.680 s | 12.38% |

| Held-out metric | Median | p95 | Maximum | Every sample within ±15% |
|---|---:|---:|---:|---:|
| Prompt-token error | 8.53% | 24.21% | 24.21% | Not an acceptance gate |
| TTFT error | 5.62% | 74.49% | 74.49% | **No** |
| Total-duration error | 9.47% | 12.38% | 12.38% | **Yes** |

With ten held-out observations, nearest-rank p95 is the maximum. The prompt-only TTFT samples all
remain below 10%; the independent overall TTFT criterion fails because the native tool-call event
is not emitted until Ollama has decoded and buffered the complete JSON call. Folding that hidden
decode delay into prefill throughput would improve this one case while biasing normal prompts, so
the measurement is retained unchanged. Total-duration accuracy passes the ±15% criterion for every
held-out sample.

The token estimator's largest held-out error is Turkish prose at 24.21%. This does not prevent the
observed TTFT fit from staying within 10% on prompt-only samples, but it limits portability to other
languages and tokenizer templates.

## Recalibration limits

These values describe this model, quantization, Ollama version, and WSL2 host. Recalibrate after
changing any of them. The script uses finite HTTPX timeouts, rejects singular regressions and
non-positive denominators, emits all training and held-out samples in JSON, and never overwrites an
evidence file. Redirect output to a new, run-specific file when durable raw evidence is required.
