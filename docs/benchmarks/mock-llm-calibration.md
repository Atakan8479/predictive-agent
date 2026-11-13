# Mock LLM calibration

The mock was calibrated on 2026-11-13 against Ollama 0.40.16 and
`qwen3.5:9b-q4_K_M` (Q4_K_M) on the Windows/WSL2 benchmark host. Run
`uv run python scripts/calibrate_mock_llm.py` to reproduce the measurement.

## Method

The calibration sends warmed, non-streaming Ollama chat requests with reasoning disabled,
temperature zero, a fixed seed, and 16 generated tokens. It measures English, Turkish, code, and
JSON prompts at 64, 128, 256, 512, 1,024, and 2,048 characters. A unique hash prefix prevents
Ollama prefix-cache reuse between samples.

The estimator is a least-squares character-class regression, not a words multiplier:

```text
prompt_tokens = 23.803578
              + 0.182658 * plain_ascii_characters
              + 0.642203 * ascii_punctuation_characters
              + 0.784786 * non_ascii_characters
```

The timing fit uses `TTFT = base + estimated_prompt_tokens / prefill_tps` and
`total = TTFT + completion_tokens / decode_tps`.

| Parameter | Calibrated value |
|---|---:|
| Base latency | 83.783 ms |
| Effective prefill throughput | 3,422.979 tokens/s |
| Decode throughput | 66.979 tokens/s |

The planned 960/67 tokens/s values were treated as starting values. Decode throughput matched,
but the local warmed measurements supported a higher effective prefill slope plus an explicit
83.783 ms base. Keeping 960 tokens/s would overstate TTFT and fail the measured error target.

## Prediction error

The table reports the worst error across the four prompt classes at each length.

| Characters | Token error | TTFT error | Total-duration error |
|---:|---:|---:|---:|
| 64 | 4.35% | 10.03% | 2.78% |
| 128 | 3.64% | 3.03% | 5.75% |
| 256 | 2.15% | 3.80% | 5.97% |
| 512 | 2.56% | 2.94% | 5.35% |
| 1,024 | 0.95% | 9.51% | 8.10% |
| 2,048 | 0.27% | 6.23% | 4.15% |

Across all 24 samples, median/max errors were:

- Prompt tokens: 0.74% / 4.35%.
- TTFT: 2.90% / 10.03%.
- Total duration: 4.29% / 8.10%.

These values describe this model and host. Recalibrate after changing the model, quantization,
Ollama version, or benchmark hardware. Request headers intentionally allow controlled experiments
without changing the stored calibration.
