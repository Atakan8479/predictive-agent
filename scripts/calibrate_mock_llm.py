"""Measure Ollama token counts and phase timings for the mock LLM."""

import argparse
import hashlib
import json
from statistics import median
from typing import Any, cast

import httpx

NS_PER_SECOND = 1_000_000_000
LENGTHS = (64, 128, 256, 512, 1024, 2048)
CORPORA = (
    "The customer asked for an order update and a concise explanation of the next action. ",
    "Müşteri sipariş durumunu ve sonraki adımın kısa bir açıklamasını istiyor. ",
    'const result = await lookup({"order_id": "ORD-0042", "include_history": true}); ',
    '{"customer":"CUST-021","items":[{"sku":"SKU-0007","quantity":2}],"priority":"high"} ',
)
CORPUS_NAMES = ("english", "turkish", "code", "json")


def _object(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Ollama returned a non-object response")
    return cast(dict[str, Any], data)


def _integer(data: dict[str, Any], field: str) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Ollama field {field!r} must be a non-negative integer")
    return value


def character_features(text: str) -> tuple[float, float, float, float]:
    non_ascii = sum(not char.isascii() for char in text)
    punctuation = sum(char.isascii() and not char.isalnum() and not char.isspace() for char in text)
    plain = len(text) - non_ascii - punctuation
    return 1.0, float(plain), float(punctuation), float(non_ascii)


def _prompt(corpus_name: str, corpus: str, length: int) -> str:
    marker = hashlib.sha256(f"{corpus_name}:{length}".encode()).hexdigest()[:16] + " "
    body_length = max(0, length - len(marker))
    return marker + (corpus * (body_length // len(corpus) + 1))[:body_length]


def _fit(samples: list[dict[str, float | int | str]]) -> tuple[float, ...]:
    rows = [character_features(str(sample["prompt"])) for sample in samples]
    targets = [float(sample["prompt_tokens"]) for sample in samples]
    size = len(rows[0])
    matrix = [
        [sum(row[left] * row[right] for row in rows) for right in range(size)]
        + [sum(row[left] * target for row, target in zip(rows, targets, strict=True))]
        for left in range(size)
    ]
    for pivot in range(size):
        best = max(range(pivot, size), key=lambda index: abs(matrix[index][pivot]))
        matrix[pivot], matrix[best] = matrix[best], matrix[pivot]
        divisor = matrix[pivot][pivot]
        if abs(divisor) < 1e-12:
            raise ValueError("Calibration matrix is singular")
        matrix[pivot] = [value / divisor for value in matrix[pivot]]
        for row_index in range(size):
            if row_index == pivot:
                continue
            factor = matrix[row_index][pivot]
            matrix[row_index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(matrix[row_index], matrix[pivot], strict=True)
            ]
    return tuple(matrix[index][-1] for index in range(size))


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / sum(
        (x - x_mean) ** 2 for x in xs
    )
    return y_mean - slope * x_mean, slope


def calibrate(base_url: str, model: str) -> dict[str, object]:
    samples: list[dict[str, float | int | str]] = []
    timeout = httpx.Timeout(120.0, connect=5.0)
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        _measure(client, model, "warmup", "warmup-unique-prefix " + CORPORA[0][:64])
        for corpus_name, corpus in zip(CORPUS_NAMES, CORPORA, strict=True):
            for length in LENGTHS:
                prompt = _prompt(corpus_name, corpus, length)
                samples.append(_measure(client, model, corpus_name, prompt))

    coefficients = _fit(samples)
    for sample in samples:
        predicted = max(
            1,
            round(
                sum(
                    coefficient * feature
                    for coefficient, feature in zip(
                        coefficients, character_features(str(sample["prompt"])), strict=True
                    )
                )
            ),
        )
        actual = int(sample["prompt_tokens"])
        sample["predicted_tokens"] = predicted
        sample["absolute_error_percent"] = abs(predicted - actual) / actual * 100
        del sample["prompt"]

    actual_tokens = [float(sample["prompt_tokens"]) for sample in samples]
    actual_ttfts = [
        float(sample["overhead_seconds"])
        + float(sample["prompt_seconds"])
        + float(sample["output_seconds"]) / int(sample["output_tokens"])
        for sample in samples
    ]
    base_seconds, seconds_per_token = _linear_fit(actual_tokens, actual_ttfts)
    decode_tps = median(float(sample["decode_tps"]) for sample in samples)
    for sample, actual_ttft in zip(samples, actual_ttfts, strict=True):
        predicted_ttft = (
            max(0.0, base_seconds) + int(sample["predicted_tokens"]) * seconds_per_token
        )
        actual_total = (
            float(sample["overhead_seconds"])
            + float(sample["prompt_seconds"])
            + float(sample["output_seconds"])
        )
        predicted_total = predicted_ttft + int(sample["output_tokens"]) / decode_tps
        sample["actual_ttft_seconds"] = actual_ttft
        sample["predicted_ttft_seconds"] = predicted_ttft
        sample["ttft_error_percent"] = abs(predicted_ttft - actual_ttft) / actual_ttft * 100
        sample["predicted_total_seconds"] = predicted_total
        sample["total_error_percent"] = abs(predicted_total - actual_total) / actual_total * 100

    return {
        "model": model,
        "ollama_url": base_url,
        "regression": {
            "formula": (
                "prompt_tokens = intercept + plain*plain_coefficient + "
                "punctuation*punctuation_coefficient + non_ascii*non_ascii_coefficient"
            ),
            "intercept": coefficients[0],
            "plain_coefficient": coefficients[1],
            "punctuation_coefficient": coefficients[2],
            "non_ascii_coefficient": coefficients[3],
            "median_absolute_error_percent": median(
                float(sample["absolute_error_percent"]) for sample in samples
            ),
            "max_absolute_error_percent": max(
                float(sample["absolute_error_percent"]) for sample in samples
            ),
        },
        "timing": {
            "base_latency_ms": max(0.0, base_seconds) * 1000,
            "prefill_tps": 1 / seconds_per_token,
            "decode_tps": decode_tps,
            "median_ttft_error_percent": median(
                float(sample["ttft_error_percent"]) for sample in samples
            ),
            "max_ttft_error_percent": max(
                float(sample["ttft_error_percent"]) for sample in samples
            ),
            "median_total_error_percent": median(
                float(sample["total_error_percent"]) for sample in samples
            ),
            "max_total_error_percent": max(
                float(sample["total_error_percent"]) for sample in samples
            ),
        },
        "samples": samples,
    }


def _measure(
    client: httpx.Client, model: str, corpus_name: str, prompt: str
) -> dict[str, float | int | str]:
    data = _object(
        client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {"num_predict": 16, "temperature": 0, "seed": 42},
            },
        )
    )
    prompt_tokens = _integer(data, "prompt_eval_count")
    cached_tokens = _integer(data, "prompt_eval_cached_count")
    prompt_seconds = _integer(data, "prompt_eval_duration") / NS_PER_SECOND
    output_tokens = _integer(data, "eval_count")
    output_seconds = _integer(data, "eval_duration") / NS_PER_SECOND
    total_seconds = _integer(data, "total_duration") / NS_PER_SECOND
    load_seconds = _integer(data, "load_duration") / NS_PER_SECOND
    uncached_tokens = max(0, prompt_tokens - cached_tokens)
    return {
        "prompt": prompt,
        "characters": len(prompt),
        "corpus": corpus_name,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "prompt_seconds": prompt_seconds,
        "prefill_tps": uncached_tokens / prompt_seconds if prompt_seconds else 0.0,
        "output_tokens": output_tokens,
        "output_seconds": output_seconds,
        "decode_tps": output_tokens / output_seconds if output_seconds else 0.0,
        "load_seconds": load_seconds,
        "overhead_seconds": max(
            0.0, total_seconds - load_seconds - prompt_seconds - output_seconds
        ),
        "total_seconds": total_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    args = parser.parse_args()
    print(json.dumps(calibrate(args.ollama_url, args.model), indent=2))


if __name__ == "__main__":
    main()
