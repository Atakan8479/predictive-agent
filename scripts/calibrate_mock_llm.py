"""Fit and independently validate the mock LLM against streamed Ollama responses."""

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from statistics import median
from time import perf_counter, time_ns
from typing import Any, TypedDict, cast

import httpx

NS_PER_SECOND = 1_000_000_000
OUTPUT_TOKEN_BUDGET = 64
ACCEPTANCE_PERCENT = 15.0
TRAINING_LENGTHS = (64, 128, 256, 512, 1024, 2048)
TRAINING_CORPORA = {
    "english": "The customer asked for an order update and a concise next action. ",
    "turkish": "Müşteri sipariş durumunu ve sonraki adımın kısa açıklamasını istiyor. ",
    "code": 'const result = await lookup({"order_id":"ORD-0042","history":true}); ',
    "json": '{"customer":"CUST-021","items":[{"sku":"SKU-0007","quantity":2}]} ',
}


class Case(TypedDict):
    name: str
    category: str
    messages: list[dict[str, str]]
    tools: list[dict[str, Any]]


def character_features(text: str) -> tuple[float, float, float, float]:
    non_ascii = sum(not char.isascii() for char in text)
    punctuation = sum(char.isascii() and not char.isalnum() and not char.isspace() for char in text)
    plain = len(text) - non_ascii - punctuation
    return 1.0, float(plain), float(punctuation), float(non_ascii)


def training_cases() -> list[Case]:
    cases = [
        _single_message_case(f"train-{category}-{length}", category, corpus, length)
        for category, corpus in TRAINING_CORPORA.items()
        for length in TRAINING_LENGTHS
    ]
    cases.extend(
        [
            {
                "name": "train-conversation-two",
                "category": "conversation",
                "messages": [
                    {"role": "system", "content": f"{_marker('train-conversation-two')} Be exact."},
                    {"role": "user", "content": "Summarize the verified order state."},
                ],
                "tools": [],
            },
            {
                "name": "train-conversation-four",
                "category": "conversation",
                "messages": [
                    {
                        "role": "system",
                        "content": f"{_marker('train-conversation-four')} Be exact.",
                    },
                    {"role": "user", "content": "Check the available record."},
                    {"role": "assistant", "content": "I will use only verified facts."},
                    {"role": "user", "content": "Return a concise answer."},
                ],
                "tools": [],
            },
            {
                "name": "train-tool-one",
                "category": "tools",
                "messages": [
                    {
                        "role": "user",
                        "content": f"{_marker('train-tool-one')} Find customer CUST-021.",
                    }
                ],
                "tools": [_lookup_tool("get_customer", "customer_id")],
            },
            {
                "name": "train-tool-two",
                "category": "tools",
                "messages": [
                    {
                        "role": "user",
                        "content": f"{_marker('train-tool-two')} Find product SKU-0007.",
                    }
                ],
                "tools": [
                    _lookup_tool("get_customer", "customer_id"),
                    _lookup_tool("get_product", "sku"),
                ],
            },
            {
                "name": "train-tool-history",
                "category": "tools-and-history",
                "messages": [
                    {
                        "role": "system",
                        "content": f"{_marker('train-tool-history')} Verify external facts.",
                    },
                    {"role": "user", "content": "Find product SKU-0007."},
                    {"role": "assistant", "content": "I will check inventory."},
                    {"role": "user", "content": "Use the tool before answering."},
                ],
                "tools": [_lookup_tool("get_product", "sku")],
            },
        ]
    )
    return cases


def validation_cases() -> list[Case]:
    cases = [
        _single_message_case(
            "heldout-english-96",
            "english",
            "A support agent needs a precise summary of the customer's current request. ",
            96,
        ),
        _single_message_case(
            "heldout-english-768",
            "english",
            "Explain the verified order state, relevant evidence, and safest next action. ",
            768,
        ),
        _single_message_case(
            "heldout-english-1536",
            "english",
            "Write a factual service summary without inventing unavailable customer details. ",
            1536,
        ),
        _single_message_case(
            "heldout-turkish-192",
            "turkish",
            "Sipariş bilgisini doğrula ve müşteriye açık, kısa bir sonraki adım sun. ",
            192,
        ),
        _single_message_case(
            "heldout-turkish-1024",
            "turkish",
            "Teslimat kaydını kanıtlara göre özetle; eksik bilgileri tahmin etme. ",
            1024,
        ),
        _single_message_case(
            "heldout-code-384",
            "code",
            "async def resolve(order_id: str) -> dict[str, object]:\n"
            "    return await fetch(order_id)\n",
            384,
        ),
        _single_message_case(
            "heldout-code-1280",
            "code",
            "type Result = { orderId: string; status: string };\n"
            "const verify = async (id: string) => id;\n",
            1280,
        ),
        _single_message_case(
            "heldout-json-320",
            "json",
            '{"event":"order.reviewed","order_id":"ORD-0042","valid":true,"notes":[]} ',
            320,
        ),
        _single_message_case(
            "heldout-json-1408",
            "json",
            '{"shipment":{"order_id":"ORD-0042","state":"in_transit"},"checks":[true,false]} ',
            1408,
        ),
    ]
    marker = _marker("heldout-agent-loop")
    cases.append(
        {
            "name": "heldout-agent-loop",
            "category": "tools-and-history",
            "messages": [
                {
                    "role": "system",
                    "content": f"{marker} Use tools for verified external facts only.",
                },
                {"role": "user", "content": "Find order ORD-0042 and summarize its state."},
                {"role": "assistant", "content": "I will verify the order before answering."},
                {"role": "user", "content": "Keep the answer concise."},
            ],
            "tools": [_lookup_tool("get_order", "order_id")],
        }
    )
    return cases


def _single_message_case(name: str, category: str, corpus: str, length: int) -> Case:
    marker = _marker(name) + " "
    body_length = max(0, length - len(marker))
    content = marker + (corpus * (body_length // len(corpus) + 1))[:body_length]
    return {
        "name": name,
        "category": category,
        "messages": [{"role": "user", "content": content}],
        "tools": [],
    }


def _marker(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:16]


def _mark_cases(cases: list[Case], run_id: str) -> list[Case]:
    marked: list[Case] = []
    for index, case in enumerate(cases):
        messages = [message.copy() for message in case["messages"]]
        messages[0]["content"] = f"{run_id}-{index:02d} {messages[0]['content']}"
        marked.append({**case, "messages": messages})
    return marked


def _lookup_tool(name: str, parameter: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Retrieve one deterministic external record.",
            "parameters": {
                "type": "object",
                "properties": {parameter: {"type": "string"}},
                "required": [parameter],
            },
        },
    }


def estimator_text(case: Case) -> str:
    parts = [message["content"] for message in case["messages"]]
    if case["tools"]:
        parts.append(json.dumps(case["tools"], sort_keys=True))
    return "\n".join(parts)


def regression_features(case: Case) -> tuple[float, ...]:
    return (
        *character_features(estimator_text(case)),
        float(max(0, len(case["messages"]) - 1)),
        float(len(case["tools"])),
    )


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


def fit_regression(rows: list[tuple[float, ...]], targets: list[float]) -> tuple[float, ...]:
    if not rows or len(rows) != len(targets):
        raise ValueError("Calibration rows and targets must be non-empty and aligned")
    size = len(rows[0])
    if size == 0 or any(len(row) != size for row in rows):
        raise ValueError("Calibration rows must have a consistent non-zero width")
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
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Timing fit requires at least two aligned observations")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 1e-12:
        raise ValueError("Timing fit requires varied prompt-token counts")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    slope /= denominator
    if slope <= 0:
        raise ValueError("Measured TTFT did not produce a positive prefill slope")
    return max(0.0, y_mean - slope * x_mean), slope


def _predict_tokens(case: Case, coefficients: tuple[float, ...]) -> int:
    predicted = round(
        sum(
            coefficient * feature
            for coefficient, feature in zip(coefficients, regression_features(case), strict=True)
        )
    )
    return max(1, predicted)


def _percent_error(predicted: float, actual: float) -> float:
    if actual <= 0:
        raise ValueError("Cannot calculate percentage error for a non-positive observation")
    return abs(predicted - actual) / actual * 100


def summarize_errors(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty sample")
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_percent": median(ordered),
        "p95_percent": ordered[p95_index],
        "max_percent": ordered[-1],
    }


def measure_stream(client: httpx.Client, model: str, case: Case) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": case["messages"],
        "stream": True,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "num_predict": OUTPUT_TOKEN_BUDGET,
            "temperature": 0,
            "seed": 42,
            "cache_prompt": False,
        },
    }
    if case["tools"]:
        payload["tools"] = case["tools"]

    started = perf_counter()
    headers_at: float | None = None
    first_event_at: float | None = None
    first_content_at: float | None = None
    final_event_at: float | None = None
    final: dict[str, Any] | None = None
    with client.stream("POST", "/api/chat", json=payload) as response:
        headers_at = perf_counter()
        response.raise_for_status()
        for line in response.iter_lines():
            observed_at = perf_counter()
            if not line:
                continue
            if first_event_at is None:
                first_event_at = observed_at
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Ollama stream event must be an object")
            typed_event = cast(dict[str, Any], event)
            message = typed_event.get("message")
            if isinstance(message, dict):
                typed_message = cast(dict[str, Any], message)
                content = typed_message.get("content")
                tool_calls = typed_message.get("tool_calls")
                if first_content_at is None and (
                    (isinstance(content, str) and bool(content)) or bool(tool_calls)
                ):
                    first_content_at = observed_at
            if typed_event.get("done") is True:
                final = typed_event
                final_event_at = observed_at
    finished = perf_counter()
    if first_event_at is None or first_content_at is None or final is None:
        raise ValueError(
            f"Ollama stream omitted required timing or final metrics for {case['name']!r}"
        )
    if final_event_at is None:
        raise ValueError("Ollama stream omitted the final done event")

    prompt_tokens = _integer(final, "prompt_eval_count")
    cached_tokens = _integer(final, "prompt_eval_cached_count")
    output_tokens = _integer(final, "eval_count")
    output_duration = _integer(final, "eval_duration") / NS_PER_SECOND
    prompt_duration = _integer(final, "prompt_eval_duration") / NS_PER_SECOND
    text = estimator_text(case)
    return {
        "name": case["name"],
        "category": case["category"],
        "characters": len(text),
        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "message_count": len(case["messages"]),
        "tool_count": len(case["tools"]),
        "actual_prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "actual_output_tokens": output_tokens,
        "observed_response_headers_seconds": headers_at - started,
        "observed_first_event_bytes_seconds": first_event_at - started,
        "observed_ttft_seconds": first_content_at - started,
        "observed_total_seconds": finished - started,
        "observed_final_event_seconds": final_event_at - started,
        "ollama_prompt_eval_seconds": prompt_duration,
        "ollama_eval_seconds": output_duration,
        "ollama_total_seconds": _integer(final, "total_duration") / NS_PER_SECOND,
        "ollama_load_seconds": _integer(final, "load_duration") / NS_PER_SECOND,
        "measured_decode_tps": output_tokens / output_duration if output_duration else 0.0,
    }


def _annotate_predictions(
    samples: list[dict[str, Any]],
    cases: list[Case],
    coefficients: tuple[float, ...],
    base_seconds: float,
    seconds_per_prompt_token: float,
    decode_tps: float,
) -> None:
    if decode_tps <= 0:
        raise ValueError("Decode throughput must be positive")
    for sample, case in zip(samples, cases, strict=True):
        predicted_tokens = _predict_tokens(case, coefficients)
        predicted_ttft = base_seconds + predicted_tokens * seconds_per_prompt_token
        predicted_total = predicted_ttft + sample["actual_output_tokens"] / decode_tps
        sample["predicted_prompt_tokens"] = predicted_tokens
        sample["predicted_ttft_seconds"] = predicted_ttft
        sample["predicted_total_seconds"] = predicted_total
        sample["token_error_percent"] = _percent_error(
            predicted_tokens, sample["actual_prompt_tokens"]
        )
        sample["ttft_error_percent"] = _percent_error(
            predicted_ttft, sample["observed_ttft_seconds"]
        )
        sample["total_error_percent"] = _percent_error(
            predicted_total, sample["observed_total_seconds"]
        )


def calibrate(base_url: str, model: str) -> dict[str, Any]:
    run_id = hashlib.sha256(str(time_ns()).encode()).hexdigest()[:16]
    train_cases = _mark_cases(training_cases(), f"{run_id}-train")
    heldout_cases = _mark_cases(validation_cases(), f"{run_id}-heldout")
    timeout = httpx.Timeout(120.0, connect=5.0, read=120.0, write=10.0, pool=5.0)
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        version = _object(client.get("/api/version")).get("version", "unknown")
        warmup = measure_stream(
            client,
            model,
            _single_message_case(
                "warmup-not-a-sample", "warmup", "Warm the model with unique content. ", 96
            ),
        )
        training = [measure_stream(client, model, case) for case in train_cases]

        coefficients = fit_regression(
            [regression_features(case) for case in train_cases],
            [float(sample["actual_prompt_tokens"]) for sample in training],
        )
        timing_training = [
            sample for sample, case in zip(training, train_cases, strict=True) if not case["tools"]
        ]
        base_seconds, seconds_per_prompt_token = _linear_fit(
            [float(sample["actual_prompt_tokens"]) for sample in timing_training],
            [float(sample["observed_ttft_seconds"]) for sample in timing_training],
        )
        valid_decode_rates = [
            float(sample["measured_decode_tps"])
            for sample in training
            if float(sample["measured_decode_tps"]) > 0
        ]
        if not valid_decode_rates:
            raise ValueError("No positive decode-throughput measurements were recorded")
        decode_tps = median(valid_decode_rates)
        heldout = [measure_stream(client, model, case) for case in heldout_cases]

    _annotate_predictions(
        training,
        train_cases,
        coefficients,
        base_seconds,
        seconds_per_prompt_token,
        decode_tps,
    )
    _annotate_predictions(
        heldout,
        heldout_cases,
        coefficients,
        base_seconds,
        seconds_per_prompt_token,
        decode_tps,
    )
    heldout_errors = {
        "prompt_tokens": summarize_errors(
            float(sample["token_error_percent"]) for sample in heldout
        ),
        "ttft": summarize_errors(float(sample["ttft_error_percent"]) for sample in heldout),
        "total_duration": summarize_errors(
            float(sample["total_error_percent"]) for sample in heldout
        ),
    }
    for metric in ("ttft", "total_duration"):
        heldout_errors[metric]["all_samples_within_acceptance"] = (
            heldout_errors[metric]["max_percent"] <= ACCEPTANCE_PERCENT
        )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "ollama_url": base_url,
            "ollama_version": version,
            "model": model,
            "python_version": platform.python_version(),
            "httpx_version": httpx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "output_token_budget": OUTPUT_TOKEN_BUDGET,
            "temperature": 0,
            "seed": 42,
            "reasoning_enabled": False,
            "warmup_excluded_from_fit": True,
            "acceptance_percent": ACCEPTANCE_PERCENT,
            "run_id": run_id,
        },
        "method": {
            "training_samples": len(training),
            "timing_fit_samples": len(timing_training),
            "heldout_samples": len(heldout),
            "ttft": "direct wall time to first non-empty content or tool-call stream event",
            "first_event_bytes": "wall time to the first non-empty NDJSON stream line",
            "total": "wall time through stream close after the final done event",
            "timing_fit_scope": (
                "Prompt-only training samples; Ollama buffers native tool calls until their "
                "JSON is complete, which is measured in validation but not fit as prefill."
            ),
            "cache_control": (
                "cache_prompt=false and a per-run, per-sample prefix prevent prior or "
                "within-run prefix reuse; cached counts are recorded without exclusions."
            ),
            "maximum_training_cached_prompt_tokens": max(
                int(sample["cached_prompt_tokens"]) for sample in training
            ),
            "maximum_heldout_cached_prompt_tokens": max(
                int(sample["cached_prompt_tokens"]) for sample in heldout
            ),
        },
        "fit": {
            "token_regression": {
                "intercept": coefficients[0],
                "plain_coefficient": coefficients[1],
                "punctuation_coefficient": coefficients[2],
                "non_ascii_coefficient": coefficients[3],
                "extra_message_coefficient": coefficients[4],
                "tool_coefficient": coefficients[5],
            },
            "timing": {
                "base_latency_ms": base_seconds * 1000,
                "prefill_tps": 1 / seconds_per_prompt_token,
                "decode_tps": decode_tps,
            },
        },
        "warmup": warmup,
        "training_summary": {
            "prompt_tokens": summarize_errors(
                float(sample["token_error_percent"]) for sample in training
            ),
            "ttft": summarize_errors(float(sample["ttft_error_percent"]) for sample in training),
            "total_duration": summarize_errors(
                float(sample["total_error_percent"]) for sample in training
            ),
        },
        "heldout_summary": heldout_errors,
        "training_samples": training,
        "heldout_samples": heldout,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    args = parser.parse_args()
    try:
        result = calibrate(args.ollama_url, args.model)
    except (httpx.HTTPError, OSError, ValueError) as error:
        raise SystemExit(f"calibration failed: {error}") from error
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
