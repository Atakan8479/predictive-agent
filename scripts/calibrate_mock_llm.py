"""Fit and independently validate the mock LLM against streamed Ollama responses."""

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter, time_ns
from typing import Any, NotRequired, TypedDict, cast

import httpx
from mocks.llm.calibration import (
    character_features,
    estimate_content_tokens,
    prompt_features,
    serialize_completion,
    serialize_prompt,
)

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
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    output_tokens: NotRequired[int]
    base_name: NotRequired[str]


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
            _final_tool_case(
                "train-tool-order",
                "Use get_order for ORD-0008.",
                "get_order",
                "order_id",
            ),
            _final_tool_case(
                "train-tool-shipment",
                "Use get_shipment to track ORD-0019.",
                "get_shipment",
                "order_id",
            ),
            _final_tool_case(
                "train-tool-policy",
                "Use get_policy for the returns policy.",
                "get_policy",
                "policy_name",
            ),
            {
                **_single_message_case(
                    "train-output-short",
                    "english",
                    "Answer with one short verified sentence. ",
                    240,
                ),
                "output_tokens": 16,
            },
            {
                **_single_message_case(
                    "train-output-medium",
                    "turkish",
                    "Doğrulanmış kayıtları kısa biçimde özetle. ",
                    480,
                ),
                "output_tokens": 32,
            },
            {
                "name": "train-after-tool-result",
                "category": "tool-result-final",
                "messages": [
                    {"role": "system", "content": "Answer only from the tool result."},
                    {"role": "user", "content": "What is the order status?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_train",
                                "type": "function",
                                "function": {
                                    "name": "get_order",
                                    "arguments": {"order_id": "ORD-0021"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_train",
                        "content": '{"order_id":"ORD-0021","status":"paid"}',
                    },
                ],
                "tools": [_lookup_tool("get_order", "order_id")],
                "output_tokens": 32,
            },
        ]
    )
    return cases


def development_cases() -> list[Case]:
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


def validation_cases() -> list[Case]:
    """Return final cases kept separate from the earlier development/held-out set."""
    return [
        {
            **_single_message_case(
                "acceptance-english-short",
                "english",
                "Summarize the confirmed request and recommend one safe next action. ",
                208,
            ),
            "output_tokens": 16,
        },
        {
            **_single_message_case(
                "acceptance-turkish-medium",
                "turkish",
                "Doğrulanmış müşteri kaydını özetle ve belirsiz ayrıntıları açıkça belirt. ",
                896,
            ),
            "output_tokens": 32,
        },
        {
            **_single_message_case(
                "acceptance-code-long",
                "code",
                "async def verify(identifier: str, payload: dict[str, object]) -> bool:\n"
                "    return bool(identifier and payload)\n",
                1664,
            ),
            "output_tokens": 64,
        },
        {
            **_single_message_case(
                "acceptance-json-medium",
                "json",
                '{"trace":{"kind":"customer","accepted":true},"items":[{"count":4}]} ',
                960,
            ),
            "output_tokens": 32,
        },
        _final_tool_case(
            "acceptance-tool-order",
            "Use the order lookup to fetch ORD-0096.",
            "get_order",
            "order_id",
        ),
        _final_tool_case(
            "acceptance-tool-shipment",
            "Track the shipment and delivery for order ORD-0114.",
            "get_shipment",
            "order_id",
            extra_tools=[_lookup_tool("get_order", "order_id")],
        ),
        _final_tool_case(
            "acceptance-tool-customer",
            "Fetch the verified profile for customer CUST-058.",
            "get_customer",
            "customer_id",
        ),
        _final_tool_case(
            "acceptance-tool-policy",
            "Use the policy lookup to retrieve returns.",
            "get_policy",
            "policy_name",
        ),
        _final_tool_case(
            "acceptance-tool-product",
            "Check product inventory for SKU-0194.",
            "get_product",
            "sku",
            extra_tools=[_lookup_tool("get_customer", "customer_id")],
        ),
        {
            "name": "acceptance-after-tool-result",
            "category": "tool-result-final",
            "messages": [
                {
                    "role": "system",
                    "content": "Answer from the supplied verified tool result only.",
                },
                {"role": "user", "content": "What is the status of order ORD-0150?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_final",
                            "type": "function",
                            "function": {
                                "name": "get_order",
                                "arguments": {"order_id": "ORD-0150"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_final",
                    "content": '{"order_id":"ORD-0150","status":"delivered"}',
                },
            ],
            "tools": [_lookup_tool("get_order", "order_id")],
            "output_tokens": 32,
        },
    ]


def _final_tool_case(
    name: str,
    prompt: str,
    tool_name: str,
    parameter: str,
    *,
    extra_tools: list[dict[str, Any]] | None = None,
) -> Case:
    return {
        "name": name,
        "category": "tool-call",
        "messages": [
            {"role": "system", "content": "Call the requested tool with the exact identifier."},
            {"role": "user", "content": prompt},
        ],
        "tools": [*(extra_tools or []), _lookup_tool(tool_name, parameter)],
        "output_tokens": 64,
    }


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


def _mark_cases(cases: list[Case], run_id: str, repetitions: int = 1) -> list[Case]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    marked: list[Case] = []
    for repetition in range(1, repetitions + 1):
        for index, case in enumerate(cases):
            messages = [message.copy() for message in case["messages"]]
            content = messages[0].get("content")
            if not isinstance(content, str):
                raise ValueError("the first calibration message must contain text")
            messages[0]["content"] = f"{run_id}-r{repetition}-{index:02d} {content}"
            marked.append(
                {
                    **case,
                    "name": f"{case['name']}-r{repetition}",
                    "base_name": case["name"],
                    "messages": messages,
                }
            )
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
    return serialize_prompt(case["messages"], case["tools"])


def regression_features(case: Case) -> tuple[float, ...]:
    return prompt_features(case["messages"], case["tools"])


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


def fit_selected_regression(
    rows: list[tuple[float, ...]], targets: list[float], indices: tuple[int, ...]
) -> tuple[float, ...]:
    compact = fit_nonnegative_regression(
        [tuple(row[index] for index in indices) for row in rows], targets
    )
    return tuple(
        compact[indices.index(index)] if index in indices else 0.0 for index in range(len(rows[0]))
    )


def fit_nonnegative_regression(
    rows: list[tuple[float, ...]], targets: list[float]
) -> tuple[float, ...]:
    if not rows:
        raise ValueError("Calibration rows must not be empty")
    active = list(range(len(rows[0])))
    result = [0.0] * len(rows[0])
    while active:
        compact = fit_regression([tuple(row[index] for index in active) for row in rows], targets)
        negative = [index for index, value in zip(active, compact, strict=True) if value < 0]
        if not negative:
            for index, value in zip(active, compact, strict=True):
                result[index] = value
            return tuple(result)
        active = [index for index in active if index not in negative]
    raise ValueError("Non-negative regression has no active features")


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
            "num_predict": case.get("output_tokens", OUTPUT_TOKEN_BUDGET),
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
    first_meaningful_at: float | None = None
    first_text_at: float | None = None
    tool_ready_at: float | None = None
    final_event_at: float | None = None
    final: dict[str, Any] | None = None
    generated_content: list[str] = []
    generated_tool_calls: list[dict[str, Any]] = []
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
                if isinstance(content, str) and content:
                    generated_content.append(content)
                    if first_text_at is None:
                        first_text_at = observed_at
                    if first_meaningful_at is None:
                        first_meaningful_at = observed_at
                if isinstance(tool_calls, list) and tool_calls:
                    generated_tool_calls = cast(list[dict[str, Any]], tool_calls)
                    if tool_ready_at is None:
                        tool_ready_at = observed_at
                    if first_meaningful_at is None:
                        first_meaningful_at = observed_at
            if typed_event.get("done") is True:
                final = typed_event
                final_event_at = observed_at
    finished = perf_counter()
    if first_event_at is None or first_meaningful_at is None or final is None:
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
    generated_text = serialize_completion("".join(generated_content), generated_tool_calls)
    return {
        "name": case["name"],
        "case_name": case.get("base_name", case["name"]),
        "category": case["category"],
        "characters": len(text),
        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "message_count": len(case["messages"]),
        "tool_count": len(case["tools"]),
        "actual_prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "actual_output_tokens": output_tokens,
        "runtime_estimated_output_tokens": estimate_content_tokens(
            generated_text, tool_call=bool(generated_tool_calls)
        ),
        "generated_output_features": list(character_features(generated_text)),
        "generated_output_sha256": hashlib.sha256(generated_text.encode()).hexdigest(),
        "generated_output_characters": len(generated_text),
        "output_kind": "tool_call" if generated_tool_calls else "text",
        "output_token_budget": case.get("output_tokens", OUTPUT_TOKEN_BUDGET),
        "observed_response_headers_seconds": headers_at - started,
        "observed_first_event_bytes_seconds": first_event_at - started,
        "observed_first_meaningful_seconds": first_meaningful_at - started,
        "observed_text_ttft_seconds": (
            first_text_at - started if first_text_at is not None else None
        ),
        "observed_tool_ready_seconds": (
            tool_ready_at - started if tool_ready_at is not None else None
        ),
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
    text_completion_coefficients: tuple[float, ...],
    tool_completion_coefficients: tuple[float, ...],
    base_seconds: float,
    seconds_per_prompt_token: float,
    decode_tps: float,
) -> None:
    if decode_tps <= 0:
        raise ValueError("Decode throughput must be positive")
    for sample, case in zip(samples, cases, strict=True):
        predicted_tokens = _predict_tokens(case, coefficients)
        completion_coefficients = (
            tool_completion_coefficients
            if sample["output_kind"] == "tool_call"
            else text_completion_coefficients
        )
        estimated_output_tokens = max(
            1,
            round(
                sum(
                    coefficient * feature
                    for coefficient, feature in zip(
                        completion_coefficients,
                        sample["generated_output_features"],
                        strict=True,
                    )
                )
            ),
        )
        predicted_ttft = base_seconds + predicted_tokens * seconds_per_prompt_token
        conditional_total = predicted_ttft + sample["actual_output_tokens"] / decode_tps
        predicted_total = predicted_ttft + estimated_output_tokens / decode_tps
        predicted_first_meaningful = (
            predicted_total if sample["output_kind"] == "tool_call" else predicted_ttft
        )
        sample["predicted_prompt_tokens"] = predicted_tokens
        sample["estimated_output_tokens"] = estimated_output_tokens
        sample["predicted_ttft_seconds"] = predicted_ttft
        sample["predicted_first_meaningful_seconds"] = predicted_first_meaningful
        sample["predicted_tool_ready_seconds"] = (
            predicted_total if sample["output_kind"] == "tool_call" else None
        )
        sample["predicted_conditional_total_seconds"] = conditional_total
        sample["predicted_total_seconds"] = predicted_total
        sample["token_error_percent"] = _percent_error(
            predicted_tokens, sample["actual_prompt_tokens"]
        )
        sample["completion_token_error_percent"] = _percent_error(
            sample["estimated_output_tokens"], sample["actual_output_tokens"]
        )
        sample["ttft_error_percent"] = _percent_error(
            predicted_first_meaningful, sample["observed_first_meaningful_seconds"]
        )
        sample["conditional_total_error_percent"] = _percent_error(
            conditional_total, sample["observed_total_seconds"]
        )
        sample["total_error_percent"] = _percent_error(
            predicted_total, sample["observed_total_seconds"]
        )
        sample["tool_ready_error_percent"] = (
            _percent_error(predicted_total, sample["observed_tool_ready_seconds"])
            if sample["observed_tool_ready_seconds"] is not None
            else None
        )


def aggregate_repetitions(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["case_name"]), []).append(sample)
    aggregated: list[dict[str, Any]] = []
    for name, repetitions in grouped.items():
        actual_prompt = median(float(sample["actual_prompt_tokens"]) for sample in repetitions)
        predicted_prompt = median(
            float(sample["predicted_prompt_tokens"]) for sample in repetitions
        )
        actual_completion = median(float(sample["actual_output_tokens"]) for sample in repetitions)
        predicted_completion = median(
            float(sample["estimated_output_tokens"]) for sample in repetitions
        )
        observed_first = median(
            float(sample["observed_first_meaningful_seconds"]) for sample in repetitions
        )
        predicted_first = median(
            float(sample["predicted_first_meaningful_seconds"]) for sample in repetitions
        )
        observed_total = median(float(sample["observed_total_seconds"]) for sample in repetitions)
        predicted_total = median(float(sample["predicted_total_seconds"]) for sample in repetitions)
        predicted_conditional_total = median(
            float(sample["predicted_conditional_total_seconds"]) for sample in repetitions
        )
        tool_observations = [
            float(sample["observed_tool_ready_seconds"])
            for sample in repetitions
            if sample["observed_tool_ready_seconds"] is not None
        ]
        tool_predictions = [
            float(sample["predicted_tool_ready_seconds"])
            for sample in repetitions
            if sample["predicted_tool_ready_seconds"] is not None
        ]
        aggregated.append(
            {
                "case_name": name,
                "category": repetitions[0]["category"],
                "output_kind": repetitions[0]["output_kind"],
                "repetitions": len(repetitions),
                "actual_prompt_tokens": actual_prompt,
                "predicted_prompt_tokens": predicted_prompt,
                "prompt_token_error_percent": _percent_error(predicted_prompt, actual_prompt),
                "actual_completion_tokens": actual_completion,
                "predicted_completion_tokens": predicted_completion,
                "completion_token_error_percent": _percent_error(
                    predicted_completion, actual_completion
                ),
                "observed_first_meaningful_seconds": observed_first,
                "predicted_first_meaningful_seconds": predicted_first,
                "first_meaningful_error_percent": _percent_error(predicted_first, observed_first),
                "observed_tool_ready_seconds": (
                    median(tool_observations) if tool_observations else None
                ),
                "predicted_tool_ready_seconds": (
                    median(tool_predictions) if tool_predictions else None
                ),
                "tool_ready_error_percent": (
                    _percent_error(median(tool_predictions), median(tool_observations))
                    if tool_observations and tool_predictions
                    else None
                ),
                "observed_total_seconds": observed_total,
                "predicted_conditional_total_seconds": predicted_conditional_total,
                "conditional_total_error_percent": _percent_error(
                    predicted_conditional_total, observed_total
                ),
                "predicted_total_seconds": predicted_total,
                "end_to_end_total_error_percent": _percent_error(predicted_total, observed_total),
                "maximum_cached_prompt_tokens": max(
                    int(sample["cached_prompt_tokens"]) for sample in repetitions
                ),
            }
        )
    return aggregated


def calibrate(
    base_url: str, model: str, repetitions: int = 2, *, use_development_cases: bool = False
) -> dict[str, Any]:
    run_id = hashlib.sha256(str(time_ns()).encode()).hexdigest()[:16]
    train_cases = _mark_cases(training_cases(), f"{run_id}-train", repetitions)
    selected_validation_cases = development_cases() if use_development_cases else validation_cases()
    heldout_cases = _mark_cases(selected_validation_cases, f"{run_id}-final", repetitions)
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

        base_pairs = [
            (case, sample)
            for case, sample in zip(train_cases, training, strict=True)
            if case["category"] in TRAINING_CORPORA
            and len(case["messages"]) == 1
            and not case["tools"]
        ]
        base_coefficients = fit_nonnegative_regression(
            [regression_features(case)[:6] for case, _ in base_pairs],
            [float(sample["actual_prompt_tokens"]) for _, sample in base_pairs],
        )
        structured_pairs = [
            (case, sample)
            for case, sample in zip(train_cases, training, strict=True)
            if len(case["messages"]) > 1 or case["tools"]
        ]
        structural_coefficients = fit_nonnegative_regression(
            [regression_features(case)[6:] for case, _ in structured_pairs],
            [
                float(sample["actual_prompt_tokens"])
                - sum(
                    coefficient * feature
                    for coefficient, feature in zip(
                        base_coefficients, regression_features(case)[:6], strict=True
                    )
                )
                for case, sample in structured_pairs
            ],
        )
        coefficients = (*base_coefficients, *structural_coefficients)
        text_completion_training = [
            sample for sample in training if sample["output_kind"] == "text"
        ]
        tool_completion_training = [
            sample for sample in training if sample["output_kind"] == "tool_call"
        ]
        text_completion_coefficients = fit_nonnegative_regression(
            [
                tuple(float(value) for value in sample["generated_output_features"])
                for sample in text_completion_training
            ],
            [float(sample["actual_output_tokens"]) for sample in text_completion_training],
        )
        tool_completion_coefficients = fit_selected_regression(
            [
                tuple(float(value) for value in sample["generated_output_features"])
                for sample in tool_completion_training
            ],
            [float(sample["actual_output_tokens"]) for sample in tool_completion_training],
            (0, 1, 2, 4),
        )
        timing_training = [
            sample
            for sample in training
            if sample["output_kind"] == "text" and sample["observed_text_ttft_seconds"] is not None
        ]
        base_seconds, seconds_per_prompt_token = _linear_fit(
            [float(sample["actual_prompt_tokens"]) for sample in timing_training],
            [float(sample["observed_text_ttft_seconds"]) for sample in timing_training],
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
        text_completion_coefficients,
        tool_completion_coefficients,
        base_seconds,
        seconds_per_prompt_token,
        decode_tps,
    )
    _annotate_predictions(
        heldout,
        heldout_cases,
        coefficients,
        text_completion_coefficients,
        tool_completion_coefficients,
        base_seconds,
        seconds_per_prompt_token,
        decode_tps,
    )
    heldout_measurement_errors = {
        "prompt_tokens": summarize_errors(
            float(sample["token_error_percent"]) for sample in heldout
        ),
        "completion_tokens": summarize_errors(
            float(sample["completion_token_error_percent"]) for sample in heldout
        ),
        "first_meaningful": summarize_errors(
            float(sample["ttft_error_percent"]) for sample in heldout
        ),
        "tool_ready": summarize_errors(
            float(sample["tool_ready_error_percent"])
            for sample in heldout
            if sample["tool_ready_error_percent"] is not None
        ),
        "conditional_total_duration": summarize_errors(
            float(sample["conditional_total_error_percent"]) for sample in heldout
        ),
        "end_to_end_total_duration": summarize_errors(
            float(sample["total_error_percent"]) for sample in heldout
        ),
    }
    heldout_case_results = aggregate_repetitions(heldout)
    heldout_case_errors = {
        "prompt_tokens": summarize_errors(
            float(case["prompt_token_error_percent"]) for case in heldout_case_results
        ),
        "completion_tokens": summarize_errors(
            float(case["completion_token_error_percent"]) for case in heldout_case_results
        ),
        "first_meaningful": summarize_errors(
            float(case["first_meaningful_error_percent"]) for case in heldout_case_results
        ),
        "tool_ready": summarize_errors(
            float(case["tool_ready_error_percent"])
            for case in heldout_case_results
            if case["tool_ready_error_percent"] is not None
        ),
        "conditional_total_duration": summarize_errors(
            float(case["conditional_total_error_percent"]) for case in heldout_case_results
        ),
        "end_to_end_total_duration": summarize_errors(
            float(case["end_to_end_total_error_percent"]) for case in heldout_case_results
        ),
    }
    measurement_acceptance_fields = {
        "first_meaningful": "ttft_error_percent",
        "tool_ready": "tool_ready_error_percent",
        "end_to_end_total_duration": "total_error_percent",
    }
    case_acceptance_fields = {
        "first_meaningful": "first_meaningful_error_percent",
        "tool_ready": "tool_ready_error_percent",
        "end_to_end_total_duration": "end_to_end_total_error_percent",
    }
    for metric, field in measurement_acceptance_fields.items():
        values = [float(sample[field]) for sample in heldout if sample[field] is not None]
        passed = sum(value <= ACCEPTANCE_PERCENT for value in values)
        heldout_measurement_errors[metric]["within_acceptance_count"] = passed
        heldout_measurement_errors[metric]["acceptance_rate_percent"] = passed / len(values) * 100
    for metric, field in case_acceptance_fields.items():
        values = [float(case[field]) for case in heldout_case_results if case[field] is not None]
        passed = sum(value <= ACCEPTANCE_PERCENT for value in values)
        heldout_case_errors[metric]["all_cases_within_acceptance"] = (
            heldout_case_errors[metric]["max_percent"] <= ACCEPTANCE_PERCENT
        )
        heldout_case_errors[metric]["within_acceptance_count"] = passed
        heldout_case_errors[metric]["acceptance_rate_percent"] = passed / len(values) * 100

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
            "output_token_budgets": sorted(
                {
                    case.get("output_tokens", OUTPUT_TOKEN_BUDGET)
                    for case in train_cases + heldout_cases
                }
            ),
            "temperature": 0,
            "seed": 42,
            "reasoning_enabled": False,
            "warmup_excluded_from_fit": True,
            "acceptance_percent": ACCEPTANCE_PERCENT,
            "run_id": run_id,
            "repetitions": repetitions,
            "validation_dataset": "development" if use_development_cases else "final",
        },
        "method": {
            "training_samples": len(training),
            "training_cases": len(training_cases()),
            "timing_fit_samples": len(timing_training),
            "heldout_samples": len(heldout),
            "heldout_cases": len(selected_validation_cases),
            "first_meaningful": (
                "direct wall time to first non-empty text or complete tool-call stream event"
            ),
            "tool_ready": "direct wall time to the first complete executable tool-call event",
            "first_event_bytes": "wall time to the first non-empty NDJSON stream line",
            "total": "wall time through stream close after the final done event",
            "prediction_scope": (
                "Text TTFT predicts base plus prefill. Buffered tool readiness and total duration "
                "add independently estimated completion tokens divided by decode throughput."
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
                "letter_coefficient": coefficients[1],
                "digit_coefficient": coefficients[2],
                "whitespace_coefficient": coefficients[3],
                "punctuation_coefficient": coefficients[4],
                "non_ascii_coefficient": coefficients[5],
                "extra_message_coefficient": coefficients[6],
                "tool_character_coefficient": coefficients[7],
            },
            "text_completion_token_regression": {
                "intercept": text_completion_coefficients[0],
                "letter_coefficient": text_completion_coefficients[1],
                "digit_coefficient": text_completion_coefficients[2],
                "whitespace_coefficient": text_completion_coefficients[3],
                "punctuation_coefficient": text_completion_coefficients[4],
                "non_ascii_coefficient": text_completion_coefficients[5],
            },
            "tool_completion_token_regression": {
                "intercept": tool_completion_coefficients[0],
                "letter_coefficient": tool_completion_coefficients[1],
                "digit_coefficient": tool_completion_coefficients[2],
                "whitespace_coefficient": tool_completion_coefficients[3],
                "punctuation_coefficient": tool_completion_coefficients[4],
                "non_ascii_coefficient": tool_completion_coefficients[5],
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
            "completion_tokens": summarize_errors(
                float(sample["completion_token_error_percent"]) for sample in training
            ),
            "first_meaningful": summarize_errors(
                float(sample["ttft_error_percent"]) for sample in training
            ),
            "tool_ready": summarize_errors(
                float(sample["tool_ready_error_percent"])
                for sample in training
                if sample["tool_ready_error_percent"] is not None
            ),
            "conditional_total_duration": summarize_errors(
                float(sample["conditional_total_error_percent"]) for sample in training
            ),
            "end_to_end_total_duration": summarize_errors(
                float(sample["total_error_percent"]) for sample in training
            ),
        },
        "heldout_summary": heldout_case_errors,
        "heldout_measurement_summary": heldout_measurement_errors,
        "heldout_case_results": heldout_case_results,
        "training_samples": training,
        "heldout_samples": heldout,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = calibrate(args.ollama_url, args.model, args.repetitions)
    except (httpx.HTTPError, OSError, ValueError) as error:
        raise SystemExit(f"calibration failed: {error}") from error
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(result, output, indent=2)
                output.write("\n")
        except FileExistsError as error:
            raise SystemExit(f"refusing to overwrite existing evidence: {args.output}") from error
        print(args.output)
    else:
        json.dump(result, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
