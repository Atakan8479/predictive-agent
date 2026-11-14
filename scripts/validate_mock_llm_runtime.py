"""Directly compare paired Ollama and running mock-LLM streams."""

import argparse
import hashlib
import json
import platform
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter, time_ns
from typing import Any, cast

import httpx
from mocks.llm.app import _validate_object_schema, _value_error
from mocks.llm.calibration import serialize_completion
from scripts.calibrate_mock_llm import (
    ACCEPTANCE_PERCENT,
    Case,
    fit_regression,
    mark_cases,
    measure_stream,
    runtime_final_cases,
    summarize_errors,
    validation_cases,
)

from predictive_agent.config import Settings


def measure_mock_stream(
    client: httpx.Client, case: Case, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "mock-calibrated",
        "messages": _openai_messages(case["messages"]),
        "tools": case["tools"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": case.get("output_tokens", 64),
    }
    started = perf_counter()
    first_meaningful_at: float | None = None
    tool_ready_at: float | None = None
    finished_at: float | None = None
    content: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    with client.stream("POST", "/v1/chat/completions", json=payload, headers=headers) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            observed_at = perf_counter()
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                finished_at = observed_at
                continue
            event = json.loads(data)
            if not isinstance(event, dict):
                raise ValueError("mock SSE event must be an object")
            typed_event = cast(dict[str, Any], event)
            event_usage = typed_event.get("usage")
            if isinstance(event_usage, dict):
                usage = cast(dict[str, Any], event_usage)
            choices = typed_event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta")
            if not isinstance(delta, dict):
                continue
            chunk = delta.get("content")
            if isinstance(chunk, str) and chunk:
                content.append(chunk)
                if first_meaningful_at is None:
                    first_meaningful_at = observed_at
            delta_calls = delta.get("tool_calls")
            if isinstance(delta_calls, list) and delta_calls:
                tool_calls = [
                    {
                        key: value
                        for key, value in cast(dict[str, Any], call).items()
                        if key != "index"
                    }
                    for call in delta_calls
                    if isinstance(call, dict)
                ]
                if tool_calls_are_executable(tool_calls, case["tools"]):
                    tool_ready_at = observed_at
                    if first_meaningful_at is None:
                        first_meaningful_at = observed_at
    if first_meaningful_at is None:
        raise ValueError(f"mock stream omitted meaningful output for {case['name']!r}")
    if finished_at is None or usage is None:
        raise ValueError(f"mock stream omitted completion or usage for {case['name']!r}")
    generated_content = "".join(content)
    canonical = serialize_completion(generated_content, tool_calls)
    return {
        "input_payload": payload,
        "generated_content": generated_content,
        "generated_tool_calls": tool_calls,
        "canonical_generated_output": canonical,
        "completion_tokens": _non_negative_integer(usage, "completion_tokens"),
        "prompt_tokens": _non_negative_integer(usage, "prompt_tokens"),
        "observed_first_meaningful_seconds": first_meaningful_at - started,
        "observed_tool_ready_seconds": (
            tool_ready_at - started if tool_ready_at is not None else None
        ),
        "observed_total_seconds": finished_at - started,
    }


def _openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = json.loads(json.dumps(messages))
    if not isinstance(normalized, list):
        raise ValueError("messages must be an array")
    for message in normalized:
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                continue
            function = call["function"]
            if not isinstance(function.get("arguments"), str):
                function["arguments"] = json.dumps(
                    function.get("arguments"), separators=(",", ":"), sort_keys=True
                )
    return cast(list[dict[str, Any]], normalized)


def tool_calls_are_executable(calls: list[dict[str, Any]], tools: list[dict[str, Any]]) -> bool:
    definitions = {
        function["name"]: function
        for tool in tools
        if isinstance(tool, dict)
        and isinstance((function := tool.get("function")), dict)
        and isinstance(function.get("name"), str)
    }
    if not calls:
        return False
    for call in calls:
        function = call.get("function")
        if not isinstance(function, dict) or function.get("name") not in definitions:
            return False
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return False
        if not isinstance(arguments, dict):
            return False
        schema = definitions[cast(str, function["name"])].get("parameters")
        if not isinstance(schema, dict) or _validate_object_schema(schema) is not None:
            return False
        properties = schema["properties"]
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return False
        if any(name not in arguments for name in required):
            return False
        if schema.get("additionalProperties") is False and any(
            name not in properties for name in arguments
        ):
            return False
        if any(
            name not in properties
            or not isinstance(properties[name], dict)
            or _value_error(properties[name], value) is not None
            for name, value in arguments.items()
        ):
            return False
    return True


def _non_negative_integer(data: dict[str, Any], field: str) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"field {field!r} must be a non-negative integer")
    return value


def compare_pair(
    case: Case,
    ollama: dict[str, Any],
    mock: dict[str, Any],
    control_mock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ollama_calls = cast(list[dict[str, Any]], ollama["generated_tool_calls"])
    mock_calls = cast(list[dict[str, Any]], mock["generated_tool_calls"])
    ollama_executable = tool_calls_are_executable(ollama_calls, case["tools"])
    mock_executable = tool_calls_are_executable(mock_calls, case["tools"])
    actions_equal = (
        ollama_executable
        and mock_executable
        and len(ollama_calls) == len(mock_calls) == 1
        and ollama["canonical_generated_output"] == mock["canonical_generated_output"]
    )
    outputs_equal = ollama["canonical_generated_output"] == mock["canonical_generated_output"]
    output_token_error = _percent_error(
        float(mock["completion_tokens"]), float(ollama["actual_output_tokens"])
    )
    length_equivalent = output_token_error <= ACCEPTANCE_PERCENT
    if actions_equal and length_equivalent:
        equivalence = "executable_action_equivalent"
    elif actions_equal:
        equivalence = "executable_action_equivalent_length_mismatch"
    elif outputs_equal:
        equivalence = "canonical_output_equivalent"
    elif ollama_calls or mock_calls:
        equivalence = "tool_action_non_equivalent"
    else:
        equivalence = "text_output_non_equivalent"
    ollama_first = float(ollama["observed_first_meaningful_seconds"])
    mock_first = float(mock["observed_first_meaningful_seconds"])
    ollama_total = float(ollama["observed_total_seconds"])
    mock_total = float(mock["observed_total_seconds"])
    ollama_tool = ollama["observed_tool_ready_seconds"]
    mock_tool = mock["observed_tool_ready_seconds"]
    return {
        "case_name": case.get("base_name", case["name"]),
        "category": case["category"],
        "input_messages": case["messages"],
        "input_tools": case["tools"],
        "output_equivalence": equivalence,
        "first_meaningful_comparable": True,
        "tool_ready_comparable": actions_equal,
        "total_duration_comparable": outputs_equal and length_equivalent,
        "ollama": ollama,
        "mock": mock,
        "control_mock": control_mock,
        "output_character_difference": len(str(mock["canonical_generated_output"]))
        - len(str(ollama["canonical_generated_output"])),
        "output_token_difference": int(mock["completion_tokens"])
        - int(ollama["actual_output_tokens"]),
        "output_token_error_percent": output_token_error,
        "prompt_token_error_percent": _percent_error(
            float(mock["prompt_tokens"]), float(ollama["actual_prompt_tokens"])
        ),
        "first_meaningful_absolute_error_seconds": abs(mock_first - ollama_first),
        "first_meaningful_error_percent": _percent_error(mock_first, ollama_first),
        "tool_ready_absolute_error_seconds": (
            abs(float(mock_tool) - float(ollama_tool))
            if actions_equal and mock_tool is not None and ollama_tool is not None
            else None
        ),
        "tool_ready_error_percent": (
            _percent_error(float(mock_tool), float(ollama_tool))
            if actions_equal and mock_tool is not None and ollama_tool is not None
            else None
        ),
        "total_absolute_error_seconds": abs(mock_total - ollama_total),
        "total_error_percent": _percent_error(mock_total, ollama_total),
    }


def _percent_error(observed: float, reference: float) -> float:
    if reference <= 0:
        raise ValueError("timing reference must be positive")
    return abs(observed - reference) / reference * 100


def aggregate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["case_name"]), []).append(sample)
    results: list[dict[str, Any]] = []
    for name, repetitions in grouped.items():
        first_errors = [float(sample["first_meaningful_error_percent"]) for sample in repetitions]
        tool_errors = [
            float(sample["tool_ready_error_percent"])
            for sample in repetitions
            if sample["tool_ready_error_percent"] is not None
        ]
        total_errors = [
            float(sample["total_error_percent"])
            for sample in repetitions
            if sample["total_duration_comparable"]
        ]
        results.append(
            {
                "case_name": name,
                "category": repetitions[0]["category"],
                "repetitions": len(repetitions),
                "equivalence": sorted(
                    {str(sample["output_equivalence"]) for sample in repetitions}
                ),
                "prompt_token_error_percent": median(
                    float(sample["prompt_token_error_percent"]) for sample in repetitions
                ),
                "first_meaningful_error_percent": median(first_errors),
                "tool_ready_error_percent": median(tool_errors) if tool_errors else None,
                "total_error_percent": median(total_errors) if total_errors else None,
                "first_meaningful_variability_seconds": _timing_range(
                    repetitions, "observed_first_meaningful_seconds"
                ),
                "maximum_cached_prompt_tokens": max(
                    int(sample["ollama"]["cached_prompt_tokens"]) for sample in repetitions
                ),
            }
        )
    return results


def _timing_range(samples: list[dict[str, Any]], field: str) -> dict[str, float]:
    return {
        provider: max(values) - min(values)
        for provider in ("ollama", "mock")
        if (values := [float(sample[provider][field]) for sample in samples])
    }


def _summary(values: Iterable[float]) -> dict[str, Any]:
    materialized = list(values)
    summary: dict[str, Any] = summarize_errors(materialized)
    passed = sum(value <= ACCEPTANCE_PERCENT for value in materialized)
    summary["within_acceptance_count"] = passed
    summary["acceptance_rate_percent"] = passed / len(materialized) * 100
    summary["all_within_acceptance"] = passed == len(materialized)
    return summary


def validate(
    ollama_url: str,
    mock_url: str,
    model: str,
    repetitions: int,
    *,
    final: bool,
) -> dict[str, Any]:
    run_id = hashlib.sha256(str(time_ns()).encode()).hexdigest()[:16]
    base_cases = runtime_final_cases() if final else validation_cases()
    cases = mark_cases(base_cases, f"{run_id}-paired", repetitions)
    timeout = httpx.Timeout(120.0, connect=5.0, read=120.0, write=10.0, pool=5.0)
    samples: list[dict[str, Any]] = []
    fast_headers = {
        "X-Mock-LLM-Base-Latency-Ms": "0",
        "X-Mock-LLM-Prefill-TPS": "1000000000",
        "X-Mock-LLM-Decode-TPS": "1000000000",
    }
    with (
        httpx.Client(base_url=ollama_url, timeout=timeout) as ollama_client,
        httpx.Client(base_url=mock_url, timeout=timeout) as mock_client,
    ):
        ollama_version = ollama_client.get("/api/version").json().get("version", "unknown")
        mock_client.get("/healthz").raise_for_status()
        warmup_case: Case = {
            "name": "paired-warmup-not-a-sample",
            "category": "warmup",
            "messages": [
                {
                    "role": "user",
                    "content": f"Warm both endpoints without reuse: {run_id}.",
                }
            ],
            "tools": [],
            "output_tokens": 16,
        }
        warmup = {
            "ollama": measure_stream(ollama_client, model, warmup_case),
            "mock": measure_mock_stream(mock_client, warmup_case),
        }
        for case in cases:
            ollama = measure_stream(ollama_client, model, case)
            control_mock = measure_mock_stream(mock_client, case, fast_headers)
            mock = measure_mock_stream(mock_client, case)
            samples.append(compare_pair(case, ollama, mock, control_mock))
    case_results = aggregate(samples)
    runtime_fit = None if final else _fit_runtime_timing(samples)
    comparable_tools = [
        float(case["tool_ready_error_percent"])
        for case in case_results
        if case["tool_ready_error_percent"] is not None
    ]
    comparable_totals = [
        float(case["total_error_percent"])
        for case in case_results
        if case["total_error_percent"] is not None
    ]
    measurement_tool_errors = [
        float(sample["tool_ready_error_percent"])
        for sample in samples
        if sample["tool_ready_error_percent"] is not None
    ]
    measurement_total_errors = [
        float(sample["total_error_percent"])
        for sample in samples
        if sample["total_duration_comparable"]
    ]
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "source_commit": _source_commit(),
            "dataset": "final" if final else "development",
            "repetitions": repetitions,
            "acceptance_percent": ACCEPTANCE_PERCENT,
            "ollama_url": ollama_url,
            "mock_url": mock_url,
            "ollama_version": ollama_version,
            "model": model,
            "python_version": platform.python_version(),
            "httpx_version": httpx.__version__,
            "platform": platform.platform(),
            "mock_timing_configuration": {
                "base_latency_ms": settings.mock_llm_base_latency_ms,
                "prefill_tps": settings.mock_llm_prefill_tps,
                "decode_tps": settings.mock_llm_decode_tps,
            },
        },
        "method": {
            "first_meaningful": (
                "direct request-start to first non-empty text or executable tool call"
            ),
            "tool_ready": "direct request-start to complete parseable schema-valid tool arguments",
            "equivalence_policy": (
                "Tool timing is comparable only for one identical schema-valid action. Total "
                "timing additionally requires canonical output and completion-token lengths "
                "within the acceptance threshold."
            ),
            "cache_control": "Ollama cache_prompt=false with unique run/repetition prefixes",
            "warmup_excluded": True,
        },
        "warmup": warmup,
        "summary": {
            "prompt_tokens": summarize_errors(
                float(case["prompt_token_error_percent"]) for case in case_results
            ),
            "first_meaningful": _summary(
                float(case["first_meaningful_error_percent"]) for case in case_results
            ),
            "tool_ready": _summary(comparable_tools) if comparable_tools else None,
            "comparable_total": _summary(comparable_totals) if comparable_totals else None,
            "comparable_tool_cases": len(comparable_tools),
            "non_comparable_cases": sum(
                case["total_error_percent"] is None for case in case_results
            ),
        },
        "measurement_summary": {
            "prompt_tokens": summarize_errors(
                float(sample["prompt_token_error_percent"]) for sample in samples
            ),
            "first_meaningful": _summary(
                float(sample["first_meaningful_error_percent"]) for sample in samples
            ),
            "tool_ready": _summary(measurement_tool_errors) if measurement_tool_errors else None,
            "comparable_total": (
                _summary(measurement_total_errors) if measurement_total_errors else None
            ),
        },
        "runtime_fit_from_direct_measurements": runtime_fit,
        "case_results": case_results,
        "samples": samples,
    }


def _fit_runtime_timing(samples: list[dict[str, Any]]) -> dict[str, float]:
    text_rows: list[tuple[float, ...]] = []
    text_targets: list[float] = []
    for sample in samples:
        control = sample["control_mock"]
        if not isinstance(control, dict):
            continue
        target = float(sample["ollama"]["observed_first_meaningful_seconds"]) - float(
            control["observed_first_meaningful_seconds"]
        )
        if target > 0 and not sample["ollama"]["generated_tool_calls"]:
            text_rows.append((1.0, float(sample["mock"]["prompt_tokens"])))
            text_targets.append(target)
    base, prompt_seconds = fit_regression(text_rows, text_targets)
    decode_rates: list[float] = []
    for sample in samples:
        if not sample["tool_ready_comparable"]:
            continue
        control = cast(dict[str, Any], sample["control_mock"])
        target = float(sample["ollama"]["observed_tool_ready_seconds"]) - float(
            control["observed_tool_ready_seconds"]
        )
        prompt_seconds_total = base + float(sample["mock"]["prompt_tokens"]) * prompt_seconds
        completion_tokens = float(sample["mock"]["completion_tokens"])
        decode_seconds = (target - prompt_seconds_total) / completion_tokens
        if decode_seconds > 0:
            decode_rates.append(1 / decode_seconds)
    if base < 0 or prompt_seconds <= 0 or not decode_rates:
        raise ValueError("direct runtime fit requires positive prefill and decode slopes")
    return {
        "base_latency_ms": base * 1000,
        "prefill_tps": 1 / prompt_seconds,
        "decode_tps": median(decode_rates),
        "text_fit_samples": float(len(text_rows)),
        "tool_fit_samples": float(len(decode_rates)),
    }


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--mock-url", default="http://127.0.0.1:8090")
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("repetitions must be at least one")
    try:
        result = validate(
            args.ollama_url,
            args.mock_url,
            args.model,
            args.repetitions,
            final=args.final,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, ensure_ascii=False)
            output.write("\n")
    except (httpx.HTTPError, OSError, ValueError) as error:
        raise SystemExit(f"runtime validation failed: {error}") from error
    print(args.output)


if __name__ == "__main__":
    main()
