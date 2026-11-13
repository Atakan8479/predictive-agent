import json
from typing import Any

import httpx
import pytest
from scripts.calibrate_mock_llm import (
    estimator_text,
    fit_regression,
    measure_stream,
    summarize_errors,
    training_cases,
    validation_cases,
)


def test_training_and_heldout_cases_are_distinct_and_cover_agent_workloads() -> None:
    training = training_cases()
    heldout = validation_cases()

    assert len(training) == 29
    assert len(heldout) == 10
    assert {case["name"] for case in training}.isdisjoint(case["name"] for case in heldout)
    assert {case["category"] for case in heldout} == {
        "english",
        "turkish",
        "code",
        "json",
        "tools-and-history",
    }
    assert any(case["tools"] and len(case["messages"]) > 1 for case in heldout)
    assert len({estimator_text(case)[:16] for case in training + heldout}) == 39


def test_regression_rejects_empty_and_singular_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty and aligned"):
        fit_regression([], [])
    with pytest.raises(ValueError, match="singular"):
        fit_regression([(1.0, 1.0), (1.0, 1.0)], [1.0, 2.0])


def test_error_summary_uses_nearest_rank_p95() -> None:
    assert summarize_errors([1.0, 4.0, 2.0, 3.0]) == {
        "median_percent": 2.5,
        "p95_percent": 4.0,
        "max_percent": 4.0,
    }


def test_stream_measurement_uses_first_meaningful_content_and_final_metrics() -> None:
    events: list[dict[str, Any]] = [
        {"message": {"role": "assistant", "content": ""}, "done": False},
        {"message": {"role": "assistant", "content": "ok"}, "done": False},
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "prompt_eval_count": 12,
            "prompt_eval_cached_count": 0,
            "eval_count": 2,
            "prompt_eval_duration": 6_000_000,
            "eval_duration": 4_000_000,
            "total_duration": 12_000_000,
            "load_duration": 1_000_000,
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = "".join(f"{json.dumps(event)}\n" for event in events)
        return httpx.Response(200, text=body)

    case = validation_cases()[0]
    with httpx.Client(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    ) as client:
        sample = measure_stream(client, "test-model", case)

    assert sample["actual_prompt_tokens"] == 12
    assert sample["actual_output_tokens"] == 2
    assert sample["cached_prompt_tokens"] == 0
    assert sample["observed_first_event_bytes_seconds"] <= sample["observed_ttft_seconds"]
    assert sample["observed_ttft_seconds"] <= sample["observed_final_event_seconds"]
    assert sample["measured_decode_tps"] == pytest.approx(500.0)
