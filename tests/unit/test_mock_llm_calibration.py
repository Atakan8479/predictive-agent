import json
from typing import Any

import httpx
import pytest
from mocks.llm.app import ChatCompletionRequest, prompt_tokens_for_request
from mocks.llm.calibration import estimate_prompt_tokens_for_payload, prompt_features
from scripts.calibrate_mock_llm import (
    estimator_text,
    fit_regression,
    mark_cases,
    measure_stream,
    runtime_final_cases,
    summarize_errors,
    supplementary_text_duration_cases,
    training_cases,
    validation_cases,
)
from scripts.validate_mock_llm_runtime import compare_pair, tool_calls_are_executable


def test_training_and_heldout_cases_are_distinct_and_cover_agent_workloads() -> None:
    training = training_cases()
    heldout = validation_cases()

    assert len(training) == 39
    assert len(heldout) == 10
    assert {case["name"] for case in training}.isdisjoint(case["name"] for case in heldout)
    assert {case["category"] for case in heldout} == {
        "english",
        "turkish",
        "code",
        "json",
        "tool-call",
        "tool-result-final",
    }
    assert any(case["tools"] and len(case["messages"]) > 1 for case in heldout)
    assert len({estimator_text(case) for case in training + heldout}) == 49


def test_runtime_final_cases_are_new_and_cover_paired_acceptance_scope() -> None:
    final = runtime_final_cases()
    prior_names = {case["name"] for case in training_cases() + validation_cases()}

    assert len(final) == 12
    assert prior_names.isdisjoint(case["name"] for case in final)
    assert {case["category"] for case in final} == {
        "english",
        "turkish",
        "code",
        "json",
        "conversation",
        "tool-call",
        "tool-result-final",
        "multi-turn-tool",
    }
    assert sum(case["category"] == "tool-call" for case in final) == 5


def test_supplementary_text_duration_cases_are_controlled_and_distinct() -> None:
    cases = supplementary_text_duration_cases()
    prior_names = {
        case["name"] for case in training_cases() + validation_cases() + runtime_final_cases()
    }

    assert len(cases) == 5
    assert prior_names.isdisjoint(case["name"] for case in cases)
    assert all(case["expected_content"] for case in cases)
    assert len({case["output_tokens"] for case in cases}) == 3
    assert sum(case["category"].startswith("controlled-post-tool") for case in cases) == 3
    assert any("Sipariş" in case["expected_content"] for case in cases)
    assert any('"customer_id"' in case["expected_content"] for case in cases)


def test_cache_nonce_cannot_be_confused_with_tool_argument() -> None:
    policy = next(
        case for case in runtime_final_cases() if case["name"] == "runtime-final-tool-policy"
    )

    marked = mark_cases([policy], "unique-run", 1)[0]

    assert marked["messages"][0]["content"].startswith("Call the requested tool")
    assert "not user data or a tool argument" in marked["messages"][0]["content"]
    assert marked["messages"][1]["content"] == "Explain the shipping policy."


@pytest.mark.parametrize(
    "text",
    [
        "a" * 2048,
        "abc123" * 320,
        "predictive" * 200,
        "ordinary natural language with several words " * 40,
        'const payload = {"order_id":"ORD-0042"};' * 30,
        '{"items":[{"sku":"SKU-0007"}]}' * 40,
        "Doğrulanmış müşteri kaydını özetle. " * 40,
    ],
)
def test_prompt_estimator_remains_length_sensitive_for_representative_text(text: str) -> None:
    short = estimate_prompt_tokens_for_payload(
        [{"role": "user", "content": text[: len(text) // 2]}]
    )
    long = estimate_prompt_tokens_for_payload([{"role": "user", "content": text}])

    assert long > short


def test_prompt_features_include_total_character_length() -> None:
    features = prompt_features([{"role": "user", "content": "abc123 !"}])

    assert features[1] == 8


def test_runtime_and_calibration_prompt_features_are_identical() -> None:
    case = validation_cases()[-1]
    runtime_messages = json.loads(json.dumps(case["messages"]))
    arguments = runtime_messages[2]["tool_calls"][0]["function"]["arguments"]
    runtime_messages[2]["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)
    request = ChatCompletionRequest.model_validate(
        {"model": "mock-calibrated", "messages": runtime_messages, "tools": case["tools"]}
    )
    expected = estimate_prompt_tokens_for_payload(case["messages"], case["tools"])

    assert prompt_tokens_for_request(request) == expected


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
    assert sample["output_kind"] == "text"
    assert sample["runtime_estimated_output_tokens"] > 0
    assert (
        sample["observed_first_event_bytes_seconds"] <= sample["observed_first_meaningful_seconds"]
    )
    assert sample["observed_first_meaningful_seconds"] <= sample["observed_final_event_seconds"]
    assert sample["measured_decode_tps"] == pytest.approx(500.0)


def test_paired_runtime_requires_complete_schema_valid_equivalent_action() -> None:
    case = next(case for case in validation_cases() if case["name"] == "acceptance-tool-policy")
    ollama_calls = [
        {
            "function": {
                "name": "get_policy",
                "arguments": {"policy_name": "returns"},
            }
        }
    ]
    mock_calls = [
        {
            "id": "call_mock",
            "type": "function",
            "function": {
                "name": "get_policy",
                "arguments": '{"policy_name":"returns"}',
            },
        }
    ]
    canonical = (
        '[{"function":{"arguments":{"policy_name":"returns"},'
        '"name":"get_policy"},"type":"function"}]'
    )
    ollama = {
        "generated_tool_calls": ollama_calls,
        "canonical_generated_output": canonical,
        "observed_first_meaningful_seconds": 0.6,
        "observed_tool_ready_seconds": 0.6,
        "observed_total_seconds": 0.62,
        "actual_output_tokens": 27,
        "actual_prompt_tokens": 300,
    }
    mock = {
        "generated_tool_calls": mock_calls,
        "canonical_generated_output": canonical,
        "observed_first_meaningful_seconds": 0.58,
        "observed_tool_ready_seconds": 0.58,
        "observed_total_seconds": 0.59,
        "completion_tokens": 26,
        "prompt_tokens": 298,
    }

    assert tool_calls_are_executable(ollama_calls, case["tools"])
    result = compare_pair(case, ollama, mock)
    assert result["output_equivalence"] == "executable_action_equivalent"
    assert result["tool_ready_comparable"] is True
    assert result["output_token_difference"] == -1


def test_paired_runtime_rejects_incomplete_tool_arguments() -> None:
    case = next(case for case in validation_cases() if case["name"] == "acceptance-tool-policy")
    calls = [{"function": {"name": "get_policy", "arguments": "{"}}]

    assert not tool_calls_are_executable(calls, case["tools"])
