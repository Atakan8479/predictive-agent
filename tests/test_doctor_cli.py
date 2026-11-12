import json
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from predictive_agent.config import Settings, get_settings
from predictive_agent.doctor.cli import BENCH_PROMPT, main, run

TEST_MODEL = "doctor-test-model"


@pytest.fixture
def doctor_settings() -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "llm_model": TEST_MODEL,
        "llm_think": False,
        "ollama_url": "http://doctor.test",
        "ollama_keep_alive": "5m",
        "http_timeout_s": 10.0,
    }
    return Settings(**values)


def _generate_response(
    *, prompt_tokens: int = 400, prompt_eval_duration: int = 400_000_000
) -> dict[str, int]:
    return {
        "load_duration": 10_000_000,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_cached_count": 12,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": 50,
        "eval_duration": 1_000_000_000,
        "total_duration": 1_500_000_000,
    }


def _transport(
    request_bodies: list[dict[str, Any]],
    generate_responses: list[dict[str, Any]] | None = None,
) -> httpx.MockTransport:
    response_index = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal response_index
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.16"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": TEST_MODEL}]})
        if request.url.path == "/api/generate":
            request_bodies.append(json.loads(request.content))
            response = (
                generate_responses[response_index]
                if generate_responses is not None
                else _generate_response()
            )
            response_index += 1
            return httpx.Response(200, json=response)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def test_uncached_runs_prepend_unique_uuid_nonces(
    doctor_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    requests: list[dict[str, Any]] = []

    assert await run(2, transport=_transport(requests), settings=doctor_settings) == 0

    assert requests[0]["prompt"] == "Merhaba."
    prompts = [request["prompt"] for request in requests[1:]]
    assert len(prompts) == 2
    assert prompts[0] != prompts[1]
    for prompt in prompts:
        nonce, benchmark = prompt.split("\n", maxsplit=1)
        assert str(uuid.UUID(nonce)) == nonce
        assert benchmark == BENCH_PROMPT
    output = capsys.readouterr().out
    assert "Benchmark (uncached)" in output
    assert "median / minimum / maximum" in output


async def test_cached_runs_are_primed_and_export_only_measured_results(
    doctor_settings: Settings, tmp_path: Path
) -> None:
    requests: list[dict[str, Any]] = []
    output = tmp_path / "doctor.json"
    responses = [
        _generate_response(),  # Model warm-up.
        _generate_response(prompt_eval_duration=4_000_000_000),  # Unmeasured cache prime.
        _generate_response(prompt_eval_duration=800_000_000),  # 500 tok/s.
        _generate_response(prompt_eval_duration=400_000_000),  # 1,000 tok/s.
        _generate_response(prompt_eval_duration=200_000_000),  # 2,000 tok/s.
    ]

    assert (
        await run(
            3,
            cached=True,
            json_path=output,
            transport=_transport(requests, responses),
            settings=doctor_settings,
        )
        == 0
    )

    assert requests[0]["prompt"] == "Merhaba."
    assert [request["prompt"] for request in requests[1:]] == [BENCH_PROMPT] * 4
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "cached"
    assert report["model"] == TEST_MODEL
    assert report["ollama_version"] == "0.40.16"
    assert [result["run"] for result in report["results"]] == [1, 2, 3]
    assert report["results"][0]["prompt_eval_cached_count"] == 12
    assert report["summaries"]["prompt_tokens_per_s"] == {
        "median": 1000.0,
        "minimum": 500.0,
        "maximum": 2000.0,
    }


async def test_malformed_generate_response_fails_cleanly(
    doctor_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.16"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": TEST_MODEL}]})
        return httpx.Response(200, json={"load_duration": 0})

    assert await run(1, transport=httpx.MockTransport(handler), settings=doctor_settings) == 1
    assert "prompt_eval_count" in capsys.readouterr().out


async def test_run_rejects_non_positive_runs() -> None:
    assert await run(0) == 2


async def test_missing_cached_count_is_exported_as_null(
    doctor_settings: Settings, tmp_path: Path
) -> None:
    requests: list[dict[str, Any]] = []
    response = _generate_response()
    response.pop("prompt_eval_cached_count")
    output = tmp_path / "doctor.json"

    assert (
        await run(
            1,
            json_path=output,
            transport=_transport(requests, [response, response]),
            settings=doctor_settings,
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["results"][0]["prompt_eval_cached_count"] is None


async def test_json_write_failure_is_reported(
    doctor_settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    requests: list[dict[str, Any]] = []

    assert (
        await run(
            1,
            json_path=tmp_path,
            transport=_transport(requests),
            settings=doctor_settings,
        )
        == 1
    )
    assert "could not write JSON report" in capsys.readouterr().out


async def test_explicit_settings_isolate_external_model_configuration(
    doctor_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, Any]] = []
    monkeypatch.setenv("PA_LLM_MODEL", "developer-model")
    get_settings.cache_clear()

    try:
        assert await run(1, transport=_transport(requests), settings=doctor_settings) == 0
    finally:
        get_settings.cache_clear()

    assert {request["model"] for request in requests} == {TEST_MODEL}


def test_cli_rejects_non_positive_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["pa-doctor", "--runs", "0"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
