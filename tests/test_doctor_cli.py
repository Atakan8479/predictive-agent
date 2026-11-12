import json
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from predictive_agent.doctor.cli import BENCH_PROMPT, main, run


def _transport(request_bodies: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.16"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b-q4_K_M"}]})
        if request.url.path == "/api/generate":
            request_bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "load_duration": 10_000_000,
                    "prompt_eval_count": 400,
                    "prompt_eval_cached_count": 12,
                    "prompt_eval_duration": 400_000_000,
                    "eval_count": 50,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 1_500_000_000,
                },
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def test_uncached_runs_prepend_unique_uuid_nonces(capsys: pytest.CaptureFixture[str]) -> None:
    requests: list[dict[str, Any]] = []

    assert await run(2, transport=_transport(requests)) == 0

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


async def test_cached_runs_repeat_prompt_and_export_json(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []
    output = tmp_path / "doctor.json"

    assert await run(2, cached=True, json_path=output, transport=_transport(requests)) == 0

    assert [request["prompt"] for request in requests[1:]] == [BENCH_PROMPT, BENCH_PROMPT]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "cached"
    assert report["model"] == "qwen3.5:9b-q4_K_M"
    assert report["ollama_version"] == "0.40.16"
    assert len(report["results"]) == 2
    assert report["results"][0]["prompt_eval_cached_count"] == 12
    assert report["summaries"]["prompt_tokens_per_s"] == {
        "median": 1000.0,
        "minimum": 1000.0,
        "maximum": 1000.0,
    }


async def test_malformed_generate_response_fails_cleanly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.16"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b-q4_K_M"}]})
        return httpx.Response(200, json={"load_duration": 0})

    assert await run(1, transport=httpx.MockTransport(handler)) == 1
    assert "prompt_eval_count" in capsys.readouterr().out


async def test_run_rejects_non_positive_runs() -> None:
    assert await run(0) == 2


def test_cli_rejects_non_positive_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["pa-doctor", "--runs", "0"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
