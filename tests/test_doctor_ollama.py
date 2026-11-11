import json

import httpx
import pytest

from predictive_agent.doctor.ollama import GenerationStats, OllamaProbe

GENERATE_RESPONSE = {
    "load_duration": 2_000_000_000,
    "prompt_eval_count": 400,
    "prompt_eval_duration": 200_000_000,
    "eval_count": 120,
    "eval_duration": 2_000_000_000,
    "total_duration": 4_300_000_000,
}


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.40.16"})
    if request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b-q4_K_M"}]})
    if request.url.path == "/api/generate":
        body = json.loads(request.content)
        assert body["think"] is False
        assert body["stream"] is False
        return httpx.Response(200, json=GENERATE_RESPONSE)
    return httpx.Response(404)


@pytest.fixture
async def probe():
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://ollama") as client:
        yield OllamaProbe(client)


async def test_version(probe: OllamaProbe) -> None:
    assert await probe.version() == "0.40.16"


async def test_installed_models(probe: OllamaProbe) -> None:
    assert await probe.installed_models() == ["qwen3.5:9b-q4_K_M"]


async def test_generate_parses_timings(probe: OllamaProbe) -> None:
    s = await probe.generate("qwen3.5:9b-q4_K_M", "hi", think=False, keep_alive="5m")
    assert s.load_s == pytest.approx(2.0)
    assert s.prompt_tokens_per_s == pytest.approx(2000.0)
    assert s.output_tokens_per_s == pytest.approx(60.0)
    assert s.total_s == pytest.approx(4.3)


def test_zero_durations_do_not_divide_by_zero() -> None:
    s = GenerationStats.from_response({})
    assert s.prompt_tokens_per_s == 0.0
    assert s.output_tokens_per_s == 0.0
