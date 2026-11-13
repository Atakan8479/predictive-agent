import json
import os
from time import perf_counter
from typing import Any, cast

import httpx
import pytest
from mocks.llm.calibration import TimingModel, estimate_content_tokens, estimate_tokens

from predictive_agent.config import Settings

pytestmark = pytest.mark.integration
FAST_HEADERS = {
    "X-Mock-LLM-Base-Latency-Ms": "0",
    "X-Mock-LLM-Prefill-TPS": "1000000000",
    "X-Mock-LLM-Decode-TPS": "1000000000",
}


@pytest.mark.skipif(
    os.getenv("PA_RUN_INTEGRATION") != "1",
    reason="set PA_RUN_INTEGRATION=1 with the Compose stack running",
)
@pytest.mark.parametrize("length", [512, 2048, 4096])
async def test_mock_llm_stream_timing_matches_calibrated_model(length: int) -> None:
    port = os.getenv("PA_MOCK_LLM_PORT", "8090")
    prompt = ("Calibrated benchmark prompt with stable ASCII text. " * 100)[:length]
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    timing = TimingModel.from_settings(settings)
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_content_tokens("Mock response complete.")
    expected_ttft = timing.ttft_seconds(prompt_tokens)
    expected_total = timing.total_seconds(prompt_tokens, completion_tokens)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}", timeout=httpx.Timeout(3.0, connect=1.0)
    ) as client:
        health = await client.get("/healthz")
        health.raise_for_status()
        control_ttft, control_total, _ = await _measure_stream(client, prompt, FAST_HEADERS)
        measured_ttft, measured_total, usage = await _measure_stream(client, prompt)

    assert usage["prompt_tokens"] == prompt_tokens
    assert (measured_ttft - control_ttft) == pytest.approx(expected_ttft, rel=0.15)
    assert (measured_total - control_total) == pytest.approx(expected_total, rel=0.15)


async def _measure_stream(
    client: httpx.AsyncClient, prompt: str, headers: dict[str, str] | None = None
) -> tuple[float, float, dict[str, int]]:
    request = client.build_request(
        "POST",
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    started = perf_counter()
    first_event_at: float | None = None
    usage: dict[str, int] | None = None
    response = await client.send(request, stream=True)
    try:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            if first_event_at is None:
                first_event_at = perf_counter()
            payload = line.removeprefix("data: ")
            if payload != "[DONE]":
                event = cast(dict[str, Any], json.loads(payload))
                if event.get("usage"):
                    usage = cast(dict[str, int], event["usage"])
        finished = perf_counter()
    finally:
        await response.aclose()
    if first_event_at is None or usage is None:
        raise AssertionError("stream did not contain first event and usage")
    return first_event_at - started, finished - started, usage
