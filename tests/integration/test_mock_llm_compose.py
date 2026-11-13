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
ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_order",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "pattern": r"^ORD-\d{4}$"}},
            "required": ["order_id"],
        },
    },
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


@pytest.mark.skipif(
    os.getenv("PA_RUN_INTEGRATION") != "1",
    reason="set PA_RUN_INTEGRATION=1 with the Compose stack running",
)
async def test_mock_llm_executes_complete_order_tool_round_trip() -> None:
    llm_port = os.getenv("PA_MOCK_LLM_PORT", "8090")
    api_port = os.getenv("PA_MOCK_APIS_PORT", "8080")
    timeout = httpx.Timeout(5.0, connect=2.0)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Show order ORD-0042"}]
    request: dict[str, Any] = {
        "model": "mock-calibrated",
        "messages": messages,
        "tools": [ORDER_TOOL],
    }

    async with (
        httpx.AsyncClient(base_url=f"http://127.0.0.1:{llm_port}", timeout=timeout) as llm,
        httpx.AsyncClient(base_url=f"http://127.0.0.1:{api_port}", timeout=timeout) as external_api,
    ):
        completion = await llm.post("/v1/chat/completions", headers=FAST_HEADERS, json=request)
        completion.raise_for_status()
        assistant = cast(dict[str, Any], completion.json()["choices"][0]["message"])
        tool_call = cast(dict[str, Any], assistant["tool_calls"][0])
        arguments = cast(dict[str, str], json.loads(tool_call["function"]["arguments"]))

        order_response = await external_api.get(f"/oms/orders/{arguments['order_id']}")
        order_response.raise_for_status()
        order = cast(dict[str, Any], order_response.json())
        tool_result = json.dumps(order, separators=(",", ":"), sort_keys=True)
        request["messages"] = [
            *messages,
            assistant,
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result,
            },
        ]
        final = await llm.post("/v1/chat/completions", headers=FAST_HEADERS, json=request)
        repeated = await llm.post("/v1/chat/completions", headers=FAST_HEADERS, json=request)

    final.raise_for_status()
    repeated.raise_for_status()
    assert arguments == {"order_id": "ORD-0042"}
    assert order["order_id"] == "ORD-0042"
    assert order_response.headers["X-Mock-Served-At"]
    final_choice = final.json()["choices"][0]
    assert final_choice["finish_reason"] == "stop"
    assert final_choice["message"]["content"] == f"Mock final answer: {tool_result}"
    assert repeated.json()["choices"] == final.json()["choices"]


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
