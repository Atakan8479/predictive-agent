import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from mocks.llm.app import app
from mocks.llm.calibration import TimingModel, estimate_content_tokens, estimate_tokens

from predictive_agent.config import Settings, get_settings

FAST_HEADERS = {
    "X-Mock-LLM-Base-Latency-Ms": "0",
    "X-Mock-LLM-Prefill-TPS": "1000000000",
    "X-Mock-LLM-Decode-TPS": "1000000000",
}


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=2.0
    ) as api_client:
        yield api_client


async def test_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_regular_completion_matches_openai_shape(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={"model": "mock-calibrated", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"].startswith("chatcmpl-mock-")
    assert data["object"] == "chat.completion"
    assert data["model"] == "mock-calibrated"
    assert data["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Mock response complete."},
            "logprobs": None,
            "finish_reason": "stop",
        }
    ]
    assert data["usage"]["total_tokens"] == (
        data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    )


async def test_tool_call_and_final_response_are_deterministic(client: httpx.AsyncClient) -> None:
    request: dict[str, Any] = {
        "model": "mock-calibrated",
        "messages": [{"role": "user", "content": "Look up order ORD-0042"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_order",
                    "description": "Fetch an order",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }

    first = await client.post("/v1/chat/completions", headers=FAST_HEADERS, json=request)
    repeated = await client.post("/v1/chat/completions", headers=FAST_HEADERS, json=request)
    first_data = first.json()
    repeated_data = repeated.json()

    assert first_data["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = first_data["choices"][0]["message"]["tool_calls"][0]
    assert tool_call == repeated_data["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"] == {"name": "get_order", "arguments": "{}"}

    request["messages"].extend(
        [
            first_data["choices"][0]["message"],
            {"role": "tool", "tool_call_id": tool_call["id"], "content": "delivered"},
        ]
    )
    final = await client.post("/v1/chat/completions", headers=FAST_HEADERS, json=request)

    assert final.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Mock final answer: delivered",
    }


async def test_streaming_completion_emits_chunks_usage_and_done(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Stream"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    payloads = [line.removeprefix("data: ") for line in response.text.splitlines() if line]
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(payload) for payload in payloads[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    content = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk["choices"]
    )
    assert content == "Mock response complete."
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] > 0


async def test_streaming_tool_call_uses_openai_delta_shape(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Use a tool"}],
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    )

    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line and line != "data: [DONE]"
    ]
    tool_delta = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_delta["index"] == 0
    assert tool_delta["function"] == {"name": "lookup", "arguments": "{}"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize(
    "header",
    [
        {"X-Mock-LLM-Base-Latency-Ms": "-1"},
        {"X-Mock-LLM-Prefill-TPS": "0"},
        {"X-Mock-LLM-Decode-TPS": "invalid"},
    ],
)
async def test_invalid_timing_headers_are_rejected(
    client: httpx.AsyncClient, header: dict[str, str]
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=header,
        json={"model": "mock", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 422


def test_calibrated_token_estimator_handles_character_classes() -> None:
    assert estimate_tokens("a" * 128) == 47
    assert estimate_tokens("{" * 128) == 106
    assert estimate_tokens("ü" * 128) == 124
    assert estimate_content_tokens("Mock response complete.") < estimate_tokens(
        "Mock response complete."
    )


def test_timing_model_uses_settings_and_overrides() -> None:
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    calibrated = TimingModel.from_settings(settings)
    overridden = TimingModel.from_settings(
        settings, base_latency_ms=10.0, prefill_tps=100.0, decode_tps=20.0
    )

    assert calibrated.base_latency_ms == pytest.approx(83.783)
    assert overridden.ttft_seconds(50) == pytest.approx(0.51)
    assert overridden.total_seconds(50, 10) == pytest.approx(1.01)
