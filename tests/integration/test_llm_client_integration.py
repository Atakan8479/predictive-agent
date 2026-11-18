import json
import os
from typing import Any, cast

import httpx
import pytest
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

from predictive_agent.config import Settings
from predictive_agent.llm.client import LLMClient, LLMResult

pytestmark = pytest.mark.integration


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


def _messages(*items: dict[str, Any]) -> list[ChatCompletionMessageParam]:
    return cast(list[ChatCompletionMessageParam], list(items))


def _order_tool() -> ChatCompletionToolUnionParam:
    return cast(
        ChatCompletionToolUnionParam,
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": "Get one order by identifier.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        },
    )


async def _consume(client: LLMClient, prompt: str) -> tuple[str, LLMResult]:
    text: list[str] = []
    result: LLMResult | None = None
    async for event in client.stream(_messages({"role": "user", "content": prompt})):
        if event.content_delta is not None:
            text.append(event.content_delta)
        if event.result is not None:
            result = event.result
    if result is None:
        raise AssertionError("stream did not return a terminal result")
    return "".join(text), result


@pytest.mark.skipif(
    os.getenv("PA_RUN_INTEGRATION") != "1",
    reason="set PA_RUN_INTEGRATION=1 with the Compose stack running",
)
async def test_mock_provider_regular_stream_and_tool_round_trip() -> None:
    llm_port = os.getenv("PA_MOCK_LLM_PORT", "8090")
    api_port = os.getenv("PA_MOCK_APIS_PORT", "8080")
    settings = _settings(
        llm_provider="mock",
        llm_base_url=f"http://127.0.0.1:{llm_port}/v1",
        llm_api_key="mock",
        llm_model="mock-calibrated",
        http_timeout_s=15,
    )
    async with LLMClient(settings) as client:
        regular = await client.complete(_messages({"role": "user", "content": "Say hello"}))
        streamed_text, streamed = await _consume(client, "Say hello")
        tool_request = await client.complete(
            _messages({"role": "user", "content": "Show order ORD-0042"}),
            tools=[_order_tool()],
        )
        assert tool_request.tool_calls
        call = tool_request.tool_calls[0]
        assert call.executable
        assert call.parsed_arguments == {"order_id": "ORD-0042"}

        async with httpx.AsyncClient(timeout=httpx.Timeout(5, connect=2)) as http:
            response = await http.get(f"http://127.0.0.1:{api_port}/oms/orders/ORD-0042")
            response.raise_for_status()
        tool_result = json.dumps(response.json(), separators=(",", ":"), sort_keys=True)
        final = await client.complete(
            _messages(
                {"role": "user", "content": "Show order ORD-0042"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call.id, "content": tool_result},
            ),
            tools=[_order_tool()],
        )

    assert regular.content == "Mock response complete."
    assert regular.metrics.usage.total_tokens is not None
    assert streamed_text == "Mock response complete."
    assert streamed.content == streamed_text
    assert streamed.metrics.usage.total_tokens is not None
    assert final.content == f"Mock final answer: {tool_result}"


@pytest.mark.skipif(
    os.getenv("PA_RUN_OLLAMA_INTEGRATION") != "1",
    reason="set PA_RUN_OLLAMA_INTEGRATION=1 with Ollama and the configured model available",
)
async def test_ollama_provider_regular_and_streaming() -> None:
    settings = _settings(
        llm_provider="ollama",
        llm_base_url=os.getenv("PA_LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_api_key="ollama",
        llm_model=os.getenv("PA_LLM_MODEL", "qwen3.5:9b-q4_K_M"),
        llm_think=False,
        http_timeout_s=120,
    )
    async with LLMClient(settings) as client:
        regular = await client.complete(
            _messages({"role": "user", "content": "Reply with exactly: ready"}),
            max_completion_tokens=32,
        )
        streamed_text, streamed = await _consume(client, "Reply with exactly: ready")

    assert regular.content
    assert streamed_text
    assert streamed.content == streamed_text


@pytest.mark.skipif(
    os.getenv("PA_RUN_OLLAMA_TOOL_INTEGRATION") != "1",
    reason="set PA_RUN_OLLAMA_TOOL_INTEGRATION=1 for a model with tool-call support",
)
async def test_ollama_provider_tool_call_when_supported() -> None:
    settings = _settings(
        llm_provider="ollama",
        llm_base_url=os.getenv("PA_LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_api_key="ollama",
        llm_model=os.getenv("PA_LLM_MODEL", "qwen3.5:9b-q4_K_M"),
        llm_think=False,
        http_timeout_s=120,
    )
    async with LLMClient(settings) as client:
        result = await client.complete(
            _messages({"role": "user", "content": "Call get_order for ORD-0042."}),
            tools=[_order_tool()],
            tool_choice="required",
        )

    assert any(call.executable and call.name == "get_order" for call in result.tool_calls)


@pytest.mark.skipif(
    os.getenv("PA_RUN_GROQ_INTEGRATION") != "1",
    reason="set PA_RUN_GROQ_INTEGRATION=1 with PA_LLM_API_KEY and PA_LLM_MODEL",
)
async def test_groq_provider_regular_and_streaming() -> None:
    api_key = os.environ["PA_LLM_API_KEY"]
    model = os.environ["PA_LLM_MODEL"]
    settings = _settings(
        llm_provider="groq",
        llm_base_url=os.getenv("PA_LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        llm_api_key=api_key,
        llm_model=model,
        http_timeout_s=60,
    )
    async with LLMClient(settings) as client:
        regular = await client.complete(
            _messages({"role": "user", "content": "Reply with exactly: ready"}),
            max_completion_tokens=32,
        )
        streamed_text, streamed = await _consume(client, "Reply with exactly: ready")

    assert regular.content
    assert streamed_text
    assert streamed.content == streamed_text
