import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from mocks.llm.app import app
from mocks.llm.calibration import (
    TimingModel,
    estimate_content_tokens,
    estimate_prompt_tokens,
    estimate_tokens,
)

from predictive_agent.config import Settings, get_settings

FAST_HEADERS = {
    "X-Mock-LLM-Base-Latency-Ms": "0",
    "X-Mock-LLM-Prefill-TPS": "1000000000",
    "X-Mock-LLM-Decode-TPS": "1000000000",
}
TOOL_PARAMETERS = {
    "get_order": "order_id",
    "get_shipment": "order_id",
    "get_customer": "customer_id",
    "get_policy": "policy_name",
    "get_product": "sku",
}
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_order": {"type": "string", "pattern": r"^ORD-\d{4}$"},
    "get_shipment": {"type": "string", "pattern": r"^ORD-\d{4}$"},
    "get_customer": {"type": "string", "pattern": r"^CUST-\d{3}$"},
    "get_policy": {
        "type": "string",
        "enum": ["returns", "cancellations", "shipping"],
    },
    "get_product": {"type": "string", "pattern": r"^SKU-\d{4}$"},
}


def _tool(name: str) -> dict[str, Any]:
    parameter = TOOL_PARAMETERS[name]
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {parameter: dict(TOOL_SCHEMAS[name])},
                "required": [parameter],
            },
        },
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
        "tools": [_tool("get_order")],
    }

    first = await client.post("/v1/chat/completions", headers=FAST_HEADERS, json=request)
    repeated = await client.post("/v1/chat/completions", headers=FAST_HEADERS, json=request)
    first_data = first.json()
    repeated_data = repeated.json()

    assert first_data["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = first_data["choices"][0]["message"]["tool_calls"][0]
    assert tool_call == repeated_data["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"] == {
        "name": "get_order",
        "arguments": '{"order_id":"ORD-0042"}',
    }

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


@pytest.mark.parametrize(
    ("name", "prompt", "expected"),
    [
        ("get_order", "Show order ORD-0042", {"order_id": "ORD-0042"}),
        ("get_shipment", "Track shipment ORD-0042", {"order_id": "ORD-0042"}),
        ("get_customer", "Show customer CUST-021", {"customer_id": "CUST-021"}),
        ("get_policy", "Explain the returns policy", {"policy_name": "returns"}),
        ("get_product", "Check inventory for SKU-0007", {"sku": "SKU-0007"}),
    ],
)
async def test_supported_tools_receive_executable_arguments(
    client: httpx.AsyncClient, name: str, prompt: str, expected: dict[str, str]
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": prompt}],
            "tools": [_tool(name)],
        },
    )

    choice = response.json()["choices"][0]
    call = choice["message"]["tool_calls"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert call["function"]["name"] == name
    assert json.loads(call["function"]["arguments"]) == expected


@pytest.mark.parametrize(
    ("prompt", "expected_fragment"),
    [
        ("Show my order", "order_id is missing"),
        ("Compare ORD-0001 and ORD-0002", "order_id is ambiguous"),
    ],
)
async def test_unsafe_tool_calls_return_explicit_fallback(
    client: httpx.AsyncClient, prompt: str, expected_fragment: str
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": prompt}],
            "tools": [_tool("get_order")],
        },
    )

    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert expected_fragment in choice["message"]["content"]
    assert "tool_calls" not in choice["message"]


async def test_multiple_tools_selects_the_scenario_named_by_user(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Track shipment ORD-0042"}],
            "tools": [_tool("get_order"), _tool("get_shipment")],
        },
    )

    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_shipment"


async def test_named_tool_choice_is_respected(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Track order ORD-0042"}],
            "tools": [_tool("get_order"), _tool("get_shipment")],
            "tool_choice": {"type": "function", "function": {"name": "get_order"}},
        },
    )

    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_order"
    assert json.loads(call["function"]["arguments"]) == {"order_id": "ORD-0042"}


async def test_overlapping_order_and_shipment_intent_selects_shipment(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Track the shipment for order ORD-0042."}],
            "tools": [_tool("get_order"), _tool("get_shipment")],
        },
    )

    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_shipment"


async def test_auto_skips_tools_when_current_request_has_no_external_intent(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Say hello"}],
            "tools": [_tool("get_order"), _tool("get_product")],
            "tool_choice": "auto",
        },
    )

    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": "Mock response complete."}


async def test_required_and_named_tool_failures_are_explicit(client: httpx.AsyncClient) -> None:
    required = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Show my order"}],
            "tools": [_tool("get_order")],
            "tool_choice": "required",
        },
    )
    unsupported: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "search_web",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    named = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Search for current news"}],
            "tools": [unsupported],
            "tool_choice": {"type": "function", "function": {"name": "search_web"}},
        },
    )

    assert required.status_code == 400
    assert "order_id is missing" in required.json()["detail"]
    assert named.status_code == 400
    assert "unsupported tool" in named.json()["detail"]


async def test_new_user_turn_ignores_historical_tool_result_and_identifiers(
    client: httpx.AsyncClient,
) -> None:
    historical_call = {
        "id": "call_order",
        "type": "function",
        "function": {"name": "get_order", "arguments": '{"order_id":"ORD-0042"}'},
    }
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [
                {"role": "user", "content": "Show order ORD-0042"},
                {"role": "assistant", "content": None, "tool_calls": [historical_call]},
                {
                    "role": "tool",
                    "tool_call_id": "call_order",
                    "content": '{"order_id":"ORD-0042","status":"delivered"}',
                },
                {"role": "assistant", "content": "The order was delivered."},
                {"role": "user", "content": "Check inventory for SKU-0007"},
            ],
            "tools": [_tool("get_order"), _tool("get_product")],
        },
    )

    choice = response.json()["choices"][0]
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_product"
    assert json.loads(call["function"]["arguments"]) == {"sku": "SKU-0007"}


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "user", "content": "Show order ORD-0042"},
            {"role": "tool", "tool_call_id": "missing", "content": "{}"},
        ],
        [
            {"role": "user", "content": "Show order ORD-0042"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": "not-json"},
                    }
                ],
            },
        ],
        [
            {"role": "user", "content": "Show order ORD-0042"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ],
        [
            {"role": "user", "content": "Show order ORD-0042"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": "{}"},
                    }
                ],
            },
        ],
    ],
)
async def test_invalid_tool_result_relationships_are_rejected(
    client: httpx.AsyncClient, messages: list[dict[str, Any]]
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={"model": "mock-calibrated", "messages": messages},
    )

    assert response.status_code == 422


async def test_unsupported_required_schema_returns_fallback(client: httpx.AsyncClient) -> None:
    tool = _tool("get_order")
    tool["function"]["parameters"]["properties"]["region"] = {"type": "string"}
    tool["function"]["parameters"]["required"].append("region")

    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Show order ORD-0042"}],
            "tools": [tool],
        },
    )

    assert (
        "required parameter 'region' is unavailable"
        in response.json()["choices"][0]["message"]["content"]
    )


async def test_supported_schema_constraints_generate_valid_boundary_values(
    client: httpx.AsyncClient,
) -> None:
    tool = _tool("get_order")
    parameters = tool["function"]["parameters"]
    parameters["additionalProperties"] = False
    parameters["properties"].update(
        {
            "region": {
                "type": "string",
                "const": "EU",
                "pattern": "^[A-Z]{2}$",
                "minLength": 2,
                "maxLength": 2,
            },
            "attempts": {"type": "integer", "default": 2, "minimum": 2, "maximum": 2},
            "ratio": {"type": "number", "enum": [0.5], "minimum": 0, "maximum": 1},
            "enabled": {"type": "boolean", "const": True},
            "note": {"type": "null", "const": None},
        }
    )
    parameters["required"].extend(["region", "attempts", "ratio", "enabled", "note"])

    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Show order ORD-0042"}],
            "tools": [tool],
        },
    )

    arguments = json.loads(
        response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    )
    assert arguments == {
        "attempts": 2,
        "enabled": True,
        "note": None,
        "order_id": "ORD-0042",
        "ratio": 0.5,
        "region": "EU",
    }


@pytest.mark.parametrize(
    ("schema_update", "expected"),
    [
        ({"oneOf": [{"type": "string"}]}, "unsupported constraint"),
        ({"type": "string", "default": "x", "minLength": 2}, "does not satisfy"),
        ({"type": "string", "const": "EU", "pattern": "["}, "invalid regular-expression"),
        ({"type": "integer", "default": 4, "maximum": 3}, "does not satisfy"),
        ({"type": "number", "default": 1, "minimum": 2}, "does not satisfy"),
        ({"type": "boolean", "default": 1}, "does not satisfy"),
        ({"type": "string", "enum": []}, "enum must be a non-empty array"),
    ],
)
async def test_malformed_or_unsatisfied_schema_constraints_are_explicit(
    client: httpx.AsyncClient, schema_update: dict[str, Any], expected: str
) -> None:
    tool = _tool("get_order")
    tool["function"]["parameters"]["properties"]["region"] = schema_update
    tool["function"]["parameters"]["required"].append("region")

    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Show order ORD-0042"}],
            "tools": [tool],
        },
    )

    assert expected in response.json()["choices"][0]["message"]["content"]


async def test_non_boolean_additional_properties_is_rejected(client: httpx.AsyncClient) -> None:
    tool = _tool("get_order")
    tool["function"]["parameters"]["additionalProperties"] = {"type": "string"}

    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Show order ORD-0042"}],
            "tools": [tool],
        },
    )

    assert (
        "additionalProperties must be a boolean"
        in response.json()["choices"][0]["message"]["content"]
    )


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
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    content = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk["choices"]
    )
    assert content == "Mock response complete."
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] > 0


async def test_tool_choice_none_disables_calls(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Use a tool"}],
            "stream": True,
            "tools": [_tool("get_order")],
            "tool_choice": "none",
        },
    )

    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line and line != "data: [DONE]"
    ]
    assert "tool_calls" not in chunks[0]["choices"][0]["delta"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_streaming_tool_call_uses_openai_delta_shape(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Show order ORD-0042"}],
            "stream": True,
            "tools": [_tool("get_order")],
        },
    )

    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line and line != "data: [DONE]"
    ]
    tool_delta = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert tool_delta["index"] == 0
    assert json.loads(tool_delta["function"]["arguments"]) == {"order_id": "ORD-0042"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


async def test_buffered_tool_call_arrives_after_full_decode_without_double_counting(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("mocks.llm.app.asyncio.sleep", record_sleep)
    payload = {
        "model": "mock-calibrated",
        "messages": [{"role": "user", "content": "Show order ORD-0042"}],
        "tools": [_tool("get_order")],
    }
    regular = await client.post(
        "/v1/chat/completions",
        headers={
            "X-Mock-LLM-Base-Latency-Ms": "100",
            "X-Mock-LLM-Prefill-TPS": "100",
            "X-Mock-LLM-Decode-TPS": "20",
        },
        json=payload,
    )
    regular_sleep = sleeps.pop()
    streamed = await client.post(
        "/v1/chat/completions",
        headers={
            "X-Mock-LLM-Base-Latency-Ms": "100",
            "X-Mock-LLM-Prefill-TPS": "100",
            "X-Mock-LLM-Decode-TPS": "20",
        },
        json={**payload, "stream": True},
    )

    assert regular.status_code == 200
    assert streamed.status_code == 200
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(regular_sleep)
    usage = regular.json()["usage"]
    assert regular_sleep == pytest.approx(
        0.1 + usage["prompt_tokens"] / 100 + usage["completion_tokens"] / 20
    )


async def test_text_stream_exposes_content_at_ttft_then_decodes(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("mocks.llm.app.asyncio.sleep", record_sleep)
    response = await client.post(
        "/v1/chat/completions",
        headers={
            "X-Mock-LLM-Base-Latency-Ms": "100",
            "X-Mock-LLM-Prefill-TPS": "100",
            "X-Mock-LLM-Decode-TPS": "20",
        },
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line and line != "data: [DONE]"
    ]
    usage = chunks[-1]["usage"]

    assert sleeps[0] == pytest.approx(0.1 + usage["prompt_tokens"] / 100)
    assert sum(sleeps[1:]) == pytest.approx(usage["completion_tokens"] / 20)


async def test_max_tokens_limits_regular_and_streaming_content(
    client: httpx.AsyncClient,
) -> None:
    payload = {
        "model": "mock-calibrated",
        "messages": [{"role": "user", "content": "Answer briefly"}],
        "max_tokens": 2,
    }
    regular = await client.post("/v1/chat/completions", headers=FAST_HEADERS, json=payload)
    streamed = await client.post(
        "/v1/chat/completions", headers=FAST_HEADERS, json={**payload, "stream": True}
    )

    regular_choice = regular.json()["choices"][0]
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in streamed.text.splitlines()
        if line and line != "data: [DONE]"
    ]
    streamed_content = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
    assert regular_choice["finish_reason"] == "length"
    assert regular_choice["message"]["content"] == streamed_content
    assert regular.json()["usage"]["completion_tokens"] <= 2
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


async def test_small_tool_token_limit_returns_explicit_error(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        headers=FAST_HEADERS,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": "Show order ORD-0042"}],
            "tools": [_tool("get_order")],
            "max_completion_tokens": 1,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "completion token limit is too small for a valid tool call"


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "mock", "messages": []},
        {"model": "mock", "messages": [{"role": "user", "content": "Hi"}], "unknown": 1},
        {
            "model": "mock",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "max_completion_tokens": 1,
        },
        {
            "model": "mock",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": "required",
        },
    ],
)
async def test_unsupported_or_invalid_request_fields_are_rejected(
    client: httpx.AsyncClient, payload: dict[str, Any]
) -> None:
    response = await client.post("/v1/chat/completions", headers=FAST_HEADERS, json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "header",
    [
        {"X-Mock-LLM-Base-Latency-Ms": "-1"},
        {"X-Mock-LLM-Prefill-TPS": "0"},
        {"X-Mock-LLM-Decode-TPS": "invalid"},
        {"X-Mock-LLM-Base-Latency-Ms": "nan"},
        {"X-Mock-LLM-Prefill-TPS": "inf"},
        {"X-Mock-LLM-Decode-TPS": "-inf"},
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
    assert estimate_tokens("a" * 128) >= estimate_tokens("a" * 64)
    assert estimate_tokens("{" * 128) > 0
    assert estimate_tokens("ü" * 128) > 0
    assert estimate_content_tokens("") == 0
    assert estimate_prompt_tokens("lookup", message_count=4, tool_count=1) > estimate_tokens(
        "lookup"
    )
    assert estimate_content_tokens("Mock response complete.") < estimate_tokens(
        "Mock response complete."
    )


def test_timing_model_uses_settings_and_overrides() -> None:
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    calibrated = TimingModel.from_settings(settings)
    overridden = TimingModel.from_settings(
        settings, base_latency_ms=10.0, prefill_tps=100.0, decode_tps=20.0
    )

    assert calibrated.base_latency_ms == pytest.approx(75.460)
    assert overridden.ttft_seconds(50) == pytest.approx(0.51)
    assert overridden.total_seconds(50, 10) == pytest.approx(1.01)
