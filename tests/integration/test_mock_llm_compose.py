import json
import os
from statistics import median
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
CONTAINER_SCHEDULING_TOLERANCE_SECONDS = 0.025
TOOL_CASES = (
    ("get_order", "order_id", "ORD-0042", "/oms/orders/ORD-0042", "Show order ORD-0042"),
    (
        "get_shipment",
        "order_id",
        "ORD-0042",
        "/shipping/shipments/ORD-0042",
        "Track the shipment for order ORD-0042",
    ),
    (
        "get_customer",
        "customer_id",
        "CUST-021",
        "/crm/customers/CUST-021",
        "Show customer CUST-021",
    ),
    ("get_policy", "policy_name", "returns", "/policies/returns", "Explain returns policy"),
    (
        "get_product",
        "sku",
        "SKU-0007",
        "/inventory/products/SKU-0007",
        "Check inventory for SKU-0007",
    ),
)


def _tool(name: str, parameter: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {parameter: {"type": "string", "minLength": 1}},
                "required": [parameter],
                "additionalProperties": False,
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
        samples: list[tuple[float, float]] = []
        usage: dict[str, int] | None = None
        for _ in range(3):
            control_ttft, control_total, _ = await _measure_stream(client, prompt, FAST_HEADERS)
            measured_ttft, measured_total, usage = await _measure_stream(client, prompt)
            samples.append((measured_ttft - control_ttft, measured_total - control_total))

    assert usage is not None
    assert usage["prompt_tokens"] == prompt_tokens
    assert median(sample[0] for sample in samples) == pytest.approx(
        expected_ttft, rel=0.15, abs=CONTAINER_SCHEDULING_TOLERANCE_SECONDS
    )
    assert median(sample[1] for sample in samples) == pytest.approx(
        expected_total, rel=0.15, abs=CONTAINER_SCHEDULING_TOLERANCE_SECONDS
    )


@pytest.mark.skipif(
    os.getenv("PA_RUN_INTEGRATION") != "1",
    reason="set PA_RUN_INTEGRATION=1 with the Compose stack running",
)
async def test_mock_llm_buffers_tool_call_until_decode_completes() -> None:
    port = os.getenv("PA_MOCK_LLM_PORT", "8090")
    prompt = "Show order ORD-0042"
    tool = _tool("get_order", "order_id")
    timing = TimingModel(base_latency_ms=50, prefill_tps=1_000_000_000, decode_tps=20)
    headers = {
        "X-Mock-LLM-Base-Latency-Ms": "50",
        "X-Mock-LLM-Prefill-TPS": "1000000000",
        "X-Mock-LLM-Decode-TPS": "20",
    }
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}", timeout=httpx.Timeout(10.0, connect=2.0)
    ) as client:
        control = await _measure_tool_ready(client, prompt, tool, FAST_HEADERS)
        measured = await _measure_tool_ready(client, prompt, tool, headers)
        completion = await client.post(
            "/v1/chat/completions",
            headers=FAST_HEADERS,
            json={
                "model": "mock-calibrated",
                "messages": [{"role": "user", "content": prompt}],
                "tools": [tool],
            },
        )
    completion.raise_for_status()
    usage = cast(dict[str, int], completion.json()["usage"])
    expected = timing.tool_ready_seconds(usage["prompt_tokens"], usage["completion_tokens"])
    assert measured - control == pytest.approx(expected, rel=0.20, abs=0.05)


@pytest.mark.skipif(
    os.getenv("PA_RUN_INTEGRATION") != "1",
    reason="set PA_RUN_INTEGRATION=1 with the Compose stack running",
)
@pytest.mark.parametrize(("tool_name", "parameter", "identifier", "path", "prompt"), TOOL_CASES)
async def test_mock_llm_executes_complete_tool_round_trip(
    tool_name: str, parameter: str, identifier: str, path: str, prompt: str
) -> None:
    llm_port = os.getenv("PA_MOCK_LLM_PORT", "8090")
    api_port = os.getenv("PA_MOCK_APIS_PORT", "8080")
    timeout = httpx.Timeout(5.0, connect=2.0)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    request: dict[str, Any] = {
        "model": "mock-calibrated",
        "messages": messages,
        "tools": [_tool(tool_name, parameter)],
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

        api_response = await external_api.get(path)
        api_response.raise_for_status()
        resource = cast(dict[str, Any], api_response.json())
        tool_result = json.dumps(resource, separators=(",", ":"), sort_keys=True)
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
    assert tool_call["function"]["name"] == tool_name
    assert arguments == {parameter: identifier}
    assert api_response.headers["X-Mock-Served-At"]
    final_choice = final.json()["choices"][0]
    assert final_choice["finish_reason"] == "stop"
    assert final_choice["message"]["content"] == f"Mock final answer: {tool_result}"
    assert repeated.json()["choices"] == final.json()["choices"]


@pytest.mark.skipif(
    os.getenv("PA_RUN_INTEGRATION") != "1",
    reason="set PA_RUN_INTEGRATION=1 with the Compose stack running",
)
async def test_mock_llm_multi_turn_request_ignores_historical_tool_result() -> None:
    port = os.getenv("PA_MOCK_LLM_PORT", "8090")
    historical_call = {
        "id": "call_order_history",
        "type": "function",
        "function": {"name": "get_order", "arguments": '{"order_id":"ORD-0042"}'},
    }
    messages = [
        {"role": "user", "content": "Show order ORD-0042"},
        {"role": "assistant", "content": None, "tool_calls": [historical_call]},
        {
            "role": "tool",
            "tool_call_id": "call_order_history",
            "content": '{"order_id":"ORD-0042"}',
        },
        {"role": "user", "content": "Check inventory for SKU-0007"},
    ]
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}", timeout=httpx.Timeout(5.0, connect=2.0)
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers=FAST_HEADERS,
            json={
                "model": "mock-calibrated",
                "messages": messages,
                "tools": [_tool("get_order", "order_id"), _tool("get_product", "sku")],
            },
        )
    response.raise_for_status()
    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_product"
    assert json.loads(call["function"]["arguments"]) == {"sku": "SKU-0007"}


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


async def _measure_tool_ready(
    client: httpx.AsyncClient,
    prompt: str,
    tool: dict[str, Any],
    headers: dict[str, str],
) -> float:
    request = client.build_request(
        "POST",
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "mock-calibrated",
            "messages": [{"role": "user", "content": prompt}],
            "tools": [tool],
            "stream": True,
        },
    )
    started = perf_counter()
    response = await client.send(request, stream=True)
    try:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = cast(dict[str, Any], json.loads(line.removeprefix("data: ")))
            delta = event["choices"][0]["delta"]
            if delta.get("tool_calls"):
                return perf_counter() - started
    finally:
        await response.aclose()
    raise AssertionError("stream did not contain an executable tool call")
