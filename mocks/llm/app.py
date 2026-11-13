"""OpenAI-compatible deterministic mock Chat Completions service."""

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator
from itertools import pairwise
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mocks.llm.calibration import TimingModel, estimate_content_tokens, estimate_tokens
from predictive_agent.config import get_settings

app = FastAPI(title="Predictive Agent Mock LLM", version="1.0.0")


class ToolFunction(BaseModel):
    name: str
    arguments: str | None = None
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolFunction


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatTool(BaseModel):
    type: Literal["function"]
    function: ToolFunction


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    tools: list[ChatTool] | None = None
    max_tokens: int | None = Field(default=None, gt=0)


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    base_latency_ms: Annotated[
        float | None, Header(alias="X-Mock-LLM-Base-Latency-Ms", ge=0)
    ] = None,
    prefill_tps: Annotated[float | None, Header(alias="X-Mock-LLM-Prefill-TPS", gt=0)] = None,
    decode_tps: Annotated[float | None, Header(alias="X-Mock-LLM-Decode-TPS", gt=0)] = None,
) -> dict[str, Any] | StreamingResponse:
    timing = TimingModel.from_settings(
        get_settings(),
        base_latency_ms=base_latency_ms,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
    )
    completion_id = _completion_id(request)
    created = int(time.time())
    message, finish_reason = _response_message(request)
    prompt_tokens = _prompt_tokens(request)
    completion_tokens = estimate_content_tokens(_generated_text(message))
    if request.stream:
        return StreamingResponse(
            _stream_completion(
                request,
                timing,
                completion_id,
                created,
                message,
                finish_reason,
                prompt_tokens,
                completion_tokens,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    await asyncio.sleep(timing.total_seconds(prompt_tokens, completion_tokens))
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def _prompt_tokens(request: ChatCompletionRequest) -> int:
    parts = [message.content or "" for message in request.messages]
    if request.tools:
        parts.append(json.dumps([tool.model_dump() for tool in request.tools], sort_keys=True))
    return estimate_tokens("\n".join(parts)) + max(0, len(request.messages) - 1) * 4


def _response_message(request: ChatCompletionRequest) -> tuple[dict[str, Any], str]:
    tool_result = next(
        (message.content for message in reversed(request.messages) if message.role == "tool"), None
    )
    if tool_result is not None:
        return {"role": "assistant", "content": f"Mock final answer: {tool_result}"}, "stop"
    if request.tools:
        tool = request.tools[0]
        call_id = "call_" + _request_digest(request)[:12]
        return (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool.function.name, "arguments": "{}"},
                    }
                ],
            },
            "tool_calls",
        )
    return {"role": "assistant", "content": "Mock response complete."}, "stop"


async def _stream_completion(
    request: ChatCompletionRequest,
    timing: TimingModel,
    completion_id: str,
    created: int,
    message: dict[str, Any],
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> AsyncIterator[str]:
    await asyncio.sleep(timing.ttft_seconds(prompt_tokens))
    yield _event(_chunk(completion_id, created, request.model, {"role": "assistant"}, None))
    if message.get("tool_calls"):
        await asyncio.sleep(completion_tokens / timing.decode_tps)
        delta = {"tool_calls": [{"index": 0, **message["tool_calls"][0]}]}
        yield _event(_chunk(completion_id, created, request.model, delta, None))
    else:
        for content in _content_chunks(str(message["content"]), completion_tokens):
            await asyncio.sleep(1 / timing.decode_tps)
            yield _event(_chunk(completion_id, created, request.model, {"content": content}, None))
    yield _event(_chunk(completion_id, created, request.model, {}, finish_reason))
    if request.stream_options and request.stream_options.include_usage:
        usage_chunk = _chunk(completion_id, created, request.model, {}, None)
        usage_chunk["choices"] = []
        usage_chunk["usage"] = _usage(prompt_tokens, completion_tokens)
        yield _event(usage_chunk)
    yield "data: [DONE]\n\n"


def _chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}],
    }


def _event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def _content_chunks(content: str, token_count: int) -> list[str]:
    boundaries = [round(index * len(content) / token_count) for index in range(token_count + 1)]
    return [content[start:end] for start, end in pairwise(boundaries) if end > start]


def _generated_text(message: dict[str, Any]) -> str:
    if message.get("tool_calls"):
        return json.dumps(message["tool_calls"], sort_keys=True)
    return str(message["content"])


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _request_digest(request: ChatCompletionRequest) -> str:
    payload = request.model_dump_json(exclude_none=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _completion_id(request: ChatCompletionRequest) -> str:
    return "chatcmpl-mock-" + _request_digest(request)[:16]
