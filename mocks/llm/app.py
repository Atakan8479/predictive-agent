"""OpenAI-compatible deterministic mock Chat Completions service."""

import asyncio
import hashlib
import json
import re
import time
from collections.abc import AsyncIterator
from itertools import pairwise
from typing import Annotated, Any, Literal, cast

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mocks.llm.calibration import TimingModel, estimate_content_tokens, estimate_prompt_tokens
from predictive_agent.config import get_settings

app = FastAPI(title="Predictive Agent Mock LLM", version="1.0.0")

ToolSpec = tuple[str, re.Pattern[str], tuple[str, ...]]
SUPPORTED_TOOLS: dict[str, ToolSpec] = {
    "get_order": ("order_id", re.compile(r"\bORD-\d{4}\b", re.IGNORECASE), ("order",)),
    "get_shipment": (
        "order_id",
        re.compile(r"\bORD-\d{4}\b", re.IGNORECASE),
        ("shipment", "tracking", "delivery"),
    ),
    "get_customer": (
        "customer_id",
        re.compile(r"\bCUST-\d{3}\b", re.IGNORECASE),
        ("customer",),
    ),
    "get_policy": (
        "policy_name",
        re.compile(r"\b(?:returns|cancellations|shipping)\b", re.IGNORECASE),
        ("policy", "return", "cancel", "shipping"),
    ),
    "get_product": (
        "sku",
        re.compile(r"\bSKU-\d{4}\b", re.IGNORECASE),
        ("product", "inventory", "sku"),
    ),
}


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


class NamedToolChoiceFunction(BaseModel):
    name: str


class NamedToolChoice(BaseModel):
    type: Literal["function"]
    function: NamedToolChoiceFunction


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    tools: list[ChatTool] | None = None
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_token_limit(self) -> "ChatCompletionRequest":
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("max_tokens and max_completion_tokens cannot both be set")
        if (
            self.tool_choice == "required" or isinstance(self.tool_choice, NamedToolChoice)
        ) and not self.tools:
            raise ValueError("tool_choice requires at least one supplied tool")
        return self

    @property
    def completion_limit(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


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
    if message.get("tool_calls") and (
        request.completion_limit is not None and completion_tokens > request.completion_limit
    ):
        raise HTTPException(
            status_code=400,
            detail="completion token limit is too small for a valid tool call",
        )
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
    return estimate_prompt_tokens(
        "\n".join(parts),
        message_count=len(request.messages),
        tool_count=len(request.tools or []),
    )


def _response_message(request: ChatCompletionRequest) -> tuple[dict[str, Any], str]:
    tool_result = next(
        (message.content for message in reversed(request.messages) if message.role == "tool"), None
    )
    if tool_result is not None:
        return _text_response(f"Mock final answer: {tool_result}", request.completion_limit)
    if request.tools and request.tool_choice != "none":
        tool, arguments, fallback = _select_tool(request)
        if fallback is not None:
            return _text_response(fallback, request.completion_limit)
        if tool is None or arguments is None:
            raise RuntimeError("tool selection returned an invalid state")
        call_id = "call_" + _request_digest(request)[:12]
        return (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool.function.name,
                            "arguments": json.dumps(
                                arguments, separators=(",", ":"), sort_keys=True
                            ),
                        },
                    }
                ],
            },
            "tool_calls",
        )
    return _text_response("Mock response complete.", request.completion_limit)


def _text_response(content: str, limit: int | None) -> tuple[dict[str, Any], str]:
    if limit is None or estimate_content_tokens(content) <= limit:
        return {"role": "assistant", "content": content}, "stop"
    truncated = ""
    for index in range(1, len(content) + 1):
        candidate = content[:index]
        if estimate_content_tokens(candidate) > limit:
            break
        truncated = candidate
    return {"role": "assistant", "content": truncated}, "length"


def _select_tool(
    request: ChatCompletionRequest,
) -> tuple[ChatTool | None, dict[str, Any] | None, str | None]:
    if not request.tools:
        return None, None, "Mock cannot call a tool safely: no tools were supplied."
    user_text = "\n".join(
        message.content or "" for message in request.messages if message.role == "user"
    )
    requested_name = (
        request.tool_choice.function.name
        if isinstance(request.tool_choice, NamedToolChoice)
        else None
    )
    available = [
        tool
        for tool in request.tools
        if requested_name is None or tool.function.name == requested_name
    ]
    if requested_name is not None and not available:
        return None, None, f"Mock cannot call tool {requested_name!r}: it was not supplied."

    candidates: list[tuple[ChatTool, dict[str, Any]]] = []
    failures: list[str] = []
    for tool in available:
        result, failure = _tool_arguments(tool, user_text)
        if result is not None:
            candidates.append((tool, result))
        elif failure is not None:
            failures.append(failure)
    if requested_name is not None and candidates:
        tool, arguments = candidates[0]
        return tool, arguments, None
    if len(candidates) == 1:
        tool, arguments = candidates[0]
        return tool, arguments, None
    if len(candidates) > 1:
        lowered = user_text.lower()
        matching = [
            candidate
            for candidate in candidates
            if any(keyword in lowered for keyword in SUPPORTED_TOOLS[candidate[0].function.name][2])
        ]
        if len(matching) == 1:
            tool, arguments = matching[0]
            return tool, arguments, None
        return None, None, "Mock cannot call a tool safely: the request is ambiguous."
    if failures:
        return None, None, failures[0]
    return None, None, "Mock cannot call a tool safely: no supported compatible tool was supplied."


def _tool_arguments(tool: ChatTool, text: str) -> tuple[dict[str, Any] | None, str | None]:
    spec = SUPPORTED_TOOLS.get(tool.function.name)
    if spec is None:
        return None, None
    identifier_name, pattern, _ = spec
    identifiers = {
        _normalize_identifier(identifier_name, match.group()) for match in pattern.finditer(text)
    }
    if not identifiers:
        return None, f"Mock cannot call {tool.function.name!r}: {identifier_name} is missing."
    if len(identifiers) > 1:
        return None, f"Mock cannot call {tool.function.name!r}: {identifier_name} is ambiguous."
    identifier = next(iter(identifiers))
    parameters = tool.function.parameters
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        return None, f"Mock cannot call {tool.function.name!r}: unsupported parameter schema."
    raw_properties = parameters.get("properties")
    raw_required = parameters.get("required", [])
    if not isinstance(raw_properties, dict) or not isinstance(raw_required, list):
        return None, f"Mock cannot call {tool.function.name!r}: unsupported parameter schema."
    properties: dict[str, Any] = {}
    for name, schema in cast(dict[Any, Any], raw_properties).items():
        if not isinstance(name, str):
            return None, f"Mock cannot call {tool.function.name!r}: unsupported parameter schema."
        properties[name] = schema
    required: list[str] = []
    for name in cast(list[Any], raw_required):
        if not isinstance(name, str):
            return None, f"Mock cannot call {tool.function.name!r}: unsupported parameter schema."
        required.append(name)
    arguments: dict[str, Any] = {}
    for name in required:
        raw_schema = properties.get(name)
        if not isinstance(raw_schema, dict):
            return None, f"Mock cannot call {tool.function.name!r}: unsupported required parameter."
        schema = cast(dict[str, Any], raw_schema)
        value = _schema_value(schema, identifier if name == identifier_name else None)
        if value is None:
            return (
                None,
                "Mock cannot call "
                f"{tool.function.name!r}: required parameter {name!r} is unavailable.",
            )
        arguments[name] = value
    if identifier_name not in arguments:
        raw_schema = properties.get(identifier_name)
        if not isinstance(raw_schema, dict):
            return None, f"Mock cannot call {tool.function.name!r}: incompatible identifier schema."
        schema = cast(dict[str, Any], raw_schema)
        if _schema_value(schema, identifier) is None:
            return None, f"Mock cannot call {tool.function.name!r}: incompatible identifier schema."
        arguments[identifier_name] = identifier
    return arguments, None


def _schema_value(schema: dict[str, Any], identifier: str | None) -> Any | None:
    value: Any = identifier
    if value is None:
        if "const" in schema:
            value = schema["const"]
        elif "default" in schema:
            value = schema["default"]
        elif isinstance(schema.get("enum"), list) and len(schema["enum"]) == 1:
            value = schema["enum"][0]
        else:
            return None
    expected = schema.get("type")
    valid_type = (
        (expected == "string" and isinstance(value, str))
        or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (
            expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        or (expected == "boolean" and isinstance(value, bool))
        or (expected == "null" and value is None)
    )
    if not valid_type:
        return None
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return None
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and isinstance(value, str) and re.fullmatch(pattern, value) is None:
        return None
    return value


def _normalize_identifier(name: str, value: str) -> str:
    return value.lower() if name == "policy_name" else value.upper()


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
    if message.get("tool_calls"):
        delta = {"role": "assistant", "tool_calls": [{"index": 0, **message["tool_calls"][0]}]}
        yield _event(_chunk(completion_id, created, request.model, delta, None))
        await asyncio.sleep(completion_tokens / timing.decode_tps)
    else:
        chunks = _content_chunks(str(message["content"]), completion_tokens)
        first, *remaining = chunks
        yield _event(
            _chunk(
                completion_id,
                created,
                request.model,
                {"role": "assistant", "content": first},
                None,
            )
        )
        for content in remaining:
            await asyncio.sleep(1 / timing.decode_tps)
            yield _event(_chunk(completion_id, created, request.model, {"content": content}, None))
        await asyncio.sleep(1 / timing.decode_tps)
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
