"""OpenAI-compatible deterministic mock Chat Completions service."""

import asyncio
import hashlib
import json
import math
import re
import time
from collections.abc import AsyncIterator
from itertools import pairwise
from typing import Annotated, Any, Literal, cast

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mocks.llm.calibration import (
    TimingModel,
    estimate_content_tokens,
    estimate_prompt_tokens_for_payload,
    serialize_completion,
)
from predictive_agent.config import get_settings

app = FastAPI(title="Predictive Agent Mock LLM", version="1.0.0")

ToolSpec = tuple[str, re.Pattern[str], tuple[str, ...]]
SUPPORTED_TOOLS: dict[str, ToolSpec] = {
    "get_order": ("order_id", re.compile(r"\bORD-\d{4}\b", re.IGNORECASE), ("order",)),
    "get_shipment": (
        "order_id",
        re.compile(r"\bORD-\d{4}\b", re.IGNORECASE),
        ("shipment", "track", "tracking", "delivery"),
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

    @model_validator(mode="after")
    def validate_role_fields(self) -> "ChatMessage":
        if self.role == "tool":
            if not self.tool_call_id or self.content is None:
                raise ValueError("tool messages require content and tool_call_id")
            if self.tool_calls:
                raise ValueError("tool messages cannot contain tool_calls")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is valid only on tool messages")
        if self.tool_calls:
            if self.role != "assistant":
                raise ValueError("tool_calls are valid only on assistant messages")
            call_ids = [call.id for call in self.tool_calls]
            if any(not call_id for call_id in call_ids) or len(call_ids) != len(set(call_ids)):
                raise ValueError("assistant tool-call IDs must be non-empty and unique")
            for call in self.tool_calls:
                if call.function.arguments is None:
                    raise ValueError("assistant tool calls require JSON arguments")
                try:
                    arguments = json.loads(call.function.arguments)
                except json.JSONDecodeError as error:
                    raise ValueError("assistant tool-call arguments must be valid JSON") from error
                if not isinstance(arguments, dict):
                    raise ValueError("assistant tool-call arguments must be a JSON object")
        return self


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
        pending_ids: set[str] = set()
        resolved_ids: set[str] = set()
        for message in self.messages:
            if message.role == "assistant" and message.tool_calls:
                if pending_ids and pending_ids != resolved_ids:
                    raise ValueError("assistant tool calls require matching tool results")
                pending_ids = {call.id for call in message.tool_calls}
                resolved_ids = set()
            elif message.role == "tool":
                if message.tool_call_id not in pending_ids:
                    raise ValueError("tool result does not match the preceding assistant tool call")
                if message.tool_call_id in resolved_ids:
                    raise ValueError("duplicate tool result for the same tool call")
                resolved_ids.add(message.tool_call_id)
            elif pending_ids:
                if pending_ids != resolved_ids:
                    raise ValueError("assistant tool calls require matching tool results")
                pending_ids = set()
                resolved_ids = set()
        if pending_ids and pending_ids != resolved_ids:
            raise ValueError("assistant tool calls require matching tool results")
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
        float | None,
        Header(alias="X-Mock-LLM-Base-Latency-Ms", ge=0, allow_inf_nan=False),
    ] = None,
    prefill_tps: Annotated[
        float | None,
        Header(alias="X-Mock-LLM-Prefill-TPS", gt=0, allow_inf_nan=False),
    ] = None,
    decode_tps: Annotated[
        float | None,
        Header(alias="X-Mock-LLM-Decode-TPS", gt=0, allow_inf_nan=False),
    ] = None,
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
    prompt_tokens = prompt_tokens_for_request(request)
    completion_tokens = estimate_content_tokens(
        _generated_text(message), tool_call=bool(message.get("tool_calls"))
    )
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


def prompt_tokens_for_request(request: ChatCompletionRequest) -> int:
    messages = [message.model_dump(exclude_none=True) for message in request.messages]
    tools = [tool.model_dump(exclude_none=True) for tool in request.tools or []]
    return estimate_prompt_tokens_for_payload(messages, tools)


def _response_message(request: ChatCompletionRequest) -> tuple[dict[str, Any], str]:
    tool_result = _current_tool_result(request.messages)
    if tool_result is not None:
        return _text_response(f"Mock final answer: {tool_result}", request.completion_limit)
    if request.tools and request.tool_choice != "none":
        tool, arguments, fallback = _select_tool(request)
        if fallback is not None:
            if request.tool_choice == "required" or isinstance(
                request.tool_choice, NamedToolChoice
            ):
                raise HTTPException(status_code=400, detail=fallback)
            return _text_response(fallback, request.completion_limit)
        if tool is None or arguments is None:
            if request.tool_choice == "required":
                raise HTTPException(
                    status_code=400,
                    detail="tool_choice='required' could not construct a safe tool call",
                )
            return _text_response("Mock response complete.", request.completion_limit)
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


def _current_tool_result(messages: list[ChatMessage]) -> str | None:
    if not messages or messages[-1].role != "tool":
        return None
    assistant = next(
        (
            message
            for message in reversed(messages[:-1])
            if message.role == "assistant" and message.tool_calls
        ),
        None,
    )
    if assistant is None or not assistant.tool_calls:
        return None
    results = {
        message.tool_call_id: message.content
        for message in messages
        if message.role == "tool"
        and message.tool_call_id in {call.id for call in assistant.tool_calls}
    }
    ordered = [
        {"tool_call_id": call.id, "content": results[call.id]} for call in assistant.tool_calls
    ]
    if len(ordered) == 1:
        return str(ordered[0]["content"])
    return json.dumps(ordered, separators=(",", ":"), sort_keys=True)


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
    user_text = next(
        (message.content or "" for message in reversed(request.messages) if message.role == "user"),
        "",
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

    candidates: list[tuple[ChatTool, dict[str, Any], int]] = []
    failures: list[tuple[int, str]] = []
    for tool in available:
        result, failure = _tool_arguments(tool, user_text)
        score = _intent_score(tool.function.name, user_text)
        if result is not None:
            candidates.append((tool, result, score))
        elif failure is not None:
            failures.append((score, failure))
    if requested_name is not None and candidates:
        tool, arguments, _ = candidates[0]
        return tool, arguments, None
    if len(candidates) == 1:
        tool, arguments, _ = candidates[0]
        return tool, arguments, None
    if len(candidates) > 1:
        highest_score = max(candidate[2] for candidate in candidates)
        matching = [candidate for candidate in candidates if candidate[2] == highest_score]
        if len(matching) == 1:
            tool, arguments, _ = matching[0]
            return tool, arguments, None
        return None, None, "Mock cannot call a tool safely: the request is ambiguous."
    relevant_failures = [failure for score, failure in failures if score > 0]
    if requested_name is not None and failures:
        return None, None, failures[0][1]
    if relevant_failures:
        return None, None, relevant_failures[0]
    return None, None, None


def _intent_score(tool_name: str, text: str) -> int:
    spec = SUPPORTED_TOOLS.get(tool_name)
    if spec is None:
        return 0
    lowered = text.lower()
    return int(spec[1].search(text) is not None) + sum(
        len(re.findall(rf"\b{re.escape(keyword)}\b", lowered)) for keyword in spec[2]
    )


def _tool_arguments(tool: ChatTool, text: str) -> tuple[dict[str, Any] | None, str | None]:
    spec = SUPPORTED_TOOLS.get(tool.function.name)
    if spec is None:
        return None, f"Mock cannot call {tool.function.name!r}: unsupported tool."
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
    if not isinstance(parameters, dict):
        return None, f"Mock cannot call {tool.function.name!r}: malformed parameter schema."
    schema_error = _validate_object_schema(parameters)
    if schema_error is not None:
        return None, f"Mock cannot call {tool.function.name!r}: {schema_error}"
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
        valid, value = _schema_value(schema, identifier if name == identifier_name else None)
        if not valid:
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
        valid, _ = _schema_value(schema, identifier)
        if not valid:
            return None, f"Mock cannot call {tool.function.name!r}: incompatible identifier schema."
        arguments[identifier_name] = identifier
    return arguments, None


def _validate_object_schema(schema: dict[str, Any]) -> str | None:
    supported = {"type", "properties", "required", "additionalProperties"}
    unsupported = sorted(set(schema) - supported)
    if unsupported:
        return f"unsupported object constraint {unsupported[0]!r}."
    if schema.get("type") != "object":
        return "parameter schema type must be 'object'."
    properties = schema.get("properties")
    required = schema.get("required", [])
    additional = schema.get("additionalProperties", True)
    if not isinstance(properties, dict) or not isinstance(required, list):
        return "properties must be an object and required must be an array."
    if not isinstance(additional, bool):
        return "additionalProperties must be a boolean."
    required_items = cast(list[Any], required)
    property_items = cast(dict[Any, Any], properties)
    if any(not isinstance(name, str) for name in required_items) or len(required_items) != len(
        set(required_items)
    ):
        return "required entries must be unique strings."
    if any(name not in property_items for name in required_items):
        return "every required parameter must exist in properties."
    for name, property_schema in property_items.items():
        if not isinstance(name, str) or not isinstance(property_schema, dict):
            return "property names must be strings and property schemas must be objects."
        error = _validate_primitive_schema(cast(dict[str, Any], property_schema))
        if error is not None:
            return f"property {name!r} {error}"
    return None


def _validate_primitive_schema(schema: dict[str, Any]) -> str | None:
    supported = {
        "type",
        "enum",
        "const",
        "default",
        "pattern",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
    }
    unsupported = sorted(set(schema) - supported)
    if unsupported:
        return f"uses unsupported constraint {unsupported[0]!r}."
    expected = schema.get("type")
    if expected not in {"string", "integer", "number", "boolean", "null"}:
        return "must declare a supported primitive type."
    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        return "enum must be a non-empty array."
    if expected != "string" and any(
        constraint in schema for constraint in ("pattern", "minLength", "maxLength")
    ):
        return "uses string constraints with a non-string type."
    if expected == "string":
        minimum_length = schema.get("minLength", 0)
        maximum_length = schema.get("maxLength")
        if (
            isinstance(minimum_length, bool)
            or not isinstance(minimum_length, int)
            or minimum_length < 0
        ):
            return "minLength must be a non-negative integer."
        if maximum_length is not None and (
            isinstance(maximum_length, bool)
            or not isinstance(maximum_length, int)
            or maximum_length < minimum_length
        ):
            return "maxLength must be an integer greater than or equal to minLength."
        pattern = schema.get("pattern")
        if pattern is not None:
            if not isinstance(pattern, str):
                return "pattern must be a string."
            try:
                re.compile(pattern)
            except re.error:
                return "contains an invalid regular-expression pattern."
    if expected not in {"integer", "number"} and any(
        constraint in schema for constraint in ("minimum", "maximum")
    ):
        return "uses numeric constraints with a non-numeric type."
    if expected in {"integer", "number"}:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        for label, boundary in (("minimum", minimum), ("maximum", maximum)):
            if boundary is not None and (
                isinstance(boundary, bool)
                or not isinstance(boundary, (int, float))
                or not math.isfinite(boundary)
            ):
                return f"{label} must be a finite number."
        if minimum is not None and maximum is not None and minimum > maximum:
            return "minimum must be less than or equal to maximum."
    for source in ("const", "default"):
        if source in schema and _value_error(schema, schema[source]) is not None:
            return f"{source} does not satisfy its declared constraints."
    if "const" in schema and "default" in schema and schema["const"] != schema["default"]:
        return "const and default must agree when both are present."
    return None


def _schema_value(schema: dict[str, Any], identifier: str | None) -> tuple[bool, Any]:
    value: Any = identifier
    if value is None:
        if "const" in schema:
            value = schema["const"]
        elif "default" in schema:
            value = schema["default"]
        elif isinstance(schema.get("enum"), list) and len(schema["enum"]) == 1:
            value = schema["enum"][0]
        else:
            return False, None
    return _value_error(schema, value) is None, value


def _value_error(schema: dict[str, Any], value: Any) -> str | None:
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
        return "type"
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return "enum"
    if "const" in schema and value != schema["const"]:
        return "const"
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and isinstance(value, str) and re.fullmatch(pattern, value) is None:
        return "pattern"
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            return "minLength"
        maximum_length = schema.get("maxLength")
        if isinstance(maximum_length, int) and len(value) > maximum_length:
            return "maxLength"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            return "minimum"
        if maximum is not None and value > maximum:
            return "maximum"
    return None


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
    if message.get("tool_calls"):
        await asyncio.sleep(timing.tool_ready_seconds(prompt_tokens, completion_tokens))
        delta = {"role": "assistant", "tool_calls": [{"index": 0, **message["tool_calls"][0]}]}
        yield _event(_chunk(completion_id, created, request.model, delta, None))
    else:
        await asyncio.sleep(timing.ttft_seconds(prompt_tokens))
        chunks = _content_chunks(str(message["content"]), completion_tokens)
        if not chunks:
            yield _event(
                _chunk(
                    completion_id,
                    created,
                    request.model,
                    {"role": "assistant", "content": ""},
                    None,
                )
            )
        else:
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
                yield _event(
                    _chunk(completion_id, created, request.model, {"content": content}, None)
                )
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
    if token_count <= 0 or not content:
        return []
    boundaries = [round(index * len(content) / token_count) for index in range(token_count + 1)]
    return [content[start:end] for start, end in pairwise(boundaries) if end > start]


def _generated_text(message: dict[str, Any]) -> str:
    return serialize_completion(
        cast(str | None, message.get("content")),
        cast(list[dict[str, Any]], message.get("tool_calls", [])),
    )


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
