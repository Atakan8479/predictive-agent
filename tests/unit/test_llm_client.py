import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractContextManager, aclosing
from types import TracebackType
from typing import Any, cast

import httpx2
import pytest
from openai import APIConnectionError, AsyncOpenAI, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionToolUnionParam
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, Tracer

from predictive_agent.config import Settings
from predictive_agent.llm.client import (
    LLMClient,
    LLMDeadlineExceeded,
    LLMPartialStreamError,
    LLMProtocolError,
    LLMRetryExhausted,
)


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


def _tool(name: str, *required: str) -> ChatCompletionToolUnionParam:
    return cast(
        ChatCompletionToolUnionParam,
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {item: {"type": "string"} for item in required},
                    "required": list(required),
                    "additionalProperties": False,
                },
            },
        },
    )


def _completion(
    *,
    content: str | None = "ok",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: bool = True,
) -> ChatCompletion:
    payload: dict[str, Any] = {
        "id": "chatcmpl_test",
        "created": 1,
        "model": "returned-model",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
            }
        ],
        "usage": (
            {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15} if usage else None
        ),
    }
    return ChatCompletion.model_validate(payload)


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: bool = False,
) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl_stream",
            "created": 1,
            "model": "returned-model",
            "object": "chat.completion.chunk",
            "choices": (
                [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "delta": {"content": content, "tool_calls": tool_calls},
                    }
                ]
                if not usage
                else []
            ),
            "usage": (
                {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20} if usage else None
            ),
        }
    )


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx2.Request("POST", "http://provider.test/v1/chat"))


def _rate_limit_error(headers: dict[str, str] | None = None) -> RateLimitError:
    request = httpx2.Request("POST", "http://provider.test/v1/chat")
    response = httpx2.Response(429, request=request, headers=headers)
    return RateLimitError("rate limited", response=response, body={"error": "limited"})


def _bad_request_error() -> BadRequestError:
    request = httpx2.Request("POST", "http://provider.test/v1/chat")
    response = httpx2.Response(400, request=request)
    return BadRequestError("invalid", response=response, body={"error": "invalid"})


class _FakeStream:
    def __init__(self, items: list[ChatCompletionChunk | Exception]) -> None:
        self.items = items
        self.closed = False

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ChatCompletionChunk]:
        for item in self.items:
            if isinstance(item, Exception):
                raise item
            yield item

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.closed = True


class _BlockingStream:
    def __init__(self, entered: asyncio.Event) -> None:
        self.entered = entered
        self.closed = False

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ChatCompletionChunk]:
        self.entered.set()
        await asyncio.Event().wait()
        if False:  # pragma: no cover - establishes the async-generator item type
            yield _chunk()

    async def __aenter__(self) -> "_BlockingStream":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.closed = True


class _FirstThenBlockingStream(_FakeStream):
    def __init__(self, waiting: asyncio.Event) -> None:
        super().__init__([])
        self.waiting = waiting

    async def _iterate(self) -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk(content="first")
        self.waiting.set()
        await asyncio.Event().wait()


def _finalized_order_stream() -> _FakeStream:
    return _FakeStream(
        [
            _chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_order",
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "arguments": '{"order_id":"ORD-0042"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
            ),
            _chunk(usage=True),
        ]
    )


class _FakeCompletions:
    def __init__(self, outcomes: list[object | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if not self.outcomes:
            raise AssertionError("unexpected completion request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, Awaitable):
            return await cast(Awaitable[object], outcome)
        return outcome


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeSDK:
    def __init__(self, outcomes: list[object | Exception]) -> None:
        self.completions = _FakeCompletions(outcomes)
        self.chat = _FakeChat(self.completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.exceptions: list[Exception] = []
        self.status: Status | None = None
        self.end_count = 0

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def record_exception(self, exception: Exception, **_: object) -> None:
        self.exceptions.append(exception)

    def set_status(self, status: Status) -> None:
        self.status = status

    def end(self, *_: object, **__: object) -> None:
        self.end_count += 1


class _SpanContext(AbstractContextManager[_FakeSpan]):
    def __init__(self, span: _FakeSpan) -> None:
        self.span = span

    def __enter__(self) -> _FakeSpan:
        return self.span

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.span.end()


class _FakeTracer:
    def __init__(self) -> None:
        self.span = _FakeSpan()
        self.names: list[str] = []

    def start_as_current_span(self, name: str, **_: object) -> _SpanContext:
        self.names.append(name)
        return _SpanContext(self.span)

    def start_span(self, name: str, **_: object) -> _FakeSpan:
        self.names.append(name)
        return self.span


def _client(
    outcomes: list[object | Exception],
    *,
    settings: Settings | None = None,
    tracer: Tracer | _FakeTracer | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[LLMClient, _FakeSDK]:
    sdk = _FakeSDK(outcomes)
    client = LLMClient(
        settings or _settings(llm_provider="mock", llm_api_key="mock"),
        sdk_client=cast(AsyncOpenAI, sdk),
        tracer=cast(Tracer, tracer),
        sleep=sleep,
    )
    return client, sdk


@pytest.mark.parametrize(
    ("provider", "expected_extra"),
    [
        ("ollama", {"think": False, "keep_alive": "30m"}),
        ("mock", None),
        ("groq", None),
        ("openai-compatible", None),
    ],
)
async def test_provider_switching_is_configuration_only(
    provider: str, expected_extra: dict[str, object] | None
) -> None:
    api_key = "groq-secret" if provider == "groq" else provider
    client, sdk = _client(
        [_completion()], settings=_settings(llm_provider=provider, llm_api_key=api_key)
    )

    result = await client.complete([{"role": "user", "content": "hello"}])

    assert result.content == "ok"
    assert sdk.completions.calls[0].get("extra_body") == expected_extra
    assert sdk.completions.calls[0]["model"] == "qwen3.5:9b-q4_K_M"


async def test_regular_completion_preserves_usage_tools_and_messages() -> None:
    tool_calls = [
        {
            "id": "call_order",
            "type": "function",
            "function": {"name": "get_order", "arguments": '{"order_id":"ORD-0042"}'},
        }
    ]
    client, sdk = _client([_completion(content=None, tool_calls=tool_calls)])
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "call_order", "content": '{"status":"paid"}'},
    ]

    result = await client.complete(
        cast(Any, messages),
        tools=[_tool("get_order", "order_id")],
        max_completion_tokens=64,
    )

    assert result.metrics.usage.prompt_tokens == 11
    assert result.metrics.usage.completion_tokens == 4
    assert result.metrics.first_meaningful_output_s is None
    assert result.metrics.first_text_output_s is None
    assert result.metrics.first_tool_call_delta_s is None
    assert result.metrics.attempts == 1
    assert result.tool_calls[0].executable
    assert result.tool_calls[0].parsed_arguments == {"order_id": "ORD-0042"}
    assert sdk.completions.calls[0]["messages"] == messages
    assert sdk.completions.calls[0]["max_completion_tokens"] == 64


async def test_regular_completion_represents_missing_usage_as_unavailable() -> None:
    client, _ = _client([_completion(usage=False)])

    result = await client.complete([{"role": "user", "content": "hello"}])

    assert result.metrics.usage.prompt_tokens is None
    assert result.metrics.usage.completion_tokens is None
    assert result.metrics.usage.total_tokens is None


async def test_stream_accumulates_text_usage_and_timings() -> None:
    stream = _FakeStream(
        [
            _chunk(content=""),
            _chunk(content="hel"),
            _chunk(content="lo", finish_reason="stop"),
            _chunk(usage=True),
        ]
    )
    client, _ = _client([stream])

    events = [event async for event in client.stream([{"role": "user", "content": "hi"}])]

    assert [event.type for event in events] == ["text_delta", "text_delta", "completed"]
    result = events[-1].result
    assert result is not None
    assert result.content == "hello"
    assert result.finish_reason == "stop"
    assert result.metrics.first_transport_event_s is not None
    assert result.metrics.first_meaningful_output_s is not None
    assert result.metrics.first_text_output_s is not None
    assert result.metrics.first_tool_call_delta_s is None
    assert result.metrics.usage.total_tokens == 20
    assert stream.closed


async def test_stream_assembles_fragmented_multiple_tool_calls() -> None:
    stream = _FakeStream(
        [
            _chunk(
                tool_calls=[
                    {
                        "index": 1,
                        "id": "call_product",
                        "type": "function",
                        "function": {"name": "get_product", "arguments": '{"sku":"'},
                    },
                    {
                        "index": 0,
                        "id": "call_order",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": '{"order_'},
                    },
                ]
            ),
            _chunk(
                tool_calls=[
                    {"index": 0, "function": {"arguments": 'id":"ORD-0042"}'}},
                    {"index": 1, "function": {"arguments": 'SKU-0007"}'}},
                ],
                finish_reason="tool_calls",
            ),
            _chunk(usage=True),
        ]
    )
    client, _ = _client([stream])

    events = [
        event
        async for event in client.stream(
            [{"role": "user", "content": "lookup"}],
            tools=[_tool("get_order", "order_id"), _tool("get_product", "sku")],
        )
    ]

    ready = [event.tool_call for event in events if event.type == "tool_call_ready"]
    assert len(ready) == 2
    assert ready[0] is not None
    assert ready[0].parsed_arguments == {"order_id": "ORD-0042"}
    assert ready[1] is not None
    assert ready[1].parsed_arguments == {"sku": "SKU-0007"}
    result = events[-1].result
    assert result is not None
    assert result.metrics.first_text_output_s is None
    assert result.metrics.first_tool_call_delta_s is not None
    assert result.metrics.first_tool_call_ready_s is not None
    assert [call.index for call in result.tool_calls] == [0, 1]
    assert sum(event.type == "completed" for event in events) == 1


async def test_stream_keeps_malformed_tool_arguments_non_executable() -> None:
    stream = _FakeStream(
        [
            _chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": "{"},
                    }
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    client, _ = _client([stream])

    events = [
        event
        async for event in client.stream(
            [{"role": "user", "content": "lookup"}],
            tools=[_tool("get_order", "order_id")],
        )
    ]

    assert all(event.type != "tool_call_ready" for event in events)
    result = events[-1].result
    assert result is not None
    assert not result.tool_calls[0].executable
    assert result.tool_calls[0].readiness_error == "arguments_not_json_object"


async def test_parseable_tool_arguments_are_not_ready_before_finalization() -> None:
    stream = _FakeStream(
        [
            _chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_order",
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "arguments": '{"order_id":"ORD-0042"}',
                        },
                    }
                ]
            ),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    client, _ = _client([stream])
    events = client.stream(
        [{"role": "user", "content": "lookup"}],
        tools=[_tool("get_order", "order_id")],
    )

    delta = await anext(events)
    assert delta.type == "tool_call_delta"
    assert delta.tool_call is not None
    assert not delta.tool_call.executable
    assert delta.tool_call.readiness_error == "generation_not_finalized"

    ready = await anext(events)
    assert ready.type == "tool_call_ready"
    assert ready.tool_call is not None
    assert ready.tool_call.executable
    await events.aclose()


@pytest.mark.parametrize(
    ("name", "arguments", "expected_error"),
    [
        ("get_order", '{"unexpected":1}', "missing_required_arguments"),
        ("unknown_tool", '{"order_id":"ORD-0042"}', "unknown_tool"),
    ],
)
async def test_finalized_tool_calls_fail_the_explicit_readiness_gate(
    name: str, arguments: str, expected_error: str
) -> None:
    stream = _FakeStream(
        [
            _chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_test",
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    client, _ = _client([stream])

    events = [
        event
        async for event in client.stream(
            [{"role": "user", "content": "lookup"}],
            tools=[_tool("get_order", "order_id")],
        )
    ]

    assert all(event.type != "tool_call_ready" for event in events)
    result = events[-1].result
    assert result is not None
    call = result.tool_calls[0]
    assert call.finalized
    assert not call.executable
    assert call.readiness_error == expected_error
    if expected_error == "missing_required_arguments":
        assert call.missing_required_arguments == ("order_id",)


async def test_tool_call_without_finish_reason_remains_unfinalized() -> None:
    stream = _FakeStream(
        [
            _chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_order",
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "arguments": '{"order_id":"ORD-0042"}',
                        },
                    }
                ]
            )
        ]
    )
    client, _ = _client([stream])

    events = [
        event
        async for event in client.stream(
            [{"role": "user", "content": "lookup"}],
            tools=[_tool("get_order", "order_id")],
        )
    ]

    assert [event.type for event in events] == ["tool_call_delta", "completed"]
    result = events[-1].result
    assert result is not None
    assert result.tool_calls[0].readiness_error == "generation_not_finalized"


async def test_repeated_identity_is_allowed_but_conflicting_identity_is_rejected() -> None:
    repeated = _FakeStream(
        [
            _chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_order",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": "{"},
                    }
                ]
            ),
            _chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_order",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": "}"},
                    }
                ],
                finish_reason="tool_calls",
            ),
        ]
    )
    repeated_client, _ = _client([repeated])
    repeated_events = [
        event
        async for event in repeated_client.stream(
            [{"role": "user", "content": "lookup"}], tools=[_tool("get_order")]
        )
    ]
    assert sum(event.type == "tool_call_ready" for event in repeated_events) == 1

    conflicting = _FakeStream(
        [
            _chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_order",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": "{"},
                    }
                ]
            ),
            _chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_changed",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": "}"},
                    }
                ]
            ),
        ]
    )
    conflicting_client, _ = _client([conflicting])

    with pytest.raises(LLMProtocolError, match="changed a streamed tool-call identifier"):
        _ = [
            event
            async for event in conflicting_client.stream(
                [{"role": "user", "content": "lookup"}], tools=[_tool("get_order")]
            )
        ]


async def test_empty_regular_and_stream_responses_are_rejected() -> None:
    empty_completion = _completion()
    empty_completion.choices = []
    regular, _ = _client([empty_completion])
    streaming, _ = _client([_FakeStream([])])

    with pytest.raises(LLMProtocolError, match="no completion choices"):
        await regular.complete([{"role": "user", "content": "hello"}])
    with pytest.raises(LLMProtocolError, match="empty stream"):
        _ = [event async for event in streaming.stream([{"role": "user", "content": "hi"}])]


async def test_connection_failure_retries_then_succeeds() -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    client, sdk = _client([_connection_error(), _completion()], sleep=record_sleep)

    result = await client.complete([{"role": "user", "content": "hello"}])

    assert result.metrics.attempts == 2
    assert result.metrics.retry_count == 1
    assert delays == [pytest.approx(0.25)]
    assert len(sdk.completions.calls) == 2


async def test_connection_retry_exhaustion_is_bounded() -> None:
    async def no_wait(_: float) -> None:
        return None

    client, sdk = _client(
        [_connection_error(), _connection_error(), _connection_error()], sleep=no_wait
    )

    with pytest.raises(LLMRetryExhausted) as captured:
        await client.complete([{"role": "user", "content": "hello"}])

    assert captured.value.attempts == 3
    assert len(sdk.completions.calls) == 3


async def test_repeated_rate_limits_exhaust_retries_and_preserve_count() -> None:
    async def no_wait(_: float) -> None:
        return None

    tracer = _FakeTracer()
    client, sdk = _client(
        [_rate_limit_error(), _rate_limit_error(), _rate_limit_error()],
        sleep=no_wait,
        tracer=tracer,
    )

    with pytest.raises(LLMRetryExhausted) as captured:
        await client.complete([{"role": "user", "content": "hello"}])

    assert captured.value.attempts == 3
    assert captured.value.rate_limit_count == 3
    assert len(sdk.completions.calls) == 3
    assert tracer.span.attributes["llm.http_429_count"] == 3


@pytest.mark.parametrize(
    ("headers", "expected_delay"),
    [
        ({"retry-after": "0.75"}, 0.75),
        ({"retry-after-ms": "500"}, 0.5),
        ({"retry-after": "invalid"}, 0.25),
        ({}, 0.25),
    ],
)
async def test_rate_limit_retry_after_is_bounded(
    headers: dict[str, str], expected_delay: float
) -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    tracer = _FakeTracer()
    client, _ = _client(
        [_rate_limit_error(headers), _completion()], sleep=record_sleep, tracer=tracer
    )

    result = await client.complete([{"role": "user", "content": "hello"}])

    assert delays == [pytest.approx(expected_delay)]
    assert result.metrics.rate_limit_count == 1
    assert tracer.span.attributes["llm.http_429_count"] == 1


async def test_stream_retries_failure_before_output_without_duplication() -> None:
    async def no_wait(_: float) -> None:
        return None

    successful = _FakeStream([_chunk(content="complete", finish_reason="stop")])
    client, sdk = _client([_connection_error(), successful], sleep=no_wait)

    events = [event async for event in client.stream([{"role": "user", "content": "hello"}])]

    assert [event.content_delta for event in events if event.type == "text_delta"] == ["complete"]
    assert events[-1].result is not None
    assert events[-1].result.metrics.attempts == 2
    assert len(sdk.completions.calls) == 2
    assert successful.closed


async def test_non_positive_completion_budget_is_rejected_before_sdk_call() -> None:
    client, sdk = _client([_completion()])

    with pytest.raises(ValueError, match="must be positive"):
        await client.complete([{"role": "user", "content": "hello"}], max_completion_tokens=0)

    assert len(sdk.completions.calls) == 0


async def test_non_retryable_provider_error_is_not_retried() -> None:
    client, sdk = _client([_bad_request_error()])

    with pytest.raises(BadRequestError):
        await client.complete([{"role": "user", "content": "hello"}])

    assert len(sdk.completions.calls) == 1


async def test_expired_deadline_fails_before_request() -> None:
    client, sdk = _client([_completion()])

    with pytest.raises(LLMDeadlineExceeded, match="already expired"):
        await client.complete(
            [{"role": "user", "content": "hello"}],
            deadline=asyncio.get_running_loop().time(),
        )

    assert not sdk.completions.calls


async def test_stream_deadline_does_not_cancel_consumer_work_between_deltas() -> None:
    stream = _FakeStream([_chunk(content="first"), _chunk(content="second", finish_reason="stop")])
    client, _ = _client([stream])
    events = client.stream(
        [{"role": "user", "content": "hello"}],
        deadline=asyncio.get_running_loop().time() + 0.05,
    )

    first = await anext(events)
    assert first.content_delta == "first"
    unrelated_work_completed = False
    await asyncio.sleep(0.08)
    unrelated_work_completed = True

    assert unrelated_work_completed
    with pytest.raises(LLMDeadlineExceeded):
        await anext(events)
    assert stream.closed


async def test_consumer_can_await_unrelated_work_before_requesting_next_delta() -> None:
    stream = _FakeStream([_chunk(content="first"), _chunk(content="second", finish_reason="stop")])
    client, _ = _client([stream])
    events = client.stream([{"role": "user", "content": "hello"}])
    work_completed = asyncio.Event()

    first = await anext(events)

    async def unrelated_work() -> None:
        await asyncio.sleep(0)
        work_completed.set()

    await unrelated_work()
    second = await anext(events)
    assert first.content_delta == "first"
    assert work_completed.is_set()
    assert second.content_delta == "second"
    await events.aclose()


async def test_deadline_expires_while_waiting_for_next_provider_chunk() -> None:
    waiting = asyncio.Event()
    stream = _FirstThenBlockingStream(waiting)
    client, _ = _client([stream])
    events = client.stream(
        [{"role": "user", "content": "hello"}],
        deadline=asyncio.get_running_loop().time() + 0.05,
    )

    first = await anext(events)
    assert first.content_delta == "first"
    with pytest.raises(LLMDeadlineExceeded):
        await anext(events)

    assert waiting.is_set()
    assert stream.closed


async def test_deadline_expires_during_initial_stream_connection() -> None:
    pending: asyncio.Future[object] = asyncio.get_running_loop().create_future()
    client, sdk = _client([pending])

    with pytest.raises(LLMDeadlineExceeded):
        _ = [
            event
            async for event in client.stream(
                [{"role": "user", "content": "hello"}],
                deadline=asyncio.get_running_loop().time() + 0.02,
            )
        ]

    assert pending.cancelled()
    assert len(sdk.completions.calls) == 1


async def test_backoff_cannot_extend_absolute_deadline() -> None:
    client, _ = _client(
        [_connection_error()],
        settings=_settings(
            llm_provider="mock",
            llm_api_key="mock",
            llm_retry_backoff_initial_s=1,
            llm_retry_backoff_max_s=1,
        ),
    )

    with pytest.raises(LLMDeadlineExceeded, match="backoff exceeds"):
        await client.complete(
            [{"role": "user", "content": "hello"}],
            deadline=asyncio.get_running_loop().time() + 0.01,
        )


async def test_cancellation_during_backoff_propagates() -> None:
    sleeping = asyncio.Event()

    async def blocked_sleep(_: float) -> None:
        sleeping.set()
        await asyncio.Event().wait()

    tracer = _FakeTracer()
    client, _ = _client([_connection_error()], sleep=blocked_sleep, tracer=tracer)
    task = asyncio.create_task(client.complete([{"role": "user", "content": "hello"}]))
    await sleeping.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tracer.span.attributes["llm.outcome"] == "cancelled"


async def test_timeout_and_cancellation_close_active_stream() -> None:
    entered_timeout = asyncio.Event()
    timeout_stream = _BlockingStream(entered_timeout)
    timeout_client, _ = _client([timeout_stream])

    with pytest.raises(LLMDeadlineExceeded):
        _ = [
            event
            async for event in timeout_client.stream(
                [{"role": "user", "content": "hello"}],
                deadline=asyncio.get_running_loop().time() + 0.02,
            )
        ]
    assert timeout_stream.closed

    entered_cancel = asyncio.Event()
    cancel_stream = _BlockingStream(entered_cancel)
    cancel_client, _ = _client([cancel_stream])

    async def consume() -> None:
        _ = [event async for event in cancel_client.stream([{"role": "user", "content": "hello"}])]

    task = asyncio.create_task(consume())
    await entered_cancel.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancel_stream.closed


async def test_cancellation_during_consumer_processing_does_not_enter_generator() -> None:
    stream = _FakeStream([_chunk(content="first"), _chunk(content="second", finish_reason="stop")])
    tracer = _FakeTracer()
    client, _ = _client([stream], tracer=tracer)
    processing = asyncio.Event()

    async def consume() -> None:
        async with aclosing(client.stream([{"role": "user", "content": "hello"}])) as events:
            first = await anext(events)
            assert first.content_delta == "first"
            processing.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(consume())
    await processing.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed
    assert tracer.span.attributes["llm.outcome"] == "abandoned"
    assert tracer.span.end_count == 1


async def test_explicit_early_stream_closure_is_deterministic() -> None:
    stream = _FakeStream([_chunk(content="first"), _chunk(content="second", finish_reason="stop")])
    tracer = _FakeTracer()
    client, _ = _client([stream], tracer=tracer)
    events = client.stream([{"role": "user", "content": "hello"}])

    assert (await anext(events)).content_delta == "first"
    await events.aclose()

    assert stream.closed
    assert tracer.span.attributes["llm.outcome"] == "abandoned"
    assert tracer.span.end_count == 1


async def test_consumer_delay_after_tool_ready_does_not_change_final_metrics() -> None:
    stream = _finalized_order_stream()
    tracer = _FakeTracer()
    client, sdk = _client([stream], tracer=tracer)
    events = client.stream(
        [{"role": "user", "content": "lookup"}],
        tools=[_tool("get_order", "order_id")],
        deadline=asyncio.get_running_loop().time() + 0.05,
    )

    assert (await anext(events)).type == "tool_call_delta"
    ready = await anext(events)
    assert ready.type == "tool_call_ready"
    assert tracer.span.attributes["llm.outcome"] == "success"
    assert tracer.span.end_count == 1
    finalized_duration_ms = tracer.span.attributes["llm.duration_ms"]

    await asyncio.sleep(0.08)
    completed = await anext(events)

    assert completed.type == "completed"
    assert completed.result is not None
    assert completed.result.metrics.total_duration_s == pytest.approx(
        cast(float, finalized_duration_ms) / 1000
    )
    assert tracer.span.attributes["llm.duration_ms"] == finalized_duration_ms
    assert tracer.span.end_count == 1
    assert len(sdk.completions.calls) == 1
    await events.aclose()


async def test_close_after_tool_ready_preserves_successful_outcome() -> None:
    stream = _finalized_order_stream()
    tracer = _FakeTracer()
    client, _ = _client([stream], tracer=tracer)
    events = client.stream(
        [{"role": "user", "content": "lookup"}],
        tools=[_tool("get_order", "order_id")],
    )

    assert (await anext(events)).type == "tool_call_delta"
    assert (await anext(events)).type == "tool_call_ready"
    await events.aclose()

    assert stream.closed
    assert tracer.span.attributes["llm.outcome"] == "success"
    assert tracer.span.end_count == 1


async def test_multiple_ready_events_share_one_finalized_completion() -> None:
    stream = _FakeStream(
        [
            _chunk(
                tool_calls=[
                    {
                        "index": 1,
                        "id": "call_product",
                        "type": "function",
                        "function": {
                            "name": "get_product",
                            "arguments": '{"sku":"SKU-0007"}',
                        },
                    },
                    {
                        "index": 0,
                        "id": "call_order",
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "arguments": '{"order_id":"ORD-0042"}',
                        },
                    },
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    tracer = _FakeTracer()
    client, _ = _client([stream], tracer=tracer)
    events = client.stream(
        [{"role": "user", "content": "lookup"}],
        tools=[_tool("get_order", "order_id"), _tool("get_product", "sku")],
    )

    assert (await anext(events)).type == "tool_call_delta"
    assert (await anext(events)).type == "tool_call_delta"
    first_ready = await anext(events)
    finalized_duration_ms = tracer.span.attributes["llm.duration_ms"]
    await asyncio.sleep(0.02)
    second_ready = await anext(events)
    await asyncio.sleep(0.02)
    completed = await anext(events)

    ready_calls = [first_ready.tool_call, second_ready.tool_call]
    assert [call.index for call in ready_calls if call is not None] == [0, 1]
    assert completed.type == "completed"
    assert completed.result is not None
    assert completed.result.metrics.total_duration_s == pytest.approx(
        cast(float, finalized_duration_ms) / 1000
    )
    assert tracer.span.end_count == 1
    await events.aclose()


async def test_real_opentelemetry_context_does_not_leak_across_yields() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("tests.llm")
    stream = _finalized_order_stream()
    client, _ = _client([stream], tracer=tracer)

    with tracer.start_as_current_span("agent.parent") as parent:
        parent_id = parent.get_span_context().span_id
        events = client.stream(
            [{"role": "user", "content": "lookup"}],
            tools=[_tool("get_order", "order_id")],
        )
        assert (await anext(events)).type == "tool_call_delta"
        assert (await anext(events)).type == "tool_call_ready"
        assert trace.get_current_span().get_span_context().span_id == parent_id
        llm_finished = [
            span for span in exporter.get_finished_spans() if span.name == "agent.llm_turn"
        ]
        assert len(llm_finished) == 1
        assert llm_finished[0].attributes is not None
        assert llm_finished[0].attributes["llm.outcome"] == "success"

        with tracer.start_as_current_span("consumer.tool"):
            pass

        completed = await anext(events)
        assert completed.type == "completed"
        assert trace.get_current_span().get_span_context().span_id == parent_id
        await events.aclose()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    llm_span = spans["agent.llm_turn"]
    consumer_span = spans["consumer.tool"]
    assert llm_span.parent is not None
    assert llm_span.parent.span_id == parent_id
    assert consumer_span.parent is not None
    assert consumer_span.parent.span_id == parent_id
    assert llm_span.context is not None
    assert consumer_span.parent.span_id != llm_span.context.span_id


async def test_real_opentelemetry_span_ends_on_early_close_without_context_leak() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("tests.llm.early-close")
    stream = _FakeStream([_chunk(content="first"), _chunk(content="second", finish_reason="stop")])
    client, _ = _client([stream], tracer=tracer)

    with tracer.start_as_current_span("agent.parent") as parent:
        parent_id = parent.get_span_context().span_id
        events = client.stream([{"role": "user", "content": "hello"}])
        assert (await anext(events)).content_delta == "first"
        await events.aclose()
        assert trace.get_current_span().get_span_context().span_id == parent_id

    llm_spans = [span for span in exporter.get_finished_spans() if span.name == "agent.llm_turn"]
    assert len(llm_spans) == 1
    assert llm_spans[0].attributes is not None
    assert llm_spans[0].attributes["llm.outcome"] == "abandoned"


async def test_stream_failure_after_partial_output_is_not_retried() -> None:
    stream = _FakeStream([_chunk(content="partial"), _connection_error()])
    client, sdk = _client([stream, _FakeStream([_chunk(content="duplicate")])])
    observed: list[str] = []

    async def consume() -> None:
        async for event in client.stream([{"role": "user", "content": "hello"}]):
            if event.content_delta:
                observed.append(event.content_delta)

    with pytest.raises(LLMPartialStreamError):
        await consume()

    assert observed == ["partial"]
    assert len(sdk.completions.calls) == 1
    assert stream.closed


async def test_success_and_failure_spans_exclude_payloads_and_record_outcomes() -> None:
    success_tracer = _FakeTracer()
    success, _ = _client([_completion()], tracer=success_tracer)

    await success.complete([{"role": "user", "content": "secret prompt"}])

    assert success_tracer.names == ["agent.llm_turn"]
    assert success_tracer.span.attributes["llm.provider"] == "mock"
    assert success_tracer.span.attributes["llm.request.model"] == "qwen3.5:9b-q4_K_M"
    assert success_tracer.span.attributes["llm.usage.prompt_tokens"] == 11
    assert "secret prompt" not in repr(success_tracer.span.attributes)

    failure_tracer = _FakeTracer()
    failure, _ = _client([_bad_request_error()], tracer=failure_tracer)
    with pytest.raises(BadRequestError):
        await failure.complete([{"role": "user", "content": "secret prompt"}])
    assert failure_tracer.span.attributes["llm.outcome"] == "error"
    assert isinstance(failure_tracer.span.attributes["llm.duration_ms"], float)
    assert failure_tracer.span.exceptions


async def test_injected_client_retains_caller_ownership_and_close_is_idempotent() -> None:
    client, sdk = _client([_completion()])

    await client.close()
    await client.close()

    assert not sdk.closed
    with pytest.raises(RuntimeError, match="closed"):
        await client.complete([{"role": "user", "content": "hello"}])
