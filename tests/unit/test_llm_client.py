import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, cast

import httpx2
import pytest
from openai import APIConnectionError, AsyncOpenAI, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
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

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def record_exception(self, exception: Exception, **_: object) -> None:
        self.exceptions.append(exception)

    def set_status(self, status: Status) -> None:
        self.status = status


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
        return None


class _FakeTracer:
    def __init__(self) -> None:
        self.span = _FakeSpan()
        self.names: list[str] = []

    def start_as_current_span(self, name: str, **_: object) -> _SpanContext:
        self.names.append(name)
        return _SpanContext(self.span)


def _client(
    outcomes: list[object | Exception],
    *,
    settings: Settings | None = None,
    tracer: _FakeTracer | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[LLMClient, _FakeSDK]:
    sdk = _FakeSDK(outcomes)
    client = LLMClient(
        settings or _settings(llm_provider="mock", llm_api_key="mock"),
        sdk_client=cast(AsyncOpenAI, sdk),
        tracer=cast(Tracer, tracer) if tracer is not None else None,
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

    result = await client.complete(cast(Any, messages), max_completion_tokens=64)

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
                        "index": 0,
                        "id": "call_order",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": '{"order_'},
                    },
                    {
                        "index": 1,
                        "id": "call_product",
                        "type": "function",
                        "function": {"name": "get_product", "arguments": '{"sku":"'},
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

    events = [event async for event in client.stream([{"role": "user", "content": "lookup"}])]

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

    events = [event async for event in client.stream([{"role": "user", "content": "lookup"}])]

    assert all(event.type != "tool_call_ready" for event in events)
    result = events[-1].result
    assert result is not None
    assert not result.tool_calls[0].executable


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
