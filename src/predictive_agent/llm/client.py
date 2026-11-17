"""Provider-independent asynchronous chat client with one absolute deadline."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal, cast

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, AsyncStream, RateLimitError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolUnionParam,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from predictive_agent.config import Settings

Provider = Literal["ollama", "mock", "groq", "openai-compatible"]
StreamEventType = Literal["text_delta", "tool_call_delta", "tool_call_ready", "completed"]
Sleep = Callable[[float], Awaitable[None]]


class LLMError(Exception):
    """Base class for client-owned failures."""


class LLMDeadlineExceeded(TimeoutError, LLMError):
    """The single absolute operation deadline expired."""


class LLMProtocolError(LLMError):
    """A provider response did not satisfy the supported contract."""


class LLMPartialStreamError(LLMError):
    """A stream failed after output was exposed and was not retried."""


class LLMRetryExhausted(LLMError):
    """All configured additional retry attempts were consumed."""

    def __init__(self, attempts: int, rate_limit_count: int, cause: Exception) -> None:
        super().__init__(f"LLM request failed after {attempts} attempts: {type(cause).__name__}")
        self.attempts = attempts
        self.rate_limit_count = rate_limit_count
        self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class LLMUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True)
class LLMToolCall:
    index: int
    id: str | None
    name: str | None
    arguments: str
    parsed_arguments: dict[str, Any] | None

    @property
    def executable(self) -> bool:
        return self.id is not None and self.name is not None and self.parsed_arguments is not None


@dataclass(frozen=True, slots=True)
class LLMMetrics:
    provider: Provider
    requested_model: str
    returned_model: str | None
    first_transport_event_s: float | None
    first_meaningful_output_s: float | None
    first_text_output_s: float | None
    first_tool_call_delta_s: float | None
    first_tool_call_ready_s: float | None
    total_duration_s: float
    attempts: int
    retry_count: int
    rate_limit_count: int
    outcome: Literal["success", "error", "timeout", "cancelled"]
    usage: LLMUsage


@dataclass(frozen=True, slots=True)
class LLMResult:
    content: str | None
    tool_calls: tuple[LLMToolCall, ...]
    finish_reason: str | None
    metrics: LLMMetrics


@dataclass(frozen=True, slots=True)
class LLMStreamEvent:
    type: StreamEventType
    content_delta: str | None = None
    tool_call: LLMToolCall | None = None
    result: LLMResult | None = None


@dataclass(slots=True)
class _ToolAccumulator:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""
    ready_emitted: bool = False

    def snapshot(self) -> LLMToolCall:
        parsed: dict[str, Any] | None = None
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            parsed = cast(dict[str, Any], value)
        return LLMToolCall(self.index, self.id, self.name, self.arguments, parsed)


class LLMClient:
    """Own one reusable SDK client; callers own absolute deadlines and stream consumption.

    ``deadline`` values use ``asyncio.get_running_loop().time()``. If omitted, the client
    creates one from ``PA_HTTP_TIMEOUT_S``. The deadline covers retries, backoff, request I/O,
    and complete stream consumption. Injected SDK clients remain owned by their caller.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        sdk_client: AsyncOpenAI | None = None,
        tracer: Tracer | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self._owns_client = sdk_client is None
        self._client = sdk_client or AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.http_timeout_s,
            max_retries=0,
        )
        self._tracer = tracer or trace.get_tracer(__name__)
        self._sleep = sleep
        self._closed = False

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close an internally created SDK client; injected clients retain caller ownership."""
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.close()

    def deadline_after(self, timeout_s: float | None = None) -> float:
        """Return an absolute event-loop deadline for parent orchestration code."""
        timeout = self.settings.http_timeout_s if timeout_s is None else timeout_s
        if timeout <= 0:
            raise ValueError("timeout_s must be positive")
        return asyncio.get_running_loop().time() + timeout

    async def complete(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        tools: Sequence[ChatCompletionToolUnionParam] = (),
        tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
        max_completion_tokens: int | None = None,
        deadline: float | None = None,
    ) -> LLMResult:
        """Return one regular completion; TTFT is unavailable for non-streaming responses."""
        self._ensure_open()
        loop = asyncio.get_running_loop()
        started = loop.time()
        absolute_deadline = self._resolve_deadline(deadline, started)
        attempts = 0
        rate_limits = 0
        with self._tracer.start_as_current_span("agent.llm_turn") as span:
            self._start_span(span)
            while True:
                attempts += 1
                try:
                    remaining = self._remaining(absolute_deadline)
                    async with asyncio.timeout_at(absolute_deadline):
                        response = cast(
                            ChatCompletion,
                            await self._client.chat.completions.create(
                                **self._request_parameters(
                                    messages,
                                    tools=tools,
                                    tool_choice=tool_choice,
                                    max_completion_tokens=max_completion_tokens,
                                    stream=False,
                                    timeout=remaining,
                                )
                            ),
                        )
                    if not response.choices:
                        raise LLMProtocolError("provider returned no completion choices")
                    choice = response.choices[0]
                    usage = self._usage(response.usage)
                    metrics = self._metrics(
                        started,
                        returned_model=response.model,
                        attempts=attempts,
                        rate_limits=rate_limits,
                        usage=usage,
                    )
                    result = LLMResult(
                        content=choice.message.content,
                        tool_calls=self._response_tool_calls(choice.message.tool_calls),
                        finish_reason=choice.finish_reason,
                        metrics=metrics,
                    )
                    self._finish_span(span, metrics)
                    return result
                except asyncio.CancelledError:
                    self._fail_span(
                        span,
                        "cancelled",
                        started=started,
                        attempts=attempts,
                        rate_limits=rate_limits,
                    )
                    raise
                except (TimeoutError, APITimeoutError) as error:
                    failure = LLMDeadlineExceeded("LLM operation deadline exceeded")
                    self._fail_span(
                        span,
                        "timeout",
                        failure,
                        started=started,
                        attempts=attempts,
                        rate_limits=rate_limits,
                    )
                    raise failure from error
                except (APIConnectionError, RateLimitError) as error:
                    if isinstance(error, RateLimitError):
                        rate_limits += 1
                    if attempts > self.settings.llm_max_retries:
                        failure = LLMRetryExhausted(attempts, rate_limits, error)
                        self._fail_span(
                            span,
                            "error",
                            failure,
                            started=started,
                            attempts=attempts,
                            rate_limits=rate_limits,
                        )
                        raise failure from error
                    try:
                        await self._backoff(error, attempts, absolute_deadline)
                    except asyncio.CancelledError:
                        self._fail_span(
                            span,
                            "cancelled",
                            started=started,
                            attempts=attempts,
                            rate_limits=rate_limits,
                        )
                        raise
                    except LLMDeadlineExceeded as failure:
                        self._fail_span(
                            span,
                            "timeout",
                            failure,
                            started=started,
                            attempts=attempts,
                            rate_limits=rate_limits,
                        )
                        raise
                except Exception as error:
                    self._fail_span(
                        span,
                        "error",
                        error,
                        started=started,
                        attempts=attempts,
                        rate_limits=rate_limits,
                    )
                    raise

    async def stream(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        tools: Sequence[ChatCompletionToolUnionParam] = (),
        tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
        max_completion_tokens: int | None = None,
        deadline: float | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Yield meaningful deltas and one terminal event containing the final result.

        Retry is allowed only before any text or tool-call delta is yielded. A failure after
        partial output raises ``LLMPartialStreamError`` to prevent duplicate content or tools.
        """
        self._ensure_open()
        loop = asyncio.get_running_loop()
        started = loop.time()
        absolute_deadline = self._resolve_deadline(deadline, started)
        attempts = 0
        rate_limits = 0
        first_event: float | None = None
        first_meaningful: float | None = None
        first_text: float | None = None
        first_tool_delta: float | None = None
        first_tool_ready: float | None = None
        exposed = False
        with self._tracer.start_as_current_span("agent.llm_turn") as span:
            self._start_span(span)
            while True:
                attempts += 1
                content: list[str] = []
                calls: dict[int, _ToolAccumulator] = {}
                returned_model: str | None = None
                finish_reason: str | None = None
                usage = LLMUsage(None, None, None)
                try:
                    remaining = self._remaining(absolute_deadline)
                    async with asyncio.timeout_at(absolute_deadline):
                        stream = cast(
                            AsyncStream[ChatCompletionChunk],
                            await self._client.chat.completions.create(
                                **self._request_parameters(
                                    messages,
                                    tools=tools,
                                    tool_choice=tool_choice,
                                    max_completion_tokens=max_completion_tokens,
                                    stream=True,
                                    timeout=remaining,
                                )
                            ),
                        )
                        async with stream:
                            async for chunk in stream:
                                observed = loop.time()
                                if first_event is None:
                                    first_event = observed - started
                                returned_model = chunk.model
                                if chunk.usage is not None:
                                    usage = self._usage(chunk.usage)
                                if not chunk.choices:
                                    continue
                                choice = chunk.choices[0]
                                if choice.finish_reason is not None:
                                    finish_reason = choice.finish_reason
                                delta = choice.delta
                                if delta.content:
                                    if first_text is None:
                                        first_text = observed - started
                                    if first_meaningful is None:
                                        first_meaningful = observed - started
                                    content.append(delta.content)
                                    exposed = True
                                    yield LLMStreamEvent("text_delta", content_delta=delta.content)
                                for part in delta.tool_calls or []:
                                    accumulator = calls.setdefault(
                                        part.index, _ToolAccumulator(index=part.index)
                                    )
                                    if part.id is not None:
                                        accumulator.id = part.id
                                    name = part.function.name if part.function is not None else None
                                    arguments = (
                                        part.function.arguments
                                        if part.function is not None
                                        else None
                                    )
                                    if name is not None:
                                        accumulator.name = name
                                    if arguments:
                                        accumulator.arguments += arguments
                                    snapshot = accumulator.snapshot()
                                    if part.id or name or arguments:
                                        if first_tool_delta is None:
                                            first_tool_delta = observed - started
                                        if first_meaningful is None:
                                            first_meaningful = observed - started
                                        exposed = True
                                        yield LLMStreamEvent("tool_call_delta", tool_call=snapshot)
                                    if snapshot.executable and not accumulator.ready_emitted:
                                        accumulator.ready_emitted = True
                                        if first_tool_ready is None:
                                            first_tool_ready = observed - started
                                        yield LLMStreamEvent("tool_call_ready", tool_call=snapshot)
                    if first_event is None:
                        raise LLMProtocolError("provider returned an empty stream")
                    metrics = self._metrics(
                        started,
                        returned_model=returned_model,
                        attempts=attempts,
                        rate_limits=rate_limits,
                        usage=usage,
                        first_event=first_event,
                        first_meaningful=first_meaningful,
                        first_text=first_text,
                        first_tool_delta=first_tool_delta,
                        first_tool_ready=first_tool_ready,
                    )
                    result = LLMResult(
                        content="".join(content) or None,
                        tool_calls=tuple(calls[index].snapshot() for index in sorted(calls)),
                        finish_reason=finish_reason,
                        metrics=metrics,
                    )
                    self._finish_span(span, metrics)
                    yield LLMStreamEvent("completed", result=result)
                    return
                except asyncio.CancelledError:
                    self._fail_span(
                        span,
                        "cancelled",
                        started=started,
                        attempts=attempts,
                        rate_limits=rate_limits,
                    )
                    raise
                except (TimeoutError, APITimeoutError) as error:
                    failure = LLMDeadlineExceeded("LLM operation deadline exceeded")
                    self._fail_span(
                        span,
                        "timeout",
                        failure,
                        started=started,
                        attempts=attempts,
                        rate_limits=rate_limits,
                    )
                    raise failure from error
                except (APIConnectionError, RateLimitError) as error:
                    if isinstance(error, RateLimitError):
                        rate_limits += 1
                    if exposed:
                        failure = LLMPartialStreamError(
                            "stream failed after output was exposed; request was not retried"
                        )
                        self._fail_span(
                            span,
                            "error",
                            failure,
                            started=started,
                            attempts=attempts,
                            rate_limits=rate_limits,
                        )
                        raise failure from error
                    if attempts > self.settings.llm_max_retries:
                        failure = LLMRetryExhausted(attempts, rate_limits, error)
                        self._fail_span(
                            span,
                            "error",
                            failure,
                            started=started,
                            attempts=attempts,
                            rate_limits=rate_limits,
                        )
                        raise failure from error
                    try:
                        await self._backoff(error, attempts, absolute_deadline)
                    except asyncio.CancelledError:
                        self._fail_span(
                            span,
                            "cancelled",
                            started=started,
                            attempts=attempts,
                            rate_limits=rate_limits,
                        )
                        raise
                    except LLMDeadlineExceeded as failure:
                        self._fail_span(
                            span,
                            "timeout",
                            failure,
                            started=started,
                            attempts=attempts,
                            rate_limits=rate_limits,
                        )
                        raise
                except Exception as error:
                    self._fail_span(
                        span,
                        "error",
                        error,
                        started=started,
                        attempts=attempts,
                        rate_limits=rate_limits,
                    )
                    raise

    def _request_parameters(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        tools: Sequence[ChatCompletionToolUnionParam],
        tool_choice: ChatCompletionToolChoiceOptionParam | None,
        max_completion_tokens: int | None,
        stream: bool,
        timeout: float,
    ) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "stream": stream,
            "timeout": timeout,
        }
        if stream:
            parameters["stream_options"] = {"include_usage": True}
        if tools:
            parameters["tools"] = tools
        if tool_choice is not None:
            parameters["tool_choice"] = tool_choice
        if max_completion_tokens is not None:
            if max_completion_tokens <= 0:
                raise ValueError("max_completion_tokens must be positive")
            parameters["max_completion_tokens"] = max_completion_tokens
        if self.settings.llm_provider == "ollama":
            parameters["extra_body"] = {
                "think": self.settings.llm_think,
                "keep_alive": self.settings.ollama_keep_alive,
            }
        return parameters

    async def _backoff(
        self,
        error: APIConnectionError | RateLimitError,
        attempts: int,
        deadline: float,
    ) -> None:
        exponential = self.settings.llm_retry_backoff_initial_s * 2 ** (attempts - 1)
        retry_after = self._retry_after(error) if isinstance(error, RateLimitError) else None
        requested = max(exponential, retry_after or 0.0)
        delay = min(self.settings.llm_retry_backoff_max_s, requested)
        loop = asyncio.get_running_loop()
        if loop.time() + delay >= deadline:
            raise LLMDeadlineExceeded("retry backoff exceeds the remaining LLM deadline")
        try:
            async with asyncio.timeout_at(deadline):
                await self._sleep(delay)
        except TimeoutError as timeout:
            raise LLMDeadlineExceeded("LLM operation deadline exceeded during backoff") from timeout

    @staticmethod
    def _retry_after(error: RateLimitError) -> float | None:
        headers = error.response.headers
        milliseconds = headers.get("retry-after-ms")
        if milliseconds is not None:
            try:
                return max(0.0, float(milliseconds) / 1000)
            except ValueError:
                return None
        value = headers.get("retry-after")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return max(0.0, (parsed - datetime.now(UTC)).total_seconds())

    def _resolve_deadline(self, deadline: float | None, now: float) -> float:
        resolved = now + self.settings.http_timeout_s if deadline is None else deadline
        if resolved <= now:
            raise LLMDeadlineExceeded("LLM operation deadline already expired")
        return resolved

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise LLMDeadlineExceeded("LLM operation deadline already expired")
        return remaining

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LLMClient is closed")

    def _usage(self, usage: object | None) -> LLMUsage:
        if usage is None:
            return LLMUsage(None, None, None)
        return LLMUsage(
            self._optional_int(getattr(usage, "prompt_tokens", None)),
            self._optional_int(getattr(usage, "completion_tokens", None)),
            self._optional_int(getattr(usage, "total_tokens", None)),
        )

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return (
            value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        )

    @staticmethod
    def _response_tool_calls(
        tool_calls: Sequence[object] | None,
    ) -> tuple[LLMToolCall, ...]:
        results: list[LLMToolCall] = []
        if not isinstance(tool_calls, list):
            return ()
        for index, raw_call in enumerate(tool_calls):
            if not isinstance(raw_call, ChatCompletionMessageFunctionToolCall):
                continue
            accumulator = _ToolAccumulator(
                index=index,
                id=raw_call.id,
                name=raw_call.function.name,
                arguments=raw_call.function.arguments,
            )
            results.append(accumulator.snapshot())
        return tuple(results)

    def _metrics(
        self,
        started: float,
        *,
        returned_model: str | None,
        attempts: int,
        rate_limits: int,
        usage: LLMUsage,
        first_event: float | None = None,
        first_meaningful: float | None = None,
        first_text: float | None = None,
        first_tool_delta: float | None = None,
        first_tool_ready: float | None = None,
    ) -> LLMMetrics:
        return LLMMetrics(
            provider=self.settings.llm_provider,
            requested_model=self.settings.llm_model,
            returned_model=returned_model,
            first_transport_event_s=first_event,
            first_meaningful_output_s=first_meaningful,
            first_text_output_s=first_text,
            first_tool_call_delta_s=first_tool_delta,
            first_tool_call_ready_s=first_tool_ready,
            total_duration_s=asyncio.get_running_loop().time() - started,
            attempts=attempts,
            retry_count=attempts - 1,
            rate_limit_count=rate_limits,
            outcome="success",
            usage=usage,
        )

    def _start_span(self, span: Span) -> None:
        span.set_attribute("llm.provider", self.settings.llm_provider)
        span.set_attribute("llm.request.model", self.settings.llm_model)

    @staticmethod
    def _finish_span(span: Span, metrics: LLMMetrics) -> None:
        if metrics.returned_model is not None:
            span.set_attribute("llm.response.model", metrics.returned_model)
        span.set_attribute("llm.attempt_count", metrics.attempts)
        span.set_attribute("llm.retry_count", metrics.retry_count)
        span.set_attribute("llm.http_429_count", metrics.rate_limit_count)
        span.set_attribute("llm.duration_ms", metrics.total_duration_s * 1000)
        if metrics.first_meaningful_output_s is not None:
            span.set_attribute("llm.first_meaningful_ms", metrics.first_meaningful_output_s * 1000)
        if metrics.first_text_output_s is not None:
            span.set_attribute("llm.first_text_output_ms", metrics.first_text_output_s * 1000)
        if metrics.first_tool_call_delta_s is not None:
            span.set_attribute("llm.tool_call_delta_ms", metrics.first_tool_call_delta_s * 1000)
        if metrics.first_tool_call_ready_s is not None:
            span.set_attribute("llm.tool_call_ready_ms", metrics.first_tool_call_ready_s * 1000)
        if metrics.usage.prompt_tokens is not None:
            span.set_attribute("llm.usage.prompt_tokens", metrics.usage.prompt_tokens)
        if metrics.usage.completion_tokens is not None:
            span.set_attribute("llm.usage.completion_tokens", metrics.usage.completion_tokens)
        if metrics.usage.total_tokens is not None:
            span.set_attribute("llm.usage.total_tokens", metrics.usage.total_tokens)
        span.set_attribute("llm.outcome", metrics.outcome)
        span.set_status(Status(StatusCode.OK))

    @staticmethod
    def _fail_span(
        span: Span,
        outcome: Literal["error", "timeout", "cancelled"],
        error: Exception | None = None,
        *,
        started: float,
        attempts: int,
        rate_limits: int,
    ) -> None:
        span.set_attribute("llm.outcome", outcome)
        span.set_attribute("llm.attempt_count", attempts)
        span.set_attribute("llm.retry_count", max(0, attempts - 1))
        span.set_attribute("llm.http_429_count", rate_limits)
        span.set_attribute("llm.duration_ms", (asyncio.get_running_loop().time() - started) * 1000)
        if error is not None:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
        elif outcome != "cancelled":
            span.set_status(Status(StatusCode.ERROR, outcome))
