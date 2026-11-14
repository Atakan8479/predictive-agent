"""Calibrated character-to-token and timing models for the mock LLM."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from predictive_agent.config import Settings

TOKEN_INTERCEPT = 16.812280022802952
LETTER_COEFFICIENT = 0.0
DIGIT_COEFFICIENT = 1.2122806074966093
WHITESPACE_COEFFICIENT = 1.025593216335329
PUNCTUATION_COEFFICIENT = 0.583400704871238
NON_ASCII_COEFFICIENT = 0.924666587868699
EXTRA_MESSAGE_COEFFICIENT = 21.648646430276557
TOOL_CHARACTER_COEFFICIENT = 0.9415216672465215
CONTENT_INTERCEPT = 3.1651549622832036
CONTENT_LETTER_COEFFICIENT = 0.11195095763177693
CONTENT_DIGIT_COEFFICIENT = 1.2041482823072511
CONTENT_WHITESPACE_COEFFICIENT = 0.47882200942583003
CONTENT_PUNCTUATION_COEFFICIENT = 0.5659700126215336
CONTENT_NON_ASCII_COEFFICIENT = 1.135423696494941
TOOL_CONTENT_INTERCEPT = 0.0
TOOL_CONTENT_LETTER_COEFFICIENT = 0.3238325441961192
TOOL_CONTENT_DIGIT_COEFFICIENT = 2.066078556064583
TOOL_CONTENT_WHITESPACE_COEFFICIENT = 0.0
TOOL_CONTENT_PUNCTUATION_COEFFICIENT = 0.20983914863622546
TOOL_CONTENT_NON_ASCII_COEFFICIENT = 0.0


def serialize_prompt(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] = ()
) -> str:
    """Return the canonical prompt representation used by calibration and runtime."""
    payload: dict[str, Any] = {"messages": _normalize(messages)}
    if tools:
        payload["tools"] = _normalize(tools)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _normalize(value: Any, key: str | None = None) -> Any:
    if key == "arguments" and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, Mapping):
        typed = cast(Mapping[Any, Any], value)
        return {
            str(child_key): _normalize(child_value, str(child_key))
            for child_key, child_value in typed.items()
            if child_value is not None
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in cast(Sequence[Any], value)]
    return value


def serialize_completion(content: str | None, tool_calls: Sequence[Mapping[str, Any]] = ()) -> str:
    """Serialize only model-generated completion fields, excluding protocol IDs."""
    if not tool_calls:
        return content or ""
    functions: list[dict[str, Any]] = []
    for call in tool_calls:
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("tool call function must be an object")
        typed_function = cast(Mapping[str, Any], function)
        arguments = typed_function.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        functions.append(
            {
                "function": {
                    "arguments": arguments,
                    "name": typed_function.get("name"),
                },
                "type": "function",
            }
        )
    return json.dumps(functions, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def character_features(text: str) -> tuple[float, ...]:
    non_ascii = sum(not character.isascii() for character in text)
    letters = sum(character.isascii() and character.isalpha() for character in text)
    digits = sum(character.isascii() and character.isdigit() for character in text)
    whitespace = sum(character.isascii() and character.isspace() for character in text)
    punctuation = sum(
        character.isascii() and not character.isalnum() and not character.isspace()
        for character in text
    )
    return (
        1.0,
        float(letters),
        float(digits),
        float(whitespace),
        float(punctuation),
        float(non_ascii),
    )


def prompt_features(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] = ()
) -> tuple[float, ...]:
    message_parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            message_parts.append(content)
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, Sequence):
            message_parts.append(
                serialize_completion(None, cast(Sequence[Mapping[str, Any]], tool_calls))
            )
    tool_text = json.dumps(
        _normalize(tools), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return (
        *character_features("\n".join(message_parts)),
        float(max(0, len(messages) - 1)),
        float(len(tool_text) if tools else 0),
    )


def estimate_tokens(text: str) -> int:
    return estimate_prompt_tokens(text)


def estimate_prompt_tokens(text: str, *, message_count: int = 1, tool_count: int = 0) -> int:
    messages: list[dict[str, Any]] = [{"role": "user", "content": text}]
    messages.extend({"role": "assistant", "content": ""} for _ in range(message_count - 1))
    tools: list[dict[str, Any]] = [{} for _ in range(tool_count)]
    return estimate_prompt_tokens_for_payload(messages, tools)


def estimate_prompt_tokens_for_payload(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] = ()
) -> int:
    (
        _,
        letters,
        digits,
        whitespace,
        punctuation,
        non_ascii,
        extra_messages,
        tool_characters,
    ) = prompt_features(messages, tools)
    estimate = (
        TOKEN_INTERCEPT
        + letters * LETTER_COEFFICIENT
        + digits * DIGIT_COEFFICIENT
        + whitespace * WHITESPACE_COEFFICIENT
        + punctuation * PUNCTUATION_COEFFICIENT
        + non_ascii * NON_ASCII_COEFFICIENT
        + extra_messages * EXTRA_MESSAGE_COEFFICIENT
        + tool_characters * TOOL_CHARACTER_COEFFICIENT
    )
    return max(1, round(estimate))


def estimate_content_tokens(text: str, *, tool_call: bool = False) -> int:
    """Estimate model-generated tokens with a completion-specific regression."""
    if not text:
        return 0
    _, letters, digits, whitespace, punctuation, non_ascii = character_features(text)
    coefficients = (
        (
            TOOL_CONTENT_INTERCEPT,
            TOOL_CONTENT_LETTER_COEFFICIENT,
            TOOL_CONTENT_DIGIT_COEFFICIENT,
            TOOL_CONTENT_WHITESPACE_COEFFICIENT,
            TOOL_CONTENT_PUNCTUATION_COEFFICIENT,
            TOOL_CONTENT_NON_ASCII_COEFFICIENT,
        )
        if tool_call
        else (
            CONTENT_INTERCEPT,
            CONTENT_LETTER_COEFFICIENT,
            CONTENT_DIGIT_COEFFICIENT,
            CONTENT_WHITESPACE_COEFFICIENT,
            CONTENT_PUNCTUATION_COEFFICIENT,
            CONTENT_NON_ASCII_COEFFICIENT,
        )
    )
    estimate = (
        coefficients[0]
        + letters * coefficients[1]
        + digits * coefficients[2]
        + whitespace * coefficients[3]
        + punctuation * coefficients[4]
        + non_ascii * coefficients[5]
    )
    return max(1, round(estimate))


@dataclass(frozen=True, slots=True)
class TimingModel:
    base_latency_ms: float
    prefill_tps: float
    decode_tps: float

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        base_latency_ms: float | None = None,
        prefill_tps: float | None = None,
        decode_tps: float | None = None,
    ) -> "TimingModel":
        return cls(
            base_latency_ms=(
                settings.mock_llm_base_latency_ms if base_latency_ms is None else base_latency_ms
            ),
            prefill_tps=settings.mock_llm_prefill_tps if prefill_tps is None else prefill_tps,
            decode_tps=settings.mock_llm_decode_tps if decode_tps is None else decode_tps,
        )

    def ttft_seconds(self, prompt_tokens: int) -> float:
        return self.base_latency_ms / 1000 + prompt_tokens / self.prefill_tps

    def decode_seconds(self, completion_tokens: int) -> float:
        return completion_tokens / self.decode_tps

    def tool_ready_seconds(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Buffered tool JSON becomes executable only after it has fully decoded."""
        return self.ttft_seconds(prompt_tokens) + self.decode_seconds(completion_tokens)

    def total_seconds(self, prompt_tokens: int, completion_tokens: int) -> float:
        return self.tool_ready_seconds(prompt_tokens, completion_tokens)
