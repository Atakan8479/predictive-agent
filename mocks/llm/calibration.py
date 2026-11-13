"""Calibrated character-to-token and timing models for the mock LLM."""

from dataclasses import dataclass

from predictive_agent.config import Settings

TOKEN_INTERCEPT = 36.757288119157025
PLAIN_CHARACTER_COEFFICIENT = 0.18303445239788455
PUNCTUATION_COEFFICIENT = 0.6880185556557008
NON_ASCII_COEFFICIENT = 0.7512997579514635
EXTRA_MESSAGE_COEFFICIENT = 16.340529833802805
TOOL_COEFFICIENT = 111.80463131999849


def estimate_tokens(text: str) -> int:
    return estimate_prompt_tokens(text)


def estimate_prompt_tokens(text: str, *, message_count: int = 1, tool_count: int = 0) -> int:
    non_ascii = sum(not character.isascii() for character in text)
    punctuation = sum(
        character.isascii() and not character.isalnum() and not character.isspace()
        for character in text
    )
    plain = len(text) - non_ascii - punctuation
    estimate = (
        TOKEN_INTERCEPT
        + plain * PLAIN_CHARACTER_COEFFICIENT
        + punctuation * PUNCTUATION_COEFFICIENT
        + non_ascii * NON_ASCII_COEFFICIENT
        + max(0, message_count - 1) * EXTRA_MESSAGE_COEFFICIENT
        + max(0, tool_count) * TOOL_COEFFICIENT
    )
    return max(1, round(estimate))


def estimate_content_tokens(text: str) -> int:
    """Estimate generated tokens without the prompt template intercept."""
    return max(1, estimate_tokens(text) - round(TOKEN_INTERCEPT))


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

    def total_seconds(self, prompt_tokens: int, completion_tokens: int) -> float:
        return self.ttft_seconds(prompt_tokens) + completion_tokens / self.decode_tps
