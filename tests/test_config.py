import pytest
from pydantic import ValidationError

from predictive_agent.config import Settings


def test_defaults_target_local_ollama_with_prefetch_off() -> None:
    # Pydantic Settings accepts _env_file at runtime, but its generated signature omits it.
    s = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_think is False
    assert s.mock_llm_base_latency_ms == pytest.approx(83.783)
    assert s.mock_llm_prefill_tps == pytest.approx(3422.979)
    assert s.mock_llm_decode_tps == pytest.approx(66.979)
    assert s.prefetch_enabled is False


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PA_LLM_MODEL", "mock-llm")
    monkeypatch.setenv("PA_PREFETCH_ENABLED", "true")
    # Pydantic Settings accepts _env_file at runtime, but its generated signature omits it.
    s = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert s.llm_model == "mock-llm"
    assert s.prefetch_enabled is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PA_MOCK_LLM_BASE_LATENCY_MS", "-1"),
        ("PA_MOCK_LLM_PREFILL_TPS", "0"),
        ("PA_MOCK_LLM_DECODE_TPS", "invalid"),
    ],
)
def test_invalid_mock_llm_timing_is_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
