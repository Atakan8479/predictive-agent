import pytest

from predictive_agent.config import Settings


def test_defaults_target_local_ollama_with_prefetch_off() -> None:
    s = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_think is False
    assert s.prefetch_enabled is False


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PA_LLM_MODEL", "mock-llm")
    monkeypatch.setenv("PA_PREFETCH_ENABLED", "true")
    s = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert s.llm_model == "mock-llm"
    assert s.prefetch_enabled is True
