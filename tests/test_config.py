import pytest
from pydantic import ValidationError

from predictive_agent.config import Settings


def test_defaults_target_local_ollama_with_prefetch_off() -> None:
    # Pydantic Settings accepts _env_file at runtime, but its generated signature omits it.
    s = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert s.llm_provider == "ollama"
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_think is False
    assert s.tool_api_base_url == "http://localhost:18100"
    assert s.mock_llm_base_latency_ms == pytest.approx(77.282)
    assert s.mock_llm_prefill_tps == pytest.approx(3089.067)
    assert s.mock_llm_decode_tps == pytest.approx(64.340)
    assert s.prefetch_enabled is False
    assert s.llm_max_retries == 2


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PA_LLM_PROVIDER", "mock")
    monkeypatch.setenv("PA_LLM_BASE_URL", "http://localhost:8090/v1")
    monkeypatch.setenv("PA_LLM_API_KEY", "mock")
    monkeypatch.setenv("PA_LLM_MODEL", "mock-llm")
    monkeypatch.setenv("PA_LLM_MAX_RETRIES", "4")
    monkeypatch.setenv("PA_LLM_RETRY_BACKOFF_INITIAL_S", "0.5")
    monkeypatch.setenv("PA_LLM_RETRY_BACKOFF_MAX_S", "3")
    monkeypatch.setenv("PA_PREFETCH_ENABLED", "true")
    monkeypatch.setenv("PA_MOCK_LLM_BASE_LATENCY_MS", "12.5")
    monkeypatch.setenv("PA_MOCK_LLM_PREFILL_TPS", "250")
    monkeypatch.setenv("PA_MOCK_LLM_DECODE_TPS", "40")
    # Pydantic Settings accepts _env_file at runtime, but its generated signature omits it.
    s = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert s.llm_provider == "mock"
    assert s.llm_base_url == "http://localhost:8090/v1"
    assert s.llm_api_key == "mock"
    assert s.llm_model == "mock-llm"
    assert s.llm_max_retries == 4
    assert s.llm_retry_backoff_initial_s == pytest.approx(0.5)
    assert s.llm_retry_backoff_max_s == pytest.approx(3)
    assert s.prefetch_enabled is True
    assert s.mock_llm_base_latency_ms == pytest.approx(12.5)
    assert s.mock_llm_prefill_tps == pytest.approx(250)
    assert s.mock_llm_decode_tps == pytest.approx(40)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PA_MOCK_LLM_BASE_LATENCY_MS", "-1"),
        ("PA_MOCK_LLM_PREFILL_TPS", "0"),
        ("PA_MOCK_LLM_DECODE_TPS", "invalid"),
        ("PA_MOCK_LLM_BASE_LATENCY_MS", "nan"),
        ("PA_MOCK_LLM_PREFILL_TPS", "inf"),
        ("PA_MOCK_LLM_DECODE_TPS", "-inf"),
    ],
)
def test_invalid_mock_llm_timing_is_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PA_LLM_PROVIDER", "unsupported"),
        ("PA_LLM_BASE_URL", "localhost:8090/v1"),
        ("PA_LLM_BASE_URL", "ftp://localhost/v1"),
        ("PA_LLM_MAX_RETRIES", "-1"),
        ("PA_LLM_RETRY_BACKOFF_INITIAL_S", "0"),
        ("PA_LLM_RETRY_BACKOFF_MAX_S", "nan"),
        ("PA_HTTP_TIMEOUT_S", "inf"),
        ("PA_TOOL_API_BASE_URL", "localhost:18100"),
        ("PA_TOOL_API_BASE_URL", "ftp://localhost"),
        ("PA_TOOL_API_BASE_URL", "http://localhost:18100/path"),
    ],
)
def test_invalid_llm_client_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # pyright: ignore[reportCallIssue]


def test_retry_backoff_maximum_cannot_be_below_initial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PA_LLM_RETRY_BACKOFF_INITIAL_S", "2")
    monkeypatch.setenv("PA_LLM_RETRY_BACKOFF_MAX_S", "1")

    with pytest.raises(ValidationError, match="at least the initial backoff"):
        Settings(_env_file=None)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize("api_key", ["ollama", "mock"])
def test_groq_rejects_local_placeholder_credentials(
    monkeypatch: pytest.MonkeyPatch, api_key: str
) -> None:
    monkeypatch.setenv("PA_LLM_PROVIDER", "groq")
    monkeypatch.setenv("PA_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("PA_LLM_API_KEY", api_key)

    with pytest.raises(ValidationError, match="Groq requires a provider API key"):
        Settings(_env_file=None)  # pyright: ignore[reportCallIssue]


def test_groq_configuration_accepts_explicit_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PA_LLM_PROVIDER", "groq")
    monkeypatch.setenv("PA_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("PA_LLM_API_KEY", "test-only-groq-key")
    monkeypatch.setenv("PA_LLM_MODEL", "test-groq-model")

    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]

    assert settings.llm_provider == "groq"
    assert settings.llm_model == "test-groq-model"
