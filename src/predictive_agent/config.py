"""Application settings, loaded from environment variables and an optional `.env` file.

Every component reads configuration from here so that switching LLM provider
(Ollama -> Groq -> mock) or toggling prefetch is a config change, never a code change.
"""

from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider (any OpenAI-compatible endpoint) ----------------------
    llm_provider: Literal["ollama", "mock", "groq", "openai-compatible"] = "ollama"
    llm_base_url: str = Field(default="http://localhost:11434/v1", min_length=1)
    llm_api_key: str = Field(default="ollama", min_length=1)
    llm_model: str = Field(default="qwen3.5:9b-q4_K_M", min_length=1)
    llm_think: bool = False  # reasoning mode dominates latency; keep off for the agent
    llm_max_retries: int = Field(default=2, ge=0)
    llm_retry_backoff_initial_s: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    llm_retry_backoff_max_s: float = Field(default=2.0, gt=0, allow_inf_nan=False)

    # --- Ollama-specific (used by the doctor and model warm-up) -------------
    ollama_url: str = "http://localhost:11434"
    ollama_keep_alive: str = "30m"  # avoid cold reloads skewing latency measurements

    # --- Local infrastructure ---------------------------------------------
    toxiproxy_api_port: int = Field(default=8474, ge=1, le=65535)
    mock_llm_base_latency_ms: float = Field(default=77.282, ge=0, allow_inf_nan=False)
    mock_llm_prefill_tps: float = Field(default=3089.067, gt=0, allow_inf_nan=False)
    mock_llm_decode_tps: float = Field(default=64.340, gt=0, allow_inf_nan=False)

    # --- Feature flags -----------------------------------------------------
    prefetch_enabled: bool = False

    # --- Timeouts ----------------------------------------------------------
    http_timeout_s: float = Field(default=120.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_llm_configuration(self) -> "Settings":
        parsed = urlsplit(self.llm_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("llm_base_url must be an absolute HTTP(S) URL")
        if self.llm_retry_backoff_max_s < self.llm_retry_backoff_initial_s:
            raise ValueError("llm_retry_backoff_max_s must be at least the initial backoff")
        if self.llm_provider == "groq" and self.llm_api_key in {"ollama", "mock"}:
            raise ValueError("Groq requires a provider API key")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
