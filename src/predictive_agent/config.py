"""Application settings, loaded from environment variables and an optional `.env` file.

Every component reads configuration from here so that switching LLM provider
(Ollama -> Groq -> mock) or toggling prefetch is a config change, never a code change.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider (any OpenAI-compatible endpoint) ----------------------
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"  # Ollama ignores it; Groq needs a real key
    llm_model: str = "qwen3.5:9b-q4_K_M"
    llm_think: bool = False  # reasoning mode dominates latency; keep off for the agent

    # --- Ollama-specific (used by the doctor and model warm-up) -------------
    ollama_url: str = "http://localhost:11434"
    ollama_keep_alive: str = "30m"  # avoid cold reloads skewing latency measurements

    # --- Local infrastructure ---------------------------------------------
    toxiproxy_api_port: int = Field(default=8474, ge=1, le=65535)
    mock_llm_base_latency_ms: float = Field(default=75.460, ge=0, allow_inf_nan=False)
    mock_llm_prefill_tps: float = Field(default=3503.696, gt=0, allow_inf_nan=False)
    mock_llm_decode_tps: float = Field(default=68.048, gt=0, allow_inf_nan=False)

    # --- Feature flags -----------------------------------------------------
    prefetch_enabled: bool = False

    # --- Timeouts ----------------------------------------------------------
    http_timeout_s: float = Field(default=120.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
