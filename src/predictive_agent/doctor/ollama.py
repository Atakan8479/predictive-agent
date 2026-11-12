"""Async Ollama probes: reachability, model availability, and a warm generation benchmark.

Ollama's native API is used here (not the OpenAI-compatible one) because it reports
per-phase timings (load, prompt eval, generation) in nanoseconds.
"""

from dataclasses import dataclass
from typing import Any, cast

import httpx

NS_PER_S = 1_000_000_000


class OllamaResponseError(ValueError):
    """Raised when Ollama returns a successful but unusable response."""


def _response_object(response: httpx.Response, endpoint: str) -> dict[str, Any]:
    try:
        data: Any = response.json()
    except ValueError as exc:
        raise OllamaResponseError(f"{endpoint} returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise OllamaResponseError(f"{endpoint} returned a non-object JSON response")
    return cast(dict[str, Any], data)


def _nonnegative_int(data: dict[str, Any], field: str) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OllamaResponseError(f"/api/generate field '{field}' must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class GenerationStats:
    load_s: float
    prompt_tokens: int
    prompt_eval_s: float
    output_tokens: int
    eval_s: float
    total_s: float
    prompt_cached_tokens: int | None

    @property
    def prompt_tokens_per_s(self) -> float:
        return self.prompt_tokens / self.prompt_eval_s if self.prompt_eval_s else 0.0

    @property
    def output_tokens_per_s(self) -> float:
        return self.output_tokens / self.eval_s if self.eval_s else 0.0

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> "GenerationStats":
        cached = data.get("prompt_eval_cached_count")
        if cached is not None and (
            isinstance(cached, bool) or not isinstance(cached, int) or cached < 0
        ):
            raise OllamaResponseError(
                "/api/generate field 'prompt_eval_cached_count' must be a non-negative integer"
            )
        return cls(
            load_s=_nonnegative_int(data, "load_duration") / NS_PER_S,
            prompt_tokens=_nonnegative_int(data, "prompt_eval_count"),
            prompt_eval_s=_nonnegative_int(data, "prompt_eval_duration") / NS_PER_S,
            output_tokens=_nonnegative_int(data, "eval_count"),
            eval_s=_nonnegative_int(data, "eval_duration") / NS_PER_S,
            total_s=_nonnegative_int(data, "total_duration") / NS_PER_S,
            prompt_cached_tokens=cached,
        )


class OllamaProbe:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def version(self) -> str:
        resp = await self._client.get("/api/version")
        resp.raise_for_status()
        version = _response_object(resp, "/api/version").get("version")
        if not isinstance(version, str) or not version:
            raise OllamaResponseError("/api/version field 'version' must be a non-empty string")
        return version

    async def installed_models(self) -> list[str]:
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        raw_models = _response_object(resp, "/api/tags").get("models")
        if not isinstance(raw_models, list):
            raise OllamaResponseError("/api/tags field 'models' must be a list")
        names: list[str] = []
        for raw_model in cast(list[Any], raw_models):
            if not isinstance(raw_model, dict):
                raise OllamaResponseError("/api/tags model entries must contain a string 'name'")
            model = cast(dict[str, Any], raw_model)
            name = model.get("name")
            if not isinstance(name, str):
                raise OllamaResponseError("/api/tags model entries must contain a string 'name'")
            names.append(name)
        return names

    async def generate(
        self, model: str, prompt: str, *, think: bool, keep_alive: str
    ) -> GenerationStats:
        resp = await self._client.post(
            "/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": think,
                "keep_alive": keep_alive,
            },
        )
        resp.raise_for_status()
        return GenerationStats.from_response(_response_object(resp, "/api/generate"))
