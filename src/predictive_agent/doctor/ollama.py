"""Async Ollama probes: reachability, model availability, and a warm generation benchmark.

Ollama's native API is used here (not the OpenAI-compatible one) because it reports
per-phase timings (load, prompt eval, generation) in nanoseconds.
"""

from dataclasses import dataclass
from typing import Any

import httpx

NS_PER_S = 1_000_000_000


@dataclass(frozen=True, slots=True)
class GenerationStats:
    load_s: float
    prompt_tokens: int
    prompt_eval_s: float
    output_tokens: int
    eval_s: float
    total_s: float

    @property
    def prompt_tokens_per_s(self) -> float:
        return self.prompt_tokens / self.prompt_eval_s if self.prompt_eval_s else 0.0

    @property
    def output_tokens_per_s(self) -> float:
        return self.output_tokens / self.eval_s if self.eval_s else 0.0

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> "GenerationStats":
        return cls(
            load_s=data.get("load_duration", 0) / NS_PER_S,
            prompt_tokens=data.get("prompt_eval_count", 0),
            prompt_eval_s=data.get("prompt_eval_duration", 0) / NS_PER_S,
            output_tokens=data.get("eval_count", 0),
            eval_s=data.get("eval_duration", 0) / NS_PER_S,
            total_s=data.get("total_duration", 0) / NS_PER_S,
        )


class OllamaProbe:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def version(self) -> str:
        resp = await self._client.get("/api/version")
        resp.raise_for_status()
        return str(resp.json()["version"])

    async def installed_models(self) -> list[str]:
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        return [str(m["name"]) for m in resp.json().get("models", [])]

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
        return GenerationStats.from_response(resp.json())
