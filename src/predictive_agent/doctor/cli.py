"""`uv run pa-doctor` — checks that the local stack is ready and records a warm LLM baseline.

Runs the model once to warm it (load into VRAM), then measures N warm generations with a
prompt long enough to give a meaningful prompt-processing rate.
"""

import argparse
import asyncio
import statistics
import sys

import httpx

from predictive_agent.config import get_settings
from predictive_agent.doctor.ollama import GenerationStats, OllamaProbe

# ~400 tokens of realistic agent context so prompt-eval throughput is meaningful.
_CONTEXT = (
    "Sipariş #48213: 1x kablosuz kulaklık, 1x şarj kablosu. Durum: hazırlanıyor. "
    "Kargo firması: henüz atanmadı. Ödeme: kredi kartı, onaylandı. "
    "İade politikası: kargoya verilmemiş siparişler ücretsiz iptal edilebilir; "
    "teslim edilmiş ürünler 14 gün içinde, kullanılmamış ve orijinal ambalajında iade edilebilir. "
) * 6
BENCH_PROMPT = (
    f"Aşağıdaki bilgileri kullanarak müşteriye kısa ve net bir yanıt yaz.\n\n{_CONTEXT}\n\n"
    "Müşteri sorusu: Dün verdiğim siparişteki kulaklık hâlâ kargoya verilmedi, iade edebilir miyim?"
)


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


async def run(runs: int) -> int:
    settings = get_settings()
    timeout = httpx.Timeout(settings.http_timeout_s, connect=3.0)

    async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=timeout) as client:
        probe = OllamaProbe(client)

        print(f"Ollama @ {settings.ollama_url}")
        try:
            _ok(f"reachable, version {await probe.version()}")
        except httpx.HTTPError as exc:
            _fail(f"unreachable: {exc!r}")
            print("        WSL -> Windows: check `wslinfo --networking-mode` is 'mirrored'")
            print("        and that Ollama is running on Windows.")
            return 1

        models = await probe.installed_models()
        if settings.llm_model not in models:
            _fail(f"model '{settings.llm_model}' not installed. Installed: {models}")
            return 1
        _ok(f"model '{settings.llm_model}' installed")

        print(f"\nWarm-up (think={settings.llm_think}, keep_alive={settings.ollama_keep_alive})")
        warm = await probe.generate(
            settings.llm_model,
            "Merhaba.",
            think=settings.llm_think,
            keep_alive=settings.ollama_keep_alive,
        )
        _ok(f"loaded in {warm.load_s:.2f}s")

        print(f"\nBenchmark: {runs} warm runs")
        results: list[GenerationStats] = []
        for i in range(runs):
            s = await probe.generate(
                settings.llm_model,
                BENCH_PROMPT,
                think=settings.llm_think,
                keep_alive=settings.ollama_keep_alive,
            )
            results.append(s)
            print(
                f"  run {i + 1}: prompt {s.prompt_tokens} tok @ {s.prompt_tokens_per_s:,.0f} tok/s"
                f" | output {s.output_tokens} tok @ {s.output_tokens_per_s:.1f} tok/s"
                f" | total {s.total_s:.2f}s (load {s.load_s:.2f}s)"
            )

    def med(values: list[float]) -> float:
        return statistics.median(values)

    print("\nMedian")
    print(f"  prompt eval : {med([r.prompt_tokens_per_s for r in results]):,.0f} tok/s")
    print(f"  generation  : {med([r.output_tokens_per_s for r in results]):.1f} tok/s")
    print(f"  total       : {med([r.total_s for r in results]):.2f}s")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the local LLM stack and record a baseline.")
    parser.add_argument("--runs", type=int, default=5, help="number of warm benchmark runs")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.runs)))


if __name__ == "__main__":
    main()
