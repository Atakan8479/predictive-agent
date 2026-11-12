"""Check the local stack and measure cached or uncached warm LLM performance."""

import argparse
import asyncio
import json
import statistics
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx

from predictive_agent.config import get_settings
from predictive_agent.doctor.ollama import GenerationStats, OllamaProbe, OllamaResponseError

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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _summaries(results: list[GenerationStats]) -> dict[str, dict[str, float]]:
    return {
        "prompt_tokens_per_s": _summary([result.prompt_tokens_per_s for result in results]),
        "output_tokens_per_s": _summary([result.output_tokens_per_s for result in results]),
        "total_s": _summary([result.total_s for result in results]),
    }


def _result_json(number: int, result: GenerationStats) -> dict[str, int | float | None]:
    return {
        "run": number,
        "load_s": result.load_s,
        "prompt_tokens": result.prompt_tokens,
        "prompt_eval_s": result.prompt_eval_s,
        "prompt_tokens_per_s": result.prompt_tokens_per_s,
        "prompt_eval_cached_count": result.prompt_cached_tokens,
        "output_tokens": result.output_tokens,
        "eval_s": result.eval_s,
        "output_tokens_per_s": result.output_tokens_per_s,
        "total_s": result.total_s,
    }


async def run(
    runs: int,
    *,
    cached: bool = False,
    json_path: Path | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    if runs < 1:
        _fail("--runs must be at least 1")
        return 2

    settings = get_settings()
    timeout = httpx.Timeout(settings.http_timeout_s, connect=3.0)
    mode = "cached" if cached else "uncached"

    try:
        async with httpx.AsyncClient(
            base_url=settings.ollama_url, timeout=timeout, transport=transport
        ) as client:
            probe = OllamaProbe(client)

            print(f"Ollama @ {settings.ollama_url}")
            version = await probe.version()
            _ok(f"reachable, version {version}")

            models = await probe.installed_models()
            if settings.llm_model not in models:
                _fail(f"model '{settings.llm_model}' not installed. Installed: {models}")
                return 1
            _ok(f"model '{settings.llm_model}' installed")

            print(
                f"\nWarm-up (think={settings.llm_think}, keep_alive={settings.ollama_keep_alive})"
            )
            warm = await probe.generate(
                settings.llm_model,
                "Merhaba.",
                think=settings.llm_think,
                keep_alive=settings.ollama_keep_alive,
            )
            _ok(f"loaded in {warm.load_s:.2f}s")

            print(f"\nBenchmark ({mode}): {runs} warm runs")
            results: list[GenerationStats] = []
            for i in range(runs):
                prompt = BENCH_PROMPT if cached else f"{uuid.uuid4()}\n{BENCH_PROMPT}"
                result = await probe.generate(
                    settings.llm_model,
                    prompt,
                    think=settings.llm_think,
                    keep_alive=settings.ollama_keep_alive,
                )
                results.append(result)
                cache = (
                    "cache n/a"
                    if result.prompt_cached_tokens is None
                    else f"cache {result.prompt_cached_tokens} tok"
                )
                print(
                    f"  run {i + 1}: prompt {result.prompt_tokens} tok"
                    f" @ {result.prompt_tokens_per_s:,.0f} tok/s ({cache})"
                    f" | output {result.output_tokens} tok @ {result.output_tokens_per_s:.1f} tok/s"
                    f" | total {result.total_s:.2f}s (load {result.load_s:.2f}s)"
                )
    except (httpx.HTTPError, OllamaResponseError) as exc:
        _fail(f"Ollama check failed: {exc}")
        print("        WSL -> Windows: check `wslinfo --networking-mode` is 'mirrored'")
        print("        and that Ollama is running on Windows.")
        return 1

    summaries = _summaries(results)
    print(f"\nSummary ({mode}): median / minimum / maximum")
    prompt_summary = summaries["prompt_tokens_per_s"]
    output_summary = summaries["output_tokens_per_s"]
    total_summary = summaries["total_s"]
    print(
        "  prompt eval : "
        f"{prompt_summary['median']:,.0f} / {prompt_summary['minimum']:,.0f} / "
        f"{prompt_summary['maximum']:,.0f} tok/s"
    )
    print(
        "  generation  : "
        f"{output_summary['median']:.1f} / {output_summary['minimum']:.1f} / "
        f"{output_summary['maximum']:.1f} tok/s"
    )
    print(
        f"  total       : {total_summary['median']:.2f} / {total_summary['minimum']:.2f} / "
        f"{total_summary['maximum']:.2f}s"
    )

    if json_path is not None:
        report: dict[str, Any] = {
            "mode": mode,
            "model": settings.llm_model,
            "ollama_version": version,
            "results": [_result_json(i, result) for i, result in enumerate(results, start=1)],
            "summaries": summaries,
        }
        try:
            await asyncio.to_thread(
                json_path.write_text, json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            _fail(f"could not write JSON report to '{json_path}': {exc}")
            return 1
        _ok(f"JSON report written to {json_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the local LLM stack and record a baseline.")
    parser.add_argument(
        "--runs", type=_positive_int, default=5, help="number of warm benchmark runs"
    )
    parser.add_argument(
        "--cached", action="store_true", help="repeat one prompt to measure the cached condition"
    )
    parser.add_argument("--json", type=Path, metavar="PATH", help="write a JSON benchmark report")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.runs, cached=args.cached, json_path=args.json)))


if __name__ == "__main__":
    main()
