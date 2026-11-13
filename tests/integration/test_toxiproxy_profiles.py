import json
import os
import platform
import sys
from importlib.metadata import version
from statistics import median
from time import perf_counter

import httpx
import pytest

from predictive_agent.bench.toxiproxy import (
    ToxiproxyClient,
    load_profile,
    toxiproxy_api_url,
)
from predictive_agent.bench.toxiproxy_diagnostics import (
    measure_independent_connections,
    run_isolated_condition,
    summarize_attempts,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("PA_RUN_INTEGRATION") != "1",
    reason="set PA_RUN_INTEGRATION=1 with the Compose stack running",
)
async def test_slow_profile_adds_downstream_latency_within_one_second() -> None:
    direct_port = os.getenv("PA_MOCK_APIS_PORT", "8080")
    proxy_port = os.getenv("PA_TOXIPROXY_MOCK_APIS_PORT", "18100")
    timeout = httpx.Timeout(3.0, connect=1.0)

    async with ToxiproxyClient(toxiproxy_api_url()) as toxiproxy:
        started = perf_counter()
        await toxiproxy.apply(load_profile("slow"))
        apply_seconds = perf_counter() - started
        try:
            async with (
                httpx.AsyncClient(
                    base_url=f"http://127.0.0.1:{direct_port}", timeout=timeout
                ) as direct,
                httpx.AsyncClient(
                    base_url=f"http://127.0.0.1:{proxy_port}", timeout=timeout
                ) as proxied,
            ):
                direct_rtts = await _measure_rtts(direct, 3)
                proxied_rtts = await _measure_rtts(proxied, 5)
        finally:
            await toxiproxy.reset()

    added_seconds = median(proxied_rtts) - median(direct_rtts)
    print(
        f"slow profile: apply={apply_seconds:.3f}s, "
        f"direct median={median(direct_rtts):.3f}s, "
        f"proxied median={median(proxied_rtts):.3f}s, added={added_seconds:.3f}s"
    )
    assert apply_seconds < 1.0
    assert 0.55 <= added_seconds <= 1.10


async def _measure_rtts(client: httpx.AsyncClient, count: int) -> list[float]:
    durations: list[float] = []
    for _ in range(count):
        started = perf_counter()
        response = await client.get("/healthz")
        response.raise_for_status()
        durations.append(perf_counter() - started)
    return durations


@pytest.mark.skipif(
    os.getenv("PA_RUN_INTEGRATION") != "1" or os.getenv("PA_RUN_FLAKY_EXPERIMENT") != "1",
    reason="set PA_RUN_INTEGRATION=1 and PA_RUN_FLAKY_EXPERIMENT=1",
)
async def test_fault_profiles_produce_controlled_diagnostics() -> None:
    proxy_port = os.getenv("PA_TOXIPROXY_MOCK_APIS_PORT", "18100")
    attempts = int(os.getenv("PA_FLAKY_DIAGNOSTIC_ATTEMPTS", "30"))
    concurrency = int(os.getenv("PA_FLAKY_DIAGNOSTIC_CONCURRENCY", "5"))
    # The read timeout exceeds the 10 s timeout toxic so injected stalls are measured, not
    # automatically mislabeled as client failures.
    timeout = httpx.Timeout(connect=2.0, read=12.0, write=2.0, pool=2.0)
    conditions = (
        ("baseline", None),
        ("latency_only", load_profile("typical")),
        ("flaky", load_profile("flaky")),
    )
    summaries: list[dict[str, object]] = []

    async with ToxiproxyClient(toxiproxy_api_url()) as toxiproxy:
        for condition, profile in conditions:
            started = perf_counter()
            results = await run_isolated_condition(
                toxiproxy,
                profile,
                lambda: measure_independent_connections(
                    f"http://127.0.0.1:{proxy_port}",
                    attempts=attempts,
                    concurrency=concurrency,
                    request_timeout=timeout,
                ),
            )
            summary = summarize_attempts(
                condition,
                profile,
                results,
                attempts=attempts,
                concurrency=concurrency,
                timeout=timeout,
                elapsed_seconds=perf_counter() - started,
            )
            summaries.append(summary.as_dict())
            assert sum(summary.outcomes.values()) == attempts

    print(
        json.dumps(
            {
                "runtime": {
                    "python": sys.version.split()[0],
                    "httpx": version("httpx"),
                    "platform": platform.platform(),
                    "toxiproxy": "2.13.0",
                },
                "conditions": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )
