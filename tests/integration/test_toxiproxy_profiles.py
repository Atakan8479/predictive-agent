import asyncio
import os
from collections import Counter
from statistics import median
from time import perf_counter

import httpx
import pytest

from predictive_agent.bench.toxiproxy import (
    ToxiproxyClient,
    load_profile,
    toxiproxy_api_url,
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
async def test_flaky_profile_records_independent_connection_outcomes() -> None:
    proxy_port = os.getenv("PA_TOXIPROXY_MOCK_APIS_PORT", "18100")
    attempts = 100
    semaphore = asyncio.Semaphore(20)

    async with ToxiproxyClient(toxiproxy_api_url()) as toxiproxy:
        await toxiproxy.apply(load_profile("flaky"))
        try:
            outcomes = await asyncio.gather(
                *(_attempt_independent_connection(proxy_port, semaphore) for _ in range(attempts))
            )
        finally:
            await toxiproxy.reset()

    counts = Counter(outcomes)
    print(
        "flaky profile observations: "
        f"attempts={attempts}, success={counts['success']}, resets={counts['reset']}, "
        f"timeouts={counts['timeout']}, other_failures={counts['other_failure']}"
    )
    assert counts.total() == attempts


async def _attempt_independent_connection(proxy_port: str, semaphore: asyncio.Semaphore) -> str:
    timeout = httpx.Timeout(1.5, connect=0.5)
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=0)
    async with semaphore:
        try:
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{proxy_port}",
                headers={"Connection": "close"},
                limits=limits,
                timeout=timeout,
            ) as client:
                response = await client.get("/healthz")
                response.raise_for_status()
        except httpx.TimeoutException:
            return "timeout"
        except httpx.TransportError:
            return "reset"
        except httpx.HTTPError:
            return "other_failure"
    return "success"
