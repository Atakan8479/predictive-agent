import os
from statistics import median
from time import perf_counter

import httpx
import pytest

from predictive_agent.bench.toxiproxy import ToxiproxyClient, load_profile

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("PA_RUN_INTEGRATION") != "1",
    reason="set PA_RUN_INTEGRATION=1 with the Compose stack running",
)
async def test_slow_profile_adds_downstream_latency_within_one_second() -> None:
    api_port = os.getenv("PA_TOXIPROXY_API_PORT", "8474")
    direct_port = os.getenv("PA_MOCK_APIS_PORT", "8080")
    proxy_port = os.getenv("PA_TOXIPROXY_MOCK_APIS_PORT", "18100")
    timeout = httpx.Timeout(3.0, connect=1.0)

    async with ToxiproxyClient(f"http://127.0.0.1:{api_port}") as toxiproxy:
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
