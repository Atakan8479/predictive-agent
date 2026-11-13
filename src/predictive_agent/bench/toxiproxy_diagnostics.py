"""Controlled measurements for local Toxiproxy fault profiles."""

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from statistics import median
from time import perf_counter
from typing import Literal

import httpx

from predictive_agent.bench.toxiproxy import ProfileDefinition, ToxiproxyClient

Outcome = Literal[
    "success",
    "connect_timeout",
    "read_timeout",
    "write_timeout",
    "pool_timeout",
    "connect_error",
    "read_error",
    "remote_protocol_error",
    "transport_error",
    "http_status_error",
]


@dataclass(frozen=True)
class AttemptResult:
    outcome: Outcome
    rtt_seconds: float
    detail: str | None = None


@dataclass(frozen=True)
class DiagnosticSummary:
    condition: str
    profile: str | None
    toxics: list[dict[str, object]]
    attempts: int
    concurrency: int
    timeout_seconds: dict[str, float]
    outcomes: dict[str, int]
    successful_rtt_seconds: dict[str, float] | None
    elapsed_seconds: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DiagnosticRunError(RuntimeError):
    """A diagnostic failed and cleanup may also have failed."""

    def __init__(self, error: Exception, cleanup_error: Exception | None = None) -> None:
        message = f"Diagnostic failed: {error}"
        if cleanup_error is not None:
            message += f"; cleanup also failed: {cleanup_error}"
        super().__init__(message)
        self.error = error
        self.cleanup_error = cleanup_error


def classify_httpx_error(error: httpx.HTTPError) -> Outcome:
    """Return the narrowest category supported by HTTPX's exception type."""
    if isinstance(error, httpx.ConnectTimeout):
        return "connect_timeout"
    if isinstance(error, httpx.ReadTimeout):
        return "read_timeout"
    if isinstance(error, httpx.WriteTimeout):
        return "write_timeout"
    if isinstance(error, httpx.PoolTimeout):
        return "pool_timeout"
    if isinstance(error, httpx.ConnectError):
        return "connect_error"
    if isinstance(error, httpx.ReadError):
        return "read_error"
    if isinstance(error, httpx.RemoteProtocolError):
        return "remote_protocol_error"
    if isinstance(error, httpx.HTTPStatusError):
        return "http_status_error"
    return "transport_error"


def summarize_attempts(
    condition: str,
    profile: ProfileDefinition | None,
    results: list[AttemptResult],
    *,
    attempts: int,
    concurrency: int,
    timeout: httpx.Timeout,
    elapsed_seconds: float,
) -> DiagnosticSummary:
    successful_rtts = sorted(
        result.rtt_seconds for result in results if result.outcome == "success"
    )
    rtt_summary = None
    if successful_rtts:
        rtt_summary = {
            "median": median(successful_rtts),
            "p95": _percentile(successful_rtts, 0.95),
            "max": successful_rtts[-1],
        }
    return DiagnosticSummary(
        condition=condition,
        profile=profile.name if profile else None,
        toxics=(
            [toxic.model_dump(mode="json", by_alias=True) for toxic in profile.toxics]
            if profile
            else []
        ),
        attempts=attempts,
        concurrency=concurrency,
        timeout_seconds={
            "connect": _required_timeout(timeout.connect),
            "read": _required_timeout(timeout.read),
            "write": _required_timeout(timeout.write),
            "pool": _required_timeout(timeout.pool),
        },
        outcomes=dict(sorted(Counter(result.outcome for result in results).items())),
        successful_rtt_seconds=rtt_summary,
        elapsed_seconds=elapsed_seconds,
    )


async def measure_independent_connections(
    base_url: str,
    *,
    attempts: int,
    concurrency: int,
    request_timeout: httpx.Timeout,
) -> list[AttemptResult]:
    """Measure requests using one non-reusable HTTP client per TCP connection."""
    semaphore = asyncio.Semaphore(concurrency)

    async def attempt() -> AttemptResult:
        async with semaphore:
            started = perf_counter()
            try:
                async with httpx.AsyncClient(
                    base_url=base_url,
                    headers={"Connection": "close"},
                    limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
                    timeout=request_timeout,
                ) as client:
                    started = perf_counter()
                    response = await client.get("/healthz")
                    response.raise_for_status()
                    rtt_seconds = perf_counter() - started
            except httpx.HTTPError as error:
                return AttemptResult(
                    classify_httpx_error(error), perf_counter() - started, type(error).__name__
                )
            return AttemptResult("success", rtt_seconds)

    return await asyncio.gather(*(attempt() for _ in range(attempts)))


async def run_isolated_condition(
    toxiproxy: ToxiproxyClient,
    profile: ProfileDefinition | None,
    measurement: Callable[[], Awaitable[list[AttemptResult]]],
) -> list[AttemptResult]:
    """Verify isolated state, measure, then verify cleanup even on failure."""
    error: Exception | None = None
    try:
        await toxiproxy.inspect()
        clean = await toxiproxy.reset()
        if clean.toxics:
            raise RuntimeError("Toxiproxy was not clean before measurement")
        if profile is not None:
            active = await toxiproxy.apply(profile)
            if active.toxics != profile.toxics:
                raise RuntimeError(f"Profile {profile.name!r} was not verified")
        return await measurement()
    except Exception as caught:
        error = caught
        raise
    finally:
        try:
            clean = await toxiproxy.reset()
            if clean.toxics:
                raise RuntimeError("Toxiproxy cleanup left active toxics")
        except Exception as cleanup_error:
            if error is not None:
                raise DiagnosticRunError(error, cleanup_error) from error
            raise DiagnosticRunError(cleanup_error) from cleanup_error


def _percentile(sorted_values: list[float], quantile: float) -> float:
    index = max(0, int(len(sorted_values) * quantile + 0.999999) - 1)
    return sorted_values[index]


def _required_timeout(value: float | None) -> float:
    if value is None:
        raise ValueError("Diagnostic timeouts must be finite")
    return value
