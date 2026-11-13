from collections.abc import AsyncIterator

import httpx
import pytest
from tests.unit.test_toxiproxy import FakeToxiproxy

from predictive_agent.bench.toxiproxy import ProfileApplyError, ProfileDefinition, ToxiproxyClient
from predictive_agent.bench.toxiproxy_diagnostics import (
    AttemptResult,
    DiagnosticRunError,
    classify_httpx_error,
    run_isolated_condition,
    summarize_attempts,
)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectTimeout("connect"), "connect_timeout"),
        (httpx.ReadTimeout("read"), "read_timeout"),
        (httpx.WriteTimeout("write"), "write_timeout"),
        (httpx.PoolTimeout("pool"), "pool_timeout"),
        (httpx.ConnectError("connect"), "connect_error"),
        (httpx.ReadError("read"), "read_error"),
        (httpx.RemoteProtocolError("protocol"), "remote_protocol_error"),
        (httpx.TransportError("other"), "transport_error"),
    ],
)
def test_classify_httpx_error(error: httpx.HTTPError, expected: str) -> None:
    assert classify_httpx_error(error) == expected


def test_classify_http_status_error() -> None:
    request = httpx.Request("GET", "http://example.test")
    response = httpx.Response(503, request=request)
    error = httpx.HTTPStatusError("status", request=request, response=response)
    assert classify_httpx_error(error) == "http_status_error"


def test_summary_counts_outcomes_and_rtts() -> None:
    summary = summarize_attempts(
        "baseline",
        None,
        [
            AttemptResult("success", 0.1),
            AttemptResult("read_timeout", 1.0, "ReadTimeout"),
            AttemptResult("success", 0.3),
        ],
        attempts=3,
        concurrency=1,
        timeout=httpx.Timeout(2.0),
        elapsed_seconds=1.4,
    )

    assert summary.outcomes == {"read_timeout": 1, "success": 2}
    assert summary.successful_rtt_seconds == {"median": 0.2, "p95": 0.3, "max": 0.3}


def test_summary_handles_all_failures() -> None:
    summary = summarize_attempts(
        "flaky",
        None,
        [AttemptResult("connect_error", 0.1, "ConnectError")],
        attempts=1,
        concurrency=1,
        timeout=httpx.Timeout(2.0),
        elapsed_seconds=0.1,
    )

    assert summary.successful_rtt_seconds is None


def test_summary_handles_empty_sample() -> None:
    summary = summarize_attempts(
        "baseline",
        None,
        [],
        attempts=0,
        concurrency=1,
        timeout=httpx.Timeout(2.0),
        elapsed_seconds=0.0,
    )

    assert summary.outcomes == {}
    assert summary.successful_rtt_seconds is None


@pytest.fixture
async def diagnostic_client() -> AsyncIterator[tuple[ToxiproxyClient, FakeToxiproxy]]:
    server = FakeToxiproxy()
    async with ToxiproxyClient(
        "http://toxiproxy.test", transport=httpx.MockTransport(server.handle)
    ) as client:
        yield client, server


def _profile() -> ProfileDefinition:
    return ProfileDefinition.model_validate(
        {
            "name": "typical",
            "toxics": [
                {
                    "name": "latency",
                    "type": "latency",
                    "stream": "downstream",
                    "toxicity": 1.0,
                    "attributes": {"latency": 200, "jitter": 50},
                }
            ],
        }
    )


async def test_condition_verifies_profile_and_cleans_up(
    diagnostic_client: tuple[ToxiproxyClient, FakeToxiproxy],
) -> None:
    client, server = diagnostic_client

    results = await run_isolated_condition(
        client, _profile(), lambda: _measurement([AttemptResult("success", 0.2)])
    )

    assert results == [AttemptResult("success", 0.2)]
    assert server.toxics == []


async def test_condition_reports_cleanup_failure_after_success(
    diagnostic_client: tuple[ToxiproxyClient, FakeToxiproxy],
) -> None:
    client, server = diagnostic_client

    async def measurement() -> list[AttemptResult]:
        assert server.toxics
        server.fail_delete = True
        return [AttemptResult("success", 0.2)]

    with pytest.raises(DiagnosticRunError, match="Diagnostic failed") as raised:
        await run_isolated_condition(client, _profile(), measurement)

    assert raised.value.cleanup_error is None
    assert isinstance(raised.value.error, httpx.HTTPStatusError)


async def test_condition_cleans_up_after_measurement_failure(
    diagnostic_client: tuple[ToxiproxyClient, FakeToxiproxy],
) -> None:
    client, server = diagnostic_client

    with pytest.raises(RuntimeError, match="measurement failed"):
        await run_isolated_condition(client, _profile(), _failed_measurement)

    assert server.toxics == []


async def test_condition_does_not_measure_unverified_profile(
    diagnostic_client: tuple[ToxiproxyClient, FakeToxiproxy],
) -> None:
    client, server = diagnostic_client
    server.verification_mismatch = True
    measured = False

    async def measurement() -> list[AttemptResult]:
        nonlocal measured
        measured = True
        return []

    with pytest.raises(ProfileApplyError):
        await run_isolated_condition(client, _profile(), measurement)

    assert measured is False
    assert server.toxics == []


async def test_condition_reports_measurement_and_cleanup_failures(
    diagnostic_client: tuple[ToxiproxyClient, FakeToxiproxy],
) -> None:
    client, server = diagnostic_client
    server.fail_delete = True

    with pytest.raises(DiagnosticRunError, match="cleanup also failed") as raised:
        await run_isolated_condition(client, _profile(), _failed_measurement)

    assert isinstance(raised.value.error, RuntimeError)
    assert isinstance(raised.value.cleanup_error, httpx.HTTPStatusError)


async def _measurement(results: list[AttemptResult]) -> list[AttemptResult]:
    return results


async def _failed_measurement() -> list[AttemptResult]:
    raise RuntimeError("measurement failed")
