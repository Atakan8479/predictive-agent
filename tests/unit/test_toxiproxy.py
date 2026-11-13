import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import httpx
import pytest

from predictive_agent.bench.toxiproxy import (
    PROXY_NAME,
    ToxiproxyClient,
    load_profile,
)


@pytest.fixture
def profiles_dir() -> Path:
    return Path(__file__).parents[2] / "deploy" / "toxiproxy" / "profiles"


@pytest.fixture
async def toxiproxy() -> AsyncIterator[tuple[ToxiproxyClient, list[dict[str, object]]]]:
    toxics: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.extensions["timeout"] == {
            "connect": 2.0,
            "read": 5.0,
            "write": 5.0,
            "pool": 5.0,
        }
        path = request.url.path
        if request.method == "GET" and path == f"/proxies/{PROXY_NAME}":
            return httpx.Response(
                200,
                json={
                    "name": PROXY_NAME,
                    "listen": "0.0.0.0:18100",
                    "upstream": "mock-apis:8100",
                    "enabled": True,
                    "toxics": toxics,
                },
            )
        if request.method == "POST" and path == f"/proxies/{PROXY_NAME}/toxics":
            toxic = cast(dict[str, object], json.loads(request.content))
            toxics.append(toxic)
            return httpx.Response(200, json=toxic)
        if request.method == "DELETE" and path.startswith(f"/proxies/{PROXY_NAME}/toxics/"):
            name = path.rsplit("/", maxsplit=1)[-1]
            toxics[:] = [toxic for toxic in toxics if toxic["name"] != name]
            return httpx.Response(204)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with ToxiproxyClient("http://toxiproxy.test", transport=transport) as client:
        yield client, toxics


def test_profiles_are_downstream_and_have_expected_latency(profiles_dir: Path) -> None:
    expected = {
        "fast": (50, 10),
        "typical": (200, 50),
        "slow": (800, 200),
        "flaky": (200, 50),
    }

    for name, (latency, jitter) in expected.items():
        profile = load_profile(name, profiles_dir)
        assert all(toxic.stream == "downstream" for toxic in profile.toxics)
        assert profile.toxics[0].attributes == {"latency": latency, "jitter": jitter}


def test_flaky_profile_defines_failures_and_timeouts(profiles_dir: Path) -> None:
    profile = load_profile("flaky", profiles_dir)

    assert [(toxic.toxic_type, toxic.toxicity) for toxic in profile.toxics[1:]] == [
        ("reset_peer", 0.05),
        ("timeout", 0.02),
    ]


async def test_apply_replaces_active_toxics(
    toxiproxy: tuple[ToxiproxyClient, list[dict[str, object]]], profiles_dir: Path
) -> None:
    client, toxics = toxiproxy
    toxics.append(
        {
            "name": "old",
            "type": "latency",
            "stream": "downstream",
            "toxicity": 1.0,
            "attributes": {"latency": 1},
        }
    )

    state = await client.apply(load_profile("slow", profiles_dir))

    assert [toxic.name for toxic in state.toxics] == ["latency"]
    assert state.toxics[0].attributes == {"latency": 800, "jitter": 200}


async def test_reset_and_inspect(
    toxiproxy: tuple[ToxiproxyClient, list[dict[str, object]]], profiles_dir: Path
) -> None:
    client, _ = toxiproxy
    await client.apply(load_profile("fast", profiles_dir))

    assert (await client.inspect()).toxics
    assert (await client.reset()).toxics == []


def test_profile_validation_rejects_upstream(tmp_path: Path) -> None:
    profile = tmp_path / "invalid.json"
    profile.write_text(
        json.dumps(
            {
                "name": "invalid",
                "toxics": [
                    {
                        "name": "latency",
                        "type": "latency",
                        "stream": "upstream",
                        "toxicity": 1.0,
                        "attributes": {"latency": 10},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="downstream"):
        load_profile("invalid", tmp_path)
