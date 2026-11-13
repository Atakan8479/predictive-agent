import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import httpx
import pytest
from pydantic import ValidationError

from predictive_agent.bench.toxiproxy import (
    PROXY_NAME,
    ProfileApplyError,
    ProfileVerificationError,
    ToxiproxyClient,
    load_profile,
    toxiproxy_api_url,
)
from predictive_agent.config import Settings


def _empty_toxics() -> list[dict[str, object]]:
    return []


@dataclass
class FakeToxiproxy:
    toxics: list[dict[str, object]] = field(default_factory=_empty_toxics)
    create_attempts: int = 0
    fail_create_at: int | None = None
    fail_delete: bool = False
    fail_inspect: bool = False
    verification_mismatch: bool = False
    mismatch_returned: bool = False

    async def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.extensions["timeout"] == {
            "connect": 2.0,
            "read": 5.0,
            "write": 5.0,
            "pool": 5.0,
        }
        path = request.url.path
        if request.method == "GET" and path == f"/proxies/{PROXY_NAME}":
            if self.fail_inspect:
                return httpx.Response(503)
            toxics = self.toxics
            if (
                self.verification_mismatch
                and self.create_attempts > 0
                and not self.mismatch_returned
            ):
                toxics = self.toxics[:-1]
                self.mismatch_returned = True
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
            self.create_attempts += 1
            if self.create_attempts == self.fail_create_at:
                return httpx.Response(503)
            toxic = cast(dict[str, object], json.loads(request.content))
            self.toxics.append(toxic)
            return httpx.Response(200, json=toxic)
        if request.method == "DELETE" and path.startswith(f"/proxies/{PROXY_NAME}/toxics/"):
            if self.fail_delete:
                return httpx.Response(503)
            name = path.rsplit("/", maxsplit=1)[-1]
            self.toxics[:] = [toxic for toxic in self.toxics if toxic["name"] != name]
            return httpx.Response(204)
        return httpx.Response(404)


@pytest.fixture
def profiles_dir() -> Path:
    return Path(__file__).parents[2] / "deploy" / "toxiproxy" / "profiles"


@pytest.fixture
async def toxiproxy() -> AsyncIterator[tuple[ToxiproxyClient, FakeToxiproxy]]:
    server = FakeToxiproxy()
    transport = httpx.MockTransport(server.handle)
    async with ToxiproxyClient("http://toxiproxy.test", transport=transport) as client:
        yield client, server


def _settings(env_file: Path | None) -> Settings:
    return Settings(_env_file=env_file)  # pyright: ignore[reportCallIssue]


def test_toxiproxy_api_url_uses_default_without_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PA_TOXIPROXY_API_PORT", raising=False)

    assert toxiproxy_api_url(_settings(None)) == "http://127.0.0.1:8474"


def test_toxiproxy_api_url_reads_custom_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PA_TOXIPROXY_API_PORT", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("PA_TOXIPROXY_API_PORT=18474\n", encoding="utf-8")

    assert toxiproxy_api_url(_settings(env_file)) == "http://127.0.0.1:18474"


def test_process_environment_overrides_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("PA_TOXIPROXY_API_PORT=18474\n", encoding="utf-8")
    monkeypatch.setenv("PA_TOXIPROXY_API_PORT", "28474")

    assert toxiproxy_api_url(_settings(env_file)) == "http://127.0.0.1:28474"


@pytest.mark.parametrize("port", ["0", "65536", "invalid"])
def test_invalid_toxiproxy_api_port_is_rejected(port: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PA_TOXIPROXY_API_PORT", port)

    with pytest.raises(ValidationError, match="toxiproxy_api_port"):
        _settings(None)


async def test_client_uses_resolved_management_endpoint(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("PA_TOXIPROXY_API_PORT=18474\n", encoding="utf-8")
    requested_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "name": PROXY_NAME,
                "listen": "0.0.0.0:18100",
                "upstream": "mock-apis:8100",
                "enabled": True,
                "toxics": [],
            },
        )

    async with ToxiproxyClient(
        toxiproxy_api_url(_settings(env_file)), transport=httpx.MockTransport(handler)
    ) as client:
        await client.inspect()

    assert requested_urls == ["http://127.0.0.1:18474/proxies/mock-apis"]


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


async def test_apply_verifies_complete_profile(
    toxiproxy: tuple[ToxiproxyClient, FakeToxiproxy], profiles_dir: Path
) -> None:
    client, _ = toxiproxy
    profile = load_profile("flaky", profiles_dir)

    state = await client.apply(profile)

    assert state.toxics == profile.toxics


async def test_apply_replaces_active_toxics(
    toxiproxy: tuple[ToxiproxyClient, FakeToxiproxy], profiles_dir: Path
) -> None:
    client, server = toxiproxy
    server.toxics.append(
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
    toxiproxy: tuple[ToxiproxyClient, FakeToxiproxy], profiles_dir: Path
) -> None:
    client, _ = toxiproxy
    await client.apply(load_profile("fast", profiles_dir))

    assert (await client.inspect()).toxics
    assert (await client.reset()).toxics == []


async def test_first_creation_failure_is_cleaned_up(
    toxiproxy: tuple[ToxiproxyClient, FakeToxiproxy], profiles_dir: Path
) -> None:
    client, server = toxiproxy
    server.fail_create_at = 1

    with pytest.raises(ProfileApplyError) as raised:
        await client.apply(load_profile("flaky", profiles_dir))

    assert isinstance(raised.value.apply_error, httpx.HTTPStatusError)
    assert raised.value.cleanup_error is None
    assert server.toxics == []


async def test_later_creation_failure_cleans_up_partial_profile(
    toxiproxy: tuple[ToxiproxyClient, FakeToxiproxy], profiles_dir: Path
) -> None:
    client, server = toxiproxy
    server.fail_create_at = 2

    with pytest.raises(ProfileApplyError) as raised:
        await client.apply(load_profile("flaky", profiles_dir))

    assert isinstance(raised.value.apply_error, httpx.HTTPStatusError)
    assert raised.value.cleanup_error is None
    assert server.toxics == []


async def test_cleanup_failure_preserves_both_errors(
    toxiproxy: tuple[ToxiproxyClient, FakeToxiproxy], profiles_dir: Path
) -> None:
    client, server = toxiproxy
    server.fail_create_at = 2
    server.fail_delete = True

    with pytest.raises(ProfileApplyError, match="cleanup also failed") as raised:
        await client.apply(load_profile("flaky", profiles_dir))

    assert isinstance(raised.value.apply_error, httpx.HTTPStatusError)
    assert isinstance(raised.value.cleanup_error, httpx.HTTPStatusError)
    assert raised.value.__cause__ is raised.value.apply_error
    assert len(server.toxics) == 1


async def test_final_state_mismatch_is_rejected_and_cleaned_up(
    toxiproxy: tuple[ToxiproxyClient, FakeToxiproxy], profiles_dir: Path
) -> None:
    client, server = toxiproxy
    server.verification_mismatch = True

    with pytest.raises(ProfileApplyError) as raised:
        await client.apply(load_profile("slow", profiles_dir))

    assert isinstance(raised.value.apply_error, ProfileVerificationError)
    assert raised.value.cleanup_error is None
    assert server.toxics == []


async def test_inspect_failure_is_propagated(
    toxiproxy: tuple[ToxiproxyClient, FakeToxiproxy],
) -> None:
    client, server = toxiproxy
    server.fail_inspect = True

    with pytest.raises(httpx.HTTPStatusError):
        await client.inspect()


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
