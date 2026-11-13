"""Async management client for the local Toxiproxy profiles."""

import argparse
import asyncio
from pathlib import Path
from types import TracebackType
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from predictive_agent.config import Settings

PROXY_NAME = "mock-apis"
PROFILES_DIR = Path(__file__).parents[3] / "deploy" / "toxiproxy" / "profiles"


class ToxicDefinition(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    toxic_type: str = Field(alias="type")
    stream: Literal["downstream"]
    toxicity: float = Field(ge=0.0, le=1.0)
    attributes: dict[str, int | float]


class ProfileDefinition(BaseModel):
    name: str
    toxics: list[ToxicDefinition]


class ProxyState(BaseModel):
    name: str
    listen: str
    upstream: str
    enabled: bool
    toxics: list[ToxicDefinition]


class ProfileVerificationError(RuntimeError):
    """The active proxy state does not match the requested safe state."""


class ProfileApplyError(RuntimeError):
    """Profile application failed and cleanup may also have failed."""

    def __init__(
        self,
        profile_name: str,
        apply_error: Exception,
        cleanup_error: Exception | None = None,
    ) -> None:
        message = f"Failed to apply profile {profile_name!r}: {apply_error}"
        if cleanup_error is not None:
            message += f"; cleanup also failed: {cleanup_error}"
        super().__init__(message)
        self.apply_error = apply_error
        self.cleanup_error = cleanup_error


def load_profile(name: str, profiles_dir: Path = PROFILES_DIR) -> ProfileDefinition:
    if Path(name).name != name:
        raise ValueError(f"Invalid profile name: {name}")
    path = profiles_dir / f"{name}.json"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in profiles_dir.glob("*.json")))
        raise ValueError(f"Unknown profile {name!r}; choose from: {available}")
    profile = ProfileDefinition.model_validate_json(path.read_text(encoding="utf-8"))
    if profile.name != name:
        raise ValueError(f"Profile name {profile.name!r} does not match {name!r}")
    return profile


class ToxiproxyClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=timeout or httpx.Timeout(5.0, connect=2.0),
        )

    async def __aenter__(self) -> "ToxiproxyClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    async def inspect(self) -> ProxyState:
        response = await self._client.get(f"/proxies/{PROXY_NAME}")
        response.raise_for_status()
        return ProxyState.model_validate(response.json())

    async def reset(self) -> ProxyState:
        proxy = await self.inspect()
        for toxic in proxy.toxics:
            response = await self._client.delete(f"/proxies/{PROXY_NAME}/toxics/{toxic.name}")
            response.raise_for_status()
        state = await self.inspect()
        if state.toxics:
            raise ProfileVerificationError(
                f"Reset left active toxics: {[toxic.name for toxic in state.toxics]}"
            )
        return state

    async def apply(self, profile: ProfileDefinition) -> ProxyState:
        await self.reset()
        try:
            for toxic in profile.toxics:
                response = await self._client.post(
                    f"/proxies/{PROXY_NAME}/toxics",
                    json=toxic.model_dump(mode="json", by_alias=True),
                )
                response.raise_for_status()
            state = await self.inspect()
            if state.toxics != profile.toxics:
                raise ProfileVerificationError(
                    f"Active toxics do not match profile {profile.name!r}"
                )
            return state
        except Exception as apply_error:
            cleanup_error: Exception | None = None
            try:
                await self.reset()
            except Exception as error:
                cleanup_error = error
            raise ProfileApplyError(profile.name, apply_error, cleanup_error) from apply_error


def toxiproxy_api_url(settings: Settings | None = None) -> str:
    resolved = settings or Settings()
    return f"http://127.0.0.1:{resolved.toxiproxy_api_port}"


async def _run(action: str, profile_name: str | None) -> None:
    async with ToxiproxyClient(toxiproxy_api_url()) as client:
        if action == "apply":
            if profile_name is None:
                raise ValueError("apply requires a profile name")
            state = await client.apply(load_profile(profile_name))
        elif action == "reset":
            state = await client.reset()
        else:
            state = await client.inspect()
    print(state.model_dump_json(indent=2, by_alias=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage local Toxiproxy fault profiles")
    parser.add_argument("action", choices=("apply", "reset", "inspect"))
    parser.add_argument("profile", nargs="?")
    args = parser.parse_args()
    if args.action == "apply" and args.profile is None:
        parser.error("apply requires a profile name")
    try:
        asyncio.run(_run(args.action, args.profile))
    except (
        httpx.HTTPError,
        ProfileApplyError,
        ProfileVerificationError,
        ValidationError,
        ValueError,
    ) as error:
        parser.exit(1, f"error: {error}\n")
