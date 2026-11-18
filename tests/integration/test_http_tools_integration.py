import asyncio
import os
from typing import Any, cast

import pytest
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

from predictive_agent.config import Settings
from predictive_agent.llm import LLMClient
from predictive_agent.tools import TOOL_REGISTRY, ExecutionContext, HTTPToolClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("PA_RUN_INTEGRATION") != "1",
        reason="set PA_RUN_INTEGRATION=1 with the Compose stack running",
    ),
]


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize(
    ("name", "arguments", "identity_field", "expected"),
    [
        ("get_order", {"order_id": "ORD-0042"}, "order_id", "ORD-0042"),
        ("get_shipment", {"order_id": "ORD-0042"}, "order_id", "ORD-0042"),
        ("get_customer", {"customer_id": "CUST-073"}, "customer_id", "CUST-073"),
        ("get_policy", {"policy_name": "returns"}, "policy_name", "returns"),
        ("get_inventory", {"sku": "SKU-0007"}, "sku", "SKU-0007"),
    ],
)
async def test_all_tools_use_the_configured_proxied_endpoint(
    name: str, arguments: dict[str, str], identity_field: str, expected: str
) -> None:
    proxy_port = os.getenv("PA_TOXIPROXY_MOCK_APIS_PORT", "18100")
    settings = _settings(tool_api_base_url=f"http://127.0.0.1:{proxy_port}", http_timeout_s=15)
    context = ExecutionContext(
        frozenset(spec.auth_scope for spec in TOOL_REGISTRY.specs()),
        subject="fixture-user",
        tenant_id="fixture-tenant",
    )
    async with HTTPToolClient(settings) as client:
        result = await client.execute(
            name,
            arguments,
            context=context,
            deadline=asyncio.get_running_loop().time() + 15,
        )
    assert result[identity_field] == expected


async def test_registry_schema_mock_llm_and_http_execution_contract() -> None:
    llm_port = os.getenv("PA_MOCK_LLM_PORT", "8090")
    proxy_port = os.getenv("PA_TOXIPROXY_MOCK_APIS_PORT", "18100")
    settings = _settings(
        llm_provider="mock",
        llm_base_url=f"http://127.0.0.1:{llm_port}/v1",
        llm_api_key="mock",
        llm_model="mock-calibrated",
        tool_api_base_url=f"http://127.0.0.1:{proxy_port}",
        http_timeout_s=15,
    )
    tools = cast(list[ChatCompletionToolUnionParam], list(TOOL_REGISTRY.openai_schemas()))
    messages = cast(
        list[ChatCompletionMessageParam],
        [{"role": "user", "content": "Check inventory for SKU-0007"}],
    )
    context = ExecutionContext(frozenset({"inventory:read"}), subject="fixture-user")
    deadline = asyncio.get_running_loop().time() + 15
    async with LLMClient(settings) as llm, HTTPToolClient(settings) as tool_client:
        result = await llm.complete(messages, tools=tools, deadline=deadline)
        call = result.tool_calls[0]
        assert call.executable
        assert call.name == "get_inventory"
        assert call.parsed_arguments is not None
        payload = await tool_client.execute(
            call.name,
            call.parsed_arguments,
            context=context,
            deadline=deadline,
        )
    assert payload["sku"] == "SKU-0007"
