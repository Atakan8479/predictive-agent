import os

import httpx
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("PA_RUN_INTEGRATION") != "1",
    reason="set PA_RUN_INTEGRATION=1 with the Compose stack running",
)
def test_mock_apis_service_is_healthy() -> None:
    port = os.getenv("PA_MOCK_APIS_PORT", "8080")
    timeout = httpx.Timeout(5.0, connect=2.0)
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=timeout) as client:
        health = client.get("/healthz")
        order = client.get("/oms/orders/ORD-0001")

    assert health.json() == {"status": "ok"}
    assert order.status_code == 200
    assert order.json()["order_id"] == "ORD-0001"
    assert order.headers["X-Mock-Served-At"]
