import os

import httpx
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("PA_RUN_INTEGRATION") != "1",
    reason="set PA_RUN_INTEGRATION=1 with the Compose stack running",
)
def test_mock_apis_service_is_healthy() -> None:
    health = httpx.get("http://127.0.0.1:8080/healthz")
    order = httpx.get("http://127.0.0.1:8080/oms/orders/ORD-0001")

    assert health.json() == {"status": "ok"}
    assert order.status_code == 200
    assert order.json()["order_id"] == "ORD-0001"
    assert order.headers["X-Mock-Served-At"]
