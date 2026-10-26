import pytest
from fastapi.testclient import TestClient

from scrm.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan (builds the planning state)
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_skus_and_forecast(client):
    skus = client.get("/api/skus").json()
    assert len(skus) == 24
    r = client.get(f"/api/skus/{skus[0]['sku_id']}/forecast?history_weeks=26")
    assert r.status_code == 200
    body = r.json()
    assert len(body["history"]) == 26
    assert len(body["forecast"]) == 12
    for p in body["forecast"]:
        assert p["p10"] <= p["qty"] <= p["p90"]


def test_unknown_sku_is_404(client):
    assert client.get("/api/skus/SKU-999/forecast").status_code == 404


def test_stockouts_sorted_and_filterable(client):
    rows = client.get("/api/risk/stockouts").json()
    probs = [r["stockout_probability"] for r in rows]
    assert probs == sorted(probs, reverse=True)
    high = client.get("/api/risk/stockouts?level=high").json()
    assert all(r["risk_level"] == "high" for r in high)
    assert client.get("/api/risk/stockouts?level=extreme").status_code == 422


def test_suppliers_and_summary(client):
    sup = client.get("/api/risk/suppliers").json()
    assert len(sup) == 4
    s = client.get("/api/summary").json()
    assert s["high_risk_skus"] >= 0 and "SKUs" in s["text"]
