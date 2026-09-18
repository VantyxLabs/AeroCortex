import pytest
from fastapi.testclient import TestClient

from api.telemetry_api import app
from config import config
from models import UAVTelemetry


@pytest.fixture
def client():
    with TestClient(app) as c:
        c.headers.update({"X-API-Key": config.api_key})
        yield c


def test_api_root(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"
    assert data["offline_mode"] is True


def test_api_healthz_no_key_required():
    with TestClient(app) as c:
        res = c.get("/healthz")
        assert res.status_code in (200, 503)
        data = res.json()
        assert "dependencies" in data
        assert "mongo" in data["dependencies"]
        assert "neo4j" in data["dependencies"]
        assert "chroma" in data["dependencies"]
        assert "ollama" in data["dependencies"]


def test_api_rejects_missing_key():
    with TestClient(app) as c:
        res = c.get("/status")
        assert res.status_code == 401


def test_api_telemetry_endpoint(client):
    t = UAVTelemetry(mission_id="API_TEST_001")
    res = client.post("/telemetry", json=t.model_dump())
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "action" in data
    assert "is_approved" in data
    assert "persisted" in data
    assert "planner_source" in data


def test_api_status_endpoint(client):
    res = client.get("/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "working_memory" in data
    assert "engine" in data["kg_summary"]


def test_api_memory_endpoint(client):
    res = client.get("/memory")
    assert res.status_code == 200
    data = res.json()
    assert "episodic_experiences_count" in data
    assert "semantic_rules_count" in data
    assert "knowledge_graph" in data


def test_api_missions_endpoint(client):
    res = client.get("/missions", params={"limit": 10, "skip": 0})
    assert res.status_code == 200
    data = res.json()
    assert "history" in data
    assert "total_events" in data


def test_api_simulate_endpoint(client):
    res = client.post("/simulate", json={"scenario": "GPS_INTERFERENCE", "steps": 1})
    assert res.status_code == 200
    data = res.json()
    assert "result" in data
    assert data["result"]["scenario"] == "GPS_INTERFERENCE"


def test_api_reset_endpoint(client):
    res = client.post("/reset")
    assert res.status_code == 200
    assert res.json()["status"] == "SUCCESS"
