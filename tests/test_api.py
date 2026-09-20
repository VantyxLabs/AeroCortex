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
        assert "pinecone" in data["dependencies"]
        assert "chroma" in data["dependencies"]
        assert "groq" in data["dependencies"]
        assert "ollama" in data["dependencies"]
        assert "version" in data
        if res.status_code == 200:
            assert data["status"] in ("ok", "degraded")
        else:
            assert data["status"] == "unavailable"


def test_api_docs_no_key_required():
    with TestClient(app) as c:
        res = c.get("/docs")
        assert res.status_code == 200


def test_healthz_degraded_when_mongo_down(monkeypatch):
    import api.telemetry_api as telemetry_api

    async def mongo_down():
        return False

    telemetry_api._HEALTH_CACHE["payload"] = None
    telemetry_api._HEALTH_CACHE["ts"] = 0.0
    with TestClient(app) as c:
        monkeypatch.setattr(telemetry_api.document_store, "ping", mongo_down)
        monkeypatch.setattr(
            telemetry_api.graph.memory_agent.episodic_memory.vector_store, "ping", lambda: True
        )
        res = c.get("/healthz")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "degraded"
        assert data["dependencies"]["mongo"] == "down"


def test_healthz_503_when_mongo_and_chroma_down(monkeypatch):
    import api.telemetry_api as telemetry_api

    async def mongo_down():
        return False

    telemetry_api._HEALTH_CACHE["payload"] = None
    telemetry_api._HEALTH_CACHE["ts"] = 0.0
    vs = telemetry_api.graph.memory_agent.episodic_memory.vector_store
    with TestClient(app) as c:
        monkeypatch.setattr(telemetry_api.document_store, "ping", mongo_down)
        monkeypatch.setattr(
            vs,
            "health",
            lambda: {"engine": "chroma", "pinecone": "down", "chroma": "down", "ok": False},
        )
        res = c.get("/healthz")
        assert res.status_code == 503
        data = res.json()
        assert data["status"] == "unavailable"
        assert data["dependencies"]["mongo"] == "down"
        assert data["dependencies"]["chroma"] == "down"


def test_compose_api_waits_for_healthy_dependencies():
    from pathlib import Path

    import yaml

    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    assert "version" not in compose
    for name in ("mongo", "neo4j", "ollama", "api", "dashboard"):
        assert name in compose["services"]
        assert "healthcheck" in compose["services"][name] or name == "dashboard"
    deps = compose["services"]["api"]["depends_on"]
    assert deps["mongo"]["condition"] == "service_healthy"
    # neo4j is optional (Aura via .env / local-neo4j profile)
    assert "neo4j" not in deps or deps["neo4j"]["condition"] == "service_healthy"
    assert deps["ollama"]["condition"] == "service_started"
    assert compose["services"]["api"].get("env_file") in (".env", [".env"])


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
    assert data["engine"] in ("Neo4j", "NetworkX Embedded")


def test_api_memory_endpoint(client):
    res = client.get("/memory")
    assert res.status_code == 200
    data = res.json()
    assert "episodic_experiences_count" in data
    assert "episodic_experiences_count" in data
    assert "vector_engine" in data
    assert data["vector_engine"] in ("pinecone", "chroma")
    assert "semantic_rules_count" in data
    assert "knowledge_graph" in data
    assert "last_retrieval" in data
    assert "candidates" in data["last_retrieval"]


def test_api_missions_endpoint(client):
    res = client.get("/missions", params={"limit": 10, "skip": 0})
    assert res.status_code == 200
    data = res.json()
    assert "history" in data
    assert "total_events" in data
    assert data["limit"] == 10
    assert data["skip"] == 0
    assert "persisted" in data
    assert isinstance(data["history"], list)


def test_api_missions_when_mongo_down(monkeypatch):
    class DownStore:
        available = False

        async def list_missions(self, limit=50, skip=0):
            return []

        async def count_missions(self):
            return 0

    import api.telemetry_api as telemetry_api

    monkeypatch.setattr(telemetry_api, "document_store", DownStore())
    with TestClient(app) as c:
        c.headers.update({"X-API-Key": config.api_key})
        res = c.get("/missions", params={"limit": 5, "skip": 0})
        assert res.status_code == 200
        data = res.json()
        assert data["history"] == []
        assert data["total_events"] == 0
        assert data["persisted"] is False


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
