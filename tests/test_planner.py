import json
import time

import httpx
import pytest

from agents.planner_agent import PlannerAgent
from llm.ollama_client import OllamaClient
from models import (
    UAVTelemetry,
    SituationReport,
    HybridMemoryContext,
    RetrievedExperience,
    EpisodicExperience,
    SemanticRule,
)


VALID_LLM_PLAN = {
    "action": "SWITCH_TO_VIO_DEAD_RECKONING",
    "reason": "GPS degraded; switch to VIO",
    "steps": ["Disengage GPS hold", "Engage VIO loop"],
    "expected_outcome": "Stable transit without GPS",
    "confidence": 0.91,
    "risk_level": "LOW",
}


@pytest.fixture
def planner(monkeypatch):
    agent = PlannerAgent()
    monkeypatch.setattr(agent.client, "generate_json", lambda *a, **k: None)
    return agent


def _gps_inputs():
    t = UAVTelemetry(altitude=120.0, gps_status="degraded", gps_accuracy=6.5)
    sit = SituationReport(
        anomaly_detected=True,
        failure_type="GPS_INTERFERENCE",
        severity="HIGH",
        description="GPS degraded during cruise",
    )
    ctx = HybridMemoryContext(top_recommended_action="SWITCH_TO_VIO_DEAD_RECKONING")
    return t, sit, ctx


def test_planner_nominal(planner):
    t = UAVTelemetry()
    sit = SituationReport(anomaly_detected=False)
    ctx = HybridMemoryContext()

    plan = planner.plan_recovery(t, sit, ctx)
    assert plan.action == "CONTINUE_MISSION"
    assert plan.confidence >= 0.95
    assert plan.risk_level == "LOW"
    assert plan.source == "Deterministic-Nominal-Controller"


def test_planner_gps_failure_reasoner(planner):
    t, sit, ctx = _gps_inputs()
    plan = planner.plan_recovery(t, sit, ctx)
    assert plan.action == "SWITCH_TO_VIO_DEAD_RECKONING"
    assert len(plan.steps) >= 2
    assert plan.confidence >= 0.85
    assert plan.source == "offline_reasoner"
    assert plan.plan_latency_ms >= 0.0


def test_planner_low_battery_reasoner(planner):
    t = UAVTelemetry(battery_level=12.0, battery_voltage=13.5)
    sit = SituationReport(
        anomaly_detected=True,
        failure_type="LOW_BATTERY",
        severity="CRITICAL"
    )
    ctx = HybridMemoryContext()
    plan = planner.plan_recovery(t, sit, ctx)
    assert plan.action == "CONTROLLED_EMERGENCY_LAND"
    assert plan.risk_level == "HIGH"
    assert plan.source == "offline_reasoner"


def test_ollama_down_returns_offline_reasoner_quickly(monkeypatch):
    agent = PlannerAgent()
    monkeypatch.setattr(agent.client, "generate_json", lambda *a, **k: None)
    t, sit, ctx = _gps_inputs()
    started = time.perf_counter()
    plan = agent.plan_recovery(t, sit, ctx)
    elapsed = time.perf_counter() - started
    assert plan.source == "offline_reasoner"
    assert plan.action == "SWITCH_TO_VIO_DEAD_RECKONING"
    assert elapsed < 1.0


def test_valid_ollama_json_used_when_confident(monkeypatch):
    agent = PlannerAgent()
    monkeypatch.setattr(agent.client, "generate_json", lambda *a, **k: dict(VALID_LLM_PLAN))
    t, sit, ctx = _gps_inputs()
    plan = agent.plan_recovery(t, sit, ctx)
    assert plan.source == "ollama"
    assert plan.action == "SWITCH_TO_VIO_DEAD_RECKONING"
    assert plan.confidence == pytest.approx(0.91)


def test_low_confidence_ollama_plan_falls_back(monkeypatch):
    agent = PlannerAgent()
    low = dict(VALID_LLM_PLAN)
    low["confidence"] = 0.4
    monkeypatch.setattr(agent.client, "generate_json", lambda *a, **k: low)
    t, sit, ctx = _gps_inputs()
    plan = agent.plan_recovery(t, sit, ctx)
    assert plan.source == "offline_reasoner"


def test_invalid_json_retries_once_then_succeeds(monkeypatch):
    agent = PlannerAgent()
    calls = []

    def fake_generate(prompt, system=None):
        calls.append(prompt)
        if len(calls) == 1:
            return {"foo": "bar"}
        return dict(VALID_LLM_PLAN)

    monkeypatch.setattr(agent.client, "generate_json", fake_generate)
    t, sit, ctx = _gps_inputs()
    plan = agent.plan_recovery(t, sit, ctx)
    assert len(calls) == 2
    assert "PREVIOUS RESPONSE WAS INVALID" in calls[1]
    assert plan.source == "ollama"


def test_invalid_json_twice_falls_back(monkeypatch):
    agent = PlannerAgent()
    monkeypatch.setattr(agent.client, "generate_json", lambda *a, **k: {"foo": "bar"})
    t, sit, ctx = _gps_inputs()
    plan = agent.plan_recovery(t, sit, ctx)
    assert plan.source == "offline_reasoner"


def test_prompt_includes_hydrated_episodes_and_rules():
    agent = PlannerAgent()
    exp = EpisodicExperience(
        episode_id="ep-hydrated",
        mission_id="M9",
        failure="GPS_INTERFERENCE",
        context="from mongo: GPS drop at 120m",
        action="SWITCH_TO_VIO_DEAD_RECKONING",
        outcome="MISSION_SUCCESS",
        success=True,
    )
    ctx = HybridMemoryContext(
        retrieved_experiences=[
            RetrievedExperience(
                experience=exp,
                episode_id="ep-hydrated",
                vector_similarity=1.0,
                graph_relevance=0.8,
                final_score=0.9,
                hydrated_from_mongo=True,
            )
        ],
        semantic_rules=[
            SemanticRule(
                rule_id="RULE_GPS_01",
                trigger="GPS_INTERFERENCE",
                condition="gps_accuracy > 3.0",
                action="SWITCH_TO_VIO_DEAD_RECKONING",
                confidence=0.92,
            )
        ],
        top_recommended_action="SWITCH_TO_VIO_DEAD_RECKONING",
    )
    t, sit, _ = _gps_inputs()
    prompt = agent.build_user_prompt(t, sit, ctx, "Preserve UAV integrity")
    assert "from mongo: GPS drop at 120m" in prompt
    assert "ep-hydrated" in prompt
    assert "hydrated_from_mongo=True" in prompt
    assert "RULE_GPS_01" in prompt
    assert "GPS_INTERFERENCE" in prompt
    assert "ids only" in prompt or "Hydrated Episodes" in prompt


def test_ollama_request_uses_json_format_and_timeouts(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"response": json.dumps(VALID_LLM_PLAN)}

    class FakeClient:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return FakeResp()

    monkeypatch.setattr("llm.ollama_client.httpx.Client", FakeClient)
    client = OllamaClient()
    parsed = client.generate_json("hello")
    assert parsed["action"] == VALID_LLM_PLAN["action"]
    assert captured["json"]["format"] == "json"
    assert captured["json"]["stream"] is False
    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == pytest.approx(0.3)
    assert timeout.read == pytest.approx(5.0)
