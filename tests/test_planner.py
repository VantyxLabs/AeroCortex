import pytest
from models import UAVTelemetry, SituationReport, HybridMemoryContext
from agents.planner_agent import PlannerAgent
from llm.ollama_client import OllamaClient

@pytest.fixture
def planner():
    return PlannerAgent()

def test_planner_nominal(planner):
    t = UAVTelemetry()
    sit = SituationReport(anomaly_detected=False)
    ctx = HybridMemoryContext()
    
    plan = planner.plan_recovery(t, sit, ctx)
    assert plan.action == "CONTINUE_MISSION"
    assert plan.confidence >= 0.95
    assert plan.risk_level == "LOW"

def test_planner_gps_failure_reasoner(planner):
    t = UAVTelemetry(altitude=120.0, gps_status="degraded", gps_accuracy=6.5)
    sit = SituationReport(
        anomaly_detected=True,
        failure_type="GPS_INTERFERENCE",
        severity="HIGH"
    )
    ctx = HybridMemoryContext(top_recommended_action="SWITCH_TO_VIO_DEAD_RECKONING")
    
    plan = planner.plan_recovery(t, sit, ctx)
    assert plan.action == "SWITCH_TO_VIO_DEAD_RECKONING"
    assert len(plan.steps) >= 2
    assert plan.confidence >= 0.85

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
