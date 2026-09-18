import pytest
from models import UAVTelemetry, SituationReport, RecoveryPlan
from agents.safety_agent import SafetyAgent
from safety.constraints import FlightSafetyConstraints
from safety.fallback import DeterministicFallbackExecutor

@pytest.fixture
def safety_agent():
    return SafetyAgent()

def test_safety_approved_nominal(safety_agent):
    t = UAVTelemetry()
    sit = SituationReport()
    plan = RecoveryPlan(action="CONTINUE_MISSION", confidence=0.95)
    
    verdict, final_plan = safety_agent.validate_plan(t, sit, plan)
    assert verdict.approved
    assert final_plan.action == "CONTINUE_MISSION"

def test_safety_reject_low_confidence(safety_agent):
    t = UAVTelemetry()
    sit = SituationReport(anomaly_detected=True, failure_type="GPS_INTERFERENCE")
    # Low confidence (< 0.70)
    plan = RecoveryPlan(action="EXPERIMENTAL_MANEUVER", confidence=0.45)
    
    verdict, final_plan = safety_agent.validate_plan(t, sit, plan)
    assert not verdict.approved
    assert "Planner confidence" in verdict.reason
    assert final_plan.source == "Deterministic-Safety-Fallback"

def test_safety_reject_continue_on_low_battery(safety_agent):
    t = UAVTelemetry(battery_level=12.0)
    sit = SituationReport(anomaly_detected=True, failure_type="LOW_BATTERY")
    plan = RecoveryPlan(action="CONTINUE_MISSION", confidence=0.95)
    
    verdict, final_plan = safety_agent.validate_plan(t, sit, plan)
    assert not verdict.approved
    assert final_plan.action in ("CONTROLLED_EMERGENCY_LAND", "POWER_CONSERVATIVE_RTH")

def test_safety_reject_gps_action_on_lost_gps(safety_agent):
    t = UAVTelemetry(gps_status="lost", gps_accuracy=99.9)
    sit = SituationReport(anomaly_detected=True, failure_type="GPS_LOSS")
    plan = RecoveryPlan(action="CONTINUE_MISSION", confidence=0.90)
    
    verdict, final_plan = safety_agent.validate_plan(t, sit, plan)
    assert not verdict.approved
    assert final_plan.action in ("SWITCH_TO_VIO_DEAD_RECKONING", "INERTIAL_DEAD_RECKONING_SAFE_RTH")

def test_fallback_executor_wind():
    t = UAVTelemetry(wind_speed=18.0)
    sit = SituationReport(anomaly_detected=True, failure_type="STRONG_WIND")
    fallback = DeterministicFallbackExecutor.get_fallback_plan(t, sit)
    assert fallback.action == "REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT"
    assert fallback.confidence >= 0.95
