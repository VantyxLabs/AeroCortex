import pytest
from models import UAVTelemetry, SituationReport, RecoveryPlan, SafetyVerdict

def test_telemetry_defaults():
    t = UAVTelemetry()
    assert t.mission_id == "MISSION_001"
    assert t.altitude == 120.0
    assert t.battery_level == 72.0
    assert t.battery_voltage == 15.4
    assert t.gps_status == "healthy"
    assert len(t.imu_acceleration) == 3
    assert len(t.imu_gyroscope) == 3

def test_telemetry_custom_values():
    t = UAVTelemetry(
        mission_id="TEST_999",
        altitude=45.0,
        battery_level=18.5,
        gps_status="degraded",
        gps_accuracy=5.2
    )
    assert t.mission_id == "TEST_999"
    assert t.altitude == 45.0
    assert t.battery_level == 18.5
    assert t.gps_status == "degraded"
    assert t.gps_accuracy == 5.2

def test_recovery_plan_serialization():
    p = RecoveryPlan(
        action="SWITCH_TO_VIO_DEAD_RECKONING",
        reason="GPS degraded",
        steps=["Step 1", "Step 2"],
        confidence=0.91,
        risk_level="LOW"
    )
    data = p.model_dump()
    assert data["action"] == "SWITCH_TO_VIO_DEAD_RECKONING"
    assert len(data["steps"]) == 2
    assert data["confidence"] == 0.91
