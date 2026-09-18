import pytest
from models import UAVTelemetry
from agents.situation_agent import SituationAgent
from simulation.failure_scenarios import FailureScenarioInjector

@pytest.fixture
def situation_agent():
    return SituationAgent()

def test_nominal_telemetry(situation_agent):
    t = UAVTelemetry()
    report = situation_agent.assess_situation(t)
    assert not report.anomaly_detected
    assert report.failure_type == "NONE"
    assert report.severity == "NOMINAL"

def test_gps_interference_detection(situation_agent):
    t = UAVTelemetry()
    t = FailureScenarioInjector.apply_scenario(t, "GPS_INTERFERENCE")
    report = situation_agent.assess_situation(t)
    assert report.anomaly_detected
    assert report.failure_type == "GPS_INTERFERENCE"
    assert report.severity in ("MEDIUM", "HIGH")
    assert report.confidence >= 0.85

def test_gps_loss_detection(situation_agent):
    t = UAVTelemetry()
    t = FailureScenarioInjector.apply_scenario(t, "GPS_LOSS")
    report = situation_agent.assess_situation(t)
    assert report.anomaly_detected
    assert report.failure_type == "GPS_LOSS"
    assert report.severity == "HIGH"

def test_low_battery_detection(situation_agent):
    t = UAVTelemetry()
    t = FailureScenarioInjector.apply_scenario(t, "LOW_BATTERY")
    report = situation_agent.assess_situation(t)
    assert report.anomaly_detected
    assert report.failure_type == "LOW_BATTERY"
    assert report.severity in ("HIGH", "CRITICAL")

def test_strong_wind_detection(situation_agent):
    t = UAVTelemetry()
    t = FailureScenarioInjector.apply_scenario(t, "STRONG_WIND")
    report = situation_agent.assess_situation(t)
    assert report.anomaly_detected
    assert report.failure_type == "STRONG_WIND"

def test_sensor_anomaly_detection(situation_agent):
    t = UAVTelemetry()
    t = FailureScenarioInjector.apply_scenario(t, "SENSOR_ANOMALY")
    report = situation_agent.assess_situation(t)
    assert report.anomaly_detected
    assert report.failure_type == "SENSOR_ANOMALY"

def test_combined_failure_detection(situation_agent):
    t = UAVTelemetry()
    t = FailureScenarioInjector.apply_scenario(t, "COMBINED_FAILURE")
    report = situation_agent.assess_situation(t)
    assert report.anomaly_detected
    assert report.failure_type == "COMBINED_FAILURE"
    assert report.severity == "CRITICAL"
