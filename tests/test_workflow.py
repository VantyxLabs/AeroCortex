import pytest
from models import UAVTelemetry
from orchestration.graph import AeroCortexGraph
from simulation.failure_scenarios import FailureScenarioInjector

@pytest.fixture
def graph():
    return AeroCortexGraph()

def test_workflow_nominal_execution(graph):
    t = UAVTelemetry(mission_id="NOMINAL_001")
    state = graph.run(t)
    
    assert "situation" in state
    assert not state["situation"].anomaly_detected
    assert state["execution_status"] == "CONTINUE"
    assert state["final_plan"].action == "CONTINUE_MISSION"

def test_workflow_failure_recovery_closed_loop(graph):
    t = UAVTelemetry(mission_id="LOOP_001", altitude=120.0)
    t = FailureScenarioInjector.apply_scenario(t, "GPS_INTERFERENCE")
    
    state = graph.run(t)
    
    # 1. Situation assessment
    assert state["situation"].anomaly_detected
    assert state["situation"].failure_type == "GPS_INTERFERENCE"
    
    # 2. Hybrid Memory retrieval
    assert state["memory_context"] is not None
    assert len(state["memory_context"].retrieved_experiences) > 0
    
    # 3. Planner reasoning
    assert state["planner_plan"] is not None
    assert state["planner_plan"].action is not None
    
    # 4. Safety validation
    assert state["safety_verdict"] is not None
    assert state["safety_verdict"].approved
    
    # 5. Execution status
    assert state["execution_status"] == "RECOVERED"
    assert "VIO" in state["final_plan"].action or "INERTIAL" in state["final_plan"].action
    
    # 6. Learning consolidation
    assert state["learning_result"] is not None
    assert state["learning_result"].get("status") == "SUCCESS"
    assert state["learning_result"].get("outcome") == "MISSION_SUCCESS"

def test_workflow_unsafe_plan_triggers_fallback(graph):
    # Craft telemetry with critical battery (< 10%)
    t = UAVTelemetry(mission_id="CRIT_001", battery_level=8.0, battery_voltage=13.2)
    t = FailureScenarioInjector.apply_scenario(t, "LOW_BATTERY")
    
    state = graph.run(t)
    
    assert state["situation"].anomaly_detected
    # Safety Agent should force emergency landing fallback
    assert state["final_plan"].action == "CONTROLLED_EMERGENCY_LAND"
    assert state["final_plan"].source in (
        "Deterministic-Safety-Fallback",
        "offline_reasoner",
        "ollama",
    )
