import pytest
import time
from models import UAVTelemetry, SituationReport, RecoveryPlan, SafetyVerdict
from agents.learning_agent import LearningAgent
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from memory.knowledge_graph import KnowledgeGraph

@pytest.fixture
def learning_agent():
    return LearningAgent()

def test_learning_skipped_on_nominal(learning_agent):
    t = UAVTelemetry()
    sit = SituationReport(anomaly_detected=False)
    plan = RecoveryPlan(action="CONTINUE_MISSION")
    verdict = SafetyVerdict(approved=True)
    
    res = learning_agent.process_mission_outcome(t, sit, plan, verdict)
    assert res["status"] == "SKIPPED"

def test_learning_stores_and_reinforces(learning_agent):
    t = UAVTelemetry(mission_id="TEST_LEARN_001", altitude=110.0, wind_speed=8.0)
    sit = SituationReport(
        anomaly_detected=True,
        failure_type="GPS_INTERFERENCE",
        severity="HIGH",
        description="GPS degraded"
    )
    plan = RecoveryPlan(
        action="SWITCH_TO_VIO_DEAD_RECKONING",
        confidence=0.92
    )
    verdict = SafetyVerdict(approved=True)
    
    res = learning_agent.process_mission_outcome(
        telemetry=t,
        situation=sit,
        plan=plan,
        verdict=verdict,
        success=True,
        duration_s=25.0
    )
    
    assert res["status"] == "SUCCESS"
    assert res["episode_id"]
    assert res["action"] == "SWITCH_TO_VIO_DEAD_RECKONING"
    assert res["outcome"] == "MISSION_SUCCESS"
    assert "persisted" in res

    # Verify memory recall immediately after learning
    episodes = learning_agent.episodic_memory.retrieve_similar_experiences(
        "GPS interference degraded",
        top_k=3
    )
    assert any("SWITCH_TO_VIO" in e["experience"].action for e in episodes)
