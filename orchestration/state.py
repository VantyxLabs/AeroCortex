from typing import TypedDict, Optional, List, Dict, Any
from models import (
    UAVTelemetry, SituationReport, HybridMemoryContext,
    RecoveryPlan, SafetyVerdict, MissionOutcome
)

class AeroCortexState(TypedDict, total=False):
    """
    Typed state dictionary representing the full data bus for LangGraph execution.
    """
    telemetry: UAVTelemetry
    situation: SituationReport
    memory_context: HybridMemoryContext
    planner_plan: RecoveryPlan
    safety_verdict: SafetyVerdict
    final_plan: RecoveryPlan
    execution_status: str # "CONTINUE", "RECOVERED", "FALLBACK_TRIGGERED"
    mission_outcome: Optional[MissionOutcome]
    learning_result: Dict[str, Any]
    logs: List[str]
