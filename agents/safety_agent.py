import time
from typing import Tuple, Optional
from models import UAVTelemetry, SituationReport, RecoveryPlan, SafetyVerdict
from safety.constraints import FlightSafetyConstraints
from safety.fallback import DeterministicFallbackExecutor

class SafetyAgent:
    """
    Safety Agent:
    Mandatory gatekeeper between the Planner Agent and UAV actuators.
    Evaluates hard deterministic physical, aerodynamic, and electrical flight envelope
    constraints. No LLM recommendation can ever execute without passing this agent.
    """
    def __init__(self, constraints: Optional[FlightSafetyConstraints] = None):
        self.constraints = constraints or FlightSafetyConstraints()

    def validate_plan(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        plan: RecoveryPlan
    ) -> Tuple[SafetyVerdict, RecoveryPlan]:
        start_t = time.time()
        
        # 1. Run deterministic envelope verification
        verdict = self.constraints.evaluate(telemetry, plan)
        verdict.latency_ms = round((time.time() - start_t) * 1000, 2)

        # 2. If approved, return original plan
        if verdict.approved:
            return verdict, plan

        # 3. If rejected, instantiate safe deterministic fallback plan
        fallback_plan = DeterministicFallbackExecutor.get_fallback_plan(
            telemetry=telemetry,
            situation=situation,
            fallback_action_override=verdict.fallback_action
        )
        return verdict, fallback_plan
