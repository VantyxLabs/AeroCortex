from typing import Optional
from models import UAVTelemetry, SituationReport, HybridMemoryContext, RecoveryPlan
from llm.ollama_client import OllamaClient

class PlannerAgent:
    """
    Planner Agent:
    Reasons over real-time UAV telemetry, detected failure severity,
    and retrieved hybrid memory context (Episodic + Semantic + KG)
    using local Gemma 3 via Ollama.
    
    Adheres strictly to the safety mandate: The LLM NEVER directly actuates
    UAV hardware or controls; it only produces a recommended structured recovery plan.
    """
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client or OllamaClient()

    def plan_recovery(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        memory_context: HybridMemoryContext,
        mission_objective: str = "Preserve UAV integrity and execute safe recovery"
    ) -> RecoveryPlan:
        return self.client.generate_plan(
            telemetry=telemetry,
            situation=situation,
            memory_context=memory_context,
            mission_objective=mission_objective
        )
