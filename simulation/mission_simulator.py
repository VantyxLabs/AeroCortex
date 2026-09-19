import time
from typing import List, Dict, Any, Optional
from models import UAVTelemetry
from simulation.telemetry_generator import TelemetryGenerator
from simulation.failure_scenarios import FailureScenarioInjector
from orchestration.graph import AeroCortexGraph

class MissionSimulator:
    """
    Coordinates simulated UAV missions.
    Streams telemetry, injects failure scenarios, and evaluates cognitive pipeline responses.
    """
    def __init__(self, mission_id: str = "SIM_MISSION_001", graph: Optional[AeroCortexGraph] = None):
        self.mission_id = mission_id
        self.generator = TelemetryGenerator(mission_id=mission_id)
        self.graph = graph or AeroCortexGraph()
        self.history: List[Dict[str, Any]] = []

    def run_step(self, scenario: str = "NORMAL") -> Dict[str, Any]:
        # Generate base telemetry
        t = self.generator.step(dt=1.0)
        
        # Apply scenario if non-normal
        if scenario != "NORMAL":
            t = FailureScenarioInjector.apply_scenario(t, scenario)
            
        # Process through AeroCortex LangGraph pipeline
        result = self.graph.run(t)
        
        step_record = {
            "step": self.generator.step_idx,
            "scenario": scenario,
            "telemetry": t.model_dump(),
            "situation": result.get("situation").model_dump() if result.get("situation") else None,
            "planner_plan": result.get("planner_plan").model_dump() if result.get("planner_plan") else None,
            "safety_verdict": result.get("safety_verdict").model_dump() if result.get("safety_verdict") else None,
            "final_plan": result.get("final_plan").model_dump() if result.get("final_plan") else None,
            "memory_context": result.get("memory_context").model_dump() if result.get("memory_context") else None,
            "execution_status": result.get("execution_status", "UNKNOWN"),
            "logs": result.get("logs", []),
        }
        self.history.append(step_record)
        return step_record

    def run_full_mission(
        self,
        total_steps: int = 15,
        inject_step: int = 5,
        scenario: str = "GPS_INTERFERENCE"
    ) -> List[Dict[str, Any]]:
        results = []
        for s in range(1, total_steps + 1):
            curr_scenario = scenario if s >= inject_step else "NORMAL"
            rec = self.run_step(scenario=curr_scenario)
            results.append(rec)
        return results

    def reset(self, new_mission_id: Optional[str] = None) -> None:
        if new_mission_id:
            self.mission_id = new_mission_id
        self.generator = TelemetryGenerator(mission_id=self.mission_id)
        self.history.clear()
