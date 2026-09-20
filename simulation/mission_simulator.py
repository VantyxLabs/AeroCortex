import json
import logging
from typing import List, Dict, Any, Optional

from models import UAVTelemetry
from simulation.telemetry_generator import TelemetryGenerator
from simulation.failure_scenarios import FailureScenarioInjector
from orchestration.graph import AeroCortexGraph

logger = logging.getLogger("aerocortex.simulator")


def _safe_dump(obj: Any) -> Any:
    """Serialize Pydantic models (and nested structures) to JSON-safe dicts."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            try:
                return json.loads(obj.model_dump_json())
            except Exception as exc:
                logger.warning("model_dump failed: %s", exc)
                return None
    if isinstance(obj, dict):
        return {k: _safe_dump(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_dump(v) for v in obj]
    return obj


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
        self._state_store = None
        try:
            from cloud.factories import sim_state_backend

            if sim_state_backend() == "dynamodb":
                from cloud.dynamo_state import SimulatorStateStore

                self._state_store = SimulatorStateStore()
                self._restore_generator()
        except Exception as exc:
            logger.warning("Simulator state store unavailable: %s", exc)

    def _generator_state(self) -> Dict[str, Any]:
        g = self.generator
        return {
            "mission_id": g.mission_id,
            "step_idx": g.step_idx,
            "lat": g.lat,
            "lon": g.lon,
            "altitude": g.altitude,
            "velocity": g.velocity,
            "heading": g.heading,
            "battery_level": g.battery_level,
            "battery_voltage": g.battery_voltage,
            "waypoint": g.waypoint,
            "mission_state": g.mission_state,
            "wind_speed": g.wind_speed,
            "wind_direction": g.wind_direction,
            "gps_status": g.gps_status,
            "gps_accuracy": g.gps_accuracy,
            "comms_status": g.comms_status,
            "payload_status": g.payload_status,
        }

    def _apply_generator_state(self, state: Dict[str, Any]) -> None:
        g = self.generator
        for key, value in state.items():
            if hasattr(g, key):
                setattr(g, key, value)
        self.mission_id = g.mission_id

    def _restore_generator(self) -> None:
        if self._state_store is None:
            return
        state = self._state_store.load()
        if state:
            self._apply_generator_state(state)

    def _persist_generator(self) -> None:
        if self._state_store is None:
            return
        self._state_store.save(self._generator_state())

    def run_step(self, scenario: str = "NORMAL") -> Dict[str, Any]:
        if self._state_store is not None:
            self._restore_generator()
        t = self.generator.step(dt=1.0)

        if scenario != "NORMAL":
            t = FailureScenarioInjector.apply_scenario(t, scenario)

        result = self.graph.run(t)
        memory_context = _safe_dump(result.get("memory_context"))
        # Prefer live agent context if LangGraph state dropped memory_context.
        if memory_context is None and getattr(self.graph, "memory_agent", None):
            last = getattr(self.graph.memory_agent, "last_context", None)
            memory_context = _safe_dump(last)

        graph_paths = []
        if isinstance(memory_context, dict):
            graph_paths = memory_context.get("graph_paths") or []

        kg_summary = {}
        try:
            kg_summary = self.graph.memory_agent.knowledge_graph.get_summary()
        except Exception:
            kg_summary = {}

        step_record = {
            "step": self.generator.step_idx,
            "scenario": scenario,
            "telemetry": _safe_dump(t),
            "situation": _safe_dump(result.get("situation")),
            "planner_plan": _safe_dump(result.get("planner_plan")),
            "safety_verdict": _safe_dump(result.get("safety_verdict")),
            "final_plan": _safe_dump(result.get("final_plan")),
            "memory_context": memory_context,
            "graph_paths": graph_paths,
            "knowledge_graph": kg_summary,
            "execution_status": result.get("execution_status", "UNKNOWN"),
            "logs": result.get("logs", []),
        }
        self.history.append(step_record)
        self._persist_generator()
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
        if self._state_store is not None:
            try:
                self._state_store.clear()
            except Exception as exc:
                logger.warning("Simulator state clear failed: %s", exc)
            self._persist_generator()
