import threading
from typing import List, Dict, Any, Optional
from collections import deque
from models import UAVTelemetry, SituationReport, RecoveryPlan, SafetyVerdict

class WorkingMemory:
    """
    Working Memory: Fast in-memory state store for the current active mission.
    Maintains active telemetry, detected anomalies, mission phase, recovery state,
    and a sliding window of recent telemetry for trend analysis.
    """
    def __init__(self, window_size: int = 100):
        self.lock = threading.Lock()
        self.window_size = window_size
        self.telemetry_history: deque = deque(maxlen=window_size)
        
        self.latest_telemetry: Optional[UAVTelemetry] = None
        self.active_anomaly: Optional[SituationReport] = None
        self.mission_phase: str = "CRUISE"
        self.current_waypoint: int = 1
        self.battery_status: Dict[str, float] = {"level": 100.0, "voltage": 16.8}
        self.communication_state: str = "connected"
        self.gps_state: Dict[str, Any] = {"status": "healthy", "accuracy": 1.0}
        self.current_recovery_plan: Optional[RecoveryPlan] = None
        self.safety_status: Optional[SafetyVerdict] = None

    def update_telemetry(self, telemetry: UAVTelemetry) -> None:
        with self.lock:
            self.latest_telemetry = telemetry
            self.telemetry_history.append(telemetry)
            self.mission_phase = telemetry.mission_state
            self.current_waypoint = telemetry.waypoint
            self.battery_status = {
                "level": telemetry.battery_level,
                "voltage": telemetry.battery_voltage
            }
            self.communication_state = telemetry.communication_status
            self.gps_state = {
                "status": telemetry.gps_status,
                "accuracy": telemetry.gps_accuracy
            }

    def set_anomaly(self, report: SituationReport) -> None:
        with self.lock:
            self.active_anomaly = report

    def set_recovery_plan(self, plan: RecoveryPlan) -> None:
        with self.lock:
            self.current_recovery_plan = plan

    def set_safety_status(self, verdict: SafetyVerdict) -> None:
        with self.lock:
            self.safety_status = verdict

    def get_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "latest_telemetry": self.latest_telemetry.model_dump() if self.latest_telemetry else None,
                "active_anomaly": self.active_anomaly.model_dump() if self.active_anomaly else None,
                "mission_phase": self.mission_phase,
                "current_waypoint": self.current_waypoint,
                "battery_status": self.battery_status,
                "communication_state": self.communication_state,
                "gps_state": self.gps_state,
                "current_recovery_plan": self.current_recovery_plan.model_dump() if self.current_recovery_plan else None,
                "safety_status": self.safety_status.model_dump() if self.safety_status else None,
                "history_length": len(self.telemetry_history)
            }

    def clear(self) -> None:
        with self.lock:
            self.telemetry_history.clear()
            self.latest_telemetry = None
            self.active_anomaly = None
            self.mission_phase = "CRUISE"
            self.current_waypoint = 1
            self.battery_status = {"level": 100.0, "voltage": 16.8}
            self.communication_state = "connected"
            self.gps_state = {"status": "healthy", "accuracy": 1.0}
            self.current_recovery_plan = None
            self.safety_status = None
