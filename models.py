from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import time

class GPSStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    LOST = "lost"

class CommsStatus(str, Enum):
    CONNECTED = "connected"
    DEGRADED = "degraded"
    LOST = "lost"

class MissionPhase(str, Enum):
    TAKEOFF = "TAKEOFF"
    CRUISE = "CRUISE"
    WAYPOINT_NAV = "WAYPOINT_NAV"
    HOVER = "HOVER"
    RTH = "RTH"
    LANDING = "LANDING"
    EMERGENCY = "EMERGENCY"

class FailureType(str, Enum):
    NONE = "NONE"
    GPS_INTERFERENCE = "GPS_INTERFERENCE"
    GPS_LOSS = "GPS_LOSS"
    BATTERY_DEGRADATION = "BATTERY_DEGRADATION"
    LOW_BATTERY = "LOW_BATTERY"
    COMMUNICATION_LOSS = "COMMUNICATION_LOSS"
    STRONG_WIND = "STRONG_WIND"
    SENSOR_ANOMALY = "SENSOR_ANOMALY"
    COMBINED_FAILURE = "COMBINED_FAILURE"

class SeverityLevel(str, Enum):
    NOMINAL = "NOMINAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class UAVTelemetry(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    mission_id: str = "MISSION_001"
    latitude: float = 11.0168
    longitude: float = 76.9558
    altitude: float = 120.0
    velocity: float = 12.5
    battery_level: float = 72.0
    battery_voltage: float = 15.4
    gps_status: str = "healthy"
    gps_accuracy: float = 1.8
    imu_acceleration: List[float] = Field(default_factory=lambda: [0.0, 0.0, 9.81])
    imu_gyroscope: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    wind_speed: float = 7.2
    wind_direction: float = 180.0
    communication_status: str = "connected"
    mission_state: str = "CRUISE"
    waypoint: int = 1
    heading: float = 90.0
    payload_status: str = "nominal"

class SituationReport(BaseModel):
    anomaly_detected: bool = False
    failure_type: str = "NONE"
    severity: str = "NOMINAL"
    confidence: float = 1.0
    current_state: Dict[str, Any] = Field(default_factory=dict)
    mission_phase: str = "CRUISE"
    timestamp: float = Field(default_factory=time.time)
    description: str = "Nominal operating conditions."

class EpisodicExperience(BaseModel):
    episode_id: Optional[str] = None
    mission_id: str
    failure: str
    context: str
    environmental_conditions: Dict[str, Any] = Field(default_factory=dict)
    action: str
    outcome: str
    success: bool
    mission_duration: float = 0.0
    confidence: float = 0.85
    timestamp: float = Field(default_factory=time.time)

class RetrievedExperience(BaseModel):
    experience: EpisodicExperience
    episode_id: Optional[str] = None
    vector_similarity: float = 0.0
    graph_relevance: float = 0.0
    final_score: float = 0.0
    hydrated_from_mongo: bool = False

class SemanticRule(BaseModel):
    rule_id: str
    trigger: str
    condition: str
    action: str
    confidence: float = 0.80
    success_count: int = 1
    failure_count: int = 0
    description: str = ""

class HybridMemoryContext(BaseModel):
    retrieved_experiences: List[RetrievedExperience] = Field(default_factory=list)
    semantic_rules: List[SemanticRule] = Field(default_factory=list)
    graph_paths: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    top_recommended_action: Optional[str] = None

class RecoveryPlan(BaseModel):
    action: str = "CONTINUE_MISSION"
    reason: str = "Nominal trajectory"
    steps: List[str] = Field(default_factory=lambda: ["Maintain scheduled path"])
    expected_outcome: str = "Mission completion"
    confidence: float = 0.95
    risk_level: str = "LOW"
    source: str = "Planner"
    plan_latency_ms: float = 0.0


class LLMRecoveryPlan(BaseModel):
    """Strict schema for Groq / Ollama JSON — missing fields fail validation."""
    action: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    steps: List[str] = Field(min_length=1)
    expected_outcome: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]

    @field_validator("risk_level", mode="before")
    @classmethod
    def _normalize_risk(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper()
        return value

class SafetyVerdict(BaseModel):
    approved: bool = True
    reason: str = "Plan satisfies all flight envelope constraints"
    risk_level: str = "LOW"
    fallback_action: str = "RETURN_TO_HOME"
    violated_constraints: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0

class MissionOutcome(BaseModel):
    mission_id: str
    failure_type: str
    action_executed: str
    safety_approved: bool
    fallback_triggered: bool
    success: bool
    final_battery: float
    time_to_stabilize_s: float
    summary: str = ""

__all__ = [
    "GPSStatus", "CommsStatus", "MissionPhase", "FailureType", "SeverityLevel",
    "UAVTelemetry", "SituationReport", "EpisodicExperience", "RetrievedExperience",
    "SemanticRule", "HybridMemoryContext", "RecoveryPlan", "LLMRecoveryPlan",
    "SafetyVerdict", "MissionOutcome"
]
