import json
from typing import Dict, Any, Optional

import httpx

from config import config
from models import UAVTelemetry, SituationReport, HybridMemoryContext, RecoveryPlan

SYSTEM_PROMPT = """You are the AeroCortex UAV Recovery Planner.
You do not directly control the UAV.
You must generate recovery recommendations using ONLY the current telemetry, mission objective, retrieved mission experiences, semantic rules, and knowledge graph context.
Never invent sensor readings.
Never recommend an action that violates provided constraints.
If information is insufficient, return LOW_CONFIDENCE.
Return structured JSON containing:
action,
reason,
steps,
expected_outcome,
confidence,
risk_level.
Prioritize UAV safety over mission completion."""


class OllamaClient:
    """
    Local Ollama HTTP client with format=json, short edge timeouts, and a
    deterministic offline reasoner when the model is unreachable.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        self.base_url = base_url or config.llm.base_url
        self.model = model or config.llm.model
        self.timeout = timeout or config.llm.timeout_seconds
        self.connect_timeout = config.llm.connect_timeout_seconds

    def ping(self) -> bool:
        try:
            with httpx.Client(timeout=httpx.Timeout(0.5, connect=self.connect_timeout)) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def generate_json(self, user_prompt: str, system: str = SYSTEM_PROMPT) -> Optional[Dict[str, Any]]:
        """POST /api/generate with format=json. Returns a dict or None on any failure."""
        try:
            return self._post_generate(user_prompt, system)
        except Exception:
            return None

    def generate_plan(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        memory_context: HybridMemoryContext,
        mission_objective: str = "Safely complete delivery route and maintain UAV integrity"
    ) -> RecoveryPlan:
        from agents.planner_agent import PlannerAgent

        return PlannerAgent(ollama_client=self).plan_recovery(
            telemetry=telemetry,
            situation=situation,
            memory_context=memory_context,
            mission_objective=mission_objective,
        )

    def _post_generate(self, user_prompt: str, system: str = SYSTEM_PROMPT) -> Optional[Dict[str, Any]]:
        with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout)) as client:
            resp = client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "system": system,
                    "prompt": user_prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                return None
            body = resp.json()
            raw_text = body.get("response", "")
            if isinstance(raw_text, dict):
                return raw_text
            if not raw_text:
                return None
            return json.loads(raw_text)

    def deterministic_reasoner(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        memory: HybridMemoryContext
    ) -> RecoveryPlan:
        """
        Deterministic cognitive edge reasoner:
        Synthesizes hybrid memory (Episodic + Semantic + KG) into a structured plan
        guaranteeing zero downtime when Ollama is offline.
        """
        recommended_action = memory.top_recommended_action or "RETURN_TO_HOME"
        ftype = situation.failure_type

        reason = f"Synthesized from retrieved experiences and rules for {ftype}."
        steps = ["Verify actuation parameters", f"Execute {recommended_action}", "Hold position and verify telemetry"]
        expected = "UAV stabilization and mission integrity preservation"
        risk = "LOW"
        conf = 0.88

        if ftype == "GPS_INTERFERENCE":
            recommended_action = "SWITCH_TO_VIO_DEAD_RECKONING"
            reason = "GPS degraded; switching to Visual-Inertial Odometry and inertial dead-reckoning."
            steps = [
                "Disengage GPS-based position hold",
                "Engage VIO / Optical Flow positioning loop",
                "Maintain current altitude and evaluate drift rate"
            ]
            expected = "Stable hovering/transit without GPS lock"
            conf = 0.92

        elif ftype == "GPS_LOSS":
            recommended_action = "INERTIAL_DEAD_RECKONING_SAFE_RTH"
            reason = "Total GPS signal lost; executing dead-reckoned Return-to-Home trajectory."
            steps = [
                "Switch to inertial navigation filter",
                "Align reverse heading towards home launch coordinates",
                "Climb/descend to safe return corridor altitude (50m)",
                "Begin steady dead-reckoning transit"
            ]
            expected = "Safe transit back to launch origin"
            risk = "MEDIUM"
            conf = 0.90

        elif ftype in ("BATTERY_DEGRADATION", "LOW_BATTERY"):
            if telemetry.battery_level < 15.0 or telemetry.battery_voltage < 13.8:
                recommended_action = "CONTROLLED_EMERGENCY_LAND"
                reason = "Battery level/voltage below critical threshold; immediate landing required."
                steps = [
                    "Halt forward transit immediately",
                    "Engage descent rate of 1.5 m/s",
                    "Scan landing zone with optical sensor",
                    "Cut throttle upon touchdown"
                ]
                expected = "Immediate safe touchdown before battery exhaustion"
                risk = "HIGH"
                conf = 0.96
            else:
                recommended_action = "POWER_CONSERVATIVE_RTH"
                reason = "Battery level low; throttling propulsion to maximum endurance speed for RTH."
                steps = [
                    "Reduce airspeed to best-endurance speed (8 m/s)",
                    "Initiate direct flight path to home base",
                    "Monitor cell voltage decay rate"
                ]
                expected = "Arrival at base with >10% reserve"
                risk = "MEDIUM"
                conf = 0.91

        elif ftype == "STRONG_WIND":
            recommended_action = "REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT"
            reason = "High winds detected; descending below boundary layer and reducing velocity."
            steps = [
                "Reduce forward groundspeed by 40%",
                "Descend by 25m to lower wind shear layer",
                "Increase attitude controller gains for gust rejection"
            ]
            expected = "Reduced aerodynamic buffeting and stable flight"
            conf = 0.89

        elif ftype == "COMMUNICATION_LOSS":
            recommended_action = "AUTONOMOUS_FAILSAFE_HOLD_THEN_RTH"
            reason = "Telemetry link severed; loitering 10s then auto-returning home."
            steps = [
                "Enter safe circular loiter at current coordinates",
                "Check for carrier signal re-acquisition (10s)",
                "If not reconnected, execute autonomous RTH along pre-planned route"
            ]
            expected = "Re-connection or safe return without pilot intervention"
            conf = 0.90

        elif ftype == "COMBINED_FAILURE":
            recommended_action = "CONTROLLED_EMERGENCY_LAND"
            reason = "Multiple simultaneous critical subsystem anomalies detected."
            steps = [
                "Cut mission objectives",
                "Locate nearest clear landing spot",
                "Execute vertical auto-land"
            ]
            expected = "Prevent hull loss through controlled landing"
            risk = "HIGH"
            conf = 0.95

        return RecoveryPlan(
            action=recommended_action,
            reason=reason,
            steps=steps,
            expected_outcome=expected,
            confidence=conf,
            risk_level=risk,
            source="offline_reasoner"
        )

    def _deterministic_cognitive_reasoner(self, *args, **kwargs) -> RecoveryPlan:
        return self.deterministic_reasoner(*args, **kwargs)
