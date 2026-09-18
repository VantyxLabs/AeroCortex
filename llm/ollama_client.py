import time
import json
from typing import Dict, Any, Optional, List
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
    Client for interacting with local Gemma 3 via Ollama.
    Supports strict JSON parsing, low-latency edge timeouts, and
    deterministic offline cognitive fallback if Ollama service is unavailable.
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

    def generate_plan(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        memory_context: HybridMemoryContext,
        mission_objective: str = "Safely complete delivery route and maintain UAV integrity"
    ) -> RecoveryPlan:
        start_t = time.time()
        
        # Nominal condition
        if not situation.anomaly_detected or situation.failure_type == "NONE":
            return RecoveryPlan(
                action="CONTINUE_MISSION",
                reason="All telemetry channels nominal. Continuing scheduled waypoint navigation.",
                steps=["Maintain planned trajectory", "Monitor remaining battery reserve", "Proceed to next waypoint"],
                expected_outcome="Nominal mission execution",
                confidence=0.99,
                risk_level="LOW",
                source="Deterministic-Nominal-Controller",
                plan_latency_ms=round((time.time() - start_t) * 1000, 2)
            )

        ollama_plan = self._call_ollama(telemetry, situation, memory_context, mission_objective)
        if ollama_plan and ollama_plan.confidence >= config.llm.min_confidence_threshold:
            ollama_plan.source = "ollama"
            ollama_plan.plan_latency_ms = round((time.time() - start_t) * 1000, 2)
            return ollama_plan

        fallback_plan = self._deterministic_cognitive_reasoner(telemetry, situation, memory_context)
        fallback_plan.source = "offline_reasoner"
        fallback_plan.plan_latency_ms = round((time.time() - start_t) * 1000, 2)
        return fallback_plan

    def _hydrated_episode_block(self, memory_context: HybridMemoryContext) -> str:
        if not memory_context.retrieved_experiences:
            return "None"
        lines = []
        for item in memory_context.retrieved_experiences[:3]:
            exp = item.experience
            lines.append(
                f"- episode_id={item.episode_id or exp.episode_id} mission={exp.mission_id} "
                f"failure={exp.failure} action={exp.action} outcome={exp.outcome} "
                f"success={exp.success} vec={item.vector_similarity:.3f} "
                f"graph={item.graph_relevance:.3f} final={item.final_score:.3f} "
                f"hydrated_from_mongo={item.hydrated_from_mongo} context={exp.context}"
            )
        return "\n".join(lines)

    def _build_user_prompt(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        memory_context: HybridMemoryContext,
        mission_objective: str,
        repair_hint: str = ""
    ) -> str:
        top_rule_text = "None"
        if memory_context.semantic_rules:
            top_rule = memory_context.semantic_rules[0]
            top_rule_text = f"{top_rule.action} (Confidence: {top_rule.confidence}, Rule: {top_rule.rule_id})"

        repair = ""
        if repair_hint:
            repair = f"\nPREVIOUS RESPONSE WAS INVALID. Fix this: {repair_hint}\nReturn ONLY valid JSON matching the schema.\n"

        return f"""Current Telemetry:
- Mission ID: {telemetry.mission_id}
- Altitude: {telemetry.altitude} m AGL
- Velocity: {telemetry.velocity} m/s
- Battery Level: {telemetry.battery_level}% (Voltage: {telemetry.battery_voltage} V)
- GPS: Status={telemetry.gps_status}, Accuracy={telemetry.gps_accuracy}
- Wind: Speed={telemetry.wind_speed} m/s, Direction={telemetry.wind_direction} deg
- Comms: {telemetry.communication_status}

Detected Anomaly:
- Failure Type: {situation.failure_type}
- Severity: {situation.severity}
- Confidence: {situation.confidence}
- Mission Phase: {situation.mission_phase}
- Description: {situation.description}

Retrieved Hydrated Episodes (from Mongo when available):
{self._hydrated_episode_block(memory_context)}

Matched Semantic Rules:
- Top Semantic Rule: {top_rule_text}

Knowledge Graph Paths: {memory_context.graph_paths[:2] if memory_context.graph_paths else 'None'}
Top Memory Recommendation: {memory_context.top_recommended_action}

Mission Objective: {mission_objective}
{repair}
Provide your recovery plan in JSON format with keys: action, reason, steps (list of strings), expected_outcome, confidence (float 0.0 to 1.0), risk_level (LOW/MEDIUM/HIGH)."""

    def _parse_plan(self, parsed: Dict[str, Any], memory_context: HybridMemoryContext) -> RecoveryPlan:
        return RecoveryPlan(
            action=parsed.get("action", memory_context.top_recommended_action or "RETURN_TO_HOME"),
            reason=parsed.get("reason", "Local Gemma 3 recovery synthesis"),
            steps=parsed.get("steps", ["Execute stabilized recovery action", "Evaluate sensor recovery"]),
            expected_outcome=parsed.get("expected_outcome", "Stabilization and safety preservation"),
            confidence=float(parsed.get("confidence", 0.85)),
            risk_level=parsed.get("risk_level", "MEDIUM"),
            source="ollama"
        )

    def _post_generate(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout)) as client:
            resp = client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "system": SYSTEM_PROMPT,
                    "prompt": user_prompt,
                    "stream": False,
                    "format": "json"
                }
            )
            if resp.status_code != 200:
                return None
            body = resp.json()
            raw_text = body.get("response", "")
            return json.loads(raw_text)

    def _call_ollama(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        memory_context: HybridMemoryContext,
        mission_objective: str
    ) -> Optional[RecoveryPlan]:
        user_prompt = self._build_user_prompt(telemetry, situation, memory_context, mission_objective)
        try:
            parsed = self._post_generate(user_prompt)
            if parsed is None:
                return None
            try:
                return self._parse_plan(parsed, memory_context)
            except Exception as exc:
                repair_prompt = self._build_user_prompt(
                    telemetry, situation, memory_context, mission_objective,
                    repair_hint=str(exc)
                )
                parsed_retry = self._post_generate(repair_prompt)
                if parsed_retry is None:
                    return None
                return self._parse_plan(parsed_retry, memory_context)
        except Exception:
            return None

    def _deterministic_cognitive_reasoner(
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
