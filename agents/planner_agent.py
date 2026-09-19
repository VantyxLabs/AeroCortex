import json
import time
from typing import Any, Dict, Optional

from config import config
from llm.groq_client import GroqClient
from llm.ollama_client import SYSTEM_PROMPT, OllamaClient
from models import (
    UAVTelemetry,
    SituationReport,
    HybridMemoryContext,
    RecoveryPlan,
    LLMRecoveryPlan,
)


_UNREACHABLE = object()


class PlannerAgent:
    """
    Planner Agent:
    Builds a grounded prompt from hydrated Mongo episodes, semantic rules, and
    the situation report; asks Groq for JSON; if Groq is unreachable, asks
    Ollama; validates against RecoveryPlan; retries once on validation failure;
    otherwise uses the offline reasoner (Raspberry Pi / no LLM).
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        groq_client: Optional[GroqClient] = None,
    ):
        self.groq = groq_client if groq_client is not None else GroqClient()
        self.client = ollama_client or OllamaClient()
        self.min_confidence = config.llm.min_confidence_threshold

    def plan_recovery(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        memory_context: HybridMemoryContext,
        mission_objective: str = "Preserve UAV integrity and execute safe recovery",
    ) -> RecoveryPlan:
        start_t = time.time()

        if not situation.anomaly_detected or situation.failure_type == "NONE":
            return RecoveryPlan(
                action="CONTINUE_MISSION",
                reason="All telemetry channels nominal. Continuing scheduled waypoint navigation.",
                steps=[
                    "Maintain planned trajectory",
                    "Monitor remaining battery reserve",
                    "Proceed to next waypoint",
                ],
                expected_outcome="Nominal mission execution",
                confidence=0.99,
                risk_level="LOW",
                source="Deterministic-Nominal-Controller",
                plan_latency_ms=round((time.time() - start_t) * 1000, 2),
            )

        groq_plan = self._plan_with_client(
            self.groq, "groq", telemetry, situation, memory_context, mission_objective
        )
        if groq_plan is not _UNREACHABLE:
            if groq_plan and groq_plan.confidence >= self.min_confidence:
                groq_plan.source = "groq"
                groq_plan.plan_latency_ms = round((time.time() - start_t) * 1000, 2)
                return groq_plan
        else:
            ollama_plan = self._plan_with_ollama(
                telemetry, situation, memory_context, mission_objective
            )
            if (
                ollama_plan
                and ollama_plan is not _UNREACHABLE
                and ollama_plan.confidence >= self.min_confidence
            ):
                ollama_plan.source = "ollama"
                ollama_plan.plan_latency_ms = round((time.time() - start_t) * 1000, 2)
                return ollama_plan

        fallback = self.client.deterministic_reasoner(telemetry, situation, memory_context)
        fallback.source = "offline_reasoner"
        fallback.plan_latency_ms = round((time.time() - start_t) * 1000, 2)
        return fallback

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

    def _semantic_rules_block(self, memory_context: HybridMemoryContext) -> str:
        if not memory_context.semantic_rules:
            return "None"
        lines = []
        for rule in memory_context.semantic_rules[:5]:
            lines.append(
                f"- {rule.rule_id}: IF {rule.trigger} / {rule.condition} "
                f"THEN {rule.action} (confidence={rule.confidence})"
            )
        return "\n".join(lines)

    def build_user_prompt(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        memory_context: HybridMemoryContext,
        mission_objective: str,
        repair_hint: str = "",
    ) -> str:
        repair = ""
        if repair_hint:
            repair = (
                "PREVIOUS RESPONSE WAS INVALID. Fix this: "
                f"{repair_hint}\nReturn ONLY valid JSON matching the schema.\n"
            )
        graph_paths = memory_context.graph_paths[:2] if memory_context.graph_paths else []
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

Retrieved Hydrated Episodes (Mongo payloads when available, not ids only):
{self._hydrated_episode_block(memory_context)}

Matched Semantic Rules:
{self._semantic_rules_block(memory_context)}

Knowledge Graph Paths: {json.dumps(graph_paths, default=str)}
Top Memory Recommendation: {memory_context.top_recommended_action}

Mission Objective: {mission_objective}
{repair}
Provide your recovery plan as JSON with keys: action, reason, steps (list of strings), expected_outcome, confidence (float 0.0 to 1.0), risk_level (LOW/MEDIUM/HIGH)."""

    def _validate_llm_plan(self, parsed: Optional[Dict[str, Any]], source: str) -> RecoveryPlan:
        if not parsed or not isinstance(parsed, dict):
            raise ValueError("empty or non-object LLM response")
        draft = LLMRecoveryPlan.model_validate(parsed)
        return RecoveryPlan(
            action=draft.action,
            reason=draft.reason,
            steps=draft.steps,
            expected_outcome=draft.expected_outcome,
            confidence=draft.confidence,
            risk_level=draft.risk_level,
            source=source,
        )

    def _plan_with_client(
        self,
        client: Any,
        source: str,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        memory_context: HybridMemoryContext,
        mission_objective: str,
    ) -> Optional[RecoveryPlan]:
        prompt = self.build_user_prompt(
            telemetry, situation, memory_context, mission_objective
        )
        parsed = client.generate_json(prompt, system=SYSTEM_PROMPT)
        if parsed is None:
            return _UNREACHABLE
        try:
            return self._validate_llm_plan(parsed, source)
        except Exception as exc:
            repair_prompt = self.build_user_prompt(
                telemetry,
                situation,
                memory_context,
                mission_objective,
                repair_hint=str(exc),
            )
            parsed_retry = client.generate_json(repair_prompt, system=SYSTEM_PROMPT)
            if parsed_retry is None:
                return _UNREACHABLE
            try:
                return self._validate_llm_plan(parsed_retry, source)
            except Exception:
                return None

    def _plan_with_ollama(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        memory_context: HybridMemoryContext,
        mission_objective: str,
    ) -> Optional[RecoveryPlan]:
        return self._plan_with_client(
            self.client,
            "ollama",
            telemetry,
            situation,
            memory_context,
            mission_objective,
        )
