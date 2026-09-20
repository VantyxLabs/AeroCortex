import time
import uuid
from typing import Any, Dict, Optional

from models import (
    UAVTelemetry, SituationReport, RecoveryPlan, SafetyVerdict,
    EpisodicExperience, HybridMemoryContext
)
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from memory.knowledge_graph import KnowledgeGraph
from memory.document_store import get_document_store


class LearningAgent:
    """
    Learning Agent:
    Write path: Mongo (source of truth) → Pinecone (Chroma fallback) → Neo4j.
    If Mongo is down the pipeline still returns a decision with persisted=false.
    """
    def __init__(
        self,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        document_store=None,
    ):
        self.episodic_memory = episodic_memory or EpisodicMemory()
        self.semantic_memory = semantic_memory or SemanticMemory()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.document_store = document_store

    def process_mission_outcome(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport,
        plan: RecoveryPlan,
        verdict: SafetyVerdict,
        success: bool = True,
        duration_s: float = 30.0,
        summary: str = "",
        memory_context: Optional[HybridMemoryContext] = None,
        latency_ms: Optional[Dict[str, float]] = None,
        forced_episode_id: Optional[str] = None,
        skip_if_duplicate: bool = False,
    ) -> Dict[str, Any]:
        if not situation.anomaly_detected or situation.failure_type == "NONE":
            return {"status": "SKIPPED", "reason": "No anomaly detected in mission", "persisted": False}

        condition_name = "MODERATE_WIND"
        if telemetry.wind_speed > 12.0:
            condition_name = "HIGH_WIND"
        elif telemetry.battery_level < 15.0:
            condition_name = "CRITICAL_BATTERY"
        elif telemetry.battery_level < 30.0:
            condition_name = "RESERVE_BATTERY"

        outcome_name = "MISSION_SUCCESS" if success else "MISSION_FAILURE"
        action_name = plan.action
        ts = time.time()
        latency = latency_ms or {}
        if memory_context is not None:
            latency.setdefault("retrieval", memory_context.retrieval_latency_ms)
        latency.setdefault("planning", getattr(plan, "plan_latency_ms", 0.0) or 0.0)
        latency.setdefault("safety", getattr(verdict, "latency_ms", 0.0) or 0.0)
        latency.setdefault(
            "total",
            round(
                float(latency.get("retrieval", 0) or 0)
                + float(latency.get("planning", 0) or 0)
                + float(latency.get("safety", 0) or 0),
                2,
            ),
        )

        retrieved = []
        if memory_context:
            for item in memory_context.retrieved_experiences:
                retrieved.append({
                    "episode_id": item.episode_id,
                    "action": item.experience.action,
                    "vector_similarity": item.vector_similarity,
                    "graph_relevance": item.graph_relevance,
                    "final_score": item.final_score,
                })

        episode_doc = {
            "mission_id": telemetry.mission_id,
            "timestamp": ts,
            "failure_type": situation.failure_type,
            "telemetry": telemetry.model_dump(),
            "situation": situation.model_dump(),
            "retrieved_context": retrieved,
            "plan": plan.model_dump(),
            "safety_verdict": verdict.model_dump(),
            "outcome": outcome_name,
            "success": success,
            "planner_source": plan.source,
            "latency_ms": latency,
        }
        if forced_episode_id:
            episode_doc["_id"] = forced_episode_id
            episode_doc["episode_id"] = forced_episode_id

        # Idempotent store put first (Dynamo conditional / Mongo insert)
        store = self.document_store if self.document_store is not None else None
        if store is None:
            try:
                from cloud.factories import get_document_store as _factory_store

                store = _factory_store()
            except Exception:
                from memory.document_store import get_document_store

                store = get_document_store()

        already = False
        if skip_if_duplicate and forced_episode_id and hasattr(store, "episode_exists"):
            already = store.episode_exists(telemetry.mission_id, forced_episode_id)

        persisted = False
        episode_id = forced_episode_id
        if not already:
            try:
                if hasattr(store, "last_insert_duplicate"):
                    store.last_insert_duplicate = False
                episode_id = store.insert_episode_sync(episode_doc)
                persisted = episode_id is not None
                if (
                    skip_if_duplicate
                    and persisted
                    and getattr(store, "last_insert_duplicate", False)
                ):
                    already = True
            except Exception:
                episode_id = None
                persisted = False

        if not episode_id:
            episode_id = forced_episode_id or str(uuid.uuid4())

        if already:
            # Duplicate SQS delivery — skip non-idempotent reinforcement
            return {
                "status": "SUCCESS",
                "episode_id": episode_id,
                "persisted": True,
                "duplicate": True,
                "failure_type": situation.failure_type,
                "action": action_name,
                "outcome": outcome_name,
                "planner_source": plan.source,
                "latency_ms": latency,
                "rule_updated": None,
                "summary": f"Skipped duplicate reinforcement for {episode_id}",
            }

        experience = EpisodicExperience(
            episode_id=episode_id,
            mission_id=telemetry.mission_id,
            failure=situation.failure_type,
            context=f"{situation.description} at altitude {telemetry.altitude:.1f}m in {condition_name}",
            environmental_conditions={
                "altitude": telemetry.altitude,
                "velocity": telemetry.velocity,
                "battery_pct": telemetry.battery_level,
                "wind_speed": telemetry.wind_speed,
                "condition": condition_name
            },
            action=action_name,
            outcome=outcome_name,
            success=success,
            mission_duration=round(duration_s, 2),
            confidence=round(plan.confidence, 2),
            timestamp=ts
        )
        self.episodic_memory.store_experience(experience, episode_id=episode_id)

        updated_rule = self.semantic_memory.reinforce_action(
            failure_type=situation.failure_type,
            action=action_name,
            success=success
        )

        self.knowledge_graph.add_mission_resolution(
            mission_id=telemetry.mission_id,
            failure=situation.failure_type,
            condition=condition_name,
            action=action_name,
            outcome=outcome_name,
            success=success,
            episode_id=episode_id,
        )

        return {
            "status": "SUCCESS",
            "episode_id": episode_id,
            "persisted": persisted,
            "failure_type": situation.failure_type,
            "action": action_name,
            "outcome": outcome_name,
            "planner_source": plan.source,
            "latency_ms": latency,
            "rule_updated": updated_rule.rule_id if updated_rule else None,
            "summary": summary or f"Stored experience {episode_id} for {situation.failure_type} -> {action_name}"
        }
