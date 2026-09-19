import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from config import config
from models import (
    UAVTelemetry, SituationReport, EpisodicExperience, RetrievedExperience,
    HybridMemoryContext
)
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from memory.knowledge_graph import KnowledgeGraph
from memory.document_store import get_document_store


def minmax_normalize(values: List[float]) -> List[float]:
    """Scale a score list to [0, 1] across the candidate set."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def chroma_distance_to_similarity(distance: float, space: str = "cosine") -> float:
    """Convert a Chroma distance into a similarity in [0, 1]."""
    d = float(distance)
    if space == "l2":
        sim = 1.0 / (1.0 + max(d, 0.0))
    else:
        sim = 1.0 - d
    return max(0.0, min(1.0, sim))


def fuse_hybrid_scores(
    vector_items: List[Dict[str, Any]],
    graph_items: List[Dict[str, Any]],
    w_vector: float = 0.6,
    w_graph: float = 0.4,
    space: str = "cosine",
) -> List[Dict[str, Any]]:
    """
    Convert Chroma distance → similarity, min-max normalize each list, then fuse.

    vector_items: [{episode_id, action, distance? | vector_similarity, experience?}]
    graph_items:  [{action, graph_relevance}]
    """
    kg_map = {item["action"]: float(item["graph_relevance"]) for item in graph_items}
    vec_raw: List[float] = []
    for item in vector_items:
        if item.get("distance") is not None:
            vec_raw.append(chroma_distance_to_similarity(item["distance"], space))
        else:
            vec_raw.append(float(item.get("vector_similarity", 0.0)))
    graph_raw = [float(kg_map.get(item.get("action"), 0.0)) for item in vector_items]
    vec_norm = minmax_normalize(vec_raw)
    graph_norm = minmax_normalize(graph_raw)

    fused: List[Dict[str, Any]] = []
    for i, item in enumerate(vector_items):
        vs = vec_norm[i] if i < len(vec_norm) else 0.0
        gr = graph_norm[i] if i < len(graph_norm) else 0.0
        fused.append({
            **item,
            "vector_similarity": round(vs, 4),
            "graph_relevance": round(gr, 4),
            "final_score": round(w_vector * vs + w_graph * gr, 4),
        })
    fused.sort(key=lambda x: x["final_score"], reverse=True)
    return fused


class MemoryAgent:
    """
    Memory Agent:
    Coordinates Hybrid Memory Retrieval across:
    1. Pinecone Vector Store with Chroma fallback (Episodic Experience)
    2. Neo4j / Embedded Knowledge Graph (Relational Paths)
    3. Semantic Memory (Operational IF-THEN Rules)
    4. MongoDB hydration of winning episode payloads
    """
    def __init__(
        self,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None
    ):
        self.episodic_memory = episodic_memory or EpisodicMemory()
        self.semantic_memory = semantic_memory or SemanticMemory()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()

        self.w_vector = config.memory.hybrid_weights.vector_similarity
        self.w_graph = config.memory.hybrid_weights.graph_relevance
        self.top_k = config.memory.top_k_episodes
        self.last_context: Optional[HybridMemoryContext] = None

    def _fetch_chroma_and_graph(
        self, query_text: str, failure_type: str, cond_hint: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Query Chroma and the knowledge graph concurrently."""

        def _ok(value: Any, fallback: List) -> List:
            if isinstance(value, BaseException) or value is None:
                return fallback
            return value

        async def _gather():
            chroma_task = asyncio.to_thread(
                self.episodic_memory.retrieve_similar_experiences, query_text, self.top_k
            )
            kg_task = asyncio.to_thread(
                self.knowledge_graph.query_action_relevance,
                failure_type,
                cond_hint,
                5,
            )
            return await asyncio.gather(chroma_task, kg_task, return_exceptions=True)

        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if not in_loop:
            raw_episodes, kg_candidates = asyncio.run(_gather())
            return _ok(raw_episodes, []), _ok(kg_candidates, [])

        with ThreadPoolExecutor(max_workers=2) as pool:
            chroma_fut = pool.submit(
                self.episodic_memory.retrieve_similar_experiences, query_text, self.top_k
            )
            kg_fut = pool.submit(
                self.knowledge_graph.query_action_relevance,
                failure_type,
                cond_hint,
                5,
            )
            try:
                raw_episodes = chroma_fut.result()
            except Exception:
                raw_episodes = []
            try:
                kg_candidates = kg_fut.result()
            except Exception:
                kg_candidates = []
        return raw_episodes or [], kg_candidates or []

    def retrieve_context(
        self,
        telemetry: UAVTelemetry,
        situation: SituationReport
    ) -> HybridMemoryContext:
        start_t = time.time()

        if not situation.anomaly_detected or situation.failure_type == "NONE":
            context = HybridMemoryContext(
                retrieved_experiences=[],
                semantic_rules=[],
                graph_paths=[],
                retrieval_latency_ms=0.0,
                top_recommended_action="CONTINUE_MISSION"
            )
            self.last_context = context
            return context

        query_text = (
            f"Failure: {situation.failure_type}. "
            f"Altitude: {telemetry.altitude}m, Wind: {telemetry.wind_speed}m/s, "
            f"Battery: {telemetry.battery_level}%, State: {situation.mission_phase}"
        )
        cond_hint = "MODERATE_WIND"
        if telemetry.wind_speed > 12.0:
            cond_hint = "HIGH_WIND"
        elif telemetry.battery_level < 15.0:
            cond_hint = "CRITICAL_BATTERY"
        elif telemetry.battery_level < 30.0:
            cond_hint = "RESERVE_BATTERY"

        raw_episodes, kg_candidates = self._fetch_chroma_and_graph(
            query_text, situation.failure_type, cond_hint
        )

        vector_items = []
        for ep_data in raw_episodes:
            exp: EpisodicExperience = ep_data["experience"]
            vector_items.append({
                "episode_id": ep_data.get("episode_id") or exp.episode_id,
                "action": exp.action,
                "distance": ep_data.get("distance"),
                "vector_similarity": ep_data.get("vector_similarity", 0.0),
                "experience": exp,
            })

        fused = fuse_hybrid_scores(
            vector_items, kg_candidates, self.w_vector, self.w_graph
        )
        ranked_experiences = self._hydrate_from_mongo(fused)

        matched_rules = self.semantic_memory.match_rules(situation.failure_type, telemetry)

        top_action = None
        if ranked_experiences and ranked_experiences[0].final_score > 0.65:
            top_action = ranked_experiences[0].experience.action
        elif matched_rules:
            top_action = matched_rules[0].action
        elif kg_candidates:
            top_action = kg_candidates[0]["action"]
        else:
            top_action = "RETURN_TO_HOME"

        latency = round((time.time() - start_t) * 1000, 2)

        context = HybridMemoryContext(
            retrieved_experiences=ranked_experiences,
            semantic_rules=matched_rules,
            graph_paths=kg_candidates,
            retrieval_latency_ms=latency,
            top_recommended_action=top_action
        )
        self.last_context = context
        return context

    def _hydrate_from_mongo(self, fused: List[Dict[str, Any]]) -> List[RetrievedExperience]:
        ids = [item.get("episode_id") for item in fused if item.get("episode_id")]
        docs_by_id: Dict[str, Dict[str, Any]] = {}
        try:
            store = get_document_store()
            if store.available and ids:
                for doc in store.get_episodes_sync(ids):
                    docs_by_id[str(doc.get("_id"))] = doc
        except Exception:
            docs_by_id = {}

        ranked: List[RetrievedExperience] = []
        for item in fused:
            exp: EpisodicExperience = item["experience"]
            episode_id = item.get("episode_id")
            hydrated = False
            doc = docs_by_id.get(episode_id) if episode_id else None
            if doc:
                hydrated = True
                sit = doc.get("situation") or {}
                plan = doc.get("plan") or {}
                outcome = doc.get("outcome") or exp.outcome
                if sit.get("description"):
                    exp.context = sit["description"]
                if plan.get("action"):
                    exp.action = plan["action"]
                if isinstance(outcome, str):
                    exp.outcome = outcome
                exp.episode_id = episode_id
            ranked.append(RetrievedExperience(
                experience=exp,
                episode_id=episode_id,
                vector_similarity=item["vector_similarity"],
                graph_relevance=item["graph_relevance"],
                final_score=item["final_score"],
                hydrated_from_mongo=hydrated,
            ))
        return ranked
