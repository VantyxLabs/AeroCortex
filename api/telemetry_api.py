import hashlib
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import config
from llm.ollama_client import OllamaClient
from memory.document_store import DocumentStore, get_document_store
from models import UAVTelemetry
from orchestration.graph import AeroCortexGraph
from simulation.failure_scenarios import FailureScenarioInjector
from simulation.mission_simulator import MissionSimulator

logger = logging.getLogger("aerocortex.api")

graph = AeroCortexGraph()
simulator = MissionSimulator(graph=graph)
document_store: DocumentStore = get_document_store()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global document_store
    document_store = get_document_store()
    try:
        await document_store.ensure_indexes()
    except Exception as exc:
        logger.warning("Mongo startup failed: %s", exc)

    kg = graph.memory_agent.knowledge_graph
    if kg.engine == "neo4j":
        try:
            if kg._neo and kg._neo.node_count() == 0:
                kg._neo.seed_ontology()
        except Exception as exc:
            logger.warning("Neo4j seed failed: %s", exc)

    try:
        graph.memory_agent.episodic_memory.vector_store.count()
    except Exception as exc:
        logger.warning("Chroma warmup failed: %s", exc)

    ollama = OllamaClient()
    if ollama.ping():
        logger.info("Planner path: ollama (%s)", config.llm.model)
    else:
        logger.info("Planner path: offline_reasoner")

    yield

    try:
        document_store.close()
    except Exception:
        pass
    try:
        graph.memory_agent.knowledge_graph.close()
    except Exception:
        pass


app = FastAPI(
    title="AeroCortex Telemetry API",
    description="Offline Edge Cognitive Memory & Flight Recovery Gateway",
    version=config.system.version,
    lifespan=lifespan,
)

EXEMPT_PATHS = {"/healthz", "/docs", "/openapi.json", "/redoc"}


def _api_key_ok(provided: str, expected: str) -> bool:
    if not expected:
        return True
    provided_digest = hashlib.sha256(provided.encode("utf-8")).digest()
    expected_digest = hashlib.sha256(expected.encode("utf-8")).digest()
    return secrets.compare_digest(provided_digest, expected_digest)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    path = request.url.path
    if path in EXEMPT_PATHS or path.startswith("/docs"):
        return await call_next(request)
    expected = config.api_key or ""
    provided = request.headers.get("X-API-Key", "")
    if not _api_key_ok(provided, expected):
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)


class SimulateRequest(BaseModel):
    scenario: str = "GPS_INTERFERENCE"
    steps: int = 1
    inject_step: int = 1


def _dump(model) -> Optional[Dict[str, Any]]:
    return model.model_dump() if model is not None else None


def _pipeline_payload(telemetry: UAVTelemetry, result: Dict[str, Any]) -> Dict[str, Any]:
    situation = result.get("situation")
    final_plan = result.get("final_plan")
    safety_verdict = result.get("safety_verdict")
    memory_context = result.get("memory_context")
    learning = result.get("learning_result") or {}
    planner_plan = result.get("planner_plan") or final_plan

    candidates = []
    if memory_context:
        for item in memory_context.retrieved_experiences:
            candidates.append({
                "episode_id": item.episode_id,
                "action": item.experience.action,
                "vector_similarity": item.vector_similarity,
                "graph_relevance": item.graph_relevance,
                "final_score": item.final_score,
                "hydrated_from_mongo": item.hydrated_from_mongo,
            })

    return {
        "status": "SUCCESS",
        "action": final_plan.action if final_plan else "CONTINUE_MISSION",
        "is_approved": safety_verdict.approved if safety_verdict else True,
        "execution_status": result.get("execution_status", "UNKNOWN"),
        "plan": _dump(final_plan),
        "safety": _dump(safety_verdict),
        "situation": _dump(situation),
        "persisted": bool(learning.get("persisted", False)),
        "episode_id": learning.get("episode_id"),
        "planner_source": getattr(planner_plan, "source", None) if planner_plan else None,
        "latency_ms": learning.get("latency_ms") or {
            "retrieval": getattr(memory_context, "retrieval_latency_ms", 0.0) if memory_context else 0.0,
            "planning": getattr(planner_plan, "plan_latency_ms", 0.0) if planner_plan else 0.0,
            "safety": getattr(safety_verdict, "latency_ms", 0.0) if safety_verdict else 0.0,
        },
        "memory": {
            "top_recommended_action": getattr(memory_context, "top_recommended_action", None) if memory_context else None,
            "candidates": candidates,
            "graph_paths": getattr(memory_context, "graph_paths", []) if memory_context else [],
        },
    }


@app.get("/")
def root():
    return {
        "system": config.system.app_name,
        "version": config.system.version,
        "status": "ONLINE",
        "offline_mode": config.system.offline_mode
    }


@app.get("/healthz")
async def healthz():
    mongo_ok = await document_store.ping()
    chroma_ok = graph.memory_agent.episodic_memory.vector_store.ping()
    kg = graph.memory_agent.knowledge_graph
    neo_ok = kg.engine == "neo4j" and kg.ping()
    ollama_ok = OllamaClient().ping()

    dependencies = {
        "mongo": "ok" if mongo_ok else "down",
        "neo4j": "ok" if neo_ok else "fallback_networkx",
        "chroma": "ok" if chroma_ok else "down",
        "ollama": "ok" if ollama_ok else "fallback_reasoner",
    }
    degraded = (not mongo_ok) or (not neo_ok) or (not chroma_ok) or (not ollama_ok)
    if not chroma_ok and not mongo_ok:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "dependencies": dependencies,
                "version": config.system.version,
            },
        )
    return {
        "status": "degraded" if degraded else "ok",
        "dependencies": dependencies,
        "version": config.system.version,
        "engine": kg.get_summary().get("engine"),
    }


@app.post("/telemetry")
def receive_telemetry(telemetry: UAVTelemetry):
    """
    Ingests live UAV telemetry, routes through the LangGraph cognitive pipeline
    (situation → hybrid memory → planner → safety → learning), and returns a
    safety-validated recovery action.
    """
    try:
        result = graph.run(telemetry)
        return _pipeline_payload(telemetry, result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def get_system_status():
    snapshot = graph.working_memory.get_snapshot()
    kg_summary = graph.memory_agent.knowledge_graph.get_summary()
    return {
        "status": "HEALTHY",
        "engine": kg_summary.get("engine"),
        "working_memory": snapshot,
        "kg_summary": kg_summary,
    }


@app.get("/memory")
def get_memory_state():
    rules = [r.model_dump() for r in graph.memory_agent.semantic_memory.get_all_rules()]
    kg_summary = graph.memory_agent.knowledge_graph.get_summary()
    chroma_count = graph.memory_agent.episodic_memory.vector_store.count()

    return {
        "episodic_experiences_count": chroma_count,
        "semantic_rules_count": len(rules),
        "semantic_rules": rules,
        "knowledge_graph": kg_summary
    }


@app.get("/missions")
async def get_missions_history(
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
):
    history = await document_store.list_missions(limit=limit, skip=skip)
    for item in history:
        if "_id" in item:
            item["_id"] = str(item["_id"])
    return {
        "total_events": len(history),
        "limit": limit,
        "skip": skip,
        "history": history,
        "persisted": document_store.available,
    }


@app.post("/simulate")
def trigger_simulation(req: SimulateRequest):
    if req.scenario not in FailureScenarioInjector.list_available_scenarios():
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario. Available: {FailureScenarioInjector.list_available_scenarios()}"
        )

    if req.steps == 1:
        step_res = simulator.run_step(scenario=req.scenario)
        return {"result": step_res}
    results = simulator.run_full_mission(
        total_steps=req.steps,
        inject_step=req.inject_step,
        scenario=req.scenario
    )
    return {"total_steps": len(results), "results": results}


@app.post("/reset")
def reset_state():
    graph.working_memory.clear()
    simulator.reset()
    return {"status": "SUCCESS", "message": "Working memory and simulation reset successfully"}
