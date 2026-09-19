import logging
import secrets
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import config
from llm.groq_client import GroqClient
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
    try:
        engine = kg.reconnect()
        logger.info("Knowledge graph engine after startup: %s", engine)
    except Exception as exc:
        logger.warning("Neo4j reconnect failed: %s", exc)

    try:
        engine = graph.memory_agent.episodic_memory.vector_store.reconnect()
        logger.info("Vector store engine after startup: %s", engine)
    except Exception as exc:
        logger.warning("Vector store warmup failed: %s", exc)

    groq = GroqClient()
    if groq.ping():
        logger.info("Planner path: groq (%s)", groq.model)
    else:
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
    return secrets.compare_digest(provided, expected)


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
    vs = graph.memory_agent.episodic_memory.vector_store
    vector_health = vs.health()
    kg = graph.memory_agent.knowledge_graph
    kg.reconnect()
    neo_ok = kg.engine == "neo4j" and kg.ping()
    groq = GroqClient()
    groq_ok = groq.ping()
    ollama_ok = OllamaClient().ping()

    if groq_ok:
        groq_status = "ok"
    elif groq.is_configured:
        groq_status = "fallback_ollama" if ollama_ok else "fallback_reasoner"
    else:
        groq_status = "unset"

    ollama_status = "ok" if ollama_ok else "fallback_reasoner"
    pine_configured = vector_health["pinecone"] != "unset"
    vector_ok = bool(vector_health.get("ok"))

    dependencies = {
        "mongo": "ok" if mongo_ok else "down",
        "neo4j": "ok" if neo_ok else "fallback_networkx",
        "pinecone": vector_health["pinecone"],
        "chroma": vector_health["chroma"],
        "groq": groq_status,
        "ollama": ollama_status,
    }
    vector_degraded = (
        vector_health["pinecone"] in ("fallback_chroma", "down")
        or (not pine_configured and vector_health["chroma"] != "ok")
    )
    llm_degraded = (groq.is_configured and not groq_ok) or (not groq.is_configured and not ollama_ok)
    degraded = (not mongo_ok) or (not neo_ok) or vector_degraded or llm_degraded
    if not vector_ok and not mongo_ok:
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
        "vector_engine": vector_health.get("engine"),
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
    vector_engine = getattr(graph.memory_agent.episodic_memory.vector_store, "engine", "chroma")
    last = graph.memory_agent.last_context
    candidates = []
    if last:
        for item in last.retrieved_experiences:
            candidates.append({
                "episode_id": item.episode_id,
                "action": item.experience.action,
                "vector_similarity": item.vector_similarity,
                "graph_relevance": item.graph_relevance,
                "final_score": item.final_score,
                "hydrated_from_mongo": item.hydrated_from_mongo,
            })

    return {
        "episodic_experiences_count": chroma_count,
        "vector_engine": vector_engine,
        "semantic_rules_count": len(rules),
        "semantic_rules": rules,
        "knowledge_graph": kg_summary,
        "last_retrieval": {
            "top_recommended_action": last.top_recommended_action if last else None,
            "retrieval_latency_ms": last.retrieval_latency_ms if last else 0.0,
            "candidates": candidates,
        },
    }


@app.get("/missions")
async def get_missions_history(
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
):
    history = await document_store.list_missions(limit=limit, skip=skip)
    total = await document_store.count_missions()
    for item in history:
        if "_id" in item:
            item["_id"] = str(item["_id"])
    return {
        "total_events": total,
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
