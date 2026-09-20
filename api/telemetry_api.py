import asyncio
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import config
from llm.groq_client import GroqClient
from llm.ollama_client import OllamaClient
from models import UAVTelemetry
from orchestration.graph import AeroCortexGraph
from simulation.failure_scenarios import FailureScenarioInjector
from simulation.mission_simulator import MissionSimulator

logger = logging.getLogger("aerocortex.api")


def _lazy_runtime() -> bool:
    return bool(os.getenv("AWS_EXECUTION_ENV") or os.getenv("LAZY_RUNTIME"))


def _build_document_store():
    try:
        from cloud.factories import get_document_store as factory_store

        return factory_store()
    except Exception:
        from memory.document_store import get_document_store

        return get_document_store()


def _build_graph() -> AeroCortexGraph:
    try:
        from cloud.factories import get_working_memory

        return AeroCortexGraph(working_memory=get_working_memory())
    except Exception:
        return AeroCortexGraph()


# Module-level singletons — eager locally (tests / uvicorn), lazy on Lambda.
graph: Optional[AeroCortexGraph] = None
simulator: Optional[MissionSimulator] = None
document_store = None

if not _lazy_runtime():
    graph = _build_graph()
    simulator = MissionSimulator(graph=graph)
    document_store = _build_document_store()


def get_graph() -> AeroCortexGraph:
    global graph, simulator
    if graph is None:
        graph = _build_graph()
    if simulator is None:
        simulator = MissionSimulator(graph=graph)
    return graph


def get_simulator() -> MissionSimulator:
    global simulator
    get_graph()
    assert simulator is not None
    return simulator


def get_store():
    global document_store
    if document_store is None:
        document_store = _build_document_store()
    return document_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_store()
    g = get_graph()
    try:
        await store.ensure_indexes()
    except Exception as exc:
        logger.warning("Mission store startup failed: %s", exc)

    kg = g.memory_agent.knowledge_graph
    try:
        engine = kg.reconnect()
        logger.info("Knowledge graph engine after startup: %s", engine)
    except Exception as exc:
        logger.warning("Neo4j reconnect failed: %s", exc)

    try:
        engine = g.memory_agent.episodic_memory.vector_store.reconnect()
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
        store.close()
    except Exception:
        pass
    try:
        g.memory_agent.knowledge_graph.close()
    except Exception:
        pass


app = FastAPI(
    title="AeroCortex Telemetry API",
    description="Offline Edge Cognitive Memory & Flight Recovery Gateway",
    version=config.system.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXEMPT_PATHS = {"/", "/healthz", "/docs", "/openapi.json", "/redoc"}


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
        "learning_mode": learning.get("learning_mode"),
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


_HEALTH_CACHE: Dict[str, Any] = {"ts": 0.0, "payload": None, "status_code": 200}
_HEALTH_TTL_S = 8.0


def _mission_store_mode() -> str:
    try:
        from cloud.factories import mission_store_mode

        return mission_store_mode()
    except Exception:
        return "mongo"


def _compute_health() -> tuple[int, Dict[str, Any]]:
    """Blocking dependency checks — always run via asyncio.to_thread."""
    store = get_store()
    g = get_graph()
    mode = _mission_store_mode()

    mongo_ok = False
    dynamo_status = None
    mongo_status = None
    if mode == "dynamodb":
        try:
            dynamo_status = "ok" if store.health() == "ok" else "down"
        except Exception:
            dynamo_status = "down"
        # DynamoDB-only mission store — do not report Mongo at all
        mission_ok = dynamo_status == "ok"
    else:
        try:
            mongo_ok = asyncio.run(store.ping())
        except Exception:
            mongo_ok = False
        mongo_status = "ok" if mongo_ok else "down"
        mission_ok = mongo_ok

    vs = g.memory_agent.episodic_memory.vector_store
    vector_health = vs.health()
    kg = g.memory_agent.knowledge_graph
    neo_ok = kg.engine == "neo4j" and kg.ping()
    if not neo_ok:
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
    chroma_status = vector_health["chroma"]

    bedrock_status = None
    try:
        from llm.bedrock_client import BedrockClient, last_call_ok

        bc = BedrockClient()
        if bc.is_configured:
            last = last_call_ok()
            if last is False:
                bedrock_status = "down"
            else:
                bedrock_status = "ok"
        else:
            bedrock_status = "unconfigured"
    except Exception:
        bedrock_status = None

    dependencies: Dict[str, Any] = {
        "neo4j": "ok" if neo_ok else "fallback_networkx",
        "pinecone": vector_health["pinecone"],
        "chroma": chroma_status,
        "groq": groq_status,
        "ollama": ollama_status,
    }
    if mongo_status is not None:
        dependencies["mongo"] = mongo_status
    if dynamo_status is not None:
        dependencies["dynamodb"] = dynamo_status
    if bedrock_status is not None:
        dependencies["bedrock"] = bedrock_status

    vector_degraded = (
        vector_health["pinecone"] in ("fallback_chroma", "down")
        or (not pine_configured and chroma_status not in ("ok", "disabled"))
    )
    # When chroma is intentionally disabled, only pinecone failure degrades vector
    if chroma_status == "disabled":
        vector_degraded = pine_configured and vector_health["pinecone"] != "ok"

    llm_degraded = (groq.is_configured and not groq_ok) or (not groq.is_configured and not ollama_ok)
    mission_degraded = not mission_ok
    # NetworkX embedded fallback is healthy (AWS / offline); only live Neo4j outage when enabled degrades
    neo_degraded = False
    degraded = mission_degraded or neo_degraded or vector_degraded or llm_degraded
    unavailable = (not vector_ok) and (not mission_ok)
    payload = {
        "status": "unavailable" if unavailable else ("degraded" if degraded else "ok"),
        "dependencies": dependencies,
        "mission_store": mode,
        "version": config.system.version,
        "engine": kg.get_summary().get("engine"),
        "vector_engine": vector_health.get("engine"),
    }
    status_code = 503 if payload["status"] == "unavailable" else 200
    return status_code, payload


@app.get("/healthz")
async def healthz():
    now = time.monotonic()
    cached = _HEALTH_CACHE.get("payload")
    if cached is not None and (now - float(_HEALTH_CACHE["ts"])) < _HEALTH_TTL_S:
        code = int(_HEALTH_CACHE["status_code"])
        if code != 200:
            return JSONResponse(status_code=code, content=cached)
        return cached

    status_code, payload = await asyncio.to_thread(_compute_health)
    _HEALTH_CACHE["ts"] = now
    _HEALTH_CACHE["payload"] = payload
    _HEALTH_CACHE["status_code"] = status_code
    if status_code != 200:
        return JSONResponse(status_code=status_code, content=payload)
    return payload


@app.post("/telemetry")
def receive_telemetry(telemetry: UAVTelemetry):
    """
    Ingests live UAV telemetry, routes through the LangGraph cognitive pipeline
    (situation → hybrid memory → planner → safety → learning), and returns a
    safety-validated recovery action.
    """
    try:
        result = get_graph().run(telemetry)
        return _pipeline_payload(telemetry, result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def get_system_status():
    g = get_graph()
    snapshot = g.working_memory.get_snapshot()
    kg_summary = g.memory_agent.knowledge_graph.get_summary()
    return {
        "status": "HEALTHY",
        "engine": kg_summary.get("engine"),
        "working_memory": snapshot,
        "kg_summary": kg_summary,
    }


@app.get("/memory")
def get_memory_state():
    g = get_graph()
    rules = [r.model_dump() for r in g.memory_agent.semantic_memory.get_all_rules()]
    kg_summary = g.memory_agent.knowledge_graph.get_summary()
    chroma_count = g.memory_agent.episodic_memory.vector_store.count()
    vector_engine = getattr(g.memory_agent.episodic_memory.vector_store, "engine", "chroma")
    last = g.memory_agent.last_context
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
            "graph_paths": (last.graph_paths if last else []) or [],
        },
    }


@app.get("/missions")
async def get_missions_history(
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
):
    store = get_store()
    history = await store.list_missions(limit=limit, skip=skip)
    total = await store.count_missions()
    for item in history:
        if "_id" in item:
            item["_id"] = str(item["_id"])
    return {
        "total_events": total,
        "limit": limit,
        "skip": skip,
        "history": history,
        "persisted": store.available,
    }


@app.post("/simulate")
def trigger_simulation(req: SimulateRequest):
    if req.scenario not in FailureScenarioInjector.list_available_scenarios():
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario. Available: {FailureScenarioInjector.list_available_scenarios()}"
        )

    sim = get_simulator()
    if req.steps == 1:
        step_res = sim.run_step(scenario=req.scenario)
        return {"result": step_res}
    results = sim.run_full_mission(
        total_steps=req.steps,
        inject_step=req.inject_step,
        scenario=req.scenario
    )
    return {"total_steps": len(results), "results": results}


@app.post("/reset")
def reset_state():
    get_graph().working_memory.clear()
    get_simulator().reset()
    return {"status": "SUCCESS", "message": "Working memory and simulation reset successfully"}
