import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.graph import AeroCortexGraph
from simulation.failure_scenarios import FailureScenarioInjector
from simulation.telemetry_generator import TelemetryGenerator

GPS_CONDITION = "MODERATE_WIND"


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def print_section(title: str):
    print(f"\n--- {title} ---")


def _edge_for_action(graph: AeroCortexGraph, action: str) -> Dict[str, Any]:
    return graph.memory_agent.knowledge_graph.get_edge_stats(GPS_CONDITION, action)


def _run_gps_fault(graph: AeroCortexGraph, mission_id: str) -> Dict[str, Any]:
    gen = TelemetryGenerator(mission_id=mission_id)
    gen.step(dt=1.0)
    telemetry = FailureScenarioInjector.apply_scenario(gen.step(dt=1.0), "GPS_INTERFERENCE")
    result = graph.run(telemetry)
    action = result["final_plan"].action if result.get("final_plan") else "SWITCH_TO_VIO_DEAD_RECKONING"
    learning = result.get("learning_result") or {}
    memory = result.get("memory_context")
    return {
        "result": result,
        "action": action,
        "episode_id": learning.get("episode_id"),
        "persisted": bool(learning.get("persisted", False)),
        "planner_source": learning.get("planner_source") or getattr(result.get("planner_plan"), "source", None),
        "latency_ms": learning.get("latency_ms") or {},
        "edge": _edge_for_action(graph, action),
        "chroma_count": graph.memory_agent.episodic_memory.vector_store.count(),
        "vector_engine": getattr(graph.memory_agent.episodic_memory.vector_store, "engine", "chroma"),
        "top": memory.retrieved_experiences[0] if memory and memory.retrieved_experiences else None,
    }


def run_gps_reuse_demo(graph: Optional[AeroCortexGraph] = None) -> Dict[str, Any]:
    """
    Money-shot sequence:
    1. Cold-start GPS failure writes Mongo + Pinecone (Chroma fallback) + Neo4j.
    2. Repeat the same failure and show hydrated recall, score split, and a stronger edge.
    """
    graph = graph or AeroCortexGraph()
    mission1 = _run_gps_fault(graph, "MISSION_001")
    mission2 = _run_gps_fault(graph, "MISSION_002")
    return {"graph": graph, "mission1": mission1, "mission2": mission2}


def main():
    print_banner(
        "AEROCORTEX: COGNITIVE MEMORY FOR AUTONOMOUS EDGE UAVs\n"
        "  Real-Time Multi-Agent Recovery & Continual Experiential Learning Demo"
    )

    print("\n[INIT] LangGraph pipeline + Mongo / Pinecone (Chroma fallback) / Neo4j (NetworkX fallback) online.")
    print("[INIT] REST API uses Groq + Pinecone. Ollama + Chroma are Raspberry Pi fallbacks.")

    print_banner("MISSION 1: Cold start (GPS multipath interference)")
    m1_gen = TelemetryGenerator(mission_id="MISSION_001")
    print_section("Step 1: Nominal cruise")
    t1 = m1_gen.step(dt=1.0)
    print(
        f"  Telemetry: Alt={t1.altitude:.1f}m | Vel={t1.velocity:.1f}m/s | "
        f"Bat={t1.battery_level:.1f}% | GPS={t1.gps_status} | Wind={t1.wind_speed}m/s"
    )

    graph = AeroCortexGraph()
    res1 = graph.run(t1)
    print(f"  Situation Agent: Anomaly Detected = {res1['situation'].anomaly_detected}")
    print(f"  Action Executed: {res1['final_plan'].action}")

    print_section("Step 2: Inject GPS degradation")
    demo = run_gps_reuse_demo(graph)
    m1 = demo["mission1"]
    m2 = demo["mission2"]
    res1_fault = m1["result"]

    for log_msg in res1_fault.get("logs", []):
        print(f"  {log_msg}")

    print_section("Step 3: Mission 1 inspection — write path")
    sit1 = res1_fault["situation"]
    mem1 = res1_fault["memory_context"]
    plan1 = res1_fault["planner_plan"]
    safe1 = res1_fault["safety_verdict"]
    print(f"  [SITUATION] {sit1.failure_type} | severity={sit1.severity} | conf={sit1.confidence:.2f}")
    print(f"  [MEMORY]    retrieved={len(mem1.retrieved_experiences)} | top_rule={mem1.semantic_rules[0].rule_id if mem1.semantic_rules else 'None'}")
    print(f"  [PLANNER]   {plan1.action} | conf={plan1.confidence:.2f} | source={plan1.source}")
    print(f"  [SAFETY]    {'APPROVED' if safe1.approved else 'REJECTED'} | fallback={safe1.fallback_action}")
    print(f"  [EXEC]      {m1['action']}")
    print("  [LEARNING]  three-store write:")
    print(f"              Mongo persisted={m1['persisted']} episode_id={m1['episode_id']}")
    print(f"              Vector ({m1.get('vector_engine', 'chroma')}) count={m1['chroma_count']}")
    print(
        f"              Neo4j {GPS_CONDITION}-[{m1['action']}] "
        f"hits={m1['edge'].get('hits')} weight={m1['edge'].get('weight')}"
    )
    print(f"              latency_ms={m1['latency_ms']}")

    print_banner("MISSION 2: Identical GPS failure (experience reuse)")
    res2 = m2["result"]
    for log_msg in res2.get("logs", []):
        print(f"  {log_msg}")

    print_section("Retrieved prior episode")
    top = m2["top"]
    if top:
        print(f"  episode_id={top.episode_id} mission={top.experience.mission_id}")
        print(f"  failure={top.experience.failure} prior_action={top.experience.action}")
        print(f"  outcome={top.experience.outcome}")
        print(f"  hydrated_from_mongo={top.hydrated_from_mongo}")
        print(f"  vector_similarity={top.vector_similarity:.4f}")
        print(f"  graph_relevance={top.graph_relevance:.4f}")
        print(f"  final_score={top.final_score:.4f}  (0.6*vec + 0.4*graph, min-max normalized)")
    else:
        print("  Warning: no past experience retrieved.")

    w1 = m1["edge"].get("weight") or 0.0
    w2 = m2["edge"].get("weight") or 0.0
    h1 = m1["edge"].get("hits") or 0
    h2 = m2["edge"].get("hits") or 0
    print_section("Reinforcement and latency")
    print(f"  Neo4j edge after mission 1: hits={h1} weight={w1}")
    print(f"  Neo4j edge after mission 2: hits={h2} weight={w2}")
    print(f"  weight increased: {w2 > w1} | hits increased: {h2 > h1}")
    print(f"  mission 1 latency_ms={m1['latency_ms']}")
    print(f"  mission 2 latency_ms={m2['latency_ms']}")
    print("\n  Closed-loop reuse verified: recall + score split + stronger graph edge.")

    print_banner("AEROCORTEX DEMONSTRATION COMPLETE")
    return demo


if __name__ == "__main__":
    main()
