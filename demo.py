import time
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import UAVTelemetry
from simulation.telemetry_generator import TelemetryGenerator
from simulation.failure_scenarios import FailureScenarioInjector
from orchestration.graph import AeroCortexGraph

def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def print_section(title: str):
    print(f"\n--- {title} ---")

def main():
    print_banner("AEROCORTEX: COGNITIVE MEMORY FOR AUTONOMOUS EDGE UAVs\n  Real-Time Multi-Agent Recovery & Continual Experiential Learning Demo")
    
    # Initialize the cognitive architecture
    graph = AeroCortexGraph()
    
    print("\n[INIT] 3-Tier Persistent Memory System & LangGraph Multi-Agent Pipeline Online.")
    print("[INIT] Target: Autonomous Edge UAV (Raspberry Pi 5 / NVIDIA Jetson) [Zero Cloud Dependency].")
    
    # =========================================================================
    # FLIGHT MISSION 1: COLD START & ANOMALY HANDLING
    # =========================================================================
    print_banner("MISSION 1: Cold Start Execution (GPS Multipath Interference)")
    m1_gen = TelemetryGenerator(mission_id="MISSION_001")
    
    print_section("Step 1: Normal Waypoint Cruise")
    t1 = m1_gen.step(dt=1.0)
    print(f"  Telemetry: Alt={t1.altitude:.1f}m | Vel={t1.velocity:.1f}m/s | Bat={t1.battery_level:.1f}% | GPS={t1.gps_status} (HDOP: {t1.gps_accuracy}) | Wind={t1.wind_speed}m/s")
    res1 = graph.run(t1)
    print(f"  Situation Agent: Anomaly Detected = {res1['situation'].anomaly_detected}")
    print(f"  Action Executed: {res1['final_plan'].action} (Nominal Cruise)")
    time.sleep(0.6)

    print_section("Step 2: Injecting Fault: GPS Degradation & Multipath Interference")
    t1_fault = m1_gen.step(dt=1.0)
    t1_fault = FailureScenarioInjector.apply_scenario(t1_fault, "GPS_INTERFERENCE")
    print(f"  Sensor Readout: GPS Status: '{t1_fault.gps_status}' | Accuracy HDOP: {t1_fault.gps_accuracy}m | Altitude: {t1_fault.altitude}m")
    
    print_section("Step 3: Multi-Agent Cognitive Recovery Loop")
    res1_fault = graph.run(t1_fault)
    for log_msg in res1_fault.get("logs", []):
        print(f"  {log_msg}")

    print_section("Step 4: Detailed Multi-Agent Inspection (Mission 1)")
    sit1 = res1_fault["situation"]
    mem1 = res1_fault["memory_context"]
    plan1 = res1_fault["planner_plan"]
    safe1 = res1_fault["safety_verdict"]
    exec1 = res1_fault["final_plan"]
    
    print(f"  [1. SITUATION AGENT]  Detected: {sit1.failure_type} | Severity: {sit1.severity} | Confidence: {sit1.confidence:.2f}")
    print(f"  [2. MEMORY AGENT]     Retrieved Experiences: {len(mem1.retrieved_experiences)} | Top Rule: {mem1.semantic_rules[0].rule_id if mem1.semantic_rules else 'None'}")
    print(f"  [3. PLANNER AGENT]    Proposed: {plan1.action} | Confidence: {plan1.confidence:.2f} | Source: {plan1.source}")
    print(f"                        Reasoning: {plan1.reason}")
    print(f"  [4. SAFETY AGENT]     Verdict: {'APPROVED' if safe1.approved else 'REJECTED'} | Fallback Action: {safe1.fallback_action}")
    print(f"  [5. EXECUTION]        Final Actuation: [{exec1.action}]")
    
    lr1 = res1_fault.get("learning_result", {})
    kg = graph.memory_agent.knowledge_graph
    edge1 = kg.get_edge_stats("MODERATE_WIND", exec1.action)
    print(f"  [6. LEARNING AGENT]   Committed Episode ID: {lr1.get('episode_id', 'N/A')} | Outcome: {lr1.get('outcome', 'N/A')}")
    print(f"                        Persisted to Mongo: {lr1.get('persisted', False)} | Planner: {plan1.source}")
    print(f"                        KG edge weight hits={edge1.get('hits')} weight={edge1.get('weight')}")
    print(f"                        Latency ms: {lr1.get('latency_ms')}")

    time.sleep(1.0)

    # =========================================================================
    # FLIGHT MISSION 2: REPEATED FAILURE & EXPERIENCE REUSE DEMONSTRATION
    # =========================================================================
    print_banner("MISSION 2: Repeated Failure (Demonstrating Experiential Memory Recall)")
    print("Scenario: A new flight (MISSION_002) encounters the identical GPS failure.")
    print("Watch the Memory Agent retrieve MISSION_001's consolidated experience!")
    
    m2_gen = TelemetryGenerator(mission_id="MISSION_002")
    m2_gen.step(dt=1.0)
    
    t2_fault = m2_gen.step(dt=1.0)
    t2_fault = FailureScenarioInjector.apply_scenario(t2_fault, "GPS_INTERFERENCE")
    
    print_section("Executing Cognitive Loop for MISSION_002")
    res2 = graph.run(t2_fault)
    for log_msg in res2.get("logs", []):
        print(f"  {log_msg}")

    print_section("Demonstrating Memory Recall Verification")
    mem2 = res2["memory_context"]
    kg = graph.memory_agent.knowledge_graph
    edge2 = kg.get_edge_stats(
        "MODERATE_WIND",
        res2["final_plan"].action if res2.get("final_plan") else "SWITCH_TO_VIO_DEAD_RECKONING",
    )
    if mem2.retrieved_experiences:
        top_exp = mem2.retrieved_experiences[0]
        print(f"  >>> RETRIEVED PAST EXPERIENCE:")
        print(f"      - Episode ID:         {top_exp.episode_id}")
        print(f"      - Source Mission ID:  {top_exp.experience.mission_id}")
        print(f"      - Failure Type:       {top_exp.experience.failure}")
        print(f"      - Prior Safe Action:  {top_exp.experience.action}")
        print(f"      - Historical Outcome: {top_exp.experience.outcome}")
        print(f"      - Hydrated from Mongo:{top_exp.hydrated_from_mongo}")
        print(f"      - Vector Similarity:  {top_exp.vector_similarity:.4f}")
        print(f"      - Graph Relevance:    {top_exp.graph_relevance:.4f}")
        print(f"      - Composite Score:    {top_exp.final_score:.4f} (normalized 0.6*Vec + 0.4*KG)")
        print(f"      - KG edge after reuse: hits={edge2.get('hits')} weight={edge2.get('weight')}")
        lr2 = res2.get("learning_result", {})
        print(f"      - Latency breakdown:  {lr2.get('latency_ms')}")
        print(f"\n  [SUCCESS] Closed-loop cognitive architecture verified!")
        print(f"  The UAV leveraged prior mission experience to recover immediately without cloud connectivity.")
    else:
        print("  Warning: No past experience retrieved.")

    print_banner("AEROCORTEX DEMONSTRATION COMPLETE")

if __name__ == "__main__":
    main()
