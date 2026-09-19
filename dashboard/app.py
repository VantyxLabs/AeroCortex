import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import config
from models import UAVTelemetry
from orchestration.graph import AeroCortexGraph
from simulation.mission_simulator import MissionSimulator
from simulation.failure_scenarios import FailureScenarioInjector

st.set_page_config(
    page_title="AeroCortex - UAV Cognitive Architecture",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .status-badge-ok {
        background-color: #10B981;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .status-badge-warn {
        background-color: #F59E0B;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .status-badge-danger {
        background-color: #EF4444;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "graph" not in st.session_state:
    st.session_state.graph = AeroCortexGraph()
    st.session_state.simulator = MissionSimulator(graph=st.session_state.graph)
    st.session_state.telemetry_history = []
    st.session_state.last_result = None

graph = st.session_state.graph
simulator = st.session_state.simulator

# Sidebar Controls
st.sidebar.title("🛸 AeroCortex Control")
st.sidebar.markdown("**Experience-Aware Edge UAV Cognitive Layer**")
st.sidebar.divider()

scenario = st.sidebar.selectbox(
    "Select Failure Scenario:",
    FailureScenarioInjector.list_available_scenarios(),
    index=0
)

col_b1, col_b2 = st.sidebar.columns(2)
run_step_btn = col_b1.button("▶ Step Mission", use_container_width=True)
reset_btn = col_b2.button("🔄 Reset", use_container_width=True)

if reset_btn:
    graph.working_memory.clear()
    simulator.reset(new_mission_id=f"MISSION_{int(time.time())%1000:03d}")
    st.session_state.telemetry_history = []
    st.session_state.last_result = None
    st.sidebar.success("Mission state reset!")
    st.rerun()

if run_step_btn:
    res = simulator.run_step(scenario=scenario)
    st.session_state.last_result = res
    st.session_state.telemetry_history.append(res["telemetry"])
    if len(st.session_state.telemetry_history) > 100:
        st.session_state.telemetry_history.pop(0)

# Main Title & Operational Mode
st.title("🛸 AeroCortex: Cognitive Memory Flight Control")
st.caption(f"Target: {config.system.device_target.upper()} | Mode: OFFLINE-EDGE | LLM: {config.llm.model}")

# Top Header KPI Cards: Section 1 (Mission Status)
latest_t = None
if st.session_state.telemetry_history:
    latest_t = st.session_state.telemetry_history[-1]
else:
    # Dummy starter telemetry
    latest_t = simulator.generator.step().model_dump()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Mission ID", latest_t.get("mission_id", "MISSION_001"))
col2.metric("Flight State", latest_t.get("mission_state", "CRUISE"))
col3.metric("Altitude (m)", f"{latest_t.get('altitude', 120.0):.1f}")
col4.metric("Velocity (m/s)", f"{latest_t.get('velocity', 12.5):.1f}")
col5.metric("Battery (%)", f"{latest_t.get('battery_level', 95.0):.1f}%", f"{latest_t.get('battery_voltage', 16.4):.2f}V")
col6.metric("Wind (m/s)", f"{latest_t.get('wind_speed', 6.5):.1f}")

st.divider()

# Left Column: Situation & Planner & Safety
# Right Column: Memory Retrieval & Knowledge Graph & Learning
left_col, right_col = st.columns([1, 1])

last_res = st.session_state.last_result
situation_data = last_res.get("situation") if last_res else None
planner_data = last_res.get("planner_plan") if last_res else None
safety_data = last_res.get("safety_verdict") if last_res else None
final_data = last_res.get("final_plan") if last_res else None

with left_col:
    # Section 2: Failure Detection
    st.subheader("⚠️ Section 2: Failure Detection (Situation Agent)")
    if situation_data and situation_data.get("anomaly_detected"):
        sev = situation_data.get("severity", "LOW")
        color_class = "status-badge-danger" if sev in ("CRITICAL", "HIGH") else "status-badge-warn"
        st.markdown(f"""
        <div class="metric-card">
            <h4><span class="{color_class}">{situation_data.get('failure_type')}</span> - Severity: {sev}</h4>
            <p><strong>Confidence:</strong> {situation_data.get('confidence', 0.9):.2f} | <strong>Phase:</strong> {situation_data.get('mission_phase')}</p>
            <p><em>{situation_data.get('description')}</em></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <h4><span class="status-badge-ok">NOMINAL</span> - All Channels Normal</h4>
            <p>Deterministic rules report zero flight envelope threshold violations.</p>
        </div>
        """, unsafe_allow_html=True)

    # Section 5: Planner Agent
    st.subheader("🧠 Section 5: Planner Agent (Local Gemma 3)")
    if planner_data:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Proposed Action: <code>{planner_data.get('action')}</code></h4>
            <p><strong>Reasoning:</strong> {planner_data.get('reason')}</p>
            <p><strong>Steps:</strong> {', '.join(planner_data.get('steps', []))}</p>
            <p><strong>Confidence:</strong> {planner_data.get('confidence', 0.85):.2f} | <strong>Risk Level:</strong> {planner_data.get('risk_level')}</p>
            <p><small>Source: {planner_data.get('source', 'Planner')}</small></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Execute mission steps to display local LLM recovery plans.")

    # Section 6: Safety Agent
    st.subheader("🛡️ Section 6: Safety Agent (Deterministic Gatekeeper)")
    if safety_data:
        is_appr = safety_data.get("approved", True)
        badge = '<span class="status-badge-ok">APPROVED</span>' if is_appr else '<span class="status-badge-danger">REJECTED</span>'
        st.markdown(f"""
        <div class="metric-card">
            <h4>Verdict: {badge}</h4>
            <p><strong>Evaluation:</strong> {safety_data.get('reason')}</p>
            <p><strong>Fallback Assigned:</strong> <code>{safety_data.get('fallback_action')}</code></p>
            <p><strong>Final Executed Action:</strong> <code>{final_data.get('action') if final_data else 'N/A'}</code></p>
            <p><strong>Violated Constraints:</strong> {safety_data.get('violated_constraints', []) or 'None'}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Safety agent awaits planner recommendation.")

with right_col:
    # Section 3: Memory Retrieval
    st.subheader("📚 Section 3: Hybrid Memory Retrieval")
    episodes = graph.memory_agent.episodic_memory.retrieve_similar_experiences(
        query_context=f"Failure: {scenario} Altitude: {latest_t.get('altitude')}m",
        top_k=2
    )
    if episodes:
        exp_table = []
        for e in episodes:
            exp = e["experience"]
            exp_table.append({
                "Mission ID": exp.mission_id,
                "Past Failure": exp.failure,
                "Action Taken": exp.action,
                "Outcome": exp.outcome,
                "Similarity": e["vector_similarity"]
            })
        st.dataframe(pd.DataFrame(exp_table), use_container_width=True, hide_index=True)
    else:
        st.info("No past experiences retrieved.")

    # Section 4: Knowledge Graph Context
    st.subheader("🕸️ Section 4: Knowledge Graph Relational Context")
    kg_paths = graph.memory_agent.knowledge_graph.query_action_relevance(scenario)
    if kg_paths:
        st.markdown("**Relational Graph Paths (Failure → Condition → Action → Outcome):**")
        for p in kg_paths[:3]:
            st.markdown(f"- **Action:** `{p['action']}` | **Graph Relevance:** `{p['graph_relevance']:.3f}`")
    else:
        st.write("Graph paths nominal.")

    # Section 7: Continual Learning
    st.subheader("📈 Section 7: Continual Learning Agent")
    kg_sum = graph.memory_agent.knowledge_graph.get_summary()
    vs = graph.memory_agent.episodic_memory.vector_store
    chroma_cnt = vs.count()
    rules_cnt = len(graph.memory_agent.semantic_memory.get_all_rules())
    
    c_l1, c_l2, c_l3 = st.columns(3)
    c_l1.metric("Episodic Experiences", chroma_cnt)
    c_l2.metric("Semantic Rules", rules_cnt)
    c_l3.metric("KG Nodes / Edges", f"{kg_sum['nodes_count']} / {kg_sum['edges_count']}")
    st.caption(f"Vector engine: {getattr(vs, 'engine', 'chroma')}")

# Section 8: Live Telemetry Charts
st.divider()
st.subheader("📊 Section 8: Live Mission Telemetry Streaming")

if st.session_state.telemetry_history:
    df_history = pd.DataFrame(st.session_state.telemetry_history)
    df_history["step"] = range(1, len(df_history) + 1)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_bat = px.line(df_history, x="step", y="battery_level", title="Battery Level (%)", markers=True)
        fig_bat.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_bat, use_container_width=True)
        
    with c2:
        fig_alt = px.line(df_history, x="step", y=["altitude", "velocity"], title="Altitude (m) & Velocity (m/s)", markers=True)
        fig_alt.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_alt, use_container_width=True)
        
    with c3:
        fig_wind = px.line(df_history, x="step", y=["wind_speed", "gps_accuracy"], title="Wind (m/s) & GPS HDOP (m)", markers=True)
        fig_wind.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_wind, use_container_width=True)
else:
    st.info("No telemetry history yet. Click 'Step Mission' to stream live telemetry.")

# System Logs Drawer
with st.expander("📝 Real-Time System Orchestration Logs", expanded=False):
    if last_res and last_res.get("logs"):
        for l in last_res["logs"]:
            st.text(l)
    else:
        st.text("AeroCortex cognitive system initialized. Ready for mission step.")
