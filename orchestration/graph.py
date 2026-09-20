import time
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from models import (
    UAVTelemetry, SituationReport, HybridMemoryContext,
    RecoveryPlan, SafetyVerdict, MissionOutcome
)
from orchestration.state import AeroCortexState
from agents.situation_agent import SituationAgent
from agents.memory_agent import MemoryAgent
from agents.planner_agent import PlannerAgent
from agents.safety_agent import SafetyAgent
from agents.learning_agent import LearningAgent
from memory.working_memory import WorkingMemory

class AeroCortexGraph:
    """
    AeroCortex LangGraph State Machine.
    Orchestrates the closed-loop cognitive flow across all 5 specialized agents
    with deterministic routing and fallback enforcement.
    """
    def __init__(
        self,
        situation_agent: Optional[SituationAgent] = None,
        memory_agent: Optional[MemoryAgent] = None,
        planner_agent: Optional[PlannerAgent] = None,
        safety_agent: Optional[SafetyAgent] = None,
        learning_agent: Optional[LearningAgent] = None,
        working_memory: Optional[WorkingMemory] = None
    ):
        self.situation_agent = situation_agent or SituationAgent()
        self.memory_agent = memory_agent or MemoryAgent()
        self.planner_agent = planner_agent or PlannerAgent()
        self.safety_agent = safety_agent or SafetyAgent()
        self.learning_agent = learning_agent or LearningAgent(
            episodic_memory=self.memory_agent.episodic_memory,
            semantic_memory=self.memory_agent.semantic_memory,
            knowledge_graph=self.memory_agent.knowledge_graph,
        )
        if working_memory is not None:
            self.working_memory = working_memory
        else:
            try:
                from cloud.factories import get_working_memory

                self.working_memory = get_working_memory()
            except Exception:
                self.working_memory = WorkingMemory()
        
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AeroCortexState)

        # 1. Add Nodes
        builder.add_node("situation_node", self._situation_step)
        builder.add_node("continue_monitoring_node", self._continue_monitoring_step)
        builder.add_node("memory_node", self._memory_step)
        builder.add_node("planner_node", self._planner_step)
        builder.add_node("safety_node", self._safety_step)
        builder.add_node("recovery_node", self._recovery_execution_step)
        builder.add_node("fallback_node", self._fallback_step)
        builder.add_node("outcome_node", self._outcome_step)
        builder.add_node("learning_node", self._learning_step)

        # 2. Add Edges & Conditional Routers
        builder.add_edge(START, "situation_node")
        
        # Router after situation assessment
        builder.add_conditional_edges(
            "situation_node",
            self._route_anomaly,
            {
                "continue": "continue_monitoring_node",
                "escalate": "memory_node"
            }
        )
        builder.add_edge("continue_monitoring_node", END)

        # Anomaly handling pipeline
        builder.add_edge("memory_node", "planner_node")
        builder.add_edge("planner_node", "safety_node")

        # Router after safety check
        builder.add_conditional_edges(
            "safety_node",
            self._route_safety,
            {
                "approved": "recovery_node",
                "rejected": "fallback_node"
            }
        )

        builder.add_edge("recovery_node", "outcome_node")
        builder.add_edge("fallback_node", "outcome_node")
        builder.add_edge("outcome_node", "learning_node")
        builder.add_edge("learning_node", END)

        return builder.compile()

    # --- Node Callbacks ---

    def _situation_step(self, state: AeroCortexState) -> Dict[str, Any]:
        telemetry = state["telemetry"]
        self.working_memory.update_telemetry(telemetry)
        
        report = self.situation_agent.assess_situation(telemetry)
        self.working_memory.set_anomaly(report)
        
        logs = state.get("logs", [])
        if report.anomaly_detected:
            logs.append(f"[{time.strftime('%H:%M:%S')}] [SITUATION] {report.failure_type} detected (Severity: {report.severity})")
        else:
            logs.append(f"[{time.strftime('%H:%M:%S')}] [SITUATION] Telemetry nominal")
            
        return {"situation": report, "logs": logs}

    def _route_anomaly(self, state: AeroCortexState) -> str:
        report = state["situation"]
        return "escalate" if report.anomaly_detected else "continue"

    def _continue_monitoring_step(self, state: AeroCortexState) -> Dict[str, Any]:
        nominal_plan = RecoveryPlan(
            action="CONTINUE_MISSION",
            reason="Telemetry within nominal boundaries",
            confidence=1.0,
            risk_level="LOW"
        )
        self.working_memory.set_recovery_plan(nominal_plan)
        return {
            "final_plan": nominal_plan,
            "execution_status": "CONTINUE"
        }

    def _memory_step(self, state: AeroCortexState) -> Dict[str, Any]:
        telemetry = state["telemetry"]
        situation = state["situation"]
        
        context = self.memory_agent.retrieve_context(telemetry, situation)
        logs = state.get("logs", [])
        logs.append(f"[{time.strftime('%H:%M:%S')}] [MEMORY] Retrieved {len(context.retrieved_experiences)} past experiences (Top: {context.top_recommended_action})")
        return {"memory_context": context, "logs": logs}

    def _planner_step(self, state: AeroCortexState) -> Dict[str, Any]:
        telemetry = state["telemetry"]
        situation = state["situation"]
        memory_context = state["memory_context"]
        
        plan = self.planner_agent.plan_recovery(telemetry, situation, memory_context)
        self.working_memory.set_recovery_plan(plan)
        
        logs = state.get("logs", [])
        logs.append(f"[{time.strftime('%H:%M:%S')}] [PLANNER] Plan generated: {plan.action} (Confidence: {plan.confidence:.2f}, Source: {plan.source})")
        return {"planner_plan": plan, "logs": logs}

    def _safety_step(self, state: AeroCortexState) -> Dict[str, Any]:
        telemetry = state["telemetry"]
        situation = state["situation"]
        planner_plan = state["planner_plan"]
        
        verdict, final_plan = self.safety_agent.validate_plan(telemetry, situation, planner_plan)
        self.working_memory.set_safety_status(verdict)
        self.working_memory.set_recovery_plan(final_plan)
        
        logs = state.get("logs", [])
        status_tag = "APPROVED" if verdict.approved else "REJECTED (FALLBACK)"
        logs.append(f"[{time.strftime('%H:%M:%S')}] [SAFETY] {status_tag} - Final Action: {final_plan.action}")
        return {
            "safety_verdict": verdict,
            "final_plan": final_plan,
            "logs": logs
        }

    def _route_safety(self, state: AeroCortexState) -> str:
        verdict = state["safety_verdict"]
        return "approved" if verdict.approved else "rejected"

    def _recovery_execution_step(self, state: AeroCortexState) -> Dict[str, Any]:
        plan = state["final_plan"]
        logs = state.get("logs", [])
        logs.append(f"[{time.strftime('%H:%M:%S')}] [EXECUTION] Executing validated recovery action: {plan.action}")
        return {"execution_status": "RECOVERED", "logs": logs}

    def _fallback_step(self, state: AeroCortexState) -> Dict[str, Any]:
        plan = state["final_plan"]
        logs = state.get("logs", [])
        logs.append(f"[{time.strftime('%H:%M:%S')}] [FALLBACK] Executing deterministic fallback action: {plan.action}")
        return {"execution_status": "FALLBACK_TRIGGERED", "logs": logs}

    def _outcome_step(self, state: AeroCortexState) -> Dict[str, Any]:
        telemetry = state["telemetry"]
        situation = state["situation"]
        plan = state["final_plan"]
        verdict = state["safety_verdict"]
        
        # Flight outcome evaluation
        success = True
        time_to_stabilize = 15.0 # seconds
        if situation.severity == "CRITICAL" and not verdict.approved:
            time_to_stabilize = 25.0
            
        outcome = MissionOutcome(
            mission_id=telemetry.mission_id,
            failure_type=situation.failure_type,
            action_executed=plan.action,
            safety_approved=verdict.approved,
            fallback_triggered=(not verdict.approved),
            success=success,
            final_battery=telemetry.battery_level - 1.5,
            time_to_stabilize_s=time_to_stabilize,
            summary=f"Mission stabilized via {plan.action}"
        )
        return {"mission_outcome": outcome}

    def _learning_step(self, state: AeroCortexState) -> Dict[str, Any]:
        telemetry = state["telemetry"]
        situation = state["situation"]
        plan = state["final_plan"]
        verdict = state["safety_verdict"]
        outcome = state["mission_outcome"]
        logs = state.get("logs", [])

        latency_ms = {
            "retrieval": (state.get("memory_context").retrieval_latency_ms
                          if state.get("memory_context") else 0.0),
            "planning": (state.get("planner_plan").plan_latency_ms
                         if state.get("planner_plan") else 0.0),
            "safety": (state.get("safety_verdict").latency_ms
                       if state.get("safety_verdict") else 0.0),
        }

        try:
            from cloud.factories import learning_mode

            mode = learning_mode()
        except Exception:
            mode = "inline"

        if mode == "async" and situation.anomaly_detected and situation.failure_type != "NONE":
            import uuid as _uuid
            from cloud.sqs_episodes import publish_episode

            episode_id = str(_uuid.uuid4())
            # Trim retrieval context for SQS 256KB limit
            retrieved = []
            mc = state.get("memory_context")
            if mc:
                for item in (mc.retrieved_experiences or [])[:3]:
                    retrieved.append({
                        "episode_id": item.episode_id,
                        "action": item.experience.action,
                        "final_score": item.final_score,
                    })
            payload = {
                "episode_id": episode_id,
                "telemetry": telemetry.model_dump(),
                "situation": situation.model_dump(),
                "plan": plan.model_dump(),
                "safety_verdict": verdict.model_dump(),
                "success": outcome.success if outcome else True,
                "duration_s": outcome.time_to_stabilize_s if outcome else 20.0,
                "latency_ms": latency_ms,
                "retrieved_context": retrieved,
            }
            queued = publish_episode(payload, episode_id)
            learning_res = {
                "status": "QUEUED" if queued else "FAILED",
                "episode_id": episode_id,
                "persisted": False,
                "learning_mode": "async",
            }
            if queued:
                logs.append(f"[{time.strftime('%H:%M:%S')}] [LEARNING] Experience queued: {episode_id}")
            else:
                logs.append(f"[{time.strftime('%H:%M:%S')}] [LEARNING] Queue publish failed; falling back inline")
                mode = "inline"

        if mode != "async":
            learning_res = self.learning_agent.process_mission_outcome(
                telemetry=telemetry,
                situation=situation,
                plan=plan,
                verdict=verdict,
                success=outcome.success if outcome else True,
                duration_s=outcome.time_to_stabilize_s if outcome else 20.0,
                memory_context=state.get("memory_context"),
                latency_ms=latency_ms,
            )
            learning_res["learning_mode"] = learning_res.get("learning_mode") or "inline"
            if learning_res.get("status") == "SUCCESS":
                logs.append(
                    f"[{time.strftime('%H:%M:%S')}] [LEARNING] Experience committed: {learning_res.get('episode_id')}"
                )

        return {"learning_result": learning_res, "logs": logs}

    # --- Public Runner ---

    def run(self, telemetry: UAVTelemetry) -> AeroCortexState:
        initial_state: AeroCortexState = {
            "telemetry": telemetry,
            "logs": []
        }
        return self.graph.invoke(initial_state)
