import json
import time
from typing import List, Dict, Any, Optional
from models import EpisodicExperience
from memory.vector_store import VectorStore

class EpisodicMemory:
    """
    Episodic Memory Layer:
    Indexes full mission experiences into the vector store (Pinecone, Chroma fallback).
    Generates embeddings and performs semantic similarity retrieval to find past
    missions facing similar anomalies and environmental contexts.
    """
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        if self.vector_store.count() == 0:
            self._seed_default_experiences()

    def _seed_default_experiences(self) -> None:
        default_episodes = [
            EpisodicExperience(
                mission_id="MISSION_PAST_001",
                failure="GPS_INTERFERENCE",
                context="GPS degraded during cruise at altitude 120m in moderate winds (7 m/s)",
                environmental_conditions={"wind_speed": 7.2, "altitude": 120.0, "battery_pct": 74.0},
                action="SWITCH_TO_VIO_DEAD_RECKONING",
                outcome="mission_completed",
                success=True,
                mission_duration=48.0,
                confidence=0.92,
                timestamp=time.time() - 86400
            ),
            EpisodicExperience(
                mission_id="MISSION_PAST_002",
                failure="GPS_LOSS",
                context="Total GPS loss during en-route navigation with high satellites drop",
                environmental_conditions={"wind_speed": 5.0, "altitude": 90.0, "battery_pct": 60.0},
                action="INERTIAL_DEAD_RECKONING_SAFE_RTH",
                outcome="mission_aborted_safe_rth",
                success=True,
                mission_duration=32.0,
                confidence=0.88,
                timestamp=time.time() - 43200
            ),
            EpisodicExperience(
                mission_id="MISSION_PAST_003",
                failure="BATTERY_DEGRADATION",
                context="Rapid voltage sag below 14.0V while 2km away from base",
                environmental_conditions={"wind_speed": 8.5, "altitude": 110.0, "battery_pct": 28.0},
                action="POWER_CONSERVATIVE_RTH",
                outcome="safe_recovery_at_home",
                success=True,
                mission_duration=25.0,
                confidence=0.95,
                timestamp=time.time() - 21600
            ),
            EpisodicExperience(
                mission_id="MISSION_PAST_004",
                failure="LOW_BATTERY",
                context="Battery below critical 15% threshold with headwind",
                environmental_conditions={"wind_speed": 9.0, "altitude": 45.0, "battery_pct": 14.0},
                action="CONTROLLED_EMERGENCY_LAND",
                outcome="emergency_landing_success",
                success=True,
                mission_duration=12.0,
                confidence=0.90,
                timestamp=time.time() - 10800
            ),
            EpisodicExperience(
                mission_id="MISSION_PAST_005",
                failure="STRONG_WIND",
                context="Wind shear gusts exceeding 15 m/s causing severe attitude oscillations",
                environmental_conditions={"wind_speed": 16.2, "altitude": 130.0, "battery_pct": 65.0},
                action="REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT",
                outcome="attitude_stabilized_safe_transit",
                success=True,
                mission_duration=35.0,
                confidence=0.89,
                timestamp=time.time() - 5400
            ),
            EpisodicExperience(
                mission_id="MISSION_PAST_006",
                failure="COMMUNICATION_LOSS",
                context="Loss of ground control telemetry link for >15 seconds",
                environmental_conditions={"wind_speed": 4.0, "altitude": 80.0, "battery_pct": 82.0},
                action="AUTONOMOUS_FAILSAFE_HOLD_THEN_RTH",
                outcome="link_restored_safe_rth",
                success=True,
                mission_duration=20.0,
                confidence=0.91,
                timestamp=time.time() - 3600
            )
        ]
        for exp in default_episodes:
            self.store_experience(exp)

    def store_experience(self, exp: EpisodicExperience, episode_id: Optional[str] = None) -> str:
        doc_id = episode_id or exp.episode_id or f"{exp.mission_id}_{int(exp.timestamp)}"
        exp.episode_id = doc_id
        doc_text = f"Failure: {exp.failure}. Context: {exp.context}. Action: {exp.action}. Outcome: {exp.outcome}. Conditions: wind={exp.environmental_conditions.get('wind_speed')}m/s, alt={exp.environmental_conditions.get('altitude')}m, bat={exp.environmental_conditions.get('battery_pct')}%"
        
        metadata = {
            "episode_id": doc_id,
            "mission_id": exp.mission_id,
            "failure": exp.failure,
            "action": exp.action,
            "outcome": exp.outcome,
            "success": exp.success,
            "duration": exp.mission_duration,
            "confidence": exp.confidence,
            "timestamp": exp.timestamp,
            "env_json": json.dumps(exp.environmental_conditions)
        }
        
        self.vector_store.add_documents(
            ids=[doc_id],
            documents=[doc_text],
            metadatas=[metadata]
        )
        return doc_id

    def retrieve_similar_experiences(self, query_context: str, top_k: int = 3) -> List[Dict[str, Any]]:
        results = self.vector_store.query(query_context, n_results=top_k)
        experiences = []
        
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for i in range(len(ids)):
            meta = metadatas[i]
            dist = distances[i] if i < len(distances) else 0.5
            # Cosine distance to similarity: sim = 1.0 - dist
            # Bound similarity to [0.0, 1.0]
            similarity = max(0.0, min(1.0, 1.0 - dist))
            
            env_cond = {}
            if "env_json" in meta:
                try:
                    env_cond = json.loads(meta["env_json"])
                except Exception:
                    pass
                    
            exp = EpisodicExperience(
                episode_id=ids[i],
                mission_id=meta.get("mission_id", ids[i]),
                failure=meta.get("failure", "UNKNOWN"),
                context=documents[i],
                environmental_conditions=env_cond,
                action=meta.get("action", "CONTINUE_MISSION"),
                outcome=meta.get("outcome", "UNKNOWN"),
                success=bool(meta.get("success", True)),
                mission_duration=float(meta.get("duration", 0.0)),
                confidence=float(meta.get("confidence", 0.85)),
                timestamp=float(meta.get("timestamp", time.time()))
            )
            experiences.append({
                "episode_id": ids[i],
                "action": exp.action,
                "experience": exp,
                "vector_similarity": round(similarity, 4),
                "distance": round(dist, 4)
            })
        return experiences
