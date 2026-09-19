import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import networkx as nx

from config import PROJECT_ROOT, config

logger = logging.getLogger("aerocortex.knowledge_graph")

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared ontology (NetworkX seed + Neo4j seed script)
# Node ids use "type:NAME" prefixes for the embedded graph.
# ---------------------------------------------------------------------------

DEFAULT_FAILURES = [
    ("failure:GPS_INTERFERENCE", {"type": "Failure", "name": "GPS_INTERFERENCE"}),
    ("failure:GPS_LOSS", {"type": "Failure", "name": "GPS_LOSS"}),
    ("failure:BATTERY_DEGRADATION", {"type": "Failure", "name": "BATTERY_DEGRADATION"}),
    ("failure:LOW_BATTERY", {"type": "Failure", "name": "LOW_BATTERY"}),
    ("failure:STRONG_WIND", {"type": "Failure", "name": "STRONG_WIND"}),
    ("failure:COMMUNICATION_LOSS", {"type": "Failure", "name": "COMMUNICATION_LOSS"}),
    ("failure:SENSOR_ANOMALY", {"type": "Failure", "name": "SENSOR_ANOMALY"}),
    ("failure:COMBINED_FAILURE", {"type": "Failure", "name": "COMBINED_FAILURE"}),
]

DEFAULT_CONDITIONS = [
    ("condition:MODERATE_WIND", {"type": "Condition", "name": "MODERATE_WIND"}),
    ("condition:HIGH_WIND", {"type": "Condition", "name": "HIGH_WIND"}),
    ("condition:CRITICAL_BATTERY", {"type": "Condition", "name": "CRITICAL_BATTERY"}),
    ("condition:RESERVE_BATTERY", {"type": "Condition", "name": "RESERVE_BATTERY"}),
    ("condition:HIGH_ALTITUDE", {"type": "Condition", "name": "HIGH_ALTITUDE"}),
]

DEFAULT_ACTIONS = [
    ("action:SWITCH_TO_VIO_DEAD_RECKONING", {"type": "RecoveryAction", "name": "SWITCH_TO_VIO_DEAD_RECKONING"}),
    ("action:INERTIAL_DEAD_RECKONING_SAFE_RTH", {"type": "RecoveryAction", "name": "INERTIAL_DEAD_RECKONING_SAFE_RTH"}),
    ("action:POWER_CONSERVATIVE_RTH", {"type": "RecoveryAction", "name": "POWER_CONSERVATIVE_RTH"}),
    ("action:CONTROLLED_EMERGENCY_LAND", {"type": "RecoveryAction", "name": "CONTROLLED_EMERGENCY_LAND"}),
    ("action:REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT", {"type": "RecoveryAction", "name": "REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT"}),
    ("action:AUTONOMOUS_FAILSAFE_HOLD_THEN_RTH", {"type": "RecoveryAction", "name": "AUTONOMOUS_FAILSAFE_HOLD_THEN_RTH"}),
    ("action:SWITCH_SECONDARY_IMU_STABILIZE", {"type": "RecoveryAction", "name": "SWITCH_SECONDARY_IMU_STABILIZE"}),
]

DEFAULT_OUTCOMES = [
    ("outcome:MISSION_SUCCESS", {"type": "Outcome", "name": "MISSION_SUCCESS", "weight": 1.0}),
    ("outcome:MISSION_ABORTED_SAFE", {"type": "Outcome", "name": "MISSION_ABORTED_SAFE", "weight": 0.8}),
    ("outcome:MISSION_FAILURE", {"type": "Outcome", "name": "MISSION_FAILURE", "weight": 0.0}),
]

# (src, dst, relation, weight)
DEFAULT_EDGES = [
    ("failure:GPS_INTERFERENCE", "condition:MODERATE_WIND", "OCCURRED_DURING", 0.8),
    ("condition:MODERATE_WIND", "action:SWITCH_TO_VIO_DEAD_RECKONING", "TRIGGERED", 0.9),
    ("action:SWITCH_TO_VIO_DEAD_RECKONING", "outcome:MISSION_SUCCESS", "RESULTED_IN", 0.95),
    ("failure:GPS_INTERFERENCE", "condition:HIGH_WIND", "OCCURRED_DURING", 0.7),
    ("condition:HIGH_WIND", "action:INERTIAL_DEAD_RECKONING_SAFE_RTH", "TRIGGERED", 0.85),
    ("action:INERTIAL_DEAD_RECKONING_SAFE_RTH", "outcome:MISSION_ABORTED_SAFE", "RESULTED_IN", 0.9),
    ("failure:GPS_LOSS", "action:INERTIAL_DEAD_RECKONING_SAFE_RTH", "RECOVERED_BY", 0.92),
    ("action:INERTIAL_DEAD_RECKONING_SAFE_RTH", "outcome:MISSION_ABORTED_SAFE", "RESULTED_IN", 0.95),
    ("failure:BATTERY_DEGRADATION", "condition:RESERVE_BATTERY", "OCCURRED_DURING", 0.9),
    ("condition:RESERVE_BATTERY", "action:POWER_CONSERVATIVE_RTH", "TRIGGERED", 0.92),
    ("action:POWER_CONSERVATIVE_RTH", "outcome:MISSION_SUCCESS", "RESULTED_IN", 0.93),
    ("failure:LOW_BATTERY", "condition:CRITICAL_BATTERY", "OCCURRED_DURING", 0.95),
    ("condition:CRITICAL_BATTERY", "action:CONTROLLED_EMERGENCY_LAND", "TRIGGERED", 0.98),
    ("action:CONTROLLED_EMERGENCY_LAND", "outcome:MISSION_ABORTED_SAFE", "RESULTED_IN", 0.96),
    ("failure:STRONG_WIND", "condition:HIGH_WIND", "OCCURRED_DURING", 0.95),
    ("condition:HIGH_WIND", "action:REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT", "TRIGGERED", 0.9),
    ("action:REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT", "outcome:MISSION_SUCCESS", "RESULTED_IN", 0.92),
    ("failure:COMMUNICATION_LOSS", "action:AUTONOMOUS_FAILSAFE_HOLD_THEN_RTH", "RECOVERED_BY", 0.91),
    ("action:AUTONOMOUS_FAILSAFE_HOLD_THEN_RTH", "outcome:MISSION_SUCCESS", "RESULTED_IN", 0.90),
    ("failure:COMBINED_FAILURE", "action:CONTROLLED_EMERGENCY_LAND", "RECOVERED_BY", 0.95),
    ("action:CONTROLLED_EMERGENCY_LAND", "outcome:MISSION_ABORTED_SAFE", "RESULTED_IN", 0.95),
]

NEO4J_CONSTRAINTS = [
    "CREATE CONSTRAINT mission_name IF NOT EXISTS FOR (m:Mission) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT failure_name IF NOT EXISTS FOR (f:Failure) REQUIRE f.name IS UNIQUE",
    "CREATE CONSTRAINT cond_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT action_name IF NOT EXISTS FOR (a:RecoveryAction) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT outcome_name IF NOT EXISTS FOR (o:Outcome) REQUIRE o.name IS UNIQUE",
]

_LABEL_FROM_PREFIX = {
    "mission": "Mission",
    "failure": "Failure",
    "condition": "Condition",
    "action": "RecoveryAction",
    "outcome": "Outcome",
}


def bare_name(node_id: str) -> str:
    return node_id.split(":", 1)[-1]


def _reinforce(weight: float, success: bool, lr: float) -> float:
    current = weight if weight is not None else 0.5
    if success:
        return current + lr * (1.0 - current)
    return current * (1.0 - lr)


@runtime_checkable
class GraphBackend(Protocol):
    def query_action_relevance(
        self, failure_type: str, condition: Optional[str] = None, k: int = 5
    ) -> List[Dict[str, Any]]:
        ...

    def add_mission_resolution(
        self,
        mission_id: str,
        failure: str,
        condition: str,
        action: str,
        outcome: str,
        success: bool,
        episode_id: Optional[str] = None,
    ) -> None:
        ...

    def get_summary(self) -> Dict[str, Any]:
        ...

    def ping(self) -> bool:
        ...


class NetworkXBackend:
    """Embedded NetworkX DiGraph backed by persistent JSON storage."""

    def __init__(self, file_path: Optional[str] = None):
        if file_path:
            self.file_path = Path(file_path)
        else:
            self.file_path = PROJECT_ROOT / config.memory.knowledge_graph_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.nx_graph = nx.DiGraph()
        self.lr = getattr(config.memory, "reinforcement_rate", 0.2)
        self._load_or_seed_graph()

    def ping(self) -> bool:
        return True

    def _load_or_seed_graph(self) -> None:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.nx_graph = nx.node_link_graph(data, edges="links")
                    return
            except Exception:
                pass
        self._seed_default_graph()
        self.save()

    def _seed_default_graph(self) -> None:
        self.nx_graph.clear()
        for n, attrs in DEFAULT_FAILURES + DEFAULT_CONDITIONS + DEFAULT_ACTIONS + DEFAULT_OUTCOMES:
            self.nx_graph.add_node(n, **attrs)
        for u, v, rel, w in DEFAULT_EDGES:
            self.nx_graph.add_edge(u, v, relation=rel, weight=w, hits=0)

    def save(self) -> None:
        try:
            data = nx.node_link_data(self.nx_graph, edges="links")
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _get_action_outcome_weight(self, action_node: str) -> float:
        for succ in self.nx_graph.successors(action_node):
            if succ.startswith("outcome:"):
                return self.nx_graph.nodes[succ].get("weight", 0.85)
        return 0.75

    def _collect_candidates(
        self, failure_type: str, condition: Optional[str] = None
    ) -> Dict[str, float]:
        target_f = f"failure:{failure_type}"
        if not self.nx_graph.has_node(target_f):
            return {}

        candidates: Dict[str, float] = {}
        cond_filter = f"condition:{condition}" if condition else None

        for succ in self.nx_graph.successors(target_f):
            if succ.startswith("action:"):
                if cond_filter:
                    continue
                action_name = self.nx_graph.nodes[succ].get("name", succ.replace("action:", ""))
                weight = self.nx_graph[target_f][succ].get("weight", 0.7)
                outcome_bonus = self._get_action_outcome_weight(succ)
                score = round(weight * outcome_bonus, 4)
                candidates[action_name] = max(candidates.get(action_name, 0.0), score)
            elif succ.startswith("condition:"):
                if cond_filter and succ != cond_filter:
                    continue
                cond_weight = self.nx_graph[target_f][succ].get("weight", 0.6)
                for cond_succ in self.nx_graph.successors(succ):
                    if cond_succ.startswith("action:"):
                        action_name = self.nx_graph.nodes[cond_succ].get(
                            "name", cond_succ.replace("action:", "")
                        )
                        act_weight = self.nx_graph[succ][cond_succ].get("weight", 0.8)
                        outcome_bonus = self._get_action_outcome_weight(cond_succ)
                        score = round(cond_weight * act_weight * outcome_bonus, 4)
                        candidates[action_name] = max(candidates.get(action_name, 0.0), score)
        return candidates

    def query_action_relevance(
        self, failure_type: str, condition: Optional[str] = None, k: int = 5
    ) -> List[Dict[str, Any]]:
        candidates = self._collect_candidates(failure_type, condition)
        if not candidates and condition:
            candidates = self._collect_candidates(failure_type, None)
        results = [
            {"action": act, "graph_relevance": score}
            for act, score in candidates.items()
        ]
        results.sort(key=lambda x: x["graph_relevance"], reverse=True)
        return results[:k]

    def _touch_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        success: bool,
        episode_id: Optional[str],
        create_weight: float = 0.5,
    ) -> None:
        if self.nx_graph.has_edge(src, dst):
            data = self.nx_graph[src][dst]
            old = float(data.get("weight", create_weight))
            data["weight"] = round(_reinforce(old, success, self.lr), 4)
            data["hits"] = int(data.get("hits", 0)) + 1
            data["relation"] = data.get("relation", relation)
            if episode_id:
                data["last_episode_id"] = episode_id
        else:
            weight = round(_reinforce(create_weight, success, self.lr), 4)
            attrs: Dict[str, Any] = {
                "relation": relation,
                "weight": weight,
                "hits": 1,
            }
            if episode_id:
                attrs["last_episode_id"] = episode_id
            self.nx_graph.add_edge(src, dst, **attrs)

    def add_mission_resolution(
        self,
        mission_id: str,
        failure: str,
        condition: str,
        action: str,
        outcome: str,
        success: bool,
        episode_id: Optional[str] = None,
    ) -> None:
        m_node = f"mission:{mission_id}"
        f_node = f"failure:{failure}"
        c_node = f"condition:{condition}"
        a_node = f"action:{action}"
        o_node = f"outcome:{outcome}"

        self.nx_graph.add_node(m_node, type="Mission", name=mission_id)
        self.nx_graph.add_node(f_node, type="Failure", name=failure)
        self.nx_graph.add_node(c_node, type="Condition", name=condition)
        self.nx_graph.add_node(a_node, type="RecoveryAction", name=action)
        self.nx_graph.add_node(o_node, type="Outcome", name=outcome, weight=1.0 if success else 0.2)

        self.nx_graph.add_edge(m_node, f_node, relation="ENCOUNTERED", weight=1.0)
        self._touch_edge(f_node, c_node, "OCCURRED_DURING", True, episode_id, create_weight=0.9)
        self._touch_edge(c_node, a_node, "TRIGGERED", success, episode_id, create_weight=0.5)
        self._touch_edge(f_node, a_node, "RECOVERED_BY", success, episode_id, create_weight=0.5)
        self._touch_edge(a_node, o_node, "RESULTED_IN", success, episode_id, create_weight=0.5)
        self.save()

    def get_edge_stats(self, condition: str, action: str) -> Dict[str, Any]:
        c_node = f"condition:{condition}"
        a_node = f"action:{action}"
        if not self.nx_graph.has_edge(c_node, a_node):
            return {"weight": None, "hits": 0, "last_episode_id": None}
        data = self.nx_graph[c_node][a_node]
        return {
            "weight": data.get("weight"),
            "hits": data.get("hits", 0),
            "last_episode_id": data.get("last_episode_id"),
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "engine": "NetworkX Embedded",
            "engine_id": "networkx",
            "nodes_count": self.nx_graph.number_of_nodes(),
            "edges_count": self.nx_graph.number_of_edges(),
            "neo4j_connected": False,
        }


class Neo4jBackend:
    """Bolt-backed knowledge graph with Cypher retrieval and real reinforcement."""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        if not NEO4J_AVAILABLE:
            raise RuntimeError("neo4j driver is not installed")
        self.lr = getattr(config.memory, "reinforcement_rate", 0.2)
        uri = uri or config.neo4j.uri
        user = user if user is not None else config.neo4j.user
        password = password if password is not None else config.neo4j.password
        self.driver = self._connect(uri, user, password)
        if not self.ping():
            self.close()
            raise RuntimeError(f"Neo4j ping failed for {uri}")
        self.ensure_constraints()

    @staticmethod
    def _connect(uri: str, user: str, password: str):
        cloud = uri.startswith("neo4j+s://") or uri.startswith("neo4j+ssc://") or uri.startswith("bolt+s://")
        timeout = 20.0 if cloud else 3.0
        driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=timeout,
            connection_acquisition_timeout=timeout,
            max_transaction_retry_time=0.0,
        )
        return driver

    @classmethod
    def connect_with_fallback(cls) -> "Neo4jBackend":
        """
        Prefer configured URI (Aura / compose). If cloud Bolt is refused,
        fall back to local Docker bolt://localhost:7687 so the API stays on Neo4j.
        """
        primary_uri = config.neo4j.uri
        attempts = [(primary_uri, config.neo4j.user, config.neo4j.password)]
        if primary_uri.startswith("neo4j+s://"):
            host = primary_uri.split("://", 1)[1]
            attempts.append((f"bolt+s://{host}", config.neo4j.user, config.neo4j.password))
        local_uri = "bolt://localhost:7687"
        if not any(u == local_uri for u, _, _ in attempts):
            attempts.append((local_uri, "neo4j", "change-me-local-dev-password"))
            # also try the configured password against local (compose may use NEO4J_PASSWORD)
            if config.neo4j.password and config.neo4j.password != "change-me-local-dev-password":
                attempts.append((local_uri, config.neo4j.user or "neo4j", config.neo4j.password))

        last_exc: Optional[Exception] = None
        for uri, user, password in attempts:
            try:
                backend = cls(uri=uri, user=user, password=password)
                if uri != primary_uri:
                    logger.warning("Neo4j primary %s unreachable; connected via %s", primary_uri, uri)
                else:
                    logger.info("Neo4j connected via %s", uri)
                return backend
            except Exception as exc:
                last_exc = exc
                logger.warning("Neo4j connect failed (%s): %s", uri, exc)
        raise RuntimeError(f"Neo4j unavailable: {last_exc}")

    def close(self) -> None:
        try:
            if self.driver:
                self.driver.close()
        except Exception:
            pass

    def ping(self) -> bool:
        try:
            with self.driver.session() as session:
                session.run("RETURN 1").consume()
            return True
        except Exception as exc:
            logger.warning("Neo4j ping failed: %s", exc)
            return False

    def ensure_constraints(self) -> None:
        try:
            with self.driver.session() as session:
                for stmt in NEO4J_CONSTRAINTS:
                    session.run(stmt)
        except Exception as exc:
            logger.warning("Failed to apply Neo4j constraints: %s", exc)

    def _run(self, cypher: str, **params) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    def query_action_relevance(
        self, failure_type: str, condition: Optional[str] = None, k: int = 5
    ) -> List[Dict[str, Any]]:
        scoped = """
        MATCH path = (f:Failure {name: $failure})-[:OCCURRED_DURING]->(c:Condition {name: $condition})
                     -[:TRIGGERED]->(a:RecoveryAction)
        OPTIONAL MATCH (a)-[:RESULTED_IN]->(o:Outcome)
        WITH a.name AS action,
             reduce(w = 1.0, rel IN relationships(path) | w * coalesce(rel.weight, 0.8)) AS path_weight,
             coalesce(o.weight, 0.75) AS outcome_weight
        RETURN action,
               round(max(path_weight * outcome_weight) * 10000) / 10000 AS graph_relevance
        ORDER BY graph_relevance DESC
        LIMIT $k
        """
        open_q = """
        MATCH path = (f:Failure {name: $failure})
                     -[:OCCURRED_DURING|TRIGGERED|RECOVERED_BY*1..3]->
                     (a:RecoveryAction)
        OPTIONAL MATCH (a)-[:RESULTED_IN]->(o:Outcome)
        WITH a.name AS action,
             reduce(w = 1.0, rel IN relationships(path) | w * coalesce(rel.weight, 0.8)) AS path_weight,
             coalesce(o.weight, 0.75) AS outcome_weight
        RETURN action,
               round(max(path_weight * outcome_weight) * 10000) / 10000 AS graph_relevance
        ORDER BY graph_relevance DESC
        LIMIT $k
        """
        rows: List[Dict[str, Any]] = []
        if condition:
            rows = self._run(scoped, failure=failure_type, condition=condition, k=k)
        if not rows:
            rows = self._run(open_q, failure=failure_type, k=k)
        return [
            {"action": r["action"], "graph_relevance": float(r["graph_relevance"])}
            for r in rows
            if r.get("action")
        ]

    def add_mission_resolution(
        self,
        mission_id: str,
        failure: str,
        condition: str,
        action: str,
        outcome: str,
        success: bool,
        episode_id: Optional[str] = None,
    ) -> None:
        cypher = """
        MERGE (m:Mission {name: $mission_id})
        MERGE (f:Failure {name: $failure})
        MERGE (c:Condition {name: $condition})
        MERGE (a:RecoveryAction {name: $action})
        MERGE (o:Outcome {name: $outcome})
        SET o.weight = CASE WHEN $success THEN 1.0 ELSE 0.2 END
        MERGE (m)-[:ENCOUNTERED]->(f)
        MERGE (f)-[od:OCCURRED_DURING]->(c)
          ON CREATE SET od.weight = 0.9
        MERGE (c)-[r:TRIGGERED]->(a)
          ON CREATE SET r.weight = 0.5, r.hits = 0
        SET r.hits = coalesce(r.hits, 0) + 1,
            r.weight = CASE WHEN $success
                            THEN coalesce(r.weight, 0.5) + $lr * (1.0 - coalesce(r.weight, 0.5))
                            ELSE coalesce(r.weight, 0.5) * (1.0 - $lr)
                       END,
            r.last_episode_id = $episode_id
        MERGE (f)-[rb:RECOVERED_BY]->(a)
          ON CREATE SET rb.weight = 0.5, rb.hits = 0
        SET rb.hits = coalesce(rb.hits, 0) + 1,
            rb.weight = CASE WHEN $success
                             THEN coalesce(rb.weight, 0.5) + $lr * (1.0 - coalesce(rb.weight, 0.5))
                             ELSE coalesce(rb.weight, 0.5) * (1.0 - $lr)
                        END,
            rb.last_episode_id = $episode_id
        MERGE (a)-[ri:RESULTED_IN]->(o)
          ON CREATE SET ri.weight = 0.5
        SET ri.weight = CASE WHEN $success
                             THEN coalesce(ri.weight, 0.5) + $lr * (1.0 - coalesce(ri.weight, 0.5))
                             ELSE coalesce(ri.weight, 0.5) * (1.0 - $lr)
                        END
        """
        self._run(
            cypher,
            mission_id=mission_id,
            failure=failure,
            condition=condition,
            action=action,
            outcome=outcome,
            success=success,
            lr=self.lr,
            episode_id=episode_id,
        )

    def get_edge_stats(self, condition: str, action: str) -> Dict[str, Any]:
        rows = self._run(
            """
            MATCH (c:Condition {name: $condition})-[r:TRIGGERED]->(a:RecoveryAction {name: $action})
            RETURN r.weight AS weight, r.hits AS hits, r.last_episode_id AS last_episode_id
            """,
            condition=condition,
            action=action,
        )
        if not rows:
            return {"weight": None, "hits": 0, "last_episode_id": None}
        return rows[0]

    def node_count(self) -> int:
        rows = self._run("MATCH (n) RETURN count(n) AS n")
        return int(rows[0]["n"]) if rows else 0

    def seed_ontology(self, reset: bool = False) -> Dict[str, int]:
        if reset:
            self._run("MATCH (n) DETACH DELETE n")
        self.ensure_constraints()
        for node_id, attrs in DEFAULT_FAILURES + DEFAULT_CONDITIONS + DEFAULT_ACTIONS + DEFAULT_OUTCOMES:
            label = attrs.get("type") or _LABEL_FROM_PREFIX[node_id.split(":")[0]]
            name = attrs["name"]
            extra = {k: v for k, v in attrs.items() if k not in ("type", "name")}
            set_clause = ""
            params: Dict[str, Any] = {"name": name}
            if extra:
                assignments = []
                for i, (k, v) in enumerate(extra.items()):
                    key = f"p{i}"
                    assignments.append(f"n.{k} = ${key}")
                    params[key] = v
                set_clause = "SET " + ", ".join(assignments)
            self._run(f"MERGE (n:{label} {{name: $name}}) {set_clause}", **params)
        for u, v, rel, w in DEFAULT_EDGES:
            u_label = _LABEL_FROM_PREFIX[u.split(":")[0]]
            v_label = _LABEL_FROM_PREFIX[v.split(":")[0]]
            self._run(
                f"""
                MATCH (a:{u_label} {{name: $u}})
                MATCH (b:{v_label} {{name: $v}})
                MERGE (a)-[r:{rel}]->(b)
                  ON CREATE SET r.weight = $w, r.hits = 0
                """,
                u=bare_name(u),
                v=bare_name(v),
                w=w,
            )
        nodes = self.node_count()
        edges = self._run("MATCH ()-[r]->() RETURN count(r) AS n")
        return {"nodes": nodes, "edges": int(edges[0]["n"]) if edges else 0}

    def get_summary(self) -> Dict[str, Any]:
        nodes = self.node_count()
        edges = self._run("MATCH ()-[r]->() RETURN count(r) AS n")
        return {
            "engine": "Neo4j",
            "engine_id": "neo4j",
            "nodes_count": nodes,
            "edges_count": int(edges[0]["n"]) if edges else 0,
            "neo4j_connected": True,
        }


class KnowledgeGraph:
    """
    Facade. Selects Neo4j when enabled and reachable, otherwise NetworkX.
    Public method names are unchanged so agents keep working.
    """

    def __init__(self, file_path: Optional[str] = None):
        self._nx = NetworkXBackend(file_path=file_path)
        self._neo: Optional[Neo4jBackend] = None
        self._backend: GraphBackend = self._nx
        self.engine = "networkx"
        self.nx_graph = self._nx.nx_graph

        if config.neo4j.enabled:
            try:
                neo = Neo4jBackend.connect_with_fallback()
                self._neo = neo
                self._backend = neo
                self.engine = "neo4j"
                if neo.node_count() == 0:
                    neo.seed_ontology()
            except Exception as exc:
                logger.warning("Neo4j unavailable, using NetworkX: %s", exc)
                self._backend = self._nx
                self.engine = "networkx"
                self._neo = None

    def _fallback_nx(self, exc: Exception, op: str) -> None:
        logger.warning("Neo4j %s failed (%s); falling back to NetworkX", op, exc)
        self._backend = self._nx
        self.engine = "networkx"

    def query_action_relevance(
        self, failure_type: str, condition: Optional[str] = None, k: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            return self._backend.query_action_relevance(failure_type, condition, k)
        except Exception as exc:
            if self.engine == "neo4j":
                self._fallback_nx(exc, "query")
                return self._nx.query_action_relevance(failure_type, condition, k)
            return []

    def add_mission_resolution(
        self,
        mission_id: str,
        failure: str,
        condition: str,
        action: str,
        outcome: str,
        success: bool,
        episode_id: Optional[str] = None,
    ) -> None:
        try:
            self._backend.add_mission_resolution(
                mission_id, failure, condition, action, outcome, success, episode_id
            )
        except Exception as exc:
            if self.engine == "neo4j":
                self._fallback_nx(exc, "write")
                self._nx.add_mission_resolution(
                    mission_id, failure, condition, action, outcome, success, episode_id
                )
        if self.engine == "neo4j":
            try:
                self._nx.add_mission_resolution(
                    mission_id, failure, condition, action, outcome, success, episode_id
                )
            except Exception:
                pass

    def get_edge_stats(self, condition: str, action: str) -> Dict[str, Any]:
        backend = self._backend
        if hasattr(backend, "get_edge_stats"):
            try:
                return backend.get_edge_stats(condition, action)
            except Exception as exc:
                if self.engine == "neo4j":
                    self._fallback_nx(exc, "edge_stats")
        return self._nx.get_edge_stats(condition, action)

    def get_summary(self) -> Dict[str, Any]:
        try:
            return self._backend.get_summary()
        except Exception as exc:
            if self.engine == "neo4j":
                self._fallback_nx(exc, "summary")
                return self._nx.get_summary()
            return self._nx.get_summary()

    def ping(self) -> bool:
        try:
            return self._backend.ping()
        except Exception:
            return False

    def reconnect(self) -> str:
        """Retry Bolt after compose deps are healthy so we do not stay on NetworkX."""
        if not config.neo4j.enabled:
            return self.engine
        if self.engine == "neo4j" and self.ping():
            try:
                self._neo.ensure_constraints()
                if self._neo.node_count() == 0:
                    self._neo.seed_ontology()
            except Exception as exc:
                logger.warning("Neo4j seed/constraints failed: %s", exc)
            return self.engine
        try:
            if self._neo:
                self._neo.close()
            neo = Neo4jBackend.connect_with_fallback()
            self._neo = neo
            self._backend = neo
            self.engine = "neo4j"
            if neo.node_count() == 0:
                neo.seed_ontology()
            logger.info("Knowledge graph engine: Neo4j")
        except Exception as exc:
            logger.warning("Neo4j unavailable, using NetworkX: %s", exc)
            self._backend = self._nx
            self.engine = "networkx"
            self._neo = None
        return self.engine

    def close(self) -> None:
        if self._neo:
            self._neo.close()
