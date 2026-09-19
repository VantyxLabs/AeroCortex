import pytest
from models import UAVTelemetry, SituationReport, EpisodicExperience
from memory.working_memory import WorkingMemory
from memory.vector_store import VectorStore
from memory.episodic_memory import EpisodicMemory
from memory.semantic_memory import SemanticMemory
from memory.knowledge_graph import KnowledgeGraph, NetworkXBackend, Neo4jBackend
from agents.memory_agent import (
    MemoryAgent,
    chroma_distance_to_similarity,
    fuse_hybrid_scores,
    minmax_normalize,
)


def test_working_memory():
    wm = WorkingMemory()
    t = UAVTelemetry(battery_level=80.0, altitude=60.0)
    wm.update_telemetry(t)
    
    snap = wm.get_snapshot()
    assert snap["latest_telemetry"]["battery_level"] == 80.0
    assert snap["battery_status"]["level"] == 80.0
    assert snap["history_length"] == 1
    
    wm.clear()
    assert wm.get_snapshot()["latest_telemetry"] is None


def test_vector_store_offline():
    vs = VectorStore(collection_name="test_vector_store")
    vs.reset()
    vs.add_documents(
        ids=["doc_1", "doc_2"],
        documents=["GPS failure during cruise in high wind", "Battery depleted emergency land"],
        metadatas=[{"tag": "gps"}, {"tag": "battery"}]
    )
    assert vs.count() == 2
    res = vs.query("GPS loss interference", n_results=1)
    assert len(res["ids"][0]) == 1
    assert res["ids"][0][0] == "doc_1"


def test_semantic_memory_matching():
    sm = SemanticMemory()
    rules = sm.match_rules("GPS_INTERFERENCE")
    assert len(rules) >= 1
    assert rules[0].trigger == "GPS_INTERFERENCE"
    assert "SWITCH_TO_VIO" in rules[0].action


def test_semantic_memory_falls_back_to_json(tmp_path, monkeypatch):
    class DownStore:
        available = False

        def get_rules_sync(self):
            raise RuntimeError("mongo down")

    monkeypatch.setattr("memory.semantic_memory.get_document_store", lambda: DownStore())
    path = tmp_path / "semantic_rules.json"
    sm = SemanticMemory(file_path=str(path))
    rules = sm.match_rules("GPS_INTERFERENCE")
    assert len(rules) >= 1
    assert path.exists()


def test_knowledge_graph_traversal():
    kg = KnowledgeGraph()
    paths = kg.query_action_relevance("GPS_INTERFERENCE")
    assert len(paths) >= 1
    assert any("SWITCH_TO_VIO" in p["action"] for p in paths)


def _assert_ranked(paths):
    assert paths
    scores = [p["graph_relevance"] for p in paths]
    assert scores == sorted(scores, reverse=True)


def test_query_action_relevance_networkx_backend(tmp_path):
    backend = NetworkXBackend(file_path=str(tmp_path / "kg.json"))
    backend._seed_default_graph()
    paths = backend.query_action_relevance("GPS_INTERFERENCE")
    _assert_ranked(paths)
    assert any("SWITCH_TO_VIO" in p["action"] for p in paths)


def test_query_action_relevance_both_backends(tmp_path):
    nx_backend = NetworkXBackend(file_path=str(tmp_path / "kg.json"))
    nx_backend._seed_default_graph()
    nx_paths = nx_backend.query_action_relevance("GPS_INTERFERENCE")
    _assert_ranked(nx_paths)

    try:
        neo = Neo4jBackend()
    except Exception:
        pytest.skip("Neo4j is not reachable")
    try:
        if neo.node_count() == 0:
            neo.seed_ontology()
        neo_paths = neo.query_action_relevance("GPS_INTERFERENCE")
        _assert_ranked(neo_paths)
        assert nx_paths[0]["action"] == neo_paths[0]["action"]
    finally:
        neo.close()


def test_networkx_reinforcement_increases_weight(tmp_path):
    backend = NetworkXBackend(file_path=str(tmp_path / "kg.json"))
    backend._seed_default_graph()
    before = backend.get_edge_stats("MODERATE_WIND", "SWITCH_TO_VIO_DEAD_RECKONING")
    backend.add_mission_resolution(
        "M1", "GPS_INTERFERENCE", "MODERATE_WIND",
        "SWITCH_TO_VIO_DEAD_RECKONING", "MISSION_SUCCESS", True, "ep-1"
    )
    after = backend.get_edge_stats("MODERATE_WIND", "SWITCH_TO_VIO_DEAD_RECKONING")
    assert after["hits"] == (before["hits"] or 0) + 1
    assert after["weight"] > (before["weight"] or 0.5)


def test_neo4j_reinforcement_increases_weight():
    try:
        neo = Neo4jBackend()
    except Exception:
        pytest.skip("Neo4j is not reachable")
    try:
        neo.seed_ontology()
        before = neo.get_edge_stats("MODERATE_WIND", "SWITCH_TO_VIO_DEAD_RECKONING")
        neo.add_mission_resolution(
            "M_REINFORCE", "GPS_INTERFERENCE", "MODERATE_WIND",
            "SWITCH_TO_VIO_DEAD_RECKONING", "MISSION_SUCCESS", True, "ep-neo-1"
        )
        after = neo.get_edge_stats("MODERATE_WIND", "SWITCH_TO_VIO_DEAD_RECKONING")
        assert after["hits"] == (before["hits"] or 0) + 1
        assert after["weight"] > (before["weight"] or 0.5)
        assert after.get("last_episode_id") == "ep-neo-1"
    finally:
        neo.close()


def test_knowledge_graph_facade_reports_engine():
    kg = KnowledgeGraph()
    assert kg.engine in ("neo4j", "networkx")
    summary = kg.get_summary()
    assert summary["engine"] in ("Neo4j", "NetworkX Embedded")
    assert summary["engine_id"] == kg.engine
    kg.close()


def test_chroma_distance_to_similarity():
    assert chroma_distance_to_similarity(0.0) == pytest.approx(1.0)
    assert chroma_distance_to_similarity(1.0) == pytest.approx(0.0)
    assert chroma_distance_to_similarity(1.0, space="l2") == pytest.approx(0.5)
    assert chroma_distance_to_similarity(-0.1) == pytest.approx(1.0)


def test_hybrid_fusion_ranking_normalized():
    vector_items = [
        {"episode_id": "a", "action": "ACT_A", "vector_similarity": 0.9},
        {"episode_id": "b", "action": "ACT_B", "vector_similarity": 0.1},
    ]
    graph_items = [
        {"action": "ACT_A", "graph_relevance": 0.2},
        {"action": "ACT_B", "graph_relevance": 0.9},
    ]
    ranked = fuse_hybrid_scores(vector_items, graph_items, w_vector=0.6, w_graph=0.4)
    assert ranked[0]["episode_id"] == "a"
    for item in ranked:
        assert 0.0 <= item["vector_similarity"] <= 1.0
        assert 0.0 <= item["graph_relevance"] <= 1.0
        assert 0.0 <= item["final_score"] <= 1.0
    assert ranked[0]["final_score"] == pytest.approx(0.6)
    assert ranked[1]["final_score"] == pytest.approx(0.4)


def test_hybrid_fusion_from_distances_and_graph_weights():
    vector_items = [
        {"episode_id": "near", "action": "ACT_A", "distance": 0.1},
        {"episode_id": "far", "action": "ACT_B", "distance": 0.9},
    ]
    graph_items = [
        {"action": "ACT_A", "graph_relevance": 0.2},
        {"action": "ACT_B", "graph_relevance": 0.8},
    ]
    ranked = fuse_hybrid_scores(vector_items, graph_items, w_vector=0.6, w_graph=0.4)
    assert [item["episode_id"] for item in ranked] == ["near", "far"]
    for item in ranked:
        assert 0.0 <= item["vector_similarity"] <= 1.0
        assert 0.0 <= item["graph_relevance"] <= 1.0
        assert 0.0 <= item["final_score"] <= 1.0
    assert ranked[0]["vector_similarity"] == pytest.approx(1.0)
    assert ranked[0]["graph_relevance"] == pytest.approx(0.0)
    assert ranked[0]["final_score"] == pytest.approx(0.6)
    assert ranked[1]["final_score"] == pytest.approx(0.4)


def test_hydrate_from_mongo_single_in_query(monkeypatch):
    calls = []

    class Store:
        available = True

        def get_episodes_sync(self, ids):
            calls.append(list(ids))
            return [{
                "_id": "ep1",
                "situation": {"description": "from mongo"},
                "plan": {"action": "SWITCH_TO_VIO_DEAD_RECKONING"},
                "outcome": "MISSION_SUCCESS",
            }]

    monkeypatch.setattr("agents.memory_agent.get_document_store", lambda: Store())
    agent = MemoryAgent()
    exp_a = EpisodicExperience(
        episode_id="ep1",
        mission_id="M1",
        failure="GPS_INTERFERENCE",
        context="chroma context",
        action="OLD_ACTION",
        outcome="unknown",
        success=True,
    )
    exp_b = EpisodicExperience(
        episode_id="ep2",
        mission_id="M2",
        failure="GPS_INTERFERENCE",
        context="chroma only",
        action="HOLD",
        outcome="unknown",
        success=True,
    )
    ranked = agent._hydrate_from_mongo([
        {
            "episode_id": "ep1",
            "action": "OLD_ACTION",
            "vector_similarity": 1.0,
            "graph_relevance": 0.5,
            "final_score": 0.8,
            "experience": exp_a,
        },
        {
            "episode_id": "ep2",
            "action": "HOLD",
            "vector_similarity": 0.0,
            "graph_relevance": 0.2,
            "final_score": 0.1,
            "experience": exp_b,
        },
    ])
    assert len(calls) == 1
    assert set(calls[0]) == {"ep1", "ep2"}
    assert ranked[0].hydrated_from_mongo is True
    assert ranked[0].experience.context == "from mongo"
    assert ranked[0].experience.action == "SWITCH_TO_VIO_DEAD_RECKONING"
    assert ranked[1].hydrated_from_mongo is False
    assert ranked[0].vector_similarity == 1.0
    assert ranked[0].graph_relevance == 0.5
    assert ranked[0].final_score == 0.8


def test_minmax_normalize_constant_list():
    assert minmax_normalize([0.4, 0.4, 0.4]) == [1.0, 1.0, 1.0]


def test_memory_agent_hybrid_scoring():
    mem_agent = MemoryAgent()
    t = UAVTelemetry(altitude=120.0, wind_speed=7.0)
    sit = SituationReport(
        anomaly_detected=True,
        failure_type="GPS_INTERFERENCE",
        severity="HIGH",
        mission_phase="CRUISE"
    )
    context = mem_agent.retrieve_context(t, sit)
    assert context.retrieved_experiences
    assert context.top_recommended_action is not None
    assert context.retrieval_latency_ms >= 0.0
    assert mem_agent.last_context is context
    top_exp = context.retrieved_experiences[0]
    assert 0.0 <= top_exp.final_score <= 1.0
    assert 0.0 <= top_exp.vector_similarity <= 1.0
    assert 0.0 <= top_exp.graph_relevance <= 1.0
    scores = [item.final_score for item in context.retrieved_experiences]
    assert scores == sorted(scores, reverse=True)
