import time
import uuid

import pytest

from memory.document_store import DocumentStore


def _live_store():
    store = DocumentStore(db_name="aerocortex_test")
    if not store.ping_sync():
        store.close()
        pytest.skip("MongoDB is not reachable")
    store.ensure_indexes_sync()
    return store


def test_document_store_ping_false_when_unreachable():
    store = DocumentStore(uri="mongodb://127.0.0.1:1", db_name="aerocortex_test")
    assert store.ping_sync() is False
    assert store.insert_episode_sync({"mission_id": "x"}) is None
    assert store.list_missions_sync() == []
    store.close()


def test_insert_episode_upserts_mission_and_lists():
    store = _live_store()
    mission_id = f"P2_{uuid.uuid4().hex[:8]}"
    ts = time.time()
    try:
        episode_id = store.insert_episode_sync({
            "mission_id": mission_id,
            "timestamp": ts,
            "failure_type": "GPS_INTERFERENCE",
            "telemetry": {"mission_id": mission_id},
            "situation": {"failure_type": "GPS_INTERFERENCE"},
            "retrieved_context": [],
            "plan": {"action": "SWITCH_TO_VIO_DEAD_RECKONING"},
            "safety_verdict": {"approved": True},
            "outcome": "MISSION_SUCCESS",
            "success": True,
            "latency_ms": {"retrieval": 1.0, "planning": 1.0, "safety": 1.0, "total": 3.0},
        })
        assert episode_id
        episodes = store.get_episodes_sync([episode_id])
        assert len(episodes) == 1
        assert episodes[0]["_id"] == episode_id

        missions = store.list_missions_sync(limit=50, skip=0)
        match = next((m for m in missions if m.get("_id") == mission_id), None)
        assert match is not None
        assert match["episode_count"] == 1
        assert match["anomaly_count"] == 1
        assert match["final_state"] == "MISSION_SUCCESS"
        assert store.count_missions_sync() >= 1
    finally:
        db = store._db_sync()
        if db is not None:
            db.episodes.delete_many({"mission_id": mission_id})
            db.missions.delete_one({"_id": mission_id})
        store.close()


def test_upsert_and_get_rules():
    store = _live_store()
    rule_id = f"RULE_TEST_{uuid.uuid4().hex[:8]}"
    try:
        store.upsert_rule_sync({
            "_id": rule_id,
            "if_condition": "GPS_INTERFERENCE",
            "then_action": "SWITCH_TO_VIO_DEAD_RECKONING",
            "confidence": 0.91,
            "hits": 2,
            "source": "test",
            "trigger": "GPS_INTERFERENCE",
            "action": "SWITCH_TO_VIO_DEAD_RECKONING",
            "condition": "gps_accuracy > 3.0",
        })
        rules = store.get_rules_sync()
        assert any(r.get("_id") == rule_id for r in rules)
    finally:
        db = store._db_sync()
        if db is not None:
            db.rules.delete_one({"_id": rule_id})
        store.close()
