"""AWS adapter unit tests (moto / stubs). Skip if moto unavailable."""

from __future__ import annotations

import json
import os

import pytest

moto = pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")


@mock_aws
def test_dynamo_missions_save_list_idempotent(aws_env, monkeypatch):
    import boto3

    dynamodb = boto3.resource("dynamodb", region_name="ap-south-1")
    dynamodb.create_table(
        TableName="missions",
        KeySchema=[
            {"AttributeName": "mission_id", "KeyType": "HASH"},
            {"AttributeName": "episode_id", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "mission_id", "AttributeType": "S"},
            {"AttributeName": "episode_id", "AttributeType": "S"},
            {"AttributeName": "gsi_pk", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "by_time",
                "KeySchema": [
                    {"AttributeName": "gsi_pk", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    monkeypatch.setenv("MISSIONS_TABLE", "missions")
    from cloud.dynamo_missions import DynamoMissionStore

    store = DynamoMissionStore(table_name="missions")
    assert store.health() == "ok"
    eid = store.insert_episode_sync(
        {"mission_id": "M1", "_id": "ep-1", "failure_type": "GPS_INTERFERENCE", "success": True}
    )
    assert eid == "ep-1"
    assert store.last_insert_duplicate is False
    eid2 = store.insert_episode_sync(
        {"mission_id": "M1", "_id": "ep-1", "failure_type": "GPS_INTERFERENCE", "success": True}
    )
    assert eid2 == "ep-1"
    assert store.last_insert_duplicate is True
    listed = store.list_missions_sync(limit=10, skip=0)
    assert len(listed) == 1
    assert store.episode_exists("M1", "ep-1")


@mock_aws
def test_dynamo_state_ttl_and_clear(aws_env, monkeypatch):
    import boto3

    dynamodb = boto3.resource("dynamodb", region_name="ap-south-1")
    dynamodb.create_table(
        TableName="state",
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    monkeypatch.setenv("STATE_TABLE", "state")
    from cloud.dynamo_state import SimulatorStateStore, DynamoWorkingMemory
    from models import UAVTelemetry
    import time

    sim = SimulatorStateStore(table_name="state")
    sim.save({"step_idx": 3, "altitude": 100.0})
    loaded = sim.load()
    assert loaded["step_idx"] == 3
    table = dynamodb.Table("state")
    item = table.get_item(Key={"pk": "SIM#default"})["Item"]
    assert "expires_at" in item
    assert int(item["expires_at"]) > int(time.time())
    sim.clear()
    assert sim.load() is None

    wm = DynamoWorkingMemory(table_name="state", mission_key="default")
    t = UAVTelemetry(
        timestamp=time.time(),
        mission_id="M1",
        latitude=1.0,
        longitude=2.0,
        altitude=50.0,
        velocity=5.0,
        battery_level=80.0,
        battery_voltage=15.0,
        gps_status="healthy",
        gps_accuracy=1.0,
        imu_acceleration=[0, 0, 9.8],
        imu_gyroscope=[0, 0, 0],
        wind_speed=3.0,
        wind_direction=90.0,
        communication_status="connected",
        mission_state="CRUISE",
        waypoint=1,
        heading=90.0,
        payload_status="nominal",
    )
    wm.update_telemetry(t)
    wm.clear()
    assert wm.latest_telemetry is None


@mock_aws
def test_s3_snapshots_seed_refresh_roundtrip(aws_env, monkeypatch):
    import boto3

    s3 = boto3.client("s3", region_name="ap-south-1")
    s3.create_bucket(
        Bucket="snap",
        CreateBucketConfiguration={"LocationConstraint": "ap-south-1"},
    )
    monkeypatch.setenv("SNAPSHOT_BUCKET", "snap")
    monkeypatch.setenv("SNAPSHOT_PREFIX", "snapshots/")
    from cloud.s3_snapshots import S3SnapshotStore

    store = S3SnapshotStore(bucket="snap", prefix="snapshots/")
    seed = {"nodes": [{"id": "a"}], "links": []}
    data = store.ensure_seed("knowledge_graph.json", seed)
    assert data["nodes"][0]["id"] == "a"
    store.save_json("knowledge_graph.json", {"nodes": [{"id": "b"}], "links": []})
    # Force refresh window
    store._etag_cache.clear()
    assert store.needs_refresh("knowledge_graph.json") is True
    loaded = store.load_json("knowledge_graph.json")
    assert loaded["nodes"][0]["id"] == "b"


@mock_aws
def test_sqs_publish_episode(aws_env, monkeypatch):
    import boto3

    sqs = boto3.client("sqs", region_name="ap-south-1")
    q = sqs.create_queue(
        QueueName="episodes.fifo",
        Attributes={"FifoQueue": "true", "ContentBasedDeduplication": "false"},
    )
    url = q["QueueUrl"]
    monkeypatch.setenv("EPISODE_QUEUE_URL", url)
    from cloud.sqs_episodes import publish_episode

    ok = publish_episode({"episode_id": "e1", "ok": True}, "e1", queue_url=url)
    assert ok is True
    msgs = sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=1)
    assert len(msgs.get("Messages") or []) == 1
    body = json.loads(msgs["Messages"][0]["Body"])
    assert body["episode_id"] == "e1"


def test_emit_metric_noop_by_default(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    from observability import emit_metric

    emit_metric("PipelineLatencyMs", 12.0, "Milliseconds")  # must not raise


def test_bedrock_unconfigured_skips(monkeypatch):
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    from llm.bedrock_client import BedrockClient

    client = BedrockClient()
    assert client.is_configured is False
    assert client.generate_json("hi") is None


def test_planner_tiers_include_bedrock(monkeypatch):
    monkeypatch.setenv("PLANNER_TIERS", "bedrock,groq,offline")
    from cloud.factories import planner_tiers

    assert planner_tiers() == ["bedrock", "groq", "offline"]


def test_vector_fallback_none(monkeypatch):
    monkeypatch.setenv("VECTOR_FALLBACK", "none")
    # Re-import path uses env at call time
    from memory.vector_store import _chroma_allowed, NullVectorBackend

    assert _chroma_allowed() is False
    nb = NullVectorBackend()
    assert nb.count() == 0
    assert nb.query("x")["ids"] == [[]]
