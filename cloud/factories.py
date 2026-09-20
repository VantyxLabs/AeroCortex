"""Factory helpers — pick backends from environment (defaults preserve local behavior)."""

from __future__ import annotations

import os
from typing import Any, Optional


def mission_store_mode() -> str:
    return (os.getenv("MISSION_STORE") or "mongo").strip().lower()


def working_memory_backend() -> str:
    return (os.getenv("WORKING_MEMORY_BACKEND") or "memory").strip().lower()


def sim_state_backend() -> str:
    return (os.getenv("SIM_STATE_BACKEND") or "memory").strip().lower()


def knowledge_snapshot_backend() -> str:
    return (os.getenv("KNOWLEDGE_SNAPSHOT_BACKEND") or "file").strip().lower()


def learning_mode() -> str:
    return (os.getenv("LEARNING_MODE") or "inline").strip().lower()


def vector_fallback() -> str:
    return (os.getenv("VECTOR_FALLBACK") or "chroma").strip().lower()


def planner_tiers() -> list[str]:
    raw = os.getenv("PLANNER_TIERS") or "groq,ollama,offline"
    return [t.strip().lower() for t in raw.split(",") if t.strip()]


def get_document_store():
    """Return Mongo DocumentStore or DynamoDB mission store."""
    if mission_store_mode() == "dynamodb":
        from cloud.dynamo_missions import DynamoMissionStore

        return DynamoMissionStore()
    from memory.document_store import get_document_store as _mongo

    return _mongo()


def get_working_memory(window_size: int = 100):
    if working_memory_backend() == "dynamodb":
        from cloud.dynamo_state import DynamoWorkingMemory

        return DynamoWorkingMemory(window_size=window_size)
    from memory.working_memory import WorkingMemory

    return WorkingMemory(window_size=window_size)


def get_snapshot_store():
    if knowledge_snapshot_backend() == "s3":
        from cloud.s3_snapshots import S3SnapshotStore

        return S3SnapshotStore()
    return None
