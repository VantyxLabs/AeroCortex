"""DynamoDB mission / episode store (MISSION_STORE=dynamodb)."""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("aerocortex.cloud.dynamo_missions")


def _to_dynamo(obj: Any) -> Any:
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dynamo(v) for v in obj]
    return obj


def _from_dynamo(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        if obj % 1 == 0:
            return int(obj)
        return float(obj)
    if isinstance(obj, dict):
        return {k: _from_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_dynamo(v) for v in obj]
    return obj


class DynamoMissionStore:
    """Mongo-compatible subset used by LearningAgent and /missions / healthz."""

    def __init__(self, table_name: Optional[str] = None):
        self.table_name = table_name or os.getenv("MISSIONS_TABLE") or ""
        self.available = False
        self._table = None
        self.last_insert_duplicate = False
        if self.table_name:
            try:
                import boto3

                self._table = boto3.resource("dynamodb").Table(self.table_name)
                self.available = True
            except Exception as exc:
                logger.warning("DynamoDB missions init failed: %s", exc)
                self.available = False

    def ping_sync(self) -> bool:
        return self.health() == "ok"

    async def ping(self) -> bool:
        return self.ping_sync()

    def health(self) -> str:
        if not self.table_name or self._table is None:
            return "down"
        try:
            import boto3

            client = boto3.client("dynamodb")
            client.describe_table(TableName=self.table_name)
            self.available = True
            return "ok"
        except Exception as exc:
            logger.warning("DynamoDB health failed: %s", exc)
            self.available = False
            return "down"

    def close(self) -> None:
        pass

    async def ensure_indexes(self) -> None:
        pass

    def ensure_indexes_sync(self) -> None:
        pass

    def insert_episode_sync(self, episode: Dict[str, Any]) -> Optional[str]:
        """Conditional put. Returns episode_id, or None on hard failure.
        Duplicate (already exists) returns episode_id with no overwrite.
        """
        if self._table is None:
            return None
        episode = dict(episode)
        episode_id = str(episode.get("_id") or episode.get("episode_id") or uuid.uuid4())
        mission_id = str(episode.get("mission_id") or "UNKNOWN")
        created = datetime.now(timezone.utc).isoformat()
        item = {
            "mission_id": mission_id,
            "episode_id": episode_id,
            "gsi_pk": "EPISODE",
            "created_at": created,
            "payload": episode,
            "_id": episode_id,
        }
        self.last_insert_duplicate = False
        try:
            self._table.put_item(
                Item=_to_dynamo(item),
                ConditionExpression="attribute_not_exists(episode_id)",
            )
            self.available = True
            return episode_id
        except Exception as exc:
            # ConditionalCheckFailedException => duplicate
            name = type(exc).__name__
            if "ConditionalCheckFailed" in name or "ConditionalCheckFailedException" in str(exc):
                self.last_insert_duplicate = True
                return episode_id
            logger.warning("Dynamo insert_episode failed: %s", exc)
            return None

    def episode_exists(self, mission_id: str, episode_id: str) -> bool:
        if self._table is None:
            return False
        try:
            resp = self._table.get_item(Key={"mission_id": mission_id, "episode_id": episode_id})
            return "Item" in resp
        except Exception:
            return False

    def get_episodes_sync(self, ids: List[str]) -> List[Dict[str, Any]]:
        # Scan GSI is expensive; for demo, query by_time and filter (small scale).
        out: List[Dict[str, Any]] = []
        want = set(str(i) for i in ids)
        for doc in self.list_missions_sync(limit=200, skip=0):
            eid = str(doc.get("_id") or doc.get("episode_id") or "")
            if eid in want:
                out.append(doc)
        return out

    def list_missions_sync(self, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        if self._table is None:
            return []
        try:
            from boto3.dynamodb.conditions import Key

            # Dynamo has no skip: fetch skip+limit then slice
            need = max(1, skip + limit)
            resp = self._table.query(
                IndexName="by_time",
                KeyConditionExpression=Key("gsi_pk").eq("EPISODE"),
                ScanIndexForward=False,
                Limit=need,
            )
            items = resp.get("Items") or []
            sliced = items[skip : skip + limit]
            result = []
            for it in sliced:
                payload = _from_dynamo(it.get("payload") or {})
                if isinstance(payload, dict):
                    doc = dict(payload)
                else:
                    doc = {}
                doc["_id"] = it.get("episode_id")
                doc.setdefault("mission_id", it.get("mission_id"))
                doc.setdefault("timestamp", it.get("created_at"))
                result.append(doc)
            return result
        except Exception as exc:
            logger.warning("Dynamo list_missions failed: %s", exc)
            return []

    async def list_missions(self, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        return self.list_missions_sync(limit=limit, skip=skip)

    def count_missions_sync(self) -> int:
        if self._table is None:
            return 0
        try:
            return int(self._table.item_count or 0)
        except Exception:
            return len(self.list_missions_sync(limit=500, skip=0))

    async def count_missions(self) -> int:
        return self.count_missions_sync()

    def get_rules_sync(self) -> List[Dict[str, Any]]:
        return []
