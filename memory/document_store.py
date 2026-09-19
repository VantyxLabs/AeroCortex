"""MongoDB document store — source of truth for episodes, missions, and rules."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from config import config

logger = logging.getLogger("aerocortex.document_store")

_STORE: Optional["DocumentStore"] = None


class DocumentStore:
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        self.uri = uri or config.mongo.uri
        self.db_name = db_name or config.mongo.db_name
        self._sync_client = None
        self._async_client = None
        self.available = False
        self._connect_sync()

    def _connect_sync(self) -> None:
        try:
            from pymongo import MongoClient

            client = MongoClient(self.uri, serverSelectionTimeoutMS=800)
            client.admin.command("ping")
            self._sync_client = client
            self.available = True
        except Exception as exc:
            logger.warning("MongoDB unavailable: %s", exc)
            self._sync_client = None
            self.available = False

    def _db_sync(self):
        if self._sync_client is None or not self.available:
            self._connect_sync()
        if not self.available or self._sync_client is None:
            return None
        return self._sync_client[self.db_name]

    def _async_db(self):
        if self._async_client is None:
            from motor.motor_asyncio import AsyncIOMotorClient

            self._async_client = AsyncIOMotorClient(
                self.uri, serverSelectionTimeoutMS=800
            )
        return self._async_client[self.db_name]

    async def ping(self) -> bool:
        try:
            if self._async_client is None:
                from motor.motor_asyncio import AsyncIOMotorClient

                self._async_client = AsyncIOMotorClient(
                    self.uri, serverSelectionTimeoutMS=800
                )
            await self._async_client.admin.command("ping")
            self.available = True
            return True
        except Exception:
            self.available = False
            return False

    def ping_sync(self) -> bool:
        try:
            if self._sync_client is None:
                self._connect_sync()
            if self._sync_client is None:
                return False
            self._sync_client.admin.command("ping")
            self.available = True
            return True
        except Exception:
            self.available = False
            return False

    async def ensure_indexes(self) -> None:
        try:
            db = self._async_db()
            await db.episodes.create_index([("mission_id", 1), ("timestamp", -1)])
            await db.episodes.create_index([("failure_type", 1), ("success", 1)])
            await db.episodes.create_index([("timestamp", -1)])
            await db.missions.create_index([("started_at", -1)])
            await db.rules.create_index([("if_condition", 1)])
            self.available = True
        except Exception as exc:
            logger.warning("Mongo index setup failed: %s", exc)
            self.available = False

    def ensure_indexes_sync(self) -> None:
        db = self._db_sync()
        if db is None:
            return
        try:
            db.episodes.create_index([("mission_id", 1), ("timestamp", -1)])
            db.episodes.create_index([("failure_type", 1), ("success", 1)])
            db.episodes.create_index([("timestamp", -1)])
            db.missions.create_index([("started_at", -1)])
            db.rules.create_index([("if_condition", 1)])
        except Exception as exc:
            logger.warning("Mongo index setup failed: %s", exc)

    def insert_episode_sync(self, episode: Dict[str, Any]) -> Optional[str]:
        db = self._db_sync()
        if db is None:
            return None
        try:
            episode = dict(episode)
            episode_id = str(episode.get("_id") or uuid.uuid4())
            episode["_id"] = episode_id
            db.episodes.insert_one(episode)
            mission_id = episode.get("mission_id")
            if mission_id:
                db.missions.update_one(
                    {"_id": mission_id},
                    {
                        "$set": {
                            "ended_at": episode.get("timestamp"),
                            "final_state": episode.get("outcome"),
                        },
                        "$setOnInsert": {"started_at": episode.get("timestamp")},
                        "$inc": {
                            "episode_count": 1,
                            "anomaly_count": 1 if episode.get("failure_type") not in (None, "NONE") else 0,
                        },
                    },
                    upsert=True,
                )
            return episode_id
        except Exception as exc:
            logger.warning("Mongo insert_episode failed: %s", exc)
            self.available = False
            return None

    async def insert_episode(self, episode: Dict[str, Any]) -> Optional[str]:
        try:
            db = self._async_db()
            episode = dict(episode)
            episode_id = str(episode.get("_id") or uuid.uuid4())
            episode["_id"] = episode_id
            await db.episodes.insert_one(episode)
            mission_id = episode.get("mission_id")
            if mission_id:
                await db.missions.update_one(
                    {"_id": mission_id},
                    {
                        "$set": {
                            "ended_at": episode.get("timestamp"),
                            "final_state": episode.get("outcome"),
                        },
                        "$setOnInsert": {"started_at": episode.get("timestamp")},
                        "$inc": {
                            "episode_count": 1,
                            "anomaly_count": 1 if episode.get("failure_type") not in (None, "NONE") else 0,
                        },
                    },
                    upsert=True,
                )
            self.available = True
            return episode_id
        except Exception as exc:
            logger.warning("Mongo insert_episode failed: %s", exc)
            self.available = False
            return None

    def get_episodes_sync(self, ids: List[str]) -> List[Dict[str, Any]]:
        db = self._db_sync()
        if db is None or not ids:
            return []
        try:
            return list(db.episodes.find({"_id": {"$in": ids}}))
        except Exception as exc:
            logger.warning("Mongo get_episodes failed: %s", exc)
            return []

    async def get_episodes(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        try:
            db = self._async_db()
            cursor = db.episodes.find({"_id": {"$in": ids}})
            return await cursor.to_list(length=len(ids))
        except Exception as exc:
            logger.warning("Mongo get_episodes failed: %s", exc)
            return []

    async def list_missions(self, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        try:
            db = self._async_db()
            cursor = db.missions.find().sort("started_at", -1).skip(skip).limit(limit)
            docs = await cursor.to_list(length=limit)
            self.available = True
            return docs
        except Exception as exc:
            logger.warning("Mongo list_missions failed: %s", exc)
            self.available = False
            return []

    def list_missions_sync(self, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        db = self._db_sync()
        if db is None:
            return []
        try:
            return list(db.missions.find().sort("started_at", -1).skip(skip).limit(limit))
        except Exception as exc:
            logger.warning("Mongo list_missions failed: %s", exc)
            self.available = False
            return []

    async def count_missions(self) -> int:
        try:
            db = self._async_db()
            total = await db.missions.count_documents({})
            self.available = True
            return int(total)
        except Exception:
            self.available = False
            return 0

    def count_missions_sync(self) -> int:
        db = self._db_sync()
        if db is None:
            return 0
        try:
            return int(db.missions.count_documents({}))
        except Exception:
            self.available = False
            return 0

    async def upsert_rule(self, rule: Dict[str, Any]) -> None:
        try:
            db = self._async_db()
            rule_id = rule.get("_id") or rule.get("rule_id")
            if not rule_id:
                return
            doc = dict(rule)
            doc["_id"] = rule_id
            await db.rules.replace_one({"_id": rule_id}, doc, upsert=True)
        except Exception as exc:
            logger.warning("Mongo upsert_rule failed: %s", exc)

    def upsert_rule_sync(self, rule: Dict[str, Any]) -> None:
        db = self._db_sync()
        if db is None:
            return
        try:
            rule_id = rule.get("_id") or rule.get("rule_id")
            if not rule_id:
                return
            doc = dict(rule)
            doc["_id"] = rule_id
            db.rules.replace_one({"_id": rule_id}, doc, upsert=True)
        except Exception as exc:
            logger.warning("Mongo upsert_rule failed: %s", exc)

    async def get_rules(self) -> List[Dict[str, Any]]:
        try:
            db = self._async_db()
            return await db.rules.find().to_list(length=500)
        except Exception:
            return []

    def get_rules_sync(self) -> List[Dict[str, Any]]:
        db = self._db_sync()
        if db is None:
            return []
        try:
            return list(db.rules.find())
        except Exception:
            return []

    def close(self) -> None:
        try:
            if self._sync_client:
                self._sync_client.close()
        except Exception:
            pass
        try:
            if self._async_client:
                self._async_client.close()
        except Exception:
            pass


def get_document_store() -> DocumentStore:
    global _STORE
    if _STORE is None:
        _STORE = DocumentStore()
    return _STORE
