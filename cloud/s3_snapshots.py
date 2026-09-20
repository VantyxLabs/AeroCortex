"""S3 snapshot store for knowledge graph + semantic rules JSON."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("aerocortex.cloud.s3_snapshots")


class S3SnapshotStore:
    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
    ):
        self.bucket = bucket or os.getenv("SNAPSHOT_BUCKET") or ""
        self.prefix = (prefix or os.getenv("SNAPSHOT_PREFIX") or "snapshots/").rstrip("/") + "/"
        self._client = None
        self._etag_cache: Dict[str, Tuple[str, float]] = {}
        self._check_interval = 5.0
        if self.bucket:
            try:
                import boto3

                self._client = boto3.client("s3")
            except Exception as exc:
                logger.warning("S3 client init failed: %s", exc)

    def _key(self, name: str) -> str:
        return f"{self.prefix}{name}"

    def load_json(self, name: str) -> Optional[Any]:
        if not self._client or not self.bucket:
            return None
        key = self._key(name)
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            body = resp["Body"].read().decode("utf-8")
            etag = (resp.get("ETag") or "").strip('"')
            self._etag_cache[name] = (etag, time.monotonic())
            return json.loads(body)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            logger.warning("S3 load %s failed: %s", key, exc)
            return None

    def save_json(self, name: str, data: Any) -> bool:
        if not self._client or not self.bucket:
            return False
        key = self._key(name)
        try:
            body = json.dumps(data, indent=2).encode("utf-8")
            resp = self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
            etag = (resp.get("ETag") or "").strip('"')
            self._etag_cache[name] = (etag, time.monotonic())
            return True
        except Exception as exc:
            logger.warning("S3 save %s failed: %s", key, exc)
            return False

    def needs_refresh(self, name: str) -> bool:
        """Cheap head_object ETag check (at most once per 5s per key)."""
        if not self._client or not self.bucket:
            return False
        cached = self._etag_cache.get(name)
        now = time.monotonic()
        if cached and (now - cached[1]) < self._check_interval:
            return False
        key = self._key(name)
        try:
            resp = self._client.head_object(Bucket=self.bucket, Key=key)
            etag = (resp.get("ETag") or "").strip('"')
            if not cached:
                self._etag_cache[name] = (etag, now)
                return True
            if etag != cached[0]:
                self._etag_cache[name] = (etag, now)
                return True
            self._etag_cache[name] = (etag, now)
            return False
        except Exception:
            return False

    def ensure_seed(self, name: str, seed_data: Any) -> Any:
        existing = self.load_json(name)
        if existing is not None:
            return existing
        self.save_json(name, seed_data)
        return seed_data
