"""DynamoDB-backed working memory + simulator state (TTL 24h)."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from decimal import Decimal
from typing import Any, Dict, Optional

from models import RecoveryPlan, SafetyVerdict, SituationReport, UAVTelemetry

logger = logging.getLogger("aerocortex.cloud.dynamo_state")

TTL_SECONDS = 24 * 3600
SIM_PK = "SIM#default"
WM_PK_PREFIX = "WM#"


def _expires_at() -> int:
    return int(time.time()) + TTL_SECONDS


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


class DynamoWorkingMemory:
    """Drop-in WorkingMemory that persists snapshot to DynamoDB."""

    def __init__(self, window_size: int = 100, table_name: Optional[str] = None, mission_key: str = "default"):
        self.window_size = window_size
        self.table_name = table_name or os.getenv("STATE_TABLE") or ""
        self.pk = f"{WM_PK_PREFIX}{mission_key}"
        self._table = None
        if self.table_name:
            try:
                import boto3

                self._table = boto3.resource("dynamodb").Table(self.table_name)
            except Exception as exc:
                logger.warning("Dynamo WM init failed: %s", exc)

        self.telemetry_history: deque = deque(maxlen=window_size)
        self.latest_telemetry: Optional[UAVTelemetry] = None
        self.active_anomaly: Optional[SituationReport] = None
        self.mission_phase: str = "CRUISE"
        self.current_waypoint: int = 1
        self.battery_status: Dict[str, float] = {"level": 100.0, "voltage": 16.8}
        self.communication_state: str = "connected"
        self.gps_state: Dict[str, Any] = {"status": "healthy", "accuracy": 1.0}
        self.current_recovery_plan: Optional[RecoveryPlan] = None
        self.safety_status: Optional[SafetyVerdict] = None
        self._load()

    def _load(self) -> None:
        if self._table is None:
            return
        try:
            resp = self._table.get_item(Key={"pk": self.pk})
            item = resp.get("Item") or {}
            snap = _from_dynamo(item.get("snapshot") or {})
            if snap.get("latest_telemetry"):
                self.latest_telemetry = UAVTelemetry(**snap["latest_telemetry"])
            if snap.get("active_anomaly"):
                self.active_anomaly = SituationReport(**snap["active_anomaly"])
            self.mission_phase = snap.get("mission_phase", self.mission_phase)
            self.current_waypoint = int(snap.get("current_waypoint", self.current_waypoint))
            self.battery_status = snap.get("battery_status", self.battery_status)
            self.communication_state = snap.get("communication_state", self.communication_state)
            self.gps_state = snap.get("gps_state", self.gps_state)
            if snap.get("current_recovery_plan"):
                self.current_recovery_plan = RecoveryPlan(**snap["current_recovery_plan"])
            if snap.get("safety_status"):
                self.safety_status = SafetyVerdict(**snap["safety_status"])
        except Exception as exc:
            logger.warning("Dynamo WM load failed: %s", exc)

    def _save(self) -> None:
        if self._table is None:
            return
        try:
            self._table.put_item(
                Item=_to_dynamo({
                    "pk": self.pk,
                    "expires_at": _expires_at(),
                    "snapshot": self.get_snapshot(),
                })
            )
        except Exception as exc:
            logger.warning("Dynamo WM save failed: %s", exc)

    def update_telemetry(self, telemetry: UAVTelemetry) -> None:
        self.latest_telemetry = telemetry
        self.telemetry_history.append(telemetry)
        self.mission_phase = telemetry.mission_state
        self.current_waypoint = telemetry.waypoint
        self.battery_status = {"level": telemetry.battery_level, "voltage": telemetry.battery_voltage}
        self.communication_state = telemetry.communication_status
        self.gps_state = {"status": telemetry.gps_status, "accuracy": telemetry.gps_accuracy}
        self._save()

    def set_anomaly(self, report: SituationReport) -> None:
        self.active_anomaly = report
        self._save()

    def set_recovery_plan(self, plan: RecoveryPlan) -> None:
        self.current_recovery_plan = plan
        self._save()

    def set_safety_status(self, verdict: SafetyVerdict) -> None:
        self.safety_status = verdict
        self._save()

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "latest_telemetry": self.latest_telemetry.model_dump() if self.latest_telemetry else None,
            "active_anomaly": self.active_anomaly.model_dump() if self.active_anomaly else None,
            "mission_phase": self.mission_phase,
            "current_waypoint": self.current_waypoint,
            "battery_status": self.battery_status,
            "communication_state": self.communication_state,
            "gps_state": self.gps_state,
            "current_recovery_plan": self.current_recovery_plan.model_dump() if self.current_recovery_plan else None,
            "safety_status": self.safety_status.model_dump() if self.safety_status else None,
            "history_length": len(self.telemetry_history),
        }

    def clear(self) -> None:
        self.telemetry_history.clear()
        self.latest_telemetry = None
        self.active_anomaly = None
        self.mission_phase = "CRUISE"
        self.current_waypoint = 1
        self.battery_status = {"level": 100.0, "voltage": 16.8}
        self.communication_state = "connected"
        self.gps_state = {"status": "healthy", "accuracy": 1.0}
        self.current_recovery_plan = None
        self.safety_status = None
        if self._table is not None:
            try:
                self._table.delete_item(Key={"pk": self.pk})
            except Exception as exc:
                logger.warning("Dynamo WM clear failed: %s", exc)


class SimulatorStateStore:
    """Persist TelemetryGenerator fields under SIM#default."""

    def __init__(self, table_name: Optional[str] = None):
        self.table_name = table_name or os.getenv("STATE_TABLE") or ""
        self._table = None
        if self.table_name:
            try:
                import boto3

                self._table = boto3.resource("dynamodb").Table(self.table_name)
            except Exception as exc:
                logger.warning("Dynamo sim store init failed: %s", exc)

    def load(self) -> Optional[Dict[str, Any]]:
        if self._table is None:
            return None
        try:
            resp = self._table.get_item(Key={"pk": SIM_PK})
            item = resp.get("Item") or {}
            state = item.get("state")
            return _from_dynamo(state) if state is not None else None
        except Exception as exc:
            logger.warning("Dynamo sim load failed: %s", exc)
            return None

    def save(self, state: Dict[str, Any]) -> None:
        if self._table is None:
            return
        try:
            self._table.put_item(
                Item=_to_dynamo({"pk": SIM_PK, "expires_at": _expires_at(), "state": state})
            )
        except Exception as exc:
            logger.warning("Dynamo sim save failed: %s", exc)

    def clear(self) -> None:
        if self._table is None:
            return
        try:
            self._table.delete_item(Key={"pk": SIM_PK})
        except Exception as exc:
            logger.warning("Dynamo sim clear failed: %s", exc)
