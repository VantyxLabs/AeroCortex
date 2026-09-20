"""Learning Lambda — SQS FIFO consumer for async episode consolidation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from cloud.secrets import load_ssm_secrets_into_env

load_ssm_secrets_into_env()

try:
    from aws_xray_sdk.core import patch_all

    patch_all()
except Exception:
    pass

from observability import emit_metric, setup_logging

setup_logging()
logger = logging.getLogger("aerocortex.learning_handler")


def _commit_payload(payload: Dict[str, Any]) -> None:
    """Reconstruct models and call LearningAgent commit (idempotent episode put first)."""
    from agents.learning_agent import LearningAgent
    from models import RecoveryPlan, SafetyVerdict, SituationReport, UAVTelemetry

    telemetry = UAVTelemetry(**payload["telemetry"])
    situation = SituationReport(**payload["situation"])
    plan = RecoveryPlan(**payload["plan"])
    verdict = SafetyVerdict(**payload["safety_verdict"])
    agent = LearningAgent()
    # Prefer pre-assigned episode_id for idempotency
    result = agent.process_mission_outcome(
        telemetry=telemetry,
        situation=situation,
        plan=plan,
        verdict=verdict,
        success=bool(payload.get("success", True)),
        duration_s=float(payload.get("duration_s", 30.0)),
        summary=payload.get("summary") or "",
        memory_context=None,
        latency_ms=payload.get("latency_ms"),
        forced_episode_id=payload.get("episode_id"),
        skip_if_duplicate=True,
    )
    logger.info("Experience committed: %s", result.get("episode_id"))
    emit_metric("EpisodesCommitted", 1, "Count")


def handler(event, context):
    batch_failures: List[Dict[str, str]] = []
    for record in event.get("Records") or []:
        msg_id = record.get("messageId") or ""
        try:
            body = record.get("body") or "{}"
            payload = json.loads(body) if isinstance(body, str) else body
            _commit_payload(payload)
        except Exception as exc:
            logger.exception("Learning record failed: %s", exc)
            if msg_id:
                batch_failures.append({"itemIdentifier": msg_id})
    return {"batchItemFailures": batch_failures}
