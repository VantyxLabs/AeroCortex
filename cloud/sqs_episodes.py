"""SQS FIFO publisher for async learning episodes."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("aerocortex.cloud.sqs_episodes")


def publish_episode(payload: Dict[str, Any], episode_id: str, queue_url: Optional[str] = None) -> bool:
    url = queue_url or os.getenv("EPISODE_QUEUE_URL") or ""
    if not url:
        logger.warning("EPISODE_QUEUE_URL unset; cannot publish episode")
        return False
    try:
        import boto3

        client = boto3.client("sqs")
        client.send_message(
            QueueUrl=url,
            MessageBody=json.dumps(payload, default=str),
            MessageGroupId="aerocortex",
            MessageDeduplicationId=str(episode_id),
        )
        return True
    except Exception as exc:
        logger.warning("SQS publish failed: %s", exc)
        return False
