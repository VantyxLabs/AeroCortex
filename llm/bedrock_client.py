"""Amazon Bedrock Converse client — mirrors GroqClient interface."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from llm.ollama_client import SYSTEM_PROMPT

logger = logging.getLogger("aerocortex.bedrock")

# Last call outcome for /healthz (never invoke from healthz)
_LAST_OK: Optional[bool] = None


def last_call_ok() -> Optional[bool]:
    return _LAST_OK


def _parse_json_content(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        fence = text.rfind("```")
        if fence >= 0:
            text = text[:fence]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class BedrockClient:
    """
    Bedrock Runtime Converse API. Skipped when BEDROCK_MODEL_ID is unset.
    Interface matches GroqClient: generate_json, ping, is_configured.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.model_id = (model_id if model_id is not None else os.getenv("BEDROCK_MODEL_ID") or "").strip()
        self.region = (
            region
            or os.getenv("BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "ap-south-1"
        )
        self.timeout = float(timeout if timeout is not None else os.getenv("BEDROCK_TIMEOUT_S") or 8)
        self.enabled = bool(self.model_id)

    @property
    def is_configured(self) -> bool:
        return self.enabled

    def ping(self) -> bool:
        """Do not call Bedrock from healthz. Reflect last generate outcome only."""
        if not self.enabled:
            return False
        if _LAST_OK is None:
            return True  # configured but unused yet — treat as ok for readiness
        return bool(_LAST_OK)

    def generate_json(self, user_prompt: str, system: str = SYSTEM_PROMPT) -> Optional[Dict[str, Any]]:
        global _LAST_OK
        if not self.enabled:
            return None
        try:
            import boto3
            from botocore.config import Config

            cfg = Config(
                connect_timeout=min(2.0, self.timeout),
                read_timeout=self.timeout,
                retries={"max_attempts": 1},
            )
            client = boto3.client("bedrock-runtime", region_name=self.region, config=cfg)
            resp = client.converse(
                modelId=self.model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={"temperature": 0.2, "maxTokens": 1024},
            )
            parts = resp.get("output", {}).get("message", {}).get("content") or []
            text = ""
            for p in parts:
                if isinstance(p, dict) and "text" in p:
                    text += p["text"]
            parsed = _parse_json_content(text)
            _LAST_OK = parsed is not None
            return parsed
        except Exception as exc:
            logger.warning("Bedrock generate failed: %s", exc)
            _LAST_OK = False
            return None
