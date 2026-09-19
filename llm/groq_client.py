import json
import logging
from typing import Any, Dict, Optional

import httpx

from config import config
from llm.ollama_client import SYSTEM_PROMPT

logger = logging.getLogger("aerocortex.groq")


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


class GroqClient:
    """
    Groq OpenAI-compatible chat client for the internet-connected REST API.
    Returns None when the API key is missing or Groq is unreachable so the
    planner can fall back to Ollama (Raspberry Pi) then the offline reasoner.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        connect_timeout: Optional[float] = None,
        enabled: Optional[bool] = None,
    ):
        self.api_key = api_key if api_key is not None else config.llm.groq_api_key
        self.model = model or config.llm.groq_model
        self.base_url = (base_url or config.llm.groq_base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else config.llm.groq_timeout_seconds
        self.connect_timeout = (
            connect_timeout
            if connect_timeout is not None
            else config.llm.groq_connect_timeout_seconds
        )
        flag = config.llm.groq_enabled if enabled is None else enabled
        self.enabled = bool(flag and self.api_key)

    @property
    def is_configured(self) -> bool:
        return self.enabled

    def ping(self) -> bool:
        if not self.enabled:
            return False
        try:
            with httpx.Client(timeout=httpx.Timeout(2.0, connect=self.connect_timeout)) as client:
                resp = client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return resp.status_code == 200
        except Exception as exc:
            logger.warning("Groq ping failed: %s", exc)
            return False

    def generate_json(self, user_prompt: str, system: str = SYSTEM_PROMPT) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            return self._post_chat(user_prompt, system)
        except Exception as exc:
            logger.warning("Groq generate failed: %s", exc)
            return None

    def _post_chat(self, user_prompt: str, system: str) -> Optional[Dict[str, Any]]:
        with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout)) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
            )
            if resp.status_code != 200:
                logger.warning("Groq HTTP %s: %s", resp.status_code, resp.text[:200])
                return None
            body = resp.json()
            choices = body.get("choices") or []
            if not choices:
                return None
            content = (choices[0].get("message") or {}).get("content")
            return _parse_json_content(content)
