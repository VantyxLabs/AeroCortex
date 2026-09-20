"""Load secrets from SSM Parameter Store into os.environ (Lambda cold start)."""

from __future__ import annotations

import logging
import os
from typing import Dict, List

logger = logging.getLogger("aerocortex.cloud.secrets")

_LOADED = False

# Local env name -> SSM suffix under SSM_PREFIX
_PARAM_MAP = {
    "API_KEY": "API_KEY",
    "GROQ_API_KEY": "GROQ_API_KEY",
    "PINECONE_API_KEY": "PINECONE_API_KEY",
}


def load_ssm_secrets_into_env(prefix: str | None = None, force: bool = False) -> None:
    """
    If SSM_PREFIX is set (e.g. /aerocortex/prod), fetch SecureString params
    and inject into os.environ when the key is missing or empty.
    Cached for the process lifetime.
    """
    global _LOADED
    if _LOADED and not force:
        return
    prefix = (prefix or os.getenv("SSM_PREFIX") or "").rstrip("/")
    if not prefix:
        _LOADED = True
        return

    names = [f"{prefix}/{suffix}" for suffix in _PARAM_MAP.values()]
    try:
        import boto3

        client = boto3.client("ssm")
        # get_parameters max 10 names
        resp = client.get_parameters(Names=names, WithDecryption=True)
        found: Dict[str, str] = {}
        for p in resp.get("Parameters") or []:
            name = p["Name"]
            for env_key, suffix in _PARAM_MAP.items():
                # Exact path match only — endswith("API_KEY") wrongly matches GROQ_API_KEY
                if name == f"{prefix}/{suffix}":
                    found[env_key] = p["Value"]
        for env_key, value in found.items():
            if value:
                # SSM is source of truth on Lambda; always overlay
                os.environ[env_key] = value
        logger.info("SSM secrets loaded for: %s", sorted(found.keys()))
        missing = [n for n in names if n not in {p["Name"] for p in (resp.get("Parameters") or [])}]
        if missing:
            logger.warning("SSM parameters missing: %s", missing)
        if resp.get("InvalidParameters"):
            logger.warning("SSM invalid parameters: %s", resp["InvalidParameters"])
    except Exception as exc:
        logger.warning("SSM secret load failed: %s", exc)
    _LOADED = True
