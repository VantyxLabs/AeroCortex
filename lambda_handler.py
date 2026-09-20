"""API Lambda entrypoint — SSM secrets first, then Mangum(FastAPI)."""

from __future__ import annotations

import os

# Secrets before any config / app import
from cloud.secrets import load_ssm_secrets_into_env

load_ssm_secrets_into_env()

try:
    from aws_xray_sdk.core import patch_all

    patch_all()
except Exception:
    pass

from mangum import Mangum

from api.telemetry_api import app
from observability import setup_logging

setup_logging()

handler = Mangum(app, lifespan="auto")
