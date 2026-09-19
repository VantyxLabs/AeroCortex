"""Minimalist Monochrome mission dashboard — FastAPI static host + API proxy."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import config

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _upstream_api() -> str:
    # Prefer explicit env; otherwise same host as the REST API port from settings.
    explicit = os.getenv("DASHBOARD_API_URL")
    if explicit:
        return explicit.rstrip("/")
    return f"http://127.0.0.1:{config.api_port}".rstrip("/")


app = FastAPI(
    title="AeroCortex Dashboard",
    description="Minimalist Monochrome live mission control",
    version=config.system.version,
)


@app.get("/config.json")
def dashboard_config():
    # Prefer same-origin proxy so the browser never needs a cross-port URL.
    return JSONResponse(
        {
            "api_base_url": "/api",
            "upstream_api_url": _upstream_api(),
            "api_key": config.api_key or "",
            "default_scenarios": [
                "NORMAL",
                "GPS_INTERFERENCE",
                "GPS_LOSS",
                "BATTERY_DEGRADATION",
                "LOW_BATTERY",
                "COMMUNICATION_LOSS",
                "STRONG_WIND",
                "SENSOR_ANOMALY",
                "COMBINED_FAILURE",
            ],
            "poll_interval_ms": 30000,
            "version": config.system.version,
        }
    )


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_api(path: str, request: Request):
    """Same-origin proxy to the AeroCortex REST API (avoids CORS / wrong-port fetch failures)."""
    upstream = f"{_upstream_api()}/{path}"
    if request.url.query:
        upstream = f"{upstream}?{request.url.query}"
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in ("host", "content-length", "connection")
    }
    body = await request.body()
    timeout = httpx.Timeout(90.0, connect=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream_resp = await client.request(
                request.method,
                upstream,
                headers=headers,
                content=body,
            )
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "detail": f"Upstream API unreachable at {_upstream_api()}: {exc}",
            },
        )
    excluded = {"content-encoding", "transfer-encoding", "connection"}
    out_headers = {
        key: value
        for key, value in upstream_resp.headers.items()
        if key.lower() not in excluded
    }
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=out_headers,
        media_type=upstream_resp.headers.get("content-type"),
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def dashboard_health():
    return {"status": "ok", "service": "dashboard", "upstream": _upstream_api()}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
