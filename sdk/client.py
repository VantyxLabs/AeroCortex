"""Thin HTTP client for the AeroCortex REST API.

Transport wrapper only — no second copy of the cognitive pipeline.
Install locally with ``pip install -e .`` (not published to PyPI).
"""

from __future__ import annotations

from typing import Any, Dict, Union

import httpx

from models import UAVTelemetry


class AeroCortexError(Exception):
    """Raised when the API returns a non-2xx status."""

    def __init__(self, status_code: int, detail: Any):
        super().__init__(f"AeroCortex API error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class AeroCortexClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AeroCortexClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        resp = self._client.request(method, path, **kwargs)
        if resp.status_code < 200 or resp.status_code >= 300:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise AeroCortexError(resp.status_code, detail)
        return resp.json()

    def send_telemetry(self, telemetry: Union[dict, UAVTelemetry]) -> dict:
        payload = telemetry.model_dump() if isinstance(telemetry, UAVTelemetry) else telemetry
        return self._request("POST", "/telemetry", json=payload)

    def simulate(self, scenario: str, steps: int = 1, inject_step: int = 1) -> dict:
        return self._request(
            "POST",
            "/simulate",
            json={"scenario": scenario, "steps": steps, "inject_step": inject_step},
        )

    def get_memory(self) -> dict:
        return self._request("GET", "/memory")

    def get_missions(self, limit: int = 50, skip: int = 0) -> dict:
        return self._request("GET", "/missions", params={"limit": limit, "skip": skip})

    def get_status(self) -> dict:
        return self._request("GET", "/status")

    def health(self) -> dict:
        return self._request("GET", "/healthz")

    def reset(self) -> dict:
        return self._request("POST", "/reset")
