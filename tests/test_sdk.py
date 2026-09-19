from sdk.client import AeroCortexClient, AeroCortexError
from models import UAVTelemetry


def test_sdk_surface():
    client = AeroCortexClient("http://localhost:8000", api_key="change-me-local-dev-key")
    try:
        assert hasattr(client, "send_telemetry")
        assert hasattr(client, "simulate")
        assert hasattr(client, "get_memory")
        assert hasattr(client, "get_missions")
        assert hasattr(client, "get_status")
        assert hasattr(client, "health")
        assert hasattr(client, "reset")
    finally:
        client.close()


def test_sdk_error_type():
    err = AeroCortexError(401, {"detail": "Invalid or missing API key"})
    assert err.status_code == 401
    assert "401" in str(err)


class _FakeResp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_sdk_send_telemetry_sets_api_key(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["headers"] = kwargs.get("headers")
            captured["base_url"] = kwargs.get("base_url")

        def request(self, method, path, **kwargs):
            captured["method"] = method
            captured["path"] = path
            captured["json"] = kwargs.get("json")
            return _FakeResp(200, {"status": "SUCCESS", "action": "CONTINUE_MISSION"})

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr("sdk.client.httpx.Client", FakeClient)
    telemetry = UAVTelemetry(mission_id="SDK_1")
    with AeroCortexClient("http://localhost:8000/", api_key="secret-key") as client:
        out = client.send_telemetry(telemetry)
    assert captured["headers"]["X-API-Key"] == "secret-key"
    assert captured["base_url"] == "http://localhost:8000"
    assert captured["method"] == "POST"
    assert captured["path"] == "/telemetry"
    assert captured["json"]["mission_id"] == "SDK_1"
    assert out["status"] == "SUCCESS"
    assert captured.get("closed") is True


def test_sdk_health_and_missions(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def request(self, method, path, **kwargs):
            calls.append((method, path, kwargs.get("params")))
            if path == "/healthz":
                return _FakeResp(200, {"status": "ok"})
            if path == "/missions":
                return _FakeResp(200, {"history": [], "limit": kwargs["params"]["limit"]})
            return _FakeResp(200, {})

        def close(self):
            pass

    monkeypatch.setattr("sdk.client.httpx.Client", FakeClient)
    client = AeroCortexClient("http://localhost:8000", api_key="k")
    assert client.health()["status"] == "ok"
    missions = client.get_missions(limit=10, skip=2)
    assert missions["limit"] == 10
    client.close()
    assert calls[0][:2] == ("GET", "/healthz")
    assert calls[1] == ("GET", "/missions", {"limit": 10, "skip": 2})


def test_sdk_non_2xx_raises(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def request(self, method, path, **kwargs):
            return _FakeResp(401, {"detail": "Invalid or missing API key"})

        def close(self):
            pass

    monkeypatch.setattr("sdk.client.httpx.Client", FakeClient)
    client = AeroCortexClient("http://localhost:8000", api_key="bad")
    try:
        raised = None
        try:
            client.get_status()
        except AeroCortexError as exc:
            raised = exc
        assert raised is not None
        assert raised.status_code == 401
        assert raised.detail["detail"] == "Invalid or missing API key"
    finally:
        client.close()
