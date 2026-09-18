from sdk.client import AeroCortexClient, AeroCortexError


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
