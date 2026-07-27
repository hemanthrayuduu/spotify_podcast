"""Smoke tests for the health/root endpoints."""


async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


async def test_health_reports_models_loaded(client):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
