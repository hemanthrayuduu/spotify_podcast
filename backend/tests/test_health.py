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


async def test_health_reports_llm_status(client):
    body = (await client.get("/health")).json()
    # Tests run without GROQ_API_KEY, so the LLM is reported as not configured.
    assert body["llm"]["configured"] is False
    assert body["llm"]["ok"] is False
    assert body["llm"]["model"]


async def test_health_reports_whether_podcast_index_is_configured(client):
    body = (await client.get("/health")).json()
    assert body["podcast_index"] == {"configured": False}  # tests run without keys
