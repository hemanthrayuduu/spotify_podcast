"""Integration tests for POST /recommend: LLM candidates → iTunes-verified, ranked shows."""

from datetime import date

import pytest_asyncio

from app.core.config import settings
from app.main import app
from app.services.catalog import ITunesCatalog
from app.services.recommender import Recommender
from tests.itunes_fakes import FakeITunes, episode, show
from tests.llm_fakes import FakeGroq, suggestions_json

VALID_BODY = {
    "age": "20_34",
    "music_genre": ["jazz"],
    "podcast_frequency": "daily",
    "podcast_duration": "long",
    "podcast_format": "storytelling",
    "podcast_content": ["true_crime"],
    "content_language": "english",
    "region": "global",
    "listening_mood": "learn",
    "podcasts_enjoyed": "Serial",
}

SHOWS = [
    show(101, "Casefile True Crime", "Casefile Presents", genres=("True Crime",)),
    show(102, "Crime Junkie", "audiochuck", genres=("True Crime",)),
    show(103, "Morbid", "Morbid Network | Wondery", genres=("True Crime", "Comedy")),
    show(104, "Criminal", "Vox Media Podcast Network", genres=("True Crime",)),
    show(105, "Swindled", "audiochuck", genres=("True Crime",)),
    show(106, "Serial", "Serial Productions & The New York Times", genres=("True Crime",)),
    show(107, "True Crime Garage", "Nic and the Captain", genres=("True Crime",)),
    show(108, "Explicit Crime Pod", "Late Night Audio", genres=("True Crime",), explicit=True),
]
EPISODES = {s["collectionId"]: [episode(s["collectionId"], 45), episode(s["collectionId"], 50)] for s in SHOWS}
CHARTS = {1488: [102, 107, 101, 108]}  # Apple's True Crime chart (genre 1488)

LLM_OUTPUT = suggestions_json(
    ("Casefile True Crime", "Casefile Presents"),
    ("Crime Junkie", "audiochuck"),
    ("Morbid", "Alaina Urquhart and Ash Kelley"),
    ("Criminal", "Phoebe Judge"),
    ("The Imaginary Murders Hour", "Nobody"),  # hallucinated: not in the catalog
    ("Serial", "Sarah Koenig"),  # already enjoyed: must be excluded
    ("Swindled", "audiochuck"),  # same publisher as Crime Junkie
)


@pytest_asyncio.fixture
async def make_client(client):
    """Install a Recommender wired to fake iTunes + fake Groq; returns (client, fake_itunes)."""
    original = app.state.recommender

    def _make(llm="default", itunes=None):
        llm = FakeGroq([LLM_OUTPUT]) if llm == "default" else llm
        itunes = itunes or FakeITunes(SHOWS, EPISODES, CHARTS)
        app.state.recommender = Recommender(
            segmenter=app.state.segmenter,
            catalog=ITunesCatalog(itunes.client(), retry_delay=0),
            llm_client=llm,
            model=settings.groq_model,
            today=lambda: date(2026, 10, 14),
        )
        return client, itunes

    yield _make
    app.state.recommender = original


async def test_ai_mode_returns_only_verified_shows(make_client):
    client, _ = make_client()
    response = await client.post("/recommend", json=VALID_BODY)
    assert response.status_code == 200
    data = response.json()
    recs = data["recommendations"]
    ids = [r["id"] for r in recs]
    catalog_ids = {s["collectionId"] for s in SHOWS}

    assert len(recs) == 5
    assert set(ids) <= catalog_ids  # nothing the catalog doesn't know
    assert 106 not in ids  # already enjoyed
    assert len({r["publisher"] for r in recs}) == 5  # one per publisher
    assert data["meta"]["mode"] == "ai"
    assert data["meta"]["llm_candidates"] == 7
    assert data["meta"]["llm_verified"] == 6  # all but the imaginary show
    assert data["meta"]["dropped_unverified"] == 1  # the imaginary show


async def test_recommendations_carry_real_catalog_metadata(make_client):
    client, _ = make_client()
    recs = (await client.post("/recommend", json=VALID_BODY)).json()["recommendations"]
    casefile = next(r for r in recs if r["id"] == 101)
    assert casefile["title"] == "Casefile True Crime"
    assert casefile["publisher"] == "Casefile Presents"
    assert casefile["artwork_url"].startswith("https://")
    assert casefile["apple_url"].endswith("/id101")
    assert casefile["spotify_url"].startswith("https://open.spotify.com/search/")
    assert casefile["typical_episode_minutes"] == 48
    assert casefile["last_episode_date"] == "2026-10-01"
    assert casefile["source"] == "ai"
    assert casefile["reason"] == "Why Casefile True Crime."
    assert 0 <= casefile["match"]["score"] <= 100
    assert set(casefile["match"]["breakdown"]) == {"topic", "duration", "endorsement", "freshness"}


async def test_short_ai_list_is_topped_up_from_catalog(make_client):
    client, _ = make_client()
    recs = (await client.post("/recommend", json=VALID_BODY)).json()["recommendations"]
    # 4 distinct-publisher AI picks survive, so one chart pick fills the 5th slot.
    catalog = [r for r in recs if r["source"] == "catalog"]
    assert len(catalog) == 1
    assert catalog[0]["id"] == 107  # next-best chart show not already recommended
    assert catalog[0]["reason"] == "#2 in Apple Podcasts' True Crime & Mystery chart (US)."


async def test_catalog_only_mode_without_llm(make_client):
    client, _ = make_client(llm=None)
    data = (await client.post("/recommend", json=VALID_BODY)).json()
    assert data["meta"]["mode"] == "catalog_only"
    assert [r["id"] for r in data["recommendations"]] == [102, 107, 101, 108]  # chart order
    assert all(r["source"] == "catalog" for r in data["recommendations"])
    assert all("chart" in r["reason"] for r in data["recommendations"])


async def test_llm_failure_degrades_to_catalog_only(make_client):
    client, _ = make_client(llm=FakeGroq([RuntimeError("Groq is down")]))
    response = await client.post("/recommend", json=VALID_BODY)
    assert response.status_code == 200
    assert response.json()["meta"]["mode"] == "catalog_only"


async def test_catalog_outage_returns_503(make_client):
    itunes = FakeITunes(SHOWS, EPISODES, CHARTS)
    itunes.fail_with = 503
    client, _ = make_client(itunes=itunes)
    response = await client.post("/recommend", json=VALID_BODY)
    assert response.status_code == 503
    assert "catalog" in response.json()["detail"].lower()


async def test_minors_never_get_explicit_shows(make_client):
    client, itunes = make_client(llm=None)
    data = (await client.post("/recommend", json={**VALID_BODY, "age": "12_19"})).json()
    assert 108 not in [r["id"] for r in data["recommendations"]]
    searches = [r for r in itunes.requests if r.url.path == "/search"]
    assert searches and all("explicit=No" in str(r.url) for r in searches)


async def test_identical_requests_are_served_from_cache(make_client):
    client, itunes = make_client()
    await client.post("/recommend", json=VALID_BODY)
    calls = len(itunes.requests)
    second = (await client.post("/recommend", json=VALID_BODY)).json()
    assert len(itunes.requests) == calls
    assert second["meta"]["cached"] is True


async def test_response_includes_persona(make_client):
    client, _ = make_client()
    data = (await client.post("/recommend", json=VALID_BODY)).json()
    assert data["persona"]["name"] and data["persona"]["traits"]


async def test_recommend_empty_list_returns_422(client):
    response = await client.post("/recommend", json={**VALID_BODY, "music_genre": []})
    assert response.status_code == 422


async def test_recommend_missing_field_returns_422(client):
    body = {k: v for k, v in VALID_BODY.items() if k != "region"}
    response = await client.post("/recommend", json=body)
    assert response.status_code == 422


async def test_recommend_unknown_option_returns_422(client):
    response = await client.post("/recommend", json={**VALID_BODY, "age": "25-34"})
    assert response.status_code == 422
    assert "20_34" in response.text  # the error lists allowed values


async def test_shows_in_the_wrong_language_are_dropped(make_client):
    itunes = FakeITunes(SHOWS, EPISODES, CHARTS, languages={102: "es", 107: "es-ES", 101: "en", 108: "es"})
    client, _ = make_client(llm=None, itunes=itunes)
    data = (await client.post("/recommend", json={**VALID_BODY, "content_language": "spanish"})).json()
    assert [r["id"] for r in data["recommendations"]] == [102, 107, 108]
    assert all(r["language"] == "es" for r in data["recommendations"])
    assert data["meta"]["dropped_language"] == 1


async def test_recommendations_report_feed_language(make_client):
    client, _ = make_client()
    recs = (await client.post("/recommend", json=VALID_BODY)).json()["recommendations"]
    assert {r["language"] for r in recs} == {"en"}


async def test_broken_chart_feed_does_not_break_ai_mode(make_client):
    itunes = FakeITunes(SHOWS, EPISODES, CHARTS)
    itunes.chart_body = "<html>maintenance</html>"
    client, _ = make_client(itunes=itunes)
    response = await client.post("/recommend", json=VALID_BODY)
    assert response.status_code == 200
    assert response.json()["meta"]["mode"] == "ai"


async def test_malformed_feed_url_does_not_break_the_request(make_client):
    shows = [dict(s, feedUrl=s["feedUrl"] + "\n") if s["collectionId"] == 107 else s for s in SHOWS]
    client, _ = make_client(llm=None, itunes=FakeITunes(shows, EPISODES, CHARTS))
    response = await client.post("/recommend", json=VALID_BODY)
    assert response.status_code == 200
    assert 107 in [r["id"] for r in response.json()["recommendations"]]


async def test_charts_are_not_fetched_when_the_llm_fills_the_list(make_client):
    output = suggestions_json(
        ("Casefile True Crime", "Casefile Presents"), ("Crime Junkie", "audiochuck"),
        ("Morbid", "Ash Kelley"), ("Criminal", "Phoebe Judge"), ("True Crime Garage", "Nic"),
    )
    client, itunes = make_client(llm=FakeGroq([output]))
    data = (await client.post("/recommend", json=VALID_BODY)).json()
    assert len(data["recommendations"]) == 5
    assert not [r for r in itunes.requests if "/rss/" in r.url.path]


async def test_enjoyed_show_is_excluded_even_if_its_search_fails(make_client):
    itunes = FakeITunes(SHOWS, EPISODES, CHARTS)
    # The enjoyed-show lookup is rate limited (both attempts); the LLM's later
    # search for the same title succeeds and finds Serial.
    itunes.fail_first = {"Serial": 2}  # first call + its retry
    client, _ = make_client(itunes=itunes)
    data = (await client.post("/recommend", json=VALID_BODY)).json()
    assert 106 not in [r["id"] for r in data["recommendations"]]


async def test_failed_lookups_are_reported_not_called_unverified(make_client):
    itunes = FakeITunes(SHOWS, EPISODES, CHARTS)
    itunes.fail_terms = {"Casefile True Crime"}
    client, _ = make_client(itunes=itunes)
    meta = (await client.post("/recommend", json=VALID_BODY)).json()["meta"]
    assert meta["dropped_unverified"] == 1  # only the imaginary show
    assert meta["catalog_errors"] >= 1


async def test_degraded_results_are_not_cached(make_client):
    itunes = FakeITunes(SHOWS, EPISODES, CHARTS)
    itunes.fail_terms = {"Casefile True Crime"}
    client, _ = make_client(itunes=itunes)
    await client.post("/recommend", json=VALID_BODY)
    itunes.fail_terms = set()
    second = (await client.post("/recommend", json=VALID_BODY)).json()
    assert second["meta"]["cached"] is False
    assert 101 in [r["id"] for r in second["recommendations"]]


async def test_results_without_the_llm_are_not_cached(make_client):
    client, _ = make_client(llm=FakeGroq([RuntimeError("Groq is down")]))
    await client.post("/recommend", json=VALID_BODY)
    second = (await client.post("/recommend", json=VALID_BODY)).json()
    assert second["meta"]["cached"] is False


async def test_persona_only_shows_traits_the_user_shares(make_client):
    client, _ = make_client(llm=None)
    body = {**VALID_BODY, "age": "60_plus", "podcast_duration": "short"}
    persona = (await client.post("/recommend", json=body)).json()["persona"]
    features = {t["feature"] for t in persona["traits"]}
    assert not features & {"age:35-60", "age:20-35", "podcast_duration:long"}
