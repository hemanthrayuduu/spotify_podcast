"""Tests for the iTunes catalog client and the candidate → show resolver."""

import json
import os
import statistics
from datetime import date

import httpx
import pytest

from app.services.catalog import (
    CatalogError,
    ITunesCatalog,
    RateBudget,
    Show,
    best_match,
    match_enjoyed,
)
from tests.itunes_fakes import FakeITunes, episode, show

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture(name):
    with open(os.path.join(FIXTURES, name)) as f:
        return f.read()


def _static_client(body: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body)))


# --- Parsing real iTunes responses ------------------------------------------


async def test_search_parses_real_itunes_response():
    catalog = ITunesCatalog(_static_client(_fixture("itunes_search_hardcore_history.json")))
    shows = await catalog.search("Hardcore History", country="us")
    first = shows[0]
    assert first.id == 173001861
    assert first.title == "Dan Carlin's Hardcore History"
    assert first.publisher == "Dan Carlin"
    assert first.genres == ["History"]  # the generic "Podcasts" genre is dropped
    assert first.apple_url.endswith("/id173001861")  # tracking query string stripped
    assert first.artwork_url.startswith("https://")
    assert first.explicit is False


async def test_lookup_parses_real_response_into_shows_and_stats():
    catalog = ITunesCatalog(_static_client(_fixture("itunes_lookup_hardcore_history.json")))
    shows, stats = await catalog.lookup([173001861], country="us")
    assert shows[173001861].title == "Dan Carlin's Hardcore History"
    raw = json.loads(_fixture("itunes_lookup_hardcore_history.json"))["results"]
    episodes = [r for r in raw if r["wrapperType"] == "podcastEpisode"]
    s = stats[173001861]
    assert s.typical_minutes == round(statistics.median(r["trackTimeMillis"] / 60000 for r in episodes))
    assert s.last_episode_date == max(date.fromisoformat(r["releaseDate"][:10]) for r in episodes)


async def test_top_chart_parses_real_rss():
    catalog = ITunesCatalog(_static_client(_fixture("itunes_chart_true_crime.json")))
    raw = json.loads(_fixture("itunes_chart_true_crime.json"))["feed"]["entry"]
    assert await catalog.top_chart(1488, country="us") == [int(e["id"]["attributes"]["im:id"]) for e in raw]


# --- Client behaviour ---------------------------------------------------------


async def test_search_is_cached():
    fake = FakeITunes([show(1, "Comedy Bang Bang", "Earwolf")])
    catalog = ITunesCatalog(fake.client())
    await catalog.search("comedy", country="us")
    await catalog.search("comedy", country="us")
    assert fake.paths() == ["/search"]


async def test_lookup_batches_ids_in_chunks():
    shows = [show(i, f"Show {i}", f"P{i}") for i in range(1, 9)]
    fake = FakeITunes(shows, {1: [episode(1, 30), episode(1, 40)], 2: [episode(2, 10)]})
    found, stats = await ITunesCatalog(fake.client()).lookup(list(range(1, 9)), country="us")
    assert fake.paths() == ["/lookup", "/lookup"]  # 8 ids → chunks of 6 + 2
    assert set(found) == set(range(1, 9))
    assert stats[1].typical_minutes == 35
    assert stats[2].typical_minutes == 10


async def test_lookup_batch_size_is_configurable():
    shows = [show(i, f"Show {i}", f"P{i}") for i in range(1, 13)]
    fake = FakeITunes(shows)
    await ITunesCatalog(fake.client()).lookup(list(range(1, 13)), country="us", chunk=10)
    assert fake.paths() == ["/lookup", "/lookup"]  # 12 ids in batches of 10


async def test_lookup_keeps_batches_that_succeed():
    shows = [show(i, f"Show {i}", f"P{i}") for i in range(1, 13)]
    fake = FakeITunes(shows)
    fake.fail_lookup_ids = {11}
    errors = []
    found, _ = await ITunesCatalog(fake.client(), retry_delay=0).lookup(
        list(range(1, 13)), country="us", chunk=10, errors=errors
    )
    assert set(found) == set(range(1, 11))  # the failed batch (11, 12) is just missing
    assert len(errors) == 1


async def test_lookup_raises_when_every_batch_fails():
    fake = FakeITunes([show(1, "Show 1", "P1")])
    fake.fail_lookup_ids = {1}
    with pytest.raises(CatalogError):
        await ITunesCatalog(fake.client(), retry_delay=0).lookup([1], country="us")


async def test_requests_ask_for_english_genre_names():
    fake = FakeITunes([show(1, "Comedy Bang Bang", "Earwolf")])
    catalog = ITunesCatalog(fake.client())
    await catalog.search("comedy", country="es")
    await catalog.lookup([1], country="es")
    assert all("lang=en_us" in str(r.url) for r in fake.requests)


async def test_explicit_filter_is_sent_to_itunes():
    fake = FakeITunes([show(1, "Clean Comedy", "A"), show(2, "Dirty Comedy", "B", explicit=True)])
    shows = await ITunesCatalog(fake.client()).search("comedy", country="us", explicit=False)
    assert [s.id for s in shows] == [1]
    assert "explicit=No" in str(fake.requests[0].url)


async def test_retries_once_on_rate_limit():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, text=json.dumps({"resultCount": 0, "results": []}))

    catalog = ITunesCatalog(httpx.AsyncClient(transport=httpx.MockTransport(handler)), retry_delay=0)
    assert await catalog.search("x", country="us") == []
    assert len(calls) == 2


async def test_raises_catalog_error_when_itunes_keeps_failing():
    fake = FakeITunes([])
    fake.fail_with = 503
    with pytest.raises(CatalogError):
        await ITunesCatalog(fake.client(), retry_delay=0).search("x", country="us")


async def test_feed_language_reads_channel_language():
    feed = "<?xml version='1.0'?><rss><channel><title>T</title><language>es-ES</language><item/>"
    catalog = ITunesCatalog(_static_client(feed))
    assert await catalog.feed_language("https://feeds.example.com/1.xml") == "es"


async def test_feed_language_reads_cdata_wrapped_language():
    # anchor.fm and audioboom feeds wrap the value in CDATA.
    feed = "<rss><channel><title>T</title><language><![CDATA[es]]></language>"
    assert await ITunesCatalog(_static_client(feed)).feed_language("https://a.example/f") == "es"


async def test_feed_language_is_none_when_undeclared_or_unreachable():
    catalog = ITunesCatalog(_static_client("<rss><channel><title>T</title></channel></rss>"))
    assert await catalog.feed_language("https://feeds.example.com/1.xml") is None
    failing = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert await ITunesCatalog(failing).feed_language("https://feeds.example.com/2.xml") is None
    assert await ITunesCatalog(failing).feed_language("file:///etc/passwd") is None


async def test_non_json_response_is_a_catalog_error():
    with pytest.raises(CatalogError):
        await ITunesCatalog(_static_client("<html>maintenance</html>")).search("x", country="us")


async def test_malformed_chart_is_a_catalog_error():
    body = json.dumps({"feed": {"entry": [{"id": {}}]}})
    with pytest.raises(CatalogError):
        await ITunesCatalog(_static_client(body)).top_chart(1488, country="us")


@pytest.mark.parametrize(
    "url",
    ["https://feeds.example.com/107.xml\n", "https://☃.example.com/rss", "https://example.com:8O80/rss"],
)
async def test_feed_language_survives_malformed_urls(url):
    assert await ITunesCatalog(_static_client("<language>en</language>")).feed_language(url) is None


async def test_call_budget_fails_fast_when_exhausted():
    fake = FakeITunes([show(1, "Comedy Bang Bang", "Earwolf")])
    budget = RateBudget(calls_per_minute=2, max_wait=0)
    catalog = ITunesCatalog(fake.client(), budget=budget)
    await catalog.search("a", country="us")
    await catalog.search("b", country="us")
    with pytest.raises(CatalogError, match="budget"):
        await catalog.search("c", country="us")
    assert len(fake.requests) == 2


async def test_cached_responses_do_not_spend_budget():
    fake = FakeITunes([show(1, "Comedy Bang Bang", "Earwolf")])
    catalog = ITunesCatalog(fake.client(), budget=RateBudget(calls_per_minute=1, max_wait=0))
    await catalog.search("a", country="us")
    await catalog.search("a", country="us")  # cached: no budget needed


# --- Resolver -----------------------------------------------------------------


def _show(id, title, publisher):
    return Show(id=id, title=title, publisher=publisher, artwork_url=None, genres=[], apple_url="",
                feed_url=None, episode_count=None, explicit=False)


HARDCORE = _show(1, "Dan Carlin's Hardcore History", "Dan Carlin")
IMPOSTOR = _show(2, "Lex Fridman Podcast", "Fridman Podcast")
LEX = _show(3, "Lex Fridman Podcast", "Lex Fridman")


def test_exact_title_is_accepted():
    assert best_match("Radiolab", "WNYC Studios", [_show(9, "Radiolab", "WNYC Studios")]).id == 9


def test_title_subset_with_matching_host_is_accepted():
    assert best_match("Hardcore History", "Dan Carlin", [HARDCORE]) is HARDCORE


def test_single_word_subset_without_host_is_rejected():
    assert best_match("History", "", [HARDCORE]) is None


def test_hallucinated_title_is_rejected():
    assert best_match("The Quantum Gardener Hour", "Jane Doe", [HARDCORE, LEX]) is None


def test_host_breaks_ties_between_same_titled_shows():
    # iTunes ranks the impostor first here; the host picks the real show.
    assert best_match("Lex Fridman Podcast", "Lex Fridman", [IMPOSTOR, LEX]) is LEX


def test_search_rank_breaks_ties_without_host():
    assert best_match("Lex Fridman Podcast", "", [LEX, IMPOSTOR]) is LEX


@pytest.mark.parametrize(
    "title, host, copycat",
    [
        ("Serial", "Sarah Koenig", ("Serials", "Random")),
        ("Stuff You Should Know", "iHeartPodcasts", ("Stuff You Should Know About Money", "Copycat")),
        ("The Joe Rogan Experience", "Joe Rogan", ("Joe Rogan Experience Review podcast", "Fan")),
    ],
)
def test_copycats_are_rejected_when_the_real_show_is_missing(title, host, copycat):
    assert best_match(title, host, [_show(5, *copycat)]) is None


def test_real_show_under_network_publisher_is_accepted():
    daily = _show(7, "The Daily", "The New York Times")
    assert best_match("The Daily", "Michael Barbaro", [daily]) is daily


def test_short_title_prefix_with_related_publisher_is_accepted():
    freak = _show(8, "Freakonomics Radio", "Freakonomics Radio + Stitcher")
    assert best_match("Freakonomics", "Stephen J. Dubner", [freak]) is freak


def test_user_typed_shorthand_matches_the_real_show():
    # Users type "Hardcore History", not "Dan Carlin's Hardcore History".
    assert match_enjoyed("Hardcore History", [HARDCORE]) is HARDCORE
    assert match_enjoyed("Quantum Gardening", [HARDCORE]) is None
