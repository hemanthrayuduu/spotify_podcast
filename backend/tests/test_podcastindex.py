"""Tests for the Podcast Index client (language-filtered trending shows)."""

import hashlib
import json

import httpx
import pytest

from app.services.podcastindex import PodcastIndex, PodcastIndexError
from tests.podcastindex_fakes import FakePodcastIndex, feed


def _pi(fake, **kw):
    return PodcastIndex(fake.client(), key="KEY", secret="SECRET", clock=lambda: 1792000000.7, **kw)


async def test_requests_are_signed_as_the_api_requires():
    fake = FakePodcastIndex([feed(1, "Show", 111)])
    await _pi(fake).trending("te", [104, 105])
    headers = fake.requests[0].headers
    assert headers["X-Auth-Key"] == "KEY"
    assert headers["X-Auth-Date"] == "1792000000"
    assert headers["Authorization"] == hashlib.sha1(b"KEYSECRET1792000000").hexdigest()
    assert headers["User-Agent"].startswith("spotify-podcast-recommender/")


async def test_trending_filters_by_language_and_categories():
    fake = FakePodcastIndex([feed(1, "Telugu Show", 111), feed(2, "Hindi Show", 222, language="hi")])
    shows = await _pi(fake).trending("te", [104, 105], max_results=15)
    url = str(fake.requests[0].url)
    assert "lang=te" in url and "cat=104%2C105" in url and "max=15" in url
    assert [(s.feed_id, s.itunes_id, s.title, s.language) for s in shows] == [(1, 111, "Telugu Show", "te")]


async def test_missing_itunes_ids_are_none():
    fake = FakePodcastIndex([feed(1, "A", None), feed(2, "B", 0)])
    shows = await _pi(fake).trending("te", [104])
    assert [s.itunes_id for s in shows] == [None, None]


async def test_trending_is_cached():
    fake = FakePodcastIndex([feed(1, "Show", 111)])
    pi = _pi(fake)
    await pi.trending("te", [104])
    await pi.trending("te", [104])
    assert len(fake.requests) == 1


async def test_errors_raise_podcast_index_error():
    fake = FakePodcastIndex([])
    fake.fail_with = 401
    with pytest.raises(PodcastIndexError):
        await _pi(fake).trending("te", [104])
    garbage = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>")))
    with pytest.raises(PodcastIndexError):
        await PodcastIndex(garbage, key="K", secret="S").trending("te", [104])


async def test_region_tagged_feeds_are_found():
    # Feeds declare "te-IN", "es-MX"...; the API matches the raw value, so ask for variants.
    fake = FakePodcastIndex([feed(1, "Telugu Show", 111, language="te-IN"),
                             feed(2, "Plain Telugu", 222, language="te")])
    shows = await _pi(fake).trending("te", [104], region="in")
    assert {s.feed_id for s in shows} == {1, 2}
    assert all(s.language == "te" for s in shows)


async def test_widely_spoken_languages_ask_for_common_regions():
    business = {"9": "Business"}
    fake = FakePodcastIndex([feed(1, "Mexico", 111, language="es-MX", categories=business),
                             feed(2, "LatAm", 222, language="es-419", categories=business)])
    shows = await _pi(fake).trending("es", [9], region="es")
    assert {s.feed_id for s in shows} == {1, 2}


async def test_malformed_feeds_are_skipped_not_fatal():
    fake = FakePodcastIndex([])
    fake.raw_body = json.dumps({"feeds": [{"id": "abc", "itunesId": 5}, feed(2, "Good", 222)]})
    shows = await _pi(fake).trending("te", [104])
    assert [s.feed_id for s in shows] == [2]
