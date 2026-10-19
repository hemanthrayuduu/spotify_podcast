"""A fake Podcast Index API served through httpx.MockTransport."""

import json
from typing import Dict, List, Optional
from urllib.parse import parse_qs

import httpx


def feed(feed_id: int, title: str, itunes_id: Optional[int], language: str = "te",
         categories: Optional[Dict[str, str]] = None) -> dict:
    return {
        "id": feed_id,
        "url": f"https://feeds.example.com/{itunes_id or feed_id}.xml",
        "title": title,
        "author": f"Author {feed_id}",
        "itunesId": itunes_id,
        "language": language,
        "categories": categories or {"104": "TV", "105": "Film"},
        "newestItemPublishTime": 1792000000,
        "trendScore": 9,
    }


class FakePodcastIndex:
    """Serves /podcasts/trending filtered by lang; records requests."""

    def __init__(self, feeds: List[dict]):
        self.feeds = feeds
        self.requests: List[httpx.Request] = []
        self.fail_with: Optional[int] = None
        self.raw_body: Optional[str] = None  # returned verbatim when set

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_with:
            return httpx.Response(self.fail_with, text="nope")
        if self.raw_body is not None:
            return httpx.Response(200, text=self.raw_body)
        params = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        # Case-insensitive *exact* match on the raw feed value ("te-IN" needs "te-in").
        langs = {lang.lower() for lang in params.get("lang", "").split(",")}
        cats = {c for c in params.get("cat", "").split(",") if c}  # any of them
        feeds = [
            f for f in self.feeds
            if str(f["language"]).lower() in langs and (not cats or cats & set(f["categories"]))
        ][: int(params.get("max", 10))]
        return httpx.Response(200, text=json.dumps({"status": "true", "feeds": feeds, "count": len(feeds)}))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
