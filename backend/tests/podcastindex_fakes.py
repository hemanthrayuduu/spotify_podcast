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

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_with:
            return httpx.Response(self.fail_with, text="nope")
        params = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        langs = params.get("lang", "").split(",")
        feeds = [f for f in self.feeds if f["language"] in langs][: int(params.get("max", 10))]
        return httpx.Response(200, text=json.dumps({"status": "true", "feeds": feeds, "count": len(feeds)}))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
