"""A fake iTunes Search API for tests, served through httpx.MockTransport."""

import json
from typing import Dict, List, Optional
from urllib.parse import parse_qs

import httpx


def show(
    id: int,
    title: str,
    artist: str,
    genres=("Comedy",),
    explicit: bool = False,
    track_count: int = 120,
) -> dict:
    return {
        "wrapperType": "track",
        "kind": "podcast",
        "collectionId": id,
        "trackId": id,
        "collectionName": title,
        "trackName": title,
        "artistName": artist,
        "collectionViewUrl": f"https://podcasts.apple.com/us/podcast/x/id{id}?uo=4",
        "feedUrl": f"https://feeds.example.com/{id}.xml",
        "artworkUrl600": f"https://img.example.com/{id}/600x600bb.jpg",
        "genres": [*genres, "Podcasts"],
        "primaryGenreName": genres[0],
        "trackCount": track_count,
        "collectionExplicitness": "explicit" if explicit else "notExplicit",
        "releaseDate": "2026-10-01T10:00:00Z",
    }


def episode(collection_id: int, minutes: float, date: str = "2026-10-01T10:00:00Z") -> dict:
    return {
        "wrapperType": "podcastEpisode",
        "kind": "podcast-episode",
        "collectionId": collection_id,
        "trackTimeMillis": int(minutes * 60_000),
        "releaseDate": date,
    }


class FakeITunes:
    """Routes /search by term and /lookup by id; records every request."""

    def __init__(
        self,
        shows: List[dict],
        episodes: Optional[Dict[int, List[dict]]] = None,
        charts: Optional[Dict[int, List[int]]] = None,  # genre id → collection ids
        languages: Optional[Dict[int, Optional[str]]] = None,  # id → feed <language> (None: none)
    ):
        self.shows = shows
        self.episodes = episodes or {}
        self.charts = charts or {}
        self.languages = languages or {}
        self.requests: List[httpx.Request] = []
        self.fail_with: Optional[int] = None  # HTTP status to return for every call
        self.fail_terms: set = set()  # /search terms that get a 403 (rate limited)
        self.fail_first: Dict[str, int] = {}  # /search term → number of initial 403s
        self.chart_body: Optional[str] = None  # raw body for chart requests
        self.fail_lookup_ids: set = set()  # /lookup batches containing these ids get a 403

    def _params(self, request: httpx.Request) -> Dict[str, str]:
        return {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_with:
            return httpx.Response(self.fail_with)
        params = self._params(request)
        if params.get("term") in self.fail_terms:
            return httpx.Response(403)
        if self.fail_first.get(params.get("term"), 0) > 0:
            self.fail_first[params["term"]] -= 1
            return httpx.Response(403)
        if request.url.host == "feeds.example.com":
            show_id = int(request.url.path.strip("/").split(".")[0])
            language = self.languages.get(show_id, "en-us")
            tag = f"<language>{language}</language>" if language else ""  # None: undeclared
            return httpx.Response(200, text=f"<?xml version='1.0'?><rss><channel><title>t</title>{tag}")
        if "/rss/toppodcasts/" in request.url.path and self.chart_body is not None:
            return httpx.Response(200, text=self.chart_body)
        if "/rss/toppodcasts/" in request.url.path:
            genre = int(request.url.path.split("genre=")[1].split("/")[0])
            entries = [{"id": {"attributes": {"im:id": str(i)}}} for i in self.charts.get(genre, [])]
            return httpx.Response(200, text=json.dumps({"feed": {"entry": entries}}))
        if request.url.path == "/search":
            words = params["term"].casefold().split()
            results = [
                s for s in self.shows
                if any(w in s["collectionName"].casefold() for w in words)
                and not (params.get("explicit") == "No" and s["collectionExplicitness"] == "explicit")
            ][: int(params.get("limit", 10))]
        else:
            ids = [int(i) for i in params["id"].split(",")]
            if self.fail_lookup_ids & set(ids):
                return httpx.Response(403)
            results = []
            for s in self.shows:
                if s["collectionId"] in ids:
                    results.append(s)
                    results.extend(self.episodes.get(s["collectionId"], [])[: int(params.get("limit", 10))])
        return httpx.Response(200, text=json.dumps({"resultCount": len(results), "results": results}))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def paths(self) -> List[str]:
        return [r.url.path for r in self.requests]
