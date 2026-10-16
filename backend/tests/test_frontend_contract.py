"""Contract tests: the frontend's test fixtures must match what the API serves.

If these fail after an API change, regenerate the fixtures (see
frontend/src/test/fixtures/README.md) and update the frontend.
"""

import json
import os

import pytest

from app.core.vocabulary import options_payload
from app.schemas.recommendation import RecommendationResponse

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "test", "fixtures")


def _load(name):
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


def test_options_fixture_matches_api():
    assert _load("options.json") == options_payload()


@pytest.mark.parametrize("name", ["recommend_ai.json", "recommend_catalog_only.json"])
def test_recommend_fixtures_match_response_schema(name):
    RecommendationResponse.model_validate(_load(name))
