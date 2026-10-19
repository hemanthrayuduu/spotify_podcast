"""Invariants for the shared form vocabulary (app.core.vocabulary)."""

import pytest

from app.core import vocabulary
from app.core.vocabulary import FIELDS


@pytest.mark.parametrize("field", FIELDS.values(), ids=lambda f: f.name)
def test_option_values_are_unique(field):
    values = [o.value for o in field.options]
    assert len(values) == len(set(values))


@pytest.mark.parametrize("field", FIELDS.values(), ids=lambda f: f.name)
def test_option_codes_are_declared_segment_codes(field):
    for option in field.options:
        assert option.code is None or option.code in field.segment_codes


@pytest.mark.parametrize("field", FIELDS.values(), ids=lambda f: f.name)
def test_every_segment_code_is_reachable_from_an_option(field):
    reachable = {o.code for o in field.options}
    assert set(field.segment_codes) <= reachable


def test_topics_have_catalog_mapping():
    for option in FIELDS["podcast_content"].options:
        assert option.chart_genre_id, option.value
        assert option.apple_genres, option.value


def test_durations_have_minute_bands():
    for option in FIELDS["podcast_duration"].options:
        assert option.minutes is not None, option.value


def test_regions_have_storefronts():
    for option in FIELDS["region"].options:
        assert option.storefront, option.value


def test_to_labels_maps_values_to_labels():
    labels = vocabulary.to_labels({"age": "20_34", "music_genre": ["jazz", "hip_hop"]})
    assert labels == {"age": "20–34", "music_genre": ["Jazz", "Hip-Hop / Rap"]}


def test_options_payload_lists_every_field():
    payload = vocabulary.options_payload()
    names = [f["name"] for f in payload["fields"]]
    assert names == list(FIELDS)
    age = payload["fields"][0]
    assert age["options"][0] == {"value": "under_12", "label": "Under 12"}


def test_languages_have_iso_codes():
    for option in FIELDS["content_language"].options:
        assert option.iso_language and len(option.iso_language) == 2, option.value


def test_topics_map_to_podcast_index_categories():
    # Category ids from the Podcast Namespace list (podcast-namespace/categories.json).
    for option in FIELDS["podcast_content"].options:
        assert option.pi_categories, option.value
        assert all(1 <= c <= 112 for c in option.pi_categories), option.value
    assert vocabulary.option("podcast_content", "tv_film").pi_categories == (104, 105)
