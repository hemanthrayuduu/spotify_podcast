"""Tests for GET /options."""


async def test_options_lists_form_fields(client):
    response = await client.get("/options")
    assert response.status_code == 200
    fields = {f["name"]: f for f in response.json()["fields"]}
    assert fields["music_genre"]["multi"] is True
    assert fields["age"]["multi"] is False
    assert {"value": "true_crime", "label": "True Crime & Mystery"} in fields["podcast_content"]["options"]
