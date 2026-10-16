"""A fake AsyncGroq client: scripted chat outputs, recorded calls."""

import json

import groq
import httpx


class _Message:
    def __init__(self, content):
        self.content = content


class _Response:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": _Message(content)})()]


class _Completions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        return _Response(out)


class _Models:
    """Mirrors the real API: ids contain slashes (openai/gpt-oss-120b)."""

    def __init__(self, error=None, available=("openai/gpt-oss-120b", "openai/gpt-oss-20b")):
        self._error = error
        self._available = available

    async def list(self):
        if self._error:
            raise self._error
        return type("ModelList", (), {"data": [type("Model", (), {"id": m})() for m in self._available]})()

    async def retrieve(self, model):
        # groq>=1.x percent-encodes the id into the URL path, so the real API
        # 404s on "openai%2Fgpt-oss-120b"; mimic that.
        raise RuntimeError(f"model_not_found: {model.replace('/', '%2F')}")


class FakeGroq:
    """Each chat call returns the next scripted output (the last one repeats)."""

    def __init__(self, outputs, models_error=None, available=("openai/gpt-oss-120b", "openai/gpt-oss-20b")):
        self.models = _Models(models_error, available)
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(outputs)

    @property
    def calls(self):
        return self.chat.completions.calls


def suggestions_json(*pairs):
    """[(title, host), ...] → the JSON the model returns."""
    return json.dumps(
        {
            "candidates": [
                {"title": t, "host": h, "summary": f"{t} summary.", "reason": f"Why {t}."}
                for t, h in pairs
            ]
        }
    )


def json_validate_failed(failed_generation="I'm sorry, but I can't find enough podcasts."):
    """The 400 Groq returns when strict JSON decoding fails (e.g. the model refuses in prose)."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {
        "error": {
            "message": "Failed to generate JSON. Please adjust your prompt.",
            "type": "invalid_request_error",
            "code": "json_validate_failed",
            "failed_generation": failed_generation,
        }
    }
    response = httpx.Response(400, request=request, json=body)
    return groq.BadRequestError("Error code: 400", response=response, body=body)
