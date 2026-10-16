"""A fake AsyncGroq client: scripted chat outputs, recorded calls."""

import json


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
    def __init__(self, error=None):
        self._error = error

    async def retrieve(self, model):
        if self._error:
            raise self._error
        return {"id": model}


class FakeGroq:
    """Each chat call returns the next scripted output (the last one repeats)."""

    def __init__(self, outputs, models_error=None):
        self.models = _Models(models_error)
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
