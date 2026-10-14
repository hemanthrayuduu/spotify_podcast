# Test fixtures

These mirror the backend API contract and are checked by
`backend/tests/test_frontend_contract.py`.

- `options.json` — `GET /options`. Regenerate from `backend/`:
  `python -c "import json; from app.core.vocabulary import options_payload; json.dump(options_payload(), open('../frontend/src/test/fixtures/options.json', 'w'), indent=2, ensure_ascii=False)"`
- `recommend_catalog_only.json` — a real `POST /recommend` response (catalog-only mode, 2026-10-14).
- `recommend_ai.json` — the same shows with the first three marked as AI picks
  (`source: "ai"`, AI-written `summary`/`reason`).
