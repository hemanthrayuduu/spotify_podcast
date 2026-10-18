# Spotify Podcast Recommender

A personalized podcast recommendation system powered by machine learning and AI.

## Overview

Spotify Podcast Recommender analyzes user preferences and listening habits to provide tailored podcast recommendations. The application uses segmentation algorithms and an open-weight LLM (gpt-oss, via Groq) to generate highly personalized content suggestions.

## Features

- **Personalized Recommendations**: Uses machine learning to match podcasts to user preferences
- **User Segmentation**: Identifies listening profiles to better understand user interests
- **AI-Enhanced Suggestions**: Leverages an open-weight LLM (gpt-oss, via Groq) to generate contextually relevant recommendations
- **Library Management**: Save favorite podcasts for future reference
- **Modern UI**: Responsive design with Spotify-inspired aesthetics

## Technology Stack

### Frontend
- React 19 + React Router 7, built with Vite and tested with Vitest
- The form is rendered from `GET /options`, so it always matches the API
- CSS with modern features (glassmorphism, animations, responsive design)
- LocalStorage for client-side podcast saving

### Backend
- FastAPI (Python) for efficient API endpoints
- KMeans listener personas trained on a 28k-response Spotify user survey (numpy-only at serving time)
- Open-weight LLM (gpt-oss via Groq) integration for recommendation intelligence
- pandas + scikit-learn for offline training only (`requirements-train.txt`)

## Installation

### Prerequisites
- Node.js 22.12+ (CI and Netlify use Node 24)
- Python 3.11+
- Groq API key (free at https://console.groq.com)

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# Create .env file with GROQ_API_KEY=your_api_key (see .env.example)
uvicorn app.main:app --reload

# To retrain the persona model from the survey data:
pip install -r requirements-train.txt
python train.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev      # http://localhost:3000, talks to VITE_API_URL (default http://localhost:8000)
npm test         # Vitest
npm run build    # outputs dist/
```

## Usage

1. Fill out the preference form with your listening habits and interests
2. Receive personalized podcast recommendations based on your profile
3. Save interesting podcasts to your library
4. Explore recommendations with direct links to find podcasts on Spotify

## API Endpoints

- `GET /options`: Form fields and the allowed option values (the frontend renders its form from this)
- `POST /recommend`: Submit option values and receive a listener persona + recommendations
- `GET /health`: Readiness, persona-model version, and LLM status
- `GET /`: API information

## Configuration

The backend reads configuration from environment variables (see `.env.example`):

- `GROQ_API_KEY` — enables LLM recommendations; falls back to a static list if unset.
- `GROQ_MODEL` — Groq model (default `openai/gpt-oss-120b`). The model is checked at startup; `GET /health` reports `llm.ok` and the error if it is unavailable.
- `ALLOWED_ORIGINS` — comma-separated CORS origins; set to your frontend URL in production.

## Docker

```bash
cd backend
docker build -t podcast-recommender .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key podcast-recommender
```

The image runs as a non-root user and serves on `$PORT` (default 8000).

## Deployment

The application is configured for deployment on:
- Frontend: Netlify
- Backend: Render — start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health check path `/health`. Rate limiting keys on the client IP Render appends to `X-Forwarded-For` (`app/core/limiter.py`).

## How recommendations work

Every show the API returns exists in Apple's podcast catalog. Its artwork, links, genres,
episode lengths, last-episode date and language come from the catalog and the show's own RSS
feed, never from the LLM.

1. **Persona.** The user's answers are mapped to a survey persona (see below), which is passed
   to the LLM as context.
2. **Candidate list.** Apple's top charts for each chosen topic (up to 3, top 15 each) in the
   user's storefront. Batched iTunes lookups give each show's real median episode length and
   last-episode date. Each show's RSS feed is read for the language it declares.
3. **Filter.** Removed from the list:
   - shows the user already enjoys, and explicit shows for under-20s;
   - trailer-only feeds, and shows in another language;
   - more than one show per publisher.

   The best 30 by the score below are kept.
4. **The AI picks.** `openai/gpt-oss-120b` on Groq sees that list (title, publisher, genres,
   real length, recency) and picks up to 8 by id, with a reason for each, through strict
   JSON-schema output. An id that isn't on the list is ignored, so picks can't be invented.

   It may also suggest up to 2 shows of its own, or more when few listed shows are in the
   user's language. Each is searched in the iTunes catalog and fuzzy-matched on title and host
   (`app/services/catalog.py`); the host check matters because copycat feeds reuse popular
   titles. A suggestion with no confident match is dropped and counted in
   `meta.dropped_unverified`.
5. **Rank.** A transparent 0–100 score over the list and the verified suggestions
   (`app/services/ranking.py`), returned with its breakdown.
   - **Score:** topic match on Apple genres (35%), fit of the real episode length to the
     requested band (30%), AI endorsement (20%), freshness (15%).
   - **Language:** shows whose feed confirms the requested language come before feeds that
     declare none.
6. **Fall back.** If the LLM is unavailable, the list is ranked directly (`meta.mode =
   "catalog_only"`), so nothing is invented. If the catalog itself is unreachable, the API
   returns 503.

The iTunes API allows roughly 20 calls per minute per IP. Responses are cached (6 h for
catalog data, 10 min per identical request), concurrency is capped, and lookups are batched.

**Measuring it.** `python -m scripts.eval_recommendations` runs 14 diverse listener profiles
against live Groq and iTunes, throttled to the production call budget. It reports how often the
AI's proposals are real shows, how much of the top 5 the AI picked, and how often the returned
shows match the requested length, language and topic.

Results on 2026-10-17, for the original 12 profiles, comparing the AI naming shows from memory
with the AI picking from the chart list:

| Metric | From memory | From the chart list |
|---|---|---|
| AI proposals that are real shows | 50% | 98% (list picks are real by construction) |
| Shown shows verified in Apple's catalog | 100% | 100% |
| Real episode length in the requested band | 53% | 68% |
| Feed language matches | 98% | 98% |
| Topic match | 100% | 100% |
| Share of the top 5 picked by the AI | — | 58% |

Length misses come from genres whose charts have few shows in the requested band, such as
short news or extra-long true crime. Languages with few shows in Apple's charts, such as
Telugu, return fewer than 5 recommendations; the AI declines to invent shows to fill the gap.

## Listener personas (survey model)

`backend/train.py` trains a KMeans model on `data/Spotify_user_research.csv`:

1. **One vocabulary.** Every form option is defined once in `app/core/vocabulary.py`. Each
   option that corresponds to a survey answer carries the cleaned survey category it maps to.
   Every raw survey answer is mapped explicitly in `app/ml/training.py`, including messy
   variants such as `classical`/`Classical`, `Shorter`, and `Classical & melody, dance`.
   An answer that isn't in the map fails training instead of being dropped silently.
2. **Same features for training and serving.** `app/ml/features.py` encodes both survey rows
   and requests, so the two can't drift. Each field is scaled to equal total variance, so no
   single always-answered question dominates the clusters.
3. **k by silhouette, searched over 3–8.** Two segments would only split on age. Personas are
   named by their most *over-represented* answers (lift vs. the whole survey), not the most
   common ones, which are the same in every segment.
4. **Plain JSON artifact.** `models/segments.json` holds the centroids, feature names, scale,
   personas and metrics. The API loads it with numpy alone (no pickle, no scikit-learn) and
   refuses to start if its feature names don't match the code. Training is deterministic.

Honest caveat: the survey's answers are only weakly correlated (median Cramér's V ≈ 0.13).
So personas are descriptive context for the user and the LLM, not a strong recommendation
signal by themselves.

## License

MIT License

## Author

Developed by [Hemanth Rayudu](https://www.linkedin.com/in/hemanthrayudu/)