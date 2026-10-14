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
- React.js for component-based UI
- CSS with modern features (glassmorphism, animations, responsive design)
- LocalStorage for client-side podcast saving

### Backend
- FastAPI (Python) for efficient API endpoints
- KMeans listener personas trained on a 28k-response Spotify user survey (numpy-only at serving time)
- Open-weight LLM (gpt-oss via Groq) integration for recommendation intelligence
- pandas + scikit-learn for offline training only (`requirements-train.txt`)

## Installation

### Prerequisites
- Node.js (v16+)
- Python (v3.9+)
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
npm start
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