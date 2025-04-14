# Spotify Podcast Recommender

A personalized podcast recommendation system powered by machine learning and AI.

## Overview

Spotify Podcast Recommender analyzes user preferences and listening habits to provide tailored podcast recommendations. The application uses segmentation algorithms and Claude AI to generate highly personalized content suggestions.

## Features

- **Personalized Recommendations**: Uses machine learning to match podcasts to user preferences
- **User Segmentation**: Identifies listening profiles to better understand user interests
- **AI-Enhanced Suggestions**: Leverages Claude AI to generate contextually relevant recommendations
- **Library Management**: Save favorite podcasts for future reference
- **Modern UI**: Responsive design with Spotify-inspired aesthetics

## Technology Stack

### Frontend
- React.js for component-based UI
- CSS with modern features (glassmorphism, animations, responsive design)
- LocalStorage for client-side podcast saving

### Backend
- FastAPI (Python) for efficient API endpoints
- KMeans clustering for user segmentation
- Claude AI integration for recommendation intelligence
- Scikit-learn for preprocessing and model training

## Installation

### Prerequisites
- Node.js (v16+)
- Python (v3.9+)
- Anthropic API key for Claude AI

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# Create .env file with ANTHROPIC_API_KEY=your_api_key
uvicorn app:app --reload
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

- `POST /recommend`: Submit user preferences and receive recommendations
- `GET /`: API health check and information

## Deployment

The application is configured for deployment on:
- Frontend: Netlify
- Backend: Render

## License

MIT License

## Author

Developed by [Hemanth Rayudu](https://www.linkedin.com/in/hemanthrayudu/)