from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import pickle
import numpy as np
import os
from typing import Dict, List, Optional, Any
import logging
from dotenv import load_dotenv
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Spotify Podcast Recommender API", 
              description="API for recommending podcasts based on user preferences")

# Setup CORS to allow frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    # Load the KMeans model
    kmeans_model = pickle.load(open(os.path.join(MODEL_DIR, "kmeans_model.pkl"), "rb"))
    
    # Load the scaler
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
    
    # Load label encoders
    label_encoders = pickle.load(open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb"))
    
    # Load valid features
    valid_features = pickle.load(open(os.path.join(MODEL_DIR, "valid_features.pkl"), "rb"))
    
    # Load segment profiles
    with open(os.path.join(MODEL_DIR, "segment_profiles.json"), "r") as f:
        segment_profiles = json.load(f)
        
    logger.info("All models and data loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

# Initialize Anthropic client
try:
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY not found in environment variables")
    
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    logger.info("Anthropic client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Anthropic client: {str(e)}")
    client = None

# Input validation model
class UserPreferences(BaseModel):
    age: str
    music_genre: str
    podcast_frequency: str
    podcast_duration: str
    podcast_format: str
    podcast_content: str
    content_language: str
    region: str
    
    # Optional fields
    gender: Optional[str] = None
    education: Optional[str] = None
    employment: Optional[str] = None

def prepare_features(preferences: Dict[str, Any]) -> np.ndarray:
    """
    Prepare user input for prediction.
    """
    # Map user inputs to feature vector
    features = {}
    
    # Add age as numeric (mid-point of range)
    age_map = {
        "18-24": 21,
        "25-34": 30,
        "35-44": 40,
        "45-54": 50,
        "55+": 60
    }
    features["age_numeric"] = age_map.get(preferences["age"], 30)
    
    # Map categorical variables using one-hot encoding
    for feature_name in valid_features:
        if feature_name == "age_numeric":
            continue
            
        if feature_name.startswith("fav_music_genre"):
            base_name = "fav_music_genre"
            value = preferences["music_genre"]
            feature_value = f"{base_name}_{value}"
            if feature_value in valid_features:
                features[feature_value] = 1
                
        elif feature_name.startswith("pod_lis_frequency"):
            base_name = "pod_lis_frequency"
            value = preferences["podcast_frequency"]
            feature_value = f"{base_name}_{value}"
            if feature_value in valid_features:
                features[feature_value] = 1
                
        elif feature_name.startswith("preffered_pod_duration"):
            base_name = "preffered_pod_duration"
            value = preferences["podcast_duration"]
            feature_value = f"{base_name}_{value}"
            if feature_value in valid_features:
                features[feature_value] = 1
                
        elif feature_name.startswith("preffered_pod_format"):
            base_name = "preffered_pod_format"
            value = preferences["podcast_format"]
            feature_value = f"{base_name}_{value}"
            if feature_value in valid_features:
                features[feature_value] = 1
                
        elif feature_name.startswith("fav_pod_genre"):
            base_name = "fav_pod_genre"
            value = preferences["podcast_content"]
            feature_value = f"{base_name}_{value}"
            if feature_value in valid_features:
                features[feature_value] = 1
    
    # Create a feature vector with all valid features
    feature_vector = np.zeros(len(valid_features))
    
    for i, feature_name in enumerate(valid_features):
        feature_vector[i] = features.get(feature_name, 0)
    
    # Scale the features
    scaled_features = scaler.transform([feature_vector])
    
    return scaled_features

def generate_podcast_recommendations(user_segment: Dict[str, Any], user_prefs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate podcast recommendations using Claude AI.
    """
    if not client:
        # Fallback recommendations if Claude client is not available
        return {
            "segment_profile": user_segment,
            "recommendations": [
                {
                    "name": "The Daily",
                    "creator": "The New York Times",
                    "description": "This is what the news should sound like. The biggest stories of our time, told by the best journalists in the world.",
                    "format": user_prefs["podcast_format"],
                    "duration": user_prefs["podcast_duration"],
                    "language": user_prefs["content_language"],
                    "region": user_prefs["region"],
                    "reason": "Popular podcast that matches your interest in current events and news.",
                    "link": "https://www.google.com/search?q=The+Daily+podcast"
                },
                {
                    "name": "TED Talks Daily",
                    "creator": "TED",
                    "description": "Every weekday, TED Talks Daily brings you the latest talks in audio. Join host and journalist Elise Hu for thought-provoking ideas on every subject imaginable.",
                    "format": "Educational",
                    "duration": "Short (< 30 min)",
                    "language": user_prefs["content_language"],
                    "region": "Global",
                    "reason": "Educational content that aligns with your listening preferences.",
                    "link": "https://www.google.com/search?q=TED+Talks+Daily+podcast"
                },
                {
                    "name": "Freakonomics Radio",
                    "creator": "Stephen J. Dubner",
                    "description": "Discover the hidden side of everything with Stephen J. Dubner, co-author of the Freakonomics books.",
                    "format": "Interview",
                    "duration": "Medium (30-60 min)",
                    "language": user_prefs["content_language"],
                    "region": "Global",
                    "reason": "Interesting economic and social topics presented in an engaging way.",
                    "link": "https://www.google.com/search?q=Freakonomics+Radio+podcast"
                }
            ]
        }
    
    # Prepare prompt for Claude
    music_genre = user_prefs.get("music_genre", "Various")
    pod_freq = user_prefs.get("podcast_frequency", "Weekly")
    pod_format = user_prefs.get("podcast_format", "Interview")
    pod_duration = user_prefs.get("podcast_duration", "Medium (30-60 min)")
    pod_content = user_prefs.get("podcast_content", "Various")
    language = user_prefs.get("content_language", "English")
    region = user_prefs.get("region", "Global")
    age = user_prefs.get("age", "25-34")
    
    # Get segment-specific information
    # Extract the top preferences for each segment feature
    top_music_genre = list(user_segment.get("fav_music_genre", {}).keys())[0] if user_segment.get("fav_music_genre") else "Various"
    top_pod_genre = list(user_segment.get("fav_pod_genre", {}).keys())[0] if user_segment.get("fav_pod_genre") else "Various"

    prompt = f"""You are a podcast recommendation expert. Based on the user profile and preferences, suggest 5 podcasts that they would enjoy.

USER INFORMATION:
- Age group: {age}
- Favorite music genre: {music_genre}
- Podcast listening frequency: {pod_freq}
- Preferred podcast duration: {pod_duration}
- Preferred podcast format: {pod_format}
- Podcast content interests: {pod_content}
- Preferred language: {language}
- Region of interest: {region}

LISTENER SEGMENT PROFILE:
- Top music genre in segment: {top_music_genre}
- Top podcast genre in segment: {top_pod_genre}
- Age demographics: {user_segment.get("age_numeric", {}).get("mean", 30)} (average)

Please provide 5 podcast recommendations in JSON format:
[
  {{
    "name": "Podcast Name",
    "creator": "Creator/Host Name",
    "description": "Brief and engaging description (1-2 sentences)",
    "format": "Format type (Interview, Narrative, Educational, etc.)",
    "duration": "Typical episode length",
    "language": "Main language",
    "region": "Content region focus",
    "reason": "Personalized reason why this matches the user (1 sentence)",
    "link": "Leave this empty, I'll fill it in later"
  }},
  ...
]

Focus on real, high-quality podcasts that genuinely match the user's interests. Be specific with your recommendations, not generic.
"""

    try:
        # Generate recommendations using Claude
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract JSON from response
        content = response.content[0].text
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[\s*{\s*"name".*}\s*\]', content, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group(0))
            else:
                # Fallback to assuming the entire response is valid JSON
                recommendations = json.loads(content)
                
            # Add Google search links if not present
            for rec in recommendations:
                if not rec.get("link") or rec.get("link") == "":
                    podcast_name = rec.get("name", "").replace(" ", "+")
                    rec["link"] = f"https://www.google.com/search?q={podcast_name}+podcast"
                    
            return {
                "segment_profile": user_segment,
                "recommendations": recommendations
            }
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from Claude response: {content}")
            # Return a simplified response
            return {
                "segment_profile": user_segment,
                "recommendations": [
                    {
                        "name": "The Daily",
                        "creator": "The New York Times",
                        "description": "This is what the news should sound like. The biggest stories of our time, told by the best journalists in the world.",
                        "format": pod_format,
                        "duration": pod_duration,
                        "language": language,
                        "region": region,
                        "reason": f"Popular podcast that matches your interest in {pod_content}.",
                        "link": "https://www.google.com/search?q=The+Daily+podcast"
                    },
                    {
                        "name": "TED Talks Daily",
                        "creator": "TED",
                        "description": "Every weekday, TED Talks Daily brings you the latest talks in audio. Join host and journalist Elise Hu for thought-provoking ideas on every subject imaginable.",
                        "format": "Educational",
                        "duration": "Short (< 30 min)",
                        "language": language,
                        "region": "Global",
                        "reason": "Educational content that aligns with your listening preferences.",
                        "link": "https://www.google.com/search?q=TED+Talks+Daily+podcast"
                    }
                ]
            }
            
    except Exception as e:
        logger.error(f"Error generating recommendations with Claude: {str(e)}")
        # Return fallback recommendations
        return {
            "segment_profile": user_segment,
            "recommendations": [
                {
                    "name": "SmartLess",
                    "creator": "Jason Bateman, Sean Hayes, Will Arnett",
                    "description": "A podcast that connects and unites people from all walks of life to learn about shared experiences through thoughtful dialogue and organic hilarity.",
                    "format": "Interview",
                    "duration": "Medium (30-60 min)",
                    "language": language,
                    "region": "Global",
                    "reason": f"Popular podcast with a {pod_format} format that many {age} listeners enjoy.",
                    "link": "https://www.google.com/search?q=SmartLess+podcast"
                },
                {
                    "name": "Stuff You Should Know",
                    "creator": "iHeartRadio",
                    "description": "Josh and Chuck explore everything from champagne to satanism, exploring the ins and outs of a variety of topics.",
                    "format": "Educational",
                    "duration": "Medium (30-60 min)",
                    "language": language,
                    "region": "Global",
                    "reason": f"Informative content about {pod_content} topics in your preferred {pod_duration} format.",
                    "link": "https://www.google.com/search?q=Stuff+You+Should+Know+podcast"
                }
            ]
        }

@app.get("/")
async def root():
    return {"message": "Welcome to the Spotify Podcast Recommender API"}

@app.post("/recommend")
async def recommend_podcasts(preferences: UserPreferences):
    try:
        # Prepare the features
        user_features = prepare_features(preferences.dict())
        
        # Predict the segment
        segment_id = kmeans_model.predict(user_features)[0]
        segment_name = f"Segment_{segment_id}"
        
        # Get the segment profile
        user_segment = segment_profiles.get(segment_name, {})
        
        # Generate recommendations
        recommendations = generate_podcast_recommendations(user_segment, preferences.dict())
        
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)