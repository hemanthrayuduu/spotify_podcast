from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import json
import pickle
import numpy as np
import os
from typing import Dict, List, Optional, Any
import logging
from dotenv import load_dotenv
import anthropic
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import status
from fastapi import Request
from pydantic import ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Spotify Podcast Recommender API", 
    description="API for recommending podcasts based on user preferences",
    # Add detailed validation error responses
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

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

# Initialize default values
kmeans_model = None
scaler = None
label_encoders = None
valid_features = None
segment_profiles = None

try:
    # Ensure models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load the KMeans model
    kmeans_model_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
    if os.path.exists(kmeans_model_path):
        kmeans_model = pickle.load(open(kmeans_model_path, "rb"))
    
    # Load the scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = pickle.load(open(scaler_path, "rb"))
    
    # Load label encoders
    encoders_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    if os.path.exists(encoders_path):
        label_encoders = pickle.load(open(encoders_path, "rb"))
    
    # Load valid features
    features_path = os.path.join(MODEL_DIR, "valid_features.pkl")
    if os.path.exists(features_path):
        valid_features = pickle.load(open(features_path, "rb"))
    
    # Load segment profiles
    segment_path = os.path.join(MODEL_DIR, "segment_profiles.json")
    if os.path.exists(segment_path):
        with open(segment_path, "r") as f:
            segment_profiles = json.load(f)
        
    logger.info("Models and data loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    # Don't raise an exception, continue with defaults

# Check if all required models are available
MODELS_AVAILABLE = all([kmeans_model, scaler, valid_features, segment_profiles])
if not MODELS_AVAILABLE:
    logger.warning("Some or all models are not available. Using fallback recommendations.")
    
    # Create simple fallback segment profiles if needed
    if not segment_profiles:
        segment_profiles = {
            "Casual Listener": {
                "segment_description": "You enjoy listening to podcasts occasionally for entertainment and to stay informed.",
                "top_genres": ["Comedy", "News & Politics", "Entertainment"]
            },
            "Knowledge Seeker": {
                "segment_description": "You use podcasts primarily as a source of learning and intellectual growth.",
                "top_genres": ["Educational", "Science & Technology", "History"]
            },
            "Daily Consumer": {
                "segment_description": "Podcasts are a daily part of your routine, keeping you company during commutes or activities.",
                "top_genres": ["News & Politics", "Daily Updates", "Comedy"]
            }
        }

# Initialize Anthropic client
try:
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY not found in environment variables")
        client = None
    else:
        logger.info("Attempting to initialize Anthropic client...")
        try:
            # Check if we're running on Render
            is_render = os.getenv('RENDER') == 'true'
            
            if is_render:
                # Render-specific initialization
                import httpx
                # Create a custom httpx client without proxy settings
                http_client = httpx.Client(
                    timeout=60.0,
                    proxies=None,
                    verify=True
                )
                client = anthropic.Anthropic(
                    api_key=anthropic_api_key,
                    http_client=http_client
                )
            else:
                # Local initialization
                client = anthropic.Anthropic(api_key=anthropic_api_key)
            
            # Test the client with a simple message
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            logger.info("Anthropic client test successful")
            logger.info("Anthropic client initialized successfully.")
        except anthropic.APIError as api_err:
            logger.error(f"Anthropic API Error: {str(api_err)}")
            client = None
        except anthropic.APIConnectionError as conn_err:
            logger.error(f"Anthropic Connection Error: {str(conn_err)}")
            client = None
        except anthropic.AuthenticationError as auth_err:
            logger.error(f"Anthropic Authentication Error: {str(auth_err)}")
            client = None
        except Exception as e:
            logger.error(f"Unexpected error initializing Anthropic client: {str(e)}")
            client = None
except Exception as e:
    logger.error(f"Error in Anthropic client setup: {str(e)}")
    client = None

# If client initialization failed, log a warning with more details
if client is None:
    logger.warning("Anthropic client not available - recommendations will use fallback mode")

# Input validation model
class UserPreferences(BaseModel):
    age: str
    music_genre: List[str]
    podcast_frequency: str
    podcast_duration: str
    podcast_format: str
    podcast_content: List[str]
    content_language: str
    region: str
    
    # New fields
    podcasts_enjoyed: str = ""
    listening_mood: str = ""
    
    # Optional fields
    gender: Optional[str] = None
    education: Optional[str] = None
    employment: Optional[str] = None
    
    # Custom validator with helpful error messages - Updated to V2 style
    @field_validator('music_genre', 'podcast_content', mode='before')
    @classmethod
    def validate_list_fields(cls, v, info):
        # If the value is None or empty, return an empty list (don't fail validation)
        if v is None:
            return []
            
        # If it's a string, try to parse it as JSON
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                # If it's a comma-separated string, split it
                if ',' in v:
                    return [item.strip() for item in v.split(',') if item.strip()]
                # If it's a single value, make it a list
                return [v]
        
        # If it's already a list, return it
        if isinstance(v, list):
            return v
            
        # If we got here, we don't know how to handle it
        raise ValueError(f"Expected a list for {info.field_name}, got {type(v).__name__}: {v}")
    
    # Post-validation to ensure lists aren't empty when submitting - Updated to V2 style
    @field_validator('music_genre', 'podcast_content')
    @classmethod
    def lists_must_not_be_empty(cls, v):
        if not v or len(v) == 0:
            raise ValueError('must contain at least one item')
        return v

def prepare_features(preferences: Dict[str, Any]) -> np.ndarray:
    """
    Prepare user input for prediction.
    """
    # If models are not available, return a dummy feature vector
    if not MODELS_AVAILABLE or not valid_features or not scaler:
        # Return a simple dummy vector that will be compatible with fallback logic
        return np.array([[0]])
        
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
            # Handle multiple music genres
            for value in preferences["music_genre"]:
                if value:  # Ensure the value is not empty
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
            # Handle multiple podcast content genres
            for value in preferences["podcast_content"]:
                if value:  # Ensure the value is not empty
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
    # Format multi-select fields for prompt
    music_genres = ", ".join(user_prefs.get("music_genre", ["Various"]) or ["Various"])
    pod_content_topics = ", ".join(user_prefs.get("podcast_content", ["Various"]) or ["Various"])
    
    pod_freq = user_prefs.get("podcast_frequency", "Weekly")
    pod_format = user_prefs.get("podcast_format", "Interview")
    pod_duration = user_prefs.get("podcast_duration", "Medium (30-60 min)")
    language = user_prefs.get("content_language", "English")
    region = user_prefs.get("region", "Global")
    age = user_prefs.get("age", "25-34")
    mood = user_prefs.get("listening_mood", "")
    podcasts_enjoyed = user_prefs.get("podcasts_enjoyed", "")
    
    # Get segment-specific information
    # Extract the top preferences for each segment feature
    top_music_genre = list(user_segment.get("fav_music_genre", {}).keys())[0] if user_segment.get("fav_music_genre") else "Various"
    top_pod_genre = list(user_segment.get("fav_pod_genre", {}).keys())[0] if user_segment.get("fav_pod_genre") else "Various"

    prompt = f"""You are a podcast recommendation expert. Based on the user profile and preferences, suggest 5 podcasts that they would enjoy.

USER INFORMATION:
- Age group: {age}
- Favorite music genres: {music_genres}
- Podcast listening frequency: {pod_freq}
- Preferred podcast duration: {pod_duration}
- Preferred podcast format: {pod_format}
- Podcast content interests: {pod_content_topics}
- Preferred language: {language}
- Region of interest: {region}
- Current listening mood: {mood}
"""

    # Add podcasts enjoyed if provided
    if podcasts_enjoyed:
        prompt += f"- Podcasts already enjoyed: {podcasts_enjoyed}\n"

    prompt += f"""
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
                        "reason": f"Popular podcast that matches your interest in {pod_content_topics}.",
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
                    "reason": f"Informative content about {pod_content_topics} topics in your preferred {pod_duration} format.",
                    "link": "https://www.google.com/search?q=Stuff+You+Should+Know+podcast"
                }
            ]
        }

@app.get("/")
async def root():
    return {"message": "Welcome to the Spotify Podcast Recommender API"}

@app.post("/recommend")
async def recommend_podcasts(request: Request):
    try:
        # Get the raw request body
        body = await request.json()
        logger.info(f"Received request body: {body}")
        
        # Try manual validation first
        try:
            # Ensure all required fields are present
            required_fields = [
                'age', 'music_genre', 'podcast_frequency', 'podcast_duration',
                'podcast_format', 'podcast_content', 'content_language', 'region',
                'listening_mood'
            ]
            
            for field in required_fields:
                if field not in body:
                    raise ValueError(f"Missing required field: {field}")
                
                if field in ['music_genre', 'podcast_content']:
                    if not isinstance(body[field], list) or len(body[field]) == 0:
                        raise ValueError(f"Field '{field}' must be a non-empty list")
                elif not body[field]:
                    raise ValueError(f"Field '{field}' cannot be empty")
            
            # Manually construct the data with defaults
            preferences = {
                'age': body['age'],
                'music_genre': body['music_genre'],
                'podcast_frequency': body['podcast_frequency'],
                'podcast_duration': body['podcast_duration'],
                'podcast_format': body['podcast_format'],
                'podcast_content': body['podcast_content'],
                'content_language': body['content_language'],
                'region': body['region'],
                'listening_mood': body['listening_mood'],
                'podcasts_enjoyed': body.get('podcasts_enjoyed', ''),
                'gender': body.get('gender', None),
                'education': body.get('education', None),
                'employment': body.get('employment', None)
            }
            
            # Check if ML models are available
            if MODELS_AVAILABLE:
                # Prepare the features
                user_features = prepare_features(preferences)
                
                # Predict the segment
                segment_id = kmeans_model.predict(user_features)[0]
                segment_name = f"Segment_{segment_id}"
                
                # Get the segment profile
                user_segment = segment_profiles.get(segment_name, {})
            else:
                # Fallback when models aren't available
                # Assign user to a basic segment based on their preferences
                preferred_genres = preferences['podcast_content']
                
                # Simple logic to assign a segment
                if any(genre in ["Educational", "Science & Technology", "History"] for genre in preferred_genres):
                    user_segment = segment_profiles.get("Knowledge Seeker", {})
                elif any(genre in ["News & Politics", "Business & Finance"] for genre in preferred_genres):
                    user_segment = segment_profiles.get("Daily Consumer", {})
                else:
                    user_segment = segment_profiles.get("Casual Listener", {})
            
            # Generate recommendations
            recommendations = generate_podcast_recommendations(user_segment, preferences)
            
            return recommendations
            
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"detail": f"Validation error: {str(ve)}"}
            )
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

# Add an exception handler for validation errors to provide clearer error messages
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {str(exc)}")
    
    # Extract more useful error messages
    error_details = []
    for error in exc.errors():
        loc = " -> ".join(str(x) for x in error["loc"])
        error_details.append(f"{loc}: {error['msg']}")
    
    error_message = "; ".join(error_details) if error_details else str(exc)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({
            "detail": f"Validation error: Please ensure all required fields are filled correctly. Details: {error_message}",
            "validation_errors": exc.errors(),
            "body": exc.body
        }),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)