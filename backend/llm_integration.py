import anthropic
import os
import json
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Anthropic client
try:
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    logger.info("Anthropic client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Anthropic client: {str(e)}")
    client = None

def generate_podcast_recommendations(
    user_preferences: Dict[str, Any], 
    segment_profile: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate podcast recommendations using Claude AI based on user preferences
    and their segment profile.
    
    Args:
        user_preferences: Dictionary containing user form responses
        segment_profile: Dictionary containing the user's segment profile data
        
    Returns:
        List of podcast recommendation objects
    """
    if not client:
        logger.warning("Anthropic client not available, returning fallback recommendations")
        return get_fallback_recommendations(user_preferences)
    
    # Extract key user preferences
    music_genre = user_preferences.get("music_genre", "Various")
    pod_freq = user_preferences.get("podcast_frequency", "Weekly")
    pod_format = user_preferences.get("podcast_format", "Interview")
    pod_duration = user_preferences.get("podcast_duration", "Medium (30-60 min)")
    pod_content = user_preferences.get("podcast_content", "Various")
    language = user_preferences.get("content_language", "English")
    region = user_preferences.get("region", "Global")
    age = user_preferences.get("age", "25-34")
    
    # Get segment-specific information
    # Extract the top preferences for each segment feature
    top_music_genre = list(segment_profile.get("fav_music_genre", {}).keys())[0] if segment_profile.get("fav_music_genre") else "Various"
    top_pod_genre = list(segment_profile.get("fav_pod_genre", {}).keys())[0] if segment_profile.get("fav_pod_genre") else "Various"
    segment_age = segment_profile.get("age_numeric", {}).get("mean", 30)

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
- Age demographics: {segment_age} (average)

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
                    podcast_creator = rec.get("creator", "").replace(" ", "+")
                    rec["link"] = f"https://www.google.com/search?q={podcast_name}+{podcast_creator}+podcast"
                    
            return recommendations
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from Claude response: {content}")
            return get_fallback_recommendations(user_preferences)
            
    except Exception as e:
        logger.error(f"Error generating recommendations with Claude: {str(e)}")
        return get_fallback_recommendations(user_preferences)

def get_fallback_recommendations(user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Provide fallback recommendations when Claude is unavailable.
    
    Args:
        user_preferences: Dictionary containing user form responses
        
    Returns:
        List of podcast recommendation objects
    """
    pod_format = user_preferences.get("podcast_format", "Interview")
    pod_duration = user_preferences.get("podcast_duration", "Medium (30-60 min)")
    pod_content = user_preferences.get("podcast_content", "Various")
    language = user_preferences.get("content_language", "English")
    region = user_preferences.get("region", "Global")
    age = user_preferences.get("age", "25-34")
    
    return [
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
        },
        {
            "name": "Freakonomics Radio",
            "creator": "Stephen J. Dubner",
            "description": "Discover the hidden side of everything with Stephen J. Dubner, co-author of the Freakonomics books.",
            "format": "Interview",
            "duration": "Medium (30-60 min)",
            "language": language,
            "region": "Global",
            "reason": "Interesting economic and social topics presented in an engaging way.",
            "link": "https://www.google.com/search?q=Freakonomics+Radio+podcast"
        },
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