import numpy as np
import pandas as pd
import pickle
import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpotifyUserAnalyzer:
    """
    Class to analyze Spotify user data and create user segments.
    """
    def __init__(self, data_path: str = None, model_dir: str = 'models'):
        """
        Initialize the analyzer with data file path and model directory.
        
        Args:
            data_path: Path to the CSV file with user data
            model_dir: Directory to save/load models and data
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.data = None
        self.features = None
        self.kmeans_model = None
        self.scaler = None
        self.label_encoders = {}
        self.valid_features = None
        self.segment_profiles = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """
        Load user data from CSV file.
        
        Returns:
            DataFrame with user data
        """
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully with {self.data.shape[0]} rows and {self.data.shape[1]} columns")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the user data for modeling.
        
        Returns:
            DataFrame with processed features
        """
        if self.data is None:
            logger.error("No data loaded. Please load data first.")
            return None
            
        try:
            # Convert age to numeric
            self.data['age_numeric'] = self.data['Age'].apply(self._convert_age_to_numeric)
            
            # Get categorical columns
            categorical_columns = self.data.select_dtypes(include=['object']).columns.tolist()
            categorical_columns.remove('Age')  # Already processed
            
            # One-hot encode categorical variables
            encoded_data = pd.get_dummies(self.data[categorical_columns])
            
            # Combine numeric and encoded features
            self.features = pd.concat([
                self.data[['age_numeric']], 
                encoded_data
            ], axis=1)
            
            # Store valid feature names
            self.valid_features = self.features.columns.tolist()
            
            logger.info(f"Data preprocessed successfully with {self.features.shape[1]} features")
            return self.features
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
            
    def _convert_age_to_numeric(self, age_category: str) -> int:
        """
        Convert age category to numeric value.
        
        Args:
            age_category: Age category string
            
        Returns:
            Numeric age value (middle of the range)
        """
        age_map = {
            '12~20': 16,
            '20~35': 28,
            '35~60': 48,
            '60+': 65
        }
        return age_map.get(age_category, 30)  # Default to 30 if unknown
        
    def train_cluster_model(self, n_clusters: int = 3) -> KMeans:
        """
        Train KMeans clustering model on the preprocessed data.
        
        Args:
            n_clusters: Number of clusters (segments)
            
        Returns:
            Trained KMeans model
        """
        if self.features is None:
            logger.error("No features available. Please preprocess data first.")
            return None
            
        try:
            # Scale the features
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(self.features)
            
            # Train KMeans model
            self.kmeans_model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            self.kmeans_model.fit(scaled_features)
            
            logger.info(f"KMeans model trained successfully with {n_clusters} clusters")
            return self.kmeans_model
        except Exception as e:
            logger.error(f"Error training cluster model: {str(e)}")
            raise
            
    def create_segment_profiles(self) -> Dict[str, Any]:
        """
        Create detailed profiles for each user segment.
        
        Returns:
            Dictionary with segment profiles
        """
        if self.kmeans_model is None or self.data is None:
            logger.error("KMeans model not trained or data not loaded.")
            return None
            
        try:
            # Assign segments to data
            self.data['segment'] = self.kmeans_model.predict(self.scaler.transform(self.features))
            
            # Create segment profiles
            self.segment_profiles = {}
            
            # Categorical features to profile
            cat_features = [
                'Age', 'Gender', 'spotify_usage_period', 'spotify_listening_device',
                'spotify_subscription_plan', 'premium_sub_willingness', 
                'preffered_premium_plan', 'preferred_listening_content',
                'fav_music_genre', 'music_time_slot', 'music_Influencial_mood',
                'music_lis_frequency', 'music_expl_method', 'pod_lis_frequency',
                'fav_pod_genre', 'preffered_pod_format', 'pod_host_preference',
                'preffered_pod_duration', 'pod_variety_satisfaction'
            ]
            
            # Numeric features to profile
            num_features = ['age_numeric']
            
            # Create profile for each segment
            for segment_id in sorted(self.data['segment'].unique()):
                segment_data = self.data[self.data['segment'] == segment_id]
                segment_name = f"Segment_{segment_id}"
                
                self.segment_profiles[segment_name] = {}
                
                # Profile categorical features
                for feature in cat_features:
                    if feature in segment_data.columns:
                        value_counts = segment_data[feature].value_counts(normalize=True)
                        top_values = value_counts.head(3).to_dict()
                        self.segment_profiles[segment_name][feature] = top_values
                
                # Profile numeric features
                for feature in num_features:
                    if feature in segment_data.columns:
                        self.segment_profiles[segment_name][feature] = {
                            'mean': segment_data[feature].mean(),
                            'median': segment_data[feature].median()
                        }
            
            logger.info(f"Created segment profiles for {len(self.segment_profiles)} segments")
            return self.segment_profiles
        except Exception as e:
            logger.error(f"Error creating segment profiles: {str(e)}")
            raise
            
    def save_models(self) -> None:
        """
        Save all models and data to the model directory.
        """
        try:
            # Save KMeans model
            with open(os.path.join(self.model_dir, 'kmeans_model.pkl'), 'wb') as f:
                pickle.dump(self.kmeans_model, f)
                
            # Save scaler
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
                
            # Save label encoders
            with open(os.path.join(self.model_dir, 'label_encoders.pkl'), 'wb') as f:
                pickle.dump(self.label_encoders, f)
                
            # Save valid features
            with open(os.path.join(self.model_dir, 'valid_features.pkl'), 'wb') as f:
                pickle.dump(self.valid_features, f)
                
            # Save segment profiles
            with open(os.path.join(self.model_dir, 'segment_profiles.json'), 'w') as f:
                json.dump(self.segment_profiles, f, indent=4)
                
            logger.info(f"All models and data saved to {self.model_dir}")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
            
    def load_models(self) -> bool:
        """
        Load all models and data from the model directory.
        
        Returns:
            Boolean indicating if loading was successful
        """
        try:
            # Load KMeans model
            with open(os.path.join(self.model_dir, 'kmeans_model.pkl'), 'rb') as f:
                self.kmeans_model = pickle.load(f)
                
            # Load scaler
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
                
            # Load label encoders
            with open(os.path.join(self.model_dir, 'label_encoders.pkl'), 'rb') as f:
                self.label_encoders = pickle.load(f)
                
            # Load valid features
            with open(os.path.join(self.model_dir, 'valid_features.pkl'), 'rb') as f:
                self.valid_features = pickle.load(f)
                
            # Load segment profiles
            with open(os.path.join(self.model_dir, 'segment_profiles.json'), 'r') as f:
                self.segment_profiles = json.load(f)
                
            logger.info(f"All models and data loaded from {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
            
    def predict_segment(self, user_data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Predict segment for a new user.
        
        Args:
            user_data: Dictionary with user preferences
            
        Returns:
            Tuple of (segment_id, segment_profile)
        """
        if self.kmeans_model is None or self.scaler is None or self.valid_features is None:
            logger.error("Models not loaded. Please load models first.")
            return None, None
            
        try:
            # Prepare features
            features = np.zeros(len(self.valid_features))
            feature_dict = {}
            
            # Add age as numeric (mid-point of range)
            age_map = {
                "18-24": 21,
                "25-34": 30,
                "35-44": 40,
                "45-54": 50,
                "55+": 60
            }
            feature_dict["age_numeric"] = age_map.get(user_data.get("age", "25-34"), 30)
            
            # Map categorical variables using one-hot encoding
            for feature_name in self.valid_features:
                if feature_name == "age_numeric":
                    continue
                    
                if feature_name.startswith("fav_music_genre"):
                    base_name = "fav_music_genre"
                    value = user_data.get("music_genre", "")
                    feature_value = f"{base_name}_{value}"
                    if feature_value in self.valid_features:
                        feature_dict[feature_value] = 1
                        
                elif feature_name.startswith("pod_lis_frequency"):
                    base_name = "pod_lis_frequency"
                    value = user_data.get("podcast_frequency", "")
                    feature_value = f"{base_name}_{value}"
                    if feature_value in self.valid_features:
                        feature_dict[feature_value] = 1
                        
                elif feature_name.startswith("preffered_pod_duration"):
                    base_name = "preffered_pod_duration"
                    value = user_data.get("podcast_duration", "")
                    feature_value = f"{base_name}_{value}"
                    if feature_value in self.valid_features:
                        feature_dict[feature_value] = 1
                        
                elif feature_name.startswith("preffered_pod_format"):
                    base_name = "preffered_pod_format"
                    value = user_data.get("podcast_format", "")
                    feature_value = f"{base_name}_{value}"
                    if feature_value in self.valid_features:
                        feature_dict[feature_value] = 1
                        
                elif feature_name.startswith("fav_pod_genre"):
                    base_name = "fav_pod_genre"
                    value = user_data.get("podcast_content", "")
                    feature_value = f"{base_name}_{value}"
                    if feature_value in self.valid_features:
                        feature_dict[feature_value] = 1
            
            # Create feature vector
            for i, feature_name in enumerate(self.valid_features):
                features[i] = feature_dict.get(feature_name, 0)
            
            # Scale the features
            scaled_features = self.scaler.transform([features])
            
            # Predict segment
            segment_id = self.kmeans_model.predict(scaled_features)[0]
            segment_name = f"Segment_{segment_id}"
            
            # Get segment profile
            segment_profile = self.segment_profiles.get(segment_name, {})
            
            logger.info(f"Predicted segment {segment_id} for user")
            return segment_id, segment_profile
        except Exception as e:
            logger.error(f"Error predicting segment: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    data_path = "../data/Spotify_user_research.csv"
    analyzer = SpotifyUserAnalyzer(data_path=data_path)
    
    # Check if models already exist
    if os.path.exists(os.path.join(analyzer.model_dir, 'kmeans_model.pkl')):
        print("Loading existing models...")
        analyzer.load_models()
    else:
        print("Training new models...")
        analyzer.load_data()
        analyzer.preprocess_data()
        analyzer.train_cluster_model(n_clusters=3)
        analyzer.create_segment_profiles()
        analyzer.save_models()
    
    # Example prediction
    user_data = {
        "age": "25-34",
        "music_genre": "Pop",
        "podcast_frequency": "Several times a week",
        "podcast_duration": "Medium (30-60 min)",
        "podcast_format": "Interview",
        "podcast_content": "Technology"
    }
    
    segment_id, segment_profile = analyzer.predict_segment(user_data)
    print(f"Predicted segment: {segment_id}")
    print(f"Segment profile: {json.dumps(segment_profile, indent=2)}")