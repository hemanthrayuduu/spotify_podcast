import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './RecommendationDisplay.css';

const RecommendationDisplay = ({ recommendations, onReset }) => {
  const [savedPodcasts, setSavedPodcasts] = useState(() => {
    try {
      const saved = localStorage.getItem('savedPodcasts');
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error('Error loading saved podcasts:', error);
      return [];
    }
  });
  
  const [savedStatus, setSavedStatus] = useState({});

  // Function to open Google search for podcast on Spotify including creator name
  const openSpotifySearch = (podcastName, creatorName) => {
    // Include both podcast name and creator in the search query for more accurate results
    const searchQuery = encodeURIComponent(`${podcastName} ${creatorName || ''} podcast spotify`);
    window.open(`https://www.google.com/search?q=${searchQuery}`, '_blank');
  };
  
  // Function to save a podcast to the library
  const savePodcast = (podcast) => {
    // Check if the podcast is already saved by matching name
    const podcastName = podcast.name || podcast.title;
    const isAlreadySaved = savedPodcasts.some(
      saved => (saved.name || saved.title) === podcastName
    );
    
    if (!isAlreadySaved) {
      const updatedSaved = [...savedPodcasts, podcast];
      localStorage.setItem('savedPodcasts', JSON.stringify(updatedSaved));
      setSavedPodcasts(updatedSaved);
      
      // Update UI to show saved status
      setSavedStatus({
        ...savedStatus,
        [podcastName]: true
      });
      
      // Show a temporary success message
      alert(`"${podcastName}" has been saved to your library!`);
    } else {
      alert(`"${podcastName}" is already in your library!`);
    }
  };

  // Extract segment profile and podcast recommendations from the API response
  const segmentProfile = recommendations?.segment_profile || {};
  const podcasts = recommendations?.recommendations || [];

  // Get segment name
  const segmentName = Object.keys(segmentProfile).length > 0 
    ? segmentProfile.segment_name || "Podcast Enthusiast" 
    : "Podcast Enthusiast";

  // Get top genres from segment profile
  const topMusicGenre = segmentProfile.fav_music_genre 
    ? Object.keys(segmentProfile.fav_music_genre)[0] 
    : "Varied";
  
  const topPodGenre = segmentProfile.fav_pod_genre 
    ? Object.keys(segmentProfile.fav_pod_genre)[0] 
    : "Varied";

  return (
    <div className="recommendation-container">
      {/* Segment info section */}
      <div className="segment-info">
        <h2>Your Listening Profile</h2>
        <div className="listener-personality">
          <div className="personality-icon">
            <i className="fas fa-headphones-alt"></i>
          </div>
          <div className="personality-details">
            <h3>{segmentName}</h3>
            <p>Based on your preferences, we've identified you as a listener who appreciates {topPodGenre.toLowerCase()} content and has an affinity for {topMusicGenre.toLowerCase()} music.</p>
            <div className="personality-traits">
              <span className="trait-tag">
                <i className="fas fa-music"></i>
                {topMusicGenre}
              </span>
              <span className="trait-tag">
                <i className="fas fa-podcast"></i>
                {topPodGenre}
              </span>
              <span className="trait-tag">
                <i className="fas fa-clock"></i>
                {recommendations?.podcast_duration || "Medium Length"}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations section */}
      <div className="recommendations">
        <h2>Your Podcast Recommendations</h2>
        <div className="recommendation-list">
          {podcasts.length > 0 ? (
            podcasts.map((podcast, index) => (
              <div className="recommendation-card" key={index}>
                <div className="podcast-info">
                  <div className="podcast-header">
                    <h3>{podcast.name || podcast.title}</h3>
                    {podcast.match_score && (
                      <span className="podcast-badge">{podcast.match_score}% Match</span>
                    )}
                  </div>
                  
                  <div className="podcast-creator">
                    <i className="fas fa-user-circle"></i>
                    {podcast.creator || podcast.author || "Unknown Creator"}
                  </div>
                  
                  <p className="podcast-description">{podcast.description}</p>
                  
                  {(podcast.reason || podcast.match_reason) && (
                    <div className="match-reason">
                      <i className="fas fa-check-circle"></i>
                      <span>{podcast.reason || podcast.match_reason}</span>
                    </div>
                  )}
                  
                  <div className="podcast-meta">
                    {(podcast.format || podcast.genre) && (
                      <div className="podcast-genre">
                        <i className="fas fa-tag"></i>
                        <span>{podcast.format || podcast.genre}</span>
                      </div>
                    )}
                    {podcast.language && (
                      <div className="podcast-language">
                        <i className="fas fa-language"></i>
                        <span>{podcast.language}</span>
                      </div>
                    )}
                    {podcast.duration && (
                      <div className="podcast-duration">
                        <i className="fas fa-clock"></i>
                        <span>{podcast.duration}</span>
                      </div>
                    )}
                    {podcast.region && (
                      <div className="podcast-region">
                        <i className="fas fa-globe-americas"></i>
                        <span>{podcast.region}</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="podcast-actions">
                    <div className="podcast-link">
                      <button 
                        className="spotify-link"
                        onClick={() => openSpotifySearch(
                        podcast.name || podcast.title,
                        podcast.creator || podcast.author
                      )}
                      >
                        <i className="fab fa-spotify"></i>
                        Listen on Spotify
                      </button>
                    </div>
                    <button 
                      className={`save-button ${savedStatus[podcast.name || podcast.title] ? 'saved' : ''}`}
                      onClick={() => savePodcast(podcast)}
                    >
                      <i className={`${savedStatus[podcast.name || podcast.title] ? 'fas' : 'far'} fa-bookmark`}></i>
                      {savedStatus[podcast.name || podcast.title] ? 'Saved to Library' : 'Save to Library'}
                    </button>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="no-recommendations">
              <p>No podcast recommendations found. Please try again with different preferences.</p>
            </div>
          )}
        </div>
        
        <div className="button-group">
          <button className="reset-button" onClick={onReset}>
            <i className="fas fa-redo"></i>
            Start Over
          </button>
          <Link to="/library" className="library-button">
            <i className="fas fa-book"></i>
            View Library
          </Link>
        </div>
        
        <div className="footer-credit">
          Developed by <a href="https://www.linkedin.com/in/hemanthrayudu/" target="_blank" rel="noopener noreferrer">Hemanth Rayudu</a>
        </div>
      </div>
    </div>
  );
};

export default RecommendationDisplay;