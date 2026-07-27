import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './Library.css';

const loadSavedPodcasts = () => {
  try {
    const saved = localStorage.getItem('savedPodcasts');
    return saved ? JSON.parse(saved) : [];
  } catch (error) {
    console.error('Error loading saved podcasts:', error);
    return [];
  }
};

const Library = () => {
  const [savedPodcasts, setSavedPodcasts] = useState(loadSavedPodcasts);

  const openSpotifySearch = (podcastName, creatorName) => {
    const searchQuery = encodeURIComponent(`${podcastName} ${creatorName || ''} podcast spotify`);
    window.open(`https://www.google.com/search?q=${searchQuery}`, '_blank');
  };

  const removePodcast = (podcastName) => {
    const updated = savedPodcasts.filter(
      (saved) => (saved.name || saved.title) !== podcastName
    );
    localStorage.setItem('savedPodcasts', JSON.stringify(updated));
    setSavedPodcasts(updated);
  };

  return (
    <div className="library-container">
      <div className="library-header">
        <h2>My Library</h2>
        <p>Podcasts you've saved for later.</p>
      </div>

      {savedPodcasts.length > 0 ? (
        <div className="library-list">
          {savedPodcasts.map((podcast, index) => {
            const name = podcast.name || podcast.title;
            const creator = podcast.creator || podcast.author || 'Unknown Creator';
            return (
              <div className="library-card" key={`${name}-${index}`}>
                <div className="library-card-header">
                  <h3>{name}</h3>
                  <button
                    className="remove-button"
                    onClick={() => removePodcast(name)}
                    aria-label={`Remove ${name} from library`}
                  >
                    <i className="fas fa-trash-alt"></i>
                  </button>
                </div>

                <div className="library-creator">
                  <i className="fas fa-user-circle"></i>
                  {creator}
                </div>

                {podcast.description && (
                  <p className="library-description">{podcast.description}</p>
                )}

                <div className="library-meta">
                  {(podcast.format || podcast.genre) && (
                    <span><i className="fas fa-tag"></i> {podcast.format || podcast.genre}</span>
                  )}
                  {podcast.duration && (
                    <span><i className="fas fa-clock"></i> {podcast.duration}</span>
                  )}
                  {podcast.language && (
                    <span><i className="fas fa-language"></i> {podcast.language}</span>
                  )}
                </div>

                <button
                  className="spotify-link"
                  onClick={() => openSpotifySearch(name, podcast.creator || podcast.author)}
                >
                  <i className="fab fa-spotify"></i>
                  Listen on Spotify
                </button>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="library-empty">
          <i className="fas fa-book"></i>
          <p>Your library is empty. Save podcasts from your recommendations to see them here.</p>
          <Link to="/" className="library-cta">
            <i className="fas fa-home"></i>
            Get Recommendations
          </Link>
        </div>
      )}
    </div>
  );
};

export default Library;
