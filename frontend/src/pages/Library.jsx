import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { keyOf, loadSaved, publisherOf, removePodcast, spotifySearchUrl, titleOf } from '../library';
import './Library.css';

const Library = () => {
  const [savedPodcasts, setSavedPodcasts] = useState(loadSaved);

  return (
    <div className="library-container">
      <div className="library-header">
        <h2>My Library</h2>
        <p>Podcasts you've saved for later.</p>
      </div>

      {savedPodcasts.length > 0 ? (
        <div className="library-list">
          {savedPodcasts.map((podcast) => {
            const key = keyOf(podcast);
            const title = titleOf(podcast);
            const minutes = podcast.typical_episode_minutes;
            return (
              <div className="library-card" key={key}>
                <div className="library-card-header">
                  {podcast.artwork_url && (
                    <img className="library-artwork" src={podcast.artwork_url} alt="" loading="lazy" />
                  )}
                  <h3>{title}</h3>
                  <button
                    className="remove-button"
                    onClick={() => setSavedPodcasts(removePodcast(key))}
                    aria-label={`Remove ${title} from library`}
                  >
                    <i className="fas fa-trash-alt"></i>
                  </button>
                </div>

                <div className="library-creator">
                  <i className="fas fa-user-circle"></i>
                  {publisherOf(podcast)}
                </div>

                {(podcast.summary || podcast.description) && (
                  <p className="library-description">{podcast.summary || podcast.description}</p>
                )}

                <div className="library-meta">
                  {podcast.genres?.length > 0 && (
                    <span><i className="fas fa-tag"></i> {podcast.genres.slice(0, 2).join(' · ')}</span>
                  )}
                  {minutes != null && <span><i className="fas fa-clock"></i> ~{minutes} min</span>}
                </div>

                <div className="library-links">
                  {podcast.apple_url && (
                    <a className="apple-link" href={podcast.apple_url} target="_blank" rel="noopener noreferrer">
                      <i className="fas fa-podcast"></i> Apple Podcasts
                    </a>
                  )}
                  <a
                    className="spotify-link"
                    href={podcast.spotify_url || spotifySearchUrl(title)}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <i className="fab fa-spotify"></i> Search on Spotify
                  </a>
                </div>
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
