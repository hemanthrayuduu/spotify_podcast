import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { keyOf, loadSaved, savePodcast } from '../library';
import './RecommendationDisplay.css';

const BREAKDOWN_LABELS = {
  topic: 'Topic',
  duration: 'Length',
  endorsement: 'Curator pick',
  freshness: 'Fresh episodes',
};

const LANGUAGE_NAMES = new Intl.DisplayNames(['en'], { type: 'language' });

function lastEpisodeText(isoDate) {
  if (!isoDate) return null;
  const days = Math.round((Date.now() - new Date(`${isoDate}T00:00:00`)) / 86_400_000);
  if (days <= 0) return 'New episode today';
  if (days === 1) return 'New episode yesterday';
  if (days < 30) return `New episode ${days} days ago`;
  return `Last episode ${new Date(`${isoDate}T00:00:00`).toLocaleDateString()}`;
}

function PersonaCard({ persona }) {
  return (
    <div className="segment-info">
      <h2>Your Listening Profile</h2>
      <div className="listener-personality">
        <div className="personality-icon">
          <i className="fas fa-headphones-alt"></i>
        </div>
        <div className="personality-details">
          <h3>{persona.name}</h3>
          <p>
            Matched from a survey of Spotify listeners: {persona.description}
          </p>
          <div className="personality-traits">
            {persona.traits.map((trait) => (
              <span key={trait.feature} className="trait-tag" title={`${Math.round(trait.share * 100)}% of this group`}>
                {trait.label} · {trait.lift.toFixed(1)}×
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function MatchBreakdown({ match }) {
  return (
    <ul className="match-breakdown" aria-label="Why this score">
      {Object.entries(BREAKDOWN_LABELS).map(([key, label]) => {
        const pct = Math.round(match.breakdown[key] * 100);
        return (
          <li key={key}>
            <span>{`${label} ${pct}%`}</span>
            <span className="match-bar" aria-hidden="true">
              <span style={{ width: `${pct}%` }}></span>
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function PodcastCard({ podcast, saved, onSave }) {
  const titleId = `podcast-${podcast.id}`;
  const lastEpisode = lastEpisodeText(podcast.last_episode_date);
  const isAi = podcast.source === 'ai';
  return (
    <article className="recommendation-card" aria-labelledby={titleId}>
      <div className="podcast-info">
        <div className="podcast-header">
          {podcast.artwork_url && (
            <img className="podcast-artwork" src={podcast.artwork_url} alt={podcast.title} loading="lazy" />
          )}
          <div className="podcast-heading">
            <h3 id={titleId}>{podcast.title}</h3>
            <div className="podcast-creator">
              <i className="fas fa-user-circle"></i>
              {podcast.publisher}
            </div>
            <span className="verified-badge">
              <i className="fas fa-check-circle"></i> Verified on Apple Podcasts
            </span>
          </div>
          <span className="podcast-badge" title="Match score">{`${podcast.match.score}% match`}</span>
        </div>

        {podcast.summary && (
          <p className="podcast-description">
            {podcast.summary} <span className="ai-label">AI-written</span>
          </p>
        )}

        {podcast.reason && (
          <div className="match-reason">
            <i className={`fas ${isAi ? 'fa-magic' : 'fa-chart-line'}`}></i>
            <span>
              {podcast.reason}
              {isAi && <span className="ai-label">AI-written</span>}
            </span>
          </div>
        )}

        <div className="podcast-meta">
          {podcast.typical_episode_minutes != null && (
            <div className="podcast-duration">
              <i className="fas fa-clock"></i>
              <span>{`~${podcast.typical_episode_minutes} min episodes`}</span>
            </div>
          )}
          {lastEpisode && (
            <div className="podcast-fresh">
              <i className="fas fa-calendar-check"></i>
              <span>{lastEpisode}</span>
            </div>
          )}
          {podcast.genres.length > 0 && (
            <div className="podcast-genre">
              <i className="fas fa-tag"></i>
              <span>{podcast.genres.slice(0, 2).join(' · ')}</span>
            </div>
          )}
          {podcast.language && (
            <div className="podcast-language">
              <i className="fas fa-language"></i>
              <span>{LANGUAGE_NAMES.of(podcast.language)}</span>
            </div>
          )}
          {podcast.explicit && <span className="explicit-tag">Explicit</span>}
        </div>

        <MatchBreakdown match={podcast.match} />

        <div className="podcast-actions">
          <a className="apple-link" href={podcast.apple_url} target="_blank" rel="noopener noreferrer">
            <i className="fas fa-podcast"></i> Apple Podcasts
          </a>
          <a className="spotify-link" href={podcast.spotify_url} target="_blank" rel="noopener noreferrer">
            <i className="fab fa-spotify"></i> Search on Spotify
          </a>
          <button
            type="button"
            className={`save-button ${saved ? 'saved' : ''}`}
            onClick={() => onSave(podcast)}
            disabled={saved}
          >
            <i className={`${saved ? 'fas' : 'far'} fa-bookmark`}></i>
            {saved ? 'Saved' : 'Save to Library'}
          </button>
        </div>
      </div>
    </article>
  );
}

const RecommendationDisplay = ({ data, onReset }) => {
  const [savedKeys, setSavedKeys] = useState(() => new Set(loadSaved().map(keyOf)));
  const { persona, recommendations, meta } = data;

  const handleSave = (podcast) => {
    setSavedKeys(new Set(savePodcast(podcast).map(keyOf)));
  };

  return (
    <div className="recommendation-container">
      <PersonaCard persona={persona} />

      <div className="recommendations">
        <h2>Your Podcast Recommendations</h2>

        {meta.mode === 'catalog_only' && (
          <div className="mode-notice" role="status">
            <i className="fas fa-info-circle"></i> The AI curator is unavailable right now, so these
            are top chart picks for your topics from Apple Podcasts, ranked against your answers.
          </div>
        )}

        <div className="recommendation-list">
          {recommendations.length > 0 ? (
            recommendations.map((podcast) => (
              <PodcastCard
                key={podcast.id}
                podcast={podcast}
                saved={savedKeys.has(keyOf(podcast))}
                onSave={handleSave}
              />
            ))
          ) : (
            <div className="no-recommendations">
              <p>No verified shows matched these preferences. Try other topics or another language.</p>
            </div>
          )}
        </div>

        {meta.dropped_unverified > 0 && (
          <p className="verification-note">
            {meta.dropped_unverified} AI suggestion{meta.dropped_unverified === 1 ? '' : 's'} couldn't
            be found in Apple Podcasts and {meta.dropped_unverified === 1 ? 'was' : 'were'} left out.
          </p>
        )}

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
          Developed by{' '}
          <a href="https://www.linkedin.com/in/hemanthrayudu/" target="_blank" rel="noopener noreferrer">
            Hemanth Rayudu
          </a>
        </div>
      </div>
    </div>
  );
};

export default RecommendationDisplay;
