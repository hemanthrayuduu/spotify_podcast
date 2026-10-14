import React, { useCallback, useEffect, useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import SpotifyHeader from './components/SpotifyHeader';
import UserForm from './components/UserForm';
import RecommendationDisplay from './components/RecommendationDisplay';
import LoadingSpinner from './components/LoadingSpinner';
import Library from './pages/Library';
import About from './pages/About';
import { getOptions, getRecommendations } from './api';
import './App.css';

function ErrorBanner({ children, action }) {
  return (
    <div className="error-banner" role="alert">
      <i className="fas fa-exclamation-circle"></i>
      <span>{children}</span>
      {action}
    </div>
  );
}

function App() {
  // The form is rendered from GET /options so it always matches the API.
  const [fields, setFields] = useState(null);
  const [optionsError, setOptionsError] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadOptions = useCallback(async () => {
    setOptionsError(null);
    try {
      setFields((await getOptions()).fields);
    } catch (err) {
      setOptionsError(err.message);
    }
  }, []);

  useEffect(() => {
    loadOptions();
  }, [loadOptions]);

  const handleSubmit = async (preferences) => {
    setIsLoading(true);
    setError(null);
    try {
      setRecommendations(await getRecommendations(preferences));
    } catch (err) {
      setError(err.message || 'Failed to get recommendations. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setRecommendations(null);
    setError(null);
  };

  let home;
  if (recommendations) {
    home = <RecommendationDisplay data={recommendations} onReset={handleReset} />;
  } else if (optionsError) {
    home = (
      <ErrorBanner
        action={
          <button type="button" className="retry-button" onClick={loadOptions}>
            Try again
          </button>
        }
      >
        We couldn't load the form ({optionsError}). The server may be waking up.
      </ErrorBanner>
    );
  } else if (!fields) {
    home = <LoadingSpinner message="Loading…" />;
  } else {
    home = (
      <>
        {error && <ErrorBanner>{error}</ErrorBanner>}
        {isLoading && <LoadingSpinner />}
        {/* Hidden, not unmounted, while loading: a failed request keeps the answers. */}
        <div hidden={isLoading}>
          <UserForm fields={fields} onSubmit={handleSubmit} />
        </div>
      </>
    );
  }

  return (
    <div className="App">
      {/* Ambient background elements */}
      <div className="ambient-bg">
        <div className="ambient-circle ambient-circle-1"></div>
        <div className="ambient-circle ambient-circle-2"></div>
      </div>

      <SpotifyHeader />

      <main>
        <Routes>
          <Route path="/" element={home} />
          <Route path="/library" element={<Library />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
