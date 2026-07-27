import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import SpotifyHeader from './components/SpotifyHeader';
import UserForm from './components/UserForm';
import RecommendationDisplay from './components/RecommendationDisplay';
import Library from './pages/Library';
import About from './pages/About';
import './App.css';

// Use environment variable for API URL with localhost as fallback
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [recommendations, setRecommendations] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (formData) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        let detail = 'Something went wrong. Please try again.';
        try {
          const errorData = await response.json();
          if (typeof errorData.detail === 'string') {
            detail = errorData.detail;
          }
        } catch (e) {
          // Non-JSON error body; keep the generic message.
        }
        throw new Error(detail);
      }

      // The backend guarantees the { segment_profile, recommendations } shape.
      const data = await response.json();
      setRecommendations(data);
    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'Failed to get recommendations. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setRecommendations(null);
    setError(null);
  };

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
          <Route
            path="/"
            element={
              recommendations ? (
                <RecommendationDisplay
                  recommendations={recommendations}
                  onReset={handleReset}
                />
              ) : (
                <>
                  {error && (
                    <div className="error-banner" role="alert">
                      <i className="fas fa-exclamation-circle"></i>
                      <span>{error}</span>
                    </div>
                  )}
                  <UserForm onSubmit={handleSubmit} isLoading={isLoading} />
                </>
              )
            }
          />
          <Route path="/library" element={<Library />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
