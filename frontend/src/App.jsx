import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import SpotifyHeader from './components/SpotifyHeader';
import UserForm from './components/UserForm';
import RecommendationDisplay from './components/RecommendationDisplay';
import Library from './pages/Library';
import About from './pages/About';
import './App.css';

function App() {
  const [recommendations, setRecommendations] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (formData) => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to get recommendations');
      }

      const data = await response.json();
      console.log("API response:", data); // Log the data to check its structure
      
      // Check if data has the expected structure from the backend API
      if (!data.recommendations || !Array.isArray(data.recommendations) || data.recommendations.length === 0) {
        // Check for old structure with podcasts array
        if (!data.podcasts || !Array.isArray(data.podcasts) || data.podcasts.length === 0) {
          console.log("API returned invalid data, using fallback data");
          // Fallback sample response if API doesn't return proper data
          setRecommendations({
            segment_profile: {
              segment_name: "Curious Explorer",
              fav_music_genre: { "Pop": 0.45, "Rock": 0.30 },
              fav_pod_genre: { "Educational": 0.55, "Interview": 0.25 }
            },
            recommendations: [
              {
                name: "The Daily",
                creator: "The New York Times",
                description: "This is what the news should sound like. The biggest stories of our time, told by the best journalists in the world.",
                format: "News & Politics",
                duration: "Short (< 30 min)",
                language: "English",
                region: "Global",
                reason: "Matches your interest in current events and short-form content"
              },
              {
                name: "RadioLab", 
                creator: "WNYC Studios",
                description: "Investigating a strange world. RadioLab is one of the most beloved podcasts in the world, exploring science, philosophy, and human experience.",
                format: "Science & Technology",
                duration: "Medium (30-60 min)",
                language: "English",
                region: "Global",
                reason: "Aligns with your curiosity about scientific discoveries"
              },
              {
                name: "Hidden Brain",
                creator: "NPR",
                description: "Explore the unconscious patterns that drive human behavior and shape our choices.",
                format: "Society & Culture",
                duration: "Medium (30-60 min)",
                language: "English",
                region: "Global",
                reason: "Perfect for your interest in psychology and human behavior"
              },
              {
                name: "Stuff You Should Know",
                creator: "iHeartRadio",
                description: "Josh and Chuck dive into a wide variety of topics and make dense subjects easy to digest.",
                format: "Educational",
                duration: "Medium (30-60 min)",
                language: "English",
                region: "Global",
                reason: "Educational and entertaining content that suits your interests"
              },
              {
                name: "Heavyweight",
                creator: "Gimlet Media",
                description: "Jonathan Goldstein goes back to the moment everything changed, with humorous and touching stories.",
                format: "Narrative",
                duration: "Medium (30-60 min)",
                language: "English",
                region: "Global",
                reason: "Engaging storytelling that matches your preference for narrative content"
              }
            ]
          });
        } else {
          // Convert old format to new format
          setRecommendations({
            segment_profile: {
              segment_name: data.user_segment || "Podcast Enthusiast",
              segment_description: data.segment_description || "You enjoy a variety of podcast content"
            },
            recommendations: data.podcasts.map(podcast => ({
              name: podcast.title,
              creator: podcast.author,
              description: podcast.description,
              format: podcast.genre,
              language: podcast.language || "English",
              region: podcast.region || "Global",
              reason: podcast.match_reason,
              match_score: podcast.match_score
            }))
          });
        }
      } else {
        // Data is already in the expected format from the backend
        setRecommendations(data);
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to get recommendations. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setRecommendations(null);
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
                <UserForm 
                  onSubmit={handleSubmit} 
                  isLoading={isLoading} 
                />
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