import React from 'react';
import { Link } from 'react-router-dom';
import './About.css';

const About = () => {
  return (
    <div className="about-container">
      <div className="about-header">
        <h2>About</h2>
        <p>Personalized podcast recommendations, powered by machine learning and AI.</p>
      </div>

      <section className="about-section">
        <h3><i className="fas fa-lightbulb"></i> How it works</h3>
        <p>
          Tell us about your listening habits and interests. A model trained on a survey of
          Spotify listeners matches you to a listener persona. An open-weight LLM then suggests
          shows for that profile.
        </p>
        <p>
          Every suggestion is checked against the Apple Podcasts catalog, and anything that can't
          be found is dropped. The artwork, links, real episode lengths, latest-episode dates and
          feed language all come from the catalog, not the AI. Each show's match score shows how
          well it fits your topics, preferred length and freshness.
        </p>
      </section>

      <section className="about-section">
        <h3><i className="fas fa-cogs"></i> Technology</h3>
        <ul className="about-tech">
          <li><span>Frontend</span> React, React Router, Vite</li>
          <li><span>Backend</span> FastAPI (Python)</li>
          <li><span>ML</span> KMeans listener personas (trained with scikit-learn, served with numpy)</li>
          <li><span>AI</span> gpt-oss-120b (open-weight, via Groq) proposes candidate shows</li>
          <li><span>Catalog</span> Apple iTunes Search API + each show's RSS feed</li>
        </ul>
      </section>

      <div className="about-footer">
        <Link to="/" className="about-cta">
          <i className="fas fa-headphones-alt"></i>
          Get Recommendations
        </Link>
        <p className="about-credit">
          Developed by{' '}
          <a
            href="https://www.linkedin.com/in/hemanthrayudu/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Hemanth Rayudu
          </a>
        </p>
      </div>
    </div>
  );
};

export default About;
