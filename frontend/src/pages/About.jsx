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
          Tell us about your listening habits and interests. A machine learning model groups you
          with similar listeners to understand your profile, and an LLM uses that profile
          together with your preferences to suggest podcasts you're likely to enjoy.
        </p>
      </section>

      <section className="about-section">
        <h3><i className="fas fa-cogs"></i> Technology</h3>
        <ul className="about-tech">
          <li><span>Frontend</span> React, React Router</li>
          <li><span>Backend</span> FastAPI (Python)</li>
          <li><span>ML</span> scikit-learn KMeans segmentation</li>
          <li><span>AI</span> Llama (open-source, via Groq) for recommendation generation</li>
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
