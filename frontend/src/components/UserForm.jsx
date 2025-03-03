import React, { useState, useEffect, useRef } from 'react';
import './UserForm.css';
import LoadingSpinner from './LoadingSpinner';

const UserForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    age: "",
    music_genre: "",
    podcast_frequency: "",
    podcast_duration: "",
    podcast_format: "",
    podcast_content: "",
    content_language: "English",
    region: "",
  });
  
  const musicNotesRef = useRef(null);
  
  // Create floating music notes animation
  useEffect(() => {
    if (!musicNotesRef.current) return;
    
    const container = musicNotesRef.current;
    const noteSymbols = ['♪', '♫', '𝅘𝅥𝅮', '𝄞', '𝅗𝅥', '𝅘𝅥𝅯'];
    const noteCount = 12;
    
    // Remove any existing notes
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
    
    // Create new notes
    for (let i = 0; i < noteCount; i++) {
      const note = document.createElement('div');
      note.className = 'music-note';
      note.textContent = noteSymbols[Math.floor(Math.random() * noteSymbols.length)];
      
      // Random position
      note.style.left = `${Math.random() * 100}%`;
      note.style.top = `${Math.random() * 100}%`;
      
      // Random movement direction
      note.style.setProperty('--move-x', `${(Math.random() * 200) - 100}px`);
      note.style.setProperty('--move-y', `${(Math.random() * -200) - 50}px`);
      note.style.setProperty('--rotate', `${(Math.random() * 360)}deg`);
      
      // Random animation duration and delay
      note.style.animationDuration = `${8 + (Math.random() * 10)}s`;
      note.style.animationDelay = `${Math.random() * 5}s`;
      
      container.appendChild(note);
    }
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  // Form options
  const ageOptions = ["18-24", "25-34", "35-44", "45-54", "55+"];
  const musicGenreOptions = ["Pop", "Rock", "Hip-Hop", "Electronic", "Jazz", "Classical", "Country", "R&B", "Metal", "Folk", "Indie"];
  const podFrequencyOptions = ["Daily", "Several times a week", "Weekly", "Monthly", "Rarely"];
  const podDurationOptions = ["Short (< 30 min)", "Medium (30-60 min)", "Long (> 60 min)"];
  const podFormatOptions = ["Interview", "Solo", "Panel discussion", "Narrative/Storytelling", "Educational", "News/Current events"];
  const podContentOptions = ["Science & Technology", "Business & Finance", "Art & Culture", "News & Politics", "Health & Wellness", "Education", "Entertainment", "Sports", "True Crime", "History", "Philosophy", "Comedy"];
  const languageOptions = ["English", "Spanish", "French", "German", "Mandarin", "Hindi", "Japanese", "Korean", "Portuguese", "Russian", "Arabic", "Italian"];
  const regionOptions = ["North America", "Europe", "Asia", "South America", "Africa", "Australia/Oceania", "Global"];

  if (isLoading) {
    return <LoadingSpinner />;
  }

  return (
    <div className="form-container">
      <div className="music-notes" ref={musicNotesRef}></div>
      
      <div className="form-header">
        <h2>Find Your <span>Perfect Podcast</span></h2>
        <p>Quick form - takes less than a minute to get personalized recommendations</p>
      </div>

      <form onSubmit={handleSubmit} className="user-form">
        <div className="form-section">
          <h3>
            <i className="fas fa-info-circle form-section-icon"></i>
            Key Information
          </h3>
          
          <div className="form-group">
            <label htmlFor="age">Age Group</label>
            <select
              id="age"
              name="age"
              value={formData.age}
              onChange={handleChange}
              required
            >
              <option value="">Select age group</option>
              {ageOptions.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="content_language">
                <i className="fas fa-language"></i> Preferred Language
              </label>
              <select
                id="content_language"
                name="content_language"
                value={formData.content_language}
                onChange={handleChange}
                required
              >
                <option value="">Select language</option>
                {languageOptions.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
              <small className="form-help">Language you prefer to listen in</small>
            </div>

            <div className="form-group">
              <label htmlFor="region">
                <i className="fas fa-globe-americas"></i> Content Region
              </label>
              <select
                id="region"
                name="region"
                value={formData.region}
                onChange={handleChange}
                required
              >
                <option value="">Select region</option>
                {regionOptions.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
              <small className="form-help">Region whose content you're interested in</small>
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3>
            <i className="fas fa-podcast form-section-icon"></i>
            Content Preferences
          </h3>
          
          <div className="form-group">
            <label htmlFor="podcast_content">
              <i className="fas fa-list-ul"></i> What topics interest you?
            </label>
            <select
              id="podcast_content"
              name="podcast_content"
              value={formData.podcast_content}
              onChange={handleChange}
              required
            >
              <option value="">Select a topic</option>
              {podContentOptions.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
            <small className="form-help">The primary topic you want to hear about</small>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="podcast_format">
                <i className="fas fa-microphone-alt"></i> Preferred format
              </label>
              <select
                id="podcast_format"
                name="podcast_format"
                value={formData.podcast_format}
                onChange={handleChange}
                required
              >
                <option value="">Select format</option>
                {podFormatOptions.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
              <small className="form-help">How you like content presented</small>
            </div>

            <div className="form-group">
              <label htmlFor="podcast_duration">
                <i className="fas fa-clock"></i> Preferred length
              </label>
              <select
                id="podcast_duration"
                name="podcast_duration"
                value={formData.podcast_duration}
                onChange={handleChange}
                required
              >
                <option value="">Select length</option>
                {podDurationOptions.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
              <small className="form-help">How long you like your podcasts</small>
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3>
            <i className="fas fa-headphones-alt form-section-icon"></i>
            Your Listening Habits
          </h3>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="music_genre">
                <i className="fas fa-music"></i> Music you enjoy
              </label>
              <select
                id="music_genre"
                name="music_genre"
                value={formData.music_genre}
                onChange={handleChange}
                required
              >
                <option value="">Select genre</option>
                {musicGenreOptions.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
              <small className="form-help">Helps us understand your audio preferences</small>
            </div>

            <div className="form-group">
              <label htmlFor="podcast_frequency">
                <i className="fas fa-calendar-alt"></i> How often do you listen?
              </label>
              <select
                id="podcast_frequency"
                name="podcast_frequency"
                value={formData.podcast_frequency}
                onChange={handleChange}
                required
              >
                <option value="">Select frequency</option>
                {podFrequencyOptions.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
              <small className="form-help">Your typical podcast consumption</small>
            </div>
          </div>
        </div>

        <div className="form-submit">
          <button type="submit" className="submit-button">
            <i className="fas fa-headphones"></i>
            <span>Get Personalized Recommendations</span>
          </button>
        </div>
        
        <div className="footer-credit">
          Developed by <a href="https://www.linkedin.com/in/hemanthrayudu/" target="_blank" rel="noopener noreferrer">Hemanth Rayudu</a>
        </div>
      </form>
    </div>
  );
};

export default UserForm;