import React, { useState, useEffect, useRef } from 'react';
import './UserForm.css';

// Layout only: the fields, labels and options themselves come from GET /options.
const SECTIONS = [
  { title: 'Key Information', icon: 'fa-info-circle', fields: ['age', 'content_language', 'region'] },
  {
    title: 'Content Preferences',
    icon: 'fa-podcast',
    fields: ['podcast_content', 'podcast_format', 'podcast_duration', 'listening_mood'],
  },
  {
    title: 'Your Listening Habits',
    icon: 'fa-headphones-alt',
    fields: ['music_genre', 'podcast_frequency'],
  },
];

const ICONS = {
  age: 'fa-user',
  content_language: 'fa-language',
  region: 'fa-globe-americas',
  podcast_content: 'fa-list-ul',
  podcast_format: 'fa-microphone-alt',
  podcast_duration: 'fa-clock',
  listening_mood: 'fa-smile',
  music_genre: 'fa-music',
  podcast_frequency: 'fa-calendar-alt',
};

const HELP = {
  content_language: 'Only shows whose feed is in this language are recommended',
  region: 'Which Apple Podcasts storefront to search',
  podcast_content: 'Select all topics that interest you',
  podcast_duration: 'Matched against real episode lengths',
  music_genre: 'Used to match you with similar survey listeners',
};

const MULTI_ERRORS = {
  podcast_content: 'Please select at least one topic',
  music_genre: 'Please select at least one music genre',
};

const DEFAULTS = { content_language: 'english' };

const initialValues = (fields) =>
  Object.fromEntries(fields.map((f) => [f.name, f.multi ? [] : DEFAULTS[f.name] ?? '']));

// Fields the layout doesn't know about still render, in the last section.
function layout(fields) {
  const byName = Object.fromEntries(fields.map((f) => [f.name, f]));
  const placed = new Set(SECTIONS.flatMap((s) => s.fields));
  const extra = fields.filter((f) => !placed.has(f.name)).map((f) => f.name);
  return SECTIONS.map((section, i) => ({
    ...section,
    fields: [...section.fields, ...(i === SECTIONS.length - 1 ? extra : [])]
      .map((name) => byName[name])
      .filter(Boolean),
  }));
}

const UserForm = ({ fields, onSubmit }) => {
  const [values, setValues] = useState(() => initialValues(fields));
  const [podcastsEnjoyed, setPodcastsEnjoyed] = useState('');
  const [multiErrors, setMultiErrors] = useState({});
  const [missing, setMissing] = useState([]);
  const musicNotesRef = useRef(null);

  // Create floating music notes animation
  useEffect(() => {
    const container = musicNotesRef.current;
    if (!container) return;
    const noteSymbols = ['♪', '♫', '𝅘𝅥𝅮', '𝄞', '𝅗𝅥', '𝅘𝅥𝅯'];
    container.replaceChildren();
    for (let i = 0; i < 12; i++) {
      const note = document.createElement('div');
      note.className = 'music-note';
      note.textContent = noteSymbols[Math.floor(Math.random() * noteSymbols.length)];
      note.style.left = `${Math.random() * 100}%`;
      note.style.top = `${Math.random() * 100}%`;
      note.style.setProperty('--move-x', `${Math.random() * 200 - 100}px`);
      note.style.setProperty('--move-y', `${Math.random() * -200 - 50}px`);
      note.style.setProperty('--rotate', `${Math.random() * 360}deg`);
      note.style.animationDuration = `${8 + Math.random() * 10}s`;
      note.style.animationDelay = `${Math.random() * 5}s`;
      container.appendChild(note);
    }
  }, []);

  const setValue = (name, value) => setValues((prev) => ({ ...prev, [name]: value }));

  const toggle = (name, value, checked) => {
    const next = checked ? [...values[name], value] : values[name].filter((v) => v !== value);
    setValue(name, next);
    if (next.length) setMultiErrors((prev) => ({ ...prev, [name]: false }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const emptyMulti = Object.fromEntries(
      fields.filter((f) => f.multi).map((f) => [f.name, values[f.name].length === 0])
    );
    const missingSingles = fields.filter((f) => !f.multi && !values[f.name]);
    setMultiErrors(emptyMulti);
    setMissing(missingSingles.map((f) => f.label));

    const firstEmpty = Object.keys(emptyMulti).find((name) => emptyMulti[name]);
    if (firstEmpty) {
      document.getElementById(`${firstEmpty}-group`)?.scrollIntoView({ behavior: 'smooth' });
      return;
    }
    if (missingSingles.length) return;
    onSubmit({ ...values, podcasts_enjoyed: podcastsEnjoyed.trim() });
  };

  const renderField = (field) => {
    const label = (
      <>
        <i className={`fas ${ICONS[field.name] || 'fa-circle'}`}></i> {field.label}
      </>
    );
    const help = HELP[field.name] && <small className="form-help">{HELP[field.name]}</small>;

    if (field.multi) {
      return (
        <div
          key={field.name}
          id={`${field.name}-group`}
          className={`form-group ${multiErrors[field.name] ? 'error' : ''}`}
          role="group"
          aria-labelledby={`${field.name}-label`}
        >
          <span id={`${field.name}-label`} className="group-label">{label}</span>
          <div className="checkbox-group">
            {field.options.map((option) => {
              const id = `${field.name}-${option.value}`;
              return (
                <div key={option.value} className="checkbox-item">
                  <input
                    type="checkbox"
                    id={id}
                    value={option.value}
                    checked={values[field.name].includes(option.value)}
                    onChange={(e) => toggle(field.name, option.value, e.target.checked)}
                  />
                  <label htmlFor={id}>{option.label}</label>
                </div>
              );
            })}
          </div>
          {multiErrors[field.name] && (
            <div className="error-message">
              {MULTI_ERRORS[field.name] || 'Please select at least one option'}
            </div>
          )}
          {help}
        </div>
      );
    }

    return (
      <div key={field.name} className="form-group">
        <label htmlFor={field.name}>{label}</label>
        <select
          id={field.name}
          name={field.name}
          value={values[field.name]}
          onChange={(e) => setValue(field.name, e.target.value)}
        >
          <option value="">Select…</option>
          {field.options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        {help}
      </div>
    );
  };

  return (
    <div className="form-container">
      <div className="music-notes" ref={musicNotesRef}></div>

      <div className="form-header">
        <h2>Find Your <span>Perfect Podcast</span></h2>
        <p>Quick form — every recommendation is a real show, verified in Apple Podcasts</p>
      </div>

      <form onSubmit={handleSubmit} className="user-form" noValidate>
        {layout(fields).map((section, i) => (
          <div key={section.title} className="form-section">
            <h3>
              <i className={`fas ${section.icon} form-section-icon`}></i>
              {section.title}
            </h3>
            {section.fields.map(renderField)}
            {i === SECTIONS.length - 1 && (
              <div className="form-group">
                <label htmlFor="podcasts_enjoyed">
                  <i className="fas fa-heart"></i> Podcasts you already enjoy (optional)
                </label>
                <textarea
                  id="podcasts_enjoyed"
                  value={podcastsEnjoyed}
                  maxLength={500}
                  onChange={(e) => setPodcastsEnjoyed(e.target.value)}
                  placeholder="Comma-separated, e.g. Serial, Radiolab, The Daily"
                  rows="2"
                  className="text-input"
                ></textarea>
                <small className="form-help">We'll find similar shows and skip these ones</small>
              </div>
            )}
          </div>
        ))}

        {missing.length > 0 && (
          <div className="form-missing" role="alert">
            Please fill in: {missing.join(', ')}
          </div>
        )}

        <div className="form-submit">
          <button type="submit" className="submit-button">
            <i className="fas fa-headphones"></i>
            <span>Get Personalized Recommendations</span>
          </button>
        </div>

        <div className="footer-credit">
          Developed by{' '}
          <a href="https://www.linkedin.com/in/hemanthrayudu/" target="_blank" rel="noopener noreferrer">
            Hemanth Rayudu
          </a>
        </div>
      </form>
    </div>
  );
};

export default UserForm;
