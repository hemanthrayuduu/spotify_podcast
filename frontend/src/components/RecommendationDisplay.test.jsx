import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import RecommendationDisplay from './RecommendationDisplay';
import recommendAi from '../test/fixtures/recommend_ai.json';
import recommendCatalog from '../test/fixtures/recommend_catalog_only.json';

const renderDisplay = (data) =>
  render(
    <MemoryRouter>
      <RecommendationDisplay data={data} onReset={() => {}} />
    </MemoryRouter>
  );

const card = (title) => screen.getByRole('article', { name: title });

test('shows the survey persona with its lifted traits', () => {
  renderDisplay(recommendAi);
  const { persona } = recommendAi;
  expect(screen.getByText(persona.name)).toBeInTheDocument();
  const trait = persona.traits[0];
  expect(screen.getByText(new RegExp(`${trait.label}.*${trait.lift.toFixed(1)}×`))).toBeInTheDocument();
});

test('renders one card per verified show with real catalog metadata', () => {
  renderDisplay(recommendAi);
  expect(screen.getAllByRole('article')).toHaveLength(recommendAi.recommendations.length);

  const rec = recommendAi.recommendations[0];
  const c = within(card(rec.title));
  expect(c.getByText(rec.publisher)).toBeInTheDocument();
  expect(c.getByRole('img', { name: rec.title })).toHaveAttribute('src', rec.artwork_url);
  expect(c.getByText(`~${rec.typical_episode_minutes} min episodes`)).toBeInTheDocument();
  expect(c.getByText(/verified on apple podcasts/i)).toBeInTheDocument();
  expect(c.getByText(`${rec.match.score}% match`)).toBeInTheDocument();
});

test('links to the show on Apple Podcasts and to a Spotify search', () => {
  renderDisplay(recommendAi);
  const rec = recommendAi.recommendations[0];
  const c = within(card(rec.title));
  const apple = c.getByRole('link', { name: /apple podcasts/i });
  expect(apple).toHaveAttribute('href', rec.apple_url);
  expect(apple).toHaveAttribute('rel', expect.stringContaining('noopener'));
  expect(c.getByRole('link', { name: /spotify/i })).toHaveAttribute('href', rec.spotify_url);
});

test('labels AI-written text and explains the match score', () => {
  renderDisplay(recommendAi);
  const ai = recommendAi.recommendations.find((r) => r.source === 'ai');
  const c = within(card(ai.title));
  expect(c.getByText(ai.summary)).toBeInTheDocument();
  expect(c.getAllByText(/ai-written/i).length).toBeGreaterThan(0);
  const { breakdown } = ai.match;
  expect(c.getByText(`Topic ${Math.round(breakdown.topic * 100)}%`)).toBeInTheDocument();
  expect(c.getByText(`Length ${Math.round(breakdown.duration * 100)}%`)).toBeInTheDocument();
});

test('explains catalog-only mode', () => {
  renderDisplay(recommendCatalog);
  expect(screen.getByRole('status')).toHaveTextContent(/top chart/i);
});

test('saves a show to the library', async () => {
  const user = userEvent.setup();
  renderDisplay(recommendAi);
  const rec = recommendAi.recommendations[0];
  const c = within(card(rec.title));
  await user.click(c.getByRole('button', { name: /save to library/i }));

  expect(c.getByRole('button', { name: /saved/i })).toBeInTheDocument();
  expect(JSON.parse(localStorage.getItem('savedPodcasts'))[0].id).toBe(rec.id);
});

test('warns when catalog lookups failed and results may be incomplete', () => {
  renderDisplay({ ...recommendAi, meta: { ...recommendAi.meta, catalog_errors: 2 } });
  expect(screen.getByRole('status')).toHaveTextContent(/incomplete/i);
});

test('renders a persona that shares no traits with the user', () => {
  const persona = { ...recommendAi.persona, name: 'Eclectic listener', traits: [], description: 'Your answers lean on themselves.' };
  renderDisplay({ ...recommendAi, persona });
  expect(screen.getByText('Eclectic listener')).toBeInTheDocument();
  expect(screen.getByText('Your answers lean on themselves.')).toBeInTheDocument();
});
