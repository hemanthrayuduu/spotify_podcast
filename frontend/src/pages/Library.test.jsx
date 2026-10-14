import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import Library from './Library';
import recommendAi from '../test/fixtures/recommend_ai.json';

const renderLibrary = () =>
  render(
    <MemoryRouter>
      <Library />
    </MemoryRouter>
  );

test('shows an empty state', () => {
  renderLibrary();
  expect(screen.getByText(/your library is empty/i)).toBeInTheDocument();
});

test('lists saved shows with their real links', () => {
  const rec = recommendAi.recommendations[0];
  localStorage.setItem('savedPodcasts', JSON.stringify([rec]));
  renderLibrary();
  expect(screen.getByText(rec.title)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: /apple podcasts/i })).toHaveAttribute('href', rec.apple_url);
});

test('still renders and removes entries saved by the old app', async () => {
  const user = userEvent.setup();
  localStorage.setItem(
    'savedPodcasts',
    JSON.stringify([{ name: 'Radiolab', creator: 'WNYC Studios', description: 'Curious.' }])
  );
  renderLibrary();
  expect(screen.getByText('Radiolab')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: /spotify/i })).toHaveAttribute(
    'href',
    'https://open.spotify.com/search/Radiolab'
  );

  await user.click(screen.getByRole('button', { name: /remove radiolab/i }));
  expect(screen.getByText(/your library is empty/i)).toBeInTheDocument();
  expect(JSON.parse(localStorage.getItem('savedPodcasts'))).toEqual([]);
});
