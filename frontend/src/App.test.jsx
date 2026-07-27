import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import App from './App';

// user-event v13 API (synchronous, no setup()).
function fillAndSubmit() {
  userEvent.selectOptions(screen.getByLabelText(/Age Group/i), '25-34');
  userEvent.selectOptions(screen.getByLabelText(/Preferred Language/i), 'English');
  userEvent.selectOptions(screen.getByLabelText(/Content Region/i), 'Global');
  userEvent.selectOptions(screen.getByLabelText(/Preferred format/i), 'Interview');
  userEvent.selectOptions(screen.getByLabelText(/Preferred length/i), 'Medium (30-60 min)');
  userEvent.selectOptions(screen.getByLabelText(/listening mood/i), 'Be Entertained');
  userEvent.selectOptions(screen.getByLabelText(/How often do you listen/i), 'Weekly');
  userEvent.click(screen.getByLabelText('Science & Technology'));
  userEvent.click(screen.getByLabelText('Pop'));
  userEvent.click(
    screen.getByRole('button', { name: /Get Personalized Recommendations/i })
  );
}

const MOCK_RESPONSE = {
  segment_profile: { fav_music_genre: { Pop: 0.5 }, fav_pod_genre: { Technology: 0.6 } },
  recommendations: [
    {
      name: 'RadioLab',
      creator: 'WNYC Studios',
      description: 'Investigating a strange world.',
      format: 'Science & Technology',
      duration: 'Medium (30-60 min)',
      language: 'English',
      region: 'Global',
      reason: 'Matches your curiosity.',
      link: 'https://www.google.com/search?q=RadioLab+podcast',
    },
  ],
};

afterEach(() => {
  jest.restoreAllMocks();
});

test('renders recommendations returned by the API', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue({
    ok: true,
    json: async () => MOCK_RESPONSE,
  });

  render(
    <MemoryRouter>
      <App />
    </MemoryRouter>
  );

  fillAndSubmit();

  expect(await screen.findByText('RadioLab')).toBeInTheDocument();
  expect(global.fetch).toHaveBeenCalledTimes(1);
});

test('shows an inline error banner when the API fails', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue({
    ok: false,
    status: 500,
    json: async () => ({ detail: 'Error generating recommendations' }),
  });

  render(
    <MemoryRouter>
      <App />
    </MemoryRouter>
  );

  fillAndSubmit();

  const alert = await screen.findByRole('alert');
  expect(alert).toHaveTextContent(/Error generating recommendations/i);
  // The form is still visible so the user can retry.
  expect(
    screen.getByRole('button', { name: /Get Personalized Recommendations/i })
  ).toBeInTheDocument();
});
