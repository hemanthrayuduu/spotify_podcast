import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, vi } from 'vitest';
import App from './App';
import { fillRequired } from './test/formHelpers';
import recommendAi from './test/fixtures/recommend_ai.json';
import { mockFetch } from './test/mockFetch';

afterEach(() => vi.restoreAllMocks());

const renderApp = () =>
  render(
    <MemoryRouter>
      <App />
    </MemoryRouter>
  );

test('builds the form from GET /options and shows verified recommendations', async () => {
  const user = userEvent.setup();
  const fetch = mockFetch({ recommend: recommendAi });
  renderApp();

  await screen.findByLabelText(/Age group/i);
  await fillRequired(user);
  await user.click(screen.getByRole('button', { name: /Get Personalized Recommendations/i }));

  expect(await screen.findByText('The Rest Is History')).toBeInTheDocument();
  expect(screen.getByText(recommendAi.persona.name)).toBeInTheDocument();
  const [, init] = fetch.mock.calls.find(([url]) => url.endsWith('/recommend'));
  expect(JSON.parse(init.body)).toMatchObject({ age: '20_34', podcast_content: ['technology'] });
});

test('shows an inline error banner when recommending fails', async () => {
  const user = userEvent.setup();
  mockFetch({
    recommend: { status: 503, body: { detail: 'The podcast catalog is unavailable right now.' } },
  });
  renderApp();

  await screen.findByLabelText(/Age group/i);
  await fillRequired(user);
  await user.click(screen.getByRole('button', { name: /Get Personalized Recommendations/i }));

  expect(await screen.findByRole('alert')).toHaveTextContent(/catalog is unavailable/i);
  // The form stays so the user can retry.
  expect(screen.getByRole('button', { name: /Get Personalized Recommendations/i })).toBeInTheDocument();
});

test('explains validation errors from the API', async () => {
  const user = userEvent.setup();
  mockFetch({
    recommend: { status: 422, body: { detail: [{ loc: ['body', 'age'], msg: "Value error, 'x' is not a valid option" }] } },
  });
  renderApp();

  await screen.findByLabelText(/Age group/i);
  await fillRequired(user);
  await user.click(screen.getByRole('button', { name: /Get Personalized Recommendations/i }));

  expect(await screen.findByRole('alert')).toHaveTextContent(/age.*not a valid option/i);
});

test('offers a retry when the form options cannot be loaded', async () => {
  const user = userEvent.setup();
  mockFetch({ options: { status: 500, body: {} } });
  renderApp();

  expect(await screen.findByRole('alert')).toHaveTextContent(/couldn't load/i);
  vi.restoreAllMocks();
  mockFetch();
  await user.click(screen.getByRole('button', { name: /try again/i }));
  expect(await screen.findByLabelText(/Age group/i)).toBeInTheDocument();
});

test('keeps the answers after a failed request so the user can retry', async () => {
  const user = userEvent.setup();
  mockFetch({ recommend: { status: 503, body: { detail: 'The podcast catalog is unavailable right now.' } } });
  renderApp();

  await screen.findByLabelText(/Age group/i);
  await fillRequired(user);
  await user.click(screen.getByRole('button', { name: /Get Personalized Recommendations/i }));
  await screen.findByRole('alert');

  expect(screen.getByLabelText(/Age group/i)).toHaveValue('20_34');
  expect(screen.getByLabelText('Technology')).toBeChecked();
});
