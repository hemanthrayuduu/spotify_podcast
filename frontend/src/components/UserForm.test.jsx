import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import UserForm from './UserForm';
import options from '../test/fixtures/options.json';
import { fillRequired } from '../test/formHelpers';

const submit = () => screen.getByRole('button', { name: /Get Personalized Recommendations/i });

test('renders every field from the options payload', () => {
  render(<UserForm fields={options.fields} onSubmit={() => {}} />);
  for (const field of options.fields) {
    expect(screen.getByText(field.label)).toBeInTheDocument();
  }
  expect(screen.getByLabelText(/Podcasts you already enjoy/i)).toBeInTheDocument();
});

test('submits option values, not labels', async () => {
  const user = userEvent.setup();
  const onSubmit = vi.fn();
  render(<UserForm fields={options.fields} onSubmit={onSubmit} />);
  await fillRequired(user);
  await user.type(screen.getByLabelText(/Podcasts you already enjoy/i), 'Radiolab');
  await user.click(submit());

  expect(onSubmit).toHaveBeenCalledTimes(1);
  expect(onSubmit.mock.calls[0][0]).toEqual({
    age: '20_34',
    music_genre: ['pop'],
    podcast_content: ['technology'],
    podcast_frequency: 'weekly',
    podcast_format: 'interview',
    podcast_duration: 'medium',
    content_language: 'english',
    region: 'global',
    listening_mood: 'learn',
    podcasts_enjoyed: 'Radiolab',
  });
});

test('blocks submit and shows inline errors when no topic or genre is picked', async () => {
  const user = userEvent.setup();
  const onSubmit = vi.fn();
  render(<UserForm fields={options.fields} onSubmit={onSubmit} />);
  await user.click(submit());

  expect(await screen.findByText(/select at least one topic/i)).toBeInTheDocument();
  expect(screen.getByText(/select at least one music genre/i)).toBeInTheDocument();
  expect(onSubmit).not.toHaveBeenCalled();
});

test('lists missing required fields inline instead of using alert()', async () => {
  const user = userEvent.setup();
  const alert = vi.spyOn(window, 'alert').mockImplementation(() => {});
  const onSubmit = vi.fn();
  render(<UserForm fields={options.fields} onSubmit={onSubmit} />);
  await user.click(screen.getByLabelText('Technology'));
  await user.click(screen.getByLabelText('Pop'));
  await user.click(submit());

  expect(await screen.findByRole('alert')).toHaveTextContent(/Age group/);
  expect(alert).not.toHaveBeenCalled();
  expect(onSubmit).not.toHaveBeenCalled();
});
