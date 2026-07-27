import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import UserForm from './UserForm';

// Fill every required field so the form is submittable (user-event v13 API).
function fillRequiredFields() {
  userEvent.selectOptions(screen.getByLabelText(/Age Group/i), '25-34');
  userEvent.selectOptions(screen.getByLabelText(/Preferred Language/i), 'English');
  userEvent.selectOptions(screen.getByLabelText(/Content Region/i), 'Global');
  userEvent.selectOptions(screen.getByLabelText(/Preferred format/i), 'Interview');
  userEvent.selectOptions(screen.getByLabelText(/Preferred length/i), 'Medium (30-60 min)');
  userEvent.selectOptions(screen.getByLabelText(/listening mood/i), 'Be Entertained');
  userEvent.selectOptions(screen.getByLabelText(/How often do you listen/i), 'Weekly');
  userEvent.click(screen.getByLabelText('Science & Technology'));
  userEvent.click(screen.getByLabelText('Pop'));
}

test('renders key form fields', () => {
  render(<UserForm onSubmit={() => {}} isLoading={false} />);
  expect(screen.getByLabelText(/Age Group/i)).toBeInTheDocument();
  expect(
    screen.getByRole('button', { name: /Get Personalized Recommendations/i })
  ).toBeInTheDocument();
});

test('blocks submit and shows errors when genre/topic checkboxes are empty', async () => {
  const onSubmit = jest.fn();
  render(<UserForm onSubmit={onSubmit} isLoading={false} />);

  userEvent.click(
    screen.getByRole('button', { name: /Get Personalized Recommendations/i })
  );

  expect(await screen.findByText(/select at least one topic/i)).toBeInTheDocument();
  expect(screen.getByText(/select at least one music genre/i)).toBeInTheDocument();
  expect(onSubmit).not.toHaveBeenCalled();
});

test('submits the entered preferences when the form is valid', () => {
  const onSubmit = jest.fn();
  render(<UserForm onSubmit={onSubmit} isLoading={false} />);

  fillRequiredFields();
  userEvent.click(
    screen.getByRole('button', { name: /Get Personalized Recommendations/i })
  );

  expect(onSubmit).toHaveBeenCalledTimes(1);
  const submitted = onSubmit.mock.calls[0][0];
  expect(submitted.age).toBe('25-34');
  expect(submitted.music_genre).toContain('Pop');
  expect(submitted.podcast_content).toContain('Science & Technology');
});
