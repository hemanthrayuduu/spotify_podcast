import { screen } from '@testing-library/react';

/** Fill every required field with known option values (user-event v14). */
export async function fillRequired(user) {
  await user.selectOptions(screen.getByLabelText(/Age group/i), '20_34');
  await user.selectOptions(screen.getByLabelText(/Content region/i), 'global');
  await user.selectOptions(screen.getByLabelText(/Preferred format/i), 'interview');
  await user.selectOptions(screen.getByLabelText(/Preferred episode length/i), 'medium');
  await user.selectOptions(screen.getByLabelText(/in the mood for/i), 'learn');
  await user.selectOptions(screen.getByLabelText(/How often do you listen/i), 'weekly');
  await user.click(screen.getByLabelText('Technology'));
  await user.click(screen.getByLabelText('Pop'));
}
