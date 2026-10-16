// Thin client for the backend API. Errors carry a message fit to show users.
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function errorMessage(response, fallback) {
  try {
    const { detail } = await response.json();
    if (typeof detail === 'string') return detail;
    // FastAPI 422: [{ loc: ['body', 'age'], msg: '...' }, ...]
    if (Array.isArray(detail) && detail.length) {
      const { loc = [], msg = '' } = detail[0];
      return `${loc[loc.length - 1] ?? 'request'}: ${msg}`;
    }
  } catch {
    // Non-JSON error body; use the fallback.
  }
  return fallback;
}

export async function getOptions() {
  const response = await fetch(`${API_URL}/options`);
  if (!response.ok) throw new Error(await errorMessage(response, 'Could not load the form.'));
  return response.json();
}

export async function getRecommendations(preferences) {
  const response = await fetch(`${API_URL}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(preferences),
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, 'Something went wrong. Please try again.'));
  }
  return response.json();
}
