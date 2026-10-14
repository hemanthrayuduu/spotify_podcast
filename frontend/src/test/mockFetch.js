import { vi } from 'vitest';
import options from './fixtures/options.json';

const json = (status, body) => Promise.resolve({ ok: status < 400, status, json: async () => body });

/**
 * Route fetch() by path. Each route is either a response body (200) or
 * { status, body }; a function is called per request for sequences.
 */
export function mockFetch({ options: opts = options, recommend } = {}) {
  const resolve = (route, init) => {
    const value = typeof route === 'function' ? route(init) : route;
    if (value && value.status) return json(value.status, value.body);
    return json(200, value);
  };
  return vi.spyOn(globalThis, 'fetch').mockImplementation((url, init) => {
    if (String(url).endsWith('/options')) return resolve(opts, init);
    if (String(url).endsWith('/recommend')) return resolve(recommend, init);
    return json(404, { detail: 'not found' });
  });
}
