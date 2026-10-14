// Saved podcasts live in localStorage. Entries saved by the old app have
// { name, creator } instead of { id, title, publisher }; both are supported.
const KEY = 'savedPodcasts';

export const titleOf = (p) => p.title || p.name || 'Untitled podcast';
export const publisherOf = (p) => p.publisher || p.creator || p.author || 'Unknown publisher';
export const keyOf = (p) => (p.id != null ? String(p.id) : titleOf(p));
export const spotifySearchUrl = (title) =>
  `https://open.spotify.com/search/${encodeURIComponent(title)}`;

export function loadSaved() {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || '[]');
    return Array.isArray(saved) ? saved : [];
  } catch {
    return [];
  }
}

function store(list) {
  localStorage.setItem(KEY, JSON.stringify(list));
  return list;
}

export function savePodcast(podcast) {
  const saved = loadSaved();
  if (saved.some((p) => keyOf(p) === keyOf(podcast))) return saved;
  return store([...saved, podcast]);
}

export function removePodcast(key) {
  return store(loadSaved().filter((p) => keyOf(p) !== key));
}
