// jest-dom adds custom matchers like toBeInTheDocument().
import '@testing-library/jest-dom';

// jsdom does not implement scrollIntoView; UserForm calls it on validation error.
window.HTMLElement.prototype.scrollIntoView = () => {};
