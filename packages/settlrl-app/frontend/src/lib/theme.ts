// Light / dark theme switching. The palettes live in index.css as CSS
// variables keyed off body[data-theme]; this module just owns the attribute
// and its persistence.

export type Theme = "light" | "dark";

const KEY = "settlrl-theme";

export function currentTheme(): Theme {
  return document.body.dataset.theme === "dark" ? "dark" : "light";
}

export function initTheme(): void {
  const stored = localStorage.getItem(KEY);
  document.body.dataset.theme = stored === "dark" ? "dark" : "light";
}

export function toggleTheme(): Theme {
  const next: Theme = currentTheme() === "dark" ? "light" : "dark";
  document.body.dataset.theme = next;
  localStorage.setItem(KEY, next);
  return next;
}
