/**
 * Signal colors for points, activations, weights, and legends.
 *
 * Each theme defines its own validated steps as CSS custom properties in
 * styles.css (violet for positive values, amber for negative, and a neutral
 * midpoint close to the chart surface). Canvas and SVG rendering read the
 * active theme's values here, so CSS stays the single source of truth.
 */
export type Theme = 'light' | 'dark';

export interface Palette {
  negative: string;
  neutral: string;
  positive: string;
}

export const THEME_CHANGE_EVENT = 'nnvisualizer:theme-change';

const FALLBACK: Palette = { negative: '#cf6d01', neutral: '#eaebf0', positive: '#744ed7' };
let cached: Palette | null = null;

export function currentTheme(): Theme {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
}

/** The active theme's signal colors. */
export function palette(): Palette {
  if (!cached) {
    const style = getComputedStyle(document.documentElement);
    const read = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
    cached = {
      negative: read('--signal-negative', FALLBACK.negative),
      neutral: read('--signal-neutral', FALLBACK.neutral),
      positive: read('--signal-positive', FALLBACK.positive),
    };
  }
  return cached;
}

/** Colors ordered for scales whose domain runs from negative through zero to positive. */
export function signalRange(): string[] {
  const { negative, neutral, positive } = palette();
  return [negative, neutral, positive];
}

/** Switches the theme and lets every canvas and SVG renderer repaint with its colors. */
export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  cached = null;
  window.dispatchEvent(new CustomEvent<Theme>(THEME_CHANGE_EVENT, { detail: theme }));
}
