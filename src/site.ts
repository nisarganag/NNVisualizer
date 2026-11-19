import { applyTheme, currentTheme, type Theme } from './palette';

/** Shared by every page: theme toggle and the model switcher in the header. */

const THEME_KEY = 'nnvisualizer-theme';
/** The playground's latest experiment, so returning from another model restores it. */
export const EXPERIMENT_KEY = 'nnvisualizer-experiment';

export const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function storedTheme(): Theme | null {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === 'light' || value === 'dark' ? value : null;
  } catch {
    return null;
  }
}

/**
 * The toggle follows the system theme until someone chooses one. A choice is
 * remembered and revealed as a circle that grows from the toggle.
 */
function initializeTheme(): void {
  const toggle = document.querySelector<HTMLButtonElement>('#theme-toggle');
  const themeColor = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  if (!toggle) return;
  const sync = () => {
    const dark = currentTheme() === 'dark';
    toggle.setAttribute('aria-pressed', String(dark));
    toggle.title = dark ? 'Switch to light theme' : 'Switch to dark theme';
    if (themeColor) themeColor.content = getComputedStyle(document.documentElement).getPropertyValue('--paper').trim();
  };
  const setTheme = (theme: Theme) => {
    if (theme !== currentTheme()) applyTheme(theme);
    sync();
  };
  sync();

  const system = window.matchMedia('(prefers-color-scheme: dark)');
  system.addEventListener('change', () => {
    if (!storedTheme()) setTheme(system.matches ? 'dark' : 'light');
  });

  toggle.addEventListener('click', () => {
    const next: Theme = currentTheme() === 'dark' ? 'light' : 'dark';
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      // Storage can be unavailable in private browsing; the switch still applies.
    }
    if (reducedMotion.matches) {
      setTheme(next);
      return;
    }
    const root = document.documentElement;
    if (typeof document.startViewTransition !== 'function') {
      root.classList.add('theme-fading');
      setTheme(next);
      window.setTimeout(() => root.classList.remove('theme-fading'), 400);
      return;
    }
    const bounds = toggle.getBoundingClientRect();
    const x = bounds.left + bounds.width / 2;
    const y = bounds.top + bounds.height / 2;
    const radius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
    root.classList.add('theme-switching');
    const transition = document.startViewTransition(() => setTheme(next));
    transition.ready.then(() => {
      root.animate(
        { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] },
        { duration: 640, easing: 'cubic-bezier(.65, 0, .35, 1)', pseudoElement: '::view-transition-new(root)' },
      );
    }).catch(() => {
      // A skipped transition has already applied the theme.
    });
    transition.finished.finally(() => root.classList.remove('theme-switching'));
  });
}

/** Places the highlight under the current model and keeps the playground link pointing at the last experiment. */
function initializeModelSwitcher(): void {
  const nav = document.querySelector<HTMLElement>('.main-nav');
  if (!nav) return;
  const links = Array.from(nav.querySelectorAll<HTMLAnchorElement>('.nav-button'));
  const active = links.find(link => link.getAttribute('aria-current') === 'page') ?? links[0];
  const indicator = nav.querySelector<HTMLElement>('.nav-indicator');
  const place = () => {
    nav.style.setProperty('--indicator-x', `${active.offsetLeft}px`);
    nav.style.setProperty('--indicator-width', `${active.offsetWidth}px`);
    nav.style.setProperty('--indicator-opacity', '1');
  };
  links.forEach(link => link.classList.add('has-indicator'));
  if (indicator) indicator.style.transition = 'none';
  place();
  requestAnimationFrame(() => requestAnimationFrame(() => { if (indicator) indicator.style.transition = ''; }));
  // On resize the highlight jumps straight to its new place, so it never trails outside the page.
  new ResizeObserver(() => {
    if (indicator) indicator.style.transition = 'none';
    place();
    requestAnimationFrame(() => { if (indicator) indicator.style.transition = ''; });
  }).observe(nav);
  document.fonts?.ready.then(place);

  // The current model's link returns to the top instead of reloading (which would reset training).
  active.addEventListener('click', event => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
    event.preventDefault();
    window.scrollTo({ top: 0, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
  });

  const playground = links.find(link => link.dataset.model === 'playground');
  if (playground && playground !== active) {
    try {
      const experiment = sessionStorage.getItem(EXPERIMENT_KEY);
      if (experiment?.startsWith('#')) playground.href = `/${experiment}`;
    } catch {
      // Without session storage the link opens the default experiment.
    }
  }
}

export function initializeSite(): void {
  initializeTheme();
  initializeModelSwitcher();
}
