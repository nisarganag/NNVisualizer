import { STATE_CHANGE_EVENT } from './state';
import { applyTheme, currentTheme, type Theme } from './palette';

/** Presentation helpers; experiment state remains owned by the playground. */
export function icon(name: 'plus' | 'minus'): string {
  return `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

const THEME_KEY = 'nnvisualizer-theme';

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
function initializeTheme(reducedMotion: MediaQueryList): void {
  const toggle = document.querySelector<HTMLButtonElement>('#theme-toggle');
  const themeColor = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  const sync = () => {
    const dark = currentTheme() === 'dark';
    toggle.setAttribute('aria-pressed', String(dark));
    toggle.title = dark ? 'Switch to light theme' : 'Switch to dark theme';
    themeColor.content = getComputedStyle(document.documentElement).getPropertyValue('--paper').trim();
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
    if (typeof document.startViewTransition !== 'function') {
      const root = document.documentElement;
      root.classList.add('theme-fading');
      setTheme(next);
      window.setTimeout(() => root.classList.remove('theme-fading'), 400);
      return;
    }
    const bounds = toggle.getBoundingClientRect();
    const x = bounds.left + bounds.width / 2;
    const y = bounds.top + bounds.height / 2;
    const radius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
    const transition = document.startViewTransition(() => setTheme(next));
    transition.ready.then(() => {
      document.documentElement.animate(
        { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] },
        { duration: 640, easing: 'cubic-bezier(.65, 0, .35, 1)', pseudoElement: '::view-transition-new(root)' },
      );
    }).catch(() => {
      // A skipped transition has already applied the theme.
    });
  });
}

/** Moves the navigation highlight and keeps it in step with the section in view. */
function initializeNavigation(reducedMotion: MediaQueryList): void {
  const nav = document.querySelector<HTMLElement>('.main-nav');
  const buttons = Array.from(nav.querySelectorAll<HTMLButtonElement>('.nav-button'));
  let active = buttons.find(button => button.classList.contains('is-active')) ?? buttons[0];
  const place = () => {
    nav.style.setProperty('--indicator-x', `${active.offsetLeft}px`);
    nav.style.setProperty('--indicator-width', `${active.offsetWidth}px`);
    nav.style.setProperty('--indicator-opacity', '1');
  };
  const activate = (target: string) => {
    const next = buttons.find(button => button.dataset.scroll === target);
    if (!next || next === active) return;
    active = next;
    buttons.forEach(button => {
      const current = button === active;
      button.classList.toggle('is-active', current);
      if (current) button.setAttribute('aria-current', 'true');
      else button.removeAttribute('aria-current');
    });
    place();
  };
  buttons.forEach(button => button.classList.add('has-indicator'));
  // Place the highlight without animating it in on load.
  const indicator = nav.querySelector<HTMLElement>('.nav-indicator');
  indicator.style.transition = 'none';
  place();
  requestAnimationFrame(() => requestAnimationFrame(() => { indicator.style.transition = ''; }));
  new ResizeObserver(place).observe(nav);
  document.fonts?.ready.then(place);

  const learn = document.getElementById('article-text');
  const classroom = document.getElementById('classroom-panel');
  let lockedUntil = 0;
  let frame = 0;
  const spy = () => {
    frame = 0;
    if (performance.now() < lockedUntil || !learn || learn.offsetParent === null) return;
    const line = window.innerHeight * .4;
    activate(classroom && classroom.getBoundingClientRect().top < line ? 'classroom-panel'
      : learn.getBoundingClientRect().top < line ? 'article-text' : 'workspace-heading');
  };
  window.addEventListener('scroll', () => { if (!frame) frame = requestAnimationFrame(spy); }, { passive: true });
  window.addEventListener('scrollend', () => { lockedUntil = 0; spy(); });
  // A reload can restore a scroll position below the playground.
  frame = requestAnimationFrame(spy);

  document.querySelectorAll<HTMLButtonElement>('[data-scroll]').forEach(button => {
    button.addEventListener('click', () => {
      const target = document.getElementById(button.dataset.scroll);
      if (!target) return;
      if (target instanceof HTMLDetailsElement) target.open = true;
      // Hold the highlight on the destination while the page scrolls past other sections.
      lockedUntil = performance.now() + 1200;
      activate(button.dataset.scroll);
      target.scrollIntoView({ behavior: reducedMotion.matches ? 'instant' : 'smooth', block: 'start' });
    });
  });
}

export function initializeStudio(): void {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  initializeTheme(reducedMotion);
  initializeNavigation(reducedMotion);

  const classroomLink = document.querySelector<HTMLAnchorElement>('.hide-controls-link');
  const synchronizeLink = () => { classroomLink.href = window.location.href; };
  window.addEventListener(STATE_CHANGE_EVENT, synchronizeLink);
  synchronizeLink();

  document.querySelectorAll<HTMLInputElement>('input[type="range"]').forEach(input => {
    const update = () => {
      const percentage = 100 * (+input.value - +input.min) / (+input.max - +input.min);
      input.style.setProperty('--range-progress', `${percentage}%`);
    };
    input.addEventListener('input', update);
    update();
  });

  // The entire dataset tile activates its existing, keyboard-accessible canvas.
  document.querySelectorAll<HTMLElement>('.dataset').forEach(tile => {
    const canvas = tile.querySelector('canvas');
    tile.addEventListener('click', event => { if (event.target !== canvas) canvas.click(); });
  });

  const shareButton = document.querySelector<HTMLButtonElement>('#share-button');
  const feedback = document.querySelector<HTMLElement>('#share-feedback');
  const dialog = document.querySelector<HTMLDialogElement>('#share-dialog');
  const urlInput = document.querySelector<HTMLInputElement>('#share-url');
  let toastTimer: ReturnType<typeof setTimeout>;
  shareButton.addEventListener('click', async () => {
    const url = window.location.href;
    try {
      await navigator.clipboard.writeText(url);
      clearTimeout(toastTimer);
      feedback.textContent = 'Experiment link copied. It shares your starting settings.';
      feedback.hidden = false;
      toastTimer = setTimeout(() => { feedback.hidden = true; }, 4000);
    } catch {
      // Clipboard restrictions should still leave a selectable, useful link.
      urlInput.value = url;
      if (!dialog.open) dialog.showModal();
      urlInput.focus();
      urlInput.select();
    }
  });
  document.querySelector('#close-share').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right ||
        event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
}
