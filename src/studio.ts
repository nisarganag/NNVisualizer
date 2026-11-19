import { STATE_CHANGE_EVENT } from './state';

/** Presentation helpers; experiment state remains owned by the playground. */
export function icon(name: 'plus' | 'minus'): string {
  return `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

export function initializeStudio(): void {
  const classroomLink = document.querySelector<HTMLAnchorElement>('.hide-controls-link');
  const synchronizeLink = () => { classroomLink.href = window.location.href; };
  window.addEventListener(STATE_CHANGE_EVENT, synchronizeLink);
  synchronizeLink();

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  document.querySelectorAll<HTMLButtonElement>('[data-scroll]').forEach(button => {
    button.addEventListener('click', () => {
      const target = document.getElementById(button.dataset.scroll);
      if (!target) return;
      if (target instanceof HTMLDetailsElement) target.open = true;
      target.scrollIntoView({ behavior: reducedMotion.matches ? 'instant' : 'smooth', block: 'start' });
      document.querySelectorAll<HTMLButtonElement>('.nav-button').forEach(nav => {
        const current = nav.dataset.scroll === button.dataset.scroll;
        nav.classList.toggle('is-active', current);
        if (current) nav.setAttribute('aria-current', 'true');
        else nav.removeAttribute('aria-current');
      });
    });
  });

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
