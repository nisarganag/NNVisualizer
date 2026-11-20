import { expect, test, type Page } from '@playwright/test';

const pageErrors = new WeakMap<Page, string[]>();
test.beforeEach(({ page }) => {
  const errors: string[] = [];
  pageErrors.set(page, errors);
  page.on('pageerror', error => errors.push(error.message));
});
test.afterEach(({ page }) => expect(pageErrors.get(page)).toEqual([]));

async function open(page: Page) {
  await page.goto('/vae/');
  await expect(page.locator('#vae-workbench')).toHaveAttribute('aria-busy', 'false');
  // The Digits page's network has loaded once the point's reading is a digit.
  await expect(page.locator('#looks-digit')).toHaveText(/^\d$/);
}

async function stroke(page: Page, points: [number, number][]) {
  const box = (await page.locator('#draw-pad').boundingBox())!;
  const at = ([x, y]: [number, number]) => [box.x + x * box.width, box.y + y * box.height] as const;
  await page.mouse.move(...at(points[0]));
  await page.mouse.down();
  for (const point of points.slice(1)) await page.mouse.move(...at(point), { steps: 3 });
  await page.mouse.up();
}

/** A hash of a canvas's pixels, to tell whether it was repainted differently. */
function signature(page: Page, selector: string) {
  return page.locator(selector).first().evaluate((canvas: HTMLCanvasElement) => {
    const { data } = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
    let hash = 0;
    for (let i = 0; i < data.length; i++) hash = (Math.imul(hash, 31) + data[i]) | 0;
    return `${canvas.width}×${canvas.height}:${hash}`;
  });
}

/** Counts a canvas's pixels that are not fully transparent. */
function paintedPixels(page: Page, selector: string) {
  return page.locator(selector).evaluate((canvas: HTMLCanvasElement) => {
    const { data } = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
    let count = 0;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) count++;
    return count;
  });
}

/** The point's two numbers, as shown beside the decoded digit. */
async function readPoint(page: Page): Promise<[number, number]> {
  const value = async (id: string) => Number((await page.locator(id).textContent())!.replace('−', '-'));
  return [await value('#z1'), await value('#z2')];
}

test.describe('autoencoder', () => {
  test('dragging the point redraws its digit, and the arrow keys move it in small steps', async ({ page }) => {
    await open(page);
    await expect(page.getByRole('navigation', { name: 'Models' }).getByRole('link', { name: 'Autoencoder', exact: true }))
      .toHaveAttribute('aria-current', 'page');
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('Explore the space between digits.');
    const probe = page.getByRole('slider', { name: 'Point on the map' });
    const start = await readPoint(page);
    const digit = await signature(page, '#decoded-view');

    // Drag the point by its handle across the map, away from where it starts.
    const handle = (await probe.boundingBox())!;
    const frame = (await page.locator('#map-frame').boundingBox())!;
    const target = [start[0] < 0 ? 0.8 : 0.2, start[1] < 0 ? 0.2 : 0.8];
    await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
    await page.mouse.down();
    await page.mouse.move(frame.x + frame.width * target[0], frame.y + frame.height * target[1], { steps: 12 });
    await page.mouse.up();
    const dragged = await readPoint(page);
    expect(Math.abs(dragged[0] - start[0])).toBeGreaterThan(1);
    expect(Math.abs(dragged[1] - start[1])).toBeGreaterThan(1);
    expect(await signature(page, '#decoded-view')).not.toBe(digit);
    await expect(probe).toHaveAttribute('aria-valuetext', /^z1 −?\d\.\d\d, z2 −?\d\.\d\d, looks like \d$/);

    // A click elsewhere moves the point there.
    await page.mouse.click(frame.x + frame.width * 0.5, frame.y + frame.height * 0.5);
    const clicked = await readPoint(page);
    expect(Math.abs(clicked[0])).toBeLessThan(0.2);
    expect(Math.abs(clicked[1])).toBeLessThan(0.2);

    // Readouts round to two decimals, so compare within a hundredth.
    const near = (actual: number, expected: number) => expect(Math.abs(actual - expected)).toBeLessThan(0.011);
    await probe.focus();
    await page.keyboard.press('ArrowRight');
    near((await readPoint(page))[0], clicked[0] + 0.05);
    await page.keyboard.press('Shift+ArrowUp');
    near((await readPoint(page))[1], clicked[1] + 0.5);
    await page.keyboard.press('ArrowLeft');
    await page.keyboard.press('ArrowDown');
    const [x, y] = await readPoint(page);
    near(x, clicked[0]);
    near(y, clicked[1] + 0.45);
    near(Number(await probe.getAttribute('aria-valuenow')), x);
  });

  test('a drawing or an example puts the point where it lands', async ({ page }) => {
    await open(page);
    const before = await readPoint(page);
    await stroke(page, Array.from({ length: 12 }, (_, i) => [0.52 - i * 0.006, 0.16 + i * 0.06] as [number, number]));
    await expect(page.locator('#drawing-readout')).toHaveText(/^It lands among the 1s, at −?\d\.\d\d, −?\d\.\d\d\.$/);
    // The point sits at the drawing's two numbers, which are also a morph end now.
    const readout = (await page.locator('#drawing-readout').textContent())!.replaceAll('−', '-');
    const [, x, y] = readout.match(/at (-?\d\.\d\d), (-?\d\.\d\d)\.$/)!;
    expect(await readPoint(page)).toEqual([Number(x), Number(y)]);
    expect(await readPoint(page)).not.toEqual(before);
    await expect(page.locator('#morph-to option[value="drawing"]')).toBeEnabled();
    expect(await paintedPixels(page, '#input-view')).toBeGreaterThan(20);
    expect(await paintedPixels(page, '#reconstruction-view')).toBeGreaterThan(20);

    await page.getByRole('button', { name: 'Clear' }).click();
    await expect(page.locator('#drawing-readout')).toHaveText('Draw a digit to see where it lands.');
    await expect(page.locator('#morph-to option[value="drawing"]')).toBeDisabled();
    expect(await paintedPixels(page, '#input-view')).toBe(0);

    const placed = await readPoint(page);
    await page.getByRole('button', { name: 'Try an example' }).click();
    await expect(page.locator('#drawing-readout')).toHaveText(/^It lands (among|between|past)/);
    expect(await paintedPixels(page, '#input-view')).toBeGreaterThan(50);
    expect(await readPoint(page)).not.toEqual(placed);
  });

  test('a morph slides the point from one digit to another', async ({ page }) => {
    await open(page);
    await page.getByLabel('From', { exact: true }).selectOption('1');
    await page.getByLabel('To', { exact: true }).selectOption('0');
    await page.getByRole('button', { name: 'Morph' }).click();
    await expect(page.getByRole('button', { name: 'Stop' })).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByRole('button', { name: 'Morph' })).toBeVisible({ timeout: 6000 });
    await expect(page.locator('#morph-strip button').last()).toHaveAttribute('aria-current', 'step');
    await expect(page.locator('#looks-digit')).toHaveText('0');

    // Run it again, sampling the point every frame: it glides, moving a little each frame.
    const trail = await page.evaluate(() => new Promise<number[]>(resolve => {
      const probe = document.getElementById('probe')!;
      const button = document.getElementById('morph-start')!;
      const values: number[] = [];
      button.click();
      const sample = () => {
        values.push(Number(probe.getAttribute('aria-valuenow')));
        if (button.classList.contains('is-running')) requestAnimationFrame(sample);
        else resolve(values);
      };
      requestAnimationFrame(sample);
    }));
    expect(trail.length).toBeGreaterThan(30);
    expect(new Set(trail).size).toBeGreaterThan(20);
    const steps = await page.locator('#morph-strip button').count();
    expect(steps).toBe(7);
    // Stop halts it where it is.
    await page.getByRole('button', { name: 'Morph' }).click();
    await page.getByRole('button', { name: 'Stop' }).click();
    await expect(page.getByRole('button', { name: 'Morph' })).toHaveAttribute('aria-pressed', 'false');
    const stopped = await readPoint(page);
    await page.waitForTimeout(300);
    expect(await readPoint(page)).toEqual(stopped);
  });

  test('with reduced motion, a morph steps instead of animating', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await open(page);
    await page.getByLabel('From', { exact: true }).selectOption('7');
    await page.getByLabel('To', { exact: true }).selectOption('3');
    await page.getByRole('button', { name: 'Morph' }).click();
    // The end at once, never a running animation.
    await expect(page.locator('#morph-strip button').last()).toHaveAttribute('aria-current', 'step');
    await expect(page.getByRole('button', { name: 'Morph' })).toHaveAttribute('aria-pressed', 'false');
    await expect(page.locator('#looks-digit')).toHaveText('3');
    // Each picture of the strip is a step to take by hand.
    const end = await readPoint(page);
    await page.locator('#morph-strip button').nth(3).click();
    await expect(page.locator('#morph-strip button').nth(3)).toHaveAttribute('aria-current', 'step');
    const middle = await readPoint(page);
    expect(middle).not.toEqual(end);
    await page.locator('#morph-strip button').nth(0).focus();
    await page.keyboard.press('Enter');
    await expect(page.locator('#looks-digit')).toHaveText('7');
    expect(await page.evaluate(() => document.getAnimations().filter(animation => animation.playState === 'running').length)).toBe(0);
  });

  test('the digit buttons highlight a class and go to it, and the grid view decodes the whole map', async ({ page }) => {
    await open(page);
    const map = await signature(page, '#map-canvas');
    await page.getByRole('button', { name: 'Go to the 4s' }).hover();
    await expect(page.getByRole('button', { name: 'Go to the 4s' })).toHaveClass(/is-highlighted/);
    expect(await signature(page, '#map-canvas')).not.toBe(map);
    await page.getByRole('button', { name: 'Go to the 4s' }).click();
    await expect(page.locator('#looks-digit')).toHaveText('4');

    const dots = await paintedPixels(page, '#map-canvas');
    await page.getByRole('radio', { name: 'Decoded grid' }).click();
    await expect(page.getByRole('radio', { name: 'Decoded grid' })).toHaveAttribute('aria-checked', 'true');
    expect(await paintedPixels(page, '#map-canvas')).toBeGreaterThan(dots * 2);
  });

  test('switching the theme repaints every canvas', async ({ page }) => {
    await open(page);
    await page.getByRole('button', { name: 'Try an example' }).click();
    await expect(page.locator('#drawing-readout')).toHaveText(/^It lands/);
    const canvases = ['#map-canvas', '#decoded-view', '#draw-pad', '#input-view', '#reconstruction-view', '#morph-strip canvas'];
    const before = await Promise.all(canvases.map(selector => signature(page, selector)));
    const theme = await page.evaluate(() => document.documentElement.dataset.theme);
    await page.locator('#theme-toggle').click();
    await expect(page.locator('html')).not.toHaveAttribute('data-theme', theme!);
    for (const [i, selector] of canvases.entries()) {
      await expect.poll(() => signature(page, selector), { message: selector }).not.toBe(before[i]);
    }
  });

  test.describe('on a phone', () => {
    test('fits 320 pixels and follows a finger', async ({ browser }) => {
      const context = await browser.newContext({ hasTouch: true, isMobile: true, viewport: { width: 320, height: 720 } });
      const page = await context.newPage();
      const errors: string[] = [];
      page.on('pageerror', error => errors.push(error.message));
      await open(page);
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);

      await page.locator('#map-frame').scrollIntoViewIfNeeded();
      const frame = (await page.locator('#map-frame').boundingBox())!;
      await page.touchscreen.tap(frame.x + frame.width * 0.5, frame.y + frame.height * 0.5);
      const tapped = await readPoint(page);
      expect(Math.abs(tapped[0])).toBeLessThan(0.3);
      expect(Math.abs(tapped[1])).toBeLessThan(0.3);

      // A finger dragged across the map carries the point with it, without scrolling the page.
      const scroll = await page.evaluate(() => window.scrollY);
      const touch = await context.newCDPSession(page);
      const at = (fx: number, fy: number) => [{ x: frame.x + frame.width * fx, y: frame.y + frame.height * fy }];
      await touch.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: at(0.5, 0.5) });
      for (let i = 1; i <= 10; i++) await touch.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: at(0.5 - i * 0.03, 0.5 + i * 0.03) });
      await touch.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
      const dragged = await readPoint(page);
      expect(dragged[0]).toBeLessThan(tapped[0] - 0.5);
      expect(dragged[1]).toBeLessThan(tapped[1] - 0.5);
      expect(await page.evaluate(() => window.scrollY)).toBe(scroll);

      await page.getByRole('button', { name: 'Try an example' }).tap();
      await expect(page.locator('#drawing-readout')).toHaveText(/^It lands/);
      await page.getByRole('button', { name: 'Morph' }).tap();
      await expect(page.getByRole('button', { name: 'Morph' })).toBeVisible({ timeout: 6000 });
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
      expect(errors).toEqual([]);
      await context.close();
    });
  });
});
