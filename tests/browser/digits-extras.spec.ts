import { expect, test, type Page } from '@playwright/test';

const pageErrors = new WeakMap<Page, string[]>();
test.beforeEach(({ page }) => {
  const errors: string[] = [];
  pageErrors.set(page, errors);
  page.on('pageerror', error => errors.push(error.message));
});
test.afterEach(({ page }) => expect(pageErrors.get(page)).toEqual([]));

async function stroke(page: Page, points: [number, number][]) {
  const box = (await page.locator('#draw-pad').boundingBox())!;
  const at = ([x, y]: [number, number]) => [box.x + x * box.width, box.y + y * box.height] as const;
  await page.mouse.move(...at(points[0]));
  await page.mouse.down();
  for (const point of points.slice(1)) await page.mouse.move(...at(point), { steps: 3 });
  await page.mouse.up();
}

/** Shows examples until the network reads `digit`; ten clicks show each of the ten samples once. */
async function showExample(page: Page, digit: string) {
  for (let i = 0; i < 10; i++) {
    await page.getByRole('button', { name: 'Try an example' }).click();
    if ((await page.locator('#predicted-digit').textContent()) === digit) return;
  }
  throw new Error(`No example reads ${digit}`);
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

const fooled = (digit: string) => new RegExp(`^It now reads ${digit}, \\d+% sure, after \\d+ pixel changes? \\((under )?\\d+% of the image\\)\\.$`);

test.describe('digits extras', () => {
  test('shows which pixels mattered, fools the network and undoes it', async ({ page }) => {
    await page.goto('/cnn/');
    await expect(page.locator('#cnn-workbench')).toHaveAttribute('aria-busy', 'false');
    await expect(page.getByRole('button', { name: 'Fool it' })).toBeDisabled();
    expect(await paintedPixels(page, '#saliency-view')).toBe(0);

    await showExample(page, '3');
    await expect.poll(() => paintedPixels(page, '#saliency-view')).toBeGreaterThan(40);
    const confidence = await page.locator('#predicted-confidence').textContent();

    // Any digit but the current answer can be the target.
    const target = page.getByLabel('Make it read');
    await expect(target.locator('option[value="3"]')).toBeDisabled();
    await expect(target).not.toHaveValue('3');
    await target.selectOption('8');
    await page.getByRole('button', { name: 'Fool it' }).click();
    await expect(page.locator('#predicted-digit')).toHaveText('8', { timeout: 10_000 });
    await expect(page.locator('#fool-status')).toHaveText(fooled('8'));
    await expect(page.locator('#digit-bars .is-best .digit-label')).toHaveText('8');
    await expect(page.locator('#cnn-workbench')).toHaveClass(/is-fooled/);

    await page.getByRole('button', { name: 'Undo' }).click();
    await expect(page.locator('#predicted-digit')).toHaveText('3');
    await expect(page.locator('#predicted-confidence')).toHaveText(confidence!);
    await expect(page.getByRole('button', { name: 'Undo' })).toBeHidden();
    await expect(page.locator('#fool-status')).toBeEmpty();
    await expect(page.locator('#cnn-workbench')).not.toHaveClass(/is-fooled/);

    // Drawing again leaves the attack and goes back to reading the drawing.
    await page.getByRole('button', { name: 'Fool it' }).click();
    await expect(page.locator('#predicted-digit')).toHaveText('8', { timeout: 10_000 });
    await stroke(page, [[.25, .9], [.75, .9]]);
    await expect(page.getByRole('button', { name: 'Undo' })).toBeHidden();
    await expect(page.locator('#fool-status')).toBeEmpty();
    await expect(page.locator('#cnn-workbench')).not.toHaveClass(/is-fooled/);
    await expect(page.locator('#predicted-digit')).toHaveText(/^\d$/);
  });

  test('works from the keyboard and skips the animation when motion is reduced', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/cnn/');
    await expect(page.locator('#cnn-workbench')).toHaveAttribute('aria-busy', 'false');
    await showExample(page, '7');
    await page.getByLabel('Make it read').selectOption('9');
    await page.getByLabel('Make it read').focus();
    await page.keyboard.press('Tab');
    await expect(page.getByRole('button', { name: 'Fool it' })).toBeFocused();
    await page.keyboard.press('Enter');
    // The final state appears at once, with no steps in between.
    await expect(page.locator('#fool-status')).toHaveText(fooled('9'));
    await expect(page.locator('#predicted-digit')).toHaveText('9');

    await page.keyboard.press('Tab');
    await expect(page.getByRole('button', { name: 'Undo' })).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page.locator('#predicted-digit')).toHaveText('7');
    await expect(page.getByRole('button', { name: 'Fool it' })).toBeFocused();
  });

  test.describe('on a narrow phone', () => {
    test.use({ viewport: { width: 320, height: 720 } });

    test('fits the screen while fooled, and Clear leaves the attack', async ({ page }) => {
      await page.goto('/cnn/');
      await expect(page.locator('#cnn-workbench')).toHaveAttribute('aria-busy', 'false');
      await page.getByRole('button', { name: 'Try an example' }).click();
      await page.getByRole('button', { name: 'Fool it' }).click();
      await expect(page.getByRole('button', { name: 'Undo' })).toBeVisible();
      // A random example and the default target: success or an honest failure, but finished.
      await expect(page.locator('#fool-status')).toHaveText(/^(It now reads|\d+ pixel changes weren’t enough|No pixel left)/, { timeout: 10_000 });
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);

      await page.getByRole('button', { name: 'Clear' }).click();
      await expect(page.locator('#predicted-digit')).toHaveText('?');
      await expect(page.getByRole('button', { name: 'Undo' })).toBeHidden();
      await expect(page.getByRole('button', { name: 'Fool it' })).toBeDisabled();
      expect(await paintedPixels(page, '#saliency-view')).toBe(0);
    });
  });
});
