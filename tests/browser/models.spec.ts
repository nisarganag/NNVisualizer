import { expect, test, type Page } from '@playwright/test';

const pageErrors = new WeakMap<Page, string[]>();
test.beforeEach(({ page }) => {
  const errors: string[] = [];
  pageErrors.set(page, errors);
  page.on('pageerror', error => errors.push(error.message));
});
test.afterEach(({ page }) => expect(pageErrors.get(page)).toEqual([]));

async function setRange(page: Page, id: string, value: string) {
  await page.locator('#' + id).evaluate((element: HTMLInputElement, v) => {
    element.value = v;
    element.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
}

async function stroke(page: Page, points: [number, number][]) {
  const box = (await page.locator('#draw-pad').boundingBox())!;
  const at = ([x, y]: [number, number]) => [box.x + x * box.width, box.y + y * box.height] as const;
  await page.mouse.move(...at(points[0]));
  await page.mouse.down();
  for (const point of points.slice(1)) await page.mouse.move(...at(point), { steps: 3 });
  await page.mouse.up();
}

test.describe('digits', () => {
  test('a drawn digit is classified live, with every layer shown', async ({ page }) => {
    await page.goto('/cnn/');
    await expect(page.locator('#cnn-workbench')).toHaveAttribute('aria-busy', 'false');
    await expect(page.locator('#predicted-digit')).toHaveText('?');
    await expect(page.locator('#conv1-maps .map-tile')).toHaveCount(8);
    await expect(page.locator('#pool1-maps .map-tile')).toHaveCount(8);
    await expect(page.locator('#conv2-maps .map-tile')).toHaveCount(16);
    await expect(page.locator('#pool2-maps .map-tile')).toHaveCount(16);

    await stroke(page, Array.from({ length: 12 }, (_, i) => [.52 - i * .006, .16 + i * .06] as [number, number]));
    await expect(page.locator('#predicted-digit')).toHaveText('1');
    await expect(page.locator('#digit-bars .is-best .digit-label')).toHaveText('1');

    await page.locator('#conv2-maps .map-tile').nth(3).hover();
    await expect(page.locator('#inspector-title')).toHaveText('Convolution 2, map 4');
    await expect(page.locator('#inspector-kernels .kernel')).toHaveCount(8);
    await page.locator('#conv1-maps .map-tile').nth(0).focus();
    await expect(page.locator('#inspector-kernels .kernel')).toHaveCount(1);

    await page.getByRole('button', { name: 'Clear' }).click();
    await expect(page.locator('#predicted-digit')).toHaveText('?');

    const ring = Array.from({ length: 33 }, (_, i) => {
      const angle = (i / 32) * 2 * Math.PI;
      return [.5 + .22 * Math.sin(angle), .5 - .3 * Math.cos(angle)] as [number, number];
    });
    await stroke(page, ring);
    await expect(page.locator('#predicted-digit')).toHaveText('0');

    await page.getByRole('button', { name: 'Clear' }).click();
    await page.getByRole('button', { name: 'Try an example' }).click();
    await expect(page.locator('#predicted-digit')).toHaveText(/^\d$/);
    await expect(page.locator('#predicted-confidence')).toHaveText(/% sure$/);
  });

  test('works with touch on a phone', async ({ browser }) => {
    const context = await browser.newContext({ hasTouch: true, isMobile: true, viewport: { width: 390, height: 844 } });
    const page = await context.newPage();
    await page.goto('/cnn/');
    await expect(page.locator('#cnn-workbench')).toHaveAttribute('aria-busy', 'false');
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
    await page.getByRole('button', { name: 'Try an example' }).tap();
    await expect(page.locator('#predicted-digit')).toHaveText(/^\d$/);
    await context.close();
  });
});

test.describe('transformer', () => {
  test('GPT-2 loads, predicts like the original model, explains a head and generates', async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto('/transformer/');
    await expect(page.locator('#token-list .token')).toHaveCount(9);
    await page.getByRole('button', { name: 'Load the model' }).click();
    await expect(page.locator('.candidate').first()).toBeVisible({ timeout: 90_000 });

    await page.fill('#prompt-input', 'The capital of France is');
    await expect(page.locator('#token-list .token')).toHaveCount(5);
    // The same five candidates, in the same order, as the original float32 model.
    await expect.poll(() => page.locator('.candidate-text').evaluateAll(items => items.slice(0, 5).map(item => item.textContent)))
      .toEqual(['the', 'now', 'a', 'Paris', 'France']);

    await page.getByRole('tab', { name: 'Block 5', exact: true }).click();
    await page.getByRole('tab', { name: 'Head 3', exact: true }).click();
    await expect(page.locator('#head-title')).toHaveText('Block 5, head 3');
    await expect(page.locator('#attention-readout')).toContainText('of its attention to');
    await page.locator('#token-list .token').nth(1).hover();
    await expect(page.locator('#attention-readout')).toContainText('“capital”');

    // Temperature reshapes the probabilities without rerunning the model.
    const before = await page.locator('.candidate-value').first().textContent();
    await setRange(page, 'temperature', '0.3');
    await expect(page.locator('#temperature-value')).toHaveText('0.3');
    await expect(page.locator('.candidate-value').first()).not.toHaveText(before!);
    await page.getByRole('radio', { name: 'Top-p' }).click();
    await expect(page.locator('#sampling-name')).toHaveText('Keep probability p');

    await page.getByRole('button', { name: 'Generate', exact: true }).click();
    await expect(page.locator('#token-list .token')).toHaveCount(6);
    await expect(page.locator('#prompt-input')).toHaveValue(/^The capital of France is./);

    // The model is cached now, so a reload opens it without asking.
    await page.reload();
    await expect(page.locator('#load-model')).toBeHidden();
    await expect(page.locator('.candidate').first()).toBeVisible({ timeout: 60_000 });
  });
});

test('the header moves between models and brings back the same experiment', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#main-part')).toHaveAttribute('aria-busy', 'false');
  await page.selectOption('#learningRate', '0.1');
  const experiment = new URL(page.url()).hash;
  await page.getByRole('link', { name: 'Digits', exact: true }).click();
  await expect(page).toHaveURL(/\/cnn\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Draw a digit. Watch a CNN read it.');
  await expect(page.getByRole('link', { name: 'Digits', exact: true })).toHaveAttribute('aria-current', 'page');
  await page.getByRole('link', { name: 'Transformer', exact: true }).click();
  await expect(page).toHaveURL(/\/transformer\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Watch GPT-2 choose its next word.');
  await page.getByRole('link', { name: 'Playground', exact: true }).click();
  await expect(page.locator('#main-part')).toHaveAttribute('aria-busy', 'false');
  expect(new URL(page.url()).hash).toBe(experiment);
  await expect(page.locator('#learningRate')).toHaveValue('0.1');
});
