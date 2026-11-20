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

/** The canvas's size and a hash of its pixels, to tell when it was repainted. */
function fingerprint(page: Page, id: string) {
  return page.locator('#' + id).evaluate((canvas: HTMLCanvasElement) => {
    const { data } = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
    let hash = 0;
    for (let i = 0; i < data.length; i += 4) hash = (hash * 31 + data[i] * 3 + data[i + 1] * 5 + data[i + 2] * 7 + data[i + 3]) % 1_000_000_007;
    return `${canvas.width}x${canvas.height}:${hash}`;
  });
}

async function openPage(page: Page) {
  await page.goto('/rnn/');
  await expect(page.locator('#rnn-workbench')).toHaveAttribute('aria-busy', 'false');
}

test.describe('recurrent', () => {
  test('reads typed text one character at a time and shows its memory', async ({ page }) => {
    await openPage(page);
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('Watch a network remember.');
    await expect(page.getByRole('link', { name: 'Recurrent', exact: true })).toHaveAttribute('aria-current', 'page');
    const cells = page.locator('#char-cells .char-cell');
    const prompt = page.locator('#prompt-input');
    await expect(cells).toHaveCount((await prompt.inputValue()).length);
    // The maps have one row per memory cell.
    const cellCount = Number(await page.locator('#cell-map').getAttribute('aria-valuemax'));
    expect(cellCount).toBeGreaterThanOrEqual(128);
    await expect(page.locator('#cell-count')).toHaveText(String(cellCount));

    // Typing updates the character cells and the memory maps, one column per character.
    await prompt.fill('The Lion');
    await expect(cells).toHaveCount(8);
    await expect(page.locator('#step-count')).toHaveText('8');
    await expect.poll(() => fingerprint(page, 'cell-map')).toMatch(new RegExp(`^8x${cellCount}:`));
    const detailBefore = await fingerprint(page, 'detail-map');
    await prompt.press('End');
    await page.keyboard.type(' said, "Wh');
    await expect(cells).toHaveCount(18);
    await expect.poll(() => fingerprint(page, 'cell-map')).toMatch(new RegExp(`^18x${cellCount}:`));
    expect(await fingerprint(page, 'detail-map')).not.toBe(detailBefore);
    // Each character is tinted by how surprising it was; nothing predicted the first one.
    await expect(cells.first()).not.toHaveAttribute('data-surprise');
    await expect(cells.nth(1)).toHaveAttribute('data-surprise', /^[0-4]$/);
    await expect(cells.nth(9)).toHaveAttribute('title', /^“s”: \d+\.\d bits of surprise/);

    // The end of the text is focused until a step is chosen.
    await expect(page.locator('#candidate-list .candidate')).toHaveCount(10);
    await expect(page.locator('#next-context q')).toHaveText('…Lion said, "Wh');
    await expect(cells.last()).toHaveAttribute('aria-selected', 'true');
    // Hovering a step moves the memory span and the next-character list there.
    const spanAtEnd = await fingerprint(page, 'span-bars');
    await cells.nth(3).hover();
    await expect(cells.nth(3)).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#next-context q')).toHaveText('The ');
    await expect(page.locator('#span-summary')).toContainText('after “The ”');
    await expect(page.locator('#next-actual')).toContainText('The text goes on with “L”');
    await expect.poll(() => fingerprint(page, 'span-bars')).not.toBe(spanAtEnd);
    // Each likely letter shows the word it would lead to; Generate picks only from the top five.
    await expect(page.locator('#candidate-list .candidate').first().locator('.candidate-word')).toHaveText(/\w/);
    await expect(page.locator('#candidate-list .candidate.is-filtered')).toHaveCount(5);
    // Inside a word, the part already written leads each preview.
    await cells.nth(5).hover();
    await expect(page.locator('#next-context q')).toHaveText('The Li');
    await expect(page.locator('#candidate-list .candidate-word .known').first()).toHaveText('Li');
    // How often it was right on this text.
    await expect(page.locator('#score-first')).toHaveText(/^\d+%$/);
    await expect(page.locator('#score-three')).toHaveText(/^\d+%$/);
    await cells.nth(12).hover();
    await expect(page.locator('#next-context q')).toHaveText('The Lion said');
    await expect(page.locator('#span-summary')).toContainText(/Half came from the last \d+ characters?, 90% from the last \d+|sure, so no single character moves it much/);
    // Arrow keys move along the text.
    await cells.nth(12).focus();
    await page.keyboard.press('ArrowLeft');
    await expect(cells.nth(11)).toHaveAttribute('aria-selected', 'true');
    await expect(cells.nth(11)).toBeFocused();
    await page.keyboard.press('End');
    await expect(cells.last()).toBeFocused();

    // Hovering a row of a map follows that memory cell in the chart below; so do arrow keys.
    const map = page.locator('#cell-map');
    await map.scrollIntoViewIfNeeded();
    const box = (await map.boundingBox())!;
    const chart = await fingerprint(page, 'unit-chart');
    await page.mouse.move(box.x + box.width / 2, box.y + (40.5 / cellCount) * box.height);
    await expect(page.locator('#unit-title')).toHaveText('Cell 41');
    await expect(page.locator('#map-readout')).toContainText('Cell 41 after');
    await expect.poll(() => fingerprint(page, 'unit-chart')).not.toBe(chart);
    await map.focus();
    await page.keyboard.press('ArrowDown');
    await expect(page.locator('#unit-title')).toHaveText('Cell 42');
    await expect(map).toHaveAttribute('aria-valuenow', '42');

    // The second map switches between the hidden state and the gates.
    const forget = await fingerprint(page, 'detail-map');
    await page.getByRole('radio', { name: /^Input/ }).click();
    await expect(page.getByRole('radio', { name: /^Input/ })).toHaveAttribute('aria-checked', 'true');
    await expect.poll(() => fingerprint(page, 'detail-map')).not.toBe(forget);

    // Temperature reshapes the candidates without rerunning the network.
    const top = await page.locator('.candidate-value').first().textContent();
    await setRange(page, 'temperature', '0.3');
    await expect(page.locator('#temperature-value')).toHaveText('0.3');
    await expect(page.locator('.candidate-value').first()).not.toHaveText(top!);

    // Generate appends characters one at a time until stopped.
    await page.getByRole('button', { name: 'Generate' }).click();
    await expect(page.getByRole('button', { name: 'Stop' })).toHaveAttribute('aria-pressed', 'true');
    await expect.poll(() => cells.count()).toBeGreaterThan(24);
    await page.getByRole('button', { name: 'Stop' }).click();
    await expect(page.getByRole('button', { name: 'Generate' })).toBeEnabled();
    const text = await prompt.inputValue();
    expect(text.startsWith('The Lion said, "Wh')).toBe(true);
    await expect(cells).toHaveCount(text.length);
    await page.waitForTimeout(300);
    await expect(cells).toHaveCount(text.length);
    await expect(cells.last()).toHaveAttribute('aria-selected', 'true');

    // A theme switch repaints every canvas in the other palette.
    const light = { map: await fingerprint(page, 'cell-map'), chart: await fingerprint(page, 'unit-chart'), span: await fingerprint(page, 'span-bars') };
    await page.locator('#theme-toggle').click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    await expect.poll(() => fingerprint(page, 'cell-map')).not.toBe(light.map);
    await expect.poll(() => fingerprint(page, 'unit-chart')).not.toBe(light.chart);
    await expect.poll(() => fingerprint(page, 'span-bars')).not.toBe(light.span);
  });

  test('simplifies typography, reports unknown characters and starts again from examples', async ({ page }) => {
    await openPage(page);
    const cells = page.locator('#char-cells .char-cell');
    await page.locator('#prompt-input').fill('“Café”—5 €');
    await expect(cells).toHaveCount(9);
    expect(await cells.allTextContents()).toEqual(['"', 'C', 'a', 'f', 'e', '"', '-', '-', '']);
    await expect(page.locator('#skipped-note')).toHaveText('Left out, since the network never learned them: “5” “€”');
    await page.selectOption('#example-select', { index: 4 });
    await expect(page.locator('#prompt-input')).toHaveValue(/^The Mole had been working very hard all the morning/);
    await expect(page.locator('#skipped-note')).toBeHidden();
    await expect(cells).toHaveCount((await page.locator('#prompt-input').inputValue()).length);
  });

  test('with reduced motion, Generate adds its characters at once', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await openPage(page);
    const cells = page.locator('#char-cells .char-cell');
    await page.locator('#prompt-input').fill('Peter');
    await expect(cells).toHaveCount(5);
    await page.getByRole('button', { name: 'Generate' }).click();
    await expect(cells).toHaveCount(125);
    await expect(page.getByRole('button', { name: 'Generate' })).toHaveAttribute('aria-pressed', 'false');

    // On a phone the text box fills up, and it keeps the newest characters in view.
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: 'Generate' }).click();
    await expect(cells).toHaveCount(245);
    const box = page.locator('#prompt-input');
    expect(await box.evaluate((element: HTMLTextAreaElement) => element.scrollHeight > element.clientHeight)).toBe(true);
    expect(await box.evaluate((element: HTMLTextAreaElement) => element.scrollTop + element.clientHeight >= element.scrollHeight - 2)).toBe(true);
  });

  test('fits a 320 px phone without sideways scrolling', async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 320, height: 700 }, hasTouch: true, isMobile: true });
    const page = await context.newPage();
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await openPage(page);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
    const cells = page.locator('#char-cells .char-cell');
    await cells.last().tap();
    await expect(cells.last()).toHaveAttribute('aria-selected', 'true');
    await page.locator('#prompt-input').fill('Alice said');
    await expect(cells).toHaveCount(10);
    await cells.nth(2).tap();
    await expect(cells.nth(2)).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#next-context q')).toHaveText('Ali');
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
    expect(errors).toEqual([]);
    await context.close();
  });
});
