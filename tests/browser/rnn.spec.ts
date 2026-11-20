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

  test('compares the LSTM with the gateless network on the same text', async ({ page }) => {
    await openPage(page);
    const panel = page.locator('#gates-panel');
    await panel.scrollIntoViewIfNeeded();
    await expect(panel).toHaveAttribute('aria-busy', 'false');
    await expect(page.locator('#gates-status')).toBeHidden();
    await expect(page.getByRole('heading', { name: 'Why gates matter' })).toBeVisible();

    // Both networks read the text, one column per character, tinted on the same scale.
    const text = await page.locator('#prompt-input').inputValue();
    const lstmCells = page.locator('#gates-lstm-chars .char-cell');
    const rnnCells = page.locator('#gates-rnn-chars .char-cell');
    await expect(lstmCells).toHaveCount(text.length);
    await expect(rnnCells).toHaveCount(text.length);
    const [lstmBox, rnnBox] = await Promise.all([lstmCells.nth(30).boundingBox(), rnnCells.nth(30).boundingBox()]);
    expect(Math.abs(lstmBox!.x - rnnBox!.x)).toBeLessThan(1);
    expect(Math.abs(lstmBox!.width - rnnBox!.width)).toBeLessThan(1);
    await expect(rnnCells.nth(1)).toHaveAttribute('data-surprise', /^[0-4]$/);
    await expect(rnnCells.nth(1)).toHaveAttribute('aria-label', /^\S+, \d+\.\d bits of surprise without gates, \d+\.\d bits with$/);

    // How often each is right on this text; the LSTM's figures are the scorecard's.
    for (const id of ['score-lstm-first', 'score-lstm-three', 'score-rnn-first', 'score-rnn-three']) {
      await expect(page.locator(`#${id}`)).toHaveText(/^\d+%$/);
    }
    await expect(page.locator('#score-lstm-first')).toHaveText((await page.locator('#score-first').textContent())!);
    await expect(page.locator('#score-lstm-three')).toHaveText((await page.locator('#score-three').textContent())!);
    await expect(page.locator('#scores-held-out')).toContainText('On text neither trained on');

    // Hovering a step shows both networks' memory spans for it, and moves the memory panel there too.
    const before = { lstm: await fingerprint(page, 'gates-lstm-span'), rnn: await fingerprint(page, 'gates-rnn-span') };
    await rnnCells.nth(60).hover();
    for (const cells of [rnnCells, lstmCells, page.locator('#char-cells .char-cell')]) {
      await expect(cells.nth(60)).toHaveAttribute('aria-selected', 'true');
    }
    await expect.poll(() => fingerprint(page, 'gates-lstm-span')).not.toBe(before.lstm);
    await expect.poll(() => fingerprint(page, 'gates-rnn-span')).not.toBe(before.rnn);
    const reading = /^(guesses .+: half of what shaped it came from the last (character|\d+ characters), 90% from the last (character|\d+ characters)|is .+ sure of .+, so no single character moves it much)$/;
    await expect(page.locator('#gates-lstm-note')).toHaveText(reading);
    await expect(page.locator('#gates-rnn-note')).toHaveText(reading);
    await expect(page.locator('#gates-rnn-span')).toHaveAttribute('aria-label', /^The gateless network (guesses|is) /);
    // Arrow keys move along the gateless network's row.
    await rnnCells.nth(60).focus();
    await page.keyboard.press('ArrowRight');
    await expect(rnnCells.nth(61)).toHaveAttribute('aria-selected', 'true');
    await expect(rnnCells.nth(61)).toBeFocused();

    // The fade chart draws both networks' held-out curves; a text this short adds none of its own.
    for (const network of ['lstm', 'rnn']) {
      await expect(page.locator(`#fade-chart .is-reference.is-${network}`)).toHaveAttribute('d', /^M[\d.]+,[\d.]+L/);
    }
    await expect(page.locator('#fade-text-hint')).toBeVisible();
    await expect(page.locator('#fade-text-key')).toBeHidden();
    await expect(page.locator('#fade-summary')).toContainText(/Further back they part\. .+ so 100 characters back it is [\d,]+ times larger/);
    // What each network's quotation cell does, from the metadata and on this text.
    await expect(page.locator('#quote-evidence')).toBeVisible();
    await expect(page.locator('#quote-lstm-r')).toHaveText('−0.74');
    await expect(page.locator('#quote-summary')).toContainText('reaches only r = ');
    await expect(page.locator('#quote-note')).toHaveText(/^on this text, r = −?\d\.\d\d for the LSTM’s cell 188 and −?\d\.\d\d for the gateless network’s unit \d+, each drawn so that up means inside the shaded quotes$/);

    // Each network continues the text as Generate would, leaving the text as it is.
    await page.getByRole('button', { name: 'Write with both' }).click();
    for (const network of ['lstm', 'rnn']) {
      const output = page.locator(`#continue-${network}`);
      await expect(output.locator('.known')).toHaveText(/^….*Oz\."$/);
      const known = (await output.locator('.known').textContent())!;
      expect((await output.textContent())!.length).toBe(known.length + 120);
    }
    expect(await page.locator('#prompt-input').inputValue()).toBe(text);

    // Typing reaches the gateless network too, and clears the continuations.
    await page.locator('#prompt-input').fill('"Hello," said the Lion.');
    await expect(rnnCells).toHaveCount(23);
    await expect(lstmCells).toHaveCount(23);
    await expect(page.locator('#continue-results')).toBeHidden();
    await expect(page.locator('#gates-rnn-chars [aria-selected="true"]')).toHaveAttribute('data-index', '22');

    // A longer text gets its own fade curves, traced a guess at a time while the section is on screen.
    await page.locator('#prompt-input').fill(`${text} The Mole had been working very hard all the morning, spring-cleaning his little home.`);
    await panel.scrollIntoViewIfNeeded();
    for (const network of ['lstm', 'rnn']) {
      await expect(page.locator(`#fade-chart .is-text.is-${network}`)).toHaveAttribute('d', /^M[\d.]+,[\d.]+L/, { timeout: 20_000 });
    }
    await expect(page.locator('#fade-text-key')).toBeVisible();
    await expect(page.locator('#fade-text-hint')).toBeHidden();

    // A theme switch repaints the canvases; the chart's lines take the theme's colors. (The spans are only
    // traced while the section is on screen.)
    await panel.scrollIntoViewIfNeeded();
    await expect(page.locator('#gates-rnn-note')).toHaveText(reading);
    const stroke = () => page.locator('#fade-chart .is-reference.is-rnn').evaluate(element => getComputedStyle(element).stroke);
    const light = {
      lstm: await fingerprint(page, 'gates-lstm-span'), rnn: await fingerprint(page, 'gates-rnn-span'),
      quote: await fingerprint(page, 'quote-chart'), stroke: await stroke(),
    };
    await page.locator('#theme-toggle').click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    await expect.poll(() => fingerprint(page, 'gates-lstm-span')).not.toBe(light.lstm);
    await expect.poll(() => fingerprint(page, 'gates-rnn-span')).not.toBe(light.rnn);
    await expect.poll(() => fingerprint(page, 'quote-chart')).not.toBe(light.quote);
    await expect.poll(stroke).not.toBe(light.stroke);
  });

  test('loads the gateless network only near its section, and fits a 320 px phone', async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 320, height: 700 }, hasTouch: true, isMobile: true });
    const page = await context.newPage();
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    const requested: string[] = [];
    page.on('request', request => requested.push(new URL(request.url()).pathname));
    await openPage(page);
    await page.waitForTimeout(500);
    expect(requested).not.toContain('/models/char-rnn-v1.bin');
    const panel = page.locator('#gates-panel');
    await panel.scrollIntoViewIfNeeded();
    await expect(panel).toHaveAttribute('aria-busy', 'false');
    expect(requested).toContain('/models/char-rnn-v1.bin');
    const cells = page.locator('#gates-rnn-chars .char-cell');
    await cells.nth(5).tap();
    await expect(cells.nth(5)).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#gates-rnn-note')).not.toBeEmpty();
    await expect(page.locator('#fade-chart .is-reference.is-lstm')).toHaveAttribute('d', /^M/);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
    expect(errors).toEqual([]);
    await context.close();
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
