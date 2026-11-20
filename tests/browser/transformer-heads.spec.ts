import { expect, test, type Page } from '@playwright/test';

const pageErrors = new WeakMap<Page, string[]>();
test.beforeEach(({ page }) => {
  const errors: string[] = [];
  pageErrors.set(page, errors);
  page.on('pageerror', error => errors.push(error.message));
});
test.afterEach(({ page }) => expect(pageErrors.get(page)).toEqual([]));

/** Every listed next token and the value beside it, in order. */
const candidates = (page: Page) => page.locator('.candidate').evaluateAll(items => items.map(item => ({
  text: item.querySelector('.candidate-text').textContent,
  value: item.querySelector('.candidate-value').textContent,
})));

/** Waits for the run to finish and for the lens to read it. */
async function settled(page: Page) {
  await expect(page.locator('#transformer-flow')).toHaveAttribute('aria-busy', 'false', { timeout: 90_000 });
  await expect(page.locator('#lens')).not.toHaveClass(/is-stale/, { timeout: 90_000 });
  await expect(page.locator('#lens-list .lens-row')).toHaveCount(13);
}

test.describe('switching heads off', () => {
  test('a switched-off head changes the prediction, shown before and after, until every head is back on', async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto('/transformer/');
    await page.getByRole('button', { name: 'Load the model' }).click();
    await expect(page.locator('.candidate').first()).toBeVisible({ timeout: 90_000 });
    await page.selectOption('#example-select', 'When Mary and John went to the store, John gave a drink to');
    await expect(page.locator('#token-list .token')).toHaveCount(14);
    await settled(page);
    const original = await candidates(page);
    expect(original.map(candidate => candidate.text).slice(0, 2)).toEqual(['Mary', 'them']);
    await expect(page.locator('#ablation')).toBeHidden();
    await expect(page.locator('#ablation-summary')).toBeEmpty();
    await expect(page.locator('#comparison-key')).toBeHidden();

    // Block 12, head 11 looks at “Mary”, yet without it “Mary” is far likelier.
    await page.getByRole('tab', { name: 'Block 12', exact: true }).click();
    await page.getByRole('tab', { name: 'Head 11', exact: true }).click();
    await expect(page.locator('#head-title')).toHaveText('Block 12, head 11');
    await expect(page.locator('#attention-readout')).toHaveText('“to” gives 52.8% of its attention to “Mary”.');
    const toggle = page.locator('#head-toggle');
    await expect(toggle).toHaveText('Switch off this head');
    await toggle.click();
    await expect(toggle).toHaveText('Switch it back on');
    await expect(page.locator('#head-state')).toHaveText('Switched off');
    await settled(page);

    // Before and after, for every candidate, and the biggest change in one sentence.
    await expect(page.locator('#ablation-summary')).toHaveText('With 1 head off, “Mary” rose from 45.3% to 73.3% and stayed 1st.');
    await expect(page.locator('#comparison-key')).toBeVisible();
    await expect(page.locator('#candidate-list')).toHaveClass(/is-compared/);
    await expect(page.locator('.candidate').first().locator('.candidate-text')).toHaveText('Mary');
    await expect(page.locator('.candidate').first().locator('.candidate-value')).toHaveText('45.3% → 73.3%');
    await expect(page.locator('.candidate').first()).toHaveAttribute('title', /^45\.3% with every head on, 73\.3% with 1 head off, before temperature and sampling\./);
    for (const { value } of await candidates(page)) expect(value).toMatch(/^(<0\.1|\d+\.\d)% → (<0\.1|\d+\.\d)%$/);
    await expect(page.locator('.candidate').first().locator('.candidate-bar b')).toHaveClass('is-gain');
    await expect(page.locator('.candidate').nth(1).locator('.candidate-bar b')).toHaveClass('is-loss');

    // The head still attends; it just passes nothing on.
    await expect(page.locator('#attention-readout')).toHaveText('“to” gives 52.8% of its attention to “Mary”. The head is switched off, so none of it is passed on.');
    await expect(page.locator('#head-output-note')).toHaveText('switched off, all zeros');

    // Marked in the heads grid and the block overview with a slash and a count, and named for screen readers.
    await expect(page.getByRole('tab', { name: 'Head 11, switched off', exact: true })).toHaveClass(/is-off/);
    await expect(page.getByRole('tab', { name: 'Head 10', exact: true })).not.toHaveClass(/is-off/);
    const block12 = page.getByRole('tab', { name: 'Block 12, 1 head switched off', exact: true });
    await expect(block12).toHaveClass(/has-off/);
    await expect(block12.locator('.off-count')).toHaveText('1');
    await expect(page.locator('#ablation')).toBeVisible();
    await expect(page.locator('#ablation .ablation-state')).toHaveText('1 head switched off in block 12');

    // The lens and the surprise strip read the run with the head off.
    await expect(page.locator('#lens-context')).toHaveText('What could follow “to”, block by block, before temperature and sampling, with 1 head off');
    const lastStage = page.locator('#lens-list .lens-row').last();
    await expect(lastStage.locator('.lens-token')).toHaveText('Mary');
    await expect(lastStage.locator('.lens-value')).toHaveText('73.3%');
    await expect(page.locator('#surprise-text span').last()).toHaveAttribute('title', /^\d+\.\d bits of surprise, rank [\d,]+; \d+\.\d bits with every head on$/);

    // A whole block at once, by keyboard.
    await page.locator('#block-toggle').focus();
    await page.keyboard.press('Enter');
    await expect(page.locator('#block-toggle')).toHaveText('Switch block 12 back on');
    await expect(page.locator('#ablation .ablation-state')).toHaveText('12 heads switched off in block 12');
    await expect(page.getByRole('tab', { name: 'Block 12, every head switched off', exact: true })).toHaveClass(/is-off/);
    await settled(page);
    await expect(page.locator('#ablation-summary')).toHaveText(/^With 12 heads off, “.+” (rose|fell) from /);

    // Turning every head back on restores the original prediction exactly, and keeps focus on a nearby control.
    await page.getByRole('button', { name: 'Turn all back on' }).click();
    await expect(page.locator('#ablation')).toBeHidden();
    await expect(toggle).toBeFocused();
    await expect(toggle).toHaveText('Switch off this head');
    await settled(page);
    await expect(page.locator('#ablation-summary')).toBeEmpty();
    await expect(page.locator('#comparison-key')).toBeHidden();
    await expect(page.locator('#candidate-list')).not.toHaveClass(/is-compared/);
    expect(await candidates(page)).toEqual(original);
    await expect(page.getByRole('tab', { name: 'Head 11', exact: true })).not.toHaveClass(/is-off/);
    await expect(page.getByRole('tab', { name: 'Block 12', exact: true })).not.toHaveClass(/has-off/);
    await expect(page.locator('#attention-readout')).toHaveText('“to” gives 52.8% of its attention to “Mary”.');

    // The focused toggle switches it off again; nothing overflows on a small phone.
    await page.keyboard.press('Space');
    await expect(page.locator('#ablation')).toBeVisible();
    await settled(page);
    await expect(page.locator('#ablation-summary')).toHaveText('With 1 head off, “Mary” rose from 45.3% to 73.3% and stayed 1st.');
    await page.setViewportSize({ width: 320, height: 800 });
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
  });

  test('which heads matter, one block at a time on the CPU', async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto('/transformer/');
    await page.getByRole('button', { name: 'Load the model' }).click();
    await expect(page.locator('.candidate').first()).toBeVisible({ timeout: 90_000 });
    await page.selectOption('#example-select', 'When Mary and John went to the store, John gave a drink to');
    await expect(page.locator('#token-list .token')).toHaveCount(14);
    await settled(page);
    await expect(page.locator('#importance-context')).toContainText('Testing all 144 needs the GPU');
    await expect(page.locator('#importance-map')).toBeHidden();

    // The CPU tests the open block's twelve heads.
    await page.getByRole('tab', { name: 'Block 12', exact: true }).click();
    const start = page.getByRole('button', { name: 'Test the heads of block 12' });
    await start.click();
    await expect(page.locator('#importance-progress')).toHaveText(/^12 heads tested in \d/, { timeout: 120_000 });
    await expect(start).toBeDisabled();
    await expect(page.locator('#importance-context')).toHaveText('Each cell switches off one head alone: how far “Mary” after “to” moves from 45.3%. Blocks run down, heads across; on the CPU, test a block at a time.');
    const cells = page.locator('.importance-cell');
    await expect(cells).toHaveCount(144);
    await expect(page.locator('.importance-cell.is-known')).toHaveCount(12);
    await expect(cells.nth(11 * 12 + 10)).toHaveAttribute('data-value', /^0\.73/);
    await expect(page.locator('#importance-readout')).toHaveText('Of the 12 heads tested, block 12, head 1 helps “Mary” most: switched off alone, it falls from 45.3% to 27.8%. Block 12, head 11 holds it back most: without it, “Mary” rises to 73.3%.');

    // Hovering a cell explains it; arrow keys move between cells; choosing one opens its head.
    await cells.nth(11 * 12 + 10).hover();
    await expect(page.locator('#importance-readout')).toHaveText('Block 12, head 11 switched off alone: “Mary” rises from 45.3% to 73.3%.');
    await cells.nth(11 * 12 + 9).focus();
    await page.keyboard.press('ArrowRight');
    await expect(cells.nth(11 * 12 + 10)).toBeFocused();
    await expect(cells.nth(11 * 12 + 10)).toHaveAttribute('tabindex', '0');
    await page.keyboard.press('Enter');
    await expect(page.locator('#head-title')).toHaveText('Block 12, head 11');
    await expect(cells.nth(11 * 12 + 10)).toHaveClass(/is-open/);
    // Switching it off agrees with the map.
    await page.getByRole('button', { name: 'Switch off this head' }).click();
    await settled(page);
    await expect(page.locator('#ablation-summary')).toHaveText('With 1 head off, “Mary” rose from 45.3% to 73.3% and stayed 1st.');
    await expect(cells.nth(11 * 12 + 10)).toHaveClass(/is-off/);

    // Another block adds its row; stopping halfway keeps what was tested.
    await page.getByRole('tab', { name: 'Block 1', exact: true }).click();
    await page.getByRole('button', { name: 'Test the heads of block 1' }).click();
    const stop = page.getByRole('button', { name: 'Stop' });
    await expect(stop).toBeFocused();
    await expect(page.locator('#importance-progress')).toHaveText(/^Testing [1-9]\d* of 12 heads$/, { timeout: 120_000 });
    await stop.click();
    await expect(stop).toBeHidden();
    await expect(page.getByRole('button', { name: 'Test the heads of block 1' })).toBeFocused();
    await expect(page.locator('#importance-progress')).toHaveText(/^\d+ heads tested$/);
    const known = await page.locator('.importance-cell.is-known').count();
    expect(known).toBeGreaterThan(12);
    expect(known).toBeLessThan(24);

    // A new prompt clears the map.
    await page.fill('#prompt-input', 'The capital of France is');
    await expect(page.locator('#token-list .token')).toHaveCount(5, { timeout: 90_000 });
    await settled(page);
    await expect(page.locator('#importance-map')).toBeHidden();
    await expect(page.getByRole('button', { name: 'Test the heads of block 1' })).toBeEnabled();
  });
});
