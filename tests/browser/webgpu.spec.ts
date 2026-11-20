import { expect, test, type Page } from '@playwright/test';

/*
 * GPT-2 on the GPU (WebGPU), checked against the CPU path on the same page.
 * Runs in its own Playwright project: full Chromium on this machine's GPU locally,
 * and the headless shell's software WebGPU (SwiftShader, slow) on CI. A CI runner
 * without any adapter skips; locally, a missing GPU fails.
 */

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

/** The five most likely next tokens and their raw scores, as the page lists them. */
const candidates = (page: Page) => page.locator('.candidate').evaluateAll(items => items.slice(0, 5).map(item => ({
  text: item.querySelector('.candidate-text').textContent,
  logit: Number((item as HTMLElement).dataset.logit),
})));

/** Bits of surprise per prompt token, from the surprise strip (null for the first token). */
const surprises = (page: Page) => page.locator('#surprise-text span').evaluateAll(spans => spans.map(span => {
  const bits = /^(\d+\.\d) bits/.exec(span.getAttribute('title') ?? '');
  return bits ? Number(bits[1]) : null;
}));

/** Waits for the run to finish and for the lens to read it. */
async function settled(page: Page) {
  await expect(page.locator('#transformer-flow')).toHaveAttribute('aria-busy', 'false', { timeout: 120_000 });
  await expect(page.locator('#lens')).not.toHaveClass(/is-stale/, { timeout: 120_000 });
  await expect(page.locator('#lens-list .lens-row')).toHaveCount(13);
}

test('GPT-2 runs on the GPU, matches the CPU, switches between them and generates', async ({ page }) => {
  test.setTimeout(900_000);
  await page.goto('/transformer/?backend=gpu');
  await page.getByRole('button', { name: 'Load the model' }).click();
  await expect(page.locator('.candidate').first()).toBeVisible({ timeout: 300_000 });

  const name = page.locator('#engine-name');
  const reason = page.locator('#engine-reason');
  if (process.env.CI && await name.textContent() !== 'Running on your GPU') {
    test.skip(true, `This CI runner offers no WebGPU adapter, so the page is ${await name.textContent()} ${await reason.textContent()}.`);
  }
  await expect(name).toHaveText('Running on your GPU');
  await expect(reason).toBeHidden();
  await expect(page.locator('#engine-time')).toHaveText(/^9 tokens in \d+(\.\d)? m?s$/);
  await expect(page.getByRole('radio', { name: 'GPU', exact: true })).toHaveAttribute('aria-checked', 'true');

  await page.fill('#prompt-input', 'The capital of France is');
  await expect(page.locator('#token-list .token')).toHaveCount(5);
  // The same five candidates, in the same order, as the original float32 model.
  await expect.poll(async () => (await candidates(page)).map(candidate => candidate.text), { timeout: 120_000 })
    .toEqual(['the', 'now', 'a', 'Paris', 'France']);
  await settled(page);
  const gpu = { candidates: await candidates(page), surprises: await surprises(page), lens: await page.locator('#lens-list .lens-token').allTextContents() };

  // The CPU recomputes the same prompt from scratch and agrees.
  await page.getByRole('radio', { name: 'CPU', exact: true }).click();
  await expect(page.getByRole('radio', { name: 'CPU', exact: true })).toHaveAttribute('aria-checked', 'true');
  await settled(page);
  await expect(name).toHaveText('Running on the CPU');
  await expect(reason).toBeHidden();
  await expect(page.locator('#engine-time')).toHaveText(/^5 tokens in /);
  const cpu = { candidates: await candidates(page), surprises: await surprises(page), lens: await page.locator('#lens-list .lens-token').allTextContents() };
  expect(cpu.candidates.map(candidate => candidate.text)).toEqual(gpu.candidates.map(candidate => candidate.text));
  cpu.candidates.forEach((candidate, i) => expect(Math.abs(candidate.logit - gpu.candidates[i].logit)).toBeLessThan(1e-3));
  expect(cpu.lens).toEqual(gpu.lens);
  expect(cpu.surprises).toHaveLength(5);
  cpu.surprises.forEach((bits, i) => (bits === null ? expect(gpu.surprises[i]).toBeNull() : expect(Math.abs(bits - gpu.surprises[i])).toBeLessThanOrEqual(0.11)));

  // Back on the GPU, greedy generation (top-k with k = 1) continues "The dog" exactly like the original model.
  await page.getByRole('radio', { name: 'GPU', exact: true }).click();
  await settled(page);
  await expect(name).toHaveText('Running on your GPU');
  await setRange(page, 'sampling-value', '1');
  await expect(page.locator('#sampling-number')).toHaveText('1');
  await page.fill('#prompt-input', 'The dog');
  await expect(page.locator('#token-list .token')).toHaveCount(2);
  await expect(page.locator('#transformer-flow')).toHaveAttribute('aria-busy', 'false');
  const generate = page.getByRole('button', { name: 'Generate', exact: true });
  for (let count = 3; count <= 20; count++) {
    await expect(generate).toBeEnabled({ timeout: 60_000 });
    await generate.click();
    await expect(page.locator('#token-list .token')).toHaveCount(count, { timeout: 60_000 });
    await expect(page.locator('#transformer-flow')).toHaveAttribute('aria-busy', 'false', { timeout: 60_000 });
  }
  await expect(page.locator('#token-list .token-id')).toHaveText(['464', '3290', '373', '1043', '287', '257', '2214', '1474', '262', '16246',
    '286', '2688', '290', '2688', '27262', '13', '198', '198', '464', '3290']);
  // The prompt box keeps the line breaks GPT-2 wrote.
  await expect(page.locator('#prompt-input')).toHaveValue('The dog was found in a field near the intersection of West and West Streets.\n\nThe dog');
  await expect(page.locator('#engine-time')).toHaveText(/^1 token in /);
  await expect(name).toHaveText('Running on your GPU');

  // A head's weights still arrive.
  await page.getByRole('tab', { name: 'Block 5', exact: true }).click();
  await page.getByRole('tab', { name: 'Head 3', exact: true }).click();
  await expect(page.locator('#query-weights')).not.toHaveClass(/is-loading/);
});

test('heads switched off on the GPU give the CPU’s prediction', async ({ page }) => {
  test.setTimeout(900_000);
  await page.goto('/transformer/?backend=gpu');
  await page.getByRole('button', { name: 'Load the model' }).click();
  await expect(page.locator('.candidate').first()).toBeVisible({ timeout: 300_000 });
  const name = page.locator('#engine-name');
  if (process.env.CI && await name.textContent() !== 'Running on your GPU') {
    test.skip(true, `This CI runner offers no WebGPU adapter, so the page is ${await name.textContent()}.`);
  }
  await expect(name).toHaveText('Running on your GPU');
  await page.selectOption('#example-select', 'When Mary and John went to the store, John gave a drink to');
  await expect(page.locator('#token-list .token')).toHaveCount(14);
  await settled(page);

  // One head early, one late, and every head of block 6.
  for (const [block, head] of [[12, 11], [1, 1]]) {
    await page.locator('#block-strip .block-tile').nth(block - 1).click();
    await page.locator('#head-grid .head-tile').nth(head - 1).click();
    await page.getByRole('button', { name: 'Switch off this head' }).click();
    await settled(page);
  }
  await page.locator('#block-strip .block-tile').nth(5).click();
  await page.getByRole('button', { name: 'Switch off block 6' }).click();
  await settled(page);
  await expect(page.locator('#ablation .ablation-state')).toHaveText('14 heads switched off in blocks 1, 6 and 12');
  await expect(page.locator('#ablation-summary')).toHaveText(/^With 14 heads off, /);
  const read = async () => ({
    candidates: await candidates(page),
    lens: await page.locator('#lens-list .lens-token').allTextContents(),
    surprises: await surprises(page),
  });
  const gpu = await read();

  // The CPU recomputes the same tokens with the same heads off and agrees.
  await page.getByRole('radio', { name: 'CPU', exact: true }).click();
  await settled(page);
  await expect(name).toHaveText('Running on the CPU');
  await expect(page.locator('#ablation .ablation-state')).toHaveText('14 heads switched off in blocks 1, 6 and 12');
  const cpu = await read();
  expect(cpu.candidates.map(candidate => candidate.text)).toEqual(gpu.candidates.map(candidate => candidate.text));
  cpu.candidates.forEach((candidate, i) => expect(Math.abs(candidate.logit - gpu.candidates[i].logit)).toBeLessThan(1e-3));
  expect(cpu.lens).toEqual(gpu.lens);
  cpu.surprises.forEach((bits, i) => (bits === null ? expect(gpu.surprises[i]).toBeNull() : expect(Math.abs(bits - gpu.surprises[i])).toBeLessThanOrEqual(0.11)));

  // Every head back on, on the GPU, is the prediction from before any were switched off.
  await page.getByRole('radio', { name: 'GPU', exact: true }).click();
  await settled(page);
  await page.getByRole('button', { name: 'Turn all back on' }).click();
  await settled(page);
  await expect.poll(async () => (await candidates(page)).map(candidate => candidate.text).slice(0, 2)).toEqual(['Mary', 'them']);
  await expect(page.locator('#ablation-summary')).toBeEmpty();
});

test('the GPU tests every head in seconds, and each cell matches switching that head off', async ({ page }) => {
  test.setTimeout(900_000);
  await page.goto('/transformer/?backend=gpu');
  await page.getByRole('button', { name: 'Load the model' }).click();
  await expect(page.locator('.candidate').first()).toBeVisible({ timeout: 300_000 });
  const name = page.locator('#engine-name');
  if (process.env.CI && await name.textContent() !== 'Running on your GPU') {
    test.skip(true, `This CI runner offers no WebGPU adapter, so the page is ${await name.textContent()}.`);
  }
  await page.selectOption('#example-select', 'When Mary and John went to the store, John gave a drink to');
  await expect(page.locator('#token-list .token')).toHaveCount(14);
  await settled(page);

  await page.getByRole('button', { name: 'Test every head' }).click();
  await expect(page.locator('#importance-progress')).toHaveText(/^Every head tested in \d+(\.\d)? m?s$/, { timeout: 120_000 });
  await expect(page.locator('.importance-cell.is-known')).toHaveCount(144);
  await expect(page.locator('#importance-readout')).toHaveText('Block 1, head 1 helps “Mary” most: switched off alone, it falls from 45.3% to 10.9%. Block 12, head 11 holds it back most: without it, “Mary” rises to 73.3%.');

  // Block 2, head 4: its cell, then the head switched off for real, on the GPU and on the CPU.
  const cell = page.locator('.importance-cell').nth(1 * 12 + 3);
  const value = Number(await cell.getAttribute('data-value'));
  expect(value).toBeGreaterThan(.2);
  const expected = `With 1 head off, “Mary” fell from 45.3% to ${(value * 100).toFixed(1)}% and “them” rose to 1st.`;
  await cell.click();
  await expect(page.locator('#head-title')).toHaveText('Block 2, head 4');
  await page.getByRole('button', { name: 'Switch off this head' }).click();
  await settled(page);
  await expect(page.locator('#ablation-summary')).toHaveText(expected);
  await page.getByRole('radio', { name: 'CPU', exact: true }).click();
  await settled(page);
  await expect(name).toHaveText('Running on the CPU');
  await expect(page.locator('#ablation-summary')).toHaveText(expected);
  await page.getByRole('radio', { name: 'GPU', exact: true }).click();
  await settled(page);
  await page.getByRole('button', { name: 'Turn all back on' }).click();
  await settled(page);

  // Changing the prompt stops a test partway (a long prompt takes a few seconds) and clears its map.
  await page.fill('#prompt-input', 'In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters.');
  await expect(page.locator('#token-list .token')).toHaveCount(44);
  await settled(page);
  await expect(page.locator('#importance-map')).toBeHidden();
  await page.getByRole('button', { name: 'Test every head' }).click();
  await expect(page.locator('#importance-progress')).toHaveText(/^Testing [1-9]\d* of 144 heads$/, { timeout: 60_000 });
  await page.fill('#prompt-input', 'The capital of France is');
  await expect(page.getByRole('button', { name: 'Stop' })).toBeHidden();
  await expect(page.locator('#token-list .token')).toHaveCount(5);
  await settled(page);
  await expect(page.locator('#importance-map')).toBeHidden();
  await expect(page.locator('#importance-progress')).toHaveText('');
  await expect(page.getByRole('button', { name: 'Test every head' })).toBeEnabled();
});

test('?backend=cpu starts on the CPU and still offers the GPU', async ({ page }) => {
  test.setTimeout(300_000);
  await page.goto('/transformer/?backend=cpu');
  await page.getByRole('button', { name: 'Load the model' }).click();
  await expect(page.locator('.candidate').first()).toBeVisible({ timeout: 240_000 });
  await expect(page.locator('#engine-name')).toHaveText('Running on the CPU');
  // CI's software GPU only counts when the URL insists on the GPU.
  if (process.env.CI && await page.locator('#engine-choice').isHidden()) {
    test.skip(true, 'This CI runner offers no hardware WebGPU adapter, so there is no GPU to switch to.');
  }
  await expect(page.locator('#engine-choice')).toBeVisible();
  await expect(page.locator('#engine-reason')).toBeHidden();
  await expect(page.getByRole('radio', { name: 'CPU', exact: true })).toHaveAttribute('aria-checked', 'true');
  await page.getByRole('radio', { name: 'GPU', exact: true }).click();
  await settled(page);
  await expect(page.locator('#engine-name')).toHaveText('Running on your GPU');
  await expect(page.locator('#engine-time')).toHaveText(/^9 tokens in /);
});
