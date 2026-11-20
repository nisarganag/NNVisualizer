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
