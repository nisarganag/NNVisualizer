import { expect, test, type Page } from '@playwright/test';

const pageErrors = new WeakMap<Page, string[]>();
test.beforeEach(({ page }) => {
  const errors: string[] = [];
  pageErrors.set(page, errors);
  page.on('pageerror', error => errors.push(error.message));
});
test.afterEach(({ page }) => expect(pageErrors.get(page)).toEqual([]));

async function ready(page: Page) {
  await expect(page.locator('#main-part')).toHaveAttribute('aria-busy', 'false');
  await page.evaluate(() => document.fonts.ready);
}
async function epoch(page: Page) {
  return Number((await page.locator('#iter-number').innerText()).replaceAll(',', ''));
}
async function step(page: Page) {
  const before = await epoch(page);
  await page.getByRole('button', { name: 'Train one epoch', exact: true }).click();
  await expect.poll(() => epoch(page)).toBe(before + 1);
  await ready(page);
}
async function setRange(page: Page, id: string, value: string) {
  await page.locator('#' + id).evaluate((element: HTMLInputElement, v) => {
    element.value = v;
    element.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await ready(page);
}

test('loads the supplied experiment using local assets and matches the original first epoch', async ({ page }) => {
  const external: string[] = [];
  const failed: string[] = [];
  const workers: string[] = [];
  page.on('request', request => {
    if (/^https?:/.test(request.url()) && !request.url().startsWith('http://127.0.0.1:4173/')) external.push(request.url());
  });
  page.on('response', response => { if (response.status() >= 400) failed.push(response.url()); });
  page.on('worker', worker => workers.push(worker.url()));
  await page.goto('/');
  await ready(page);
  await expect(page).toHaveTitle('NNVisualizer — Neural Network Playground');
  await expect(page.locator('#num-layers')).toHaveText('5');
  await expect(page.locator('#network div.canvas')).toHaveCount(48);
  await expect(page.locator('#heatmap g.train circle')).toHaveCount(250);
  await expect(page.locator('#loss-train')).toHaveText('0.483');
  await expect(page.locator('#loss-test')).toHaveText('0.479');
  await step(page);
  await expect(page.locator('#loss-train')).toHaveText('0.477');
  const weight = await page.locator('#linkx-1').evaluate(element => (element as any).__data__.weight);
  expect(weight).toBeCloseTo(-.24596917717075642, 12);
  expect(workers).toHaveLength(1);
  expect(external).toEqual([]);
  expect(failed).toEqual([]);
});

test('plays, pauses, updates the loss chart and rejects old results after reset', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  await page.getByRole('button', { name: 'Run training', exact: true }).click();
  await expect.poll(() => epoch(page)).toBeGreaterThan(1);
  await page.getByRole('button', { name: 'Pause training', exact: true }).click();
  await ready(page);
  const paused = await epoch(page);
  await page.waitForTimeout(150);
  expect(await epoch(page)).toBe(paused);
  // The loss curve has drawn segments after its starting point (straight or smoothed).
  expect(await page.locator('#linechart path.line').first().getAttribute('d')).toMatch(/^M[^M]+[LCS]/);
  await page.getByRole('button', { name: 'Run training', exact: true }).click();
  await page.getByRole('button', { name: 'Reset the network', exact: true }).click();
  await ready(page);
  await expect(page.locator('#iter-number')).toHaveText('000,000');
  await page.waitForTimeout(250);
  expect(await epoch(page)).toBe(0);
  await expect(page.locator('#play-pause-button')).toHaveAttribute('aria-pressed', 'false');
});

test('trains on all classification and regression datasets', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  for (const dataset of ['circle','xor','gauss','spiral']) {
    await page.locator(`canvas[data-dataset="${dataset}"]`).click();
    await ready(page);
    await step(page);
    expect(Number(await page.locator('#loss-train').innerText())).toBeGreaterThanOrEqual(0);
    expect(new URL(page.url()).hash).toContain(`dataset=${dataset}`);
  }
  await page.selectOption('#problem', 'regression');
  await ready(page);
  for (const dataset of ['reg-plane','reg-gauss']) {
    await page.locator(`canvas[data-regdataset="${dataset}"]`).click();
    await ready(page);
    await step(page);
    expect(Number.isFinite(Number(await page.locator('#loss-train').innerText()))).toBe(true);
    await expect(page.locator('#heatmap g.train circle')).toHaveCount(600);
  }
});

test('activation and regularization choices work, and sliders change the experiment', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  for (const activation of ['relu','sigmoid','linear','tanh']) {
    await page.selectOption('#activations', activation);
    await ready(page);
    await step(page);
    await expect(page.locator('#simulation-status')).toBeHidden();
  }
  for (const regularization of ['none','L2','L1']) {
    await page.selectOption('#regularizations', regularization);
    await ready(page);
    await step(page);
  }
  await page.selectOption('#regularRate', '0.003');
  await ready(page);
  await page.selectOption('#learningRate', '0.01');
  await setRange(page, 'batchSize', '30');
  await setRange(page, 'percTrainData', '70');
  await expect(page.locator('#heatmap g.train circle')).toHaveCount(350);
  await setRange(page, 'noise', '20');
  const seed = new URLSearchParams(new URL(page.url()).hash.slice(1)).get('seed');
  await page.locator('#data-regen-button').click();
  await ready(page);
  expect(new URLSearchParams(new URL(page.url()).hash.slice(1)).get('seed')).not.toBe(seed);
  await expect(page.locator('#batchSize')).toHaveValue('30');
});

test('network editing respects maximum and minimum layer and neuron counts', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  await page.locator('#add-layers').click();
  await ready(page);
  await expect(page.locator('#num-layers')).toHaveText('6');
  await expect(page.locator('#add-layers')).toBeDisabled();
  for (let i = 2; i < 8; i++) {
    await page.getByRole('button', { name: 'Add a neuron to layer 6', exact: true }).click();
    await ready(page);
  }
  await expect(page.getByRole('button', { name: 'Add a neuron to layer 6', exact: true })).toBeDisabled();
  for (let i = 8; i > 1; i--) {
    await page.getByRole('button', { name: 'Remove a neuron from layer 6', exact: true }).click();
    await ready(page);
  }
  await expect(page.getByRole('button', { name: 'Remove a neuron from layer 6', exact: true })).toBeDisabled();
});

test('zero inputs and zero hidden layers remain trainable and survive sharing', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  for (const feature of ['x','y','xSquared','ySquared','xTimesY','sinX','sinY']) {
    await page.locator('#canvas-' + feature).click();
    await ready(page);
  }
  for (let i = 0; i < 5; i++) {
    await page.locator('#remove-layers').click();
    await ready(page);
  }
  await expect(page.locator('#num-layers')).toHaveText('0');
  await expect(page.locator('#remove-layers')).toBeDisabled();
  await step(page);
  expect(Number.isFinite(Number(await page.locator('#loss-train').innerText()))).toBe(true);
  await page.reload();
  await ready(page);
  await expect(page.locator('#num-layers')).toHaveText('0');
  await expect(page.locator('#network path.link')).toHaveCount(0);
  await step(page);
});

test('neuron inspection, test-point visibility and discrete heatmaps work', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  const canvas = page.locator('#heatmap canvas');
  const initial = await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL());
  await page.locator('#canvas-sinX').hover();
  expect(await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())).not.toBe(initial);
  await expect(page.locator('#inspection-label')).toHaveText('Input sin(x₁)');
  // An input feeds every neuron of the first hidden layer; only those connections stay lit.
  await expect(page.locator('#network path.link.related')).toHaveCount(8);
  await page.locator('header').hover();
  expect(await canvas.evaluate((element: HTMLCanvasElement) => element.toDataURL())).toBe(initial);
  await expect(page.locator('#network path.link.related')).toHaveCount(0);
  await expect(page.locator('#inspection-label')).toHaveText('Output');
  await page.locator('label[for="show-test-data"]').click();
  await expect(page.locator('#show-test-data')).toBeChecked();
  await expect(page.locator('#heatmap g.test circle')).toHaveCount(250);
  await page.locator('label[for="discretize"]').click();
  await expect(page.locator('#discretize')).toBeChecked();
  const colors = await canvas.evaluate((element: HTMLCanvasElement) => {
    const pixels = element.getContext('2d')!.getImageData(0,0,100,100).data;
    const colors = new Set<string>();
    for (let i=0;i<pixels.length;i+=4) colors.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);
    return colors.size;
  });
  expect(colors).toBeLessThanOrEqual(2);
  await page.reload();
  await ready(page);
  await expect(page.locator('#discretize')).toBeChecked();
  await expect(page.locator('#show-test-data')).toBeChecked();
});

test('weights and biases can be edited and are reflected by the worker', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  const position = await page.locator('[data-weight-id="x-1"]').evaluate((element: SVGPathElement) => {
    const point = element.getPointAtLength(element.getTotalLength() * .4);
    const screen = new DOMPoint(point.x, point.y).matrixTransform(element.getScreenCTM()!);
    return { x: screen.x, y: screen.y };
  });
  await page.mouse.move(position.x, position.y);
  await page.mouse.click(position.x, position.y);
  const input = page.getByRole('spinbutton', { name: 'Weight or bias value' });
  await expect(input).toBeVisible();
  await input.fill('0.75');
  await ready(page);
  expect(await page.locator('#linkx-1').evaluate(element => (element as any).__data__.weight)).toBe(.75);
  await input.press('Escape');
  await page.locator('#bias-1').hover();
  await page.locator('#bias-1').click();
  await input.fill('-0.5');
  await ready(page);
  expect(await page.locator('#linkx-1').evaluate(element => (element as any).__data__.dest.bias)).toBe(-.5);
  expect(await epoch(page)).toBe(0);
});

test('teaching controls, optional lessons and invalid URLs remain usable', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  await page.locator('.hide-controls label').filter({ hasText: /^Learning rate$/ }).click();
  await expect(page.locator('.hide-controls').getByRole('checkbox', { name: 'Learning rate', exact: true })).not.toBeChecked();
  await page.reload();
  await ready(page);
  await expect(page.locator('#top-controls .ui-learningRate')).toBeHidden();
  await page.goto('/#networkShape=&x=false&y=false&xSquared=false&ySquared=false&xTimesY=false&sinX=false&sinY=false&initZero=true&hideText=true');
  await ready(page);
  await expect(page.locator('header')).toBeHidden();
  await expect(page.locator('#article-text')).toBeHidden();
  await step(page);
  await page.goto('/#networkShape=9999,8,8,8,8,8,8&batchSize=0&learningRate=NaN&percTrainData=100');
  await ready(page);
  await expect(page.locator('#num-layers')).toHaveText('6');
  await step(page);
  await page.goto('/#tutorial=intro');
  await ready(page);
  await expect(page.getByRole('heading', { name: 'Start with a prediction' })).toBeVisible();
  await page.goto('/#tutorial=missing-lesson');
  await ready(page);
  await expect(page.getByRole('heading', { name: 'Understand every connection.', exact: true })).toBeVisible();
});

test('numerical divergence is reported and a reset recovers the interface', async ({ page }) => {
  await page.goto('/#problem=regression&activation=linear&learningRate=10&regularization=none');
  await ready(page);
  await page.getByRole('button', { name: 'Run training', exact: true }).click();
  await expect(page.locator('#simulation-status')).toContainText('Training diverged');
  await expect(page.locator('#play-pause-button')).toHaveAttribute('aria-pressed', 'false');
  await page.selectOption('#learningRate', '0.001');
  await page.locator('#reset-button').click();
  await ready(page);
  await expect(page.locator('#simulation-status')).toBeHidden();
  await expect(page.locator('#iter-number')).toHaveText('000,000');
});

test('narrow screens scroll the complete network without overflowing the page', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await ready(page);
  const sizes = await page.evaluate(() => {
    const region = document.querySelector('.playground-scroll')!;
    return { body: document.documentElement.scrollWidth, viewport: window.innerWidth,
      content: region.scrollWidth, region: region.clientWidth };
  });
  expect(sizes.body).toBeLessThanOrEqual(sizes.viewport);
  expect(sizes.content).toBeGreaterThan(sizes.region);
  await page.getByRole('button', { name: 'Train one epoch', exact: true }).click();
  await expect.poll(() => epoch(page)).toBe(1);
});

test('resizing preserves accessible network controls and classroom visibility', async ({ page }) => {
  await page.goto('/#numNodes1_hide=true');
  await ready(page);
  await expect(page.locator('.ui-numNodes1')).toBeHidden();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Add a neuron to layer 2', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Toggle input x', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.ui-numNodes1')).toBeHidden();
  await page.getByRole('button', { name: 'Toggle input x', exact: true }).focus();
  await page.keyboard.press('Space');
  await ready(page);
  await expect(page.getByRole('button', { name: 'Toggle input x', exact: true })).toHaveAttribute('aria-pressed', 'false');
  await expect(page.locator('.ui-numNodes1')).toBeHidden();
});

test('studio navigation preserves the current experiment and training progress', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  await step(page);
  const experiment = page.url();
  await page.getByRole('button', { name: 'Learning guide', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Understand every connection.', exact: true })).toBeVisible();
  expect(page.url()).toBe(experiment);
  expect(await epoch(page)).toBe(1);
  await page.getByRole('button', { name: 'Classroom settings', exact: true }).click();
  await expect(page.locator('.hide-controls')).toBeVisible();
  expect(page.url()).toBe(experiment);
  expect(await epoch(page)).toBe(1);
  await page.locator('.brand').click();
  await ready(page);
  expect(page.url()).toBe(experiment);
  expect(await epoch(page)).toBe(1);
});

test('copying an experiment link includes its current settings without resetting it', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/');
  await ready(page);
  await page.selectOption('#learningRate', '0.03');
  await step(page);
  await page.getByRole('button', { name: 'Copy experiment link', exact: true }).click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(page.url());
  expect(await epoch(page)).toBe(1);
  await expect(page.locator('#share-feedback')).toContainText('copied');
});

test('the classroom link opens the latest configuration without disturbing the active experiment', async ({ page, context }) => {
  await page.goto('/');
  await ready(page);
  await page.selectOption('#learningRate', '0.03');
  await step(page);
  const link = page.locator('.hide-controls-link');
  await expect(link).toHaveAttribute('href', page.url());
  await page.locator('.hide-controls label').filter({ hasText: /^Learning rate$/ }).click();
  const experiment = page.url();
  const opened = context.waitForEvent('page');
  await link.click();
  const classroom = await opened;
  await ready(classroom);
  expect(classroom.url()).toBe(experiment);
  await expect(classroom.locator('#learningRate')).toHaveValue('0.03');
  await expect(classroom.locator('#top-controls .ui-learningRate')).toBeHidden();
  expect(await epoch(page)).toBe(1);
  expect(await epoch(classroom)).toBe(0);
  await classroom.close();
});

test.describe('themes', () => {
  test.use({ colorScheme: 'dark' });
  test('follow the system, switch without resetting the experiment, and are remembered', async ({ page }) => {
    await page.goto('/');
    await ready(page);
    const root = page.locator('html');
    const toggle = page.getByRole('button', { name: 'Dark theme', exact: true });
    await expect(root).toHaveAttribute('data-theme', 'dark');
    await expect(toggle).toHaveAttribute('aria-pressed', 'true');
    await step(page);
    const experiment = page.url();
    const heatmap = page.locator('#heatmap canvas');
    const thumbnail = page.locator('canvas[data-dataset="gauss"]');
    const paint = (locator: typeof heatmap) => locator.evaluate((element: HTMLCanvasElement) => element.toDataURL());
    const before = { heatmap: await paint(heatmap), thumbnail: await paint(thumbnail),
      link: await page.locator('#linkx-1').evaluate(element => getComputedStyle(element).stroke) };
    await toggle.click();
    await expect(root).toHaveAttribute('data-theme', 'light');
    await expect(toggle).toHaveAttribute('aria-pressed', 'false');
    await expect.poll(() => paint(heatmap)).not.toBe(before.heatmap);
    expect(await paint(thumbnail)).not.toBe(before.thumbnail);
    await expect.poll(() => page.locator('#linkx-1').evaluate(element => getComputedStyle(element).stroke)).not.toBe(before.link);
    expect(await page.locator('meta[name="theme-color"]').getAttribute('content')).toBe('#f3f4f9');
    expect(page.url()).toBe(experiment);
    expect(await epoch(page)).toBe(1);
    await expect(page.locator('#network div.canvas')).toHaveCount(48);
    await page.reload();
    await ready(page);
    await expect(root).toHaveAttribute('data-theme', 'light');
  });
});

test.describe('reduced motion', () => {
  test.use({ reducedMotion: 'reduce' });
  test('keeps the graph still and switches themes instantly', async ({ page }) => {
    await page.goto('/');
    await ready(page);
    await page.locator('#add-layers').click();
    await ready(page);
    expect(await page.evaluate(() => document.getAnimations().length)).toBe(0);
    await page.getByRole('button', { name: 'Dark theme', exact: true }).click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    expect(await page.evaluate(() => document.getAnimations().length)).toBe(0);
  });
});

test.describe('touch parameter editing', () => {
  test.use({ hasTouch: true, isMobile: true, viewport: { width: 390, height: 844 } });
  test('biases and weights can be edited by tapping their visible targets', async ({ page }) => {
    await page.goto('/#networkShape=2&hideText=true');
    await ready(page);
    const bias = page.locator('#bias-1');
    await bias.scrollIntoViewIfNeeded();
    const bounds = await bias.boundingBox();
    await page.touchscreen.tap(bounds!.x + bounds!.width / 2, bounds!.y + bounds!.height / 2);
    const input = page.getByRole('spinbutton', { name: 'Weight or bias value' });
    await expect(input).toBeVisible();
    await input.fill('0.25');
    await ready(page);
    expect(await page.locator('#linkx-1').evaluate(el => (el as any).__data__.dest.bias)).toBe(.25);
    await page.locator('#network-heading').tap();
    await page.locator('.playground-scroll').evaluate(el => { el.scrollLeft = 0; });
    const point = await page.locator('[data-weight-id="x-1"]').evaluate((el: SVGPathElement) => {
      const local = el.getPointAtLength(el.getTotalLength() * .4);
      const screen = new DOMPoint(local.x, local.y).matrixTransform(el.getScreenCTM()!);
      return { x: screen.x, y: screen.y };
    });
    await page.touchscreen.tap(point.x, point.y);
    await expect(input).toBeVisible();
    await input.fill('0.75');
    await ready(page);
    expect(await page.locator('#linkx-1').evaluate(el => (el as any).__data__.weight)).toBe(.75);
    expect(await epoch(page)).toBe(0);
  });
});
