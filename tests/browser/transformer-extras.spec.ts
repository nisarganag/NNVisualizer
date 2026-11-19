import { expect, test, type Page } from '@playwright/test';

const pageErrors = new WeakMap<Page, string[]>();
test.beforeEach(({ page }) => {
  const errors: string[] = [];
  pageErrors.set(page, errors);
  page.on('pageerror', error => errors.push(error.message));
});
test.afterEach(({ page }) => expect(pageErrors.get(page)).toEqual([]));

test.describe('transformer extras', () => {
  test('the guess forms block by block, and a typo reads as surprising', async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto('/transformer/');
    await page.getByRole('button', { name: 'Load the model' }).click();
    await expect(page.locator('.candidate').first()).toBeVisible({ timeout: 90_000 });

    // "Pairs" is split into " P" and "airs"; the misspelled ending is about 17 bits.
    await page.fill('#prompt-input', 'The capital of France is Pairs');
    await expect(page.locator('#token-list .token')).toHaveCount(7);
    const strip = page.locator('#surprise-text span');
    await expect(strip).toHaveCount(7);
    await expect(strip.last()).toHaveText('airs');
    await expect(strip.last()).toHaveAttribute('data-surprise', '4');
    await expect(strip.last()).toHaveAttribute('title', /^\d+\.\d bits of surprise, rank [\d,]+$/);
    await expect(strip.nth(2)).toHaveText('of');
    await expect(strip.nth(2)).toHaveAttribute('data-surprise', /^[01]$/);
    // The first token has nothing before it to be surprised by.
    await expect(strip.first()).not.toHaveAttribute('data-surprise');
    await expect(strip.first()).toHaveAttribute('title', 'No context yet');
    await expect(page.locator('#token-list .token').last()).toHaveAttribute('title', /\n\d+\.\d bits of surprise, rank [\d,]+$/);

    // The lens reads the last token; its last row is the model's own prediction.
    const rows = page.locator('#lens-list .lens-row');
    await expect(page.locator('#lens-context')).toContainText('“airs”');
    await expect(page.locator('#lens')).not.toHaveClass(/is-stale/);
    await expect(rows).toHaveCount(13);
    await expect(rows.first().locator('.lens-stage')).toHaveText('Embedding');
    await expect(rows.last().locator('.lens-stage')).toHaveText('Block 12');
    const top = await page.locator('.candidate-text').first().textContent();
    await expect(rows.last().locator('.lens-token')).toHaveText(top!);
    await expect(rows.last()).toHaveClass(/is-final/);

    // A row opens its block, by pointer or keyboard.
    await expect(page.locator('#head-title')).toHaveText('Block 1, head 1');
    await rows.nth(5).click();
    await expect(page.locator('#head-title')).toHaveText('Block 5, head 1');
    await expect(page.getByRole('tab', { name: 'Block 5', exact: true })).toHaveAttribute('aria-selected', 'true');
    await expect(rows.nth(5)).toHaveAttribute('aria-current', 'true');
    await rows.nth(9).focus();
    await page.keyboard.press('Enter');
    await expect(page.locator('#head-title')).toHaveText('Block 9, head 1');

    // Focusing an earlier token reads the lens there.
    await page.locator('#token-list .token').nth(3).hover();
    await expect(page.locator('#lens-context')).toContainText('“France”');
    await expect(page.locator('#lens')).not.toHaveClass(/is-stale/);
    await expect(rows).toHaveCount(13);

    // Nothing overflows on a small phone once the header's indicator has followed the resize.
    await page.setViewportSize({ width: 320, height: 800 });
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
  });
});
