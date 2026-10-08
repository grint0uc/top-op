import { expect, test } from '@playwright/test';

const PANELS = ['Import', 'Domain', 'Loads', 'Supports', 'Reference models', 'Run', 'Results'];

test('sidebar panels and viewport canvas render', async ({ page }) => {
  await page.goto('/');
  const sidebar = page.locator('aside');
  for (const name of PANELS) {
    await expect(sidebar.getByRole('heading', { name, exact: true })).toBeVisible();
  }
  await expect(page.locator('canvas#viewport')).toBeVisible();
});
