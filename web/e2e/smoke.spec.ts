import { expect, test } from '@playwright/test';
import { openPage, type PageId } from './helpers';

const TABS = ['Model', 'Boundary Conditions', 'Domain', 'Optimize', 'Results'];
const TREE: [PageId, string][] = [
  ['part', 'Design body'],
  ['refs', 'Reference bodies'],
  ['loads', 'Loads'],
  ['supports', 'Supports'],
  ['domain', 'Domain / Mesh'],
  ['study', 'Study settings'],
  ['material', 'Material'],
  ['results', 'Results'],
];

test('workbench shell: ribbon, feature tree, property manager, job monitor, status bar, view cube and canvas render', async ({ page }) => {
  await page.goto('/');
  const ribbon = page.getByRole('tablist', { name: 'Ribbon' });
  for (const name of TABS) await expect(ribbon.getByRole('tab', { name, exact: true })).toBeVisible();
  const tree = page.getByTestId('feature-tree');
  for (const [id, label] of TREE) await expect(tree.getByTestId(`tree-${id}`)).toContainText(label);
  await expect(page.getByTestId('property-manager')).toBeVisible();
  await expect(page.getByTestId('panel-run')).toBeVisible();
  await expect(page.getByTestId('status-bar')).toContainText('units: as mesh');
  await expect(page.getByTestId('view-cube')).toBeVisible();
  await expect(page.getByTestId('view-toolbar')).toBeVisible();
  await expect(page.locator('canvas#viewport')).toBeVisible();

  // every tree node opens its page, and the ribbon follows the workflow stage
  for (const [id] of TREE) await openPage(page, id);
  await expect(ribbon.getByRole('tab', { name: 'Results', exact: true })).toHaveAttribute('aria-selected', 'true');
  await ribbon.getByRole('tab', { name: 'Domain', exact: true }).click();
  await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'domain');
  await expect(page.getByTestId('domain-elements')).toBeVisible();
});

test('theme toggle switches the tokens; the choice survives a reload', async ({ page }) => {
  await page.goto('/');
  const theme = () => page.evaluate(() => document.documentElement.dataset.theme);
  const first = await theme();
  await page.getByTestId('theme-toggle').click();
  const second = await theme();
  expect(second).not.toBe(first);
  const bg = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  const before = await bg();
  await page.reload();
  expect(await theme()).toBe(second);
  expect(await bg()).toBe(before);
});
