// Screenshots of the real server flow for the docs: bracket, a load + a support + a keep-out, a finished run.
// Run with `npx playwright test -c playwright.shots.config.ts`; writes docs/img/gui-*.png.
import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';
import { BRACKET, openPage, state } from '../e2e/helpers';

const img = (name: string) => fileURLToPath(new URL(`../../docs/img/${name}.png`, import.meta.url));

test('GUI screenshots: setup, results (light and dark), tablet', async ({ browser }) => {
  const baseURL = String(test.info().project.use.baseURL);
  const page = await (await browser.newContext({ baseURL, viewport: { width: 1440, height: 900 }, colorScheme: 'light' })).newPage();
  const settle = async () => {
    await page.getByTestId('hud-fit').click();
    const bar = (await page.getByTestId('status-bar').boundingBox())!;
    await page.mouse.move(bar.x + bar.width / 2, bar.y + bar.height / 2); // no hover highlight in the pictures
    await page.waitForTimeout(600);
  };

  await page.goto('/');
  await page.getByTestId('mesh-file-input').setInputFiles(BRACKET);
  await expect.poll(() => page.evaluate(() => window.__topopViewport?.stats().designFaces ?? 0)).toBeGreaterThan(0);
  await page.getByTestId('project-name').fill('Bracket - stiffest 30 %');

  // support: the plate bottom (largest facet)
  await page.getByTestId('mode-query').click();
  await page.getByTestId('query-tab-facets').click();
  await page.getByTestId('facet-row').first().click();
  await page.getByTestId('add-support').click();
  // load: the wall top, pushed down
  const top = await state(page, (s) => s.meshes[s.project.design_mesh!.mesh_id!]!.info.bbox[1]![2]!);
  await page.getByTestId('query-tab-normal').click();
  await page.getByTestId('normal-dir-+Z').click();
  await page.getByTestId('normal-within-toggle').check();
  await page.getByTestId('within-min-z').fill(String(top - 5));
  await page.getByTestId('add-load').click();
  await page.getByTestId('load-name').fill('Wall top');
  await page.getByTestId('force-z').fill('-100');
  // keep-out: a bolt clearance through the plate
  await page.getByTestId('mode-orbit').click();
  await openPage(page, 'refs');
  await page.getByTestId('add-ref-cylinder').click();
  await expect.poll(() => state(page, (s) => s.project.ref_models.length)).toBe(1);
  for (const [id, v] of [
    ['ref-scale-x', 6],
    ['ref-scale-y', 30],
    ['ref-scale-z', 6],
    ['ref-rot-x', 90],
    ['ref-pos-x', 55],
    ['ref-pos-y', 30],
    ['ref-pos-z', 5],
  ] as const)
    await page.getByTestId(id).fill(String(v));
  await page.getByTestId('mode-orbit').click();
  await openPage(page, 'domain');
  await page.getByTestId('domain-elements').fill('40');
  await expect.poll(() => state(page, (s) => s.voxel.stats?.nx ?? 0), { timeout: 60_000 }).toBe(42);
  await page.getByTestId('load-row').first().click();
  await settle();
  await page.screenshot({ path: img('gui-setup-light') });

  // run
  await openPage(page, 'study');
  await page.getByTestId('p-max-iter').fill('40');
  await page.getByTestId('run-start').click();
  await expect.poll(() => state(page, (s) => s.run.status), { timeout: 170_000 }).toBe('done');
  await openPage(page, 'results');
  await settle();
  await page.screenshot({ path: img('gui-results-light') });

  await page.getByTestId('theme-toggle').click();
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark');
  await settle();
  await page.screenshot({ path: img('gui-results-dark') });

  await page.getByTestId('theme-toggle').click();
  await page.setViewportSize({ width: 1024, height: 768 });
  await settle();
  await page.screenshot({ path: img('gui-tablet') });
  await page.context().close();
});
