// The CAD workbench itself against the mock backend (project `chromium`): view cube and heads-up view toolbar,
// feature-tree context menu and inline actions, ribbon commands, property pages following the active item.
import { expect, test } from '@playwright/test';
import { importBracket, openPage, selectFlatFaceAtCentre, state, viewportStats } from './helpers';

const ISO = [1, -1.2, 0.8].map((v) => v / Math.hypot(1, 1.2, 0.8));

test.describe('workbench (mock backend)', () => {
  test('view cube faces and the standard-views menu snap the camera; fit all keeps the direction', async ({ page }) => {
    await importBracket(page);
    const dir = async () => (await viewportStats(page)).viewDir;
    const start = await dir();
    for (let k = 0; k < 3; k++) expect(start[k]!).toBeCloseTo(ISO[k]!, 3);

    await page.getByTestId('viewcube-top').click();
    await expect.poll(async () => (await dir())[2]!).toBeGreaterThan(0.999);
    await page.getByTestId('hud-views').click();
    await page.getByTestId('hud-view-front').click();
    await expect.poll(async () => (await dir())[1]!).toBeCloseTo(-1, 3);
    await page.getByTestId('viewcube-iso').click();
    await expect.poll(async () => (await dir())[0]!).toBeCloseTo(ISO[0]!, 3);
    await page.getByTestId('viewcube-right').click();
    await expect.poll(async () => (await dir())[0]!).toBeCloseTo(1, 3);

    // fit all (button and the f key) re-frames from the current direction
    await page.getByTestId('hud-fit').click();
    expect((await dir())[0]!).toBeCloseTo(1, 3);
    await page.keyboard.press('f');
    expect((await dir())[0]!).toBeCloseTo(1, 3);
    expect(await page.evaluate(() => window.__topopViewport!.coverage())).toBeGreaterThan(0.03);
  });

  test('display styles, section plane and ghosting from the heads-up toolbar', async ({ page }) => {
    await importBracket(page);
    const st = () => viewportStats(page);
    expect((await st()).displayStyle).toBe('edges');
    expect((await st()).edgesVisible).toBe(true);
    await page.getByTestId('hud-style-wireframe').click();
    await expect.poll(async () => (await st()).wireframe).toBe(true);
    expect((await st()).edgesVisible).toBe(false);
    await page.getByTestId('hud-style-shaded').click();
    await expect.poll(async () => [(await st()).wireframe, (await st()).edgesVisible]).toEqual([false, false]);
    await page.getByTestId('hud-style-edges').click();
    await expect.poll(async () => (await st()).edgesVisible).toBe(true);

    // section: clips the design (faces and edges) at the middle of the box, along the chosen axis
    const full = await page.evaluate(() => window.__topopViewport!.coverage());
    await page.getByTestId('hud-section').click();
    await expect(page.getByTestId('section-pop')).toBeVisible();
    await expect.poll(async () => (await st()).section).toEqual({ axis: 'x', t: 0.5 });
    await expect.poll(async () => (await st()).clippedMaterials).toBeGreaterThanOrEqual(2);
    await page.getByTestId('section-axis-z').click();
    await page.getByTestId('section-pos').fill('0.1');
    await expect.poll(async () => (await st()).section).toEqual({ axis: 'z', t: 0.1 });
    const cut = await page.evaluate(() => window.__topopViewport!.coverage());
    expect(cut).toBeLessThan(full * 0.8); // only the bottom tenth of the bracket is left
    await page.getByTestId('section-off').click();
    await expect.poll(async () => [(await st()).section, (await st()).clippedMaterials]).toEqual([null, 0]);

    await page.getByTestId('hud-ghost').click();
    expect(await state(page, (s) => s.ghostDesign)).toBe(false);
    await page.getByTestId('hud-ghost').click();
    expect(await state(page, (s) => s.ghostDesign)).toBe(true);
  });

  test('feature tree: active rows, rename, context menu, show/hide; Esc closes the menu only', async ({ page }) => {
    await importBracket(page);
    await selectFlatFaceAtCentre(page);
    await page.getByTestId('add-load').click();
    const row = page.getByTestId('load-row').first();
    await expect(row).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'load');
    await expect(page.getByTestId('pm-title')).toHaveText('Load 1');

    // double-click renames inline
    await row.dblclick();
    await page.getByTestId('tree-rename').fill('Bolt load');
    await page.getByTestId('tree-rename').press('Enter');
    expect(await state(page, (s) => s.project.loads[0]!.name)).toBe('Bolt load');
    await expect(page.getByTestId('pm-title')).toHaveText('Bolt load');
    await expect(page.getByTestId('load-name')).toHaveValue('Bolt load');

    // the context menu: Esc closes it and leaves the selection alone
    await selectFlatFaceAtCentre(page);
    await row.click({ button: 'right' });
    await expect(page.getByTestId('context-menu')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('context-menu')).toHaveCount(0);
    expect(await state(page, (s) => s.selection.faceIds.length)).toBeGreaterThan(1);
    // ... "Use current selection" replaces the load's region, "Rename" and "Delete" do what they say
    await row.click({ button: 'right' });
    await page.getByTestId('ctx-replace').click();
    const faces = await state(page, (s) => (s.project.loads[0]!.selection as { face_ids: number[] }).face_ids.length);
    expect(faces).toBeGreaterThan(1);
    await row.getByTestId('tree-more').click();
    await page.getByTestId('ctx-rename').click();
    await page.getByTestId('tree-rename').fill('Renamed');
    await page.getByTestId('tree-rename').press('Enter');
    expect(await state(page, (s) => s.project.loads[0]!.name)).toBe('Renamed');
    await row.click({ button: 'right' });
    await page.getByTestId('ctx-delete').click();
    await expect.poll(() => state(page, (s) => s.project.loads.length)).toBe(0);
    await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'loads');

    // reference bodies: the eye in the row toggles visibility, the menu switches keep-in / keep-out
    await openPage(page, 'refs');
    await page.getByTestId('add-ref-box').click();
    await expect.poll(async () => (await viewportStats(page)).refs).toBe(1);
    const ref = page.getByTestId('ref-row').first();
    await ref.getByTestId('ref-visible').uncheck();
    expect(await state(page, (s) => s.project.ref_models[0]!.visible)).toBe(false);
    await ref.getByTestId('ref-visible').check();
    await ref.click({ button: 'right' });
    await page.getByTestId('ctx-mode').click();
    expect(await state(page, (s) => s.project.ref_models[0]!.mode)).toBe('keep_in');
    await expect(ref).toContainText('keep-in');
  });

  test('ribbon: tabs follow the workflow stage, commands act like the panels', async ({ page }) => {
    await importBracket(page);
    await page.getByTestId('ribbon-tab-bc').click();
    await page.getByTestId('rb-mode-pick').click();
    expect(await state(page, (s) => s.tool)).toBe('pick');
    await selectFlatFaceAtCentre(page);
    await page.getByTestId('rb-add-load').click();
    await expect.poll(() => state(page, (s) => s.project.loads.length)).toBe(1);
    await page.getByTestId('rb-add-box').click();
    await page.getByTestId('rb-add-support').click();
    await expect.poll(() => state(page, (s) => s.project.supports.length)).toBe(1);

    await page.getByTestId('ribbon-tab-optimize').click();
    await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'study');
    await page.getByTestId('rb-symmetry').click();
    expect(await state(page, (s) => s.project.params.symmetry)).toEqual([{ axis: 'x', position: null }]);
    await page.getByTestId('ribbon-tab-domain').click();
    await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'domain');
    await page.getByTestId('domain-elements').fill('24');
    await page.getByTestId('ribbon-tab-optimize').click();
    await page.getByTestId('p-max-iter').fill('3');
    await page.getByTestId('rb-run').click();
    await expect.poll(() => state(page, (s) => s.run.status), { timeout: 15_000 }).toBe('done');
    await expect(page.getByTestId('sb-status')).toHaveText('done');
    await expect(page.getByTestId('sb-iter')).toContainText('3');
    await expect(page.getByTestId('tree-run')).toContainText('done');

    await page.getByTestId('ribbon-tab-results').click();
    await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'results');
    await expect(page.getByTestId('hud-threshold')).toBeVisible();
    await page.getByTestId('hud-threshold').fill('0.8');
    expect(await state(page, (s) => s.threshold)).toBeCloseTo(0.8, 5);
    await expect(page.getByTestId('threshold')).toHaveValue('0.8');
    await page.getByTestId('rb-load-result').click();
    await expect.poll(async () => (await viewportStats(page)).result).toBe(true);
    await page.getByTestId('rb-hide-result').click();
    await expect.poll(async () => (await viewportStats(page)).result).toBe(false);
  });
});
