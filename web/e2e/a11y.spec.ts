// WCAG 2.1 A/AA audit with axe-core over the main GUI states, in both themes (project `chromium`, mock backend).
// Serious and critical violations fail the test; moderate and minor ones are printed to the console.
//
// The one exclusion: the WebGL <canvas id="viewport"> is left out of the color-contrast run only (every other rule
// still sees it). It draws the 3D scene, not text, so 1.4.3 has nothing to measure on it.
// axe leaves text over translucent overlays and gradients (selection bar, hint, view cube, triad, group headers)
// "incomplete"; contrastIssues() (a11y-helpers.ts) measures those, plus placeholders and input boundaries (1.4.11).
// Then: landmarks / headings, live announcements, single-key shortcuts, reduced motion and reflow at 320 px / 200 %.
import AxeBuilder from '@axe-core/playwright';
import { expect, type Page, test } from '@playwright/test';
import { contrastIssues } from './a11y-helpers';
import { BRACKET, defineBoundaries, importBracket, openPage, state, viewportStats } from './helpers';

const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'];
const THEMES = ['light', 'dark'] as const;

async function audit(page: Page, label: string): Promise<void> {
  // let transitions and the theme switch settle
  await page.waitForTimeout(150);
  const all = await new AxeBuilder({ page }).withTags(TAGS).disableRules(['color-contrast']).analyze();
  const contrast = await new AxeBuilder({ page }).withRules(['color-contrast']).exclude('canvas#viewport').analyze();
  const violations = [...all.violations, ...contrast.violations];
  const blocking = violations.filter((v) => v.impact === 'serious' || v.impact === 'critical');
  const soft = violations.filter((v) => !(v.impact === 'serious' || v.impact === 'critical'));
  for (const v of soft) console.log(`[a11y ${label}] ${v.impact} ${v.id}: ${v.help} (${v.nodes.length}) ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`);
  const report = blocking.map((v) => `${v.impact} ${v.id}: ${v.help}\n${v.nodes.map((n) => `    ${n.target.join(' ')}  ${n.failureSummary?.split('\n').slice(1).join('; ')}`).join('\n')}`);
  expect(report, `${label}: serious/critical axe violations`).toEqual([]);
  expect(await contrastIssues(page), `${label}: contrast (text 4.5:1 / large 3:1, input boundaries 3:1)`).toEqual([]);
}

for (const theme of THEMES) {
  test.describe(`axe, ${theme} theme`, () => {
    test.beforeEach(async ({ page }) => {
      await page.addInitScript((t) => localStorage.setItem('topop.ui.theme', t), theme);
    });

    test('empty start', async ({ page }) => {
      await page.goto('/');
      await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
      await expect(page.getByTestId('welcome')).toBeVisible();
      await audit(page, `${theme} empty`);
    });

    test('bracket loaded', async ({ page }) => {
      await importBracket(page);
      await expect(page.getByTestId('voxel-active')).toBeVisible();
      await audit(page, `${theme} bracket`);
    });

    test('load, support and keep-out defined, property panel open', async ({ page }) => {
      await importBracket(page);
      await defineBoundaries(page);
      await openPage(page, 'refs');
      await page.getByTestId('add-ref-box').click();
      await expect.poll(() => state(page, (s) => s.project.ref_models.length)).toBe(1);
      await audit(page, `${theme} ref page`);
      await page.getByTestId('load-row').first().click();
      await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'load');
      await audit(page, `${theme} load page`);
      await page.getByTestId('support-row').first().click();
      await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'support');
      await audit(page, `${theme} support page`);
      // the query tool's facet table and the study page with every group open
      await page.getByTestId('mode-query').click();
      await page.getByTestId('query-tab-facets').click();
      await page.getByTestId('facets-fetch').click();
      await expect(page.getByTestId('facet-list')).toBeVisible();
      await openPage(page, 'study');
      for (const b of await page.getByTestId('panel-study').locator('.group-toggle[aria-expanded="false"]').all()) await b.click();
      await page.getByTestId('p-stress-limit').fill('2');
      await audit(page, `${theme} query + study`);
    });

    test('run finished, results tab', async ({ page }) => {
      await importBracket(page);
      await defineBoundaries(page);
      await openPage(page, 'domain');
      await page.getByTestId('domain-elements').fill('24');
      await openPage(page, 'study');
      await page.getByTestId('p-max-iter').fill('4');
      await page.getByTestId('run-start').click();
      await expect.poll(() => state(page, (s) => s.run.status), { timeout: 30_000 }).toBe('done');
      await page.getByTestId('ribbon-tab-results').click();
      await page.getByTestId('color-stress').check();
      await expect(page.getByTestId('stress-legend')).toBeVisible();
      await audit(page, `${theme} results`);
    });

    test('menus and popovers open', async ({ page }) => {
      await importBracket(page);
      await defineBoundaries(page);
      await page.getByTestId('load-row').first().click({ button: 'right' });
      await expect(page.getByTestId('context-menu')).toBeVisible();
      await audit(page, `${theme} context menu`);
      await page.keyboard.press('Escape');
      await page.getByTestId('hud-views').click();
      await expect(page.getByTestId('hud-view-front')).toBeVisible();
      await audit(page, `${theme} views menu`);
      await page.keyboard.press('Escape');
      await page.getByTestId('hud-section').click();
      await expect(page.getByTestId('section-pop')).toBeVisible();
      await audit(page, `${theme} section popover`);
    });
  });
}

test.describe('structure and behaviour', () => {
  test('landmarks, one h1, heading levels in order, named canvas with instructions', async ({ page }) => {
    await importBracket(page);
    await defineBoundaries(page);
    for (const role of ['banner', 'main', 'contentinfo'] as const) await expect(page.getByRole(role)).toHaveCount(1);
    await expect(page.getByRole('navigation', { name: 'Feature tree' })).toBeVisible();
    await expect(page.getByRole('complementary', { name: 'Property manager' })).toBeVisible();
    await expect(page.getByRole('region', { name: 'Command ribbon' })).toBeVisible();
    await expect(page.locator('h1')).toHaveCount(1);
    const levels = await page.locator('h1, h2, h3, h4, h5, h6').evaluateAll((hs) =>
      hs.filter((h) => (h as HTMLElement).offsetParent !== null).map((h) => Number(h.tagName[1])),
    );
    expect(levels[0]).toBe(1);
    for (let i = 1; i < levels.length; i++) expect(levels[i]! - levels[i - 1]!, `heading ${i}: h${levels[i - 1]} -> h${levels[i]}`).toBeLessThanOrEqual(1);
    const canvas = page.locator('canvas#viewport');
    await expect(canvas).toHaveAccessibleName('3D viewport');
    await expect(canvas).toHaveAccessibleDescription(/Without a mouse: .*Query tool/);
  });

  test('live region: run start, every 10th iteration, done; notices are announced', async ({ page }) => {
    test.setTimeout(60_000);
    await importBracket(page);
    await defineBoundaries(page);
    await openPage(page, 'domain');
    await page.getByTestId('domain-elements').fill('24');
    await openPage(page, 'study');
    await page.getByTestId('p-max-iter').fill('25');
    await page.evaluate(() => {
      const w = window as unknown as { __said: string[] };
      w.__said = [];
      const el = document.querySelector('[data-testid="live-status"]')!;
      new MutationObserver(() => w.__said.push(el.textContent!.trim())).observe(el, { childList: true, characterData: true, subtree: true });
    });
    await page.getByTestId('run-start').click();
    await expect.poll(() => state(page, (s) => s.run.status), { timeout: 30_000 }).toBe('done');
    const said = await page.evaluate(() => (window as unknown as { __said: string[] }).__said.filter(Boolean));
    expect(said).toEqual([
      'Run started.',
      expect.stringMatching(/^Iteration 10 of 25, compliance [\d.]+\.$/),
      expect.stringMatching(/^Iteration 20 of 25, compliance [\d.]+\.$/),
      expect.stringMatching(/^Run finished after 25 iterations, compliance [\d.]+\.$/),
    ]);
    await expect(page.getByTestId('live-status')).toHaveAttribute('aria-live', 'polite');

    // a warning notice goes to the polite region, an error to the assertive one
    await page.getByTestId('ribbon-tab-bc').click();
    await page.keyboard.press('Escape'); // clears the selection and releases the active item
    await page.getByTestId('rb-preview').click();
    await expect(page.getByTestId('notice')).toBeVisible();
    await expect(page.getByTestId('live-status')).toContainText('Nothing to preview');
    await page.getByTestId('project-file-input').setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{"nope": 1}') });
    await expect(page.getByTestId('live-alert')).not.toBeEmpty();
    await expect(page.getByTestId('live-alert')).toHaveText(await page.getByTestId('notice').innerText());
  });

  test('single-key shortcuts can be turned off (WCAG 2.1.4)', async ({ page }) => {
    await importBracket(page);
    const tool = () => state(page, (s) => s.tool);
    await page.keyboard.press('2');
    expect(await tool()).toBe('pick');
    await page.getByTestId('shortcuts-toggle').click();
    await expect(page.getByTestId('shortcuts-toggle')).toHaveAttribute('aria-pressed', 'false');
    await page.locator('body').click({ position: { x: 5, y: 5 } }).catch(() => undefined);
    await page.keyboard.press('1');
    await page.keyboard.press('5');
    expect(await tool()).toBe('pick');
    await page.reload();
    await page.getByTestId('mesh-file-input').setInputFiles(BRACKET);
    await expect(page.getByTestId('shortcuts-toggle')).toHaveAttribute('aria-pressed', 'false'); // remembered
    await page.getByTestId('shortcuts-toggle').click();
    await page.keyboard.press('5');
    await expect.poll(tool).toBe('query');
  });

  test('reduced motion: no camera tween, no spinner or caret animation', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await importBracket(page);
    await page.getByTestId('hud-views').click();
    await page.getByTestId('hud-view-top').click();
    // without the 280 ms tween the camera is already there on the next frame
    await page.evaluate(() => new Promise(requestAnimationFrame));
    expect((await viewportStats(page)).viewDir[2]!).toBeGreaterThan(0.999);
    const dur = await page.locator('.tree-caret .icon').first().evaluate((el) => parseFloat(getComputedStyle(el).transitionDuration));
    expect(dur).toBeLessThan(0.001);
  });

  for (const [w, h, label] of [
    [320, 640, '320 CSS px'],
    [640, 360, '1280 x 720 at 200 % zoom'],
  ] as const) {
    test(`reflow at ${label}: no horizontal scroll, every pane reachable`, async ({ page }) => {
      await page.setViewportSize({ width: w, height: h });
      await importBracket(page);
      // boundaries from region primitives (the canvas centre may be scrolled out of view here)
      await page.getByTestId('add-box').click();
      await page.getByTestId('add-load').click();
      await page.getByTestId('add-sphere').click();
      await page.getByTestId('add-support').click();
      await expect.poll(() => state(page, (s) => s.project.loads.length + s.project.supports.length)).toBe(2);
      const overflowX = () => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      expect(await overflowX()).toBeLessThanOrEqual(0);
      for (const id of ['ribbon-tab-results', 'new-project', 'theme-toggle', 'tree-loads', 'load-row', 'run-start', 'property-manager', 'sb-status', 'hud-fit', 'mode-query', 'add-load']) {
        const el = page.getByTestId(id).first();
        await el.scrollIntoViewIfNeeded();
        await expect(el, id).toBeVisible();
        const box = (await el.boundingBox())!;
        expect(box.x, `${id} left edge`).toBeGreaterThanOrEqual(-1);
        expect(box.x + box.width, `${id} right edge`).toBeLessThanOrEqual(w + 1);
      }
      // the property panel stacks under the viewport and still works
      await page.getByTestId('load-row').first().click();
      await page.getByTestId('force-z').scrollIntoViewIfNeeded();
      await page.getByTestId('force-z').fill('-42');
      expect(await state(page, (s) => s.project.loads[0]!.force[2])).toBe(-42);
      await page.getByTestId('ribbon-tab-optimize').click();
      await expect(page.getByTestId('rb-run')).toBeVisible();
      expect(await overflowX()).toBeLessThanOrEqual(0);
    });
  }
});
