// Keyboard-only use of the workbench (project `chromium`, mock backend): WCAG 2.1.1 keyboard, 2.1.2 no trap, 2.4.3
// focus order, 2.4.7 focus visible. No mouse event is sent in these tests except where a comment says so; the
// viewport face picking is the one mouse-only feature, and the query / primitive selections cover what it does.
import { expect, type Page, test } from '@playwright/test';
import { focusedRole, focusedTestId, focusInfo, tabTo } from './a11y-helpers';
import { state, viewportStats } from './helpers';

const press = (page: Page, ...keys: string[]) => keys.reduce((p, k) => p.then(() => page.keyboard.press(k)), Promise.resolve());

/** Replaces the value of the focused field, as a keyboard user would (select all, type). */
async function typeInto(page: Page, testId: string, text: string): Promise<void> {
  await tabTo(page, testId);
  await press(page, 'ControlOrMeta+a');
  await page.keyboard.type(text);
}

/** Loads the bracket with the Example button, by keyboard. */
async function keyboardImport(page: Page): Promise<void> {
  await page.goto('/');
  await tabTo(page, 'welcome-example');
  await press(page, 'Enter');
  await expect.poll(async () => (await viewportStats(page)).designFaces).toBeGreaterThan(0);
  // the welcome card is gone: focus stays on the page (the Design body page title), not on <body>
  await expect.poll(() => focusedTestId(page)).toBe('pm-title');
}

/** Tabs to the treeitem in the tab order (the tree is one tab stop). */
async function tabToTree(page: Page): Promise<void> {
  for (let i = 0; i < 200; i++) {
    if ((await focusedRole(page)) === 'treeitem') return;
    await press(page, 'Tab');
  }
  throw new Error('tree not reached');
}

/** Arrow-down through the tree until the treeitem with this test id has focus. */
async function treeDownTo(page: Page, testId: string): Promise<void> {
  await press(page, 'Home');
  for (let i = 0; i < 40; i++) {
    if ((await focusedTestId(page)) === testId) return;
    await press(page, 'ArrowDown');
  }
  throw new Error(`${testId} not in the tree`);
}

test.describe('keyboard only', () => {
  test('ribbon tablist: one tab stop, arrows / Home / End select, aria-selected and the panel follow', async ({ page }) => {
    await page.goto('/');
    await tabTo(page, 'ribbon-tab-model');
    const tab = (id: string) => page.getByTestId(`ribbon-tab-${id}`);
    await expect(tab('model')).toHaveAttribute('aria-selected', 'true');
    await expect(tab('model')).toHaveAttribute('aria-controls', 'ribbon-panel');
    await press(page, 'ArrowRight');
    expect(await focusedTestId(page)).toBe('ribbon-tab-bc');
    await expect(tab('bc')).toHaveAttribute('aria-selected', 'true');
    await expect(tab('model')).toHaveAttribute('aria-selected', 'false');
    await expect(page.getByRole('tabpanel')).toHaveAccessibleName('Boundary Conditions');
    await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'loads');
    await press(page, 'End');
    expect(await focusedTestId(page)).toBe('ribbon-tab-results');
    await press(page, 'ArrowRight'); // wraps
    expect(await focusedTestId(page)).toBe('ribbon-tab-model');
    await press(page, 'ArrowLeft');
    expect(await focusedTestId(page)).toBe('ribbon-tab-results');
    await press(page, 'Home', 'ArrowRight', 'ArrowRight');
    await expect(tab('domain')).toHaveAttribute('aria-selected', 'true');
    // Tab leaves the tablist for the panel's first command, Shift+Tab comes back to the selected tab
    await press(page, 'Tab');
    expect(await focusedTestId(page)).toBe('rb-domain');
    await press(page, 'Shift+Tab');
    expect(await focusedTestId(page)).toBe('ribbon-tab-domain');
    expect(await page.getByRole('tab').evaluateAll((ts) => ts.map((t) => t.getAttribute('tabindex')))).toEqual(['-1', '-1', '0', '-1', '-1']);
  });

  test('main flow: import, query load, primitive support, keep-out, tree, property panel, run and stop', async ({ page }) => {
    test.setTimeout(90_000);
    await keyboardImport(page);

    // ribbon command: the Query tool, then a normal +Z selection becomes a load
    await tabTo(page, 'ribbon-tab-model', { back: true });
    await press(page, 'ArrowRight');
    await tabTo(page, 'rb-mode-query');
    await press(page, 'Enter');
    expect(await state(page, (s) => s.tool)).toBe('query');
    await tabTo(page, 'query-tab-normal');
    await press(page, 'Space');
    await tabTo(page, 'normal-dir-+Z');
    await press(page, 'Enter');
    await tabTo(page, 'query-apply');
    await press(page, 'Enter');
    expect(await state(page, (s) => s.selection.query?.kind)).toBe('normal');
    await tabTo(page, 'add-load');
    await press(page, 'Enter');
    await expect.poll(() => state(page, (s) => s.project.loads.length)).toBe(1);
    await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'load');

    // edit the load's page
    await typeInto(page, 'force-z', '-250');
    await typeInto(page, 'load-name', 'Top load');
    expect(await state(page, (s) => [s.project.loads[0]!.force[2], s.project.loads[0]!.name])).toEqual([-250, 'Top load']);
    await tabTo(page, 'pm-done');
    await press(page, 'Enter');
    await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'loads');
    expect(await focusedTestId(page)).toBe('pm-title'); // the Done button went away; focus stays in the panel

    // a box region (fields instead of the gizmo) becomes a roller support
    await tabTo(page, 'add-box');
    await press(page, 'Enter');
    await typeInto(page, 'prim-size-sz', '4');
    await tabTo(page, 'add-support');
    await press(page, 'Enter');
    await expect.poll(() => state(page, (s) => s.project.supports.length)).toBe(1);
    await tabTo(page, 'fix-z');
    await press(page, 'Space');
    expect(await state(page, (s) => s.project.supports[0]!.fix)).toEqual([true, true, false]);

    // keep-out box from the Domain tab
    await tabTo(page, 'ribbon-tab-bc', { back: true });
    await press(page, 'ArrowRight');
    await tabTo(page, 'rb-add-ref-box');
    await press(page, 'Enter');
    await expect.poll(() => state(page, (s) => s.project.ref_models.length)).toBe(1);
    await typeInto(page, 'ref-pos-x', '5');
    expect(await state(page, (s) => s.project.ref_models[0]!.transform![12])).toBeCloseTo(5, 6);

    // feature tree: arrows move and fold, Enter opens, F2 renames, Delete removes the focused row
    await tabToTree(page);
    await treeDownTo(page, 'load-row');
    await press(page, 'Enter');
    await expect(page.getByTestId('load-row')).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByTestId('pm-title')).toHaveText('Top load');
    await press(page, 'F2');
    expect(await focusedTestId(page)).toBe('tree-rename');
    await press(page, 'ControlOrMeta+a');
    await page.keyboard.type('Bolt load');
    await press(page, 'Enter');
    expect(await state(page, (s) => s.project.loads[0]!.name)).toBe('Bolt load');
    expect(await focusedTestId(page)).toBe('load-row');
    await press(page, 'ArrowLeft');
    expect(await focusedTestId(page)).toBe('tree-loads');
    await press(page, 'ArrowLeft');
    await expect(page.getByTestId('tree-loads')).toHaveAttribute('aria-expanded', 'false');
    await expect(page.getByTestId('load-row')).toHaveCount(0);
    await press(page, 'ArrowRight');
    await expect(page.getByTestId('tree-loads')).toHaveAttribute('aria-expanded', 'true');
    await press(page, 'ArrowRight');
    expect(await focusedTestId(page)).toBe('load-row');
    await treeDownTo(page, 'tree-study');
    await press(page, 'Enter');
    await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'study');

    // start and stop a run (Stop is the next tab stop after Run)
    await typeInto(page, 'p-max-iter', '400');
    await tabTo(page, 'run-start');
    await press(page, 'Enter');
    await expect.poll(() => state(page, (s) => s.run.status)).toBe('running');
    // Run turned disabled under the focus: the focus moved on to Stop instead of dropping to <body>
    await expect.poll(() => focusedTestId(page)).toBe('run-stop');
    await press(page, 'Enter');
    await expect.poll(() => state(page, (s) => s.run.status)).toBe('cancelled');
    await expect(page.getByTestId('live-status')).toContainText('Run stopped');
    await expect.poll(() => focusedTestId(page)).toBe('run-start');

    // a short run from the ribbon, then the Results tab and an export link
    await typeInto(page, 'p-max-iter', '3');
    await tabTo(page, 'ribbon-tab-optimize', { back: true });
    await tabTo(page, 'rb-run');
    await press(page, 'Enter');
    await expect.poll(() => state(page, (s) => s.run.status), { timeout: 15_000 }).toBe('done');
    await expect.poll(() => focusedTestId(page)).toBe('rb-run'); // Stop -> Run again once it is over
    await tabTo(page, 'ribbon-tab-optimize', { back: true });
    await press(page, 'ArrowRight');
    await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'results');
    await tabTo(page, 'download-stl');
    await expect(page.getByTestId('download-stl')).toHaveAttribute('href', /result\.stl/);

    // Delete on a focused tree row removes that row's item, and focus moves to a neighbouring row
    await tabToTree(page);
    await treeDownTo(page, 'support-row');
    await press(page, 'Delete');
    await expect.poll(() => state(page, (s) => s.project.supports.length)).toBe(0);
    expect(await focusedRole(page)).toBe('treeitem');
  });

  test('menus and the section popover: focus moves in, Tab stays in, Esc closes and returns focus', async ({ page }) => {
    await keyboardImport(page);
    await tabTo(page, 'add-box');
    await press(page, 'Enter');
    await tabTo(page, 'add-load');
    await press(page, 'Enter');
    await expect.poll(() => state(page, (s) => s.project.loads.length)).toBe(1);

    const inside = (sel: string) => page.evaluate((s) => !!document.activeElement?.closest(s), sel);

    // context menu of a tree row: Shift+F10 (and the Menu key)
    await tabToTree(page);
    await treeDownTo(page, 'load-row');
    for (const key of ['Shift+F10', 'ContextMenu']) {
      await press(page, key);
      await expect(page.getByTestId('context-menu')).toBeVisible();
      expect(await focusedTestId(page)).toBe('ctx-rename');
      await press(page, 'ArrowDown');
      expect(await focusedTestId(page)).toBe('ctx-select');
      await press(page, 'End');
      expect(await focusedTestId(page)).toBe('ctx-delete');
      await press(page, 'ArrowDown'); // wraps
      expect(await focusedTestId(page)).toBe('ctx-rename');
      for (let i = 0; i < 8; i++) {
        await press(page, i % 3 ? 'Tab' : 'Shift+Tab');
        expect(await inside('[data-testid="context-menu"]')).toBe(true);
      }
      await press(page, 'Escape');
      await expect(page.getByTestId('context-menu')).toHaveCount(0);
      expect(await focusedTestId(page)).toBe('load-row');
    }
    // a menu command runs with Enter: Preview nodes, and focus comes back to the row
    await press(page, 'Shift+F10', 'ArrowDown', 'ArrowDown');
    expect(await focusedTestId(page)).toBe('ctx-preview');
    await press(page, 'Enter');
    await expect(page.getByTestId('preview-count')).toBeVisible();
    expect(await focusedTestId(page)).toBe('load-row');
    expect(await state(page, (s) => s.project.loads.length)).toBe(1); // Esc in the menu did not touch the project

    // standard views menu in the heads-up toolbar
    await tabTo(page, 'hud-views');
    await expect(page.getByTestId('hud-views')).toHaveAttribute('aria-expanded', 'false');
    await press(page, 'Enter');
    await expect(page.getByTestId('hud-views')).toHaveAttribute('aria-expanded', 'true');
    expect(await focusedTestId(page)).toBe('hud-view-iso');
    await press(page, 'ArrowRight', 'ArrowDown');
    expect(await focusedTestId(page)).toBe('hud-view-back');
    for (let i = 0; i < 9; i++) {
      await press(page, 'Tab');
      expect(await inside('.hud-views')).toBe(true);
    }
    await press(page, 'Escape');
    await expect(page.getByTestId('hud-view-iso')).toHaveCount(0);
    expect(await focusedTestId(page)).toBe('hud-views');
    await press(page, 'Enter', 'End', 'Enter'); // Bottom
    await expect.poll(async () => (await viewportStats(page)).viewDir[2]!).toBeLessThan(-0.999);
    expect(await focusedTestId(page)).toBe('hud-views');

    // section plane popover (a non-modal dialog)
    await tabTo(page, 'hud-section');
    await press(page, 'Enter');
    await expect(page.getByTestId('section-pop')).toBeVisible();
    await expect(page.getByRole('dialog', { name: 'Section plane' })).toBeVisible();
    expect(await focusedTestId(page)).toBe('section-axis-x');
    await press(page, 'Tab', 'Tab', 'Enter'); // Z
    await expect(page.getByTestId('section-axis-z')).toHaveAttribute('aria-pressed', 'true');
    await press(page, 'Tab', 'ArrowLeft', 'ArrowLeft');
    expect(await page.getByTestId('section-pos').inputValue()).toBe('0.48');
    for (let i = 0; i < 7; i++) {
      await press(page, 'Shift+Tab');
      expect(await inside('[data-testid="section-pop"]')).toBe(true);
    }
    await press(page, 'Escape');
    await expect(page.getByTestId('section-pop')).toHaveCount(0);
    expect(await focusedTestId(page)).toBe('hud-section');
    expect((await viewportStats(page)).section).toEqual({ axis: 'z', t: 0.48 });
    expect(await state(page, (s) => s.project.loads.length)).toBe(1);
  });

  for (const theme of ['light', 'dark'] as const) {
    test(`no keyboard trap; every tab stop shows a focus ring with 3:1 contrast (${theme})`, async ({ page }) => {
      test.setTimeout(90_000);
      await page.addInitScript((t) => localStorage.setItem('topop.ui.theme', t), theme);
      await keyboardImport(page);
      // a state with every control type on screen: a load in the tree, the pick tool's slider, the study page
      await tabTo(page, 'add-box');
      await press(page, 'Enter');
      await tabTo(page, 'add-load');
      await press(page, 'Enter');
      await tabTo(page, 'mode-pick');
      await press(page, 'Enter');
      await tabToTree(page);
      await treeDownTo(page, 'tree-study');
      await press(page, 'Enter');
      await expect(page.getByTestId('property-manager')).toHaveAttribute('data-page', 'study');

      const seen = new Map<string, number>();
      const bad: string[] = [];
      const visit = async (back: boolean) => {
        const first = await page.evaluate(() => {
          const el = document.activeElement as HTMLElement;
          el.dataset.a11yStart = '1';
          return el.tagName;
        });
        expect(first).not.toBe('BODY');
        for (let i = 0; i < 400; i++) {
          await press(page, back ? 'Shift+Tab' : 'Tab');
          const at = await page.evaluate(() => {
            const el = document.activeElement as HTMLElement | null;
            if (!el || el === document.body) return { body: true, start: false, kind: '' };
            const role = el.getAttribute('role');
            const kind = role ? `role=${role}` : el instanceof HTMLInputElement ? `input[${el.type}]` : el.tagName.toLowerCase();
            return { body: false, start: el.dataset.a11yStart === '1', kind };
          });
          if (at.start) {
            seen.set(at.kind, (seen.get(at.kind) ?? 0) + 1);
            return; // back where we started: the whole page is a cycle, nothing traps
          }
          if (at.body) continue;
          seen.set(at.kind, (seen.get(at.kind) ?? 0) + 1);
          const f = (await focusInfo(page))!;
          if (!f.visible || f.ratio < 3) bad.push(`${f.what}: visible=${f.visible} ratio=${f.ratio}`);
        }
        throw new Error(`Tab ${back ? 'backwards ' : ''}never came back to the start: a keyboard trap`);
      };
      await visit(false);
      await visit(true);
      expect([...seen.values()].reduce((a, b) => a + b, 0), 'tab stops checked').toBeGreaterThan(60);
      expect(bad, 'focus indicators').toEqual([]);
      for (const kind of ['button', 'input[number]', 'input[text]', 'input[checkbox]', 'input[range]', 'select', 'role=tab', 'role=treeitem'])
        expect(seen.get(kind) ?? 0, `${kind} in the tab order`).toBeGreaterThan(0);
    });
  }
});
