// Query selections, project import, queueing and generated keep-out volumes, against the mock backend (project `chromium`).
// The same flows run against the real server in real.spec.ts.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { BRACKET, defineBoundaries, importBracket, state, viewportStats } from './helpers';

const BRACKET_ID = createHash('sha256').update(readFileSync(BRACKET)).digest('hex').slice(0, 16);

test.describe('query selections (mock backend)', () => {
  test('normal form: presets, free vector, angle and a padded within box build a NormalSelection', async ({ page }) => {
    await importBracket(page);
    await page.keyboard.press('5');
    await expect(page.getByTestId('query-panel')).toBeVisible();
    // opening the tab selects nothing; the first edit does
    expect(await state(page, (s) => s.selection.query)).toBeNull();

    await page.getByTestId('normal-dir--Z').click();
    expect(await state(page, (s) => s.selection.query)).toEqual({
      kind: 'normal',
      mesh_id: BRACKET_ID,
      direction: [0, 0, -1],
      angle_deg: 10,
      within: null,
    });
    await page.getByTestId('normal-angle').fill('15');
    await page.getByTestId('normal-vec-x').fill('1');
    const sel = await state(page, (s) => s.selection.query as { direction: number[]; angle_deg: number });
    expect(sel.direction).toEqual([1, 0, -1]);
    expect(sel.angle_deg).toBe(15);
    await expect(page.getByTestId('query-status')).toContainText('normal [1, 0, -1] ±15°');

    // within: pre-filled from the design bbox grown by one voxel, with the h/2 explanation
    await expect(page.getByTestId('normal-within')).toHaveCount(0);
    await page.getByTestId('normal-within-toggle').check();
    const { bbox, h } = await state(page, (s) => {
      const info = s.meshes[s.project.design_mesh!.mesh_id!]!.info;
      const b = info.bbox;
      return { bbox: b, h: Math.max(b[1]![0]! - b[0]![0]!, b[1]![1]! - b[0]![1]!, b[1]![2]! - b[0]![2]!) / s.project.grid.elements_along_longest };
    });
    const within = await state(page, (s) => (s.selection.query as { within: number[][] }).within);
    for (let k = 0; k < 3; k++) {
      expect(within[0]![k]!).toBeCloseTo(bbox[0]![k]! - h, 3);
      expect(within[1]![k]!).toBeCloseTo(bbox[1]![k]! + h, 3);
    }
    await expect(page.getByTestId('within-hint')).toContainText('h/2');
    await page.getByTestId('within-max-z').fill('20');
    expect(await state(page, (s) => (s.selection.query as { within: number[][] }).within[1]![2])).toBe(20);

    // resolve preview works for a query selection, and the selection becomes a load labelled readably
    await page.getByTestId('resolve-preview').click();
    await expect.poll(() => state(page, (s) => s.preview?.count ?? 0)).toBeGreaterThan(0);
    await page.getByTestId('add-load').click();
    await expect(page.getByTestId('load-row').first()).toContainText('normal [1, 0, -1] ±15° in box');
    expect(await state(page, (s) => s.selection.query)).toBeNull(); // consumed, like a primitive
    expect(await state(page, (s) => s.project.loads[0]!.selection.kind)).toBe('normal');

    // an invalid form (zero direction) leaves no selection
    await page.getByTestId('normal-vec-x').fill('0');
    await page.getByTestId('normal-vec-z').fill('0');
    expect(await state(page, (s) => s.selection.query)).toBeNull();
    await expect(page.getByTestId('query-apply')).toBeDisabled();
    await expect(page.getByTestId('query-status')).toHaveText('incomplete');

    // "Select" on the load brings its query back into the form
    await page.getByTestId('load-row').first().getByTestId('item-select').click();
    expect(await state(page, (s) => s.tool)).toBe('query');
    await expect(page.getByTestId('normal-vec-x')).toHaveValue('1');
    expect(await state(page, (s) => s.selection.query)).toEqual(await state(page, (s) => s.project.loads[0]!.selection));
  });

  test('facets form: top facets by area, hover highlights their faces, rows build a FacetSelection', async ({ page }) => {
    await importBracket(page);
    await page.getByTestId('mode-query').click();
    await page.getByTestId('query-tab-facets').click();
    await expect(page.getByTestId('facet-list')).toBeVisible();
    const rows = page.getByTestId('facet-row');
    expect(await rows.count()).toBe(12);
    const areas = await rows.evaluateAll((els) => els.map((e) => Number(e.querySelector('.facet-area')!.textContent)));
    expect([...areas].sort((a, b) => b - a)).toEqual(areas); // area-sorted, largest first
    await expect(page.getByTestId('facets-summary')).toContainText('facets at 5');

    await rows.nth(1).hover();
    const facet = await state(page, (s) => Object.values(s.facetCache)[0]!.facets[1]!);
    await expect.poll(() => state(page, (s) => s.hoverFaces.length)).toBe(facet.n_faces);
    await page.mouse.move(900, 600);
    await expect.poll(() => state(page, (s) => s.hoverFaces.length)).toBe(0);

    await rows.nth(0).click();
    await rows.nth(2).click();
    const ids = await state(page, (s) => (s.selection.query as { facet_ids: number[] }).facet_ids);
    expect(ids).toEqual([0, 2]);
    await expect(rows.nth(0)).toHaveClass(/picked/);
    await rows.nth(0).click(); // toggles off
    expect(await state(page, (s) => (s.selection.query as { facet_ids: number[] }).facet_ids)).toEqual([2]);
    await rows.nth(0).click();

    await page.getByTestId('resolve-preview').click();
    await expect.poll(() => state(page, (s) => s.preview?.count ?? 0)).toBeGreaterThan(0);
    await page.getByTestId('add-support').click();
    await expect(page.getByTestId('support-row').first()).toContainText('facets #2, #0 (5°)');
    await expect.poll(async () => (await viewportStats(page)).glyphs).toBe(1);

    // another angle is another table: the chosen ids no longer apply
    await page.getByTestId('facets-angle').fill('20');
    expect(await state(page, (s) => s.selection.query)).toBeNull();
    await page.getByTestId('facets-fetch').click();
    await expect(page.getByTestId('facets-summary')).toContainText('at 20');
    await page.getByTestId('facets-more').click();
    expect(await rows.count()).toBeGreaterThan(12);
  });

  test('plane form: point, normal presets and tolerance build a PlaneSelection', async ({ page }) => {
    await importBracket(page);
    await page.getByTestId('mode-query').click();
    await page.getByTestId('query-tab-plane').click();
    await page.getByTestId('plane-point-z').fill('60');
    await page.getByTestId('plane-dir-+Z').click();
    await page.getByTestId('plane-tol').fill('1.5');
    const sel = await state(page, (s) => s.selection.query);
    expect(sel).toMatchObject({ kind: 'plane', normal: [0, 0, 1], tol: 1.5 });
    expect((sel as { point: number[] }).point[2]).toBe(60);
    await expect(page.getByTestId('query-status')).toContainText('plane z=60 ±1.5');
    await page.getByTestId('resolve-preview').click();
    await expect.poll(() => state(page, (s) => s.preview?.count ?? 0)).toBeGreaterThan(0);
    await page.getByTestId('add-support').click();
    await expect(page.getByTestId('support-row').first()).toContainText('plane z=60');
    await expect.poll(async () => (await viewportStats(page)).glyphs).toBe(1); // anchored at the plane point
  });
});

test.describe('project import and run handling (mock backend)', () => {
  test('a run that cannot start (422 not runnable) is explained in the Run panel', async ({ page }) => {
    await importBracket(page);
    await page.getByTestId('run-start').click();
    await expect(page.getByTestId('run-error')).toHaveText('project is not runnable: no loads defined; no supports defined');
    await expect(page.getByTestId('run-status')).toHaveText('idle');
    await expect(page.getByTestId('run-start')).toBeEnabled();
  });

  test('exports before completion answer 409', async ({ page }) => {
    await importBracket(page);
    await defineBoundaries(page);
    await page.getByTestId('domain-elements').fill('24');
    await page.getByTestId('p-max-iter').fill('200');
    await page.getByTestId('run-start').click();
    await expect.poll(() => state(page, (s) => s.run.history.length)).toBeGreaterThan(0);
    const id = await state(page, (s) => s.run.id);
    for (const ext of ['stl', 'vti', 'npz']) {
      const res = await page.request.get(`/api/runs/${id}/result.${ext}`);
      expect(res.status(), ext).toBe(409);
      expect((await res.json()).detail).toContain('no result to export');
    }
    await page.getByTestId('run-stop').click();
    await expect.poll(() => state(page, (s) => s.run.status)).toBe('cancelled');
  });

  test('a second run waits in the queue: queued, then running once the first one stops', async ({ page, browser }) => {
    const baseURL = String(test.info().project.use.baseURL);
    await importBracket(page);
    await defineBoundaries(page);
    await page.getByTestId('domain-elements').fill('24');
    await page.getByTestId('p-max-iter').fill('300');
    await page.getByTestId('run-start').click();
    await expect.poll(() => state(page, (s) => s.run.history.length)).toBeGreaterThan(1);

    const other = await (await browser.newContext({ baseURL })).newPage();
    try {
      await importBracket(other);
      await defineBoundaries(other);
      await other.getByTestId('domain-elements').fill('24');
      await other.getByTestId('p-max-iter').fill('3');
      await other.evaluate(() => {
        const w = window as unknown as { __statuses: string[] };
        w.__statuses = [];
        window.__topop!.subscribe((s, p) => {
          if (s.run.status !== p.run.status) w.__statuses.push(s.run.status);
        });
      });
      await other.getByTestId('run-start').click();
      await expect(other.getByTestId('run-status')).toHaveText('queued');
      await other.waitForTimeout(800);
      expect(await state(other, (s) => [s.run.status, s.run.history.length])).toEqual(['queued', 0]);

      await page.getByTestId('run-stop').click(); // frees the slot
      await expect.poll(() => state(other, (s) => s.run.status), { timeout: 20_000 }).toBe('done');
      expect(await other.evaluate(() => (window as unknown as { __statuses: string[] }).__statuses)).toEqual(['queued', 'running', 'done']);
      expect(await state(other, (s) => s.run.history.length)).toBe(3);
      expect(await state(other, (s) => s.run.stats?.origin.length)).toBe(3);
    } finally {
      await other.context().close();
    }
  });

  test('Load project.json: restores the document, fetches known meshes by id, shows the run history', async ({ page }) => {
    await importBracket(page); // the mock now knows the mesh
    await page.getByTestId('new-project').click();
    expect(await state(page, (s) => s.project.design_mesh)).toBeNull();

    const loadSel = { kind: 'normal', mesh_id: BRACKET_ID, direction: [0, 0, 1], angle_deg: 10, within: [[-2, -2, 50], [12, 62, 63]] };
    const exported = {
      project: {
        id: 'proj-x',
        created_at: 'x',
        updated_at: 'x',
        name: 'from file',
        design_mesh: { mesh_id: BRACKET_ID, transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] },
        grid: { elements_along_longest: 30, padding: 1 },
        params: { max_iter: 5, volfrac: 0.4 },
        loads: [{ id: 'l1', name: 'top', selection: loadSel, force: [0, 0, -5], case: 1 }],
        supports: [{ id: 's1', name: 'base', selection: { kind: 'plane', point: [0, 0, 0], normal: [0, 0, 1], tol: 0 }, fix: [true, true, false] }],
      },
      run: {
        id: 'run-from-file',
        project_id: 'proj-x',
        status: 'done',
        created_at: '2026-01-01T00:00:00Z',
        finished_at: '2026-01-01T00:01:00Z',
        history: [1, 2, 3, 4, 5].map((it) => ({ it, compliance: 100 / it, volume: 0.4, change: 0.1 / it, t_iter: 0.1 })),
        stats: null,
        error: null,
      },
    };
    await page.getByTestId('project-file-input').setInputFiles({ name: 'p.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(exported)) });
    await expect.poll(() => state(page, (s) => s.project.name)).toBe('from file');
    const doc = await state(page, (s) => s.project);
    expect(doc.grid).toEqual({ elements_along_longest: 30, padding: 1 });
    expect(doc.params).toMatchObject({ max_iter: 5, volfrac: 0.4, penal: 3, density_every: 1 }); // defaults filled in
    expect(doc.loads[0]).toEqual({ id: 'l1', name: 'top', selection: loadSel, force: [0, 0, -5], case: 1 });
    expect(doc.supports[0]!.fix).toEqual([true, true, false]);

    // the mock (like the server) still has the mesh: loaded by id, no re-upload prompt
    await expect.poll(async () => (await viewportStats(page)).designFaces).toBe(1326);
    await expect(page.getByTestId('reupload-notice')).toHaveCount(0);
    await expect(page.getByTestId('loaded-run')).toBeVisible();
    await expect(page.getByTestId('loaded-run-iters')).toHaveText('5');
    await expect(page.getByTestId('loaded-run-gone')).toBeVisible(); // this run id is not on the server
    await expect(page.getByTestId('load-row').first()).toContainText('normal +Z ±10° in box');
    await expect(page.getByTestId('support-row').first()).toContainText('plane z=0');
    expect((await viewportStats(page)).arrows).toBe(1);
    await expect(page.getByTestId('voxel-active')).toBeVisible();
  });

  test('Load project.json with a mesh nobody has: re-upload prompt; the wrong file is detected, the right one restores', async ({ page }) => {
    const unknown = 'f00dfeedf00dfeed';
    const make = (meshId: string) => ({
      name: 'needs upload',
      design_mesh: { mesh_id: meshId },
      loads: [{ id: 'l1', selection: { kind: 'normal', mesh_id: meshId, direction: [0, 0, 1], angle_deg: 10 }, force: [0, 0, -1] }],
      supports: [{ id: 's1', selection: { kind: 'facets', mesh_id: meshId, facet_ids: [0], angle_deg: 5 } }],
    });
    await page.goto('/');
    await page.getByTestId('project-file-input').setInputFiles({ name: 'p.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(make(unknown))) });
    await expect(page.getByTestId('reupload-notice')).toBeVisible();
    const row = page.getByTestId('required-mesh');
    await expect(row).toHaveCount(1);
    await expect(row).toHaveAttribute('data-mesh-id', unknown);
    expect(await state(page, (s) => [s.project.loads.length, s.project.supports.length])).toEqual([1, 1]);

    // "Fetch from server": the server does not have it either
    await page.getByTestId('fetch-from-server').click();
    await expect(page.getByTestId('notice')).toContainText('no longer has');

    // a different file: the facets support (indexes the other mesh's faces) is dropped, the normal load follows
    await row.getByTestId('reupload-input').setInputFiles(BRACKET);
    await expect.poll(() => state(page, (s) => s.project.design_mesh?.mesh_id)).toBe(BRACKET_ID);
    await expect(page.getByTestId('notice')).toContainText('not the file the project was made with');
    expect(await state(page, (s) => [s.project.loads.length, s.project.supports.length])).toEqual([1, 0]);
    expect(await state(page, (s) => (s.project.loads[0]!.selection as { mesh_id: string }).mesh_id)).toBe(BRACKET_ID);
    await expect(page.getByTestId('reupload-notice')).toHaveCount(0);
  });

  test('a restored project: re-uploading the same file keeps every selection (ids are content hashes)', async ({ page }) => {
    await importBracket(page);
    await defineBoundaries(page);
    const before = await state(page, (s) => JSON.stringify([s.project.loads, s.project.supports]));
    await page.waitForTimeout(500);
    await page.reload();
    await expect(page.getByTestId('reupload-notice')).toBeVisible();
    await page.getByTestId('mesh-file-input').setInputFiles(BRACKET);
    await expect.poll(() => state(page, (s) => Object.keys(s.meshes).length)).toBe(1);
    expect(await state(page, (s) => JSON.stringify([s.project.loads, s.project.supports]))).toBe(before);
    await expect(page.getByTestId('reupload-notice')).toHaveCount(0);
    expect((await viewportStats(page)).arrows).toBe(1);

    // and "Fetch from server" gets the mesh back without a file
    await page.waitForTimeout(500);
    await page.reload();
    await page.getByTestId('fetch-from-server').click();
    await expect.poll(() => state(page, (s) => Object.keys(s.meshes).length)).toBe(1);
    await expect(page.getByTestId('reupload-notice')).toHaveCount(0);
  });

  test('a file that is not a project is refused with a message', async ({ page }) => {
    await page.goto('/');
    await page.getByTestId('project-file-input').setInputFiles({ name: 'x.json', mimeType: 'application/json', buffer: Buffer.from('{"hello": 1}') });
    await expect(page.getByTestId('notice')).toContainText('not a top-op project');
    await page.getByTestId('project-file-input').setInputFiles({
      name: 'case.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify({ design_mesh: { path: 'examples/bracket.stl' }, loads: [] })),
    });
    await expect(page.getByTestId('notice')).toContainText('disk path');
  });
});

test.describe('generated keep-out volumes (mock backend)', () => {
  test('+ cylinder uploads a unit cylinder; the transform matrix carries translation, rotation and scale', async ({ page }) => {
    await importBracket(page);
    await page.getByTestId('add-ref-cylinder').click();
    await expect.poll(() => state(page, (s) => s.project.ref_models.length)).toBe(1);
    await expect.poll(async () => (await viewportStats(page)).refs).toBe(1);
    const info = await state(page, (s) => Object.values(s.meshes).find((m) => m.info.name.startsWith('keep-out'))!.info);
    expect(info.is_watertight).toBe(true);
    expect(info.bbox).toEqual([
      [-1, -0.5, -1],
      [1, 0.5, 1],
    ]); // radius 1, height 1, axis Y

    await page.getByTestId('ref-scale-x').fill('4');
    await page.getByTestId('ref-scale-y').fill('30');
    await page.getByTestId('ref-scale-z').fill('4');
    await page.getByTestId('ref-rot-x').fill('90');
    await page.getByTestId('ref-pos-x').fill('50');
    await page.getByTestId('ref-pos-y').fill('30');
    await page.getByTestId('ref-pos-z').fill('5');
    const m = await state(page, (s) => s.project.ref_models[0]!.transform!);
    [4, 0, 0, 0, 0, 0, 30, 0, 0, -4, 0, 0, 50, 30, 5, 1].forEach((v, i) => expect(m[i]!).toBeCloseTo(v, 6));
    expect(await state(page, (s) => s.project.design_mesh!.transform)).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

    // the gizmo path writes the same kind of matrix: dragging keeps the scale part
    await page.keyboard.press('g');
    const bboxCentre = [50, 30, 5];
    const at = await page.evaluate((p) => window.__topopViewport!.screenPoint(p as [number, number, number]), bboxCentre);
    await page.mouse.move(at.x, at.y);
    await page.mouse.down();
    await page.mouse.move(at.x + 60, at.y + 20, { steps: 6 });
    await page.mouse.up();
    const m2 = await state(page, (s) => s.project.ref_models[0]!.transform!);
    expect(Math.hypot(m2[12]! - 50, m2[13]! - 30, m2[14]! - 5)).toBeGreaterThan(0.5);
    expect(Math.hypot(m2[0]!, m2[1]!, m2[2]!)).toBeCloseTo(4, 5); // x scale untouched
    expect(Math.hypot(m2[4]!, m2[5]!, m2[6]!)).toBeCloseTo(30, 5);
    expect(Math.hypot(m2[8]!, m2[9]!, m2[10]!)).toBeCloseTo(4, 5);
  });
});
