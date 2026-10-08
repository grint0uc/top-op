import { expect, test } from '@playwright/test';
import {
  canvasCentre,
  defineBoundaries,
  importBracket,
  selectFlatFaceAtCentre,
  state,
  viewportStats,
} from './helpers';

test.describe('frontend against the mock backend', () => {
  test('1. upload bracket.stl: design mesh in the store and the canvas renders it', async ({ page }) => {
    await importBracket(page);
    const s = await state(page, (st) => ({
      meshId: st.project.design_mesh?.mesh_id ?? null,
      loaded: Object.keys(st.meshes).length,
      faces: st.meshes[st.project.design_mesh?.mesh_id ?? '']?.info.n_faces ?? 0,
      watertight: st.meshes[st.project.design_mesh?.mesh_id ?? '']?.info.is_watertight ?? false,
    }));
    expect(s.meshId).toBeTruthy();
    expect(s.loaded).toBe(1);
    expect(s.faces).toBe(1326);
    expect(s.watertight).toBe(true);
    await expect(page.getByTestId('mesh-faces')).toHaveText('1,326');
    await expect(page.getByTestId('watertight-warning')).toHaveCount(0);
    // domain panel voxelizes through the server
    await expect(page.getByTestId('voxel-active')).toBeVisible();

    const coverage = await page.evaluate(() => window.__topopViewport!.coverage());
    expect(coverage, 'design mesh covers a visible part of the canvas').toBeGreaterThan(0.05);
    await page.screenshot({ path: 'test-results/viewport-bracket.png' });
  });

  test('1b. the mock serves examples/bracket.stl (Load example button)', async ({ page }) => {
    await page.goto('/');
    await page.getByTestId('load-example').click();
    await expect.poll(() => state(page, (s) => Object.values(s.meshes)[0]?.info.n_faces ?? 0)).toBe(1326);
  });

  test('2. pick: click selects a face, shift+click grows the flat face, ctrl+click removes', async ({ page }) => {
    await importBracket(page);
    await page.getByTestId('mode-pick').click();
    const { x, y } = await canvasCentre(page);

    await page.mouse.click(x, y);
    await expect.poll(() => state(page, (s) => s.selection.faceIds.length)).toBeGreaterThanOrEqual(1);
    const before = await state(page, (s) => s.selection.faceIds.length);

    await page.keyboard.down('Shift');
    await page.mouse.click(x, y);
    await page.keyboard.up('Shift');
    await expect.poll(() => state(page, (s) => s.selection.faceIds.length)).toBeGreaterThan(before);
    const grown = await state(page, (s) => s.selection.faceIds.length);

    // the selection is drawn by recolouring vertices in place
    expect(await page.evaluate(() => window.__topopViewport!.coverage())).toBeGreaterThan(0.05);

    await page.keyboard.down('Control');
    await page.mouse.click(x, y);
    await page.keyboard.up('Control');
    await expect.poll(() => state(page, (s) => s.selection.faceIds.length)).toBe(grown - 1);

    await page.keyboard.press('Escape');
    await expect.poll(() => state(page, (s) => s.selection.faceIds.length)).toBe(0);
  });

  test('2b. a drag (orbit) does not select, and keys 1-4 switch modes', async ({ page }) => {
    await importBracket(page);
    await page.keyboard.press('2');
    expect(await state(page, (s) => s.tool)).toBe('pick');
    const { x, y } = await canvasCentre(page);
    await page.mouse.move(x, y);
    await page.mouse.down();
    await page.mouse.move(x + 60, y + 20, { steps: 6 });
    await page.mouse.up();
    expect(await state(page, (s) => s.selection.faceIds.length)).toBe(0);
    await page.keyboard.press('3');
    expect(await state(page, (s) => s.tool)).toBe('paint');
    await page.keyboard.press('4');
    expect(await state(page, (s) => s.tool)).toBe('gizmo');
    await page.keyboard.press('1');
    expect(await state(page, (s) => s.tool)).toBe('orbit');
  });

  test('2c. paint mode adds faces under the brush, ctrl-drag removes them', async ({ page }) => {
    await importBracket(page);
    await page.getByTestId('mode-paint').click();
    const { x, y } = await canvasCentre(page);
    await page.mouse.move(x, y);
    await page.mouse.down();
    await page.mouse.move(x + 40, y + 10, { steps: 5 });
    await page.mouse.up();
    await expect.poll(() => state(page, (s) => s.selection.faceIds.length)).toBeGreaterThan(0);
    const painted = await state(page, (s) => s.selection.faceIds.length);

    await page.keyboard.down('Control');
    await page.mouse.move(x, y);
    await page.mouse.down();
    await page.mouse.move(x + 40, y + 10, { steps: 5 });
    await page.mouse.up();
    await page.keyboard.up('Control');
    await expect.poll(() => state(page, (s) => s.selection.faceIds.length)).toBeLessThan(painted);
  });

  test('3. add a load from the selection, set its force, resolve preview adds markers', async ({ page }) => {
    await importBracket(page);
    await selectFlatFaceAtCentre(page);
    await page.getByTestId('add-load').click();
    await expect.poll(() => state(page, (s) => s.project.loads.length)).toBe(1);
    expect(await state(page, (s) => s.selection.faceIds.length)).toBe(0);

    const row = page.getByTestId('load-row').first();
    await row.getByTestId('force-x').fill('10');
    await row.getByTestId('force-z').fill('-250');
    await expect.poll(() => state(page, (s) => s.project.loads[0]?.force)).toEqual([10, 0, -250]);
    const load = await state(page, (s) => s.project.loads[0]!);
    expect(load.selection.kind).toBe('faces');
    expect(load.case).toBe(0);

    expect((await viewportStats(page)).arrows).toBe(1);
    expect((await viewportStats(page)).resolvedPoints).toBe(0);
    await page.getByTestId('resolve-preview').click();
    await expect(page.getByTestId('preview-count')).toHaveText(/^\d+ nodes/);
    await expect.poll(async () => (await viewportStats(page)).resolvedPoints).toBeGreaterThan(0);
    expect(await state(page, (s) => s.preview?.count ?? 0)).toBeGreaterThan(100);
  });

  test('4. a box primitive becomes a support, with numeric fields bound to the gizmo state', async ({ page }) => {
    await importBracket(page);
    await page.getByTestId('add-box').click();
    await expect(page.getByTestId('prim-fields')).toBeVisible();
    expect(await state(page, (s) => s.selection.primitive?.kind)).toBe('box');
    expect(await state(page, (s) => s.tool)).toBe('gizmo');

    await page.getByTestId('prim-pos-x').fill('12.5');
    await page.getByTestId('prim-size-sx').fill('7');
    await page.getByTestId('prim-rot-z').fill('30');
    const prim = await state(page, (s) => s.selection.primitive!);
    expect(prim.transform).toHaveLength(16);
    expect(prim.transform[12]).toBeCloseTo(12.5);
    expect(prim.size[0]).toBeCloseTo(7);
    expect(prim.transform[0]).toBeCloseTo(Math.cos(Math.PI / 6)); // rotation about Z, column-major

    // store -> fields
    await page.evaluate(() => {
      const st = window.__topop!.getState();
      const p = st.selection.primitive!;
      const t = [...p.transform];
      t[13] = 33;
      st.setPrimitive({ ...p, transform: t });
    });
    await expect(page.getByTestId('prim-pos-y')).toHaveValue('33');

    await page.getByTestId('add-support').click();
    await expect.poll(() => state(page, (s) => s.project.supports.length)).toBe(1);
    const sup = await state(page, (s) => s.project.supports[0]!);
    expect(sup.selection.kind).toBe('box');
    expect(sup.fix).toEqual([true, true, true]);
    expect(await state(page, (s) => s.selection.primitive)).toBeNull();
    await expect.poll(async () => (await viewportStats(page)).glyphs).toBe(1);

    // fix checkboxes edit the support
    await page.getByTestId('support-row').first().getByTestId('fix-z').uncheck();
    await expect.poll(() => state(page, (s) => s.project.supports[0]?.fix)).toEqual([true, true, false]);
  });

  test('4b. dragging the gizmo moves the primitive and keys g/r/s change the gizmo mode', async ({ page }) => {
    await importBracket(page);
    await page.getByTestId('add-box').click();
    const before = await state(page, (s) => s.selection.primitive!.transform.slice(12, 15));
    const at = await page.evaluate((p) => window.__topopViewport!.screenPoint(p as [number, number, number]), before);
    await page.mouse.move(at.x, at.y);
    await page.mouse.down();
    await page.mouse.move(at.x + 80, at.y + 30, { steps: 8 });
    await page.mouse.up();
    const after = await state(page, (s) => s.selection.primitive!.transform.slice(12, 15));
    expect(Math.hypot(...after.map((v, i) => v - before[i]!))).toBeGreaterThan(1);
    // numeric field follows the drag
    await expect(page.getByTestId('prim-pos-x')).toHaveValue(String(Math.round(after[0]! * 1e6) / 1e6));

    await page.keyboard.press('r');
    expect(await state(page, (s) => s.gizmoMode)).toBe('rotate');
    await page.keyboard.press('s');
    expect(await state(page, (s) => s.gizmoMode)).toBe('scale');
    await page.keyboard.press('g');
    expect(await state(page, (s) => s.gizmoMode)).toBe('translate');
  });

  test('4c. Delete removes the active load, Esc clears the preview', async ({ page }) => {
    await importBracket(page);
    await selectFlatFaceAtCentre(page);
    await page.getByTestId('add-load').click();
    expect(await state(page, (s) => s.activeItem?.kind)).toBe('load');
    await page.getByTestId('resolve-preview').click();
    await expect.poll(() => state(page, (s) => s.preview?.count ?? 0)).toBeGreaterThan(0);
    await page.keyboard.press('Escape');
    expect(await state(page, (s) => s.preview)).toBeNull();
    // Esc also deselects, so re-activate the row before deleting
    await page.getByTestId('load-row').first().click();
    await page.keyboard.press('Delete');
    await expect.poll(() => state(page, (s) => s.project.loads.length)).toBe(0);
    await expect.poll(async () => (await viewportStats(page)).arrows).toBe(0);
  });

  test('5. run 8 iterations: progress records, a density frame, DensityView instances, done', async ({ page }) => {
    await importBracket(page);
    await defineBoundaries(page);
    await page.getByTestId('domain-elements').fill('24');
    await page.getByTestId('p-max-iter').fill('8');
    await page.getByTestId('run-start').click();

    await expect.poll(() => state(page, (s) => s.run.history.length), { timeout: 15_000 }).toBeGreaterThanOrEqual(8);
    await expect.poll(() => state(page, (s) => s.run.status), { timeout: 15_000 }).toBe('done');
    const run = await state(page, (s) => ({
      frames: s.run.densityFrames,
      its: s.run.history.map((r) => r.it),
      c: s.run.history.map((r) => r.compliance),
      shape: s.run.densityFrame?.shape,
    }));
    expect(run.frames).toBeGreaterThanOrEqual(1);
    expect(run.its.slice(0, 8)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(run.c[7]).toBeLessThan(run.c[0]!);
    expect(run.shape).toBeTruthy();

    const stats = await viewportStats(page);
    expect(stats.densityCount).toBeGreaterThan(0);
    expect(stats.densityMode).toBe('instanced');
    await expect(page.getByTestId('density-info')).toContainText('cells');
    await expect(page.getByTestId('run-iter')).toContainText('it 8 / 8');
    expect(await page.getByTestId('sparkline').getAttribute('data-points')).toBe('8');

    // threshold slider re-thresholds the same frame: fewer cells at 0.9 than at 0.1
    await page.getByTestId('threshold').fill('0.1');
    const low = (await viewportStats(page)).densityCount;
    await page.getByTestId('threshold').fill('0.9');
    const high = (await viewportStats(page)).densityCount;
    expect(low).toBeGreaterThan(high);
    await page.screenshot({ path: 'test-results/viewport-density.png' });
  });

  test('5b. stop cancels a running job', async ({ page }) => {
    await importBracket(page);
    await defineBoundaries(page);
    await page.getByTestId('domain-elements').fill('24');
    await page.getByTestId('p-max-iter').fill('200');
    await page.getByTestId('run-start').click();
    await expect.poll(() => state(page, (s) => s.run.history.length), { timeout: 10_000 }).toBeGreaterThanOrEqual(2);
    await page.getByTestId('run-stop').click();
    await expect.poll(() => state(page, (s) => s.run.status)).toBe('cancelled');
    // records already in flight may still arrive; after that the history must stop growing
    let last = -1;
    await expect
      .poll(async () => {
        const n = await state(page, (s) => s.run.history.length);
        const settled = n === last;
        last = n;
        return settled;
      }, { intervals: [500] })
      .toBe(true);
    expect(await state(page, (s) => s.run.history.length)).toBeLessThan(200);
  });

  test('6. result downloads respond and the result mesh loads', async ({ page }) => {
    await importBracket(page);
    await defineBoundaries(page);
    await page.getByTestId('domain-elements').fill('24');
    await page.getByTestId('p-max-iter').fill('3');
    await page.getByTestId('run-start').click();
    await expect.poll(() => state(page, (s) => s.run.status), { timeout: 15_000 }).toBe('done');

    const stl = page.getByTestId('download-stl');
    const href = await stl.getAttribute('href');
    expect(href).toMatch(/\/api\/runs\/.+\/result\.stl\?threshold=0\.5&smooth=3/);
    const res = await page.request.get(href!);
    expect(res.ok()).toBe(true);
    expect((await res.body()).length).toBeGreaterThan(84);
    for (const id of ['download-vti', 'download-npz', 'download-project']) {
      const r = await page.request.get((await page.getByTestId(id).getAttribute('href'))!);
      expect(r.ok(), id).toBe(true);
    }
    const exported = await (await page.request.get((await page.getByTestId('download-project').getAttribute('href'))!)).json();
    expect(exported.project.loads).toHaveLength(1);
    expect(exported.run.status).toBe('done');

    await page.getByTestId('load-result').click();
    await expect.poll(async () => (await viewportStats(page)).result).toBe(true);
    await page.screenshot({ path: 'test-results/viewport-result.png' });
  });

  test('7. the project survives a reload (geometry must be re-uploaded)', async ({ page }) => {
    await importBracket(page);
    await defineBoundaries(page);
    await page.getByTestId('project-name').fill('my bracket');
    await page.waitForTimeout(500); // persistence is debounced
    await page.reload();
    await expect(page.getByTestId('notice')).toContainText('re-upload');
    expect(await state(page, (s) => s.project.name)).toBe('my bracket');
    expect(await state(page, (s) => s.project.loads.length)).toBe(1);
    expect(await state(page, (s) => Object.keys(s.meshes).length)).toBe(0);
    await expect(page.getByTestId('reupload-notice')).toContainText('bracket.stl');

    await page.getByTestId('mesh-file-input').setInputFiles(
      (await import('./helpers')).BRACKET,
    );
    await expect.poll(() => state(page, (s) => Object.keys(s.meshes).length)).toBe(1);
    // same mesh -> same id, loads kept and still drawn on the faces
    expect(await state(page, (s) => s.project.loads.length)).toBe(1);
    expect((await viewportStats(page)).arrows).toBe(1);
  });

  test('8. a reference model can be added, switched to keep-in and moved with the gizmo state', async ({ page }) => {
    await importBracket(page);
    await page.getByTestId('ref-file-input').setInputFiles((await import('./helpers')).BRACKET);
    await expect.poll(() => state(page, (s) => s.project.ref_models.length)).toBe(1);
    await expect.poll(async () => (await viewportStats(page)).refs).toBe(1);
    expect(await state(page, (s) => s.tool)).toBe('gizmo');
    await page.getByTestId('ref-mode').selectOption('keep_in');
    expect(await state(page, (s) => s.project.ref_models[0]?.mode)).toBe('keep_in');
    await page.keyboard.press('r');
    expect(await state(page, (s) => s.gizmoMode)).toBe('rotate');
    await page.keyboard.press('g');
    // drag the ref model with the gizmo: its transform in the store changes
    const refTransform = () => state(page, (s) => (s.project.ref_models[0]!.transform ?? []).slice(12, 15));
    const t0 = await refTransform();
    // the gizmo sits on the geometry (transform * bbox centre), not at the mesh origin
    const bboxCentre = await state(page, (s) => {
      const b = Object.values(s.meshes)[0]!.data.bbox;
      return [0, 1, 2].map((k) => (b.min[k]! + b.max[k]!) / 2);
    });
    const at = await page.evaluate((p) => window.__topopViewport!.screenPoint(p as [number, number, number]), bboxCentre);
    await page.mouse.move(at.x, at.y);
    await page.mouse.down();
    await page.mouse.move(at.x + 60, at.y + 20, { steps: 6 });
    await page.mouse.up();
    const t1 = await refTransform();
    expect(Math.hypot(...t1.map((v, i) => v - (t0[i] ?? 0)))).toBeGreaterThan(0.5);
    await page.getByTestId('ref-visible').uncheck();
    expect(await state(page, (s) => s.project.ref_models[0]?.visible)).toBe(false);
    await page.getByTestId('ref-delete').click();
    await expect.poll(async () => (await viewportStats(page)).refs).toBe(0);
  });

  test('9. domain warnings: amber above 150k active elements, red above 300k', async ({ page }) => {
    await importBracket(page);
    const set = (v: string) => page.getByTestId('domain-elements').fill(v);
    await set('60');
    await expect(page.getByTestId('voxel-stats')).toHaveAttribute('data-level', 'ok');
    await set('110');
    await expect(page.getByTestId('voxel-stats')).toHaveAttribute('data-level', 'amber');
    await expect(page.getByTestId('voxel-warning')).toBeVisible();
    await set('150');
    await expect(page.getByTestId('voxel-stats')).toHaveAttribute('data-level', 'red');
    await expect(page.getByTestId('voxel-memory')).toContainText('GB');
  });
});
