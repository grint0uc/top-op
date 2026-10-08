// Contract v0.2 features against the mock backend (project `chromium`): optimizer/symmetry/stress/overhang params,
// stress colouring, trim to CAD, facet kinds and exact faces, design mesh transform, STEP import.
// The same stages run against the real server in real.spec.ts.
import { readFileSync } from 'node:fs';
import { type Page, expect, test } from '@playwright/test';
import { BRACKET_STEP, defineBoundaries, importBracket, runBracket, state, viewportStats } from './helpers';

type Body = { method: string; url: string; json: Record<string, any> }; // eslint-disable-line @typescript-eslint/no-explicit-any -- request bodies are inspected loosely

/** Records every POST/PUT of the project document. */
function recordProjectWrites(page: Page): Body[] {
  const out: Body[] = [];
  page.on('request', (r) => {
    if ((r.method() === 'POST' || r.method() === 'PUT') && /\/api\/projects(\/[^/]+)?$/.test(r.url())) {
      out.push({ method: r.method(), url: r.url(), json: r.postDataJSON() });
    }
  });
  return out;
}

const MATRIX_ID = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

test.describe('run parameters (mock backend)', () => {
  test('optimizer, symmetry planes, stress limit, p-norm and overhang round-trip through the PUT body and a project.json', async ({ page }) => {
    const writes = recordProjectWrites(page);
    await importBracket(page);
    await defineBoundaries(page);

    // optimizer: oc/mma; a stress limit forces mma and says so, without touching the stored choice
    const opt = page.getByTestId('p-optimizer');
    await expect(opt).toBeEnabled();
    await opt.selectOption('mma');
    expect(await state(page, (s) => s.project.params.optimizer)).toBe('mma');
    await opt.selectOption('oc');
    await page.getByTestId('p-stress-limit').fill('2.5');
    await expect(opt).toBeDisabled();
    await expect(opt.locator('option:checked')).toHaveText('mma (forced by stress limit)');
    expect(await state(page, (s) => s.project.params.optimizer)).toBe('oc');
    await page.getByTestId('p-stress-limit').fill(''); // empty = off
    expect(await state(page, (s) => s.project.params.stress_limit)).toBeNull();
    await expect(opt).toBeEnabled();
    await expect(opt.locator('option:checked')).toHaveText('oc');
    await opt.selectOption('mma');
    await expect(opt.locator('option:checked')).toHaveText('mma');
    await page.getByTestId('p-stress-limit').fill('2.5');
    await expect(opt.locator('option:checked')).toHaveText('mma (forced by stress limit)');
    await page.getByTestId('p-stress-pnorm').fill('12');
    await page.getByTestId('p-overhang').selectOption('+z');

    // symmetry rows: add / axis / center vs numeric position / remove
    await expect(page.getByTestId('sym-row')).toHaveCount(0);
    await page.getByTestId('sym-add').click();
    await page.getByTestId('sym-add').click();
    await page.getByTestId('sym-add').click();
    const rows = page.getByTestId('sym-row');
    await expect(rows).toHaveCount(3);
    await expect(rows.nth(0).getByTestId('sym-center')).toBeChecked();
    await expect(rows.nth(0).getByTestId('sym-pos')).toHaveCount(0);
    await rows.nth(1).getByTestId('sym-axis').selectOption('y');
    await rows.nth(1).getByTestId('sym-center').uncheck();
    await expect(rows.nth(1).getByTestId('sym-pos')).toHaveValue('30'); // starts at the middle of the part
    await rows.nth(1).getByTestId('sym-pos').fill('12.5');
    await rows.nth(2).getByTestId('sym-axis').selectOption('z');
    await rows.nth(2).getByTestId('sym-remove').click();
    await expect(rows).toHaveCount(2);

    await page.getByTestId('p-max-iter').fill('3');
    await page.getByTestId('domain-elements').fill('24');
    await page.getByTestId('run-start').click();
    await expect.poll(() => state(page, (s) => s.run.status), { timeout: 30_000 }).toBe('done');

    const body = writes.at(-1)!.json;
    expect(body.params).toMatchObject({
      optimizer: 'mma',
      symmetry: [
        { axis: 'x', position: null },
        { axis: 'y', position: 12.5 },
      ],
      stress_limit: 2.5,
      stress_pnorm: 12,
      overhang: '+z',
      max_iter: 3,
    });
    // the server keeps exactly what was sent, and the run export carries it
    const projectId = await state(page, (s) => s.projectMeta!.id);
    expect((await (await page.request.get(`/api/projects/${projectId}`)).json()).params).toMatchObject(body.params);
    const exported = await (await page.request.get((await page.getByTestId('download-project').getAttribute('href'))!)).json();
    expect(exported.project.params).toMatchObject(body.params);

    // ... and a project.json brings them back into the panel
    await page.getByTestId('new-project').click();
    await expect(page.getByTestId('sym-row')).toHaveCount(0);
    await expect(page.getByTestId('p-stress-limit')).toHaveValue('');
    await page.getByTestId('project-file-input').setInputFiles({ name: 'p.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(exported)) });
    await expect(page.getByTestId('sym-row')).toHaveCount(2);
    await expect(page.getByTestId('p-stress-limit')).toHaveValue('2.5');
    await expect(page.getByTestId('p-stress-pnorm')).toHaveValue('12');
    await expect(page.getByTestId('p-overhang')).toHaveValue('+z');
    await expect(page.getByTestId('sym-row').nth(1).getByTestId('sym-axis')).toHaveValue('y');
    await expect(page.getByTestId('sym-row').nth(1).getByTestId('sym-pos')).toHaveValue('12.5');
    await expect(page.getByTestId('p-optimizer').locator('option:checked')).toHaveText('mma (forced by stress limit)');
  });

  test('symmetry planes span the domain in the viewport; the overhang arrow and base plate sit on the build face', async ({ page }) => {
    await importBracket(page);
    await expect(page.getByTestId('voxel-active')).toBeVisible();
    const dom = await state(page, (s) => {
      const st = s.voxel.stats!;
      return { min: st.origin, max: st.origin.map((o, k) => o + st.h * [st.nx, st.ny, st.nz][k]!) };
    });
    const overlays = async () => (await viewportStats(page)).overlays;
    expect((await overlays()).symmetry).toHaveLength(0);
    expect((await overlays()).overhang).toBeNull();

    // x plane through the middle of the part (80 mm long): a quad in x = 40 spanning the grid in y and z
    await page.getByTestId('sym-add').click();
    await expect.poll(async () => (await overlays()).symmetry.length).toBe(1);
    let q = (await overlays()).symmetry[0]!;
    expect(q.axis).toBe('x');
    expect(q.position).toBeCloseTo(40, 3);
    expect([q.min[0], q.max[0]]).toEqual([q.position, q.position]);
    for (const k of [1, 2]) {
      expect(q.min[k]!).toBeCloseTo(dom.min[k]!, 3);
      expect(q.max[k]!).toBeCloseTo(dom.max[k]!, 3);
    }

    // y plane at an explicit coordinate
    await page.getByTestId('sym-axis').selectOption('y');
    await expect.poll(async () => (await overlays()).symmetry[0]!.axis).toBe('y');
    expect((await overlays()).symmetry[0]!.position).toBeCloseTo(30, 3);
    await page.getByTestId('sym-center').uncheck();
    await page.getByTestId('sym-pos').fill('12');
    await expect.poll(async () => (await overlays()).symmetry[0]!.position).toBe(12);
    q = (await overlays()).symmetry[0]!;
    expect(q.min[0]!).toBeCloseTo(dom.min[0]!, 3);
    expect(q.max[2]!).toBeCloseTo(dom.max[2]!, 3);
    await page.getByTestId('sym-add').click();
    await expect.poll(async () => (await overlays()).symmetry.length).toBe(2);
    await page.getByTestId('sym-remove').first().click();
    await expect.poll(async () => (await overlays()).symmetry.length).toBe(1);
    await expect(page.getByTestId('sym-row')).toHaveCount(1);

    // overhang: +z builds from the min-z face, -x from the max-x face; the arrow starts on that face
    await page.getByTestId('p-overhang').selectOption('+z');
    await expect.poll(async () => (await overlays()).overhang?.dir).toBe('+z');
    let am = (await overlays()).overhang!;
    expect(am.axis).toBe('z');
    expect(am.plateAt).toBeCloseTo(dom.min[2]!, 3);
    expect(am.min[0]!).toBeCloseTo(dom.min[0]!, 3);
    expect(am.max[1]!).toBeCloseTo(dom.max[1]!, 3);
    await page.getByTestId('p-overhang').selectOption('-x');
    await expect.poll(async () => (await overlays()).overhang?.dir).toBe('-x');
    am = (await overlays()).overhang!;
    expect(am.axis).toBe('x');
    expect(am.plateAt).toBeCloseTo(dom.max[0]!, 3);
    await page.getByTestId('p-overhang').selectOption('');
    await expect.poll(async () => (await overlays()).overhang).toBeNull();
    expect(await state(page, (s) => s.project.params.overhang)).toBeNull();
    expect(await page.evaluate(() => window.__topopViewport!.coverage())).toBeGreaterThan(0.03); // overlays do not break rendering
  });

  test('progress carries stress_max (third sparkline series) and the constraint with a green/red tag', async ({ page }) => {
    await importBracket(page);
    await defineBoundaries(page);
    await page.getByTestId('domain-elements').fill('24');
    await page.getByTestId('p-max-iter').fill('3');
    await page.getByTestId('run-start').click();
    await expect.poll(() => state(page, (s) => s.run.status), { timeout: 30_000 }).toBe('done');
    // no stress limit: stress_max is reported, there is no constraint
    expect(await state(page, (s) => s.run.history.every((r) => typeof r.stress_max === 'number' && r.constraint === null))).toBe(true);
    await expect(page.getByTestId('run-stress')).toContainText('stress');
    await expect(page.getByTestId('run-constraint')).toHaveCount(0);
    await expect(page.getByTestId('sparkline')).toHaveAttribute('data-series', '3');

    // with a limit the constraint g appears: violated while g > 0, satisfied once g <= 0
    await page.getByTestId('p-stress-limit').fill('2.5');
    await page.getByTestId('p-max-iter').fill('16');
    await page.getByTestId('run-start').click();
    const tag = page.getByTestId('run-constraint-tag');
    await expect(tag).toHaveText('violated');
    await expect(page.getByTestId('run-constraint')).toHaveAttribute('data-ok', 'false');
    await expect(tag).toHaveClass(/tag-bad/);
    await expect.poll(() => state(page, (s) => s.run.status), { timeout: 30_000 }).toBe('done');
    await expect(tag).toHaveText('satisfied');
    await expect(page.getByTestId('run-constraint')).toHaveAttribute('data-ok', 'true');
    await expect(tag).toHaveClass(/tag-ok/);
    const g = await state(page, (s) => s.run.history.map((r) => r.constraint));
    expect(g[0]!).toBeGreaterThan(0);
    expect(g.at(-1)!).toBeLessThanOrEqual(0);
  });
});

test.describe('results (mock backend)', () => {
  test('Color by stress fetches /stress and recolours the density cells; the threshold slider keeps working', async ({ page }) => {
    await runBracket(page, 4);
    const base = await viewportStats(page);
    expect(base.densityMode).toBe('instanced');
    expect(base.densityColorMode).toBe('density');
    expect(base.densityCount).toBeGreaterThan(0);
    await expect(page.getByTestId('stress-legend')).toHaveCount(0);

    const reply = page.waitForResponse((r) => /\/api\/runs\/[^/]+\/stress$/.test(r.url()));
    await page.getByTestId('color-stress').check();
    const res = await reply;
    expect(res.status()).toBe(200);
    const st = await state(page, (s) => s.run.stats!);
    expect((await res.body()).length).toBe(12 + 4 * st.nx * st.ny * st.nz);

    await expect.poll(async () => (await viewportStats(page)).densityColorMode).toBe('stress');
    const stressed = await viewportStats(page);
    expect(stressed.densityCount).toBe(base.densityCount); // same cells, other colours
    expect(stressed.densityColorSum).not.toBeCloseTo(base.densityColorSum, 1);
    await expect(page.getByTestId('stress-legend')).toBeVisible();
    const max = Number((await page.getByTestId('stress-max').innerText()).replace('max ', ''));
    expect(max).toBeGreaterThan(0);
    expect(await state(page, (s) => s.stress!.max)).toBeCloseTo(max, 2);
    await page.screenshot({ path: 'test-results/viewport-stress.png' });

    // threshold slider still thresholds on density
    await page.getByTestId('threshold').fill('0.1');
    const low = (await viewportStats(page)).densityCount;
    await page.getByTestId('threshold').fill('0.9');
    const high = await viewportStats(page);
    expect(low).toBeGreaterThan(high.densityCount);
    expect(high.densityColorMode).toBe('stress');

    // off again: back to density colours
    await page.getByTestId('threshold').fill('0.5');
    await page.getByTestId('color-stress').uncheck();
    await expect.poll(async () => (await viewportStats(page)).densityColorMode).toBe('density');
    expect((await viewportStats(page)).densityColorSum).toBeCloseTo(base.densityColorSum, 2);
    await expect(page.getByTestId('stress-legend')).toHaveCount(0);

    // the result mesh stays density-coloured: turning stress on does not touch it
    await page.getByTestId('load-result').click();
    await expect.poll(async () => (await viewportStats(page)).result).toBe(true);
    await page.getByTestId('color-stress').check(); // cached: no new request needed
    await expect(page.getByTestId('stress-legend')).toBeVisible();
    await expect(page.getByTestId('stress-legend')).toContainText('result mesh stays density-coloured');
  });

  test('/stress answers 409 while the run is going, and the toggle waits for the end', async ({ page }) => {
    await importBracket(page);
    await defineBoundaries(page);
    await page.getByTestId('domain-elements').fill('24');
    await page.getByTestId('p-max-iter').fill('300');
    await page.getByTestId('run-start').click();
    await expect.poll(() => state(page, (s) => s.run.history.length)).toBeGreaterThan(0);
    await expect(page.getByTestId('color-stress')).toBeDisabled();
    const id = await state(page, (s) => s.run.id);
    const early = await page.request.get(`/api/runs/${id}/stress`);
    expect(early.status()).toBe(409);
    await page.getByTestId('run-stop').click();
    await expect.poll(() => state(page, (s) => s.run.status)).toBe('cancelled');
    await expect(page.getByTestId('color-stress')).toBeEnabled();
    expect((await page.request.get(`/api/runs/${id}/stress`)).status()).toBe(200); // a cancelled run keeps its partial result
  });

  test('Trim to CAD: the STL link and the result-mesh load carry trim=, and X-Topop-Warnings is shown', async ({ page }) => {
    await runBracket(page, 3);
    const link = page.getByTestId('download-stl');
    await expect(link).toHaveAttribute('href', /\/api\/runs\/[^/]+\/result\.stl\?threshold=0\.5&smooth=3&trim=false$/);
    const plain = await page.request.get((await link.getAttribute('href'))!);
    expect(plain.status()).toBe(200);
    expect(plain.headers()['x-topop-warnings']).toBeUndefined();

    await page.getByTestId('trim-cad').check();
    await expect(link).toHaveAttribute('href', /smooth=3&trim=true$/);
    const trimmed = await page.request.get((await link.getAttribute('href'))!);
    expect(trimmed.status()).toBe(200);
    expect((await trimmed.body()).length).toBeGreaterThan(84);
    expect(trimmed.headers()['x-topop-warnings']).toContain('trim');

    // result mesh load: the request is trimmed and the server's warning is shown next to it
    await expect(page.getByTestId('result-warnings')).toHaveCount(0);
    const req = page.waitForRequest((r) => /result\.stl\?.*trim=true/.test(r.url()));
    await page.getByTestId('load-result').click();
    await req;
    await expect(page.getByTestId('result-warnings')).toContainText('trim to CAD not applied');
    await expect.poll(async () => (await viewportStats(page)).result).toBe(true);

    // switching trim off while the mesh is on screen reloads it untrimmed (and the warning goes)
    const again = page.waitForRequest((r) => /result\.stl\?.*trim=false/.test(r.url()));
    await page.getByTestId('trim-cad').uncheck();
    await again;
    await expect(page.getByTestId('result-warnings')).toHaveCount(0);
    await expect(link).toHaveAttribute('href', /trim=false$/);
  });
});

test.describe('facets (mock backend)', () => {
  test('rows show kind, radius and B-rep ids; highlight uses the exact faces endpoint, cached per facet', async ({ page }) => {
    const faceReqs: string[] = [];
    page.on('request', (r) => {
      if (/\/facets\/\d+\/faces/.test(r.url())) faceReqs.push(r.url());
    });
    await importBracket(page);
    await page.getByTestId('mode-query').click();
    await page.getByTestId('query-tab-facets').click();
    await expect(page.getByTestId('facet-list')).toBeVisible();
    await expect(page.getByTestId('facets-source')).toHaveCount(0); // a plain mesh: no STEP badge
    await page.getByTestId('facets-more').click();

    const table = await state(page, (s) => Object.values(s.facetCache)[0]!.facets);
    const hole = table.find((f) => f.kind === 'cylinder' && Math.abs((f.radius ?? 0) - 6) < 0.25);
    expect(hole, 'a cylinder facet for the 12 mm hole').toBeTruthy();
    expect(hole!.axis).toEqual([1, 0, 0]);
    expect(table.some((f) => f.kind === 'plane')).toBe(true);
    const row = page.locator(`[data-testid="facet-row"][data-facet-id="${hole!.id}"]`);
    await expect(row.getByTestId('facet-kind')).toHaveText('cylinder');
    await expect(row.getByTestId('facet-radius')).toHaveText(/^r 6(\.\d+)?$/);
    await expect(row.getByTestId('facet-brep')).toHaveCount(0); // not a STEP mesh
    const planeRow = page.locator('[data-testid="facet-row"][data-facet-id="0"]');
    await expect(planeRow.getByTestId('facet-kind')).toHaveAttribute('data-kind', 'plane');
    await expect(planeRow).toContainText('n ');

    // hover: the highlight is exactly the facet's triangles, fetched once
    const reqsFor = (id: number) => faceReqs.filter((u) => u.includes(`/facets/${id}/faces?angle_deg=5`)).length;
    expect(faceReqs).toHaveLength(0);
    await row.hover();
    await expect.poll(() => state(page, (s) => s.hoverFaces.length)).toBe(hole!.n_faces);
    expect(reqsFor(hole!.id)).toBe(1);
    expect(faceReqs[0]).toContain(`/api/meshes/`);
    const exact = await state(page, (s) => [...s.hoverFaces]);
    expect(new Set(exact).size).toBe(hole!.n_faces);
    await page.mouse.move(900, 600);
    await expect.poll(() => state(page, (s) => s.hoverFaces.length)).toBe(0);
    await row.hover();
    await expect.poll(() => state(page, (s) => s.hoverFaces.length)).toBe(hole!.n_faces);
    expect(reqsFor(hole!.id)).toBe(1); // cached

    // every cylinder: the highlight is the endpoint's face list (the normal + bbox reconstruction would pick up neighbours)
    const meshId = await state(page, (s) => s.project.design_mesh!.mesh_id!);
    for (const cyl of table.filter((f) => f.kind === 'cylinder')) {
      await page.locator(`[data-testid="facet-row"][data-facet-id="${cyl.id}"]`).hover();
      const want = ((await (await page.request.get(`/api/meshes/${meshId}/facets/${cyl.id}/faces?angle_deg=5`)).json()) as { face_ids: number[] }).face_ids;
      expect(want).toHaveLength(cyl.n_faces);
      await expect.poll(() => state(page, (s) => [...s.hoverFaces].sort((a, b) => a - b)), { message: `facet ${cyl.id}` }).toEqual(want);
    }

    // selecting a facet colours the same exact faces (and prefetches them)
    await planeRow.click();
    await expect.poll(() => reqsFor(0)).toBe(1);
    const cached = await state(page, (s) => Object.keys(s.facetFaceCache));
    expect(cached.filter((k) => k.endsWith('#0') || k.endsWith(`#${hole!.id}`))).toHaveLength(2);
    await row.click();
    expect(await state(page, (s) => (s.selection.query as { facet_ids: number[] }).facet_ids)).toEqual([0, hole!.id]);
    expect(reqsFor(0)).toBe(1);
    expect(reqsFor(hole!.id)).toBe(1);

    // the answer of the endpoint is what gets highlighted: serve a made-up list, late, to see both phases.
    // In flight, the old normal + bbox reconstruction stands in; once the answer is in, its ids win.
    const other = table.find((f) => f.id !== 0 && f.id !== hole!.id && f.kind === 'plane')!;
    await page.route(`**/facets/${other.id}/faces*`, async (route) => {
      await new Promise((r) => setTimeout(r, 800));
      await route.fulfill({ json: { face_ids: [3, 5, 7] } });
    });
    await page.locator(`[data-testid="facet-row"][data-facet-id="${other.id}"]`).hover();
    await expect.poll(() => state(page, (s) => s.hoverFaces.length)).toBeGreaterThan(0);
    expect(await state(page, (s) => [...s.hoverFaces])).not.toEqual([3, 5, 7]);
    await expect.poll(() => state(page, (s) => [...s.hoverFaces]), { timeout: 5_000 }).toEqual([3, 5, 7]);
    // leaving the row before the answer arrives must not light it up later
    const third = table.find((f) => ![0, hole!.id, other.id].includes(f.id) && f.kind === 'plane')!;
    await page.route(`**/facets/${third.id}/faces*`, async (route) => {
      await new Promise((r) => setTimeout(r, 500));
      await route.fulfill({ json: { face_ids: [11, 13] } });
    });
    await page.locator(`[data-testid="facet-row"][data-facet-id="${third.id}"]`).hover();
    await page.mouse.move(900, 600);
    await expect.poll(() => state(page, (s) => s.hoverFaces.length)).toBe(0);
    await expect.poll(async () => (await state(page, (s) => Object.keys(s.facetFaceCache))).some((k) => k.endsWith(`#${third.id}`))).toBe(true); // the answer was cached ...
    expect(await state(page, (s) => s.hoverFaces.length)).toBe(0); // ... but not shown
  });
});

test.describe('design mesh transform (mock backend)', () => {
  test('pos/rot/scale fields: the full matrix goes to the server, the grid follows, picking and markers work in world space', async ({ page }) => {
    const writes = recordProjectWrites(page);
    await importBracket(page);
    await expect(page.getByTestId('design-transform')).toBeVisible();
    await expect(page.getByTestId('design-reset')).toBeDisabled();
    await expect.poll(() => state(page, (s) => s.voxel.stats?.origin[0] ?? null)).not.toBeNull();
    const x0 = (await state(page, (s) => s.voxel.stats!.origin[0]))!;
    const lastMatrix = () => writes.at(-1)!.json.design_mesh.transform as number[];
    expect(lastMatrix()).toEqual(MATRIX_ID);

    await page.getByTestId('design-pos-x').fill('10');
    await expect(page.getByTestId('design-reset')).toBeEnabled();
    await expect.poll(() => state(page, (s) => s.voxel.stats!.origin[0])).toBeCloseTo(x0 + 10, 3); // re-voxelized with the new pose
    const m = lastMatrix();
    expect(m).not.toEqual(MATRIX_ID);
    MATRIX_ID.forEach((v, i) => expect(m[i]!).toBeCloseTo(i === 12 ? 10 : v, 6));
    expect((await viewportStats(page)).designMatrix[12]).toBeCloseTo(10, 6);

    // rotation and scale are part of the same column-major matrix
    await page.getByTestId('design-scale-x').fill('2');
    await page.getByTestId('design-rot-z').fill('90');
    await expect.poll(async () => (await state(page, (s) => s.project.design_mesh!.transform!))[1]).toBeCloseTo(2, 5); // x axis -> +y, scaled by 2
    const m2 = await state(page, (s) => s.project.design_mesh!.transform!);
    expect(Math.hypot(m2[0]!, m2[1]!, m2[2]!)).toBeCloseTo(2, 5);
    expect(m2[12]).toBeCloseTo(10, 5);
    await page.getByTestId('design-reset').click();
    await expect.poll(() => state(page, (s) => s.project.design_mesh!.transform)).toEqual(MATRIX_ID);
    await expect(page.getByTestId('design-pos-x')).toHaveValue('0');
    await page.getByTestId('design-pos-x').fill('10');
    await expect.poll(() => state(page, (s) => s.voxel.stats!.origin[0])).toBeCloseTo(x0 + 10, 3);

    // picking: the ray through the screen position of a WORLD point hits the mesh at that point
    const world = [50, 30, 10]; // plate top, local (40, 30, 10) moved by +10 in x
    const hit = await page.evaluate(([x, y, z]) => {
      const vp = window.__topopViewport!;
      const at = vp.screenPoint([x!, y!, z!]);
      const h = vp.pickFace(at.x, at.y);
      return h ? { p: h.point.toArray(), n: h.normal.toArray(), face: h.face, at } : null;
    }, world);
    expect(hit).not.toBeNull();
    world.forEach((v, k) => expect(hit!.p[k]!).toBeCloseTo(v, 0));
    expect(hit!.n[2]!).toBeCloseTo(1, 3);

    // clicking there selects that face; a load made from it is anchored at the face centroid moved by the pose
    await page.getByTestId('mode-pick').click();
    await page.mouse.click(hit!.at.x, hit!.at.y);
    await expect.poll(() => state(page, (s) => s.selection.faceIds.length)).toBe(1);
    const centroid = await state(page, (s) => {
      const f = s.selection.faceIds[0]!;
      const c = s.meshes[s.project.design_mesh!.mesh_id!]!.data.centroids;
      return [c[f * 3]!, c[f * 3 + 1]!, c[f * 3 + 2]!];
    });
    await page.getByTestId('add-load').click();
    await expect.poll(async () => (await viewportStats(page)).arrowAt.length).toBe(1);
    const arrow = (await viewportStats(page)).arrowAt[0]!;
    expect(arrow[0]!).toBeCloseTo(centroid[0]! + 10, 2);
    expect(arrow[1]!).toBeCloseTo(centroid[1]!, 2);
    expect(arrow[2]!).toBeCloseTo(centroid[2]!, 2);
    // moving the design moves the marker with it
    await page.getByTestId('design-pos-x').fill('20');
    await expect.poll(async () => (await viewportStats(page)).arrowAt[0]![0]).toBeCloseTo(centroid[0]! + 20, 2);
    await page.getByTestId('design-pos-x').fill('10');
    await expect.poll(async () => (await viewportStats(page)).arrowAt[0]![0]).toBeCloseTo(centroid[0]! + 10, 2);

    // painting: the brush works in world space, i.e. on the faces whose centroids, moved by the pose, are within the radius
    const R = 25;
    await page.getByTestId('mode-paint').click();
    await page.evaluate((r) => window.__topop!.getState().setBrushRadius(r), R);
    await page.mouse.move(hit!.at.x, hit!.at.y);
    await page.mouse.down();
    await page.mouse.up();
    const { sel, cent } = await state(page, (s) => ({
      sel: [...s.selection.faceIds],
      cent: Array.from(s.meshes[s.project.design_mesh!.mesh_id!]!.data.centroids),
    }));
    const dist = (f: number) => Math.hypot(cent[f * 3]! + 10 - hit!.p[0]!, cent[f * 3 + 1]! - hit!.p[1]!, cent[f * 3 + 2]! - hit!.p[2]!);
    const nTri = cent.length / 3;
    const inside = Array.from({ length: nTri }, (_, f) => f).filter((f) => dist(f) < R - 0.5);
    expect(inside.length).toBeGreaterThan(5);
    for (const f of inside) expect(sel, `face ${f} is inside the brush`).toContain(f);
    const underCursor = await page.evaluate(({ x, y }) => window.__topopViewport!.pickFace(x, y)!.face, hit!.at);
    for (const f of sel) if (f !== underCursor) expect(dist(f), `face ${f} is outside the brush`).toBeLessThan(R + 0.5);
  });

  test('the gizmo moves the design mesh and the fields follow; Escape releases it', async ({ page }) => {
    const writes = recordProjectWrites(page);
    await importBracket(page);
    await expect.poll(() => state(page, (s) => s.voxel.stats !== null)).toBe(true);
    await page.getByTestId('design-gizmo').click();
    expect(await state(page, (s) => [s.tool, s.activeItem?.kind])).toEqual(['gizmo', 'design']);
    await expect(page.getByTestId('design-gizmo')).toHaveAttribute('aria-pressed', 'true');
    for (const m of ['rotate', 'scale', 'translate'] as const) {
      await page.getByTestId(`design-gizmo-${m}`).click();
      expect(await state(page, (s) => s.gizmoMode)).toBe(m);
    }

    // the handles sit on the geometry (pose * bbox centre), not at the mesh origin
    const centre = await state(page, (s) => {
      const b = s.meshes[s.project.design_mesh!.mesh_id!]!.data.bbox;
      return [0, 1, 2].map((k) => (b.min[k]! + b.max[k]!) / 2);
    });
    const at = await page.evaluate((p) => window.__topopViewport!.screenPoint(p as [number, number, number]), centre);
    await page.mouse.move(at.x, at.y);
    await page.mouse.down();
    await page.mouse.move(at.x + 60, at.y + 20, { steps: 6 });
    await page.mouse.up();
    const t = await state(page, (s) => s.project.design_mesh!.transform!);
    expect(Math.hypot(t[12]!, t[13]!, t[14]!)).toBeGreaterThan(0.5);
    // the three position fields show the same numbers
    await expect(page.getByTestId('design-pos-x')).toHaveValue(String(Math.round(t[12]! * 1e6) / 1e6));
    await expect(page.getByTestId('design-pos-y')).toHaveValue(String(Math.round(t[13]! * 1e6) / 1e6));
    expect((await viewportStats(page)).designMatrix[12]).toBeCloseTo(t[12]!, 6);
    // it reaches the server (debounced voxelize syncs the project)
    await expect.poll(() => writes.at(-1)!.json.design_mesh.transform[12], { timeout: 10_000 }).toBeCloseTo(t[12]!, 5);

    await page.keyboard.press('Escape'); // releases the gizmo
    expect(await state(page, (s) => s.activeItem)).toBeNull();
    await page.keyboard.press('Delete'); // and Delete does not remove the design mesh
    expect(await state(page, (s) => s.project.design_mesh?.mesh_id)).toBeTruthy();
  });

  test('a project.json with a non-identity design transform loads without an "unsupported" warning and shows the pose', async ({ page }) => {
    await importBracket(page);
    const id = await state(page, (s) => s.project.design_mesh!.mesh_id!);
    await page.getByTestId('new-project').click();
    const t = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1];
    const doc = {
      name: 'moved',
      design_mesh: { mesh_id: id, transform: t },
      loads: [{ id: 'l', selection: { kind: 'normal', mesh_id: id, direction: [0, 0, 1], angle_deg: 10 }, force: [0, 0, -1] }],
      supports: [{ id: 's', selection: { kind: 'plane', point: [0, 0, 0], normal: [0, 0, 1], tol: 0 } }],
    };
    await page.getByTestId('project-file-input').setInputFiles({ name: 'moved.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(doc)) });
    await expect.poll(async () => (await viewportStats(page)).designFaces).toBeGreaterThan(0);
    await expect(page.getByTestId('notice')).toContainText('Loaded "moved"');
    await expect(page.getByTestId('notice')).not.toContainText(/unsupported|non-identity/i);
    expect(await state(page, (s) => s.project.design_mesh!.transform)).toEqual(t);
    await expect(page.getByTestId('design-pos-x')).toHaveValue('5');
    expect((await viewportStats(page)).designMatrix[12]).toBeCloseTo(5, 6);
  });
});

test.describe('STEP import (mock backend)', () => {
  test('a .step file is accepted by the input; source and B-rep faces show up, facets are the B-rep faces', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByTestId('mesh-file-input')).toHaveAttribute('accept', /\.step/);
    await expect(page.getByTestId('mesh-file-input')).toHaveAttribute('accept', /\.stp/);
    await page.getByTestId('mesh-file-input').setInputFiles(BRACKET_STEP);
    await expect.poll(async () => (await viewportStats(page)).designFaces).toBeGreaterThan(0);

    const info = await state(page, (s) => s.meshes[s.project.design_mesh!.mesh_id!]!.info);
    expect(info.source).toBe('step');
    expect(info.n_brep_faces).toBeGreaterThan(5);
    expect(info.name).toBe('bracket.step');
    await expect(page.getByTestId('mesh-source')).toHaveText(/STEP/);
    await expect(page.getByTestId('mesh-source')).toHaveAttribute('data-source', 'step');
    await expect(page.getByTestId('mesh-brep-faces')).toHaveText(String(info.n_brep_faces));

    await page.getByTestId('mode-query').click();
    await page.getByTestId('query-tab-facets').click();
    await expect(page.getByTestId('facet-list')).toBeVisible();
    await expect(page.getByTestId('facets-source')).toContainText('STEP: ids are B-rep faces');
    await expect(page.getByTestId('facets-angle')).toBeDisabled();
    const rows = page.getByTestId('facet-row');
    expect(await rows.count()).toBeGreaterThan(3);
    await expect(rows.first().getByTestId('facet-brep')).toContainText('B-rep #');
    const table = await state(page, (s) => Object.values(s.facetCache)[0]!.facets);
    expect(table.every((f) => typeof f.brep_face === 'number')).toBe(true);
    expect(new Set(table.map((f) => f.brep_face)).size).toBe(table.length);
    expect(table.some((f) => f.kind === 'cylinder' && f.radius != null)).toBe(true);
    expect(await state(page, (s) => Object.values(s.facetCache)[0]!.n_facets_total)).toBe(info.n_brep_faces);

    // hover highlights the B-rep face's triangles through the faces endpoint
    const first = table[0]!;
    await rows.first().hover();
    await expect.poll(() => state(page, (s) => s.hoverFaces.length)).toBe(first.n_faces);
  });

  test('drag-drop takes .stp, and a server answer 400 with the install hint is shown verbatim', async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => {
      const dt = new DataTransfer();
      dt.items.add(new File(['ISO-10303-21;\nHEADER;\nENDSEC;\nEND-ISO-10303-21;\n'], 'part.stp'));
      window.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }));
    });
    await expect.poll(() => state(page, (s) => Object.values(s.meshes)[0]?.info.source ?? null)).toBe('step');
    expect(await state(page, (s) => Object.values(s.meshes)[0]!.info.name)).toBe('part.stp');

    const hint = 'STEP import needs OpenCascade: run `uv sync --extra step` (or `pip install "topop[step]"`) and retry';
    await page.getByTestId('new-project').click();
    await page.getByTestId('mesh-file-input').setInputFiles({ name: 'nostep.step', mimeType: 'application/step', buffer: readFileSync(BRACKET_STEP) });
    await expect(page.getByTestId('notice')).toHaveText(new RegExp(hint.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
    expect(await state(page, (s) => s.notice!.text)).toBe(hint);
    expect(await state(page, (s) => s.notice!.kind)).toBe('error');
    expect(await state(page, (s) => s.project.design_mesh)).toBeNull();
  });
});
