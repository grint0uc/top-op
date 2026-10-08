// The frontend against the real server (`uv run topop serve`, fresh data dir; see playwright.config.ts, project `real`).
// One serial scenario on examples/bracket.stl, each test a stage of the workflow. Screenshots: test-results/real-*.png.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { type APIResponse, type Page, expect, test } from '@playwright/test';
import { BRACKET, selectFlatFaceAtCentre, state, viewportStats } from './helpers';

test.describe.configure({ mode: 'serial', timeout: 120_000 });

let page: Page;
let baseURL: string;
const shot = (name: string) => page.screenshot({ path: `test-results/real-${name}.png` });

/** Facts carried from one stage to the next. */
const ctx: {
  meshId: string;
  bbox: number[][];
  h: number;
  nActive: number; // active elements before / after the keep-out
  nActiveKeepOut: number;
  runId: string;
  projectJson: string;
} = { meshId: '', bbox: [], h: 0, nActive: 0, nActiveKeepOut: 0, runId: '', projectJson: '' };

test.beforeAll(async ({ browser }) => {
  baseURL = String(test.info().project.use.baseURL);
  page = await (await browser.newContext({ baseURL, viewport: { width: 1360, height: 860 } })).newPage();
});

test.afterAll(async () => {
  await page.context().close();
});

const text = async (r: APIResponse) => (await r.body()).toString('latin1');

test('1. upload bracket.stl: content-hash id, watertight MeshInfo, mesh in the viewport', async () => {
  await page.goto('/');
  await page.getByTestId('mesh-file-input').setInputFiles(BRACKET);
  await expect
    .poll(async () => (await page.evaluate(() => window.__topopViewport?.stats().designFaces)) ?? 0, { message: 'design mesh in viewport' })
    .toBeGreaterThan(0);

  const info = await state(page, (s) => s.meshes[s.project.design_mesh?.mesh_id ?? '']!.info);
  expect(info.is_watertight).toBe(true);
  expect(info.n_faces).toBe(1326);
  expect(info.volume).toBeGreaterThan(70_000);
  expect(info.bbox).toEqual([
    [0, 0, 0],
    [80, 60, 60],
  ]);
  // ids are content hashes: first 16 hex digits of the sha256 of the file
  expect(info.id).toBe(createHash('sha256').update(readFileSync(BRACKET)).digest('hex').slice(0, 16));
  expect(await state(page, (s) => s.project.design_mesh?.mesh_id)).toBe(info.id);
  await expect(page.getByTestId('mesh-faces')).toHaveText('1,326');
  await expect(page.getByTestId('watertight-warning')).toHaveCount(0);
  ctx.meshId = info.id;
  ctx.bbox = info.bbox;
  expect(await page.evaluate(() => window.__topopViewport!.coverage())).toBeGreaterThan(0.03);

  // a project without loads/supports is rejected with 422 {detail: "project is not runnable: ..."}; the Run panel says why
  await page.getByTestId('run-start').click();
  await expect(page.getByTestId('run-error')).toContainText('project is not runnable');
  await expect(page.getByTestId('run-error')).toContainText('no loads defined');
  expect(await state(page, (s) => s.run.status)).toBe('idle');
});

test('2. resolution 24: VoxelStats come from the server', async () => {
  await page.getByTestId('domain-elements').fill('24');
  // (the default 60-element grid is voxelized first; the debounced request for 24 replaces it)
  await expect.poll(() => state(page, (s) => s.voxel.stats?.nx ?? 0), { timeout: 30_000 }).toBe(26);
  const st = await state(page, (s) => s.voxel.stats!);
  expect(st.n_active).toBeGreaterThan(0);
  expect([st.nx, st.ny, st.nz]).toEqual([26, 20, 20]); // 80 x 60 x 60 at h = 80/24, plus one padding cell per side
  expect(st.h).toBeCloseTo(80 / 24, 6);
  expect(st.origin.map((v) => +v.toFixed(4))).toEqual([-3.3333, -3.3333, -3.3333]);
  expect(st.n_nodes).toBeGreaterThan(st.n_active); // nodes of the active elements only
  expect(st.n_dof).toBe(3 * st.n_nodes);
  expect(st.est_bytes).toBeGreaterThan(0);
  await expect(page.getByTestId('voxel-active')).toHaveText(st.n_active.toLocaleString());
  ctx.h = st.h;
  ctx.nActive = st.n_active;
});

test('2b. pick and shift+click grow work on the real /buffer and /adjacency', async () => {
  const grown = await selectFlatFaceAtCentre(page);
  expect(grown).toBeGreaterThan(10);
  await page.keyboard.press('Escape');
  expect(await state(page, (s) => s.selection.faceIds.length)).toBe(0);
  await page.getByTestId('mode-orbit').click();
});

test('3. support from the facets query: the largest facet is the plate bottom', async () => {
  await page.getByTestId('mode-query').click();
  await page.getByTestId('query-tab-facets').click(); // fetches GET /facets?angle_deg=5
  await expect(page.getByTestId('facet-list')).toBeVisible();
  const rows = page.getByTestId('facet-row');
  expect(await rows.count()).toBeGreaterThan(5);
  expect(await rows.count()).toBeLessThanOrEqual(12); // top 12 by area
  const first = rows.first();
  await expect(first).toHaveAttribute('data-facet-id', '0');
  await expect(first).toContainText('n -Z');
  const facet = await state(page, (s) => Object.values(s.facetCache)[0]!.facets[0]!);
  expect(facet.normal).toEqual([0, 0, -1]);
  expect(facet.bbox).toEqual([
    [0, 0, 0],
    [80, 60, 0],
  ]);

  // hovering highlights its faces in the viewport (reconstructed client-side from normal + bbox)
  await first.hover();
  await expect.poll(() => state(page, (s) => s.hoverFaces.length)).toBe(facet.n_faces);
  await shot('facets-hover');
  await page.mouse.move(700, 600);
  await expect.poll(() => state(page, (s) => s.hoverFaces.length)).toBe(0);
  // the top of the plate (#1) is visible from the iso camera: same relation, and a picture of it
  const top = await state(page, (s) => Object.values(s.facetCache)[0]!.facets[1]!);
  await rows.nth(1).hover();
  await expect.poll(() => state(page, (s) => s.hoverFaces.length)).toBe(top.n_faces);
  await shot('facets-hover-top');
  await page.mouse.move(700, 600);
  await expect.poll(() => state(page, (s) => s.hoverFaces.length)).toBe(0);

  await first.click();
  expect(await state(page, (s) => s.selection.query)).toEqual({ kind: 'facets', mesh_id: ctx.meshId, facet_ids: [0], angle_deg: 5 });
  await expect(page.getByTestId('query-status')).toContainText('facet #0');
  // resolve preview works for a facets selection
  await page.getByTestId('resolve-preview').click();
  await expect.poll(() => state(page, (s) => s.preview?.count ?? 0)).toBeGreaterThan(50);
  await expect.poll(async () => (await viewportStats(page)).resolvedPoints).toBeGreaterThan(0);

  await page.getByTestId('add-support').click();
  const sup = await state(page, (s) => s.project.supports[0]!);
  expect(sup.selection.kind).toBe('facets');
  await expect(page.getByTestId('support-row').first()).toContainText('facet #0');
  await expect.poll(async () => (await viewportStats(page)).glyphs).toBe(1); // marker anchored from the cached facet table
  await page.getByTestId('item-preview').first().click();
  await expect.poll(() => state(page, (s) => s.preview?.source ?? '')).toContain('Support');
});

test('4. load from the normal query: +Z inside a box over the wall top, force (0,0,-100)', async () => {
  await page.getByTestId('mode-query').click();
  await page.getByTestId('query-tab-normal').click();
  await page.getByTestId('normal-dir-+Z').click();
  await expect(page.getByTestId('query-status')).toContainText('normal +Z');

  // box over the wall top, built from MeshInfo.bbox (top of the wall: z = 60) grown by one voxel: nodes sit up to h/2 outside the surface
  await page.getByTestId('normal-within-toggle').check();
  const [lo, hi] = ctx.bbox as [number[], number[]];
  const h = ctx.h;
  // pre-filled from the design bbox padded by h ...
  await expect(page.getByTestId('within-min-x')).toHaveValue(String(+(lo[0]! - h).toFixed(4)));
  await expect(page.getByTestId('within-max-z')).toHaveValue(String(+(hi[2]! + h).toFixed(4)));
  await expect(page.getByTestId('within-hint')).toContainText('h/2');
  // ... then shrunk to the top two layers (only the wall top faces +Z up there)
  await page.getByTestId('within-min-z').fill(String(hi[2]! - 2 * h));
  const query = await state(page, (s) => s.selection.query);
  expect(query).toMatchObject({
    kind: 'normal',
    mesh_id: ctx.meshId,
    direction: [0, 0, 1],
    angle_deg: 10,
  });
  const within = (query as { within: number[][] }).within;
  expect(within[0]![2]!).toBeCloseTo(hi[2]! - 2 * h, 3);
  expect(within[1]![2]!).toBeCloseTo(hi[2]! + h, 3);

  // matching faces light up (client-side preview): the 2 triangles of the wall top
  await shot('normal-query');
  await page.getByTestId('clear-preview').click(); // (the previous stage's preview is still up)
  await expect.poll(() => state(page, (s) => s.preview?.count ?? 0)).toBe(0);
  await page.getByTestId('resolve-preview').click();
  await expect.poll(() => state(page, (s) => s.preview?.count ?? 0)).toBeGreaterThan(10);
  const nodes = await state(page, (s) => s.preview!.count);

  await page.getByTestId('add-load').click();
  const row = page.getByTestId('load-row').first();
  await expect(row).toContainText('normal +Z');
  await expect(row).toContainText('in box');
  await row.getByTestId('force-z').fill('-100');
  await expect.poll(() => state(page, (s) => s.project.loads[0]?.force)).toEqual([0, 0, -100]);
  await expect.poll(async () => (await viewportStats(page)).arrows).toBe(1); // arrow anchored at the matching faces

  // resolve the stored load: same nodes as the query itself
  await row.getByTestId('item-preview').click();
  await expect.poll(() => state(page, (s) => s.preview?.source ?? '')).toContain('Load');
  expect(await state(page, (s) => s.preview!.count)).toBe(nodes);
  expect(await state(page, (s) => s.preview!.xyz.length / 3)).toBeGreaterThan(0);
  await expect.poll(async () => (await viewportStats(page)).resolvedPoints).toBeGreaterThan(0);

  // plane query resolves too (z = wall top); not kept as a load
  await page.getByTestId('query-tab-plane').click();
  await page.getByTestId('plane-point-z').fill(String(hi[2]));
  await page.getByTestId('plane-dir-+Z').click();
  await page.getByTestId('clear-preview').click();
  await page.getByTestId('resolve-preview').click();
  await expect.poll(() => state(page, (s) => s.preview?.count ?? 0)).toBeGreaterThan(0);
  expect(await state(page, (s) => s.preview!.source)).toBe('selection');
  await shot('query-forms');
  await page.getByTestId('clear-selection').click();
  await page.getByTestId('clear-preview').click();
});

test('5. keep-out cylinder through the plate: the full matrix (with scale) reaches the voxelizer', async () => {
  await page.getByTestId('mode-orbit').click();
  await page.getByTestId('add-ref-cylinder').click();
  await expect.poll(() => state(page, (s) => s.project.ref_models.length)).toBe(1);
  await expect.poll(async () => (await viewportStats(page)).refs).toBe(1);
  expect(await state(page, (s) => s.project.ref_models[0]!.mode)).toBe('keep_out');

  // unit cylinder (radius 1, height 1, axis Y) -> radius 4, 30 long, axis along Z, through the middle of the plate (z in 0..10)
  const set = async (id: string, v: number) => page.getByTestId(id).fill(String(v));
  await set('ref-scale-x', 4);
  await set('ref-scale-y', 30);
  await set('ref-scale-z', 4);
  await set('ref-rot-x', 90);
  await set('ref-pos-x', 50);
  await set('ref-pos-y', 30);
  await set('ref-pos-z', 5);
  const m = await state(page, (s) => s.project.ref_models[0]!.transform!);
  // column-major: columns are the scaled, rotated axes
  const near = (a: number[], b: number[]) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i]!, 6));
  near(m, [4, 0, 0, 0, /* y axis -> z */ 0, 0, 30, 0, /* z axis -> -y */ 0, -4, 0, 0, 50, 30, 5, 1]);
  const design = await state(page, (s) => s.project.design_mesh!.transform);
  expect(design).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]); // design mesh transform stays identity

  // keep_out bodies remove elements from the design (n_active drops; there is no "passive void" count for them)
  await expect.poll(() => state(page, (s) => s.voxel.stats?.n_active ?? 0), { timeout: 30_000 }).toBeLessThan(ctx.nActive);
  const st = await state(page, (s) => s.voxel.stats!);
  const removed = ctx.nActive - st.n_active;
  // pi * 4^2 * 10 / h^3 ~ 14 cells; a wrong scale or an unrotated cylinder removes 0 or ~40
  expect(removed).toBeGreaterThanOrEqual(6);
  expect(removed).toBeLessThanOrEqual(30);
  expect((st.warnings ?? []).join(' ')).not.toContain('removes no elements');
  ctx.nActiveKeepOut = st.n_active;
  await page.getByTestId('mode-orbit').click();
  await shot('keep-out');
});

test('6. run 8 iterations behind another run: queued -> started with stats -> progress, density, done', async () => {
  await page.getByTestId('p-max-iter').fill('8');
  await page.getByTestId('p-density-every').fill('1');
  expect(await state(page, (s) => [s.project.params.max_iter, s.project.params.density_every])).toEqual([8, 1]);

  // occupy the server's single run slot with a long run of the same case, started through the API
  const doc = await state(page, (s) => ({ ...s.project, ref_models: s.project.ref_models }));
  const created = await page.request.post('/api/projects', { data: { ...doc, params: { ...doc.params, max_iter: 2000 } } });
  expect(created.status()).toBe(200);
  const blocker = await page.request.post('/api/runs', { data: { project_id: (await created.json()).id } });
  expect(blocker.status()).toBe(200);
  const blockerId: string = (await blocker.json()).id;

  // what actually goes over the run WebSocket (text frames: JSON, binary frames: density)
  const wire: { type: string; status?: string; stats?: boolean; bytes?: number }[] = [];
  page.on('websocket', (ws) => {
    if (!ws.url().includes('/stream')) return;
    ws.on('framereceived', (f) => {
      if (typeof f.payload === 'string') {
        const m = JSON.parse(f.payload) as { type: string; run?: { status: string; stats: unknown } };
        wire.push({ type: m.type, status: m.run?.status, stats: !!m.run?.stats });
      } else wire.push({ type: 'density', bytes: f.payload.length });
    });
  });
  // record every run.status transition the UI goes through
  await page.evaluate(() => {
    const w = window as unknown as { __statuses: string[] };
    w.__statuses = [];
    window.__topop!.subscribe((s, p) => {
      if (s.run.status !== p.run.status) w.__statuses.push(s.run.status);
    });
  });
  await page.getByTestId('run-start').click();
  await expect(page.getByTestId('run-status')).toHaveText('queued', { timeout: 30_000 });
  await expect.poll(() => state(page, (s) => s.run.id), { timeout: 30_000 }).not.toBeNull(); // POST /api/runs answered
  const run = await state(page, (s) => ({ id: s.run.id, status: s.run.status, n: s.run.history.length }));
  expect(run.n).toBe(0);
  ctx.runId = run.id!;
  await shot('run-queued');

  // exports before completion are 409 with a message
  const early = await page.request.get(`/api/runs/${ctx.runId}/result.stl`);
  expect(early.status()).toBe(409);
  expect((await early.json()).detail).toContain('no result to export');
  expect((await page.request.get(`/api/runs/${ctx.runId}/result.vti`)).status()).toBe(409);
  expect((await page.request.get(`/api/runs/${ctx.runId}/result.npz`)).status()).toBe(409);

  // release the slot: the queued run starts (second `started` frame, status running)
  const cancelled = await page.request.post(`/api/runs/${blockerId}/cancel`);
  expect(cancelled.status()).toBe(200);
  await expect.poll(() => page.evaluate(() => (window as unknown as { __statuses: string[] }).__statuses.includes('running')), { timeout: 60_000 }).toBe(true);
  const stats = await state(page, (s) => s.run.stats);
  expect(stats).not.toBeNull();
  expect(stats!.origin).toHaveLength(3);
  expect([stats!.nx, stats!.ny, stats!.nz]).toEqual([26, 20, 20]);
  expect(stats!.h).toBeCloseTo(ctx.h, 9);

  await expect.poll(() => state(page, (s) => s.run.status), { timeout: 120_000 }).toBe('done');
  // wire level: `started` (queued, then running) carry run.stats, >= 8 progress, density frames of nx*ny*nz+16 bytes, then `done`
  const started = wire.filter((m) => m.type === 'started');
  expect(started.map((m) => m.status)).toEqual(['queued', 'running']);
  expect(started.every((m) => m.stats)).toBe(true);
  expect(wire[0]!.type).toBe('started');
  expect(wire.filter((m) => m.type === 'progress').length).toBeGreaterThanOrEqual(8);
  const dens = wire.filter((m) => m.type === 'density');
  expect(dens.length).toBeGreaterThanOrEqual(8); // density_every = 1
  expect(dens.every((m) => m.bytes === 16 + 26 * 20 * 20)).toBe(true);
  expect(wire[wire.length - 1]!.type).toBe('done');
  const r = await state(page, (s) => ({
    its: s.run.history.map((x) => x.it),
    c: s.run.history.map((x) => x.compliance),
    vol: s.run.history.map((x) => x.volume),
    frames: s.run.densityFrames,
    shape: s.run.densityFrame?.shape,
    message: s.run.message,
    error: s.run.error,
  }));
  expect(r.error).toBeNull();
  expect(await page.evaluate(() => (window as unknown as { __statuses: string[] }).__statuses)).toEqual(['queued', 'running', 'done']);
  expect(r.its.slice(0, 8)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  expect(r.its.length).toBeGreaterThanOrEqual(8);
  expect(r.c.every((v) => Number.isFinite(v) && v > 0)).toBe(true);
  expect(r.frames).toBeGreaterThanOrEqual(1);
  expect(r.shape).toEqual([26, 20, 20]);
  expect(r.message).toContain('max_iter');
  await expect(page.getByTestId('run-iter')).toContainText('it 8 / 8');
  await expect(page.getByTestId('run-message')).toContainText('max_iter');
  expect(await page.getByTestId('sparkline').getAttribute('data-points')).toBe('8');

  // density cells are drawn, positioned from run.stats (origin, h): every visible cell lies inside the design bbox, and
  // the loaded wall (x in 0..10, top at z = 60) is still there after 8 iterations
  const s0 = await viewportStats(page);
  expect(s0.densityCount).toBeGreaterThan(0);
  await page.getByTestId('threshold').fill('0.05');
  const s1 = await viewportStats(page);
  expect(s1.densityMode).toBe('instanced');
  expect(s1.densityCount).toBeGreaterThanOrEqual(s0.densityCount);
  const b = s1.densityBounds!;
  const [lo, hi] = ctx.bbox as [number[], number[]];
  const tol = ctx.h * 1.01;
  for (let k = 0; k < 3; k++) {
    expect(b.min[k]!, `min[${k}]`).toBeGreaterThanOrEqual(lo[k]! - tol);
    expect(b.max[k]!, `max[${k}]`).toBeLessThanOrEqual(hi[k]! + tol);
  }
  expect(b.min[0]!).toBeLessThanOrEqual(lo[0]! + tol); // the wall root at x = 0
  expect(b.max[2]!).toBeGreaterThanOrEqual(hi[2]! - tol); // the loaded wall top
  expect(b.max[1]! - b.min[1]!).toBeGreaterThan(0.8 * (hi[1]! - lo[1]!)); // across the full width (y)
  await page.getByTestId('threshold').fill('0.5');
  expect((await viewportStats(page)).densityCount).toBeGreaterThan(0);
  await shot('run-density');
});

test('7. result mesh loads; the four export links return bytes; project.json is a RunExport', async () => {
  await page.getByTestId('load-result').click();
  await expect.poll(() => state(page, (s) => (s.resultStl?.byteLength ?? 0) > 84)).toBe(true);
  await expect.poll(async () => (await viewportStats(page)).result).toBe(true);
  await shot('result-mesh');

  const links = {
    stl: await page.getByTestId('download-stl').getAttribute('href'),
    vti: await page.getByTestId('download-vti').getAttribute('href'),
    npz: await page.getByTestId('download-npz').getAttribute('href'),
    project: await page.getByTestId('download-project').getAttribute('href'),
  };
  const got: Record<string, APIResponse> = {};
  for (const [name, href] of Object.entries(links)) {
    expect(href, name).toContain(`/api/runs/${ctx.runId}/`);
    got[name] = await page.request.get(href!);
    expect(got[name]!.status(), name).toBe(200);
    expect((await got[name]!.body()).length, name).toBeGreaterThan(100);
  }
  expect((await text(got.stl!)).startsWith('solid') || (await got.stl!.body()).length > 84).toBe(true);
  expect((await text(got.vti!)).startsWith('<?xml')).toBe(true);
  expect((await text(got.npz!)).startsWith('PK')).toBe(true);

  const exported = await got.project!.json();
  expect(Object.keys(exported).sort()).toEqual(['project', 'run']);
  expect(exported.run.id).toBe(ctx.runId);
  expect(exported.run.status).toBe('done');
  expect(exported.run.history).toHaveLength(8);
  expect(exported.run.stats.nx).toBe(26);
  expect(exported.project.loads).toHaveLength(1);
  expect(exported.project.loads[0].selection.kind).toBe('normal');
  expect(exported.project.supports[0].selection.kind).toBe('facets');
  expect(exported.project.ref_models).toHaveLength(1);
  ctx.projectJson = JSON.stringify(exported);
});

test('8. a page reload restores the document; "Fetch from server" brings the meshes back by id', async () => {
  await page.waitForTimeout(500); // persistence is debounced
  await page.reload();
  await expect(page.getByTestId('notice')).toContainText('re-upload');
  await expect(page.getByTestId('reupload-notice')).toContainText('bracket.stl');
  expect(await state(page, (s) => Object.keys(s.meshes).length)).toBe(0);
  expect(await state(page, (s) => [s.project.loads.length, s.project.supports.length, s.project.ref_models.length])).toEqual([1, 1, 1]);
  expect(await page.getByTestId('required-mesh').count()).toBe(2); // design + the keep-out cylinder

  // re-upload only the design mesh: same bytes, same id, nothing to remap
  await page.getByTestId('required-mesh').first().getByTestId('reupload-input').setInputFiles(BRACKET);
  await expect.poll(() => state(page, (s) => Object.keys(s.meshes).length)).toBe(1);
  expect(await state(page, (s) => s.project.design_mesh?.mesh_id)).toBe(ctx.meshId);
  expect(await state(page, (s) => [s.project.loads[0]!.selection.kind, s.project.supports[0]!.selection.kind])).toEqual(['normal', 'facets']);
  await expect(page.getByTestId('required-mesh')).toHaveCount(1);

  // the generated cylinder is not a file we hold: ask the server
  await page.getByTestId('fetch-from-server').click();
  await expect.poll(() => state(page, (s) => Object.keys(s.meshes).length)).toBe(2);
  await expect(page.getByTestId('reupload-notice')).toHaveCount(0);
  await expect.poll(async () => (await viewportStats(page)).refs).toBe(1);
  await expect.poll(async () => (await viewportStats(page)).arrows).toBe(1);
  await expect.poll(async () => (await viewportStats(page)).glyphs).toBe(1);
});

test('9. Load project.json (RunExport): document restored, meshes fetched by id, run history shown', async () => {
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  expect(await state(page, (s) => s.project.loads.length)).toBe(0);

  await page.getByTestId('project-file-input').setInputFiles({
    name: 'project.json',
    mimeType: 'application/json',
    buffer: Buffer.from(ctx.projectJson),
  });
  await expect.poll(() => state(page, (s) => s.project.loads.length)).toBe(1);
  const doc = await state(page, (s) => s.project);
  const exported = JSON.parse(ctx.projectJson).project;
  expect(doc.name).toBe(exported.name);
  expect(doc.design_mesh!.mesh_id).toBe(ctx.meshId);
  expect(doc.loads[0]!.selection).toEqual(exported.loads[0].selection);
  expect(doc.loads[0]!.force).toEqual([0, 0, -100]);
  expect(doc.supports[0]!.selection).toEqual(exported.supports[0].selection);
  expect(doc.ref_models[0]!.transform).toEqual(exported.ref_models[0].transform);
  expect(doc.params.max_iter).toBe(8);
  expect(doc.grid.elements_along_longest).toBe(24);

  // the server still has the meshes (content hash ids): nothing to re-upload
  await expect.poll(() => state(page, (s) => Object.keys(s.meshes).length), { timeout: 30_000 }).toBe(2);
  await expect(page.getByTestId('reupload-notice')).toHaveCount(0);
  await expect.poll(async () => (await viewportStats(page)).designFaces).toBe(1326);
  await expect.poll(async () => (await viewportStats(page)).refs).toBe(1);

  // run history from the file, and the run is still on the server so Results works
  await expect(page.getByTestId('loaded-run')).toBeVisible();
  await expect(page.getByTestId('loaded-run-iters')).toHaveText('8');
  await expect(page.getByTestId('loaded-run-attached')).toBeVisible();
  expect(await state(page, (s) => [s.run.id, s.run.status])).toEqual([ctx.runId, 'done']);
  await expect(page.getByTestId('download-stl')).toHaveAttribute('href', new RegExp(`/api/runs/${ctx.runId}/result.stl`));

  // same document -> same voxel domain as before the export
  await expect.poll(() => state(page, (s) => s.voxel.stats?.n_active ?? -1), { timeout: 30_000 }).toBe(ctx.nActiveKeepOut);
  await shot('project-loaded');

  // re-load result mesh from the attached run
  await page.getByTestId('load-result').click();
  await expect.poll(async () => (await viewportStats(page)).result).toBe(true);
});

test('10. a project.json whose mesh the server does not know asks for a re-upload; a different file is detected', async () => {
  const fake = '0123456789abcdef';
  const edited = JSON.parse(ctx.projectJson);
  edited.project.design_mesh.mesh_id = fake;
  for (const it of [...edited.project.loads, ...edited.project.supports]) it.selection.mesh_id = fake;
  edited.run.id = 'gone-run';

  await page.getByTestId('new-project').click();
  await page.getByTestId('project-file-input').setInputFiles({ name: 'edited.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(edited)) });
  await expect(page.getByTestId('reupload-notice')).toBeVisible();
  const row = page.getByTestId('required-mesh').first();
  await expect(row).toHaveAttribute('data-mesh-id', fake);
  await expect(page.getByTestId('loaded-run-gone')).toBeVisible(); // that run id is unknown to the server
  expect(await state(page, (s) => [s.project.loads.length, s.project.supports.length])).toEqual([1, 1]);

  // the file we have is not the one the project asks for: the facets support (face ids of another mesh) goes,
  // the normal query follows the new id
  await row.getByTestId('reupload-input').setInputFiles(BRACKET);
  await expect.poll(() => state(page, (s) => s.project.design_mesh?.mesh_id)).toBe(ctx.meshId);
  await expect(page.getByTestId('notice')).toContainText('not the file the project was made with');
  expect(await state(page, (s) => s.project.supports.length)).toBe(0);
  expect(await state(page, (s) => s.project.loads[0]!.selection)).toMatchObject({ kind: 'normal', mesh_id: ctx.meshId });
});

test('11. Stop cancels a running job; the partial result is exportable', async () => {
  await page.getByTestId('project-file-input').setInputFiles({ name: 'project.json', mimeType: 'application/json', buffer: Buffer.from(ctx.projectJson) });
  await expect.poll(() => state(page, (s) => [s.project.loads.length, s.project.supports.length])).toEqual([1, 1]);
  await expect.poll(() => state(page, (s) => Object.keys(s.meshes).length), { timeout: 30_000 }).toBe(2);
  await page.getByTestId('p-max-iter').fill('400');
  await page.getByTestId('run-start').click();
  await expect.poll(() => state(page, (s) => s.run.history.length), { timeout: 60_000 }).toBeGreaterThanOrEqual(2);
  await page.getByTestId('run-stop').click();
  await expect.poll(() => state(page, (s) => s.run.status), { timeout: 30_000 }).toBe('cancelled');
  expect(await state(page, (s) => s.run.history.length)).toBeLessThan(400);
  const href = await page.getByTestId('download-stl').getAttribute('href');
  const res = await page.request.get(href!);
  expect(res.status()).toBe(200);
  expect((await res.body()).length).toBeGreaterThan(84);
  expect((await page.request.get((await page.getByTestId('download-project').getAttribute('href'))!)).status()).toBe(200);
  await expect(page.getByTestId('run-start')).toBeEnabled();
});

test('12. the built app is also served by the API server itself', async () => {
  // only meaningful once `npm run build` has written topop/server/static; skipped otherwise
  const res = await page.request.get('http://127.0.0.1:8765/');
  test.skip(!(await text(res)).includes('<div id="root">'), 'frontend not built');
  expect(res.status()).toBe(200);
  expect(res.headers()['content-type']).toContain('text/html');
});
