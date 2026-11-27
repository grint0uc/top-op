// Keep regions: face handles, voxel paint/erase, convert, remap on resolution change, passive in a run.
// Usage: python -m http.server 8920 -d lite, then
//   PW_MODULE=web/node_modules/playwright-core THREE_DIR=/path/to/three-0.147.0/package \
//   CHROMIUM=/opt/pw-browsers/chromium-1243/chrome-linux64/chrome node lite/tests/keep.mjs
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PW_MODULE || 'playwright-core');
const BASE = process.env.BASE_URL || 'http://127.0.0.1:8920/index.html';
const THREE_DIR = process.env.THREE_DIR;
const SHOTS = process.env.SHOTS || path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'screens');
fs.mkdirSync(SHOTS, { recursive: true });
const SKELETON_HEAD = '<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">'
  + '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
  + 'body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style></head><body>';

const failures = [];
const check = (name, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? ' — ' + detail : ''}`);
  if (!ok) failures.push(name);
};

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM || undefined,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});

async function newPage(opts) {
  const ctx = await browser.newContext(opts);
  const page = await ctx.newPage();
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.route('https://cdn.jsdelivr.net/npm/three@0.147.0/**', async (route) => {
    if (!THREE_DIR) return route.continue();
    const rel = new URL(route.request().url()).pathname.replace('/npm/three@0.147.0/', '');
    route.fulfill({ status: 200, contentType: 'text/javascript', body: fs.readFileSync(path.join(THREE_DIR, rel)) });
  });
  await page.route(BASE, async (route) => {
    const res = await route.fetch();
    route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: SKELETON_HEAD + (await res.text()) + '</body></html>' });
  });
  await page.goto(BASE);
  await page.waitForFunction(() => window.topopLite?.grid && window.topopLite?.selftest && window.topopLite.downloads !== undefined, null, { timeout: 30000 });
  return { ctx, page, errors };
}

async function setSlider(page, id, value) {
  await page.evaluate(([id, v]) => {
    const el = document.getElementById(id);
    el.value = String(v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, [id, value]);
}

const project = (page, p) => page.evaluate((p) => window.topopLite.api.project(p), p);
const sel = (page) => page.evaluate(() => {
  const S = window.topopLite, kp = S.keeps.find((k) => k.id === S.keepSel);
  return kp && { id: kp.id, mode: kp.mode, min: kp.min, max: kp.max, n: kp.cells ? kp.cells.size : 0 };
});
const fields = (page) => page.evaluate(() => ['x0', 'y0', 'z0', 'x1', 'y1', 'z1'].map((a) => +document.getElementById('keep-' + a).value));
const rows = (page) => page.evaluate(() => [...document.querySelectorAll('#keep-list li button.t')].map((b) => b.firstChild.textContent));
const isMult = (d, h) => Math.abs(d / h - Math.round(d / h)) < 1e-6;

async function mouseStroke(page, a, b, steps = 14) {
  const p0 = await project(page, a), p1 = await project(page, b);
  await page.mouse.move(p0[0], p0[1]);
  await page.mouse.down();
  await page.mouse.move(p1[0], p1[1], { steps });
  await page.mouse.up();
  await page.waitForTimeout(80);
}

// Drag from the middle of a handle arrow by px pixels along (sign=1) or against (sign=-1) it.
async function dragHandle(page, axis, side, px, sign = 1) {
  const hs = await page.evaluate(() => window.topopLite.api.keepHandles());
  const hd = hs.find((x) => x.axis === axis && x.side === side);
  const m = [(hd.base[0] + hd.tip[0]) / 2, (hd.base[1] + hd.tip[1]) / 2];
  const d = [hd.tip[0] - hd.base[0], hd.tip[1] - hd.base[1]];
  const l = Math.hypot(...d);
  const to = [m[0] + (sign * px * d[0]) / l, m[1] + (sign * px * d[1]) / l];
  return { from: m, to };
}

// ---- desktop: handles, fields, convert, paint, erase, run, remap
{
  const { ctx, page, errors } = await newPage({ viewport: { width: 1280, height: 800 }, colorScheme: 'light' });
  await page.click('#sec-keep summary');
  const h = await page.evaluate(() => window.topopLite.grid.h);

  // 1. Add a box; it becomes the selected region with six handles.
  for (const [id, v] of [['keep-x0', 30], ['keep-y0', 20], ['keep-z0', 0], ['keep-x1', 50], ['keep-y1', 40], ['keep-z1', 10]]) await page.fill('#' + id, String(v));
  await page.selectOption('#keep-mode', 'solid');
  await page.click('#keep-add');
  const k0 = await sel(page);
  const nh = await page.evaluate(() => window.topopLite.api.keepHandles().length);
  check('Add box selects the new region and shows 6 face handles', k0 && nh === 6 && /· box · \d[\d,]* cells in part$/.test((await rows(page))[0]), JSON.stringify({ k0, nh, rows: await rows(page) }));

  // 2. Drag the +x handle outward: max x moves by whole voxels, the fields follow, the camera does not orbit.
  const cam0 = await page.evaluate(() => window.topopLite.api.camera());
  let dr = await dragHandle(page, 0, 1, 70);
  await page.mouse.move(...dr.from);
  await page.mouse.down();
  const ctlDuring = await page.evaluate(() => window.topopLite.api.controls().enabled);
  await page.mouse.move(...dr.to, { steps: 12 });
  await page.mouse.up();
  await page.waitForTimeout(100);
  const k1 = await sel(page), f1 = await fields(page);
  const cam1 = await page.evaluate(() => window.topopLite.api.camera());
  const camMoved = Math.hypot(...cam1.map((v, i) => v - cam0[i]));
  const ctlAfter = await page.evaluate(() => window.topopLite.api.controls().enabled);
  const dmax = k1.max[0] - k0.max[0];
  check('drag +x handle: max x changes by a multiple of h, other faces unchanged',
    dmax > 0 && isMult(dmax, h) && k1.min.join() === k0.min.join() && k1.max[1] === k0.max[1] && k1.max[2] === k0.max[2],
    `max x ${k0.max[0]} -> ${k1.max[0]} (${(dmax / h).toFixed(3)} h, h=${h})`);
  check('fields show the dragged box live', Math.abs(f1[3] - k1.max[0]) < 1e-6 && Math.abs(f1[0] - k1.min[0]) < 1e-6, JSON.stringify(f1));
  check('orbit disabled while dragging, restored after', !ctlDuring && ctlAfter && camMoved < 1e-9, `during=${ctlDuring} after=${ctlAfter} camera moved ${camMoved}`);

  // min < max with at least one voxel: drag the -x face far past the +x face.
  dr = await dragHandle(page, 0, 0, 600, -1);
  await page.mouse.move(...dr.from);
  await page.mouse.down();
  await page.mouse.move(...dr.to, { steps: 20 });
  await page.mouse.up();
  const k2 = await sel(page);
  check('dragging min past max stops one voxel short', k2.max[0] - k2.min[0] >= h - 1e-9 && k2.max[0] - k2.min[0] < 2 * h && isMult(k2.min[0] - k0.min[0], h),
    `x ${k2.min[0]} .. ${k2.max[0]}`);
  // Put it back for the rest, by typing (two-way binding: fields -> box).
  await page.fill('#keep-x0', '30');
  await page.fill('#keep-y1', '44');
  const k3 = await sel(page);
  const small = await page.evaluate(() => document.querySelector('#keep-list li small').textContent);
  check('typing in the fields edits the selected box', k3.min[0] === 30 && k3.max[1] === 44 && /44\.0/.test(small), `${JSON.stringify(k3)} "${small}"`);

  // 3. Convert box to voxels keeps the same cell count.
  const nBox = await page.evaluate(() => { const S = window.topopLite; return S.api.regionCells(S.grid, S.keeps.find((k) => k.id === S.keepSel)).size; });
  await page.click('#keep-convert');
  const kc = await sel(page);
  const rowC = (await rows(page))[0];
  const nh2 = await page.evaluate(() => window.topopLite.api.keepHandles().length);
  check('Convert box to voxels: same cell count, no box, no handles, row says voxels (N)',
    kc.min === null && kc.n === nBox && nBox > 0 && nh2 === 0 && rowC.endsWith(`voxels (${nBox.toLocaleString('en-US')})`), `${nBox} -> ${kc.n}, row "${rowC}"`);
  await page.screenshot({ path: path.join(SHOTS, 'keep-converted.png') });

  // 4. Paint three strokes on the plate top into a new keep-in region.
  await page.click('#keep-list li button.t'); // deselect
  check('clicking the selected row deselects it', (await sel(page)) === undefined || (await sel(page)) === null);
  await page.selectOption('#keep-mode', 'solid');
  await page.click('#keep-paint');
  const pOn = await page.evaluate(() => ({ on: window.topopLite.paint.on, opts: !document.getElementById('paint-opts').hidden, pill: !document.getElementById('vp-paint').hidden }));
  const nk0 = await page.evaluate(() => window.topopLite.keeps.length);
  const counts = [];
  for (const y of [10, 30, 50]) {
    await mouseStroke(page, [60, y, 10], [76, y, 10]);
    counts.push((await sel(page))?.n ?? 0);
  }
  const painted = await page.evaluate(() => {
    const S = window.topopLite, g = S.grid, kp = S.keeps.find((k) => k.id === S.keepSel);
    const at = (x, y, z) => (Math.floor(x / g.h - g.origin[0] / g.h) * g.ny + Math.floor((y - g.origin[1]) / g.h)) * g.nz + Math.floor((z - g.origin[2]) / g.h);
    return { mode: kp.mode, n: kp.cells.size, allActive: [...kp.cells].every((e) => g.active[e]), under: [10, 30, 50].map((y) => kp.cells.has(at(68, y, 9))), sel: S.selection, keeps: S.keeps.length };
  });
  check('paint mode: toggle on, options and pill shown', pOn.on && pOn.opts && pOn.pill, JSON.stringify(pOn));
  check('3 strokes add cells to a new keep-in region (cells under each stroke, all active)',
    painted.keeps === nk0 + 1 && painted.mode === 'solid' && counts[0] > 0 && counts[1] > counts[0] && counts[2] > counts[1] && painted.allActive && painted.under.every(Boolean),
    `counts after each stroke ${counts}, ${JSON.stringify({ ...painted, sel: !!painted.sel })}`);
  check('paint mode does not pick faces', painted.sel === null);
  const rowP = (await rows(page))[1];
  check('row shows voxels (N)', rowP.endsWith(`voxels (${painted.n})`), rowP);
  await page.screenshot({ path: path.join(SHOTS, 'keep-painted.png') });

  // 5. Erase the middle stroke.
  await page.click('[data-brush="erase"]');
  await mouseStroke(page, [58, 30, 10], [78, 30, 10], 18);
  const er = await page.evaluate(() => {
    const S = window.topopLite, g = S.grid, kp = S.keeps.find((k) => k.id === S.keepSel);
    const at = (x, y, z) => (Math.floor((x - g.origin[0]) / g.h) * g.ny + Math.floor((y - g.origin[1]) / g.h)) * g.nz + Math.floor((z - g.origin[2]) / g.h);
    return { n: kp.cells.size, under: [10, 30, 50].map((y) => kp.cells.has(at(68, y, 9))) };
  });
  check('erase removes cells under the stroke only', er.n < painted.n && er.n > 0 && er.under[0] && !er.under[1] && er.under[2], `${painted.n} -> ${er.n}, under ${er.under}`);
  await page.click('[data-brush="add"]');

  // 6. A keep-out box plus painted keep-out voxels: row says box + voxels (N).
  await page.keyboard.press('Escape');
  const pOff = await page.evaluate(() => window.topopLite.paint.on);
  check('Esc leaves paint mode', pOff === false);
  await page.keyboard.press('Escape'); // deselect
  for (const [id, v] of [['keep-x0', 0], ['keep-y0', 0], ['keep-z0', 30], ['keep-x1', 10], ['keep-y1', 10], ['keep-z1', 40]]) await page.fill('#' + id, String(v));
  await page.selectOption('#keep-mode', 'void');
  await page.click('#keep-add');
  await page.click('#keep-paint');
  await mouseStroke(page, [10, 18, 25], [10, 46, 25]);
  const kv = await sel(page);
  const rowV = (await rows(page))[2];
  check('painting into a box region: row says box + voxels (N)', kv.mode === 'void' && kv.min && kv.n > 0 && rowV.includes(`box + voxels (${kv.n}) · `) && /cells in part$/.test(rowV), rowV);
  await page.keyboard.press('Escape');

  // Clicking the outline of the keep-out box selects it.
  await page.keyboard.press('Escape');
  const corner = await project(page, [10, 0, 35]);
  await page.mouse.click(corner[0], corner[1]);
  await page.waitForTimeout(100);
  const ko = await sel(page);
  check('clicking a box outline in the view selects that region', ko && ko.id === kv.id, JSON.stringify(ko));

  // 7. Run: passive solid cells stay 1, passive void cells stay 0.
  await setSlider(page, 'maxit', 6);
  await page.click('#run');
  await page.waitForFunction(() => ['done', 'error'].includes(window.topopLite.status), null, { timeout: 240000, polling: 250 });
  const run = await page.evaluate(() => {
    const S = window.topopLite, g = S.grid, rho = S.rhoFinal;
    const out = { status: S.status, err: S.lastError, solid: 0, void: 0, badSolid: 0, badVoid: 0, paintedSolid: 0, paintedVoid: 0 };
    const solid = S.keeps.filter((k) => k.mode === 'solid'), vd = S.keeps.filter((k) => k.mode === 'void');
    for (const kp of solid) for (const e of kp.cells || []) { out.paintedSolid++; if (rho[e] !== 1) out.badSolid++; }
    for (const kp of vd) for (const e of S.api.regionCells(g, kp)) { out.paintedVoid++; if (rho[e] !== 0) out.badVoid++; }
    return out;
  });
  check('run keeps painted keep-in cells at density 1 and keep-out cells at 0',
    run.status === 'done' && run.paintedSolid > 0 && run.paintedVoid > 0 && run.badSolid === 0 && run.badVoid === 0, JSON.stringify(run));
  await page.screenshot({ path: path.join(SHOTS, 'keep-run.png') });

  // 8. Resolution change remaps painted cells by cell centre (coarsen, then refine).
  const snap = () => page.evaluate(() => {
    const S = window.topopLite, g = S.grid;
    return { g: { nx: g.nx, ny: g.ny, nz: g.nz, h: g.h, origin: g.origin }, cells: S.keeps.map((k) => (k.cells ? [...k.cells] : null)) };
  });
  const verify = (old) => page.evaluate((old) => {
    const S = window.topopLite, g = S.grid, o = old.g;
    const cellAt = (G, x, y, z) => {
      const i = Math.floor((x - G.origin[0]) / G.h), j = Math.floor((y - G.origin[1]) / G.h), k = Math.floor((z - G.origin[2]) / G.h);
      return i < 0 || j < 0 || k < 0 || i >= G.nx || j >= G.ny || k >= G.nz ? -1 : (i * G.ny + j) * G.nz + k;
    };
    const res = { missA: 0, missB: 0, extra: 0, before: 0, after: 0 };
    S.keeps.forEach((kp, r) => {
      const oc = old.cells[r];
      if (!oc) return;
      const os = new Set(oc);
      res.before += oc.length;
      res.after += kp.cells.size;
      const expect = new Set();
      for (const e of oc) {
        const k = e % o.nz, t = (e - k) / o.nz, j = t % o.ny, i = (t - j) / o.ny;
        const n = cellAt(g, o.origin[0] + (i + 0.5) * o.h, o.origin[1] + (j + 0.5) * o.h, o.origin[2] + (k + 0.5) * o.h);
        if (n >= 0 && g.active[n]) { expect.add(n); if (!kp.cells.has(n)) res.missA++; }
      }
      for (let i = 0; i < g.nx; i++) for (let j = 0; j < g.ny; j++) for (let k = 0; k < g.nz; k++) {
        const n = (i * g.ny + j) * g.nz + k;
        if (!g.active[n]) continue;
        const c = cellAt(o, g.origin[0] + (i + 0.5) * g.h, g.origin[1] + (j + 0.5) * g.h, g.origin[2] + (k + 0.5) * g.h);
        if (c >= 0 && os.has(c)) { expect.add(n); if (!kp.cells.has(n)) res.missB++; }
      }
      for (const n of kp.cells) if (!expect.has(n)) res.extra++;
    });
    res.note = document.getElementById('keep-note').hidden ? null : document.getElementById('keep-note').textContent;
    return res;
  }, old);
  let old = await snap();
  await setSlider(page, 'res', 30);
  await page.waitForFunction(() => window.topopLite.grid.res === 30);
  const v1 = await verify(old);
  check('coarsen 40 -> 30 remaps painted cells by centre, with a note', v1.after > 0 && v1.missA === 0 && v1.missB === 0 && v1.extra === 0 && /remapped/.test(v1.note || ''), JSON.stringify(v1));
  old = await snap();
  await setSlider(page, 'res', 52);
  await page.waitForFunction(() => window.topopLite.grid.res === 52);
  const v2 = await verify(old);
  check('refine 30 -> 52 fills new cells whose centres fall in painted cells', v2.after > v2.before && v2.missA === 0 && v2.missB === 0 && v2.extra === 0, JSON.stringify(v2));
  const pas = await page.evaluate(() => {
    const S = window.topopLite, p = S.api.buildPassive(S.grid, S.keeps);
    let s = 0, v = 0;
    for (const x of p) { if (x === 1) s++; else if (x === -1) v++; }
    return { s, v };
  });
  check('passive mask after remap has keep-in and keep-out cells', pas.s > 0 && pas.v > 0, JSON.stringify(pas));
  check('no console errors (keep desktop)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

// ---- dark theme: handles and voxels render without errors
{
  const { ctx, page, errors } = await newPage({ viewport: { width: 1280, height: 800 }, colorScheme: 'dark' });
  await page.click('#sec-keep summary');
  await page.click('#keep-add');
  await page.click('#keep-convert');
  await page.click('#keep-add');
  await page.waitForTimeout(200);
  await page.screenshot({ path: path.join(SHOTS, 'keep-dark.png') });
  const axis = await page.evaluate(() => ['--axis-x', '--axis-y', '--axis-z'].map((t) => getComputedStyle(document.documentElement).getPropertyValue(t).trim()));
  check('dark theme: axis tokens defined, no console errors', axis.every((c) => /^#[0-9a-f]{6}$/i.test(c)) && errors.length === 0, `${axis} ${errors.join(' | ')}`);
  await ctx.close();
}

// ---- phone: 390 px, touch drag of a handle, touch paint
{
  const { ctx, page, errors } = await newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, colorScheme: 'light' });
  await page.click('#sec-keep summary');
  await page.click('#keep-add');
  await page.click('#keep-paint');
  await page.waitForTimeout(150);
  const ov = await page.evaluate(() => {
    const W = document.documentElement.clientWidth, over = [];
    for (const el of document.querySelectorAll('#sec-keep *, .viewport *')) {
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height || el.classList.contains('visually-hidden')) continue;
      if (r.right > W + 0.5 || r.left < -0.5) over.push(`${el.tagName}#${el.id}.${el.className} ${Math.round(r.left)}..${Math.round(r.right)}`);
    }
    return { W, doc: document.documentElement.scrollWidth, body: document.body.scrollWidth, over: over.slice(0, 6) };
  });
  check('390px with keep panel and paint options: no horizontal scroll', ov.W === 390 && ov.doc <= 390 && ov.body <= 390 && ov.over.length === 0, JSON.stringify(ov));
  await page.screenshot({ path: path.join(SHOTS, 'keep-phone.png') });
  await page.click('#keep-paint'); // off: the handle drag below must not paint
  const h = await page.evaluate(() => window.topopLite.grid.h);
  const k0 = await sel(page);
  const cam0 = await page.evaluate(() => window.topopLite.api.camera());
  const cdp = await ctx.newCDPSession(page);
  const touchDrag = async (from, to, n = 10) => {
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: from[0], y: from[1] }] });
    for (let i = 1; i <= n; i++) await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: from[0] + ((to[0] - from[0]) * i) / n, y: from[1] + ((to[1] - from[1]) * i) / n }] });
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    await page.waitForTimeout(100);
  };
  const dr = await dragHandle(page, 2, 1, 40);
  await touchDrag(dr.from, dr.to);
  const k1 = await sel(page);
  const cam1 = await page.evaluate(() => window.topopLite.api.camera());
  const dz = k1.max[2] - k0.max[2];
  check('touch drag of the +z handle moves max z by whole voxels without orbiting', dz > 0 && isMult(dz, h) && Math.hypot(...cam1.map((v, i) => v - cam0[i])) < 1e-9,
    `max z ${k0.max[2]} -> ${k1.max[2]} (${(dz / h).toFixed(3)} h)`);
  await page.click('#keep-paint');
  await page.evaluate(() => window.topopLite.api.selectKeep(null));
  const a = await project(page, [62, 50, 10]), b = await project(page, [76, 50, 10]);
  await touchDrag(a, b, 12);
  const tp = await sel(page);
  check('touch paint stroke adds cells', tp && tp.n > 0 && tp.min === null, JSON.stringify(tp));
  check('no console errors (keep phone)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

await browser.close();
console.log(failures.length ? `\n${failures.length} check(s) failed: ${failures.join('; ')}` : '\nall checks passed');
process.exit(failures.length ? 1 : 0);
