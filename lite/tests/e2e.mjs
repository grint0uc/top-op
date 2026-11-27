// End-to-end check of top-op lite in headless Chromium.
// Usage: serve lite/ (python -m http.server 8910 -d lite), then
//   PW_MODULE=/path/to/node_modules/playwright-core THREE_DIR=/path/to/three-0.147.0/package \
//   CHROMIUM=/opt/pw-browsers/chromium-1243/chrome-linux64/chrome node lite/tests/e2e.mjs
// THREE_DIR serves the pinned jsDelivr three.js files locally when the CDN is unreachable.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PW_MODULE || 'playwright-core');
const BASE = process.env.BASE_URL || 'http://127.0.0.1:8910/index.html';
const THREE_DIR = process.env.THREE_DIR;
const SHOTS = process.env.SHOTS || path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'screens');
const MEASURE = process.env.MEASURE !== '0';
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
  // Wrap the page in the skeleton the Artifact host adds at publish time.
  await page.route(BASE, async (route) => {
    const res = await route.fetch();
    const body = SKELETON_HEAD + (await res.text()) + '</body></html>';
    route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body });
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

async function runAndWait(page, timeout = 240000) {
  await page.click('#run');
  await page.waitForFunction(() => ['done', 'error'].includes(window.topopLite.status), null, { timeout, polling: 250 });
  return page.evaluate(() => ({
    status: window.topopLite.status,
    error: window.topopLite.lastError,
    history: window.topopLite.history,
    ne: window.topopLite.grid.count,
    surface: window.topopLite.surface && { ntri: window.topopLite.surface.ntri, volume: window.topopLite.surface.volume },
  }));
}

// ---- desktop, light: the default example runs 20 iterations
{
  const { ctx, page, errors } = await newPage({ viewport: { width: 1440, height: 900 }, colorScheme: 'light' });
  const st = await page.evaluate(() => window.topopLite.selftest);
  check('KE self-test (symmetric, 6 zero modes, kernels)', st.ok, `zeroModes=${st.zeroModes} asym=${st.asym}`);
  const info = await page.evaluate(() => ({ bcs: window.topopLite.bcs.map((b) => [b.kind, b.nodes.length]), ne: window.topopLite.grid.count, name: window.topopLite.mesh.name }));
  check('example loads with a support and a load', info.bcs.length === 2 && info.bcs.every(([, n]) => n > 0), JSON.stringify(info));
  await setSlider(page, 'maxit', 20);
  const r = await runAndWait(page);
  check('run finished', r.status === 'done', r.error || '');
  const c = r.history.map((x) => x.c);
  let mono = true, worst = 0;
  for (let i = 1; i < c.length; i++) {
    const rise = (c[i] - c[i - 1]) / c[i - 1];
    worst = Math.max(worst, rise);
    if (i < 5 ? rise > 0 : rise > 0.02) mono = false;
  }
  check('20 iterations', r.history.length === 20, `${r.history.length}`);
  check('compliance decreases', mono, `c0=${c[0].toExponential(3)} c20=${c[c.length - 1].toExponential(3)} worst rise ${(worst * 100).toFixed(2)}%`);
  const vol = r.history[r.history.length - 1].vol;
  check('final volume within 0.01 of 0.30', Math.abs(vol - 0.3) <= 0.01, vol.toFixed(4));
  check('marching cubes surface non-empty', r.surface && r.surface.ntri > 0, JSON.stringify(r.surface));
  const avg = r.history.slice(1).reduce((s, x) => s + x.ms, 0) / (r.history.length - 1);
  console.log(`  default example: ${r.ne} elements, ${avg.toFixed(0)} ms/iter avg, CG ${r.history.map((x) => x.cg).join(',')}`);
  const exportState = await page.evaluate(() => ({ exportHidden: document.getElementById('export').hidden, copyDisabled: document.getElementById('copy').disabled, note: document.getElementById('export-note').textContent }));
  check('export hidden without downloads capability; copy available', exportState.exportHidden && !exportState.copyDisabled, JSON.stringify(exportState));
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(SHOTS, 'desktop-light.png') });
  // density view screenshot too
  await page.click('.seg [data-view="voxels"]');
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(SHOTS, 'desktop-light-density.png') });
  // click-to-select on the design view
  await page.click('.seg [data-view="design"]');
  const vp = await page.locator('#viewport').boundingBox();
  await page.mouse.click(vp.x + vp.width * 0.5, vp.y + vp.height * 0.55);
  await page.waitForTimeout(200);
  const sel = await page.evaluate(() => window.topopLite.selection && { tris: window.topopLite.selection.tris.length, nodes: window.topopLite.selection.nodes.length });
  check('clicking a face selects a flat face group', !!sel && sel.tris > 0 && sel.nodes > 0, JSON.stringify(sel));
  await page.screenshot({ path: path.join(SHOTS, 'desktop-light-selection.png') });
  check('no console errors (desktop light)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

// ---- desktop, dark
{
  const { ctx, page, errors } = await newPage({ viewport: { width: 1440, height: 900 }, colorScheme: 'dark' });
  await page.selectOption('#example', 'cantilever');
  await setSlider(page, 'res', 40);
  await page.waitForTimeout(400);
  await setSlider(page, 'maxit', 30);
  const r = await runAndWait(page);
  check('dark run finished', r.status === 'done', `${r.ne} elements, ${r.history.length} it`);
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(SHOTS, 'desktop-dark.png') });
  check('no console errors (desktop dark)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

// ---- phone
{
  const { ctx, page, errors } = await newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, colorScheme: 'light' });
  const sw = await page.evaluate(() => ({ doc: document.documentElement.scrollWidth, body: document.body.scrollWidth, w: window.innerWidth }));
  check('no horizontal scroll at 390px', sw.doc <= sw.w && sw.body <= sw.w, JSON.stringify(sw));
  await setSlider(page, 'maxit', 12);
  const r = await runAndWait(page);
  check('phone run finished', r.status === 'done');
  const sw2 = await page.evaluate(() => ({ doc: document.documentElement.scrollWidth, w: window.innerWidth }));
  check('no horizontal scroll at 390px after run', sw2.doc <= sw2.w, JSON.stringify(sw2));
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(SHOTS, 'phone.png') });
  check('no console errors (phone)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

// ---- downloads capability (mocked) and a keep-out box
{
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  await ctx.addInitScript(() => {
    window.__saved = [];
    const downloads = {
      save: async (req) => {
        const size = req.data.byteLength ?? req.data.size;
        window.__saved.push({ filename: req.filename, size });
        if (!req.filename.endsWith('.zip')) throw { code: 'rejected_extension', message: 'not on the allowlist' };
        return { status: 'saved' };
      },
    };
    window.claude = { use: async (name) => (name === 'downloads' ? downloads : null) };
  });
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
  await page.waitForFunction(() => window.topopLite?.grid && window.topopLite.downloads, null, { timeout: 30000 });
  await page.selectOption('#example', 'cantilever');
  await setSlider(page, 'res', 30);
  await page.waitForFunction(() => window.topopLite.grid.res === 30);
  await page.click('#sec-keep summary');
  for (const [id, v] of [['keep-x0', 25], ['keep-y0', -1], ['keep-z0', 6], ['keep-x1', 35], ['keep-y1', 21], ['keep-z1', 14]]) await page.fill('#' + id, String(v));
  await page.selectOption('#keep-mode', 'void');
  await page.click('#keep-add');
  await setSlider(page, 'maxit', 6);
  const r = await runAndWait(page);
  const keep = await page.evaluate(() => {
    const g = window.topopLite.grid, rho = window.topopLite.rhoFinal;
    let n = 0, maxRho = 0;
    for (let i = 0; i < g.nx; i++) for (let j = 0; j < g.ny; j++) for (let k = 0; k < g.nz; k++) {
      const x = g.origin[0] + (i + 0.5) * g.h, z = g.origin[2] + (k + 0.5) * g.h;
      if (x >= 25 && x <= 35 && z >= 6 && z <= 14) { n++; maxRho = Math.max(maxRho, rho[(i * g.ny + j) * g.nz + k]); }
    }
    return { n, maxRho };
  });
  check('keep-out box stays void', r.status === 'done' && keep.n > 0 && keep.maxRho === 0, JSON.stringify(keep));
  const visible = await page.evaluate(() => !document.getElementById('export').hidden && !document.getElementById('export').disabled);
  check('Save STL shown when downloads capability exists', visible);
  await page.click('#export');
  await page.waitForFunction(() => window.__saved.length >= 1, null, { timeout: 5000 }).catch(() => {});
  const saved = await page.evaluate(() => ({ saved: window.__saved, note: document.getElementById('export-note').textContent }));
  check('Save STL saves topop-result.zip (allowlisted) in one prompt', saved.saved.length === 1 && saved.saved[0].filename === 'topop-result.zip' && saved.saved[0].size > 84 + 50, JSON.stringify(saved));
  check('no console errors (downloads/keep)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

// ---- timing at ~10k and ~30k active elements
if (MEASURE) {
  const { ctx, page } = await newPage({ viewport: { width: 1280, height: 800 } });
  for (const res of [40, 57]) {
    await setSlider(page, 'res', res);
    await page.waitForFunction((r) => window.topopLite.grid.res === r, res, { timeout: 10000 });
    await setSlider(page, 'maxit', 8);
    const r = await runAndWait(page, 600000);
    const ms = r.history.slice(1).map((x) => x.ms);
    console.log(`  timing: res ${res}: ${r.ne} active elements, ${(ms.reduce((a, b) => a + b, 0) / ms.length).toFixed(0)} ms/iter (iters 2-8), CG ${r.history.map((x) => x.cg).join(',')}`);
  }
  await ctx.close();
}

await browser.close();
console.log(failures.length ? `\n${failures.length} check(s) failed: ${failures.join('; ')}` : '\nall checks passed');
process.exit(failures.length ? 1 : 0);
