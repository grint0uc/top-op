// Review checks for top-op lite: Artifact rules, numerics, interaction, robustness.
// Usage: python -m http.server 8910 -d lite, then
//   PW_MODULE=web/node_modules/playwright-core THREE_DIR=/path/to/three-0.147.0/package \
//   CHROMIUM=/opt/pw-browsers/chromium-1243/chrome-linux64/chrome node lite/tests/review.mjs
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PW_MODULE || 'playwright-core');
const BASE = process.env.BASE_URL || 'http://127.0.0.1:8910/index.html';
const THREE_DIR = process.env.THREE_DIR;
const LITE = path.join(path.dirname(new URL(import.meta.url).pathname), '..');
const SKELETON_HEAD = '<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">'
  + '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
  + 'body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style></head><body>';

const results = [];
const check = (name, ok, detail = '') => {
  results.push({ name, passed: !!ok, detail: String(detail) });
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? ' - ' + detail : ''}`);
};

// ---------------------------------------------------------------- static Artifact rules
{
  const html = fs.readFileSync(path.join(LITE, 'index.html'), 'utf8');
  const js = ['app.js', 'solver.js', 'mc.js'].map((f) => fs.readFileSync(path.join(LITE, f), 'utf8')).join('\n');
  const skel = html.match(/<!doctype|<\/?html[\s>]|<\/?head[\s>]|<\/?body[\s>]/gi);
  check('no doctype/html/head/body tags', !skel, skel ? skel.join(' ') : 'none');
  const firstLines = html.split('\n').slice(0, 5).join('\n');
  check('<title> near the top', /<title>[^<]+<\/title>/.test(firstLines), firstLines.split('\n')[0]);
  const srcs = [...html.matchAll(/<script[^>]*\ssrc="([^"]+)"/g)].map((m) => m[1]);
  const ext = srcs.filter((s) => /^https?:/.test(s));
  const pinned = (s) => /^https:\/\/cdn\.jsdelivr\.net\/npm\/(@[\w.-]+\/)?[\w.-]+@\d+\.\d+\.\d+\//.test(s)
    || /^https:\/\/cdnjs\.cloudflare\.com\/ajax\/libs\/[\w.-]+\/\d+(\.\d+)*\//.test(s);
  check('external scripts pinned to exact versions on jsDelivr/cdnjs', ext.length > 0 && ext.every(pinned), ext.join(' '));
  const links = [...html.matchAll(/<link[^>]*href="([^"]+)"/g)].map((m) => m[1]);
  check('no external stylesheets/preconnects', links.length === 0, links.join(' ') || 'none');
  const banned = (js + html).match(/\b(alert|confirm|prompt)\s*\(|window\.print|\bprint\s*\(\s*\)|<a[^>]*\sdownload\b|\.download\s*=/g);
  check('no alert/confirm/prompt/print/<a download>', !banned, banned ? banned.join(' ') : 'none');
  check('export goes through window.claude?.use?.("downloads")', /window\.claude\?\.use\?\.\(\s*['"]downloads['"]\s*\)/.test(js));
}

// ---------------------------------------------------------------- helpers
function boxTris(x0, y0, z0, x1, y1, z1, div = 1, skipTop = false) {
  // Outward-facing triangles of an axis-aligned box, each face split into div x div quads.
  const tris = [];
  const lerp = (a, b, t) => a + (b - a) * t;
  const face = (P) => {
    for (let i = 0; i < div; i++) for (let j = 0; j < div; j++) {
      const a = P(i / div, j / div), b = P((i + 1) / div, j / div), c = P((i + 1) / div, (j + 1) / div), d = P(i / div, (j + 1) / div);
      tris.push([a, b, c], [a, c, d]);
    }
  };
  face((u, v) => [lerp(x0, x1, v), lerp(y0, y1, u), z0]); // -z
  if (!skipTop) face((u, v) => [lerp(x0, x1, u), lerp(y0, y1, v), z1]); // +z
  face((u, v) => [lerp(x0, x1, u), y0, lerp(z0, z1, v)]); // -y
  face((u, v) => [lerp(x0, x1, v), y1, lerp(z0, z1, u)]); // +y
  face((u, v) => [x0, lerp(y0, y1, v), lerp(z0, z1, u)]); // -x
  face((u, v) => [x1, lerp(y0, y1, u), lerp(z0, z1, v)]); // +x
  return tris;
}
function stlAscii(tris) {
  const out = ['solid test'];
  for (const t of tris) {
    out.push(' facet normal 0 0 0', '  outer loop');
    for (const v of t) out.push(`   vertex ${v[0]} ${v[1]} ${v[2]}`);
    out.push('  endloop', ' endfacet');
  }
  out.push('endsolid test', '');
  return Buffer.from(out.join('\n'));
}
function stlBinary(tris) {
  const b = Buffer.alloc(84 + 50 * tris.length);
  b.write('solid but actually binary', 0); // header starting with "solid" must still parse as binary
  b.writeUInt32LE(tris.length, 80);
  tris.forEach((t, i) => {
    const o = 84 + 50 * i + 12;
    t.forEach((v, k) => v.forEach((x, d) => b.writeFloatLE(x, o + 12 * k + 4 * d)));
  });
  return b;
}

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM || undefined,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const origin = new URL(BASE).origin;
const allowedHost = (u) => u.startsWith(origin + '/') || u.startsWith('blob:') || u.startsWith('data:')
  || /^https:\/\/cdn\.jsdelivr\.net\/npm\//.test(u) || /^https:\/\/cdnjs\.cloudflare\.com\/ajax\/libs\//.test(u);

async function newPage(opts = {}, { wrap = true, init } = {}) {
  const ctx = await browser.newContext(opts);
  if (init) await ctx.addInitScript(init);
  const page = await ctx.newPage();
  const errors = [], requests = [], dialogs = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('request', (r) => requests.push(r.url()));
  page.on('dialog', (d) => { dialogs.push(d.type()); d.dismiss(); });
  await page.route(/^https:\/\/cdn\.jsdelivr\.net\/npm\/three@0\.147\.0\//, async (route) => {
    if (!THREE_DIR) return route.continue();
    const rel = new URL(route.request().url()).pathname.replace('/npm/three@0.147.0/', '');
    route.fulfill({ status: 200, contentType: 'text/javascript', body: fs.readFileSync(path.join(THREE_DIR, rel)) });
  });
  if (wrap) {
    await page.route(BASE, async (route) => {
      const res = await route.fetch();
      route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: SKELETON_HEAD + (await res.text()) + '</body></html>' });
    });
  }
  await page.goto(BASE);
  await page.waitForFunction(() => window.topopLite?.grid && window.topopLite?.selftest && window.topopLite.downloads !== undefined, null, { timeout: 30000 });
  return { ctx, page, errors, requests, dialogs };
}

async function setSlider(page, id, value) {
  await page.evaluate(([id, v]) => {
    const el = document.getElementById(id);
    el.value = String(v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, [id, value]);
}
async function waitDone(page, timeout = 300000) {
  await page.waitForFunction(() => ['done', 'error'].includes(window.topopLite.status), null, { timeout, polling: 200 });
  return page.evaluate(() => ({
    status: window.topopLite.status, error: window.topopLite.lastError, history: window.topopLite.history,
    ne: window.topopLite.grid.count, statusText: document.getElementById('vp-status-text').textContent,
  }));
}
async function run(page, timeout) {
  await page.click('#run');
  return waitDone(page, timeout);
}

// ---------------------------------------------------------------- numerics (desktop, light)
{
  const { ctx, page, errors, requests, dialogs } = await newPage({ viewport: { width: 1440, height: 900 }, colorScheme: 'light' });
  const st = await page.evaluate(() => window.topopLite.selftest);
  check('KE self-test: symmetric, 6 rigid modes', st.ok && st.symmetric && st.zeroModes === 6 && st.negative === 0,
    `asym=${st.asym.toExponential(1)} zeroModes=${st.zeroModes} negative=${st.negative} kernelErr=${st.kernelError.toExponential(1)}`);

  // Export UI degrades when window.claude is absent.
  const ex = await page.evaluate(() => ({ hasClaude: !!window.claude, hidden: document.getElementById('export').hidden, note: document.getElementById('export-note').textContent }));
  check('Save hidden with a note when downloads resolves null/undefined', !ex.hasClaude && ex.hidden && /not available/i.test(ex.note), JSON.stringify(ex));

  // Cantilever: two chords.
  await page.selectOption('#example', 'cantilever');
  await setSlider(page, 'res', 48);
  await page.waitForFunction(() => window.topopLite.grid.res === 48);
  await setSlider(page, 'maxit', 40);
  const r = await run(page);
  const c = r.history.map((x) => x.c);
  let rises = 0, worst = 0;
  for (let i = 1; i < c.length; i++) { const d = (c[i] - c[i - 1]) / c[i - 1]; worst = Math.max(worst, d); if (d > 1e-3) rises++; }
  const finite = r.history.every((x) => Number.isFinite(x.c) && Number.isFinite(x.vol));
  check('cantilever compliance decreasing', r.status === 'done' && finite && c[c.length - 1] < 0.5 * c[0] && worst < 0.01,
    `${r.ne} el, ${c.length} it, c ${c[0].toExponential(3)} -> ${c[c.length - 1].toExponential(3)}, worst rise ${(worst * 100).toFixed(3)}%, rises>0.1%: ${rises}`);
  const vol = r.history[r.history.length - 1].vol;
  check('cantilever volume within 0.01 of target', Math.abs(vol - 0.3) <= 0.01, vol.toFixed(4));
  const bands = await page.evaluate(() => {
    const g = window.topopLite.grid, rho = window.topopLite.rhoFinal;
    const out = {};
    for (const [name, lo, hi] of [['nearSupport', 0, 0.5], ['farHalf', 0.5, 1]]) {
      const s = [0, 0, 0], n = [0, 0, 0];
      for (let i = Math.floor(lo * g.nx); i < Math.floor(hi * g.nx); i++) for (let j = 0; j < g.ny; j++) for (let k = 0; k < g.nz; k++) {
        const b = Math.min(2, Math.floor((3 * k) / g.nz));
        s[b] += rho[(i * g.ny + j) * g.nz + k]; n[b]++;
      }
      out[name] = s.map((v, b) => +(v / n[b]).toFixed(3)); // [bottom, middle, top]
    }
    return out;
  });
  const chords = (b) => b[0] > b[1] && b[2] > b[1];
  check('cantilever two-chord layout (top and bottom thirds denser than middle)', chords(bands.farHalf) && chords(bands.nearSupport),
    `[bottom, middle, top] mean density: support half ${JSON.stringify(bands.nearSupport)}, far half ${JSON.stringify(bands.farHalf)}`);

  // Stop mid-run.
  await setSlider(page, 'maxit', 200);
  await page.click('#run');
  await page.waitForFunction(() => window.topopLite.history.length >= 3, null, { timeout: 60000 });
  await page.click('#stop');
  const rs = await waitDone(page, 60000);
  const ui = await page.evaluate(() => ({ run: document.getElementById('run').disabled, stop: document.getElementById('stop').disabled, res: document.getElementById('res').disabled }));
  check('Stop works mid-run', rs.status === 'done' && rs.history.length < 10 && /Stopped/.test(rs.statusText) && !ui.run && ui.stop && !ui.res,
    `${rs.history.length} it, "${rs.statusText}", ${JSON.stringify(ui)}`);

  // Change resolution and re-run.
  const ne0 = rs.ne;
  await setSlider(page, 'res', 30);
  await page.waitForFunction(() => window.topopLite.grid.res === 30);
  await setSlider(page, 'maxit', 8);
  const r2 = await run(page);
  const fin2 = r2.history.every((x) => Number.isFinite(x.c));
  const bcNodes = await page.evaluate(() => window.topopLite.bcs.map((b) => b.nodes.length));
  check('changing resolution and re-running works', r2.status === 'done' && r2.ne !== ne0 && r2.history.length === 8 && fin2 && bcNodes.every((n) => n > 0),
    `${ne0} -> ${r2.ne} el, ${r2.history.length} it, bc nodes ${bcNodes}`);

  // Load placed on the support-only face.
  await page.evaluate(() => {
    const S = window.topopLite;
    const load = S.bcs.find((b) => b.kind === 'load');
    S.api.removeBC(load.id);
    const sup = S.bcs.find((b) => b.kind === 'support');
    S.api.addBC('load', sup.tris, [0, 0, -1000]);
  });
  await page.click('#run');
  await page.waitForTimeout(1500);
  const r3 = await page.evaluate(() => ({ status: window.topopLite.status, hist: window.topopLite.history.map((x) => x.c), msg: document.getElementById('run-note').textContent, vp: document.getElementById('vp-status-text').textContent }));
  check('load on a support-only face is rejected with a message', r3.status === 'error' && r3.hist.length === 0 && /fixed|support/i.test(r3.msg), JSON.stringify(r3));

  check('no dialogs', dialogs.length === 0, dialogs.join(','));
  const bad = requests.filter((u) => !allowedHost(u));
  check('no network requests outside the page and pinned CDN', bad.length === 0, bad.slice(0, 5).join(' ') || `${requests.length} requests, all allowed`);
  check('no console errors (numerics)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

// ---------------------------------------------------------------- interaction: face pick on a finely tessellated part
{
  const { ctx, page, errors } = await newPage({ viewport: { width: 1280, height: 800 }, colorScheme: 'light' });
  // Default bracket: click the plate top (z = 10, x 10..80, y 0..60).
  await page.evaluate(() => { document.getElementById('sel-clear').click(); });
  const pickAt = async (p) => {
    const xy = await page.evaluate((p) => window.topopLite.api.project(p), p);
    await page.mouse.click(xy[0], xy[1]);
    await page.waitForTimeout(150);
    return page.evaluate(() => {
      const S = window.topopLite, sel = S.selection;
      if (!sel) return null;
      return { tris: sel.tris.length, area: +sel.area.toFixed(3), normal: sel.normal.map((x) => +x.toFixed(3)), nodes: sel.nodes.length };
    });
  };
  const s1 = await pickAt([45, 30, 10]);
  check('click on bracket plate top selects the whole planar face', s1 && Math.abs(s1.area - 4200) < 1e-3 && s1.normal[2] === 1, JSON.stringify(s1));
  // Imported box with every face split into 8 x 8 quads (128 triangles per face).
  await page.setInputFiles('#file', { name: 'fine-box.stl', mimeType: 'application/octet-stream', buffer: stlBinary(boxTris(0, 0, 0, 40, 20, 10, 8)) });
  await page.waitForFunction(() => window.topopLite.mesh.name === 'fine-box.stl');
  const s2 = await pickAt([20, 10, 10]);
  check('face pick + flat grow on a tessellated face selects all of it', s2 && s2.tris === 128 && Math.abs(s2.area - 800) < 1e-2, JSON.stringify(s2));
  check('no console errors (picking)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

// ---------------------------------------------------------------- phone: touch orbit, width, tap select
{
  const { ctx, page, errors } = await newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, colorScheme: 'light' });
  const ov = await page.evaluate(() => {
    const W = document.documentElement.clientWidth;
    const over = [];
    for (const el of document.querySelectorAll('body *')) {
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      // Ignore content inside a horizontal scroller (the toolbar) and visually-hidden inputs.
      let p = el.parentElement, scrolled = false;
      while (p && p !== document.body) { const cs = getComputedStyle(p); if (/(auto|scroll|hidden)/.test(cs.overflowX) && p.scrollWidth > p.clientWidth) { scrolled = true; break; } p = p.parentElement; }
      if (scrolled || el.classList.contains('visually-hidden')) continue;
      if (r.right > W + 0.5 || r.left < -0.5) over.push(`${el.tagName}.${el.className}#${el.id} ${Math.round(r.left)}..${Math.round(r.right)}`);
    }
    return { W, doc: document.documentElement.scrollWidth, body: document.body.scrollWidth, over: over.slice(0, 6) };
  });
  check('phone 390px: no horizontal scroll or overflowing elements', ov.W === 390 && ov.doc <= 390 && ov.body <= 390 && ov.over.length === 0, JSON.stringify(ov));
  const ta = await page.evaluate(() => ({ touchAction: getComputedStyle(document.getElementById('gl')).touchAction, enabled: window.topopLite.api.controls().enabled, rotate: window.topopLite.api.controls().enableRotate }));
  const cam0 = await page.evaluate(() => window.topopLite.api.camera());
  const box = await page.locator('#gl').boundingBox();
  const cdp = await ctx.newCDPSession(page);
  const x = box.x + box.width / 2, y = box.y + box.height / 2;
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] });
  for (let i = 1; i <= 10; i++) await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: x + 8 * i, y: y + 3 * i }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await page.waitForTimeout(100);
  const cam1 = await page.evaluate(() => window.topopLite.api.camera());
  const moved = Math.hypot(...cam1.map((v, i) => v - cam0[i]));
  check('touch orbit enabled (one-finger drag rotates the camera)', ta.touchAction === 'none' && ta.enabled && ta.rotate && moved > 1, `${JSON.stringify(ta)} camera moved ${moved.toFixed(2)}`);
  // Tap selects a face.
  await page.evaluate(() => window.topopLite.api.fit());
  const xy = await page.evaluate(() => window.topopLite.api.project([45, 30, 10]));
  await page.touchscreen.tap(xy[0], xy[1]);
  await page.waitForTimeout(200);
  const sel = await page.evaluate(() => window.topopLite.selection && window.topopLite.selection.area);
  check('tap selects a face on touch screens', Math.abs((sel || 0) - 4200) < 1e-3, String(sel));
  check('no console errors (phone)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

// ---------------------------------------------------------------- themes: contrast of text tokens
function lum([r, g, b]) {
  const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}
const contrast = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
for (const [label, scheme, attr] of [['light', 'light', null], ['dark', 'dark', null], ['data-theme=dark on light OS', 'light', 'dark'], ['data-theme=light on dark OS', 'dark', 'light']]) {
  const { ctx, page, errors } = await newPage({ viewport: { width: 1280, height: 800 }, colorScheme: scheme });
  if (attr) await page.evaluate((a) => document.documentElement.setAttribute('data-theme', a), attr);
  await page.waitForTimeout(100);
  const cols = await page.evaluate(() => {
    const rgb = (s) => s.match(/[\d.]+/g).slice(0, 3).map(Number);
    const cs = (sel, p = 'color') => rgb(getComputedStyle(document.querySelector(sel))[p]);
    return {
      body: [cs('body'), cs('body', 'backgroundColor')],
      kv: [cs('#part-name'), cs('.panel', 'backgroundColor')],
      muted: [cs('.kv dt'), cs('.panel', 'backgroundColor')],
      tag: [cs('#grid-tag'), cs('.panel', 'backgroundColor')],
      empty: [cs('.bc-list li small'), cs('.bc-list li', 'backgroundColor')],
      hint: [cs('.vp-hint'), cs('.about', 'backgroundColor')], // pill over the view is ~--surface
      warnNote: [rgb(getComputedStyle(document.documentElement).getPropertyValue('--warn').trim().replace(/^#(..)(..)(..)$/, (m, a, b, c) => `rgb(${parseInt(a, 16)},${parseInt(b, 16)},${parseInt(c, 16)})`)), cs('.panel', 'backgroundColor')],
    };
  });
  const ratios = Object.fromEntries(Object.entries(cols).map(([k, [fg, bg]]) => [k, +contrast(fg, bg).toFixed(2)]));
  const ok = ratios.body >= 7 && ratios.kv >= 7 && ratios.muted >= 4.5 && ratios.tag >= 4.5 && ratios.empty >= 4.5 && ratios.warnNote >= 4.5 && ratios.hint >= 4.5;
  check(`theme ${label}: text legible (contrast)`, ok && errors.length === 0, JSON.stringify(ratios));
  await ctx.close();
}

// ---------------------------------------------------------------- robustness: STL import
{
  const { ctx, page, errors } = await newPage({ viewport: { width: 1280, height: 800 } });
  const imp = async (name, buffer) => {
    await page.evaluate(() => { window.topopLite.__n = (window.topopLite.__n || 0) + 1; });
    await page.setInputFiles('#file', { name, mimeType: 'application/octet-stream', buffer });
    await page.waitForTimeout(400);
    return page.evaluate(() => ({
      name: window.topopLite.mesh.name, ntri: window.topopLite.mesh.ntri, active: window.topopLite.grid.count,
      status: document.getElementById('vp-status').dataset.state, text: document.getElementById('vp-status-text').textContent,
      partNote: document.getElementById('part-note')?.textContent || '', partNoteHidden: document.getElementById('part-note')?.hidden ?? true,
      runNote: document.getElementById('run-note').textContent,
    }));
  };
  const a = await imp('ascii-box.stl', stlAscii(boxTris(0, 0, 0, 30, 20, 10, 2)));
  check('ASCII STL import', a.name === 'ascii-box.stl' && a.ntri === 48 && a.active > 0 && a.status !== 'error' && a.partNoteHidden, JSON.stringify(a));
  const b = await imp('binary-box.stl', stlBinary(boxTris(0, 0, 0, 30, 20, 10, 1)));
  check('binary STL import (header starts with "solid")', b.name === 'binary-box.stl' && b.ntri === 12 && b.active > 0 && b.status !== 'error' && b.partNoteHidden, JSON.stringify(b));
  const o = await imp('open-box.stl', stlBinary(boxTris(0, 0, 0, 30, 20, 10, 1, true)));
  check('non-watertight STL gives a warning, not a crash', o.name === 'open-box.stl' && !o.partNoteHidden && /not (closed|watertight)|open edge/i.test(o.partNote), JSON.stringify(o));
  const e = await imp('empty.stl', Buffer.alloc(0));
  check('empty file gives a clear message', e.name === 'open-box.stl' && e.status === 'error' && /is empty/i.test(e.text), JSON.stringify(e));
  const g = await imp('garbage.stl', Buffer.from('hello world, not an stl'));
  check('garbage file gives a clear message', g.status === 'error' && /no triangles/i.test(g.text) && !/DataView|Offset/.test(g.text), JSON.stringify(g));
  check('no console errors (import)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

// ---------------------------------------------------------------- downloads capability (mocked) and raw (unwrapped) load
{
  const { ctx, page, errors } = await newPage({ viewport: { width: 1280, height: 800 } }, {
    init: () => {
      window.__saved = [];
      const downloads = {
        save: async (req) => {
          const bytes = req.data instanceof Uint8Array ? req.data : new Uint8Array(req.data);
          window.__saved.push({ filename: req.filename, size: bytes.byteLength, head: Array.from(bytes.slice(0, 4)), b64: btoa(String.fromCharCode(...bytes)) });
          if (!/\.(gif|png|jpg|jpeg|webp|mp4|webm|mov|txt|json|md|docx|pptx|epub|csv|ttf|html|svg|pdf|xlsx|zip|fig)$/.test(req.filename)) throw { code: 'rejected_extension', message: 'no' };
          return { status: 'saved' };
        },
      };
      window.claude = { use: async (n) => (n === 'downloads' ? downloads : null) };
    },
  });
  await page.selectOption('#example', 'cantilever');
  await setSlider(page, 'res', 24);
  await page.waitForFunction(() => window.topopLite.grid.res === 24);
  await setSlider(page, 'maxit', 5);
  await run(page);
  await page.click('#export');
  await page.waitForFunction(() => window.__saved.length >= 1, null, { timeout: 5000 }).catch(() => {});
  await page.waitForTimeout(200);
  const s = await page.evaluate(() => ({ saved: window.__saved, note: document.getElementById('export-note').textContent, ntri: window.topopLite.surface.ntri }));
  let zipOk = s.saved.length === 1 && s.saved[0].filename.endsWith('.zip') && s.saved[0].head.join() === '80,75,3,4';
  let zipInfo = '';
  if (zipOk) {
    // Validate the archive with Python's zipfile: one entry, CRC ok, a binary STL of the right size.
    const { execFileSync } = await import('node:child_process');
    const py = 'import sys,io,zipfile,struct;z=zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read()));assert z.testzip() is None;n=z.namelist();d=z.read(n[0]);print(n, len(d), struct.unpack("<I",d[80:84])[0])';
    zipInfo = execFileSync('python3', ['-I', '-c', py], { input: Buffer.from(s.saved[0].b64, 'base64') }).toString().trim();
    zipOk = zipInfo === `['topop-result.stl'] ${84 + 50 * s.ntri} ${s.ntri}`;
  }
  for (const x of s.saved) delete x.b64;
  check('Save offers an allowlisted file (zip with a valid binary STL), one prompt per click', zipOk && /saved/i.test(s.note), JSON.stringify(s) + ' ' + zipInfo);
  check('no console errors (downloads)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}
{
  const { ctx, page, errors } = await newPage({ viewport: { width: 1280, height: 800 } }, { wrap: false });
  check('raw page (no host skeleton) loads without errors', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

await browser.close();
fs.writeFileSync(process.env.OUT || '/dev/null', JSON.stringify(results, null, 1));
const failed = results.filter((r) => !r.passed);
console.log(failed.length ? `\n${failed.length} failed` : '\nall passed');
process.exit(failed.length ? 1 : 0);
