// In-page checks the axe run cannot resolve: text over translucent overlays and gradients, placeholders, control
// boundaries (WCAG 1.4.3 / 1.4.11) and the focus indicator of whatever has focus (2.4.7 + 1.4.11).
import type { Page } from '@playwright/test';

export interface ContrastIssue {
  what: string;
  text: string;
  ratio: number;
  need: number;
}

export interface FocusInfo {
  what: string;
  /** an outline or box-shadow is drawn */
  visible: boolean;
  /** outline / box-shadow colour against the colour behind it */
  ratio: number;
}

// Everything below runs in the page: one self-contained source string shared by both checks.
const LIB = `
const parse = (c) => {
  if (!c || c === 'transparent') return [0, 0, 0, 0];
  let m = /^rgba?\\(([^)]+)\\)/.exec(c);
  if (m) {
    const p = m[1].split(/[\\s,/]+/).filter(Boolean).map(Number);
    return [p[0], p[1], p[2], p[3] ?? 1];
  }
  m = /^color\\(srgb ([\\d.e-]+) ([\\d.e-]+) ([\\d.e-]+)(?: \\/ ([\\d.]+))?\\)/.exec(c);
  if (m) return [+m[1] * 255, +m[2] * 255, +m[3] * 255, m[4] === undefined ? 1 : +m[4]];
  return null;
};
const over = (f, b) => {
  const a = f[3];
  return [f[0] * a + b[0] * (1 - a), f[1] * a + b[1] * (1 - a), f[2] * a + b[2] * (1 - a), 1];
};
const lum = (c) => {
  const ch = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * ch(c[0]) + 0.7152 * ch(c[1]) + 0.0722 * ch(c[2]);
};
const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
const COLOR_RE = /(rgba?\\([^)]+\\)|color\\(srgb[^)]+\\))(\\s+[\\d.]+%)?/g;
const colorsIn = (img) => (img.match(COLOR_RE) || []).map((m) => parse(m.replace(/\\s+[\\d.]+%$/, ''))).filter(Boolean);
/** Colours of a top-to-bottom linear gradient on n over the vertical span of el (the stops in it and both ends). */
const gradientAt = (img, n, el) => {
  const stops = [...img.matchAll(COLOR_RE)].map((m, i, all) => ({ c: parse(m[1]), t: m[2] ? parseFloat(m[2]) / 100 : i / Math.max(1, all.length - 1) }));
  if (stops.length === 0) return [];
  const vertical = !/linear-gradient\\((\\d+deg|to )/.test(img) || /linear-gradient\\((180deg|to bottom)/.test(img);
  if (!vertical || !/^linear-gradient/.test(img)) return stops.map((s) => s.c);
  const r = n.getBoundingClientRect();
  const e = el.getBoundingClientRect();
  const t0 = Math.max(0, Math.min(1, (e.top - r.top) / (r.height || 1)));
  const t1 = Math.max(0, Math.min(1, (e.bottom - r.top) / (r.height || 1)));
  const at = (t) => {
    let k = 0;
    while (k < stops.length - 1 && stops[k + 1].t < t) k++;
    const a = stops[k], b = stops[Math.min(k + 1, stops.length - 1)];
    const f = b.t > a.t ? Math.max(0, Math.min(1, (t - a.t) / (b.t - a.t))) : 0;
    return a.c.map((v, i) => v + (b.c[i] - v) * f);
  };
  return [at(t0), at(t1), ...stops.filter((s) => s.t > t0 && s.t < t1).map((s) => s.c)];
};
/** Every colour the backdrop of el can take (gradients give several), composed from the root down to el (inclusive). */
const backdrops = (el) => {
  const chain = [];
  for (let n = el; n && n.nodeType === 1; n = n.parentElement) chain.push(n);
  let cands = [[255, 255, 255, 1]];
  for (const n of chain.reverse()) {
    const cs = getComputedStyle(n);
    const stops = cs.backgroundImage && cs.backgroundImage !== 'none' ? gradientAt(cs.backgroundImage, n, el) : [];
    if (stops.length) cands = stops.flatMap((s) => cands.map((c) => over(s, c)));
    const bg = parse(cs.backgroundColor);
    if (bg && bg[3] > 0) cands = cands.map((c) => over(bg, c));
    if (cands.length > 64) cands = cands.slice(0, 64);
  }
  return cands;
};
const opacityOf = (el) => { let o = 1; for (let n = el; n && n.nodeType === 1; n = n.parentElement) o *= +getComputedStyle(n).opacity; return o; };
const describe = (el) => {
  const id = el.getAttribute('data-testid');
  const cls = typeof el.className === 'string' ? el.className.trim().split(/\\s+/).slice(0, 2).join('.') : '';
  return el.tagName.toLowerCase() + (id ? '[data-testid=' + id + ']' : '') + (cls ? '.' + cls : '') + (el.getAttribute('role') ? '[role=' + el.getAttribute('role') + ']' : '');
};
const shown = (el) => {
  if (el.closest('.visually-hidden, [hidden]')) return false;
  const r = el.getBoundingClientRect();
  if (r.width < 2 || r.height < 2) return false;
  const cs = getComputedStyle(el);
  return cs.visibility !== 'hidden' && cs.display !== 'none' && opacityOf(el) > 0.01;
};
`;

/** Text (incl. input values and placeholders) under 4.5:1 (3:1 large) and input/select borders under 3:1. */
export async function contrastIssues(page: Page): Promise<ContrastIssue[]> {
  return page.evaluate(`(() => {
    ${LIB}
    const out = [];
    const inactive = (el) => !!el.closest(':disabled, [aria-disabled="true"], .disabled');
    const check = (el, fg, text, sizeEl) => {
      const cs = getComputedStyle(sizeEl);
      const px = parseFloat(cs.fontSize);
      const bold = +cs.fontWeight >= 700;
      const need = px >= 24 || (bold && px >= 18.66) ? 3 : 4.5;
      const o = opacityOf(el);
      const f = [fg[0], fg[1], fg[2], fg[3] * o];
      const min = Math.min(...backdrops(el).map((b) => ratio(over(f, b), b)));
      if (min < need - 0.005) out.push({ what: describe(el), text: text.slice(0, 40), ratio: Math.round(min * 100) / 100, need });
    };
    for (const el of document.querySelectorAll('body *')) {
      if (el.closest('canvas, script, style') || !shown(el) || inactive(el)) continue;
      const own = [...el.childNodes].filter((n) => n.nodeType === 3 && n.textContent.trim()).map((n) => n.textContent.trim()).join(' ');
      if (own) {
        const cs = getComputedStyle(el);
        const fg = parse(el instanceof SVGElement ? cs.fill : cs.color);
        if (fg) check(el, fg, own, el);
      }
      if (el.matches('input:not([type]), input[type=text], input[type=number], input[type=search], textarea, select')) {
        const cs = getComputedStyle(el);
        if (el.value) check(el, parse(cs.color), el.value, el);
        else if (el.placeholder) check(el, parse(getComputedStyle(el, '::placeholder').color), 'placeholder: ' + el.placeholder, el);
        // 1.4.11: the boundary that identifies the field, against what is around it
        const border = parse(cs.borderTopColor);
        // a field drawn without any boundary (the inline project name) is outside 1.4.11: nothing to measure
        if (!border || border[3] === 0 || parseFloat(cs.borderTopWidth) === 0) continue;
        const around = backdrops(el.parentElement);
        const inner = backdrops(el);
        const fillDiffers = Math.min(...inner.map((c, i) => ratio(c, around[i % around.length]))) >= 3;
        const b = Math.min(...around.map((c) => ratio(over(border, c), c)));
        if (!fillDiffers && b < 3 - 0.005) out.push({ what: describe(el) + ' border', text: '', ratio: Math.round(b * 100) / 100, need: 3 });
      }
    }
    return out;
  })()`) as Promise<ContrastIssue[]>;
}

/** The focus indicator of document.activeElement: drawn at all, and its contrast against the colour behind it. */
export async function focusInfo(page: Page): Promise<FocusInfo | null> {
  return page.evaluate(`(() => {
    ${LIB}
    const el = document.activeElement;
    if (!el || el === document.body) return null;
    const cs = getComputedStyle(el);
    const ow = parseFloat(cs.outlineWidth) || 0;
    const outline = cs.outlineStyle !== 'none' && ow >= 1 ? parse(cs.outlineColor) : null;
    let color = outline && outline[3] > 0 ? outline : null;
    if (!color && cs.boxShadow && cs.boxShadow !== 'none') color = colorsIn(cs.boxShadow)[0] || null;
    if (!color) return { what: describe(el), visible: false, ratio: 0 };
    // an outline inside the box (negative offset) sits on the element, otherwise on what is around it
    const inside = outline && parseFloat(cs.outlineOffset) < 0;
    const behind = backdrops(inside ? el : el.parentElement || el);
    const r = Math.min(...behind.map((b) => ratio(over(color, b), b)));
    return { what: describe(el), visible: true, ratio: Math.round(r * 100) / 100 };
  })()`) as Promise<FocusInfo | null>;
}

/** Presses Tab until the element with this test id has focus (proves it is in the tab order). */
export async function tabTo(page: Page, testId: string, { back = false, max = 400 } = {}): Promise<void> {
  for (let i = 0; i < max; i++) {
    if (await page.evaluate((id) => document.activeElement?.getAttribute('data-testid') === id, testId)) return;
    await page.keyboard.press(back ? 'Shift+Tab' : 'Tab');
  }
  throw new Error(`${testId} not reached with ${back ? 'Shift+Tab' : 'Tab'}`);
}

export const focusedTestId = (page: Page) => page.evaluate(() => document.activeElement?.getAttribute('data-testid') ?? null);
export const focusedRole = (page: Page) => page.evaluate(() => document.activeElement?.getAttribute('role') ?? document.activeElement?.tagName.toLowerCase() ?? null);
