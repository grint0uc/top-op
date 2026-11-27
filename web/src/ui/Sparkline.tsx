import { useEffect, useRef } from 'react';
import type { IterationRecord } from '../api/client';
import { useUi } from './uiStore';

interface Props {
  history: readonly IterationRecord[];
  height?: number;
}

export const SERIES_COLORS = { compliance: '#2f7bf6', volume: '#e8930c', stress: '#e5484d' };

function series(g: CanvasRenderingContext2D, vals: readonly number[], color: string, w: number, h: number, log: boolean): [number, number] {
  const v = log ? vals.map((x) => Math.log10(Math.max(x, 1e-12))) : [...vals];
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  const span = hi - lo || 1;
  // A series that barely moves (volume under OC sits exactly on volfrac) would autoscale rounding noise into a
  // mountain range: judge on the raw values and draw it flat in the middle of its band instead.
  const rawLo = Math.min(...vals);
  const rawHi = Math.max(...vals);
  const flat = rawHi - rawLo <= 0.01 * Math.max(...vals.map(Math.abs));
  g.strokeStyle = color;
  g.lineWidth = 1.5;
  g.beginPath();
  v.forEach((y, i) => {
    const px = vals.length === 1 ? w / 2 : (i / (vals.length - 1)) * (w - 4) + 2;
    const py = flat ? h / 2 : h - 3 - ((y - lo) / span) * (h - 8);
    if (i === 0) g.moveTo(px, py);
    else g.lineTo(px, py);
  });
  g.stroke();
  return [rawLo, rawHi];
}

/**
 * Compliance (log scale), volume fraction and, when the run reports it, max von Mises stress, each normalised to its
 * own range (flat if it moves < 1 %). Plain canvas, no chart lib.
 */
export function Sparkline({ history, height = 90 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const hasStress = history.some((r) => typeof r.stress_max === 'number');
  const dark = useUi((s) => s.dark);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    let lastW = -1;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth;
      if (w === lastW || w === 0) return;
      lastW = w;
      const h = height;
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
      const g = c.getContext('2d');
      if (!g) return;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const css = getComputedStyle(c);
      const COLORS = {
        ...SERIES_COLORS,
        grid: css.getPropertyValue('--border').trim() || '#c9d1dc',
        text: css.getPropertyValue('--fg-dim').trim() || '#6b7686',
      };
      // faint quarter lines behind the series
      g.strokeStyle = COLORS.grid;
      g.lineWidth = 1;
      g.setLineDash([2, 3]);
      for (const f of [0.25, 0.5, 0.75]) {
        g.beginPath();
        g.moveTo(0, Math.round(h * f) + 0.5);
        g.lineTo(w, Math.round(h * f) + 0.5);
        g.stroke();
      }
      g.setLineDash([]);
      g.strokeRect(0.5, 0.5, w - 1, h - 1);
      if (history.length === 0) {
        g.fillStyle = COLORS.text;
        g.font = '11px Inter, system-ui, sans-serif';
        g.fillText('convergence: compliance / volume', 8, h / 2 + 4);
        return;
      }
      series(
        g,
        history.map((r) => r.compliance),
        COLORS.compliance,
        w,
        h,
        true,
      );
      series(
        g,
        history.map((r) => r.volume),
        COLORS.volume,
        w,
        h,
        false,
      );
      if (hasStress) {
        // records without a value (stress not evaluated yet) repeat the neighbouring one instead of dropping to 0
        const first = history.find((r) => typeof r.stress_max === 'number')!.stress_max as number;
        let prev = first;
        series(
          g,
          history.map((r) => (prev = typeof r.stress_max === 'number' ? r.stress_max : prev)),
          COLORS.stress,
          w,
          h,
          false,
        );
      }
    };
    draw();
    // a canvas on a page that was hidden has no width yet: draw once it is laid out (and on resize)
    const ro = new ResizeObserver(() => draw());
    ro.observe(c);
    return () => ro.disconnect();
  }, [history, height, hasStress, dark]);

  return (
    <canvas
      ref={ref}
      className="sparkline"
      style={{ height }}
      data-testid="sparkline"
      data-points={history.length}
      data-series={history.length === 0 ? 0 : hasStress ? 3 : 2}
    />
  );
}
