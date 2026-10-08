import { useEffect, useRef } from 'react';
import type { IterationRecord } from '../api/client';

interface Props {
  history: readonly IterationRecord[];
  height?: number;
}

const COLORS = { compliance: '#4f9cf9', volume: '#ffa534', grid: '#2b3038', text: '#9aa1ab' };

function series(
  g: CanvasRenderingContext2D,
  vals: readonly number[],
  color: string,
  w: number,
  h: number,
  log: boolean,
): [number, number] {
  const v = log ? vals.map((x) => Math.log10(Math.max(x, 1e-12))) : [...vals];
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  const span = hi - lo || 1;
  g.strokeStyle = color;
  g.lineWidth = 1.5;
  g.beginPath();
  v.forEach((y, i) => {
    const px = vals.length === 1 ? w / 2 : (i / (vals.length - 1)) * (w - 4) + 2;
    const py = h - 3 - ((y - lo) / span) * (h - 8);
    if (i === 0) g.moveTo(px, py);
    else g.lineTo(px, py);
  });
  g.stroke();
  return [Math.min(...vals), Math.max(...vals)];
}

/** Compliance (log scale) and volume fraction, each normalised to its own range. Plain canvas, no chart lib. */
export function Sparkline({ history, height = 90 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = height;
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    const g = c.getContext('2d');
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    g.strokeStyle = COLORS.grid;
    g.lineWidth = 1;
    g.strokeRect(0.5, 0.5, w - 1, h - 1);
    if (history.length === 0) {
      g.fillStyle = COLORS.text;
      g.font = '11px system-ui, sans-serif';
      g.fillText('compliance / volume', 8, h / 2 + 4);
      return;
    }
    series(g, history.map((r) => r.compliance), COLORS.compliance, w, h, true);
    series(g, history.map((r) => r.volume), COLORS.volume, w, h, false);
  }, [history, height]);

  return <canvas ref={ref} className="sparkline" style={{ height }} data-testid="sparkline" data-points={history.length} />;
}
