// View cube (clickable faces snap to the standard views) and orientation triad. Both are DOM overlays driven by the
// camera: the viewport calls back after every render and they rewrite their transforms directly (no React render).
import { useEffect, useRef } from 'react';
import type { StandardView, Viewport } from '../viewport/Viewport';

const SIZE = 64;

/** World-space face frames (Z up): outward normal n, screen-right u and screen-down v as seen from outside. */
const FACES: { view: StandardView; label: string; n: number[]; u: number[]; v: number[] }[] = [
  { view: 'front', label: 'Front', n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, -1] },
  { view: 'back', label: 'Back', n: [0, 1, 0], u: [-1, 0, 0], v: [0, 0, -1] },
  { view: 'right', label: 'Right', n: [1, 0, 0], u: [0, 1, 0], v: [0, 0, -1] },
  { view: 'left', label: 'Left', n: [-1, 0, 0], u: [0, -1, 0], v: [0, 0, -1] },
  { view: 'top', label: 'Top', n: [0, 0, 1], u: [1, 0, 0], v: [0, -1, 0] },
  { view: 'bottom', label: 'Bottom', n: [0, 0, -1], u: [1, 0, 0], v: [0, 1, 0] },
];

/** CSS matrix3d (column-major) placing a SIZE x SIZE face, centred on the origin, on the cube surface. */
function faceMatrix(f: (typeof FACES)[number]): string {
  const h = SIZE / 2;
  const m = [...f.u, 0, ...f.v, 0, ...f.n, 0, f.n[0]! * h, f.n[1]! * h, f.n[2]! * h, 1];
  return `matrix3d(${m.join(',')})`;
}

/** Rotation part of the camera's view matrix with y flipped (camera y up -> CSS y down), as CSS matrix3d. */
function viewMatrix(e: ArrayLike<number>): string {
  const m = [e[0]!, -e[1]!, e[2]!, 0, e[4]!, -e[5]!, e[6]!, 0, e[8]!, -e[9]!, e[10]!, 0, 0, 0, 0, 1];
  return `matrix3d(${m.map((x) => x.toFixed(6)).join(',')})`;
}

const AXES = [
  { k: 'x', color: 'var(--axis-x)', v: [1, 0, 0] },
  { k: 'y', color: 'var(--axis-y)', v: [0, 1, 0] },
  { k: 'z', color: 'var(--axis-z)', v: [0, 0, 1] },
] as const;

export function ViewCube({ vp }: { vp: Viewport }) {
  const cube = useRef<HTMLDivElement>(null);
  const triad = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const update = () => {
      const e = vp.camera.matrixWorldInverse.elements;
      if (cube.current) cube.current.style.transform = viewMatrix(e);
      const svg = triad.current;
      if (!svg) return;
      // depth-sort so the axis pointing at the viewer is drawn last
      const items = AXES.map((a) => {
        const [x, y, z] = a.v;
        const sx = e[0]! * x + e[4]! * y + e[8]! * z;
        const sy = e[1]! * x + e[5]! * y + e[9]! * z;
        const sz = e[2]! * x + e[6]! * y + e[10]! * z;
        return { k: a.k, sx, sy, sz };
      }).sort((p, q) => p.sz - q.sz);
      items.forEach((it) => {
        const g = svg.querySelector<SVGGElement>(`[data-axis="${it.k}"]`);
        if (!g) return;
        const x2 = 32 + it.sx * 22;
        const y2 = 32 - it.sy * 22;
        const line = g.querySelector('line')!;
        line.setAttribute('x2', x2.toFixed(2));
        line.setAttribute('y2', y2.toFixed(2));
        const t = g.querySelector('text')!;
        t.setAttribute('x', (32 + it.sx * 29).toFixed(2));
        t.setAttribute('y', (32 - it.sy * 29 + 3.5).toFixed(2));
        line.style.opacity = it.sz < -0.6 ? '0.55' : '1'; // the letter stays at full contrast
        svg.appendChild(g); // re-append in depth order
      });
    };
    update();
    return vp.addCameraListener(update);
  }, [vp]);

  return (
    <>
      <div className="viewcube" data-testid="view-cube" role="group" aria-label="View cube">
        <div className="viewcube-stage">
          <div className="viewcube-cube" ref={cube}>
            {FACES.map((f) => (
              <button
                key={f.view}
                type="button"
                className={`viewcube-face face-${f.view}`}
                style={{ transform: faceMatrix(f) }}
                onClick={() => vp.setView(f.view)}
                tabIndex={-1} // faces turned away are hidden but focusable: keyboard users use the Standard views menu
                data-testid={`viewcube-${f.view}`}
                title={`${f.label} view`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>
        <button
          type="button"
          className="viewcube-home"
          onClick={() => vp.setView('iso')}
          title="Isometric view"
          data-testid="viewcube-iso"
          aria-label="Isometric view"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round">
            <path d="M12 4 5 8v8l7 4 7-4V8z" />
            <path d="m5 8 7 4 7-4M12 12v8" />
          </svg>
        </button>
      </div>
      <svg className="triad" ref={triad} width="64" height="64" viewBox="0 0 64 64" data-testid="triad" role="img" aria-label="Orientation triad (X, Y, Z axes)">
        <circle cx="32" cy="32" r="2.4" className="triad-origin" />
        {AXES.map((a) => (
          <g key={a.k} data-axis={a.k} style={{ color: a.color }}>
            <line x1="32" y1="32" x2="32" y2="32" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
            <text textAnchor="middle" fill="currentColor" fontSize="10" fontWeight="700">
              {a.k.toUpperCase()}
            </text>
          </g>
        ))}
      </svg>
    </>
  );
}
