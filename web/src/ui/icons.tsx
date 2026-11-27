// Inline line icons (24-unit grid, stroked with currentColor). Drawn for top-op; simple geometric shapes only.
import type { ReactNode } from 'react';

const P: Record<string, ReactNode> = {
  import: (
    <>
      <path d="M12 3v11m-4-4 4 4 4-4" />
      <path d="M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4" />
    </>
  ),
  export: (
    <>
      <path d="M12 14V3m-4 4 4-4 4 4" />
      <path d="M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4" />
    </>
  ),
  example: (
    <>
      <path d="M4 20V9h7V4h9v16z" />
      <circle cx="15.5" cy="12" r="2" />
    </>
  ),
  new: (
    <>
      <path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z" />
      <path d="M14 3v5h5M12 11v6m-3-3h6" />
    </>
  ),
  open: <path d="M3 7a1 1 0 0 1 1-1h5l2 2h8a1 1 0 0 1 1 1v2M3 7v11a1 1 0 0 0 1 1h13l4-8H8l-4 8" />,
  save: (
    <>
      <path d="M5 4h11l3 3v12a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1z" />
      <path d="M8 4v5h7V4M8 20v-6h8v6" />
    </>
  ),
  part: (
    <>
      <path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" />
      <path d="M4 7.5 12 12l8-4.5M12 12v9" />
    </>
  ),
  project: (
    <>
      <rect x="4" y="4" width="16" height="16" rx="1" />
      <path d="M8 9h8M8 13h8M8 17h5" />
    </>
  ),
  gizmo: (
    <>
      <path d="M12 3v18M3 12h18" />
      <path d="m9 6 3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3" />
    </>
  ),
  reset: (
    <>
      <path d="M4 12a8 8 0 1 0 2.4-5.7" />
      <path d="M4 4v5h5" />
    </>
  ),
  fit: (
    <>
      <path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5" />
      <rect x="8" y="8" width="8" height="8" />
    </>
  ),
  iso: (
    <>
      <path d="M12 4 5 8v8l7 4 7-4V8z" />
      <path d="m5 8 7 4 7-4M12 12v8" />
    </>
  ),
  home: (
    <>
      <path d="m4 11 8-7 8 7" />
      <path d="M6 10v10h12V10" />
    </>
  ),
  orbit: (
    <>
      <ellipse cx="12" cy="12" rx="9" ry="4" />
      <path d="M17 6.2A9 4 0 0 1 21 12" />
      <circle cx="12" cy="12" r="1.5" />
    </>
  ),
  pick: <path d="m5 3 14 7-6 2-2 6z" />,
  paint: (
    <>
      <path d="M14 4 20 10 11 19H5v-6z" />
      <path d="m12 6 6 6" />
    </>
  ),
  query: (
    <>
      <circle cx="10.5" cy="10.5" r="6" />
      <path d="m15 15 6 6M8 10.5h5M10.5 8v5" />
    </>
  ),
  box: (
    <>
      <path d="M4 8h12v12H4z" />
      <path d="m4 8 4-4h12v12l-4 4M16 8l4-4" />
    </>
  ),
  sphere: (
    <>
      <circle cx="12" cy="12" r="8" />
      <ellipse cx="12" cy="12" rx="8" ry="3" />
    </>
  ),
  cylinder: (
    <>
      <ellipse cx="12" cy="6" rx="7" ry="2.5" />
      <path d="M5 6v12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5V6" />
    </>
  ),
  force: (
    <>
      <path d="M12 3v11m-4-4 4 4 4-4" />
      <path d="M3 18h18M5 21l2-3M10 21l2-3M15 21l2-3" />
    </>
  ),
  support: (
    <>
      <path d="m12 5-6 9h12z" />
      <path d="M4 17h16M6 20l2-3M11 20l2-3M16 20l2-3" />
    </>
  ),
  preview: (
    <>
      <circle cx="6" cy="6" r="1.3" />
      <circle cx="12" cy="6" r="1.3" />
      <circle cx="18" cy="6" r="1.3" />
      <circle cx="6" cy="12" r="1.3" />
      <circle cx="12" cy="12" r="1.3" />
      <circle cx="18" cy="12" r="1.3" />
      <circle cx="6" cy="18" r="1.3" />
      <circle cx="12" cy="18" r="1.3" />
      <circle cx="18" cy="18" r="1.3" />
    </>
  ),
  clear: <path d="m6 6 12 12M18 6 6 18" />,
  domain: (
    <>
      <rect x="4" y="4" width="16" height="16" />
      <path d="M4 9.3h16M4 14.7h16M9.3 4v16M14.7 4v16" />
    </>
  ),
  voxelize: (
    <>
      <path d="M4 14h6v6H4zM10 14h6v6h-6zM10 8h6v6h-6z" />
      <path d="m17 4 1 2 2 1-2 1-1 2-1-2-2-1 2-1z" />
    </>
  ),
  ref: (
    <>
      <path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" strokeDasharray="2.5 2" />
      <path d="M12 9v6M9 12h6" />
    </>
  ),
  keepout: (
    <>
      <path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" />
      <path d="m6 18 12-12" />
    </>
  ),
  keepin: (
    <>
      <path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" />
      <path d="m8.5 12 2.5 2.5 4.5-5" />
    </>
  ),
  study: (
    <>
      <path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" />
      <circle cx="16" cy="6" r="2" />
      <circle cx="10" cy="12" r="2" />
      <circle cx="18" cy="18" r="2" />
    </>
  ),
  material: (
    <>
      <path d="m12 3 9 5-9 5-9-5z" />
      <path d="m3 12 9 5 9-5M3 16l9 5 9-5" />
    </>
  ),
  symmetry: (
    <>
      <path d="M12 3v18" strokeDasharray="2 2" />
      <path d="M9 7 4 17h5zM15 7l5 10h-5z" />
    </>
  ),
  overhang: (
    <>
      <path d="M4 20h16M12 17V5m-3 3 3-3 3 3" />
      <path d="m6 17 3-5M18 17l-3-5" />
    </>
  ),
  run: <path d="M7 4.5v15l12-7.5z" />,
  stop: <rect x="6" y="6" width="12" height="12" rx="1" />,
  results: (
    <>
      <path d="M4 20h16" />
      <path d="M6 16h3v4H6zM10.5 10h3v10h-3zM15 6h3v14h-3z" />
    </>
  ),
  density: (
    <>
      <path d="M4 4h5v5H4zM9.5 9.5h5v5h-5zM15 15h5v5h-5zM15 4h5v5h-5z" />
      <path d="M4 15h5v5H4z" strokeDasharray="1.5 1.5" />
    </>
  ),
  stress: (
    <>
      <path d="M5 20c0-9 14-7 14-16" />
      <path d="M5 4h4M5 8h2M17 20h2M15 16h4" />
    </>
  ),
  mesh: (
    <>
      <path d="M4 18 12 4l8 14z" />
      <path d="m8 11 8 0M12 18l-4-7M12 18l4-7" />
    </>
  ),
  hide: (
    <>
      <path d="M2 12s3.6-6 10-6 10 6 10 6-3.6 6-10 6S2 12 2 12z" />
      <path d="m4 20 16-16" />
    </>
  ),
  eye: (
    <>
      <path d="M2 12s3.6-6 10-6 10 6 10 6-3.6 6-10 6S2 12 2 12z" />
      <circle cx="12" cy="12" r="2.8" />
    </>
  ),
  trash: (
    <>
      <path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" />
      <path d="M10 11v6M14 11v6" />
    </>
  ),
  more: (
    <>
      <circle cx="6" cy="12" r="1.2" />
      <circle cx="12" cy="12" r="1.2" />
      <circle cx="18" cy="12" r="1.2" />
    </>
  ),
  chevron: <path d="m9 6 6 6-6 6" />,
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>
  ),
  moon: <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />,
  section: (
    <>
      <path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" />
      <path d="M2 14 22 9" strokeDasharray="2.5 1.8" />
    </>
  ),
  shaded: (
    <>
      <path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" fill="currentColor" fillOpacity="0.25" />
      <path d="M4 7.5 12 12l8-4.5" />
    </>
  ),
  edges: (
    <>
      <path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" fill="currentColor" fillOpacity="0.25" strokeWidth="2.2" />
      <path d="M4 7.5 12 12l8-4.5M12 12v9" strokeWidth="2.2" />
    </>
  ),
  wireframe: (
    <>
      <path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" />
      <path d="M4 7.5 12 12l8-4.5M12 12v9M4 16.5 12 12l8 4.5M12 3v9" strokeDasharray="2 1.6" />
    </>
  ),
  ghost: (
    <>
      <path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" strokeDasharray="3 2" />
      <path d="M9 10h6v6H9z" fill="currentColor" fillOpacity="0.35" />
    </>
  ),
  threshold: (
    <>
      <path d="M3 18h18" />
      <path d="M3 18c4 0 5-12 9-12s5 12 9 12" />
      <path d="M12 3v18" strokeDasharray="2 2" />
    </>
  ),
  list: <path d="M8 6h12M8 12h12M8 18h12M4 6h0M4 12h0M4 18h0" />,
  rename: (
    <>
      <path d="m15 5 4 4L9 19H5v-4z" />
      <path d="M13 7l4 4" />
    </>
  ),
  target: (
    <>
      <circle cx="12" cy="12" r="7" />
      <path d="M12 2v5M12 17v5M2 12h5M17 12h5" />
    </>
  ),
  check: <path d="m5 12 4.5 4.5L19 7" />,
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v6M12 7.5v.5" />
    </>
  ),
  warn: (
    <>
      <path d="M12 3 2 20h20z" />
      <path d="M12 10v5M12 17.5v.5" />
    </>
  ),
  close: <path d="m7 7 10 10M17 7 7 17" />,
};

export type IconName = keyof typeof P;

export function Icon({ name, size = 16, className }: { name: IconName; size?: number; className?: string }) {
  return (
    <svg
      className={`icon ${className ?? ''}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {P[name]}
    </svg>
  );
}

/** The app mark: a voxel lattice thinned to a strut, drawn in the accent colour. */
export function AppMark({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" className="app-mark">
      <rect x="2" y="2" width="20" height="20" rx="4" fill="var(--accent)" />
      <path d="M6 18 12 6l6 12" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
      <path d="M8.7 13h6.6" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}
