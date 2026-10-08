// Unit primitive meshes as binary STL, uploaded as keep-out / keep-in reference models. The reference transform
// (translation * rotation * scale, applied to these coordinates) then places and sizes them:
//   box: unit cube centred at the origin; sphere: radius 1; cylinder: radius 1, height 1, axis local Y.
// These are the same conventions as the box/sphere/cylinder selections (CLAUDE.md), with `size` becoming the scale.
export type RefPrimitiveKind = 'box' | 'sphere' | 'cylinder';

type V3 = [number, number, number];
type Tri = [V3, V3, V3];

function box(): Tri[] {
  const v = (i: number): V3 => [i & 1 ? 0.5 : -0.5, i & 2 ? 0.5 : -0.5, i & 4 ? 0.5 : -0.5];
  const quads = [
    [0, 2, 3, 1], // z-
    [4, 5, 7, 6], // z+
    [0, 1, 5, 4], // y-
    [2, 6, 7, 3], // y+
    [0, 4, 6, 2], // x-
    [1, 3, 7, 5], // x+
  ];
  return quads.flatMap(([a, b, c, d]) => [
    [v(a!), v(b!), v(c!)] as Tri,
    [v(a!), v(c!), v(d!)] as Tri,
  ]);
}

function cylinder(n = 48): Tri[] {
  const ring = (i: number, y: number): V3 => [Math.cos((i / n) * Math.PI * 2), y, Math.sin((i / n) * Math.PI * 2)];
  const top: V3 = [0, 0.5, 0];
  const bottom: V3 = [0, -0.5, 0];
  const out: Tri[] = [];
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    out.push([ring(i, -0.5), ring(i, 0.5), ring(j, -0.5)], [ring(j, -0.5), ring(i, 0.5), ring(j, 0.5)]);
    out.push([top, ring(j, 0.5), ring(i, 0.5)], [bottom, ring(i, -0.5), ring(j, -0.5)]);
  }
  return out;
}

/** Icosphere (two subdivisions, 320 triangles): watertight, no poles. */
function sphere(levels = 2): Tri[] {
  const t = (1 + Math.sqrt(5)) / 2;
  let verts: V3[] = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ];
  const unit = (p: V3): V3 => {
    const l = Math.hypot(...p);
    return [p[0] / l, p[1] / l, p[2] / l];
  };
  verts = verts.map(unit);
  let faces: [number, number, number][] = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  for (let l = 0; l < levels; l++) {
    const mid = new Map<string, number>();
    const midpoint = (a: number, b: number): number => {
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      let idx = mid.get(key);
      if (idx === undefined) {
        const p = verts[a]!;
        const q = verts[b]!;
        verts.push(unit([(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2]));
        idx = verts.length - 1;
        mid.set(key, idx);
      }
      return idx;
    };
    faces = faces.flatMap(([a, b, c]): [number, number, number][] => {
      const ab = midpoint(a, b);
      const bc = midpoint(b, c);
      const ca = midpoint(c, a);
      return [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]];
    });
  }
  return faces.map(([a, b, c]) => [verts[a]!, verts[b]!, verts[c]!]);
}

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Binary STL (80-byte header, u32 count, 50 bytes per triangle) with outward-facing winding. */
export function primitiveStl(kind: RefPrimitiveKind): ArrayBuffer {
  let tris = kind === 'box' ? box() : kind === 'sphere' ? sphere() : cylinder();
  // all three generators are consistently wound; flip them all if that turned out to be inward
  const volume = tris.reduce((s, [a, b, c]) => s + (a[0] * cross(b, c)[0] + a[1] * cross(b, c)[1] + a[2] * cross(b, c)[2]) / 6, 0);
  if (volume < 0) tris = tris.map(([a, b, c]) => [a, c, b] as Tri);
  const buf = new ArrayBuffer(84 + tris.length * 50);
  const dv = new DataView(buf);
  dv.setUint32(80, tris.length, true);
  tris.forEach(([a, b, c], i) => {
    const o = 84 + i * 50;
    const n = cross(sub(b, a), sub(c, a));
    const len = Math.hypot(...n) || 1;
    [[n[0] / len, n[1] / len, n[2] / len] as V3, a, b, c].forEach((p, k) => {
      for (let j = 0; j < 3; j++) dv.setFloat32(o + k * 12 + j * 4, p[j]!, true);
    });
  });
  return buf;
}
