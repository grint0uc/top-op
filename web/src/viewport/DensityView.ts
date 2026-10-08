// Live density display: an InstancedMesh of unit cubes (scaled by h) for cells with rho >= threshold,
// coloured by rho with a viridis ramp, or by von Mises stress (inferno ramp) once a stress field is set. The
// InstancedMesh is reused between frames; only the instance count and matrices are rewritten. Above POINT_FALLBACK
// visible cells it switches to Points.
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import type { DensityFrame, StressField } from '../api/client';
import type { Viewport } from './Viewport';

export const POINT_FALLBACK = 300_000;

// 8-stop viridis ramp (matplotlib viridis sampled at i/7)
const VIRIDIS = [0x440154, 0x46327e, 0x365c8d, 0x277f8e, 0x1fa187, 0x4ac16d, 0x9fda3a, 0xfde725];
// sequential ramp for stress (matplotlib inferno sampled at i/9): dark = low, bright = high; the legend bar uses the same stops
export const INFERNO = [0x000004, 0x1b0c41, 0x4a0c6b, 0x781c6d, 0xa52c60, 0xcf4446, 0xed6925, 0xfb9b06, 0xf7d13d, 0xfcffa4];

/** 256-entry rgb lookup in three's linear working space, interpolated between the stops. */
function buildLut(ramp: readonly number[]): Float32Array {
  const stops = ramp.map((h) => new THREE.Color(h));
  const lut = new Float32Array(256 * 3);
  const c = new THREE.Color();
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(t));
    c.copy(stops[k]!).lerp(stops[k + 1]!, t - k);
    lut.set([c.r, c.g, c.b], i * 3);
  }
  return lut;
}

export interface DensityGrid {
  origin: readonly number[];
  h: number;
}

export class DensityView {
  readonly group = new THREE.Group();
  /** instanced cubes / points; hidden while the final result mesh is shown */
  private readonly cells = new THREE.Group();
  private resultMesh: THREE.Mesh | null = null;
  mode: 'none' | 'instanced' | 'points' = 'none';
  count = 0;
  it = 0;
  /** world-space extent of the visible cells (diagnostic: lets e2e check the grid is positioned on the design) */
  bounds: { min: number[]; max: number[] } | null = null;

  private inst: THREE.InstancedMesh | null = null;
  private instCap = 0;
  private pts: THREE.Points | null = null;
  private ptsCap = 0;
  private frame: DensityFrame | null = null;
  private grid: DensityGrid | null = null;
  private threshold = 0.5;
  private readonly lut = buildLut(VIRIDIS);
  private readonly stressLut = buildLut(INFERNO);
  private stress: StressField | null = null;
  /** what the cells are coloured by right now (stress only when the field matches the frame's grid) */
  colorMode: 'density' | 'stress' = 'density';
  private readonly cube = new THREE.BoxGeometry(1, 1, 1);
  private readonly cubeMat = new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0 });
  private readonly pointMat = new THREE.PointsMaterial({ size: 2, sizeAttenuation: false, vertexColors: true });

  constructor(private readonly vp: Viewport) {
    this.group.name = 'density';
    this.group.add(this.cells);
  }

  /** Final result: the server's STL (world coordinates) drawn solid; replaces any previous one. */
  showResultMesh(stl: ArrayBuffer): number {
    this.clearResultMesh();
    const geometry = new STLLoader().parse(stl);
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({ color: 0x3ddc97, roughness: 0.6, metalness: 0.1, side: THREE.DoubleSide });
    this.resultMesh = new THREE.Mesh(geometry, material);
    this.resultMesh.name = 'result';
    this.group.add(this.resultMesh);
    this.vp.requestRender();
    return (geometry.getAttribute('position')?.count ?? 0) / 3;
  }

  clearResultMesh(): void {
    if (!this.resultMesh) return;
    this.group.remove(this.resultMesh);
    this.resultMesh.geometry.dispose();
    (this.resultMesh.material as THREE.Material).dispose();
    this.resultMesh = null;
    this.vp.requestRender();
  }

  get hasResult(): boolean {
    return this.resultMesh !== null;
  }

  setFrame(frame: DensityFrame | null, grid: DensityGrid | null, threshold: number): void {
    this.frame = frame;
    this.grid = grid;
    this.threshold = threshold;
    this.rebuild();
  }

  /** Colour the cells by this per-element field (null: back to density). Ignored while its grid differs from the frame's. */
  setStress(field: StressField | null): void {
    if (field === this.stress) return;
    this.stress = field;
    this.rebuild();
  }

  /** Sum of the visible instance colours (diagnostic: lets e2e see that recolouring happened). */
  colorSum(): number {
    const attr = this.mode === 'points' ? this.pts?.geometry.getAttribute('color') : this.inst?.instanceColor;
    if (!attr || this.count === 0) return 0;
    const arr = attr.array as Float32Array;
    let sum = 0;
    for (let i = 0; i < this.count * 3; i++) sum += arr[i]!;
    return Math.round(sum * 1e3) / 1e3;
  }

  setThreshold(t: number): void {
    if (t === this.threshold) return;
    this.threshold = t;
    this.rebuild();
  }

  /** Shows or hides the density cells (the result mesh is unaffected). */
  setVisible(v: boolean): void {
    this.cells.visible = v;
    this.vp.requestRender();
  }

  clear(): void {
    this.setFrame(null, null, this.threshold);
  }

  private rebuild(): void {
    const f = this.frame;
    const g = this.grid;
    if (!f || !g) {
      this.count = 0;
      this.mode = 'none';
      this.colorMode = 'density';
      this.it = 0;
      this.bounds = null;
      if (this.inst) this.inst.count = 0;
      if (this.pts) this.pts.geometry.setDrawRange(0, 0);
      this.vp.requestRender();
      return;
    }
    const [nx, ny, nz] = f.shape;
    const rho = f.rho;
    const cut = Math.max(1, Math.round(this.threshold * 255));
    let n = 0;
    for (let i = 0; i < rho.length; i++) if (rho[i]! >= cut) n++;
    const sf = this.stress && this.stress.shape[0] === nx && this.stress.shape[1] === ny && this.stress.shape[2] === nz ? this.stress : null;
    this.colorMode = sf ? 'stress' : 'density';
    const smax = sf && sf.max > 0 ? sf.max : 1;
    // lookup row of one cell: its density byte, or its stress scaled to [0, max]
    const lutRow = (idx: number, v: number): number => (sf ? Math.round(Math.min(1, Math.max(0, sf.data[idx]! / smax)) * 255) : v);
    const lut = sf ? this.stressLut : this.lut;
    this.it = f.it;
    this.count = n;
    const asPoints = n > POINT_FALLBACK;
    this.mode = asPoints ? 'points' : 'instanced';
    if (this.inst) this.inst.visible = !asPoints;
    if (this.pts) this.pts.visible = asPoints;

    const h = g.h;
    const [ox, oy, oz] = [g.origin[0] ?? 0, g.origin[1] ?? 0, g.origin[2] ?? 0];
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    this.fill(rho, nx, ny, nz, cut, (_j, x, y, z) => {
      if (x < lo[0]!) lo[0] = x;
      if (x > hi[0]!) hi[0] = x;
      if (y < lo[1]!) lo[1] = y;
      if (y > hi[1]!) hi[1] = y;
      if (z < lo[2]!) lo[2] = z;
      if (z > hi[2]!) hi[2] = z;
    });
    this.bounds = n > 0 ? { min: [ox + h * lo[0]!, oy + h * lo[1]!, oz + h * lo[2]!], max: [ox + h * (hi[0]! + 1), oy + h * (hi[1]! + 1), oz + h * (hi[2]! + 1)] } : null;
    if (asPoints) {
      const pts = this.ensurePoints(n);
      const pos = pts.geometry.getAttribute('position') as THREE.BufferAttribute;
      const col = pts.geometry.getAttribute('color') as THREE.BufferAttribute;
      this.fill(rho, nx, ny, nz, cut, (j, x, y, z, v, idx) => {
        (pos.array as Float32Array).set([ox + h * (x + 0.5), oy + h * (y + 0.5), oz + h * (z + 0.5)], j * 3);
        const k = lutRow(idx, v);
        (col.array as Float32Array).set(lut.subarray(k * 3, k * 3 + 3), j * 3);
      });
      pos.needsUpdate = true;
      col.needsUpdate = true;
      pts.geometry.setDrawRange(0, n);
    } else {
      const inst = this.ensureInstanced(n);
      const mat = inst.instanceMatrix.array as Float32Array;
      const col = inst.instanceColor!.array as Float32Array;
      this.fill(rho, nx, ny, nz, cut, (j, x, y, z, v, idx) => {
        const o = j * 16;
        mat[o] = h;
        mat[o + 1] = 0;
        mat[o + 2] = 0;
        mat[o + 3] = 0;
        mat[o + 4] = 0;
        mat[o + 5] = h;
        mat[o + 6] = 0;
        mat[o + 7] = 0;
        mat[o + 8] = 0;
        mat[o + 9] = 0;
        mat[o + 10] = h;
        mat[o + 11] = 0;
        mat[o + 12] = ox + h * (x + 0.5);
        mat[o + 13] = oy + h * (y + 0.5);
        mat[o + 14] = oz + h * (z + 0.5);
        mat[o + 15] = 1;
        const k = lutRow(idx, v);
        col.set(lut.subarray(k * 3, k * 3 + 3), j * 3);
      });
      inst.count = n;
      inst.instanceMatrix.needsUpdate = true;
      inst.instanceColor!.needsUpdate = true;
    }
    this.vp.requestRender();
  }

  /** C-order walk ([ix][iy][iz], iz fastest) calling `put` for each visible cell. */
  private fill(
    rho: Uint8Array,
    nx: number,
    ny: number,
    nz: number,
    cut: number,
    put: (j: number, x: number, y: number, z: number, byte: number, idx: number) => void,
  ): void {
    let j = 0;
    let idx = 0;
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        for (let z = 0; z < nz; z++, idx++) {
          const v = rho[idx]!;
          if (v >= cut) put(j++, x, y, z, v, idx);
        }
      }
    }
  }

  private ensureInstanced(n: number): THREE.InstancedMesh {
    if (!this.inst || this.instCap < n) {
      if (this.inst) {
        this.cells.remove(this.inst);
        this.inst.dispose();
      }
      this.instCap = Math.max(1024, Math.ceil(n * 1.5));
      this.inst = new THREE.InstancedMesh(this.cube, this.cubeMat, this.instCap);
      this.inst.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(this.instCap * 3), 3);
      this.inst.frustumCulled = false;
      this.inst.count = 0;
      this.cells.add(this.inst);
    }
    return this.inst;
  }

  private ensurePoints(n: number): THREE.Points {
    if (!this.pts || this.ptsCap < n) {
      if (this.pts) {
        this.cells.remove(this.pts);
        this.pts.geometry.dispose();
      }
      this.ptsCap = Math.ceil(n * 1.25);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.ptsCap * 3), 3));
      geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.ptsCap * 3), 3));
      this.pts = new THREE.Points(geo, this.pointMat);
      this.pts.frustumCulled = false;
      this.cells.add(this.pts);
    }
    return this.pts;
  }

  dispose(): void {
    this.clearResultMesh();
    if (this.inst) this.inst.dispose();
    this.pts?.geometry.dispose();
    this.cube.dispose();
    this.cubeMat.dispose();
    this.pointMat.dispose();
  }
}
