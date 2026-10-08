// Run-parameter overlays in the viewport: translucent quads for the symmetry planes (spanning the domain box) and,
// for an overhang build direction, an arrow plus the outline of the base plate the part grows from.
import * as THREE from 'three';
import type { Viewport } from './Viewport';

export interface Box {
  min: readonly number[];
  max: readonly number[];
}

export interface SymmetryPlane {
  axis: 'x' | 'y' | 'z';
  position: number;
}

export type BuildDir = '+x' | '-x' | '+y' | '-y' | '+z' | '-z';

/** World-space extent of a drawn quad (diagnostic: lets e2e check that it spans the domain). */
interface Extent {
  min: number[];
  max: number[];
}

export interface OverlayInfo {
  symmetry: (SymmetryPlane & Extent)[];
  overhang: ({ dir: BuildDir; axis: 'x' | 'y' | 'z'; plateAt: number; length: number } & Extent) | null;
}

const extentOf = (pts: THREE.Vector3[]): Extent => ({
  min: [0, 1, 2].map((k) => Math.min(...pts.map((p) => p.getComponent(k)))),
  max: [0, 1, 2].map((k) => Math.max(...pts.map((p) => p.getComponent(k)))),
});

const PLANE_HEX = { x: 0xff5a5f, y: 0x3ddc97, z: 0x4f9cf9 } as const;
const AM_HEX = 0xffd23f;
const AXES = ['x', 'y', 'z'] as const;

/** Corners of the domain box's cross-section perpendicular to `axis`, at coordinate `at`. */
function rect(axis: number, at: number, box: Box): THREE.Vector3[] {
  const [u, v] = [0, 1, 2].filter((k) => k !== axis) as [number, number];
  return [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ].map(([i, j]) => {
    const p = [0, 0, 0];
    p[axis] = at;
    p[u] = i ? box.max[u]! : box.min[u]!;
    p[v] = j ? box.max[v]! : box.min[v]!;
    return new THREE.Vector3(p[0], p[1], p[2]);
  });
}

function quad(corners: THREE.Vector3[], color: number, opacity: number, name: string): THREE.Group {
  const g = new THREE.Group();
  g.name = name;
  const geo = new THREE.BufferGeometry().setFromPoints(corners);
  geo.setIndex([0, 1, 2, 0, 2, 3]);
  g.add(
    new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }),
    ),
    new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(corners), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 })),
  );
  return g;
}

export class Overlays {
  readonly group = new THREE.Group();
  private info: OverlayInfo = { symmetry: [], overhang: null };

  constructor(private readonly vp: Viewport) {
    this.group.name = 'overlays';
  }

  /** `box` is the domain (voxel grid, or the transformed design bbox before the first voxelization). */
  set(box: Box | null, planes: readonly SymmetryPlane[], build: BuildDir | null): void {
    this.clear();
    this.info = { symmetry: [], overhang: null };
    if (box) {
      planes.forEach((p, i) => {
        const axis = AXES.indexOf(p.axis);
        const corners = rect(axis, p.position, box);
        this.group.add(quad(corners, PLANE_HEX[p.axis], 0.16, `symmetry:${i}`));
        this.info.symmetry.push({ ...p, ...extentOf(corners) });
      });
      if (build) this.addBuild(box, build);
    }
    this.vp.requestRender();
  }

  /** Diagnostics for e2e. */
  describe(): OverlayInfo {
    return structuredClone(this.info);
  }

  private addBuild(box: Box, dir: BuildDir): void {
    const axis = AXES.indexOf(dir[1] as 'x' | 'y' | 'z');
    const sign = dir[0] === '+' ? 1 : -1;
    // the part grows from the base plate into the domain: the min face for +, the max face for -
    const at = sign > 0 ? box.min[axis]! : box.max[axis]!;
    const corners = rect(axis, at, box);
    const plate = quad(corners, AM_HEX, 0.08, 'overhang-plate');
    const size = [0, 1, 2].map((k) => box.max[k]! - box.min[k]!);
    const length = Math.max(...size) * 0.18;
    const origin = new THREE.Vector3(...[0, 1, 2].map((k) => (k === axis ? at : (box.min[k]! + box.max[k]!) / 2)) as [number, number, number]);
    const d = new THREE.Vector3();
    d.setComponent(axis, sign);
    const arrow = new THREE.ArrowHelper(d, origin, length, AM_HEX, length * 0.35, length * 0.2);
    arrow.name = 'overhang-arrow';
    for (const part of [arrow.line, arrow.cone]) {
      (part.material as THREE.Material).depthTest = false;
      part.renderOrder = 6;
    }
    this.group.add(plate, arrow);
    this.info.overhang = { dir, axis: AXES[axis]!, plateAt: at, length, ...extentOf(corners) };
  }

  private clear(): void {
    for (const child of [...this.group.children]) {
      this.group.remove(child);
      child.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose();
        (m.material as THREE.Material | undefined)?.dispose();
      });
    }
  }

  dispose(): void {
    this.clear();
  }
}
