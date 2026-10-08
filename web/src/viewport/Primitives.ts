// Box / sphere / cylinder selection helpers (wireframe + 20% fill) driven by the shared TransformControls.
// Wire convention (CLAUDE.md): box = unit cube scaled by `size`; sphere radius size[0]; cylinder axis = local Y,
// radius size[0], height size[1]. `transform` carries rotation + translation only; size is kept separately.
import * as THREE from 'three';
import type { Prim, PrimitiveKind } from '../state/store';
import type { Viewport } from './Viewport';

const ONE = new THREE.Vector3(1, 1, 1);

function bodyScale(kind: PrimitiveKind, size: readonly number[]): [number, number, number] {
  const a = size[0] ?? 1;
  const b = size[1] ?? 1;
  const c = size[2] ?? 1;
  if (kind === 'box') return [a, b, c];
  if (kind === 'sphere') return [a, a, a];
  return [a, b, a]; // cylinder: radius, height, radius
}

/** Clean outlines: box edges, three great circles for a sphere, cap circles + 4 ribs for a cylinder. */
function outline(kind: PrimitiveKind): THREE.BufferGeometry {
  if (kind === 'box') return new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
  const pts: number[] = [];
  const N = 48;
  const circle = (at: (a: number) => [number, number, number]) => {
    for (let i = 0; i < N; i++) pts.push(...at((i / N) * Math.PI * 2), ...at(((i + 1) / N) * Math.PI * 2));
  };
  if (kind === 'sphere') {
    circle((a) => [Math.cos(a), Math.sin(a), 0]);
    circle((a) => [0, Math.cos(a), Math.sin(a)]);
    circle((a) => [Math.cos(a), 0, Math.sin(a)]);
  } else {
    circle((a) => [Math.cos(a), 0.5, Math.sin(a)]);
    circle((a) => [Math.cos(a), -0.5, Math.sin(a)]);
    for (let k = 0; k < 4; k++) {
      const a = (k * Math.PI) / 2;
      pts.push(Math.cos(a), 0.5, Math.sin(a), Math.cos(a), -0.5, Math.sin(a));
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  return g;
}

function unitGeometry(kind: PrimitiveKind): THREE.BufferGeometry {
  if (kind === 'box') return new THREE.BoxGeometry(1, 1, 1);
  if (kind === 'sphere') return new THREE.SphereGeometry(1, 20, 14);
  return new THREE.CylinderGeometry(1, 1, 1, 28, 1);
}

/** group (pose) -> body (scaled by size) -> [fill, wire] */
export function makePrimitiveHelper(kind: PrimitiveKind, color: number, fill = 0.2): THREE.Group {
  const geo = unitGeometry(kind);
  const body = new THREE.Group();
  body.name = 'body';
  const fillMesh = new THREE.Mesh(
    geo,
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: fill, depthWrite: false, side: THREE.DoubleSide }),
  );
  const wire = new THREE.LineSegments(outline(kind), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }));
  body.add(fillMesh, wire);
  const group = new THREE.Group();
  group.add(body);
  return group;
}

export function setPrimitivePose(group: THREE.Group, prim: Prim): void {
  const body = group.getObjectByName('body');
  if (body) body.scale.set(...bodyScale(prim.kind, prim.size));
  const m = new THREE.Matrix4().fromArray(prim.transform);
  m.decompose(group.position, group.quaternion, group.scale);
  group.scale.copy(ONE);
  group.updateMatrixWorld(true);
}

export function disposeHelper(group: THREE.Object3D): void {
  group.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.geometry) m.geometry.dispose();
    const mat = m.material as THREE.Material | undefined;
    mat?.dispose();
  });
}

export class Primitives {
  /** emitted when the gizmo moves the active primitive */
  onChange: ((p: Prim) => void) | null = null;

  private active: { group: THREE.Group; prim: Prim } | null = null;
  private readonly committed = new Map<string, { group: THREE.Group; kind: PrimitiveKind }>();
  private readonly root = new THREE.Group();

  constructor(private readonly vp: Viewport) {
    vp.scene.add(this.root);
  }

  dispose(): void {
    this.setActive(null);
    for (const id of [...this.committed.keys()]) this.removeCommitted(id);
    this.vp.scene.remove(this.root);
  }

  /** Mirror the store's active primitive into the scene (no-op when it already matches the helper). */
  setActive(prim: Prim | null): void {
    if (!prim) {
      if (this.active) {
        this.root.remove(this.active.group);
        disposeHelper(this.active.group);
        this.active = null;
        this.vp.requestRender();
      }
      return;
    }
    if (!this.active || this.active.prim.kind !== prim.kind) {
      this.setActive(null);
      const group = makePrimitiveHelper(prim.kind, 0xffa534, 0.2);
      this.root.add(group);
      this.active = { group, prim };
      setPrimitivePose(group, prim);
    } else {
      const cur = this.active.prim;
      const same =
        cur.transform.every((v, i) => Math.abs(v - prim.transform[i]!) < 1e-9) &&
        cur.size.every((v, i) => Math.abs(v - prim.size[i]!) < 1e-9);
      if (!same) {
        setPrimitivePose(this.active.group, prim);
        this.active.prim = prim;
      }
    }
    this.vp.requestRender();
  }

  /** Hand the shared gizmo to the active primitive (bind.ts decides who owns it). */
  attachGizmo(): void {
    if (this.active) this.vp.attachGizmo(this.active.group, this.emit, this.finish);
  }

  /** Static outlines of the primitives already used by loads/supports. */
  setCommitted(items: readonly { id: string; prim: Prim; color: number }[]): void {
    const keep = new Set(items.map((i) => i.id));
    for (const id of [...this.committed.keys()]) if (!keep.has(id)) this.removeCommitted(id);
    for (const it of items) {
      let entry = this.committed.get(it.id);
      if (!entry || entry.kind !== it.prim.kind) {
        if (entry) this.removeCommitted(it.id);
        const group = makePrimitiveHelper(it.prim.kind, it.color, 0.08);
        this.root.add(group);
        entry = { group, kind: it.prim.kind };
        this.committed.set(it.id, entry);
      }
      setPrimitivePose(entry.group, it.prim);
    }
    this.vp.requestRender();
  }

  private removeCommitted(id: string): void {
    const e = this.committed.get(id);
    if (!e) return;
    this.root.remove(e.group);
    disposeHelper(e.group);
    this.committed.delete(id);
  }

  /** Pose + size as a PrimitiveSelection; scale applied by the gizmo is left out (folded on mouse-up). */
  private snapshot(): Prim | null {
    const a = this.active;
    if (!a) return null;
    const m = new THREE.Matrix4().compose(a.group.position, a.group.quaternion, ONE);
    return { kind: a.prim.kind, transform: m.toArray(), size: [...a.prim.size], surface_only: a.prim.surface_only };
  }

  private emit = (): void => {
    const p = this.snapshot();
    if (!p || !this.active) return;
    this.active.prim = p;
    this.onChange?.(p);
  };

  /** Scale gizmo: fold group.scale into `size` once the drag ends (folding mid-drag would compound). */
  private finish = (): void => {
    const a = this.active;
    if (!a) return;
    const s = a.group.scale;
    if (Math.abs(s.x - 1) + Math.abs(s.y - 1) + Math.abs(s.z - 1) > 1e-9) {
      const size = [...a.prim.size];
      if (a.prim.kind === 'box') {
        size[0] = size[0]! * s.x;
        size[1] = size[1]! * s.y;
        size[2] = size[2]! * s.z;
      } else if (a.prim.kind === 'sphere') {
        const dev = [s.x, s.y, s.z].reduce((best, v) => (Math.abs(v - 1) > Math.abs(best - 1) ? v : best), 1);
        size[0] = size[0]! * dev;
      } else {
        const devR = Math.abs(s.x - 1) >= Math.abs(s.z - 1) ? s.x : s.z;
        size[0] = size[0]! * devR;
        size[1] = size[1]! * s.y;
      }
      a.group.scale.copy(ONE);
      a.prim = { ...a.prim, size };
      setPrimitivePose(a.group, a.prim);
    }
    this.emit();
  };
}
