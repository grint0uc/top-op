// Resolved-node preview (Points), load arrows and support glyphs. Colours are per load case.
import * as THREE from 'three';
import type { Viewport } from './Viewport';

export interface LoadMarker {
  id: string;
  at: readonly [number, number, number];
  force: readonly number[];
  color: number;
}

export interface SupportMarker {
  id: string;
  at: readonly [number, number, number];
  color: number;
}

export class Markers {
  readonly group = new THREE.Group();
  private points: THREE.Points | null = null;
  private readonly arrows = new THREE.Group();
  private readonly glyphs = new THREE.Group();
  private scale = 10;
  private lastLoads: readonly LoadMarker[] = [];
  private lastSupports: readonly SupportMarker[] = [];

  constructor(private readonly vp: Viewport) {
    this.group.name = 'markers';
    this.group.add(this.arrows, this.glyphs);
  }

  /** Characteristic length (arrows are 12% of it, cones 5%); set from the design bbox diagonal. */
  setScale(len: number): void {
    if (Math.abs(len - this.scale) < 1e-9) return;
    this.scale = len;
    this.setLoads(this.lastLoads);
    this.setSupports(this.lastSupports);
  }

  counts(): { points: number; arrows: number; glyphs: number } {
    return {
      points: this.points?.geometry.getAttribute('position')?.count ?? 0,
      arrows: this.arrows.children.length,
      glyphs: this.glyphs.children.length,
    };
  }

  setPoints(xyz: Float32Array | null, color = 0xffd23f): void {
    if (this.points) {
      this.group.remove(this.points);
      this.points.geometry.dispose();
      (this.points.material as THREE.Material).dispose();
      this.points = null;
    }
    if (xyz && xyz.length > 0) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
      const mat = new THREE.PointsMaterial({ color, size: 3, sizeAttenuation: false, depthTest: false });
      this.points = new THREE.Points(geo, mat);
      this.points.frustumCulled = false;
      this.points.renderOrder = 5;
      this.group.add(this.points);
    }
    this.vp.requestRender();
  }

  setLoads(items: readonly LoadMarker[]): void {
    this.lastLoads = items;
    this.clear(this.arrows);
    const len = this.scale * 0.12;
    for (const it of items) {
      const dir = new THREE.Vector3(...(it.force as [number, number, number]));
      if (dir.lengthSq() === 0) continue;
      dir.normalize();
      const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(...it.at), len, it.color, len * 0.35, len * 0.2);
      arrow.name = `load:${it.id}`;
      for (const part of [arrow.line, arrow.cone]) {
        (part.material as THREE.Material).depthTest = false;
        part.renderOrder = 6;
      }
      this.arrows.add(arrow);
    }
    this.vp.requestRender();
  }

  setSupports(items: readonly SupportMarker[]): void {
    this.lastSupports = items;
    this.clear(this.glyphs);
    const h = this.scale * 0.05;
    for (const it of items) {
      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(h * 0.6, h, 14),
        new THREE.MeshBasicMaterial({ color: it.color, depthTest: false }),
      );
      cone.name = `support:${it.id}`;
      cone.rotation.x = Math.PI / 2; // apex towards +Z, touching the support centroid from below
      cone.position.set(it.at[0], it.at[1], it.at[2] - h / 2);
      cone.renderOrder = 6;
      this.glyphs.add(cone);
    }
    this.vp.requestRender();
  }

  private clear(g: THREE.Group): void {
    for (const child of [...g.children]) {
      g.remove(child);
      child.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose();
        (m.material as THREE.Material | undefined)?.dispose();
      });
    }
  }

  dispose(): void {
    this.setPoints(null);
    this.clear(this.arrows);
    this.clear(this.glyphs);
  }
}
