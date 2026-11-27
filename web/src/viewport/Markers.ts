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

  /** Where each load arrow is anchored, i.e. its tip (diagnostic: lets e2e check the markers follow the design transform). */
  arrowOrigins(): number[][] {
    return this.arrows.children.map((a) => (a.userData.anchor as THREE.Vector3).toArray().map((v) => Math.round(v * 1e4) / 1e4));
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
      // CAD convention: a solid arrow pointing at the region, its tip on the anchor
      const anchor = new THREE.Vector3(...it.at);
      const arrow = new THREE.Group();
      arrow.name = `load:${it.id}`;
      arrow.userData.anchor = anchor;
      arrow.position.copy(anchor).addScaledVector(dir, -len);
      arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      const mat = new THREE.MeshBasicMaterial({ color: it.color, depthTest: false });
      const head = len * 0.32;
      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(len * 0.035, len * 0.035, len - head, 10), mat);
      shaft.position.y = (len - head) / 2;
      const cone = new THREE.Mesh(new THREE.ConeGeometry(len * 0.11, head, 18), mat.clone());
      cone.position.y = len - head / 2;
      for (const part of [shaft, cone]) part.renderOrder = 6;
      arrow.add(shaft, cone);
      this.arrows.add(arrow);
    }
    this.vp.requestRender();
  }

  setSupports(items: readonly SupportMarker[]): void {
    this.lastSupports = items;
    this.clear(this.glyphs);
    const h = this.scale * 0.05;
    for (const it of items) {
      // fixture symbol: a cone touching the support centroid from below, standing on a hatched ground plate
      const glyph = new THREE.Group();
      glyph.name = `support:${it.id}`;
      glyph.position.set(it.at[0], it.at[1], it.at[2]);
      const cone = new THREE.Mesh(new THREE.ConeGeometry(h * 0.6, h, 16), new THREE.MeshBasicMaterial({ color: it.color, depthTest: false }));
      cone.rotation.x = Math.PI / 2; // apex towards +Z
      cone.position.z = -h / 2;
      const plate = new THREE.Mesh(
        new THREE.CylinderGeometry(h * 0.95, h * 0.95, h * 0.08, 24),
        new THREE.MeshBasicMaterial({ color: it.color, depthTest: false, transparent: true, opacity: 0.55 }),
      );
      plate.rotation.x = Math.PI / 2;
      plate.position.z = -h * 1.04;
      const hatch: number[] = [];
      for (let k = -2; k <= 2; k++) {
        const x = k * h * 0.4;
        hatch.push(x, 0, -h * 1.08, x - h * 0.3, 0, -h * 1.4);
      }
      const lines = new THREE.LineSegments(
        new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(hatch, 3)),
        new THREE.LineBasicMaterial({ color: it.color, depthTest: false }),
      );
      for (const part of [cone, plate, lines]) part.renderOrder = 6;
      glyph.add(cone, plate, lines);
      this.glyphs.add(glyph);
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
