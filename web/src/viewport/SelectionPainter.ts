// Brush selection: in paint mode a left-drag adds every triangle whose centroid lies within the
// brush radius (world units) of the hit point; ctrl/cmd-drag removes. A camera-facing circle shows the brush.
import * as THREE from 'three';
import type { Viewport } from './Viewport';

export interface PainterDeps {
  getSelection(): readonly number[];
  setSelection(ids: number[]): void;
  getRadius(): number;
}

function circleTexture(): THREE.CanvasTexture {
  const size = 128;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  if (g) {
    g.lineWidth = 6;
    g.strokeStyle = '#ffa534';
    g.fillStyle = 'rgba(255,165,52,0.15)';
    g.beginPath();
    g.arc(size / 2, size / 2, size / 2 - 5, 0, Math.PI * 2);
    g.fill();
    g.stroke();
  }
  return new THREE.CanvasTexture(c);
}

export class SelectionPainter {
  private readonly brush: THREE.Sprite;
  private active = false;
  private painting = false;
  private removing = false;
  private working: Set<number> | null = null;
  private publishQueued = false;

  constructor(
    private readonly vp: Viewport,
    private readonly deps: PainterDeps,
  ) {
    this.brush = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: circleTexture(), depthTest: false, transparent: true }),
    );
    this.brush.visible = false;
    this.brush.renderOrder = 10;
    vp.scene.add(this.brush);
    const c = vp.canvas;
    c.addEventListener('pointerdown', this.onDown);
    c.addEventListener('pointermove', this.onMove);
    c.addEventListener('pointerup', this.onUp);
    c.addEventListener('pointercancel', this.onUp);
    c.addEventListener('pointerleave', this.onLeave);
  }

  setActive(on: boolean): void {
    this.active = on;
    if (!on) {
      this.brush.visible = false;
      this.painting = false;
      this.vp.requestRender();
    }
  }

  dispose(): void {
    const c = this.vp.canvas;
    c.removeEventListener('pointerdown', this.onDown);
    c.removeEventListener('pointermove', this.onMove);
    c.removeEventListener('pointerup', this.onUp);
    c.removeEventListener('pointercancel', this.onUp);
    c.removeEventListener('pointerleave', this.onLeave);
    this.vp.scene.remove(this.brush);
    this.brush.material.map?.dispose();
    this.brush.material.dispose();
  }

  private onDown = (e: PointerEvent): void => {
    if (!this.active || e.button !== 0) return;
    this.painting = true;
    this.removing = e.ctrlKey || e.metaKey;
    this.working = new Set(this.deps.getSelection());
    this.vp.canvas.setPointerCapture(e.pointerId);
    this.stamp(e);
  };

  private onMove = (e: PointerEvent): void => {
    if (!this.active) return;
    const hit = this.vp.pickFace(e.clientX, e.clientY);
    this.brush.visible = !!hit;
    if (hit) {
      const r = this.deps.getRadius();
      this.brush.position.copy(hit.point);
      this.brush.scale.setScalar(r * 2);
    }
    this.vp.requestRender();
    if (this.painting) this.stamp(e, hit);
  };

  private onUp = (): void => {
    if (!this.painting) return;
    this.painting = false;
    this.flush();
    this.working = null;
  };

  private onLeave = (): void => {
    this.brush.visible = false;
    this.vp.requestRender();
  };

  private stamp(e: PointerEvent, hit = this.vp.pickFace(e.clientX, e.clientY)): void {
    const design = this.vp.design;
    const p = hit?.point;
    if (!design || !hit || !p || !this.working) return;
    const r2 = this.deps.getRadius() ** 2;
    // the triangle under the cursor is always painted, even when its centroid is outside the brush (large triangles)
    if (this.removing) this.working.delete(hit.face);
    else this.working.add(hit.face);
    const c = design.data.centroids;
    for (let f = 0; f < design.data.nTri; f++) {
      const dx = c[f * 3]! - p.x;
      const dy = c[f * 3 + 1]! - p.y;
      const dz = c[f * 3 + 2]! - p.z;
      if (dx * dx + dy * dy + dz * dz <= r2) {
        if (this.removing) this.working.delete(f);
        else this.working.add(f);
      }
    }
    // publish at most once per frame while dragging
    if (!this.publishQueued) {
      this.publishQueued = true;
      requestAnimationFrame(() => {
        this.publishQueued = false;
        this.flush();
      });
    }
  }

  private flush(): void {
    if (this.working) this.deps.setSelection([...this.working]);
  }
}
