// Click picking on the design mesh. Selection state lives in the store; this class only computes
// the new face set and hands it to deps.setSelection (the colour attribute is updated in place by the viewport).
import { type Adjacency, growFlat } from './meshData';
import type { Viewport } from './Viewport';

export interface PickerDeps {
  getSelection(): readonly number[];
  setSelection(ids: number[]): void;
  getGrowAngle(): number;
  /** Resolves the adjacency of the design mesh (fetches /adjacency on first use). */
  ensureAdjacency(): Promise<Adjacency | null>;
}

const CLICK_SLOP_PX = 4;

export class Picker {
  private down: { x: number; y: number } | null = null;

  constructor(
    private readonly vp: Viewport,
    private readonly deps: PickerDeps,
  ) {
    const c = vp.canvas;
    c.addEventListener('pointerdown', this.onDown);
    c.addEventListener('pointerup', this.onUp);
    c.addEventListener('pointercancel', this.onCancel);
  }

  dispose(): void {
    const c = this.vp.canvas;
    c.removeEventListener('pointerdown', this.onDown);
    c.removeEventListener('pointerup', this.onUp);
    c.removeEventListener('pointercancel', this.onCancel);
  }

  private onDown = (e: PointerEvent): void => {
    if (this.vp.mode !== 'pick' || e.button !== 0) return;
    this.down = { x: e.clientX, y: e.clientY };
  };

  private onCancel = (): void => {
    this.down = null;
  };

  private onUp = (e: PointerEvent): void => {
    const d = this.down;
    this.down = null;
    if (!d || this.vp.mode !== 'pick' || e.button !== 0) return;
    // a drag is an orbit/pan, not a click
    if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > CLICK_SLOP_PX) return;
    void this.click(e);
  };

  private async click(e: PointerEvent): Promise<void> {
    const remove = e.ctrlKey || e.metaKey;
    const grow = e.shiftKey;
    const hit = this.vp.pickFace(e.clientX, e.clientY);
    if (!hit) {
      if (!remove && !grow) this.deps.setSelection([]); // click on empty space clears
      return;
    }
    let faces = [hit.face];
    if (grow) {
      const adj = await this.deps.ensureAdjacency();
      const design = this.vp.design;
      if (adj && design) faces = growFlat(design.data, adj, hit.face, this.deps.getGrowAngle());
    }
    const next = new Set(remove || grow ? this.deps.getSelection() : []);
    for (const f of faces) {
      if (remove) next.delete(f);
      else next.add(f);
    }
    this.deps.setSelection([...next]);
  }
}
