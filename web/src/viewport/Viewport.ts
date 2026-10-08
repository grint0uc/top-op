// Vanilla three.js viewport: owns the canvas, renderer, camera, controls and the design/ref/result meshes.
// React never touches three objects; viewport/bind.ts mirrors store state into this class.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import type { GizmoMode, ToolMode } from '../state/store';
import { DensityView } from './DensityView';
import { Markers } from './Markers';
import { type MeshData, parseMeshBuffer } from './meshData';
import { Picker, type PickerDeps } from './Picker';
import { Primitives } from './Primitives';
import { SelectionPainter, type PainterDeps } from './SelectionPainter';

export const BASE_HEX = 0xb8bec8;
export const SELECT_HEX = 0xffa534;
const BG_HEX = 0x14161a;

export type DesignAppearance = 'solid' | 'ghost' | 'hidden';

export interface DesignMesh {
  meshId: string | null;
  data: MeshData;
  geometry: THREE.BufferGeometry;
  mesh: THREE.Mesh;
  material: THREE.MeshStandardMaterial;
}

export interface FaceGroup {
  faces: readonly number[];
  color: number;
}

export interface FaceHit {
  face: number;
  point: THREE.Vector3;
  normal: THREE.Vector3;
}

interface RefEntry {
  mesh: THREE.Mesh; // matrix = the RefModel transform (matrixAutoUpdate off)
  material: THREE.MeshStandardMaterial;
  /** gizmo proxy sitting at transform * bboxCentre, so the handles appear on the geometry, not at the mesh origin */
  pivot: THREE.Object3D;
  centre: THREE.Vector3;
}

export const REF_COLORS = { keep_in: 0x3ddc97, keep_out: 0xff5a5f } as const;

export class Viewport {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly gizmo: TransformControls;
  readonly picker: Picker;
  readonly painter: SelectionPainter;
  readonly primitives: Primitives;
  readonly markers: Markers;
  readonly density: DensityView;

  mode: ToolMode = 'orbit';
  design: DesignMesh | null = null;
  /** emitted (column-major Matrix4.toArray) while a reference model is dragged with the gizmo */
  onRefTransform: ((id: string, matrix: number[]) => void) | null = null;
  renderCount = 0;

  private readonly refs = new Map<string, RefEntry>();
  private grid: THREE.GridHelper | null = null;
  private axes: THREE.AxesHelper | null = null;
  private faceHex = new Uint32Array(0);
  private faceNext = new Uint32Array(0);
  private gizmoTarget: { object: THREE.Object3D; onChange: () => void; onEnd: () => void } | null = null;
  private gizmoDragging = false;
  private dirty = true;
  private raf = 0;
  private disposed = false;
  private readonly ro: ResizeObserver;
  private readonly raycaster = new THREE.Raycaster();
  private readonly ndc = new THREE.Vector2();

  constructor(
    readonly canvas: HTMLCanvasElement,
    deps: { picker: PickerDeps; painter: PainterDeps },
  ) {
    // preserveDrawingBuffer keeps screenshots/readPixels valid between the on-demand renders
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(BG_HEX);

    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 5000);
    this.camera.up.set(0, 0, 1); // engineering convention: Z up
    this.camera.position.set(120, -140, 100);
    this.scene.add(this.camera);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x3a3f4a, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(1, 0.6, 1.4);
    this.camera.add(sun); // light travels with the camera so every orbit angle is lit

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.addEventListener('change', this.requestRender);

    this.gizmo = new TransformControls(this.camera, canvas);
    this.scene.add(this.gizmo.getHelper());
    this.gizmo.addEventListener('dragging-changed', (e) => {
      this.gizmoDragging = !!e.value;
      this.controls.enabled = !e.value;
    });
    this.gizmo.addEventListener('change', this.requestRender);
    this.gizmo.addEventListener('objectChange', () => this.gizmoTarget?.onChange());
    this.gizmo.addEventListener('mouseUp', () => this.gizmoTarget?.onEnd());

    this.markers = new Markers(this);
    this.density = new DensityView(this);
    this.primitives = new Primitives(this);
    this.picker = new Picker(this, deps.picker);
    this.painter = new SelectionPainter(this, deps.painter);
    this.scene.add(this.markers.group, this.density.group);

    this.rebuildGrid(new THREE.Box3(new THREE.Vector3(-50, -50, 0), new THREE.Vector3(50, 50, 0)));

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas.parentElement ?? canvas);
    this.resize();
    const loop = () => {
      this.raf = requestAnimationFrame(loop);
      if (this.dirty) this.renderNow();
    };
    loop();
  }

  // ---------------------------------------------------------------- render loop
  /** Marks the scene dirty; the rAF loop renders at most once per frame, and only when dirty. */
  readonly requestRender = (): void => {
    this.dirty = true;
  };

  renderNow(): void {
    if (this.disposed) return;
    this.dirty = false;
    this.renderer.render(this.scene, this.camera);
    this.renderCount++;
  }

  private resize(): void {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  // ---------------------------------------------------------------- mode
  setMode(mode: ToolMode): void {
    this.mode = mode;
    // paint mode owns the left button: orbit moves to the right button
    this.controls.mouseButtons =
      mode === 'paint'
        ? { LEFT: null, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE }
        : { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    this.canvas.style.cursor = mode === 'pick' || mode === 'paint' ? 'crosshair' : '';
    this.painter.setActive(mode === 'paint');
    this.syncGizmo();
    this.requestRender();
  }

  setGizmoMode(mode: GizmoMode): void {
    this.gizmo.setMode(mode);
    this.requestRender();
  }

  /** The gizmo is only live in gizmo mode; the target is remembered across mode switches. */
  attachGizmo(object: THREE.Object3D | null, onChange: () => void = () => {}, onEnd: () => void = () => {}): void {
    this.gizmoTarget = object ? { object, onChange, onEnd } : null;
    this.syncGizmo();
  }

  private syncGizmo(): void {
    if (this.mode === 'gizmo' && this.gizmoTarget) {
      if (this.gizmo.object !== this.gizmoTarget.object) this.gizmo.attach(this.gizmoTarget.object);
    } else if (!this.gizmoDragging) {
      this.gizmo.detach();
    }
    this.requestRender();
  }

  // ---------------------------------------------------------------- design mesh
  loadDesignMesh(buffer: ArrayBuffer, opts: { meshId?: string; data?: MeshData } = {}): THREE.BufferGeometry {
    this.clearDesignMesh();
    const data = opts.data ?? parseMeshBuffer(buffer);
    const n = data.nTri;
    // non-indexed: three vertices per triangle, so one face's colour never bleeds into its neighbours
    const pos = new Float32Array(n * 9);
    const nor = new Float32Array(n * 9);
    const col = new Float32Array(n * 9);
    const base = new THREE.Color(BASE_HEX);
    for (let f = 0; f < n; f++) {
      for (let v = 0; v < 3; v++) {
        const vi = data.tris[f * 3 + v]! * 3;
        for (let k = 0; k < 3; k++) {
          pos[f * 9 + v * 3 + k] = data.positions[vi + k]!;
          nor[f * 9 + v * 3 + k] = data.normals[f * 3 + k]!;
        }
        col.set([base.r, base.g, base.b], f * 9 + v * 3);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.75,
      metalness: 0.05,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'design';
    this.scene.add(mesh);
    this.design = { meshId: opts.meshId ?? null, data, geometry, mesh, material };
    this.appearance = 'solid';
    this.faceHex = new Uint32Array(n).fill(BASE_HEX);
    this.faceNext = new Uint32Array(n);
    this.rebuildGrid(geometry.boundingBox!);
    this.requestRender();
    return geometry;
  }

  clearDesignMesh(): void {
    if (!this.design) return;
    this.scene.remove(this.design.mesh);
    this.design.geometry.dispose();
    this.design.material.dispose();
    this.design = null;
    this.requestRender();
  }

  private appearance: DesignAppearance = 'solid';

  setDesignAppearance(mode: DesignAppearance): void {
    const d = this.design;
    if (!d || mode === this.appearance) return;
    this.appearance = mode;
    d.mesh.visible = mode !== 'hidden';
    d.material.transparent = mode === 'ghost';
    d.material.opacity = mode === 'ghost' ? 0.1 : 1;
    d.material.depthWrite = mode !== 'ghost';
    d.material.needsUpdate = true;
    this.requestRender();
  }

  /** Recolours only the faces whose colour changed (in place; the geometry is never rebuilt). */
  setFaceLayers(selected: readonly number[], groups: readonly FaceGroup[]): void {
    const d = this.design;
    if (!d) return;
    const n = d.data.nTri;
    const next = this.faceNext;
    next.fill(BASE_HEX);
    for (const g of groups) for (const f of g.faces) if (f < n) next[f] = g.color;
    for (const f of selected) if (f < n) next[f] = SELECT_HEX;
    const attr = d.geometry.getAttribute('color') as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    const c = new THREE.Color();
    let lo = n;
    let hi = -1;
    for (let f = 0; f < n; f++) {
      const hex = next[f]!;
      if (hex === this.faceHex[f]) continue;
      this.faceHex[f] = hex;
      c.setHex(hex);
      for (let v = 0; v < 3; v++) arr.set([c.r, c.g, c.b], f * 9 + v * 3);
      if (f < lo) lo = f;
      hi = f;
    }
    if (hi >= 0) {
      attr.clearUpdateRanges();
      attr.addUpdateRange(lo * 9, (hi - lo + 1) * 9);
      attr.needsUpdate = true;
      this.requestRender();
    }
  }

  /** Raycast the design mesh from a client-space pointer position. */
  pickFace(clientX: number, clientY: number): FaceHit | null {
    const d = this.design;
    if (!d || !d.mesh.visible) return null;
    const r = this.canvas.getBoundingClientRect();
    this.ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(this.ndc, this.camera);
    const hit = this.raycaster.intersectObject(d.mesh, false)[0];
    if (!hit || hit.faceIndex == null) return null;
    const f = hit.faceIndex;
    const nn = d.data.normals;
    return { face: f, point: hit.point.clone(), normal: new THREE.Vector3(nn[f * 3], nn[f * 3 + 1], nn[f * 3 + 2]) };
  }

  // ---------------------------------------------------------------- camera
  private contentBox(): THREE.Box3 {
    const box = new THREE.Box3();
    if (this.design?.geometry.boundingBox) box.copy(this.design.geometry.boundingBox);
    for (const r of this.refs.values()) box.union(new THREE.Box3().setFromObject(r.mesh));
    if (box.isEmpty()) box.set(new THREE.Vector3(-50, -50, 0), new THREE.Vector3(50, 50, 50));
    return box;
  }

  /**
   * Iso view of the design. The orbit target is nudged onto the surface when the bbox centre is
   * empty space (an L-bracket), so the pixel at the viewport centre always lands on geometry.
   */
  fitCamera(): void {
    this.resize();
    const box = this.contentBox();
    const centre = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-6);
    const dir = new THREE.Vector3(1, -1.2, 0.8).normalize();
    const halfFov = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const fitHalf = Math.atan(Math.tan(halfFov) * Math.min(1, this.camera.aspect));
    const dist = (radius / Math.sin(fitHalf)) * 1.1;

    const place = (target: THREE.Vector3, d: number) => {
      this.camera.position.copy(target).addScaledVector(dir, d);
      this.camera.near = Math.max(d / 1000, 1e-4);
      this.camera.far = d + radius * 20;
      this.camera.lookAt(target);
      this.camera.updateProjectionMatrix();
      this.camera.updateMatrixWorld(true);
    };
    place(centre, dist);
    let target = centre;
    const d = this.design;
    if (d) {
      d.mesh.updateMatrixWorld(true);
      target = this.nearestSurfaceToCentre() ?? centre;
    }
    // with the target possibly off-centre, back the camera off until every bbox corner is on screen
    const corners = [0, 1, 2, 3, 4, 5, 6, 7].map(
      (i) => new THREE.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z),
    );
    let fit = dist;
    for (let i = 0; i < 8; i++) {
      place(target, fit);
      let worst = 0;
      for (const c of corners) {
        const p = c.clone().project(this.camera);
        worst = Math.max(worst, Math.abs(p.x), Math.abs(p.y));
      }
      fit = Math.min(Math.max(fit * (worst / 0.85), radius * 1.2), radius * 50);
    }
    place(target, fit);
    this.controls.target.copy(target);
    this.controls.update();
    this.requestRender();
  }

  /** First ray hit on a spiral of NDC offsets around the screen centre (centre itself first). */
  private nearestSurfaceToCentre(): THREE.Vector3 | null {
    const mesh = this.design!.mesh;
    for (let ring = 0; ring <= 12; ring++) {
      const r = ring * 0.05;
      const steps = ring === 0 ? 1 : 8 * ring;
      for (let i = 0; i < steps; i++) {
        const a = (i / steps) * Math.PI * 2;
        this.raycaster.setFromCamera(new THREE.Vector2(Math.cos(a) * r, Math.sin(a) * r), this.camera);
        const hit = this.raycaster.intersectObject(mesh, false)[0];
        if (hit) return hit.point.clone();
      }
    }
    return null;
  }

  private rebuildGrid(box: THREE.Box3): void {
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry.dispose();
    }
    if (this.axes) {
      this.scene.remove(this.axes);
      this.axes.geometry.dispose();
    }
    const size = box.getSize(new THREE.Vector3());
    const span = Math.max(size.x, size.y, size.z, 1e-3);
    const mag = 10 ** Math.floor(Math.log10(span * 2));
    const gridSize = Math.ceil((span * 2) / mag) * mag;
    this.grid = new THREE.GridHelper(gridSize, 20, 0x3a414c, 0x262b33);
    this.grid.rotation.x = Math.PI / 2; // GridHelper lies in XZ; we want XY (Z up)
    this.grid.position.set((box.min.x + box.max.x) / 2, (box.min.y + box.max.y) / 2, box.min.z - span * 0.002);
    this.axes = new THREE.AxesHelper(span * 0.6);
    this.axes.position.copy(box.min);
    this.scene.add(this.grid, this.axes);
  }

  // ---------------------------------------------------------------- reference models
  loadRefMesh(id: string, buffer: ArrayBuffer): THREE.Mesh {
    this.removeRefMesh(id);
    const data = parseMeshBuffer(buffer);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(data.tris, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    const material = new THREE.MeshStandardMaterial({
      color: REF_COLORS.keep_out,
      transparent: true,
      opacity: 0.35,
      depthWrite: false,
      side: THREE.DoubleSide,
      roughness: 0.8,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `ref:${id}`;
    mesh.matrixAutoUpdate = false;
    const pivot = new THREE.Object3D();
    pivot.name = `ref-pivot:${id}`;
    const centre = geometry.boundingBox!.getCenter(new THREE.Vector3());
    pivot.position.copy(centre);
    this.scene.add(mesh, pivot);
    this.refs.set(id, { mesh, material, pivot, centre });
    this.requestRender();
    return mesh;
  }

  hasRefMesh(id: string): boolean {
    return this.refs.has(id);
  }

  refIds(): string[] {
    return [...this.refs.keys()];
  }

  removeRefMesh(id: string): void {
    const r = this.refs.get(id);
    if (!r) return;
    if (this.gizmoTarget?.object === r.pivot) this.attachGizmo(null);
    this.scene.remove(r.mesh, r.pivot);
    r.mesh.geometry.dispose();
    r.material.dispose();
    this.refs.delete(id);
    this.requestRender();
  }

  setRefAppearance(id: string, mode: keyof typeof REF_COLORS, visible: boolean): void {
    const r = this.refs.get(id);
    if (!r) return;
    r.material.color.setHex(REF_COLORS[mode]);
    r.mesh.visible = visible;
    this.requestRender();
  }

  setRefTransform(id: string, matrix: readonly number[]): void {
    const r = this.refs.get(id);
    if (!r || matrix.length !== 16) return;
    if (r.mesh.matrix.elements.every((v, i) => Math.abs(v - matrix[i]!) < 1e-9)) return;
    r.mesh.matrix.fromArray(matrix as number[]);
    r.mesh.matrixWorldNeedsUpdate = true;
    const pose = r.mesh.matrix.clone().multiply(new THREE.Matrix4().makeTranslation(r.centre.x, r.centre.y, r.centre.z));
    pose.decompose(r.pivot.position, r.pivot.quaternion, r.pivot.scale);
    r.pivot.updateMatrixWorld(true);
    this.requestRender();
  }

  /** Attach the gizmo to a reference model (or detach with null). */
  setActiveRef(id: string | null): void {
    const r = id ? this.refs.get(id) : undefined;
    if (!r || !id) {
      if (this.gizmoTarget && [...this.refs.values()].some((x) => x.pivot === this.gizmoTarget?.object)) {
        this.attachGizmo(null);
      }
      return;
    }
    const emit = () => {
      r.pivot.updateMatrix();
      const m = r.pivot.matrix.clone().multiply(new THREE.Matrix4().makeTranslation(-r.centre.x, -r.centre.y, -r.centre.z));
      r.mesh.matrix.copy(m);
      r.mesh.matrixWorldNeedsUpdate = true;
      this.requestRender();
      this.onRefTransform?.(id, m.toArray());
    };
    this.attachGizmo(r.pivot, emit, emit);
  }

  // ---------------------------------------------------------------- result mesh
  showResultMesh(stl: ArrayBuffer): number {
    return this.density.showResultMesh(stl);
  }

  clearResultMesh(): void {
    this.density.clearResultMesh();
  }

  // ---------------------------------------------------------------- diagnostics (e2e)
  stats(): {
    renders: number;
    designFaces: number;
    result: boolean;
    refs: number;
    densityCount: number;
    densityMode: string;
    resolvedPoints: number;
    arrows: number;
    glyphs: number;
  } {
    return {
      renders: this.renderCount,
      designFaces: this.design?.data.nTri ?? 0,
      result: this.density.hasResult,
      refs: this.refs.size,
      densityCount: this.density.count,
      densityMode: this.density.mode,
      resolvedPoints: this.markers.counts().points,
      arrows: this.markers.counts().arrows,
      glyphs: this.markers.counts().glyphs,
    };
  }

  /** Client-space pixel of a world point (used by e2e to aim at gizmo handles). */
  screenPoint(world: readonly [number, number, number]): { x: number; y: number } {
    const r = this.canvas.getBoundingClientRect();
    const v = new THREE.Vector3(...world).project(this.camera);
    return { x: r.left + ((v.x + 1) / 2) * r.width, y: r.top + ((1 - v.y) / 2) * r.height };
  }

  /** Fraction of canvas pixels that differ from the background with grid/axes/gizmo hidden. */
  coverage(): number {
    const hidden: THREE.Object3D[] = [this.gizmo.getHelper()];
    if (this.grid) hidden.push(this.grid);
    if (this.axes) hidden.push(this.axes);
    const was = hidden.map((o) => o.visible);
    hidden.forEach((o) => (o.visible = false));
    this.renderNow();
    const gl = this.renderer.getContext();
    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const ref = [px[0]!, px[1]!, px[2]!];
    let diff = 0;
    for (let i = 0; i < px.length; i += 4) {
      if (Math.abs(px[i]! - ref[0]!) + Math.abs(px[i + 1]! - ref[1]!) + Math.abs(px[i + 2]! - ref[2]!) > 24) diff++;
    }
    hidden.forEach((o, i) => (o.visible = was[i]!));
    this.requestRender();
    return diff / (w * h);
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    this.picker.dispose();
    this.painter.dispose();
    this.primitives.dispose();
    this.markers.dispose();
    this.density.dispose();
    this.clearDesignMesh();
    for (const id of [...this.refs.keys()]) this.removeRefMesh(id);
    this.gizmo.dispose();
    this.controls.dispose();
    this.renderer.dispose();
  }
}
