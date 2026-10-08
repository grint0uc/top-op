// Mirrors store state into a Viewport (store -> viewport) and viewport edits back (viewport -> store).
import type { DensityFrame, Selection } from '../api/client';
import {
  SUPPORT_COLOR,
  bboxDiagonal,
  caseColor,
  designBox,
  designEntry,
  designPose,
  designWorld,
  domainBox,
  facetKey,
  isPrimitive,
  selectionAnchor,
  toPrim,
} from '../state/derived';
import { IDENTITY } from '../state/defaults';
import { selectionFaces } from '../state/query';
import { type State, useStore } from '../state/store';
import type { DensityGrid } from './DensityView';
import type { LoadMarker, SupportMarker } from './Markers';
import type { SymmetryPlane } from './Overlays';
import type { FaceGroup, Viewport } from './Viewport';

function gridFor(frame: DensityFrame, s: State): DensityGrid {
  const st = s.run.stats ?? s.voxel.stats;
  const [nx, ny, nz] = frame.shape;
  if (st && st.nx === nx && st.ny === ny && st.nz === nz) return { origin: st.origin, h: st.h };
  // no matching stats: fit the frame to the (transformed) design bbox (padding cells on every side)
  const box = designBox(s);
  const pad = s.project.grid.padding;
  if (!box) return { origin: [0, 0, 0], h: 1 };
  const { min, max } = box;
  const h = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / Math.max(1, Math.max(nx, ny, nz) - 2 * pad);
  return { origin: [min[0] - pad * h, min[1] - pad * h, min[2] - pad * h], h };
}

export function bindViewport(vp: Viewport): () => void {
  const store = useStore;
  const unsubs: Array<() => void> = [];

  function watch<T>(sel: (s: State) => T, cb: (v: T, s: State) => void): void {
    cb(sel(store.getState()), store.getState());
    unsubs.push(
      store.subscribe((s, prev) => {
        const v = sel(s);
        if (!Object.is(v, sel(prev))) cb(v, s);
      }),
    );
  }

  // ---- viewport -> store
  vp.primitives.onChange = (p) => store.getState().setPrimitive(p);
  vp.onRefTransform = (id, m) => store.getState().updateRef(id, { transform: m });
  vp.onDesignTransform = (m) => store.getState().setDesignTransform(m);

  // ---- helpers
  /**
   * Faces a load/support/query selection covers on the design mesh (faces: as given; facets: the exact ids from the
   * faces endpoint, approximated until they arrive; normal: client preview in world space).
   */
  const facesOf = (sel: Selection, s: State, design: NonNullable<ReturnType<typeof designEntry>>): readonly number[] => {
    if (!('mesh_id' in sel) || sel.mesh_id !== design.info.id) return [];
    if (sel.kind === 'faces') return sel.face_ids;
    const table = sel.kind === 'facets' ? (s.facetCache[facetKey(sel.mesh_id, sel.angle_deg)] ?? null) : null;
    return selectionFaces(sel, design.data, table, s.facetFaceCache, designWorld(s) ?? design.data);
  };

  const applyFaces = () => {
    const s = store.getState();
    const design = designEntry(s);
    if (!design) return;
    const groups: FaceGroup[] = [];
    for (const l of s.project.loads) groups.push({ faces: facesOf(l.selection, s, design), color: caseColor(l.case) });
    for (const x of s.project.supports) groups.push({ faces: facesOf(x.selection, s, design), color: SUPPORT_COLOR });
    const selected = s.selection.query ? facesOf(s.selection.query, s, design) : s.selection.faceIds;
    vp.setFaceLayers(selected, groups, s.hoverFaces);
  };

  const applyAppearance = () => {
    const s = store.getState();
    const overlay = (!!s.run.densityFrame && !s.resultStl) || !!s.resultStl;
    vp.setDesignAppearance(overlay ? (s.ghostDesign ? 'ghost' : 'hidden') : 'solid');
    vp.density.setVisible(!s.resultStl);
  };

  const reportDensity = () =>
    store.getState().setDensityInfo({ mode: vp.density.mode, count: vp.density.count, it: vp.density.it });

  const syncMarkers = () => {
    const s = store.getState();
    const loads: LoadMarker[] = [];
    const supports: SupportMarker[] = [];
    const committed: { id: string; prim: ReturnType<typeof toPrim>; color: number }[] = [];
    const pose = designPose(s);
    for (const l of s.project.loads) {
      const at = selectionAnchor(l.selection, s.meshes, s.facetCache, pose);
      if (at) loads.push({ id: l.id, at, force: l.force, color: caseColor(l.case) });
      if (isPrimitive(l.selection)) committed.push({ id: l.id, prim: toPrim(l.selection), color: caseColor(l.case) });
    }
    for (const x of s.project.supports) {
      const at = selectionAnchor(x.selection, s.meshes, s.facetCache, pose);
      if (at) supports.push({ id: x.id, at, color: SUPPORT_COLOR });
      if (isPrimitive(x.selection)) committed.push({ id: x.id, prim: toPrim(x.selection), color: SUPPORT_COLOR });
    }
    vp.markers.setLoads(loads);
    vp.markers.setSupports(supports);
    vp.primitives.setCommitted(committed);
  };

  const syncRefs = () => {
    const s = store.getState();
    const live = new Set<string>();
    for (const r of s.project.ref_models) {
      const entry = r.mesh_id ? s.meshes[r.mesh_id] : undefined;
      if (!entry) continue;
      live.add(r.id);
      if (!vp.hasRefMesh(r.id)) vp.loadRefMesh(r.id, entry.buffer);
      vp.setRefAppearance(r.id, r.mode, r.visible);
      vp.setRefTransform(r.id, r.transform ?? IDENTITY);
    }
    for (const id of vp.refIds()) if (!live.has(id)) vp.removeRefMesh(id);
    syncGizmoTarget();
  };

  /** One gizmo, two possible owners: an active primitive wins, otherwise the active reference model. */
  function syncGizmoTarget(): void {
    const s = store.getState();
    if (s.selection.primitive) vp.primitives.attachGizmo();
    else if (s.activeItem?.kind === 'ref') vp.setActiveRef(s.activeItem.id);
    else if (s.activeItem?.kind === 'design') vp.setActiveDesign(true);
    else vp.attachGizmo(null);
  }

  /** Symmetry planes and the overhang base plate, drawn on the domain box (voxel grid, else the transformed design bbox). */
  const syncOverlays = () => {
    const s = store.getState();
    const box = domainBox(s);
    const near = designBox(s) ?? box; // "center" = the middle of the part, as the server picks the active region's centre
    const planes: SymmetryPlane[] = [];
    for (const sym of s.project.params.symmetry ?? []) {
      const a = 'xyz'.indexOf(sym.axis);
      const position = sym.position ?? (near ? (near.min[a]! + near.max[a]!) / 2 : 0);
      planes.push({ axis: sym.axis, position });
    }
    vp.overlays.set(box, planes, s.project.params.overhang ?? null);
  };

  // ---- store -> viewport
  watch(
    (s) => designEntry(s),
    (entry) => {
      if (entry) {
        vp.loadDesignMesh(entry.buffer, { meshId: entry.info.id, data: entry.data });
        vp.setDesignTransform(store.getState().project.design_mesh?.transform ?? IDENTITY);
        vp.markers.setScale(bboxDiagonal(designWorld(store.getState()) ?? entry.data));
        vp.fitCamera();
        applyFaces();
        applyAppearance();
        syncMarkers();
        syncGizmoTarget();
        syncOverlays();
      } else {
        vp.clearDesignMesh();
      }
    },
  );
  watch(
    (s) => s.project.design_mesh?.transform,
    (t) => {
      vp.setDesignTransform(t ?? IDENTITY);
      applyFaces();
      syncMarkers();
      syncOverlays();
    },
  );
  watch((s) => s.project.ref_models, syncRefs);
  watch((s) => s.meshes, syncRefs);
  watch((s) => s.activeItem, syncGizmoTarget);
  watch((s) => s.tool, (t) => vp.setMode(t));
  watch((s) => s.gizmoMode, (m) => vp.setGizmoMode(m));
  watch((s) => s.selection.faceIds, applyFaces);
  watch((s) => s.selection.query, applyFaces);
  watch((s) => s.hoverFaces, applyFaces);
  watch((s) => s.facetCache, () => {
    applyFaces();
    syncMarkers();
  });
  watch((s) => s.facetFaceCache, applyFaces);
  watch((s) => s.project.params.symmetry, syncOverlays);
  watch((s) => s.project.params.overhang, syncOverlays);
  watch((s) => s.voxel.stats, syncOverlays);
  watch((s) => s.selection.primitive, (p) => {
    vp.primitives.setActive(p);
    syncGizmoTarget();
  });
  watch((s) => s.project.loads, () => {
    applyFaces();
    syncMarkers();
  });
  watch((s) => s.project.supports, () => {
    applyFaces();
    syncMarkers();
  });
  watch((s) => s.preview, (p) => vp.markers.setPoints(p?.xyz ?? null));
  watch(
    (s) => s.run.densityFrame,
    (f, s) => {
      if (f) vp.density.setFrame(f, gridFor(f, s), s.threshold);
      else vp.density.clear();
      reportDensity();
      applyAppearance();
    },
  );
  // run.stats (origin, h) from the `started` frame can land after the first density frame: re-place the cells
  watch(
    (s) => s.run.stats,
    (_st, s) => {
      const f = s.run.densityFrame;
      if (f) vp.density.setFrame(f, gridFor(f, s), s.threshold);
    },
  );
  watch((s) => s.threshold, (t) => {
    vp.density.setThreshold(t);
    reportDensity();
  });
  watch((s) => (s.colorByStress ? s.stress : null), (st) => {
    vp.density.setStress(st);
    reportDensity();
  });
  watch((s) => s.ghostDesign, applyAppearance);
  watch(
    (s) => s.resultStl,
    (buf) => {
      if (buf) vp.showResultMesh(buf);
      else vp.clearResultMesh();
      applyAppearance();
    },
  );

  return () => {
    unsubs.forEach((u) => u());
    vp.primitives.onChange = null;
    vp.onRefTransform = null;
    vp.onDesignTransform = null;
  };
}
