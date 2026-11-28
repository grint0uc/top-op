import { useEffect, useMemo, useRef, useState } from 'react';
import { requiredMeshes } from '../state/derived';
import { useStore } from '../state/store';
import { mountViewport } from '../viewport/mount';
import type { Viewport } from '../viewport/Viewport';
import { AppMark, Icon } from './icons';
import { IS_MOCK, loadExample, pickFile } from './ImportPanel';
import { NoticeBar } from './NoticeBar';
import { SelectionToolbar } from './SelectionToolbar';
import { useUi } from './uiStore';
import { ViewCube } from './ViewCube';
import { ViewToolbar } from './ViewToolbar';

const HINTS: Record<string, string> = {
  orbit: 'Orbit: left-drag rotate, right-drag pan, wheel zoom',
  pick: 'Pick: click face, Shift+click grow flat face, Ctrl/Cmd+click remove, Esc clear',
  paint: 'Paint: drag to add faces, Ctrl/Cmd-drag to remove, right-drag orbits',
  gizmo: 'Gizmo: drag handles; g move, r rotate, s scale',
  query: 'Query: describe the region (facet / normal / plane); matching faces light up, Nodes shows the grid nodes',
};

/** Mirrors the UI store (theme, display style, section plane) into the viewport. */
function useViewportUi(vp: Viewport | null): void {
  useEffect(() => {
    if (!vp) return;
    const apply = () => {
      const u = useUi.getState();
      vp.setTheme(u.dark);
      vp.setDisplayStyle(u.display);
      vp.setSection(u.section.on ? { axis: u.section.axis, t: u.section.t } : null);
    };
    apply();
    const unsubUi = useUi.subscribe((s, p) => {
      if (s.dark !== p.dark || s.display !== p.display || s.section !== p.section) apply();
    });
    // the section plane is placed on the content box: follow the design pose and new geometry
    const unsubStore = useStore.subscribe((s, p) => {
      if (s.project.design_mesh !== p.project.design_mesh || s.meshes !== p.meshes) apply();
    });
    return () => {
      unsubUi();
      unsubStore();
    };
  }, [vp]);
}

function Welcome() {
  const hasDesign = useStore((s) => !!s.project.design_mesh?.mesh_id);
  const busy = useStore((s) => s.busy);
  // the welcome card (and the button that imported) goes away: keep keyboard focus on the page, on the Design body page
  useEffect(() => {
    if (hasDesign && (document.activeElement === document.body || document.activeElement === null))
      document.querySelector<HTMLElement>('[data-testid="pm-title"]')?.focus();
  }, [hasDesign]);
  if (hasDesign) return null;
  return (
    <div className="welcome" data-testid="welcome">
      <AppMark size={36} />
      <h2>Start a topology study</h2>
      <p className="dim">Import the design body (STL or STEP), drop a file anywhere, or open a saved project.json.</p>
      <div className="row center">
        <button type="button" className="btn primary" onClick={() => pickFile('mesh')} disabled={busy === 'Upload'} data-testid="welcome-import">
          <Icon name="import" size={14} />
          Import STL / STEP
        </button>
        <button type="button" className="btn" onClick={() => pickFile('project')} data-testid="welcome-open">
          <Icon name="open" size={14} />
          Open project
        </button>
        {IS_MOCK && (
          <button type="button" className="btn" onClick={() => void loadExample()} data-testid="welcome-example">
            <Icon name="example" size={14} />
            Example bracket
          </button>
        )}
      </div>
      <ol className="steps">
        <li>
          <b>Model</b> import the part
        </li>
        <li>
          <b>Boundary conditions</b> select faces, add loads and fixtures
        </li>
        <li>
          <b>Domain</b> resolution and keep-in / keep-out bodies
        </li>
        <li>
          <b>Optimize</b> volume fraction, constraints, run
        </li>
        <li>
          <b>Results</b> density, stress, export STL
        </li>
      </ol>
    </div>
  );
}

export function ViewportHost({ dropping }: { dropping: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [vp, setVp] = useState<Viewport | null>(null);
  const tool = useStore((s) => s.tool);
  const hasMesh = useStore((s) => !!(s.project.design_mesh?.mesh_id && s.meshes[s.project.design_mesh.mesh_id]));
  const project = useStore((s) => s.project);
  const meshes = useStore((s) => s.meshes);
  const meshMemo = useStore((s) => s.meshMemo);
  const missing = useMemo(() => requiredMeshes({ project, meshes, meshMemo }).some((m) => !m.loaded), [project, meshes, meshMemo]);

  useEffect(() => {
    if (ref.current) setVp(mountViewport(ref.current));
  }, []);
  useViewportUi(vp);

  return (
    <div className="viewport">
      <canvas id="viewport" ref={ref} role="img" aria-label="3D viewport" aria-describedby="viewport-help" />
      <p id="viewport-help" className="visually-hidden">
        {hasMesh ? HINTS[tool] : 'Import an STL or STEP file to begin.'} Without a mouse: the View toolbar (fit, standard
        views, display style, section) and the view cube's Isometric button change the view; the selection toolbar's Query
        tool (normal, plane or facet selections) and its Box, Sphere and Cylinder regions select faces and nodes for loads
        and supports.
      </p>
      <SelectionToolbar />
      <NoticeBar />
      {!missing && <Welcome />}
      {vp && <ViewCube vp={vp} />}
      {vp && <ViewToolbar vp={vp} />}
      <div className="hint" data-testid="hud">
        {hasMesh ? HINTS[tool] : 'Import an STL or STEP file to begin'}
      </div>
      {dropping && (
        <div className="drop-overlay">
          <Icon name="import" size={28} />
          Drop an STL / STEP to import it, or a project.json to open it
        </div>
      )}
    </div>
  );
}
