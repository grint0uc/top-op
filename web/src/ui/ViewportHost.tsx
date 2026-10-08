import { useEffect, useRef } from 'react';
import { mountViewport } from '../viewport/mount';
import { useStore } from '../state/store';
import { SelectionToolbar } from './SelectionToolbar';

const HINTS: Record<string, string> = {
  orbit: 'Orbit: left-drag rotate, right-drag pan, wheel zoom',
  pick: 'Pick: click face, Shift+click grow flat face, Ctrl/Cmd+click remove, Esc clear',
  paint: 'Paint: drag to add faces, Ctrl/Cmd-drag to remove, right-drag orbits',
  gizmo: 'Gizmo: drag handles; g move, r rotate, s scale',
  query: 'Query: describe the region (facet / normal / plane); matching faces light up, Resolve preview shows the grid nodes',
};

export function ViewportHost() {
  const ref = useRef<HTMLCanvasElement>(null);
  const tool = useStore((s) => s.tool);
  const hasMesh = useStore((s) => !!(s.project.design_mesh?.mesh_id && s.meshes[s.project.design_mesh.mesh_id]));

  useEffect(() => {
    if (ref.current) mountViewport(ref.current);
  }, []);

  return (
    <>
      <canvas id="viewport" ref={ref} />
      <SelectionToolbar />
      <div className="hud" data-testid="hud">
        {hasMesh ? HINTS[tool] : 'Import an STL to begin'}
      </div>
    </>
  );
}
