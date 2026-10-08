import { designEntry } from '../state/derived';
import { useStore } from '../state/store';
import { isIdentity } from '../state/worldMesh';
import { Btn } from './controls';
import { PoseFields } from './PoseFields';

/** Pose of the design mesh: the same fields and gizmo as a reference model; the matrix goes to the server as MeshRef.transform. */
export function DesignTransform() {
  const present = useStore((s) => designEntry(s) !== null);
  const transform = useStore((s) => s.project.design_mesh?.transform);
  const active = useStore((s) => s.activeItem?.kind === 'design');
  const gizmoMode = useStore((s) => s.gizmoMode);
  const { setDesignTransform, setActiveItem, setTool, setGizmoMode } = useStore.getState();
  if (!present) return null;

  return (
    <div className="stack" data-testid="design-transform">
      <div className="row">
        <span className="dim grow">Design transform</span>
        <Btn
          active={active}
          onClick={() => {
            if (active) setActiveItem(null);
            else {
              setActiveItem({ kind: 'design', id: 'design' });
              setTool('gizmo');
            }
          }}
          testId="design-gizmo"
          title="Move / rotate / scale the design mesh with the gizmo (g / r / s)"
        >
          Gizmo
        </Btn>
        <Btn onClick={() => setDesignTransform([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])} disabled={isIdentity(transform)} testId="design-reset">
          Reset
        </Btn>
      </div>
      <PoseFields prefix="design" transform={transform} onChange={setDesignTransform} />
      {active && (
        <div className="row">
          {(['translate', 'rotate', 'scale'] as const).map((m) => (
            <Btn key={m} active={gizmoMode === m} onClick={() => setGizmoMode(m)} testId={`design-gizmo-${m}`} title={`Key ${m[0]}`}>
              {m}
            </Btn>
          ))}
        </div>
      )}
    </div>
  );
}
