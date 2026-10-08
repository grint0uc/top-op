import { useRef } from 'react';
import type { RefModel } from '../api/client';
import { addRefPrimitive, importRefMesh, reuploadRefMesh } from '../state/actions';
import { designEntry } from '../state/derived';
import { useStore } from '../state/store';
import { Btn, Section } from './controls';
import { PoseFields } from './PoseFields';
import { MESH_ACCEPT } from './ImportPanel';

/** Pose of a reference model: `transform` is the full column-major matrix, the same value the gizmo writes. */
function RefFields({ r }: { r: RefModel }) {
  const updateRef = useStore((s) => s.updateRef);
  return <PoseFields prefix="ref" transform={r.transform} onChange={(transform) => updateRef(r.id, { transform })} />;
}

export function RefModelsPanel() {
  const refs = useStore((s) => s.project.ref_models);
  const meshes = useStore((s) => s.meshes);
  const active = useStore((s) => s.activeItem);
  const gizmoMode = useStore((s) => s.gizmoMode);
  const designLoaded = useStore((s) => designEntry(s) !== null);
  const { updateRef, removeRef, setActiveItem, setTool, setGizmoMode } = useStore.getState();
  const add = useRef<HTMLInputElement>(null);

  return (
    <Section title="Reference models">
      <input
        ref={add}
        type="file"
        accept={MESH_ACCEPT}
        className="visually-hidden"
        data-testid="ref-file-input"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void importRefMesh(f);
          e.target.value = '';
        }}
      />
      <div className="row wrap">
        <Btn onClick={() => add.current?.click()} disabled={!designLoaded} testId="add-ref" title="Keep-in / keep-out volumes, placed with the gizmo">
          Add reference STL
        </Btn>
      </div>
      <div className="row wrap">
        <span className="dim">Keep-out</span>
        {(['box', 'sphere', 'cylinder'] as const).map((k) => (
          <Btn key={k} onClick={() => void addRefPrimitive(k)} disabled={!designLoaded} testId={`add-ref-${k}`} title={`Generated unit ${k}, sized by the scale fields / gizmo`}>
            + {k}
          </Btn>
        ))}
      </div>
      {refs.length === 0 && <p className="dim">Keep-out: material is forbidden inside. Keep-in: material is forced.</p>}
      <ul className="items">
        {refs.map((r) => {
          const loaded = !!(r.mesh_id && meshes[r.mesh_id]);
          const isActive = active?.kind === 'ref' && active.id === r.id;
          return (
            <li
              key={r.id}
              className={`item ${isActive ? 'active' : ''}`}
              data-testid="ref-row"
              onClick={() => {
                setActiveItem({ kind: 'ref', id: r.id });
                setTool('gizmo');
              }}
            >
              <div className="row">
                <span className="grow ellipsis" title={r.name}>
                  {r.name || r.id}
                </span>
                <label className="check" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={r.visible}
                    data-testid="ref-visible"
                    onChange={(e) => updateRef(r.id, { visible: e.target.checked })}
                  />
                  show
                </label>
              </div>
              <div className="row" onClick={(e) => e.stopPropagation()}>
                <select
                  className="select"
                  value={r.mode}
                  data-testid="ref-mode"
                  onChange={(e) => updateRef(r.id, { mode: e.target.value as 'keep_in' | 'keep_out' })}
                >
                  <option value="keep_out">keep out</option>
                  <option value="keep_in">keep in</option>
                </select>
                <Btn onClick={() => removeRef(r.id)} variant="danger" testId="ref-delete">
                  Delete
                </Btn>
              </div>
              {!loaded && (
                <div className="row wrap" onClick={(e) => e.stopPropagation()} data-testid="ref-missing">
                  <span className="dim">Geometry missing after reload ({r.mesh_id}): re-upload the same file.</span>
                  <input
                    type="file"
                    accept=".stl"
                    data-testid="ref-reupload-input"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) void reuploadRefMesh(r.id, f);
                    }}
                  />
                </div>
              )}
              {isActive && loaded && <RefFields r={r} />}
              {isActive && (
                <div className="row" onClick={(e) => e.stopPropagation()}>
                  {(['translate', 'rotate', 'scale'] as const).map((m) => (
                    <Btn key={m} active={gizmoMode === m} onClick={() => setGizmoMode(m)} testId={`gizmo-${m}`} title={`Key ${m[0]}`}>
                      {m}
                    </Btn>
                  ))}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </Section>
  );
}
