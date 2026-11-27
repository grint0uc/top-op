import type { RefModel } from '../api/client';
import { addRefPrimitive, reuploadRefMesh } from '../state/actions';
import { cssHex, designEntry } from '../state/derived';
import { useStore } from '../state/store';
import { REF_COLORS } from '../viewport/Viewport';
import { Btn, Field, Group } from './controls';
import { Icon } from './icons';
import { pickFile } from './ImportPanel';
import { PoseFields } from './PoseFields';

export const refColor = (r: RefModel): string => cssHex(REF_COLORS[r.mode]);
export const modeLabel = (r: RefModel): string => (r.mode === 'keep_in' ? 'keep-in' : 'keep-out');

/** Reference bodies group: commands to add one, and the list. */
export function RefsPage() {
  const refs = useStore((s) => s.project.ref_models);
  const designLoaded = useStore((s) => designEntry(s) !== null);
  const { setActiveItem, setTool } = useStore.getState();

  return (
    <>
      <Group title="Add reference body" icon="ref">
        <p className="dim">Keep-out: material is forbidden inside. Keep-in: material is forced. Placed with the gizmo or the pose fields.</p>
        <div className="row wrap">
          <Btn
            icon="import"
            onClick={() => pickFile('ref')}
            disabled={!designLoaded}
            testId="add-ref"
            title="Keep-in / keep-out volume from an STL, placed with the gizmo"
          >
            Reference STL
          </Btn>
        </div>
        <div className="row wrap">
          <span className="field-label">Keep-out</span>
          {(['box', 'sphere', 'cylinder'] as const).map((k) => (
            <Btn
              key={k}
              icon={k}
              onClick={() => void addRefPrimitive(k)}
              disabled={!designLoaded}
              testId={`add-ref-${k}`}
              title={`Generated unit ${k}, sized by the scale fields / gizmo`}
            >
              {k}
            </Btn>
          ))}
        </div>
      </Group>
      <Group title="Reference bodies" icon="list" aside={<span className="count">{refs.length}</span>}>
        {refs.length === 0 && <p className="dim">None yet.</p>}
        <ul className="pm-list">
          {refs.map((r) => (
            <li key={r.id}>
              <button
                type="button"
                className="pm-list-row"
                onClick={() => {
                  setActiveItem({ kind: 'ref', id: r.id });
                  setTool('gizmo');
                }}
              >
                <span className="swatch square" style={{ background: refColor(r) }} />
                <span className="grow ellipsis">{r.name || r.id}</span>
                <span className="dim">{modeLabel(r)}</span>
              </button>
            </li>
          ))}
        </ul>
      </Group>
    </>
  );
}

/** One reference body: mode, pose (the same matrix the gizmo writes) and gizmo modes. */
export function RefPage({ r }: { r: RefModel }) {
  const meshes = useStore((s) => s.meshes);
  const gizmoMode = useStore((s) => s.gizmoMode);
  const { updateRef, removeRef, setGizmoMode } = useStore.getState();
  const loaded = !!(r.mesh_id && meshes[r.mesh_id]);

  return (
    <>
      <Group title="Reference body" icon={r.mode === 'keep_in' ? 'keepin' : 'keepout'}>
        <Field label="Name">
          <input type="text" className="text" value={r.name} data-testid="ref-name" onChange={(e) => updateRef(r.id, { name: e.target.value })} />
        </Field>
        <div className="row">
          <Field label="Type">
            <select
              className="select"
              value={r.mode}
              data-testid="ref-mode"
              onChange={(e) => updateRef(r.id, { mode: e.target.value as 'keep_in' | 'keep_out' })}
            >
              <option value="keep_out">keep out</option>
              <option value="keep_in">keep in</option>
            </select>
          </Field>
          <span className="swatch square big" style={{ background: refColor(r) }} />
        </div>
        {!loaded && (
          <div className="stack" data-testid="ref-missing">
            <p className="amber">Geometry missing after reload ({r.mesh_id}): re-upload the same file.</p>
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
      </Group>
      {loaded && (
        <Group title="Placement" icon="gizmo">
          <PoseFields prefix="ref" transform={r.transform} onChange={(transform) => updateRef(r.id, { transform })} />
          <div className="seg" role="group" aria-label="Gizmo mode">
            {(['translate', 'rotate', 'scale'] as const).map((m) => (
              <button
                key={m}
                type="button"
                className={`seg-btn ${gizmoMode === m ? 'on' : ''}`}
                aria-pressed={gizmoMode === m}
                onClick={() => setGizmoMode(m)}
                data-testid={`gizmo-${m}`}
                title={`Key ${m[0]}`}
              >
                {m}
              </button>
            ))}
          </div>
        </Group>
      )}
      <div className="pm-actions">
        <Btn onClick={() => removeRef(r.id)} variant="danger" icon="trash" testId="ref-delete">
          Delete reference body
        </Btn>
      </div>
    </>
  );
}

export function RefIcon({ r }: { r: RefModel }) {
  return (
    <span style={{ color: refColor(r) }} className="tree-glyph">
      <Icon name={r.mode === 'keep_in' ? 'keepin' : 'keepout'} size={14} />
    </span>
  );
}
