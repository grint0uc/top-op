import { useState } from 'react';
import { addLoadFromSelection, addPrimitive, addSupportFromSelection, resolvePreview } from '../state/actions';
import { hasSelection, selectionSummary } from '../state/derived';
import { type Prim, type PrimitiveKind, type ToolMode, useStore } from '../state/store';
import { composeTRS, decomposeTRS } from '../state/transform';
import { Btn, NumField, Slider } from './controls';
import { Icon, type IconName } from './icons';
import { QueryPanel } from './QueryPanel';

const MODES: { id: ToolMode; label: string; key: string; hint: string; icon: IconName }[] = [
  { id: 'orbit', label: 'Orbit', key: '1', hint: 'Rotate / pan / zoom', icon: 'orbit' },
  { id: 'pick', label: 'Pick', key: '2', hint: 'Click: face. Shift+click: grow flat face. Ctrl/Cmd+click: remove', icon: 'pick' },
  { id: 'paint', label: 'Paint', key: '3', hint: 'Drag to paint faces, Ctrl/Cmd-drag to erase. Right-drag orbits', icon: 'paint' },
  { id: 'gizmo', label: 'Gizmo', key: '4', hint: 'Move / rotate / scale the active primitive or reference model (g / r / s)', icon: 'gizmo' },
  { id: 'query', label: 'Query', key: '5', hint: 'Select by facet, surface normal or plane, as an agent would (no clicking)', icon: 'query' },
];

// A primitive's transform carries rotation + translation only (size is separate), so the scale part is dropped.
const decompose = (prim: Prim) => decomposeTRS(prim.transform);
const compose = (pos: number[], rot: number[]) => composeTRS(pos, rot);

/** Numeric fields bound two-way to the active primitive (the gizmo writes the same store field). */
function PrimitiveFields({ prim }: { prim: Prim }) {
  const setPrimitive = useStore((s) => s.setPrimitive);
  const { pos, rot } = decompose(prim);
  const axes = ['x', 'y', 'z'] as const;
  const sizeFields: { label: string; idx: number }[] =
    prim.kind === 'box'
      ? [
          { label: 'sx', idx: 0 },
          { label: 'sy', idx: 1 },
          { label: 'sz', idx: 2 },
        ]
      : prim.kind === 'sphere'
        ? [{ label: 'radius', idx: 0 }]
        : [
            { label: 'radius', idx: 0 },
            { label: 'height', idx: 1 },
          ];
  const setSize = (idx: number, v: number) => {
    const size = [...prim.size];
    if (prim.kind === 'sphere') size.fill(v);
    else if (prim.kind === 'cylinder' && idx === 0) {
      size[0] = v;
      size[2] = v;
    } else size[idx] = v;
    setPrimitive({ ...prim, size });
  };
  return (
    <div className="prim-fields" data-testid="prim-fields">
      <div className="prim-row">
        <span>pos</span>
        {axes.map((a, k) => (
          <NumField
            key={a}
            value={pos[k]!}
            label={`Region position ${a}`}
            testId={`prim-pos-${a}`}
            onChange={(v) =>
              setPrimitive({
                ...prim,
                transform: compose(
                  pos.map((x, i) => (i === k ? v : x)),
                  rot,
                ),
              })
            }
          />
        ))}
      </div>
      <div className="prim-row">
        <span>rot&deg;</span>
        {axes.map((a, k) => (
          <NumField
            key={a}
            value={rot[k]!}
            step={5}
            label={`Region rotation ${a} (degrees)`}
            testId={`prim-rot-${a}`}
            onChange={(v) =>
              setPrimitive({
                ...prim,
                transform: compose(
                  pos,
                  rot.map((x, i) => (i === k ? v : x)),
                ),
              })
            }
          />
        ))}
      </div>
      <div className="prim-row">
        <span>size</span>
        {sizeFields.map((f) => (
          <NumField
            key={f.label}
            value={prim.size[f.idx]!}
            min={1e-6}
            label={`Region size ${f.label}`}
            testId={`prim-size-${f.label}`}
            title={f.label}
            onChange={(v) => setSize(f.idx, v)}
          />
        ))}
      </div>
    </div>
  );
}

export function SelectionToolbar() {
  const tool = useStore((s) => s.tool);
  const grow = useStore((s) => s.growAngleDeg);
  const brush = useStore((s) => s.brushRadius);
  const gizmoMode = useStore((s) => s.gizmoMode);
  const prim = useStore((s) => s.selection.primitive);
  const summary = useStore((s) => selectionSummary(s));
  const hasSel = useStore((s) => hasSelection(s));
  const preview = useStore((s) => s.preview);
  const bcActive = useStore((s) => s.activeItem?.kind === 'load' || s.activeItem?.kind === 'support');
  const diag = useStore((s) => {
    const id = s.project.design_mesh?.mesh_id;
    const b = id ? s.meshes[id]?.data.bbox : undefined;
    return b ? Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) : 100;
  });
  const hasDesign = useStore((s) => !!(s.project.design_mesh?.mesh_id && s.meshes[s.project.design_mesh.mesh_id]));
  const { setTool, setGrowAngle, setBrushRadius, clearSelection, setGizmoMode, setPreview } = useStore.getState();
  const [folded, setFolded] = useState(false);
  // orbiting with nothing selected: just the header, so the panel hides as little of the part as possible
  const options = tool !== 'orbit' || !!prim;
  const actions = tool !== 'orbit' || hasSel || !!preview || bcActive;

  return (
    <div className={`selbar ${tool === 'query' ? 'wide' : ''} ${folded ? 'folded' : ''}`} data-testid="selection-toolbar">
      <div className="selbar-head">
        <button
          type="button"
          className="selbar-fold"
          onClick={() => setFolded(!folded)}
          aria-expanded={!folded}
          aria-label={folded ? 'Show the tool options' : 'Hide the tool options'}
          title={folded ? 'Show the tool options' : 'Hide the tool options'}
          data-testid="selection-fold"
          disabled={!options}
        >
          <Icon name="chevron" size={12} />
        </button>
        <div className="seg modes" role="toolbar" aria-label="Selection tool">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              className={`seg-btn icon ${tool === m.id ? 'on' : ''}`}
              aria-pressed={tool === m.id}
              onClick={() => setTool(m.id)}
              data-testid={`mode-${m.id}`}
              title={`${m.label}: ${m.hint} [${m.key}]`}
              aria-label={`${m.label} ${m.key}`}
              aria-keyshortcuts={m.key}
              disabled={!hasDesign && m.id !== 'orbit' && m.id !== 'gizmo'}
            >
              <Icon name={m.icon} size={15} />
              <kbd>{m.key}</kbd>
            </button>
          ))}
        </div>
        <span className="grow" />
        <div className="seg" role="toolbar" aria-label="Region primitive">
          {(['box', 'sphere', 'cylinder'] as PrimitiveKind[]).map((k) => (
            <button
              key={k}
              type="button"
              className="seg-btn icon"
              onClick={() => addPrimitive(k)}
              disabled={!hasDesign}
              data-testid={`add-${k}`}
              title={`Region primitive: ${k} (the grid nodes inside it, on the surface)`}
              aria-label={`Add a ${k} region`}
            >
              <Icon name={k} size={15} />
            </button>
          ))}
        </div>
      </div>
      {options && (
        <div className="selbar-body">
          {tool === 'query' && <QueryPanel />}
          {tool === 'pick' && (
            <Slider label="Grow angle" value={grow} min={1} max={60} step={1} format={(v) => `${v}°`} testId="grow-angle" onChange={setGrowAngle} />
          )}
          {tool === 'paint' && (
            <Slider
              label="Brush radius"
              value={brush}
              min={diag / 200}
              max={diag / 4}
              step={diag / 400}
              format={(v) => v.toPrecision(3)}
              testId="brush-radius"
              onChange={setBrushRadius}
            />
          )}
          {tool === 'gizmo' && (
            <div className="seg" role="group" aria-label="Gizmo mode">
              {(['translate', 'rotate', 'scale'] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  className={`seg-btn ${gizmoMode === m ? 'on' : ''}`}
                  aria-pressed={gizmoMode === m}
                  onClick={() => setGizmoMode(m)}
                  data-testid={`toolbar-gizmo-${m}`}
                  title={`Key ${m[0]}`}
                  aria-keyshortcuts={m[0]}
                >
                  {m} <kbd>{m[0]}</kbd>
                </button>
              ))}
            </div>
          )}
          {prim && <PrimitiveFields prim={prim} />}
        </div>
      )}
      {actions && (
        <div className="selbar-foot">
          <div className="toolbar-row">
            <span className={`sel-summary ${hasSel ? 'has' : ''}`} data-testid="selection-summary">
              {summary}
            </span>
            <Btn variant="ghost" icon="clear" onClick={clearSelection} disabled={!hasSel} testId="clear-selection" title="Clear the selection [Esc]" />
          </div>
          <div className="toolbar-row">
            <Btn
              variant="primary"
              icon="force"
              onClick={() => addLoadFromSelection()}
              disabled={!hasSel}
              testId="add-load"
              title="Create a load (force) from the selection"
            >
              Load
            </Btn>
            <Btn
              variant="primary"
              icon="support"
              onClick={() => addSupportFromSelection()}
              disabled={!hasSel}
              testId="add-support"
              title="Create a support (fixture) from the selection"
            >
              Support
            </Btn>
            <span className="grow" />
            <Btn
              icon="preview"
              onClick={() => void resolvePreview()}
              testId="resolve-preview"
              title="Resolve the selection (or the active load/support) to grid nodes on the server"
            >
              Nodes
            </Btn>
          </div>
          {preview && (
            <div className="toolbar-row preview-row">
              <span className="dim" data-testid="preview-count">
                {preview.count} nodes{preview.truncated ? '+' : ''}
              </span>
              <span className="dim ellipsis grow">{preview.source}</span>
              <Btn variant="ghost" icon="close" onClick={() => setPreview(null)} testId="clear-preview" title="Hide the resolved nodes" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
