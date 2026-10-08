import { addPrimitive, resolvePreview } from '../state/actions';
import { hasSelection, selectionSummary } from '../state/derived';
import { type Prim, type PrimitiveKind, type ToolMode, useStore } from '../state/store';
import { composeTRS, decomposeTRS } from '../state/transform';
import { Btn, NumField, Slider } from './controls';
import { QueryPanel } from './QueryPanel';

const MODES: { id: ToolMode; label: string; key: string; hint: string }[] = [
  { id: 'orbit', label: 'Orbit', key: '1', hint: 'Rotate / pan / zoom' },
  { id: 'pick', label: 'Pick', key: '2', hint: 'Click: face. Shift+click: grow flat face. Ctrl/Cmd+click: remove' },
  { id: 'paint', label: 'Paint', key: '3', hint: 'Drag to paint faces, Ctrl/Cmd-drag to erase. Right-drag orbits' },
  { id: 'gizmo', label: 'Gizmo', key: '4', hint: 'Move / rotate / scale the active primitive or reference model (g / r / s)' },
  { id: 'query', label: 'Query', key: '5', hint: 'Select by facet, surface normal or plane, as an agent would (no clicking)' },
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
            testId={`prim-pos-${a}`}
            onChange={(v) => setPrimitive({ ...prim, transform: compose(pos.map((x, i) => (i === k ? v : x)), rot) })}
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
            testId={`prim-rot-${a}`}
            onChange={(v) => setPrimitive({ ...prim, transform: compose(pos, rot.map((x, i) => (i === k ? v : x))) })}
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
  const diag = useStore((s) => {
    const id = s.project.design_mesh?.mesh_id;
    const b = id ? s.meshes[id]?.data.bbox : undefined;
    return b ? Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) : 100;
  });
  const hasDesign = useStore((s) => !!(s.project.design_mesh?.mesh_id && s.meshes[s.project.design_mesh.mesh_id]));
  const { setTool, setGrowAngle, setBrushRadius, clearSelection, setGizmoMode, setPreview } = useStore.getState();

  return (
    <div className="toolbar" data-testid="selection-toolbar">
      <div className="toolbar-row">
        {MODES.map((m) => (
          <Btn key={m.id} active={tool === m.id} onClick={() => setTool(m.id)} testId={`mode-${m.id}`} title={`${m.hint} [${m.key}]`} disabled={!hasDesign && m.id !== 'orbit' && m.id !== 'gizmo'}>
            {m.label} <kbd>{m.key}</kbd>
          </Btn>
        ))}
      </div>
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
        <div className="toolbar-row">
          {(['translate', 'rotate', 'scale'] as const).map((m) => (
            <Btn key={m} active={gizmoMode === m} onClick={() => setGizmoMode(m)} testId={`toolbar-gizmo-${m}`} title={`Key ${m[0]}`}>
              {m} <kbd>{m[0]}</kbd>
            </Btn>
          ))}
        </div>
      )}
      <div className="toolbar-row">
        <span className="dim">Primitive</span>
        {(['box', 'sphere', 'cylinder'] as PrimitiveKind[]).map((k) => (
          <Btn key={k} onClick={() => addPrimitive(k)} disabled={!hasDesign} testId={`add-${k}`}>
            + {k}
          </Btn>
        ))}
      </div>
      {prim && <PrimitiveFields prim={prim} />}
      <div className="toolbar-row">
        <span className="dim" data-testid="selection-summary">
          {summary}
        </span>
        <Btn onClick={clearSelection} disabled={!hasSel} testId="clear-selection" title="Esc">
          Clear
        </Btn>
        <Btn onClick={() => void resolvePreview()} testId="resolve-preview" title="Resolve the selection (or the active load/support) to grid nodes on the server">
          Resolve preview
        </Btn>
        {preview && (
          <>
            <span className="dim" data-testid="preview-count">
              {preview.count} nodes{preview.truncated ? '+' : ''}
            </span>
            <Btn onClick={() => setPreview(null)} testId="clear-preview">
              &times;
            </Btn>
          </>
        )}
      </div>
    </div>
  );
}
