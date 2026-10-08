import { IDENTITY } from '../state/defaults';
import { composeTRS, decomposeTRS } from '../state/transform';
import { NumField } from './controls';

const AXES = ['x', 'y', 'z'] as const;

/**
 * Position / rotation (degrees, XYZ) / scale of a column-major matrix as number fields. The gizmo writes the same
 * matrix, so fields and gizmo edit one value. test ids: `${prefix}-fields`, `${prefix}-pos-x`, `${prefix}-rot-x`, `${prefix}-scale-x`.
 */
export function PoseFields({ prefix, transform, onChange }: { prefix: string; transform: readonly number[] | undefined; onChange: (m: number[]) => void }) {
  const { pos, rot, scale } = decomposeTRS(transform?.length === 16 ? transform : IDENTITY);
  const set = (p = pos, ro = rot, sc = scale) => onChange(composeTRS(p, ro, sc));
  const put = (arr: number[], k: number, v: number) => arr.map((x, i) => (i === k ? v : x));
  return (
    <div className="prim-fields" data-testid={`${prefix}-fields`} onClick={(e) => e.stopPropagation()}>
      <div className="prim-row">
        <span>pos</span>
        {AXES.map((a, k) => (
          <NumField key={a} value={pos[k]!} testId={`${prefix}-pos-${a}`} onChange={(v) => set(put(pos, k, v))} />
        ))}
      </div>
      <div className="prim-row">
        <span>rot&deg;</span>
        {AXES.map((a, k) => (
          <NumField key={a} value={rot[k]!} step={5} testId={`${prefix}-rot-${a}`} onChange={(v) => set(pos, put(rot, k, v))} />
        ))}
      </div>
      <div className="prim-row">
        <span>scale</span>
        {AXES.map((a, k) => (
          <NumField key={a} value={scale[k]!} min={1e-6} testId={`${prefix}-scale-${a}`} onChange={(v) => set(pos, rot, put(scale, k, v))} />
        ))}
      </div>
    </div>
  );
}
