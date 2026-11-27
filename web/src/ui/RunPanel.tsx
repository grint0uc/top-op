import type { IterationRecord, ParamsSpec } from '../api/client';
import { startRun, stopRun } from '../state/actions';
import { designBox } from '../state/derived';
import { useStore } from '../state/store';
import { Btn, Field, Group, NumField, OptNumField } from './controls';
import { Icon } from './icons';
import { Sparkline, SERIES_COLORS } from './Sparkline';

type Axis = 'x' | 'y' | 'z';
const BUILD_DIRS = ['+x', '-x', '+y', '-y', '+z', '-z'] as const;

/** Centre of the design bbox along an axis: the plane a symmetry row means with "center" (and its starting value when made numeric). */
function centreAlong(axis: Axis): number {
  const box = designBox(useStore.getState());
  const a = 'xyz'.indexOf(axis);
  return box ? Math.round(((box.min[a]! + box.max[a]!) / 2) * 1e4) / 1e4 : 0;
}

function SymmetryEditor({ planes }: { planes: NonNullable<ParamsSpec['symmetry']> }) {
  const { setParams } = useStore.getState();
  const set = (next: typeof planes) => setParams({ symmetry: next });
  const edit = (i: number, patch: Partial<(typeof planes)[number]>) => set(planes.map((p, k) => (k === i ? { ...p, ...patch } : p)));
  return (
    <div className="stack" data-testid="symmetry-editor">
      <div className="row">
        <span className="field-label grow">Symmetry planes</span>
        <Btn icon="symmetry" onClick={() => set([...planes, { axis: 'x', position: null }])} testId="sym-add" title="Mirror the design across a plane (the optimizer averages mirrored cells)">
          Add plane
        </Btn>
      </div>
      {planes.map((p, i) => (
        <div className="row" key={i} data-testid="sym-row">
          <select
            className="select"
            value={p.axis}
            data-testid="sym-axis"
            title="The plane is perpendicular to this axis"
            onChange={(e) => {
              const axis = e.target.value as Axis;
              edit(i, { axis, position: p.position == null ? null : centreAlong(axis) });
            }}
          >
            {(['x', 'y', 'z'] as const).map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
          <label className="check" title="Mirror plane through the middle of the design">
            <input
              type="checkbox"
              checked={p.position == null}
              data-testid="sym-center"
              onChange={(e) => edit(i, { position: e.target.checked ? null : centreAlong(p.axis) })}
            />
            center
          </label>
          {p.position != null && (
            <NumField value={p.position} testId="sym-pos" title={`${p.axis} coordinate of the plane`} onChange={(v) => edit(i, { position: v })} />
          )}
          <span className="grow" />
          <Btn variant="danger" icon="trash" onClick={() => set(planes.filter((_, k) => k !== i))} testId="sym-remove" title="Remove this plane" />
        </div>
      ))}
    </div>
  );
}

export const lastNumber = (hist: readonly IterationRecord[], key: 'stress_max' | 'constraint'): number | null => {
  for (let i = hist.length - 1; i >= 0; i--) {
    const v = hist[i]![key];
    if (typeof v === 'number') return v;
  }
  return null;
};

export function StudyPage() {
  const params = useStore((s) => s.project.params);
  const { setParams } = useStore.getState();
  const stressOn = params.stress_limit != null;

  return (
    <>
      <Group title="Objective and constraints" icon="study" testId="constraints">
        <p className="dim">Minimize compliance (maximize stiffness) for a volume budget.</p>
        <div className="grid2">
          <Field label="Volume fraction">
            <NumField value={params.volfrac} min={0.01} max={0.99} step={0.01} testId="p-volfrac" onChange={(v) => setParams({ volfrac: v })} />
          </Field>
          <Field label="Optimizer" hint="A stress limit needs the MMA optimizer (two constraints); the server switches to it by itself">
            <select
              className="select"
              value={stressOn ? 'mma' : params.optimizer}
              disabled={stressOn}
              data-testid="p-optimizer"
              onChange={(e) => setParams({ optimizer: e.target.value as ParamsSpec['optimizer'] })}
            >
              <option value="oc">oc</option>
              <option value="mma">{stressOn ? 'mma (forced by stress limit)' : 'mma'}</option>
            </select>
          </Field>
          <Field label="Stress limit" hint="Von Mises limit in the units of E; empty = no stress constraint">
            <OptNumField value={params.stress_limit ?? null} placeholder="off" testId="p-stress-limit" onChange={(v) => setParams({ stress_limit: v })} />
          </Field>
          <Field
            label="Stress p-norm (final)"
            hint="Final exponent of the automatic p-continuation (starts at 8, doubles as the run settles; 4..256). The move limit is capped at 0.1/0.05 while the constraint is active"
          >
            <NumField value={params.stress_pnorm} min={4} max={256} step={1} testId="p-stress-pnorm" onChange={(v) => setParams({ stress_pnorm: v })} />
          </Field>
        </div>
      </Group>
      <Group title="Manufacturing" icon="overhang">
        <Field label="Overhang build direction" hint="Additive manufacturing: no overhangs steeper than 45 degrees when building along this direction">
          <select
            className="select"
            value={params.overhang ?? ''}
            data-testid="p-overhang"
            onChange={(e) => setParams({ overhang: e.target.value === '' ? null : (e.target.value as (typeof BUILD_DIRS)[number]) })}
          >
            <option value="">none</option>
            {BUILD_DIRS.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </Field>
        <SymmetryEditor planes={params.symmetry ?? []} />
      </Group>
      <Group title="SIMP and filter" icon="density">
        <div className="grid2">
          <Field label="Penalization p">
            <NumField value={params.penal} min={1} max={6} step={0.5} testId="p-penal" onChange={(v) => setParams({ penal: v })} />
          </Field>
          <Field label="Filter radius rmin" hint="In cells">
            <NumField value={params.rmin} min={1} step={0.5} testId="p-rmin" onChange={(v) => setParams({ rmin: v })} />
          </Field>
        </div>
        <div className="row wrap">
          <label className="check">
            <input type="checkbox" checked={params.heaviside} onChange={(e) => setParams({ heaviside: e.target.checked })} data-testid="p-heaviside" />
            Heaviside projection
          </label>
          <label className="check">
            <input type="checkbox" checked={params.continuation} onChange={(e) => setParams({ continuation: e.target.checked })} data-testid="p-continuation" />
            Continuation
          </label>
        </div>
      </Group>
      <Group title="Convergence" icon="results">
        <div className="grid2">
          <Field label="Max iterations">
            <NumField value={params.max_iter} min={1} max={2000} step={1} testId="p-max-iter" onChange={(v) => setParams({ max_iter: Math.round(v) })} />
          </Field>
          <Field label="Tolerance">
            <NumField value={params.tol} min={1e-6} step={0.001} testId="p-tol" onChange={(v) => setParams({ tol: v })} />
          </Field>
          <Field label="Move limit">
            <NumField value={params.move} min={0.01} max={1} step={0.05} testId="p-move" onChange={(v) => setParams({ move: v })} />
          </Field>
          <Field label="Density frame every" hint="Iterations between density frames streamed to the viewport">
            <NumField value={params.density_every} min={1} step={1} testId="p-density-every" onChange={(v) => setParams({ density_every: Math.round(v) })} />
          </Field>
        </div>
      </Group>
      <Group title="Solver" icon="voxelize" defaultOpen={false}>
        <div className="grid2">
          <Field label="Linear solver">
            <select className="select" value={params.solver} data-testid="p-solver" onChange={(e) => setParams({ solver: e.target.value as typeof params.solver })}>
              <option value="auto">auto</option>
              <option value="amg">amg</option>
              <option value="direct">direct</option>
            </select>
          </Field>
          <Field label="Precision">
            <select className="select" value={params.dtype} data-testid="p-dtype" onChange={(e) => setParams({ dtype: e.target.value as typeof params.dtype })}>
              <option value="float64">float64</option>
              <option value="float32">float32</option>
            </select>
          </Field>
        </div>
      </Group>
    </>
  );
}

export function MaterialPage() {
  const material = useStore((s) => s.project.material);
  const { setMaterial } = useStore.getState();
  return (
    <Group title="Isotropic linear elastic" icon="material">
      <div className="grid2">
        <Field label="Young's modulus E" hint="Units follow the mesh (e.g. MPa for mm and N)">
          <NumField value={material.E} min={1e-9} testId="m-e" onChange={(v) => setMaterial({ E: v })} />
        </Field>
        <Field label="Poisson's ratio ν">
          <NumField value={material.nu} min={0} max={0.499} step={0.01} testId="m-nu" onChange={(v) => setMaterial({ nu: v })} />
        </Field>
      </div>
      <p className="dim">Compliance and stress scale with E; the optimal layout does not.</p>
    </Group>
  );
}

/** Solver job monitor: always visible under the feature tree. Run / Stop, progress, convergence plot, constraint. */
export function JobMonitor() {
  const params = useStore((s) => s.project.params);
  const run = useStore((s) => s.run);
  const nLoads = useStore((s) => s.project.loads.length);
  const nSupports = useStore((s) => s.project.supports.length);
  const hasDesign = useStore((s) => !!s.project.design_mesh?.mesh_id);
  const active = run.status === 'queued' || run.status === 'running';
  const last = run.history[run.history.length - 1];
  const max = run.maxIter || params.max_iter;
  const stressMax = lastNumber(run.history, 'stress_max');
  const constraint = lastNumber(run.history, 'constraint');

  return (
    <section className="monitor" data-testid="panel-run" aria-label="Job monitor">
      <header className="monitor-head">
        <Icon name="results" size={14} />
        <span className="grow">Job monitor</span>
        <span className={`status status-${run.status}`} data-testid="run-status" title={run.status === 'queued' ? 'Waiting for the previous run to finish' : undefined}>
          {run.status}
        </span>
      </header>
      <div className="monitor-body">
        <div className="row">
          <Btn variant="primary" icon="run" onClick={() => void startRun()} disabled={active || !hasDesign} testId="run-start" title="Run the optimization">
            Run
          </Btn>
          <Btn variant="danger" icon="stop" onClick={() => void stopRun()} disabled={!active} testId="run-stop" title="Stop the running job">
            Stop
          </Btn>
          <span className="grow" />
          <span data-testid="run-iter" className="num-out">
            it {last?.it ?? 0} / {max}
          </span>
        </div>
        <progress className="meter" max={max} value={last?.it ?? 0} />
        {hasDesign && (nLoads === 0 || nSupports === 0) && (
          <p className="amber" data-testid="run-hint">
            {nLoads === 0 ? 'No load defined. ' : ''}
            {nSupports === 0 ? 'No support defined.' : ''}
          </p>
        )}
        <Sparkline history={run.history} height={64} />
        {last && (
          <div className="legend">
            <span style={{ color: SERIES_COLORS.compliance }}>C {last.compliance.toPrecision(5)}</span>
            <span style={{ color: SERIES_COLORS.volume }}>V {last.volume.toFixed(3)}</span>
            {stressMax !== null && (
              <span style={{ color: SERIES_COLORS.stress }} data-testid="run-stress">
                stress {stressMax.toPrecision(4)}
              </span>
            )}
          </div>
        )}
        {constraint !== null && (
          <div className="row" data-testid="run-constraint" data-ok={constraint <= 0}>
            <span className="dim grow">stress constraint g = {constraint.toPrecision(3)}</span>
            <span className={`tag ${constraint <= 0 ? 'tag-ok' : 'tag-bad'}`} data-testid="run-constraint-tag" title="g <= 0 means the stress limit is met">
              {constraint <= 0 ? 'satisfied' : 'violated'}
            </span>
          </div>
        )}
        {run.message && run.status !== 'idle' && (
          <p className="dim" data-testid="run-message">
            {run.message}
          </p>
        )}
        {run.error && (
          <p className="red" data-testid="run-error">
            {run.error}
          </p>
        )}
      </div>
    </section>
  );
}
