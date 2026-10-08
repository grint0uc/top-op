import { startRun, stopRun } from '../state/actions';
import { useStore } from '../state/store';
import { Btn, Field, NumField, Section } from './controls';
import { Sparkline } from './Sparkline';

export function RunPanel() {
  const params = useStore((s) => s.project.params);
  const material = useStore((s) => s.project.material);
  const run = useStore((s) => s.run);
  const nLoads = useStore((s) => s.project.loads.length);
  const nSupports = useStore((s) => s.project.supports.length);
  const hasDesign = useStore((s) => !!s.project.design_mesh?.mesh_id);
  const { setParams, setMaterial } = useStore.getState();
  const active = run.status === 'queued' || run.status === 'running';
  const last = run.history[run.history.length - 1];
  const max = run.maxIter || params.max_iter;

  return (
    <Section title="Run">
      <div className="grid2">
        <Field label="Volume frac">
          <NumField value={params.volfrac} min={0.01} max={0.99} step={0.01} testId="p-volfrac" onChange={(v) => setParams({ volfrac: v })} />
        </Field>
        <Field label="Penal p">
          <NumField value={params.penal} min={1} max={6} step={0.5} testId="p-penal" onChange={(v) => setParams({ penal: v })} />
        </Field>
        <Field label="Filter rmin">
          <NumField value={params.rmin} min={1} step={0.5} testId="p-rmin" onChange={(v) => setParams({ rmin: v })} />
        </Field>
        <Field label="Max iter">
          <NumField value={params.max_iter} min={1} max={2000} step={1} testId="p-max-iter" onChange={(v) => setParams({ max_iter: Math.round(v) })} />
        </Field>
        <Field label="Tol">
          <NumField value={params.tol} min={1e-6} step={0.001} testId="p-tol" onChange={(v) => setParams({ tol: v })} />
        </Field>
        <Field label="Move">
          <NumField value={params.move} min={0.01} max={1} step={0.05} testId="p-move" onChange={(v) => setParams({ move: v })} />
        </Field>
        <Field label="Frame every">
          <NumField value={params.density_every} min={1} step={1} testId="p-density-every" onChange={(v) => setParams({ density_every: Math.round(v) })} />
        </Field>
        <Field label="Solver">
          <select className="select" value={params.solver} onChange={(e) => setParams({ solver: e.target.value as typeof params.solver })}>
            <option value="auto">auto</option>
            <option value="amg">amg</option>
            <option value="direct">direct</option>
          </select>
        </Field>
        <Field label="Young E">
          <NumField value={material.E} min={1e-9} testId="m-e" onChange={(v) => setMaterial({ E: v })} />
        </Field>
        <Field label="Poisson nu">
          <NumField value={material.nu} min={0} max={0.499} step={0.01} testId="m-nu" onChange={(v) => setMaterial({ nu: v })} />
        </Field>
      </div>
      <div className="row">
        <label className="check">
          <input type="checkbox" checked={params.heaviside} onChange={(e) => setParams({ heaviside: e.target.checked })} />
          Heaviside
        </label>
        <label className="check">
          <input type="checkbox" checked={params.continuation} onChange={(e) => setParams({ continuation: e.target.checked })} />
          Continuation
        </label>
        <select className="select" value={params.dtype} onChange={(e) => setParams({ dtype: e.target.value as typeof params.dtype })}>
          <option value="float64">float64</option>
          <option value="float32">float32</option>
        </select>
      </div>

      {hasDesign && (nLoads === 0 || nSupports === 0) && (
        <p className="amber" data-testid="run-hint">
          {nLoads === 0 ? 'No load defined. ' : ''}
          {nSupports === 0 ? 'No support defined.' : ''}
        </p>
      )}

      <div className="row">
        <Btn variant="primary" onClick={() => void startRun()} disabled={active || !hasDesign} testId="run-start">
          Start
        </Btn>
        <Btn variant="danger" onClick={() => void stopRun()} disabled={!active} testId="run-stop">
          Stop
        </Btn>
        <span className={`status status-${run.status}`} data-testid="run-status" title={run.status === 'queued' ? 'Waiting for the previous run to finish' : undefined}>
          {run.status}
        </span>
      </div>
      <div className="row">
        <span data-testid="run-iter">
          it {last?.it ?? 0} / {max}
        </span>
        <progress className="grow" max={max} value={last?.it ?? 0} />
      </div>
      {run.message && run.status !== 'idle' && (
        <p className="dim" data-testid="run-message">
          {run.message}
        </p>
      )}
      <Sparkline history={run.history} />
      <div className="legend">
        <span style={{ color: '#4f9cf9' }}>compliance {last ? last.compliance.toPrecision(5) : '-'}</span>
        <span style={{ color: '#ffa534' }}>volume {last ? last.volume.toFixed(3) : '-'}</span>
      </div>
      {run.error && (
        <p className="red" data-testid="run-error">
          {run.error}
        </p>
      )}
    </Section>
  );
}
