import { useEffect } from 'react';
import { runVoxelize } from '../state/actions';
import { designEntry } from '../state/derived';
import { useStore } from '../state/store';
import { Banner, Field, NumField, Section, Slider } from './controls';

const AMBER_ACTIVE = 150_000;
const RED_ACTIVE = 300_000;

export function DomainPanel() {
  const loaded = useStore((s) => designEntry(s)?.info.id ?? null);
  const grid = useStore((s) => s.project.grid);
  const refs = useStore((s) => s.project.ref_models);
  const maxIter = useStore((s) => s.project.params.max_iter);
  const voxel = useStore((s) => s.voxel);
  const setGrid = useStore((s) => s.setGrid);

  // debounced: every change to resolution/padding/ref placement re-voxelizes on the server
  useEffect(() => {
    if (!loaded) return;
    const t = setTimeout(() => void runVoxelize(), 400);
    return () => clearTimeout(t);
  }, [loaded, grid, refs]);

  const st = voxel.stats;
  const level = !st ? 'ok' : st.n_active > RED_ACTIVE ? 'red' : st.n_active > AMBER_ACTIVE ? 'amber' : 'ok';
  const gb = st ? st.est_bytes / 1e9 : 0;

  return (
    <Section title="Domain">
      <Slider
        label="Elements along longest"
        value={grid.elements_along_longest}
        min={10}
        max={300}
        step={1}
        testId="domain-elements"
        onChange={(v) => setGrid({ elements_along_longest: v })}
      />
      <Field label="Padding (cells)">
        <NumField value={grid.padding} min={0} max={10} step={1} testId="domain-padding" onChange={(v) => setGrid({ padding: Math.round(v) })} />
      </Field>
      {!loaded && <p className="dim">Import a mesh to see the voxel grid.</p>}
      {voxel.loading && <p className="dim" data-testid="voxel-loading">Voxelizing...</p>}
      {voxel.error && <Banner kind="error">Voxelize failed: {voxel.error}</Banner>}
      {st && (
        <dl className={`kv voxel level-${level}`} data-testid="voxel-stats" data-level={level}>
          <dt>Grid</dt>
          <dd>
            {st.nx} x {st.ny} x {st.nz} (h = {st.h.toPrecision(4)})
          </dd>
          <dt>Active elements</dt>
          <dd data-testid="voxel-active">{st.n_active.toLocaleString()}</dd>
          <dt>DOF</dt>
          <dd>{st.n_dof.toLocaleString()}</dd>
          <dt>Est. memory</dt>
          <dd data-testid="voxel-memory">{gb.toFixed(2)} GB</dd>
          <dt>Est. time</dt>
          <dd>
            {st.est_sec_per_iter.toFixed(2)} s/iter
            <span className="dim"> (~{formatDuration(st.est_sec_per_iter * maxIter)} for {maxIter})</span>
          </dd>
          {(st.n_passive_solid > 0 || st.n_passive_void > 0) && (
            <>
              <dt>Passive</dt>
              <dd>
                {st.n_passive_solid} solid / {st.n_passive_void} void
              </dd>
            </>
          )}
        </dl>
      )}
      {st && level !== 'ok' && (
        <Banner kind={level === 'red' ? 'error' : 'warn'} testId="voxel-warning">
          {level === 'red'
            ? `Over ${RED_ACTIVE.toLocaleString()} active elements: this will likely exhaust memory or take hours. Reduce the resolution.`
            : `Over ${AMBER_ACTIVE.toLocaleString()} active elements: iterations will be slow.`}
        </Banner>
      )}
      {st?.warnings?.map((w) => (
        <p className="dim" key={w}>
          Server: {w}
        </p>
      ))}
    </Section>
  );
}

function formatDuration(sec: number): string {
  if (sec < 90) return `${Math.round(sec)} s`;
  if (sec < 5400) return `${Math.round(sec / 60)} min`;
  return `${(sec / 3600).toFixed(1)} h`;
}
