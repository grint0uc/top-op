import { useEffect } from 'react';
import type { VoxelStats } from '../api/client';
import { runVoxelize } from '../state/actions';
import { designEntry } from '../state/derived';
import { useStore } from '../state/store';
import { Banner, Btn, Field, Group, NumField, Slider } from './controls';

export const AMBER_ACTIVE = 150_000;
export const RED_ACTIVE = 300_000;

export const voxelLevel = (st: VoxelStats | null): 'ok' | 'amber' | 'red' =>
  !st ? 'ok' : st.n_active > RED_ACTIVE ? 'red' : st.n_active > AMBER_ACTIVE ? 'amber' : 'ok';

/** Debounced: every change to resolution/padding/ref placement/design pose re-voxelizes on the server. */
export function useAutoVoxelize(): void {
  const loaded = useStore((s) => designEntry(s)?.info.id ?? null);
  const grid = useStore((s) => s.project.grid);
  const refs = useStore((s) => s.project.ref_models);
  const designTransform = useStore((s) => s.project.design_mesh?.transform);
  useEffect(() => {
    if (!loaded) return;
    const t = setTimeout(() => void runVoxelize(), 400);
    return () => clearTimeout(t);
  }, [loaded, grid, refs, designTransform]);
}

export function DomainPage() {
  const loaded = useStore((s) => designEntry(s)?.info.id ?? null);
  const grid = useStore((s) => s.project.grid);
  const maxIter = useStore((s) => s.project.params.max_iter);
  const voxel = useStore((s) => s.voxel);
  const setGrid = useStore((s) => s.setGrid);

  const st = voxel.stats;
  const level = voxelLevel(st);
  const gb = st ? st.est_bytes / 1e9 : 0;

  return (
    <>
      <Group title="Resolution" icon="domain">
        <Slider
          label="Elements along longest side"
          value={grid.elements_along_longest}
          min={10}
          max={300}
          step={1}
          testId="domain-elements"
          onChange={(v) => setGrid({ elements_along_longest: v })}
        />
        <Field label="Padding (cells)" hint="Empty cells around the part on every side">
          <NumField value={grid.padding} min={0} max={10} step={1} testId="domain-padding" onChange={(v) => setGrid({ padding: Math.round(v) })} />
        </Field>
      </Group>
      <Group
        title="Voxel grid"
        icon="voxelize"
        aside={
          <Btn
            icon="voxelize"
            variant="ghost"
            onClick={() => void runVoxelize()}
            disabled={!loaded || voxel.loading}
            testId="voxelize-now"
            title="Voxelize again now"
          />
        }
      >
        {!loaded && <p className="dim">Import a design body to see the voxel grid.</p>}
        {voxel.loading && (
          <p className="dim" data-testid="voxel-loading">
            Voxelizing...
          </p>
        )}
        {voxel.error && <Banner kind="error">Voxelize failed: {voxel.error}</Banner>}
        {st && (
          <dl className={`kv voxel level-${level}`} data-testid="voxel-stats" data-level={level}>
            <dt>Grid</dt>
            <dd>
              {st.nx} &times; {st.ny} &times; {st.nz}
            </dd>
            <dt>Cell size h</dt>
            <dd>{st.h.toPrecision(4)}</dd>
            <dt>Active elements</dt>
            <dd>{st.n_active.toLocaleString()}</dd>
            <dt>Nodes</dt>
            <dd>{st.n_nodes.toLocaleString()}</dd>
            <dt>DOF</dt>
            <dd>{st.n_dof.toLocaleString()}</dd>
            <dt>Est. memory</dt>
            <dd>{gb.toFixed(2)} GB</dd>
            <dt>Est. time</dt>
            <dd>
              {st.est_sec_per_iter.toFixed(2)} s/iter
              <span className="dim">
                {' '}
                (~{formatDuration(st.est_sec_per_iter * maxIter)} for {maxIter})
              </span>
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
      </Group>
    </>
  );
}

export function formatDuration(sec: number): string {
  if (sec < 90) return `${Math.round(sec)} s`;
  if (sec < 5400) return `${Math.round(sec / 60)} min`;
  return `${(sec / 3600).toFixed(1)} h`;
}
