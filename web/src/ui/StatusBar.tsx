// Bottom status bar: busy state, units, grid, element count and memory estimate, run status, iteration, compliance.
import { useStore } from '../state/store';
import { voxelLevel } from './DomainPanel';
import { Icon } from './icons';

export function StatusBar() {
  const busy = useStore((s) => s.busy);
  const st = useStore((s) => s.voxel.stats);
  const voxLoading = useStore((s) => s.voxel.loading);
  const run = useStore((s) => s.run);
  const maxIter = useStore((s) => s.project.params.max_iter);
  const tool = useStore((s) => s.tool);
  const last = run.history[run.history.length - 1];
  const level = voxelLevel(st);
  const max = run.maxIter || maxIter;

  return (
    <footer className="statusbar" data-testid="status-bar">
      <span className="sb-item sb-state">
        {busy ? (
          <span className="busy" data-testid="busy">
            <span className="spinner" />
            {busy}...
          </span>
        ) : (
          <span className="dim">Ready</span>
        )}
      </span>
      <span className="sb-item" title="Units are not converted: forces, E and compliance are in the mesh's units (e.g. mm, N, MPa)" data-testid="sb-units">
        units: as mesh
      </span>
      <span className="sb-item" title="Active tool">
        <Icon name={tool === 'orbit' ? 'orbit' : tool === 'pick' ? 'pick' : tool === 'paint' ? 'paint' : tool === 'gizmo' ? 'gizmo' : 'query'} size={12} />
        {tool}
      </span>
      {st && (
        <>
          <span className="sb-item" title="Voxel grid">
            <Icon name="domain" size={12} />
            {st.nx}&times;{st.ny}&times;{st.nz}
          </span>
          <span className={`sb-item level-${level}`} title="Active elements">
            Elements <b data-testid="voxel-active">{st.n_active.toLocaleString()}</b>
          </span>
          <span className="sb-item" title="Degrees of freedom">
            DOF <b>{st.n_dof.toLocaleString()}</b>
          </span>
          <span className={`sb-item level-${level}`} title="Estimated solver memory">
            Mem <b data-testid="voxel-memory">{(st.est_bytes / 1e9).toFixed(2)} GB</b>
          </span>
        </>
      )}
      {voxLoading && <span className="sb-item dim">voxelizing...</span>}
      <span className="grow" />
      <span className="sb-item">
        <span className={`dot status-${run.status}`} />
        <span data-testid="sb-status">{run.status}</span>
      </span>
      <span className="sb-item" data-testid="sb-iter">
        it <b>{last?.it ?? 0}</b> / {max}
      </span>
      <span className="sb-item" title="Compliance (the objective)" data-testid="sb-compliance">
        C <b>{last ? last.compliance.toPrecision(5) : '–'}</b>
      </span>
      <span className="sb-item" title="Volume fraction">
        V <b>{last ? last.volume.toFixed(3) : '–'}</b>
      </span>
    </footer>
  );
}
