import { useEffect, useMemo } from 'react';
import { fetchMissingMeshes, importDesignMesh, importRefMesh, loadProjectFile, reuploadRefMesh } from '../state/actions';
import { type RequiredMesh, designEntry, requiredMeshes } from '../state/derived';
import { useStore } from '../state/store';
import { Banner, Btn, Group, Props } from './controls';
import { DesignTransform } from './DesignTransform';
import { Sparkline } from './Sparkline';

export const MESH_ACCEPT = '.stl,.step,.stp,model/stl';
export const IS_MOCK = import.meta.env.VITE_MOCK === '1';

function fileList(f: File): FileList {
  const dt = new DataTransfer();
  dt.items.add(f);
  return dt.files;
}

/** Required meshes the browser does not hold: one row each with a re-upload input (ids are content hashes). */
export function MissingMeshes() {
  const project = useStore((s) => s.project);
  const meshes = useStore((s) => s.meshes);
  const meshMemo = useStore((s) => s.meshMemo);
  const required = useMemo(() => requiredMeshes({ project, meshes, meshMemo }), [project, meshes, meshMemo]);
  const missing = required.filter((m) => !m.loaded);
  if (missing.length === 0) return null;
  return <MissingList missing={missing} />;
}

function MissingList({ missing }: { missing: RequiredMesh[] }) {
  const memo = useStore((s) => s.meshMemo);
  const busy = useStore((s) => s.busy);
  const names = missing.map((m) => {
    const known = memo[m.id];
    return known ? `"${known.name}" (${known.n_faces} faces)` : m.role === 'design' ? 'the design mesh' : `"${m.label}"`;
  });
  return (
    <Banner kind="warn" testId="reupload-notice">
      <div className="stack">
        <span>
          The project's geometry is not loaded. Re-upload {names.join(', ')} to continue; loads and supports are kept (mesh ids are
          content hashes, so the same file brings every selection back).
        </span>
        <ul className="items required-meshes" data-testid="required-meshes">
          {missing.map((m) => (
            <li key={`${m.id}:${m.refId ?? ''}`} className="row wrap" data-testid="required-mesh" data-mesh-id={m.id}>
              <span className="grow ellipsis" title={m.id}>
                {m.role}: {m.label} <code>{m.id}</code>
              </span>
              <input
                type="file"
                accept={MESH_ACCEPT}
                aria-label={`re-upload ${m.label}`}
                data-testid="reupload-input"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (!f) return;
                  if (m.role === 'design') void importDesignMesh(f);
                  else if (m.refId) void reuploadRefMesh(m.refId, f);
                  e.target.value = '';
                }}
              />
            </li>
          ))}
        </ul>
        <div className="row">
          <Btn onClick={() => void fetchMissingMeshes()} disabled={busy !== null} testId="fetch-from-server" title="The server keeps uploaded meshes on disk; ask it for these ids">
            Fetch from server
          </Btn>
        </div>
      </div>
    </Banner>
  );
}

export function LoadedRun() {
  const loaded = useStore((s) => s.loadedRun);
  if (!loaded) return null;
  const { run, fileName, onServer } = loaded;
  const hist = run.history ?? [];
  const first = hist[0];
  const last = hist[hist.length - 1];
  return (
    <div className="loaded-run" data-testid="loaded-run">
      <dl className="kv">
        <dt>Run</dt>
        <dd>
          <code>{run.id}</code> <span className={`status status-${run.status}`}>{run.status}</span>
        </dd>
        <dt>From</dt>
        <dd className="ellipsis" title={fileName}>
          {fileName}
        </dd>
        <dt>Iterations</dt>
        <dd data-testid="loaded-run-iters">{hist.length}</dd>
        {first && last && (
          <>
            <dt>Compliance</dt>
            <dd>
              {first.compliance.toPrecision(4)} &rarr; {last.compliance.toPrecision(4)}
            </dd>
            <dt>Volume</dt>
            <dd>{last.volume.toFixed(3)}</dd>
          </>
        )}
        {run.stats && (
          <>
            <dt>Grid</dt>
            <dd>
              {run.stats.nx} x {run.stats.ny} x {run.stats.nz}
            </dd>
          </>
        )}
        {run.finished_at && (
          <>
            <dt>Finished</dt>
            <dd>{run.finished_at.replace('T', ' ').slice(0, 19)}</dd>
          </>
        )}
      </dl>
      {run.error && <p className="red">{run.error}</p>}
      <Sparkline history={hist} height={60} />
      {onServer === false && (
        <p className="amber" data-testid="loaded-run-gone">
          The connected server does not have this run: the history is shown, but result exports need a new run.
        </p>
      )}
      {onServer === true && (
        <p className="dim" data-testid="loaded-run-attached">
          This run is still on the server: its exports and result mesh are available in Results.
        </p>
      )}
    </div>
  );
}

/** Opens one of the app's hidden file inputs (they are mounted once, in FileInputs). */
export function pickFile(which: 'mesh' | 'project' | 'ref'): void {
  document.getElementById(`file-${which}`)?.click();
}

/** Mock backend only: serves examples/bracket.stl from the repo root. */
export async function loadExample(): Promise<void> {
  const res = await fetch('/mock/examples/bracket.stl');
  if (res.ok) takeFiles(fileList(new File([await res.arrayBuffer()], 'bracket.stl')));
}

function takeFiles(files: FileList | null | undefined): void {
  const f = files?.[0];
  if (!f) return;
  // a .json is a project, anything else a design mesh
  if (/\.json$/i.test(f.name)) void loadProjectFile(f);
  else void importDesignMesh(f);
}

/** The hidden inputs behind every Import / Open / Reference STL command (and the e2e file uploads). */
export function FileInputs() {
  return (
    <div className="visually-hidden">
      <input
        id="file-mesh"
        type="file"
        accept={MESH_ACCEPT}
        data-testid="mesh-file-input"
        tabIndex={-1}
        onChange={(e) => {
          takeFiles(e.target.files);
          e.target.value = ''; // allow re-selecting the same file
        }}
      />
      <input
        id="file-project"
        type="file"
        accept=".json,application/json"
        data-testid="project-file-input"
        tabIndex={-1}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void loadProjectFile(f);
          e.target.value = '';
        }}
      />
      <input
        id="file-ref"
        type="file"
        accept={MESH_ACCEPT}
        data-testid="ref-file-input"
        tabIndex={-1}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void importRefMesh(f);
          e.target.value = '';
        }}
      />
    </div>
  );
}

/** Dropping a file anywhere on the window imports it as the design mesh (or loads it as a project if it is .json). */
export function useWindowDrop(onOver: (over: boolean) => void): void {
  useEffect(() => {
    const over = (e: DragEvent) => {
      e.preventDefault();
      onOver(true);
    };
    const leave = (e: DragEvent) => {
      if (e.relatedTarget === null) onOver(false);
    };
    const drop = (e: DragEvent) => {
      e.preventDefault();
      onOver(false);
      takeFiles(e.dataTransfer?.files);
    };
    window.addEventListener('dragover', over);
    window.addEventListener('dragleave', leave);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragover', over);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('drop', drop);
    };
  }, [onOver]);
}

/** Design body page: what was imported and where it sits. */
export function PartPage() {
  const entry = useStore((s) => designEntry(s));
  const busy = useStore((s) => s.busy);
  const info = entry?.info;
  const size = info ? info.bbox[1]!.map((v, k) => v - info.bbox[0]![k]!) : null;

  return (
    <>
      <Group title="Design body" icon="part" testId="group-design">
        {!info && (
          <div className="stack">
            <p className="dim">No design body yet. Import an STL or STEP file, or drop one anywhere on the window.</p>
            <div className="row">
              <Btn variant="primary" icon="import" onClick={() => pickFile('mesh')} disabled={busy === 'Upload'} testId="choose-mesh">
                {busy === 'Upload' ? 'Uploading...' : 'Import STL / STEP'}
              </Btn>
            </div>
          </div>
        )}
        {info && (
          <Props testId="mesh-info">
            <dt>Name</dt>
            <dd className="ellipsis" title={info.name}>
              {info.name}
            </dd>
            <dt>Source</dt>
            <dd data-testid="mesh-source" data-source={info.source}>
              {info.source === 'step' ? 'STEP (tessellated)' : 'mesh'}
            </dd>
            {info.n_brep_faces != null && (
              <>
                <dt>B-rep faces</dt>
                <dd data-testid="mesh-brep-faces">{info.n_brep_faces.toLocaleString()}</dd>
              </>
            )}
            <dt>Faces</dt>
            <dd data-testid="mesh-faces">{info.n_faces.toLocaleString()}</dd>
            <dt>Vertices</dt>
            <dd>{info.n_vertices.toLocaleString()}</dd>
            {size && (
              <>
                <dt>Size</dt>
                <dd>{size.map((v) => v.toPrecision(4)).join(' \u00d7 ')}</dd>
              </>
            )}
            {info.volume != null && (
              <>
                <dt>Volume</dt>
                <dd>{info.volume.toPrecision(5)}</dd>
              </>
            )}
            <dt>Watertight</dt>
            <dd>{info.is_watertight ? 'yes' : 'no'}</dd>
          </Props>
        )}
        {info && !info.is_watertight && (
          <Banner kind="error" testId="watertight-warning">
            Mesh is not watertight. Voxelization may leak or fill incorrectly; repair the STL before optimizing.
          </Banner>
        )}
      </Group>
      {info && (
        <Group title="Placement" icon="gizmo">
          <DesignTransform />
        </Group>
      )}
    </>
  );
}
