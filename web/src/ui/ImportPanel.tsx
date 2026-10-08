import { useEffect, useMemo, useRef, useState } from 'react';
import { exportProjectJson, fetchMissingMeshes, importDesignMesh, loadProjectFile, reuploadRefMesh } from '../state/actions';
import { type RequiredMesh, designEntry, requiredMeshes } from '../state/derived';
import { useStore } from '../state/store';
import { Banner, Btn, Section } from './controls';
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
function MissingMeshes({ missing }: { missing: RequiredMesh[] }) {
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

function LoadedRun() {
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

export function ImportPanel() {
  const entry = useStore((s) => designEntry(s));
  const project = useStore((s) => s.project);
  const meshes = useStore((s) => s.meshes);
  const meshMemo = useStore((s) => s.meshMemo);
  const required = useMemo(() => requiredMeshes({ project, meshes, meshMemo }), [project, meshes, meshMemo]);
  const name = useStore((s) => s.project.name);
  const busy = useStore((s) => s.busy);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const projectInput = useRef<HTMLInputElement>(null);

  const take = (files: FileList | null | undefined) => {
    const f = files?.[0];
    if (!f) return;
    // a .json dropped on the window is a project, anything else a design mesh
    if (/\.json$/i.test(f.name)) void loadProjectFile(f);
    else void importDesignMesh(f);
  };

  // dropping a file anywhere on the window imports it as the design mesh (or loads it as a project if it is .json)
  useEffect(() => {
    const over = (e: DragEvent) => e.preventDefault();
    const drop = (e: DragEvent) => {
      e.preventDefault();
      setOver(false);
      take(e.dataTransfer?.files);
    };
    window.addEventListener('dragover', over);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragover', over);
      window.removeEventListener('drop', drop);
    };
  }, []);

  const info = entry?.info;
  const size = info ? info.bbox[1]!.map((v, k) => v - info.bbox[0]![k]!) : null;
  const missing = required.filter((m) => !m.loaded);

  return (
    <Section title="Import">
      <div
        className={`dropzone ${over ? 'over' : ''}`}
        onDragEnter={() => setOver(true)}
        onDragLeave={() => setOver(false)}
        onDrop={() => setOver(false)}
      >
        <input
          ref={input}
          type="file"
          accept={MESH_ACCEPT}
          data-testid="mesh-file-input"
          className="visually-hidden"
          onChange={(e) => {
            take(e.target.files);
            e.target.value = ''; // allow re-selecting the same file
          }}
        />
        <Btn variant="primary" onClick={() => input.current?.click()} disabled={busy === 'Upload'} testId="choose-mesh">
          {busy === 'Upload' ? 'Uploading...' : 'Choose STL / STEP'}
        </Btn>
        <span className="dim">or drop a file anywhere</span>
      </div>
      {IS_MOCK && (
        <div className="row">
          <Btn
            onClick={async () => {
              const res = await fetch('/mock/examples/bracket.stl');
              if (res.ok) take(fileList(new File([await res.arrayBuffer()], 'bracket.stl')));
            }}
            testId="load-example"
            title="Mock backend only: serves examples/bracket.stl from the repo root"
          >
            Load example bracket
          </Btn>
        </div>
      )}

      {missing.length > 0 && <MissingMeshes missing={missing} />}

      {info && (
        <dl className="kv" data-testid="mesh-info">
          <dt>Name</dt>
          <dd>{info.name}</dd>
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
              <dd>{size.map((v) => v.toPrecision(4)).join(' x ')}</dd>
            </>
          )}
          {info.volume != null && (
            <>
              <dt>Volume</dt>
              <dd>{info.volume.toPrecision(5)}</dd>
            </>
          )}
        </dl>
      )}
      <DesignTransform />
      {info && !info.is_watertight && (
        <Banner kind="error" testId="watertight-warning">
          Mesh is not watertight. Voxelization may leak or fill incorrectly; repair the STL before optimizing.
        </Banner>
      )}

      <label className="field">
        <span className="field-label">Project</span>
        <input
          type="text"
          className="text"
          value={name}
          data-testid="project-name"
          onChange={(e) => useStore.getState().setProjectName(e.target.value)}
        />
      </label>
      <input
        ref={projectInput}
        type="file"
        accept=".json,application/json"
        className="visually-hidden"
        data-testid="project-file-input"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void loadProjectFile(f);
          e.target.value = '';
        }}
      />
      <div className="row wrap">
        <Btn
          onClick={() => projectInput.current?.click()}
          disabled={busy === 'Load project'}
          testId="load-project"
          title="A project.json from Results (run export) or a saved project: restores loads, supports, parameters and the run history"
        >
          Load project.json
        </Btn>
        <Btn onClick={exportProjectJson} disabled={!info} testId="export-project" title="Save the current setup as a project.json (no run)">
          Save project
        </Btn>
        <Btn
          onClick={() => {
            useStore.getState().resetProject();
          }}
          testId="new-project"
          title="Discard the current project"
        >
          New project
        </Btn>
        {IS_MOCK && <span className="badge" title="Served by web/mock, not the Python server">MOCK API</span>}
      </div>
      <LoadedRun />
    </Section>
  );
}
