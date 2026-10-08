import { useEffect, useRef, useState } from 'react';
import { importDesignMesh } from '../state/actions';
import { designEntry } from '../state/derived';
import { useStore } from '../state/store';
import { Banner, Btn, Section } from './controls';

export const IS_MOCK = import.meta.env.VITE_MOCK === '1';

function fileList(f: File): FileList {
  const dt = new DataTransfer();
  dt.items.add(f);
  return dt.files;
}

export function ImportPanel() {
  const entry = useStore((s) => designEntry(s));
  const designId = useStore((s) => s.project.design_mesh?.mesh_id ?? null);
  const memo = useStore((s) => (designId ? s.meshMemo[designId] : undefined));
  const name = useStore((s) => s.project.name);
  const busy = useStore((s) => s.busy);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const take = (files: FileList | null | undefined) => {
    const f = files?.[0];
    if (f) void importDesignMesh(f);
  };

  // dropping a file anywhere on the window imports it as the design mesh
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
          accept=".stl,model/stl"
          data-testid="mesh-file-input"
          className="visually-hidden"
          onChange={(e) => {
            take(e.target.files);
            e.target.value = ''; // allow re-selecting the same file
          }}
        />
        <Btn variant="primary" onClick={() => input.current?.click()} disabled={busy === 'Upload'} testId="choose-mesh">
          {busy === 'Upload' ? 'Uploading...' : 'Choose STL'}
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

      {designId && !entry && (
        <Banner kind="warn" testId="reupload-notice">
          Project restored without its geometry. Re-upload {memo ? `"${memo.name}" (${memo.n_faces} faces)` : 'the design mesh'} to
          continue; loads and supports are kept.
        </Banner>
      )}

      {info && (
        <dl className="kv" data-testid="mesh-info">
          <dt>Name</dt>
          <dd>{info.name}</dd>
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
      <div className="row">
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
    </Section>
  );
}
