import type { MouseEvent } from 'react';
import { api } from '../api/client';
import { downloadExport, hideResultMesh, loadResultMesh, resultOptions, setColorByStress, setTrimToCad } from '../state/actions';
import { cssHex } from '../state/derived';
import { useStore } from '../state/store';
import { INFERNO } from '../viewport/DensityView';
import { Btn, Field, Group, NumField, Slider } from './controls';
import { Icon } from './icons';
import { LoadedRun } from './ImportPanel';

const STRESS_RAMP = `linear-gradient(to right, ${INFERNO.map(cssHex).join(', ')})`;

export function ResultsPage() {
  const runId = useStore((s) => s.run.id);
  const status = useStore((s) => s.run.status);
  const threshold = useStore((s) => s.threshold);
  const smooth = useStore((s) => s.smoothIterations);
  const trim = useStore((s) => s.trimToCad);
  const ghost = useStore((s) => s.ghostDesign);
  const info = useStore((s) => s.densityInfo);
  const hasResult = useStore((s) => s.resultStl !== null);
  const warnings = useStore((s) => s.resultWarnings);
  const exportNote = useStore((s) => s.exportNote);
  const byStress = useStore((s) => s.colorByStress);
  const stress = useStore((s) => s.stress);
  const stressUi = useStore((s) => s.stressUi);
  const frameShape = useStore((s) => s.run.densityFrame?.shape);
  const { setThreshold, setSmoothIterations, setGhostDesign } = useStore.getState();
  const finished = status === 'done' || status === 'cancelled';
  const ready = !!runId && finished;
  const opts = resultOptions({ threshold, smoothIterations: smooth, trimToCad: trim });
  const sameGrid = !stress || !frameShape || stress.shape.every((n, k) => n === frameShape[k]);

  // A plain click fetches the file so the server's warning/error can be shown; the href stays for right-click,
  // open-in-new-tab and modified clicks, which keep the browser's own behaviour.
  const link = (label: string, href: string | null, testId: string, download: string) =>
    href ? (
      <a
        className="btn export"
        href={href}
        download={download}
        data-testid={testId}
        onClick={(e: MouseEvent<HTMLAnchorElement>) => {
          if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
          e.preventDefault();
          void downloadExport(label, href, download);
        }}
      >
        <Icon name="export" size={14} />
        {label}
      </a>
    ) : (
      <span className="btn export disabled" aria-disabled="true" data-testid={testId}>
        <Icon name="export" size={14} />
        {label}
      </span>
    );

  return (
    <>
      <Group title="Density" icon="density">
        <p className="dim" data-testid="density-info">
          {info.mode === 'none' ? 'No density frames yet: run the study.' : `Showing ${info.count.toLocaleString()} cells (${info.mode}) at iteration ${info.it}.`}
        </p>
        <Slider
          label="Density threshold"
          value={threshold}
          min={0.05}
          max={0.95}
          step={0.01}
          format={(v) => v.toFixed(2)}
          testId="threshold"
          onChange={setThreshold}
        />
        <label className="check">
          <input type="checkbox" checked={ghost} onChange={(e) => setGhostDesign(e.target.checked)} data-testid="ghost-design" />
          Ghost the design body (off = hide) while showing results
        </label>
      </Group>
      <Group title="Von Mises stress" icon="stress">
        <label className="check" title={ready ? 'Colour the density cells by von Mises stress (per element)' : 'Available once the run has finished'}>
          <input
            type="checkbox"
            checked={byStress}
            disabled={!ready || stressUi.loading}
            onChange={(e) => void setColorByStress(e.target.checked)}
            data-testid="color-stress"
          />
          Color by stress
        </label>
        {stressUi.loading && <p className="dim">Fetching stress...</p>}
        {stressUi.error && (
          <p className="amber" data-testid="stress-error">
            Stress not available: {stressUi.error}
          </p>
        )}
        {byStress && stress && (
          <div className="stack" data-testid="stress-legend">
            <div className="ramp" style={{ background: STRESS_RAMP }} />
            <div className="legend">
              <span>0</span>
              <span data-testid="stress-max">max {stress.max.toPrecision(4)}</span>
            </div>
            {hasResult && <p className="dim">Stress colours the density cells; the result mesh stays density-coloured. Hide it to see them.</p>}
            {!sameGrid && <p className="amber">The stress grid differs from the density frames: the cells keep their density colours.</p>}
          </div>
        )}
      </Group>
      <Group title="Result body" icon="mesh">
        <div className="grid2">
          <Field label="Smoothing iterations">
            <NumField value={smooth} min={0} max={20} step={1} testId="smooth" onChange={(v) => setSmoothIterations(Math.round(v))} />
          </Field>
        </div>
        <label className="check" title="Intersect the result with the original CAD (STL download and result mesh)">
          <input type="checkbox" checked={trim} onChange={(e) => setTrimToCad(e.target.checked)} data-testid="trim-cad" />
          Trim to CAD
        </label>
        <div className="row wrap">
          <Btn variant="primary" icon="mesh" onClick={() => void loadResultMesh()} disabled={!ready} testId="load-result">
            Load result mesh
          </Btn>
          {hasResult && (
            <Btn icon="hide" onClick={hideResultMesh} testId="hide-result">
              Hide result mesh
            </Btn>
          )}
        </div>
        {warnings && (
          <p className="amber" data-testid="result-warnings">
            Server warning: {warnings}
          </p>
        )}
      </Group>
      <Group title="Export" icon="export">
        <div className="export-row">
          {link('STL', ready ? api.resultStlUrl(runId, opts) : null, 'download-stl', `${runId}.stl`)}
          {link('VTI', ready ? api.resultVtiUrl(runId) : null, 'download-vti', `${runId}.vti`)}
          {link('NPZ', ready ? api.resultNpzUrl(runId) : null, 'download-npz', `${runId}.npz`)}
          {link('project.json', ready ? api.projectJsonUrl(runId) : null, 'download-project', `${runId}-project.json`)}
        </div>
        {!ready && <p className="dim">Exports are available once a run has finished or been stopped.</p>}
        {exportNote && (
          <p className={exportNote.kind === 'error' ? 'red' : 'amber'} data-testid={exportNote.kind === 'error' ? 'export-error' : 'export-warnings'}>
            {exportNote.kind === 'error' ? `${exportNote.label} download failed: ${exportNote.text}` : `${exportNote.label} download, server warning: ${exportNote.text}`}
          </p>
        )}
      </Group>
      <LoadedRunGroup />
    </>
  );
}

function LoadedRunGroup() {
  const loaded = useStore((s) => s.loadedRun !== null);
  if (!loaded) return null;
  return (
    <Group title="Loaded run" icon="open">
      <LoadedRun />
    </Group>
  );
}
