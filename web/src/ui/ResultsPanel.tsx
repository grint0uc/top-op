import { api } from '../api/client';
import { hideResultMesh, loadResultMesh } from '../state/actions';
import { useStore } from '../state/store';
import { Btn, Field, NumField, Section, Slider } from './controls';

export function ResultsPanel() {
  const runId = useStore((s) => s.run.id);
  const status = useStore((s) => s.run.status);
  const threshold = useStore((s) => s.threshold);
  const smooth = useStore((s) => s.smoothIterations);
  const ghost = useStore((s) => s.ghostDesign);
  const info = useStore((s) => s.densityInfo);
  const hasResult = useStore((s) => s.resultStl !== null);
  const { setThreshold, setSmoothIterations, setGhostDesign } = useStore.getState();
  const finished = status === 'done' || status === 'cancelled';
  const ready = !!runId && finished;
  const opts = { threshold, smooth };

  const link = (label: string, href: string | null, testId: string, download: string) =>
    href ? (
      <a className="btn" href={href} download={download} data-testid={testId}>
        {label}
      </a>
    ) : (
      <span className="btn disabled" aria-disabled="true" data-testid={testId}>
        {label}
      </span>
    );

  return (
    <Section title="Results">
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
      <Field label="Smoothing iterations">
        <NumField value={smooth} min={0} max={20} step={1} testId="smooth" onChange={(v) => setSmoothIterations(Math.round(v))} />
      </Field>
      <label className="check">
        <input type="checkbox" checked={ghost} onChange={(e) => setGhostDesign(e.target.checked)} data-testid="ghost-design" />
        Ghost the design mesh (off = hide) while showing results
      </label>
      <p className="dim" data-testid="density-info">
        {info.mode === 'none' ? 'No density frames yet.' : `Showing ${info.count.toLocaleString()} cells (${info.mode}) at iteration ${info.it}.`}
      </p>
      <div className="row wrap">
        <Btn variant="primary" onClick={() => void loadResultMesh()} disabled={!ready} testId="load-result">
          Load result mesh
        </Btn>
        {hasResult && (
          <Btn onClick={hideResultMesh} testId="hide-result">
            Hide result mesh
          </Btn>
        )}
      </div>
      <div className="row wrap">
        {link('STL', ready ? api.resultStlUrl(runId, opts) : null, 'download-stl', `${runId}.stl`)}
        {link('VTI', ready ? api.resultVtiUrl(runId) : null, 'download-vti', `${runId}.vti`)}
        {link('NPZ', ready ? api.resultNpzUrl(runId) : null, 'download-npz', `${runId}.npz`)}
        {link('project.json', ready ? api.projectJsonUrl(runId) : null, 'download-project', `${runId}-project.json`)}
      </div>
    </Section>
  );
}
