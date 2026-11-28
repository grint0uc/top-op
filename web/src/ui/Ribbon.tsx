// Title bar (quick access toolbar, document name, theme) and the command ribbon. Ribbon commands call the same
// actions as the property pages and the selection toolbar; choosing a tab opens that workflow stage's page.
import { type KeyboardEvent, type ReactNode, type Ref, useRef } from 'react';
import { api } from '../api/client';
import {
  addLoadFromSelection,
  addPrimitive,
  addRefPrimitive,
  downloadExport,
  exportProjectJson,
  hideResultMesh,
  loadResultMesh,
  resolvePreview,
  resultOptions,
  runVoxelize,
  setColorByStress,
  startRun,
  stopRun,
  addSupportFromSelection,
} from '../state/actions';
import { designEntry, hasSelection } from '../state/derived';
import { type ToolMode, useStore } from '../state/store';
import { getViewport } from '../viewport/mount';
import { useFocusHandoff } from './a11y';
import { AppMark, Icon, type IconName } from './icons';
import { IS_MOCK, loadExample, pickFile } from './ImportPanel';
import { PAGE_TAB, type PageId, type RibbonTab, TAB_PAGE, useUi } from './uiStore';

const TABS: { id: RibbonTab; label: string }[] = [
  { id: 'model', label: 'Model' },
  { id: 'bc', label: 'Boundary Conditions' },
  { id: 'domain', label: 'Domain' },
  { id: 'optimize', label: 'Optimize' },
  { id: 'results', label: 'Results' },
];

interface CmdProps {
  icon: IconName;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
  testId: string;
  title?: string;
  size?: 'lg' | 'sm';
  tone?: 'run' | 'stop';
  btnRef?: Ref<HTMLButtonElement>;
}

function Cmd({ icon, label, onClick, disabled, active, testId, title, size = 'lg', tone, btnRef }: CmdProps) {
  return (
    <button
      ref={btnRef}
      type="button"
      className={`cmd cmd-${size} ${active ? 'on' : ''} ${tone ? `tone-${tone}` : ''}`}
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
      data-testid={testId}
      title={title ?? label}
    >
      <Icon name={icon} size={size === 'lg' ? 22 : 15} />
      <span className="cmd-label">{label}</span>
    </button>
  );
}

function RGroup({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="rgroup" role="group" aria-label={label}>
      <div className="rgroup-cmds">{children}</div>
      <div className="rgroup-label">{label}</div>
    </div>
  );
}

/** A column of small commands stacked inside a group. */
const Stack = ({ children }: { children: ReactNode }) => <div className="cmd-stack">{children}</div>;

const open = (p: PageId) => useUi.getState().openPage(p);

function ModeCmd({ mode, icon, label, k, disabled }: { mode: ToolMode; icon: IconName; label: string; k: string; disabled?: boolean }) {
  const tool = useStore((s) => s.tool);
  return (
    <Cmd
      icon={icon}
      label={label}
      size="sm"
      active={tool === mode}
      disabled={disabled}
      onClick={() => useStore.getState().setTool(mode)}
      testId={`rb-mode-${mode}`}
      title={`${label} [${k}]`}
    />
  );
}

function ModelTab({ hasDesign }: { hasDesign: boolean }) {
  const busy = useStore((s) => s.busy);
  const designActive = useStore((s) => s.activeItem?.kind === 'design');
  const st = useStore.getState();
  return (
    <>
      <RGroup label="Geometry">
        <Cmd
          icon="import"
          label="Import"
          onClick={() => pickFile('mesh')}
          disabled={busy === 'Upload'}
          testId="rb-import"
          title="Import a design body (STL or STEP); or drop a file anywhere"
        />
        {IS_MOCK && (
          <Cmd
            icon="example"
            label="Example"
            onClick={() => void loadExample()}
            testId="load-example"
            title="Mock backend only: serves examples/bracket.stl from the repo root"
          />
        )}
      </RGroup>
      <RGroup label="Project">
        <Stack>
          <Cmd icon="new" label="New" size="sm" onClick={() => st.resetProject()} testId="rb-new" />
          <Cmd icon="open" label="Open project.json" size="sm" onClick={() => pickFile('project')} testId="rb-open" />
          <Cmd icon="save" label="Save project.json" size="sm" onClick={exportProjectJson} disabled={!hasDesign} testId="rb-save" />
        </Stack>
      </RGroup>
      <RGroup label="Design body">
        <Cmd icon="part" label="Properties" onClick={() => open('part')} testId="rb-part" />
        <Stack>
          <Cmd
            icon="gizmo"
            label="Move / rotate"
            size="sm"
            active={designActive}
            disabled={!hasDesign}
            onClick={() => {
              if (designActive) st.setActiveItem(null);
              else {
                st.setActiveItem({ kind: 'design', id: 'design' });
                st.setTool('gizmo');
              }
            }}
            testId="rb-design-gizmo"
          />
          <Cmd
            icon="reset"
            label="Reset pose"
            size="sm"
            disabled={!hasDesign}
            onClick={() => st.setDesignTransform([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])}
            testId="rb-design-reset"
          />
        </Stack>
      </RGroup>
      <RGroup label="View">
        <Cmd icon="fit" label="Fit all" onClick={() => getViewport()?.fitAll()} testId="rb-fit" title="Fit all [F]" />
        <Cmd icon="iso" label="Isometric" onClick={() => getViewport()?.setView('iso')} testId="rb-iso" />
      </RGroup>
    </>
  );
}

function BcTab({ hasDesign }: { hasDesign: boolean }) {
  const hasSel = useStore((s) => hasSelection(s));
  return (
    <>
      <RGroup label="Select">
        <Stack>
          <ModeCmd mode="orbit" icon="orbit" label="Rotate view" k="1" />
          <ModeCmd mode="pick" icon="pick" label="Pick faces" k="2" disabled={!hasDesign} />
          <ModeCmd mode="paint" icon="paint" label="Paint faces" k="3" disabled={!hasDesign} />
        </Stack>
        <Stack>
          <ModeCmd mode="gizmo" icon="gizmo" label="Gizmo" k="4" />
          <ModeCmd mode="query" icon="query" label="Query" k="5" disabled={!hasDesign} />
        </Stack>
      </RGroup>
      <RGroup label="Region">
        {(['box', 'sphere', 'cylinder'] as const).map((k) => (
          <Cmd
            key={k}
            icon={k}
            label={k[0]!.toUpperCase() + k.slice(1)}
            onClick={() => addPrimitive(k)}
            disabled={!hasDesign}
            testId={`rb-add-${k}`}
            title={`Region primitive: ${k}`}
          />
        ))}
      </RGroup>
      <RGroup label="Define">
        <Cmd
          icon="force"
          label="Force"
          onClick={() => addLoadFromSelection()}
          disabled={!hasSel}
          testId="rb-add-load"
          title="Create a load from the current selection"
        />
        <Cmd
          icon="support"
          label="Fixture"
          onClick={() => addSupportFromSelection()}
          disabled={!hasSel}
          testId="rb-add-support"
          title="Create a support from the current selection"
        />
      </RGroup>
      <RGroup label="Check">
        <Stack>
          <Cmd icon="preview" label="Preview nodes" size="sm" onClick={() => void resolvePreview()} testId="rb-preview" />
          <Cmd icon="clear" label="Clear selection" size="sm" disabled={!hasSel} onClick={() => useStore.getState().clearSelection()} testId="rb-clear" />
          <Cmd icon="list" label="Loads / Supports" size="sm" onClick={() => open('loads')} testId="rb-bc-list" />
        </Stack>
      </RGroup>
    </>
  );
}

function DomainTab({ hasDesign }: { hasDesign: boolean }) {
  const loading = useStore((s) => s.voxel.loading);
  return (
    <>
      <RGroup label="Design space">
        <Cmd icon="domain" label="Resolution" onClick={() => open('domain')} testId="rb-domain" />
        <Cmd icon="voxelize" label="Voxelize" onClick={() => void runVoxelize()} disabled={!hasDesign || loading} testId="rb-voxelize" />
      </RGroup>
      <RGroup label="Reference bodies">
        <Cmd icon="ref" label="Reference STL" onClick={() => pickFile('ref')} disabled={!hasDesign} testId="rb-add-ref" />
        <Stack>
          {(['box', 'sphere', 'cylinder'] as const).map((k) => (
            <Cmd key={k} icon={k} label={`Keep-out ${k}`} size="sm" onClick={() => void addRefPrimitive(k)} disabled={!hasDesign} testId={`rb-add-ref-${k}`} />
          ))}
        </Stack>
        <Cmd icon="list" label="Bodies" onClick={() => open('refs')} testId="rb-refs" />
      </RGroup>
    </>
  );
}

function OptimizeTab({ hasDesign }: { hasDesign: boolean }) {
  const status = useStore((s) => s.run.status);
  const active = status === 'queued' || status === 'running';
  const st = useStore.getState();
  const runBtn = useRef<HTMLButtonElement>(null);
  const stopBtn = useRef<HTMLButtonElement>(null);
  useFocusHandoff(active, runBtn, stopBtn);
  return (
    <>
      <RGroup label="Study">
        <Cmd icon="study" label="Settings" onClick={() => open('study')} testId="rb-study" />
        <Cmd icon="material" label="Material" onClick={() => open('material')} testId="rb-material" />
        <Stack>
          <Cmd
            icon="symmetry"
            label="Symmetry plane"
            size="sm"
            onClick={() => {
              st.setParams({ symmetry: [...(st.project.params.symmetry ?? []), { axis: 'x', position: null }] });
              open('study');
            }}
            testId="rb-symmetry"
          />
          <Cmd icon="overhang" label="Overhang (AM)" size="sm" onClick={() => open('study')} testId="rb-overhang" />
        </Stack>
      </RGroup>
      <RGroup label="Solve">
        <Cmd icon="run" label="Run" tone="run" btnRef={runBtn} onClick={() => void startRun()} disabled={active || !hasDesign} testId="rb-run" />
        <Cmd icon="stop" label="Stop" tone="stop" btnRef={stopBtn} onClick={() => void stopRun()} disabled={!active} testId="rb-stop" />
      </RGroup>
    </>
  );
}

function ResultsTab() {
  const runId = useStore((s) => s.run.id);
  const status = useStore((s) => s.run.status);
  const byStress = useStore((s) => s.colorByStress);
  const hasResult = useStore((s) => s.resultStl !== null);
  const stressLoading = useStore((s) => s.stressUi.loading);
  const threshold = useStore((s) => s.threshold);
  const smooth = useStore((s) => s.smoothIterations);
  const trim = useStore((s) => s.trimToCad);
  const ready = !!runId && (status === 'done' || status === 'cancelled');
  const dl = (label: string, url: string, file: string) => void downloadExport(label, url, file);
  return (
    <>
      <RGroup label="Display">
        <Cmd icon="density" label="Density" onClick={() => open('results')} testId="rb-results" />
        <Cmd
          icon="stress"
          label="Von Mises"
          active={byStress}
          disabled={!ready || stressLoading}
          onClick={() => void setColorByStress(!byStress)}
          testId="rb-stress"
          title="Colour the density cells by von Mises stress"
        />
      </RGroup>
      <RGroup label="Result body">
        <Cmd icon="mesh" label="Result mesh" disabled={!ready} onClick={() => void loadResultMesh()} testId="rb-load-result" />
        <Cmd icon="hide" label="Hide" disabled={!hasResult} onClick={hideResultMesh} testId="rb-hide-result" />
      </RGroup>
      <RGroup label="Export">
        <Stack>
          <Cmd
            icon="export"
            label="STL"
            size="sm"
            disabled={!ready}
            onClick={() => runId && dl('STL', api.resultStlUrl(runId, resultOptions({ threshold, smoothIterations: smooth, trimToCad: trim })), `${runId}.stl`)}
            testId="rb-export-stl"
          />
          <Cmd
            icon="export"
            label="VTI"
            size="sm"
            disabled={!ready}
            onClick={() => runId && dl('VTI', api.resultVtiUrl(runId), `${runId}.vti`)}
            testId="rb-export-vti"
          />
        </Stack>
        <Stack>
          <Cmd
            icon="export"
            label="NPZ"
            size="sm"
            disabled={!ready}
            onClick={() => runId && dl('NPZ', api.resultNpzUrl(runId), `${runId}.npz`)}
            testId="rb-export-npz"
          />
          <Cmd
            icon="export"
            label="project.json"
            size="sm"
            disabled={!ready}
            onClick={() => runId && dl('project.json', api.projectJsonUrl(runId), `${runId}-project.json`)}
            testId="rb-export-project"
          />
        </Stack>
      </RGroup>
    </>
  );
}

export function Ribbon() {
  const tab = useUi((s) => s.tab);
  const hasDesign = useStore((s) => designEntry(s) !== null);
  const list = useRef<HTMLDivElement>(null);
  // the tab follows the workflow stage: it opens that stage's page unless one of its pages is already open
  const choose = (t: RibbonTab) => {
    const ui = useUi.getState();
    if (PAGE_TAB[ui.page] !== t) ui.openPage(TAB_PAGE[t]);
    ui.setTab(t);
  };
  // tablist pattern: one tab stop, arrows / Home / End move and select (automatic activation)
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = TABS.findIndex((t) => t.id === tab);
    const n = TABS.length;
    const next =
      e.key === 'ArrowRight' ? (i + 1) % n : e.key === 'ArrowLeft' ? (i - 1 + n) % n : e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : -1;
    if (next < 0) return;
    e.preventDefault();
    choose(TABS[next]!.id);
    list.current?.querySelector<HTMLElement>(`#ribbon-tab-${TABS[next]!.id}`)?.focus();
  };
  return (
    <section className="ribbon" data-testid="ribbon" aria-label="Command ribbon">
      <div className="ribbon-tabs" role="tablist" aria-label="Ribbon" ref={list} onKeyDown={onKey}>
        {TABS.map((t) => (
          <button
            key={t.id}
            id={`ribbon-tab-${t.id}`}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            aria-controls="ribbon-panel"
            tabIndex={tab === t.id ? 0 : -1}
            className={`ribbon-tab ${tab === t.id ? 'on' : ''}`}
            onClick={() => choose(t.id)}
            data-testid={`ribbon-tab-${t.id}`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="ribbon-body" role="tabpanel" id="ribbon-panel" aria-labelledby={`ribbon-tab-${tab}`}>
        {tab === 'model' && <ModelTab hasDesign={hasDesign} />}
        {tab === 'bc' && <BcTab hasDesign={hasDesign} />}
        {tab === 'domain' && <DomainTab hasDesign={hasDesign} />}
        {tab === 'optimize' && <OptimizeTab hasDesign={hasDesign} />}
        {tab === 'results' && <ResultsTab />}
      </div>
    </section>
  );
}

export function TitleBar() {
  const name = useStore((s) => s.project.name);
  const hasDesign = useStore((s) => designEntry(s) !== null);
  const busy = useStore((s) => s.busy);
  const dark = useUi((s) => s.dark);
  const shortcuts = useUi((s) => s.shortcuts);
  const st = useStore.getState();
  const q = (icon: IconName, label: string, onClick: () => void, testId: string, disabled?: boolean) => (
    <button type="button" className="qat-btn" title={label} aria-label={label} onClick={onClick} disabled={disabled} data-testid={testId}>
      <Icon name={icon} size={16} />
    </button>
  );
  return (
    <header className="titlebar">
      <div className="brand">
        <AppMark size={20} />
        <h1 className="brand-name">top-op</h1>
        <span className="brand-sub">Topology Optimization</span>
      </div>
      <div className="qat" role="toolbar" aria-label="Quick access">
        {q('new', 'New project (discard the current one)', () => st.resetProject(), 'new-project')}
        {q(
          'open',
          'Load project.json: a run export or a saved project (loads, supports, parameters and run history)',
          () => pickFile('project'),
          'load-project',
          busy === 'Load project',
        )}
        {q('save', 'Save the current setup as a project.json (no run)', exportProjectJson, 'export-project', !hasDesign)}
        <span className="qat-sep" />
        {q('import', 'Import design body (STL / STEP)', () => pickFile('mesh'), 'qat-import', busy === 'Upload')}
      </div>
      <div className="doc-name">
        <Icon name="part" size={14} />
        <input
          type="text"
          className="doc-input"
          value={name}
          aria-label="Project name"
          spellCheck={false}
          data-testid="project-name"
          onChange={(e) => st.setProjectName(e.target.value)}
        />
      </div>
      <div className="titlebar-right">
        {IS_MOCK && (
          <span className="badge" title="Served by web/mock, not the Python server">
            MOCK API
          </span>
        )}
        <button
          type="button"
          className={`qat-btn ${shortcuts ? '' : 'off'}`}
          onClick={() => useUi.getState().setShortcuts(!shortcuts)}
          title={`Single-key shortcuts (1-5, g / r / s, f): ${shortcuts ? 'on' : 'off'}`}
          aria-label="Single-key shortcuts"
          aria-pressed={shortcuts}
          data-testid="shortcuts-toggle"
        >
          <Icon name="keyboard" size={16} />
        </button>
        <button
          type="button"
          className="qat-btn"
          onClick={() => useUi.getState().setThemePref(dark ? 'light' : 'dark')}
          title={dark ? 'Light theme' : 'Dark theme'}
          aria-label={dark ? 'Switch to the light theme' : 'Switch to the dark theme'}
          data-testid="theme-toggle"
        >
          <Icon name={dark ? 'sun' : 'moon'} size={16} />
        </button>
      </div>
    </header>
  );
}
