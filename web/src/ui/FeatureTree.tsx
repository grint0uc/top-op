// Specification tree: the project as features. Group nodes open their property page; item rows (loads, supports,
// reference bodies) activate the item. Right-click (or the row's menu button) opens a context menu; double-click renames.
import { type MouseEvent, type ReactNode, useEffect, useRef } from 'react';
import type { Selection } from '../api/client';
import { reselect, resolvePreview } from '../state/actions';
import { designEntry } from '../state/derived';
import { currentSelectionSpec, useStore } from '../state/store';
import { Icon, type IconName } from './icons';
import { caseColor, SUPPORT_COLOR, selectionLabel } from './ItemRows';
import { fixLabel } from './SupportsPanel';
import { RefIcon, modeLabel } from './RefModelsPanel';
import { cssHex } from '../state/derived';
import { type ContextMenuState, type PageId, useUi } from './uiStore';

function GroupNode({
  id,
  icon,
  label,
  count,
  children,
  detail,
}: {
  id: PageId;
  icon: IconName;
  label: string;
  count?: number;
  children?: ReactNode;
  detail?: ReactNode;
}) {
  const page = useUi((s) => s.page);
  const folded = useUi((s) => !!s.folded[id]);
  const hasItem = useStore((s) => s.activeItem !== null && s.activeItem.kind !== 'design');
  const { openPage, toggleFold } = useUi.getState();
  const selected = page === id && !hasItem;
  return (
    <li className="tree-group" role="treeitem" aria-expanded={children ? !folded : undefined} aria-selected={selected}>
      <div className={`tree-row group-row ${selected ? 'selected' : ''}`} data-testid={`tree-${id}`} onClick={() => openPage(id)}>
        {children ? (
          <button
            type="button"
            className={`tree-caret ${folded ? '' : 'open'}`}
            aria-label={folded ? 'Expand' : 'Collapse'}
            data-testid={`tree-fold-${id}`}
            onClick={(e) => {
              e.stopPropagation();
              toggleFold(id);
            }}
          >
            <Icon name="chevron" size={12} />
          </button>
        ) : (
          <span className="tree-caret" />
        )}
        <span className="tree-glyph">
          <Icon name={icon} size={15} />
        </span>
        <span className="tree-label grow ellipsis">{label}</span>
        {detail}
        {count !== undefined && <span className="count">{count}</span>}
      </div>
      {children && !folded && (
        <ul className="tree-children" role="group">
          {children}
        </ul>
      )}
    </li>
  );
}

function InfoRow({ children, page, testId }: { children: ReactNode; page: PageId; testId?: string }) {
  return (
    <li className="tree-row info-row" role="treeitem" onClick={() => useUi.getState().openPage(page)} data-testid={testId}>
      {children}
    </li>
  );
}

/** Inline rename: Enter or blur commits, Esc cancels. */
function RenameInput({ value, onCommit }: { value: string; onCommit: (v: string) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    ref.current?.focus();
    ref.current?.select();
  }, []);
  const done = () => useUi.getState().setRenaming(null);
  return (
    <input
      ref={ref}
      className="text tree-rename"
      defaultValue={value}
      data-testid="tree-rename"
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === 'Enter') {
          onCommit(e.currentTarget.value);
          done();
        } else if (e.key === 'Escape') done();
      }}
      onBlur={(e) => {
        onCommit(e.currentTarget.value);
        done();
      }}
    />
  );
}

interface ItemRowProps {
  id: string;
  testId: string;
  name: string;
  active: boolean;
  glyph: ReactNode;
  detail: string;
  tag?: ReactNode;
  onActivate: () => void;
  onRename: (name: string) => void;
  /** built when the menu opens, so enabled states reflect the current selection */
  menu: () => ContextMenuState['items'];
  extra?: ReactNode;
  warn?: string;
}

function ItemRow({ id, testId, name, active, glyph, detail, tag, onActivate, onRename, menu, extra, warn }: ItemRowProps) {
  const renaming = useUi((s) => s.renaming === id);
  const openMenu = (e: MouseEvent, x?: number, y?: number) => {
    e.preventDefault();
    e.stopPropagation();
    useUi.getState().setMenu({ x: x ?? e.clientX, y: y ?? e.clientY, items: menu() });
  };
  return (
    <li
      className={`tree-row item-row ${active ? 'selected' : ''}`}
      role="treeitem"
      aria-selected={active}
      data-testid={testId}
      onClick={onActivate}
      onDoubleClick={(e) => {
        e.stopPropagation();
        useUi.getState().setRenaming(id);
      }}
      onContextMenu={(e) => openMenu(e)}
      title={`${name}: ${detail}`}
    >
      <span className="tree-caret" />
      {glyph}
      <span className="tree-text grow">
        {renaming ? <RenameInput value={name} onCommit={(v) => v.trim() && onRename(v.trim())} /> : <span className="tree-label ellipsis">{name}</span>}
        <span className="tree-detail ellipsis">{detail}</span>
      </span>
      {warn && (
        <span className="tree-warn" title={warn}>
          <Icon name="warn" size={13} />
        </span>
      )}
      {tag}
      {extra}
      <button
        type="button"
        className="tree-more"
        aria-label="More actions"
        data-testid="tree-more"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          openMenu(e, r.left, r.bottom);
        }}
      >
        <Icon name="more" size={14} />
      </button>
    </li>
  );
}

function selectionMenu(sel: Selection, label: string, onReplace: (s: Selection) => void): ContextMenuState['items'] {
  const hasCurrent = currentSelectionSpec(useStore.getState()) !== null;
  return [
    { label: 'Edit selection', testId: 'ctx-select', run: () => reselect(sel) },
    { label: 'Preview nodes', testId: 'ctx-preview', run: () => void resolvePreview(sel, label) },
    {
      label: 'Use current selection',
      testId: 'ctx-replace',
      disabled: !hasCurrent,
      run: () => {
        const cur = currentSelectionSpec(useStore.getState());
        if (cur) onReplace(cur);
      },
    },
  ];
}

export function FeatureTree() {
  const name = useStore((s) => s.project.name);
  const design = useStore((s) => designEntry(s));
  const designRef = useStore((s) => s.project.design_mesh);
  const memo = useStore((s) => s.meshMemo);
  const loads = useStore((s) => s.project.loads);
  const supports = useStore((s) => s.project.supports);
  const refs = useStore((s) => s.project.ref_models);
  const meshes = useStore((s) => s.meshes);
  const active = useStore((s) => s.activeItem);
  const voxel = useStore((s) => s.voxel.stats);
  const grid = useStore((s) => s.project.grid);
  const params = useStore((s) => s.project.params);
  const material = useStore((s) => s.project.material);
  const run = useStore((s) => s.run);
  const loadedRun = useStore((s) => s.loadedRun);
  const st = useStore.getState();
  const setRenaming = useUi((s) => s.setRenaming);
  const isActive = (kind: string, id: string) => active?.kind === kind && active.id === id;
  const memoName = designRef?.mesh_id ? memo[designRef.mesh_id]?.name : undefined;

  return (
    <nav className="tree" aria-label="Feature tree" data-testid="feature-tree">
      <div className="tree-root" title={name}>
        <Icon name="project" size={15} />
        <span className="ellipsis grow">{name || 'untitled'}</span>
      </div>
      <ul className="tree-list" role="tree">
        <GroupNode id="part" icon="part" label="Design body">
          {design ? (
            <InfoRow page="part" testId="tree-design">
              <span className="tree-caret" />
              <span className="tree-glyph accent">
                <Icon name="part" size={14} />
              </span>
              <span className="tree-text grow">
                <span className="tree-label ellipsis">{design.info.name}</span>
                <span className="tree-detail">
                  {design.info.n_faces.toLocaleString()} faces{design.info.source === 'step' ? ' · STEP' : ''}
                </span>
              </span>
            </InfoRow>
          ) : (
            <InfoRow page="part">
              <span className="tree-caret" />
              <span className="tree-detail">{designRef?.mesh_id ? `${memoName ?? designRef.mesh_id} (not loaded)` : 'not imported'}</span>
            </InfoRow>
          )}
        </GroupNode>
        <GroupNode id="refs" icon="ref" label="Reference bodies" count={refs.length}>
          {refs.map((r) => (
            <ItemRow
              key={r.id}
              id={r.id}
              testId="ref-row"
              name={r.name || r.id}
              active={isActive('ref', r.id)}
              glyph={<RefIcon r={r} />}
              detail={modeLabel(r)}
              warn={r.mesh_id && meshes[r.mesh_id] ? undefined : 'Geometry missing: re-upload'}
              onActivate={() => {
                st.setActiveItem({ kind: 'ref', id: r.id });
                st.setTool('gizmo');
              }}
              onRename={(n) => st.updateRef(r.id, { name: n })}
              extra={
                <label className={`tree-eye ${r.visible ? '' : 'off'}`} title={r.visible ? 'Hide' : 'Show'} onClick={(e) => e.stopPropagation()}>
                  <input type="checkbox" checked={r.visible} data-testid="ref-visible" onChange={(e) => st.updateRef(r.id, { visible: e.target.checked })} />
                  <Icon name={r.visible ? 'eye' : 'hide'} size={14} />
                </label>
              }
              menu={() => [
                { label: 'Rename', testId: 'ctx-rename', run: () => setRenaming(r.id) },
                { label: r.visible ? 'Hide' : 'Show', testId: 'ctx-visibility', run: () => st.updateRef(r.id, { visible: !r.visible }) },
                {
                  label: r.mode === 'keep_in' ? 'Make keep-out' : 'Make keep-in',
                  testId: 'ctx-mode',
                  run: () => st.updateRef(r.id, { mode: r.mode === 'keep_in' ? 'keep_out' : 'keep_in' }),
                },
                { label: 'Delete', testId: 'ctx-delete', danger: true, run: () => st.removeRef(r.id) },
              ]}
            />
          ))}
        </GroupNode>
        <GroupNode id="loads" icon="force" label="Loads" count={loads.length}>
          {loads.map((l) => (
            <ItemRow
              key={l.id}
              id={l.id}
              testId="load-row"
              name={l.name || l.id}
              active={isActive('load', l.id)}
              glyph={
                <span className="tree-glyph" style={{ color: cssHex(caseColor(l.case)) }}>
                  <Icon name="force" size={14} />
                </span>
              }
              detail={selectionLabel(l.selection)}
              tag={<span className="tree-tag">C{l.case}</span>}
              onActivate={() => st.setActiveItem({ kind: 'load', id: l.id })}
              onRename={(n) => st.updateLoad(l.id, { name: n })}
              menu={() => [
                { label: 'Rename', testId: 'ctx-rename', run: () => setRenaming(l.id) },
                ...selectionMenu(l.selection, l.name || l.id, (selection) => st.updateLoad(l.id, { selection })),
                { label: 'Delete', testId: 'ctx-delete', danger: true, run: () => st.removeLoad(l.id) },
              ]}
            />
          ))}
        </GroupNode>
        <GroupNode id="supports" icon="support" label="Supports" count={supports.length}>
          {supports.map((sp) => (
            <ItemRow
              key={sp.id}
              id={sp.id}
              testId="support-row"
              name={sp.name || sp.id}
              active={isActive('support', sp.id)}
              glyph={
                <span className="tree-glyph" style={{ color: cssHex(SUPPORT_COLOR) }}>
                  <Icon name="support" size={14} />
                </span>
              }
              detail={selectionLabel(sp.selection)}
              tag={<span className="tree-tag">{fixLabel(sp)}</span>}
              onActivate={() => st.setActiveItem({ kind: 'support', id: sp.id })}
              onRename={(n) => st.updateSupport(sp.id, { name: n })}
              menu={() => [
                { label: 'Rename', testId: 'ctx-rename', run: () => setRenaming(sp.id) },
                ...selectionMenu(sp.selection, sp.name || sp.id, (selection) => st.updateSupport(sp.id, { selection })),
                { label: 'Delete', testId: 'ctx-delete', danger: true, run: () => st.removeSupport(sp.id) },
              ]}
            />
          ))}
        </GroupNode>
        <GroupNode id="domain" icon="domain" label="Domain / Mesh">
          <InfoRow page="domain">
            <span className="tree-caret" />
            <span className="tree-detail">
              {voxel ? `${voxel.nx} × ${voxel.ny} × ${voxel.nz} · h ${voxel.h.toPrecision(3)}` : `${grid.elements_along_longest} elements along longest`}
            </span>
          </InfoRow>
        </GroupNode>
        <GroupNode id="study" icon="study" label="Study settings">
          <InfoRow page="study">
            <span className="tree-caret" />
            <span className="tree-detail">
              Compliance · V {params.volfrac.toFixed(2)} · {params.stress_limit != null ? 'MMA, stress' : params.optimizer.toUpperCase()}
              {(params.symmetry?.length ?? 0) > 0 ? ` · sym ${params.symmetry!.map((p) => p.axis).join('')}` : ''}
              {params.overhang ? ` · AM ${params.overhang}` : ''}
            </span>
          </InfoRow>
        </GroupNode>
        <GroupNode
          id="material"
          icon="material"
          label="Material"
          detail={
            <span className="tree-detail inline">
              E {+material.E.toPrecision(4)} · ν {material.nu}
            </span>
          }
        />
        <GroupNode id="results" icon="results" label="Results">
          <InfoRow page="results" testId="tree-run">
            <span className="tree-caret" />
            <span className={`dot status-${run.status}`} />
            <span className="tree-text grow">
              <span className="tree-label ellipsis">{run.id ? `Run ${run.id}` : 'No run yet'}</span>
              <span className="tree-detail">
                {run.status}
                {run.history.length > 0 ? ` · ${run.history.length} it` : ''}
              </span>
            </span>
          </InfoRow>
          {loadedRun && (
            <InfoRow page="results" testId="tree-loaded-run">
              <span className="tree-caret" />
              <span className="tree-glyph">
                <Icon name="open" size={14} />
              </span>
              <span className="tree-text grow">
                <span className="tree-label ellipsis">{loadedRun.fileName}</span>
                <span className="tree-detail">
                  {loadedRun.run.status} · {loadedRun.run.history?.length ?? 0} it
                </span>
              </span>
            </InfoRow>
          )}
        </GroupNode>
      </ul>
    </nav>
  );
}

/** The floating context menu (one at a time); closes on outside click, Esc or after a command. */
export function ContextMenu() {
  const menu = useUi((s) => s.menu);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!menu) return;
    const close = (e: Event) => {
      if (e instanceof KeyboardEvent) {
        if (e.key !== 'Escape') return;
        e.stopPropagation(); // this Esc only closes the menu
      }
      if (e.target instanceof Node && ref.current?.contains(e.target) && !(e instanceof KeyboardEvent)) return;
      useUi.getState().setMenu(null);
    };
    window.addEventListener('pointerdown', close, true);
    window.addEventListener('keydown', close, true);
    window.addEventListener('blur', close);
    return () => {
      window.removeEventListener('pointerdown', close, true);
      window.removeEventListener('keydown', close, true);
      window.removeEventListener('blur', close);
    };
  }, [menu]);
  if (!menu) return null;
  const x = Math.min(menu.x, window.innerWidth - 200);
  const y = Math.min(menu.y, window.innerHeight - 28 * menu.items.length - 12);
  return (
    <div className="ctx-menu" role="menu" ref={ref} style={{ left: x, top: y }} data-testid="context-menu">
      {menu.items.map((it) => (
        <button
          key={it.label}
          type="button"
          role="menuitem"
          className={`ctx-item ${it.danger ? 'danger' : ''}`}
          disabled={it.disabled}
          data-testid={it.testId}
          onClick={() => {
            useUi.getState().setMenu(null);
            it.run();
          }}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}
