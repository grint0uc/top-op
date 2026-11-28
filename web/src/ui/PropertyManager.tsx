// Property manager: the page of the selected feature-tree node. Group pages stay mounted (hidden) so their local
// state survives switching; an active load/support/reference body shows its own page instead.
import { type ReactNode, useEffect, useRef } from 'react';
import { useStore } from '../state/store';
import { DomainPage } from './DomainPanel';
import { Icon, type IconName } from './icons';
import { PartPage } from './ImportPanel';
import { LoadPage, LoadsPage } from './LoadsPanel';
import { RefPage, RefsPage } from './RefModelsPanel';
import { ResultsPage } from './ResultsPanel';
import { MaterialPage, StudyPage } from './RunPanel';
import { SupportPage, SupportsPage } from './SupportsPanel';
import { type PageId, useUi } from './uiStore';

export const PAGES: { id: PageId; title: string; icon: IconName; testId: string; render: () => ReactNode }[] = [
  { id: 'part', title: 'Design body', icon: 'part', testId: 'panel-import', render: () => <PartPage /> },
  { id: 'refs', title: 'Reference bodies', icon: 'ref', testId: 'panel-reference-models', render: () => <RefsPage /> },
  { id: 'loads', title: 'Loads', icon: 'force', testId: 'panel-loads', render: () => <LoadsPage /> },
  { id: 'supports', title: 'Supports', icon: 'support', testId: 'panel-supports', render: () => <SupportsPage /> },
  { id: 'domain', title: 'Domain / Mesh', icon: 'domain', testId: 'panel-domain', render: () => <DomainPage /> },
  { id: 'study', title: 'Study settings', icon: 'study', testId: 'panel-study', render: () => <StudyPage /> },
  { id: 'material', title: 'Material', icon: 'material', testId: 'panel-material', render: () => <MaterialPage /> },
  { id: 'results', title: 'Results', icon: 'results', testId: 'panel-results', render: () => <ResultsPage /> },
];

function ItemPage() {
  const active = useStore((s) => s.activeItem);
  const load = useStore((s) => (active?.kind === 'load' ? s.project.loads.find((l) => l.id === active.id) : undefined));
  const support = useStore((s) => (active?.kind === 'support' ? s.project.supports.find((x) => x.id === active.id) : undefined));
  const ref = useStore((s) => (active?.kind === 'ref' ? s.project.ref_models.find((r) => r.id === active.id) : undefined));
  if (load) return <LoadPage key={load.id} load={load} />;
  if (support) return <SupportPage key={support.id} support={support} />;
  if (ref) return <RefPage key={ref.id} r={ref} />;
  return null;
}

export function PropertyManager() {
  const page = useUi((s) => s.page);
  const active = useStore((s) => s.activeItem);
  const itemName = useStore((s) => {
    const a = s.activeItem;
    if (!a) return null;
    if (a.kind === 'load') return s.project.loads.find((l) => l.id === a.id)?.name ?? a.id;
    if (a.kind === 'support') return s.project.supports.find((x) => x.id === a.id)?.name ?? a.id;
    if (a.kind === 'ref') return s.project.ref_models.find((r) => r.id === a.id)?.name ?? a.id;
    return null;
  });
  // the design body's gizmo keeps the Design body page; other items get their own page
  const item = active && active.kind !== 'design' ? active : null;
  const meta = PAGES.find((p) => p.id === page)!;
  const icon: IconName = item ? (item.kind === 'load' ? 'force' : item.kind === 'support' ? 'support' : 'ref') : meta.icon;
  const kindLabel = item ? (item.kind === 'load' ? 'Force' : item.kind === 'support' ? 'Fixture' : 'Reference body') : null;
  const title = useRef<HTMLHeadingElement>(null);
  const itemKey = item ? `${item.kind}:${item.id}` : null;
  const prevKey = useRef(itemKey);
  // the item page unmounts when it is released (Done, Esc, delete): if focus was in it, keep it in the panel
  useEffect(() => {
    const was = prevKey.current;
    prevKey.current = itemKey;
    if (was && was !== itemKey && (document.activeElement === document.body || document.activeElement === null)) title.current?.focus();
  }, [itemKey]);

  return (
    <aside className="pm" aria-label="Property manager" data-testid="property-manager" data-page={item ? item.kind : page}>
      <header className="pm-head">
        <Icon name={icon} size={18} />
        <div className="pm-title grow">
          <span className="pm-kicker">{kindLabel ?? 'Properties'}</span>
          <h2 className="pm-name ellipsis" data-testid="pm-title" tabIndex={-1} ref={title}>
            {item ? itemName : meta.title}
          </h2>
        </div>
        {item && (
          <button
            type="button"
            className="icon-btn"
            title="Done (Esc)"
            aria-label="Done"
            data-testid="pm-done"
            onClick={() => useStore.getState().setActiveItem(null)}
          >
            <Icon name="check" size={16} />
          </button>
        )}
      </header>
      <div className="pm-body">
        {item && (
          <div className="pm-page" data-testid="panel-item">
            <ItemPage />
          </div>
        )}
        {PAGES.map((p) => (
          <div key={p.id} className="pm-page" data-testid={p.testId} hidden={!!item || p.id !== page}>
            {p.render()}
          </div>
        ))}
      </div>
    </aside>
  );
}
