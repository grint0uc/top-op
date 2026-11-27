import type { SupportSpec } from '../api/client';
import { addSupportFromSelection } from '../state/actions';
import { hasSelection, selectionSummary } from '../state/derived';
import { useStore } from '../state/store';
import { Btn, Field, Group } from './controls';
import { Icon } from './icons';
import { ItemActions, SUPPORT_COLOR, Swatch, selectionLabel } from './ItemRows';

export const fixLabel = (sp: SupportSpec): string => {
  const fixed = ['X', 'Y', 'Z'].filter((_, k) => sp.fix?.[k] ?? true);
  return fixed.length === 3 ? 'fixed' : fixed.length === 0 ? 'free' : fixed.join('');
};

export function SupportsPage() {
  const supports = useStore((s) => s.project.supports);
  const hasSel = useStore((s) => hasSelection(s));
  const summary = useStore((s) => selectionSummary(s));
  const { setActiveItem } = useStore.getState();
  return (
    <>
      <Group title="New fixture" icon="support">
        <p className="dim">Fix at least one region. Select it first (faces, a query or a primitive).</p>
        <div className="row">
          <Btn variant="primary" icon="support" onClick={() => addSupportFromSelection()} disabled={!hasSel} testId="pm-add-support">
            Fixture from selection
          </Btn>
          <span className="dim ellipsis grow">{summary}</span>
        </div>
      </Group>
      <Group title="Supports" icon="list" aside={<span className="count">{supports.length}</span>}>
        {supports.length === 0 && <p className="dim">No supports yet.</p>}
        <ul className="pm-list">
          {supports.map((sp) => (
            <li key={sp.id}>
              <button type="button" className="pm-list-row" onClick={() => setActiveItem({ kind: 'support', id: sp.id })}>
                <Swatch color={SUPPORT_COLOR} />
                <span className="grow ellipsis">{sp.name || sp.id}</span>
                <span className="dim">{fixLabel(sp)}</span>
              </button>
            </li>
          ))}
        </ul>
      </Group>
    </>
  );
}

export function SupportPage({ support: sp }: { support: SupportSpec }) {
  const { updateSupport, removeSupport } = useStore.getState();
  return (
    <>
      <Group title="Fixture" icon="support">
        <Field label="Name">
          <input type="text" className="text" value={sp.name} data-testid="support-name" onChange={(e) => updateSupport(sp.id, { name: e.target.value })} />
        </Field>
      </Group>
      <Group title="Fixed translations" icon="gizmo" aside={<span className="dim">{fixLabel(sp)}</span>}>
        <div className="fix-row">
          {(['x', 'y', 'z'] as const).map((axis, k) => (
            <label key={axis} className="check fix">
              <input
                type="checkbox"
                checked={sp.fix?.[k] ?? true}
                data-testid={`fix-${axis}`}
                onChange={(e) =>
                  updateSupport(sp.id, {
                    fix: [0, 1, 2].map((i) => (i === k ? e.target.checked : (sp.fix?.[i] ?? true))),
                  })
                }
              />
              U{axis}
            </label>
          ))}
        </div>
        <p className="dim">Unchecked directions may slide (roller support).</p>
      </Group>
      <Group title="Region" icon="target">
        <p className="region-label">
          <Icon name="target" size={14} /> {selectionLabel(sp.selection)}
        </p>
        <ItemActions
          selection={sp.selection}
          label={sp.name || sp.id}
          onReplace={(selection) => updateSupport(sp.id, { selection })}
          onDelete={() => removeSupport(sp.id)}
        />
      </Group>
    </>
  );
}
