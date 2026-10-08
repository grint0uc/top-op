import { addSupportFromSelection } from '../state/actions';
import { selectionSummary } from '../state/derived';
import { useStore } from '../state/store';
import { Btn, Section } from './controls';
import { ItemActions, SUPPORT_COLOR, Swatch, selectionLabel } from './ItemRows';

export function SupportsPanel() {
  const supports = useStore((s) => s.project.supports);
  const active = useStore((s) => s.activeItem);
  const hasSel = useStore((s) => s.selection.faceIds.length > 0 || s.selection.primitive !== null);
  const summary = useStore((s) => selectionSummary(s));
  const { updateSupport, removeSupport, setActiveItem } = useStore.getState();

  return (
    <Section title="Supports">
      <div className="row">
        <Btn variant="primary" onClick={() => addSupportFromSelection()} disabled={!hasSel} testId="add-support">
          Add from selection
        </Btn>
        <span className="dim">{summary}</span>
      </div>
      {supports.length === 0 && <p className="dim">No supports. Fix at least one region.</p>}
      <ul className="items">
        {supports.map((sp) => (
          <li
            key={sp.id}
            className={`item ${active?.kind === 'support' && active.id === sp.id ? 'active' : ''}`}
            data-testid="support-row"
            onClick={() => setActiveItem({ kind: 'support', id: sp.id })}
          >
            <div className="row">
              <Swatch color={SUPPORT_COLOR} />
              <input
                type="text"
                className="text grow"
                value={sp.name}
                data-testid="support-name"
                onChange={(e) => updateSupport(sp.id, { name: e.target.value })}
              />
              <span className="dim">{selectionLabel(sp.selection)}</span>
            </div>
            <div className="row">
              <span className="field-label">Fix</span>
              {(['x', 'y', 'z'] as const).map((axis, k) => (
                <label key={axis} className="check">
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
                  {axis.toUpperCase()}
                </label>
              ))}
            </div>
            <ItemActions
              selection={sp.selection}
              onReplace={(selection) => updateSupport(sp.id, { selection })}
              onDelete={() => removeSupport(sp.id)}
            />
          </li>
        ))}
      </ul>
    </Section>
  );
}
