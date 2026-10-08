import { addLoadFromSelection } from '../state/actions';
import { hasSelection, selectionSummary } from '../state/derived';
import { useStore } from '../state/store';
import { Btn, Field, NumField, Section } from './controls';
import { ItemActions, Swatch, caseColor, selectionLabel } from './ItemRows';

export function LoadsPanel() {
  const loads = useStore((s) => s.project.loads);
  const active = useStore((s) => s.activeItem);
  const hasSel = useStore((s) => hasSelection(s));
  const summary = useStore((s) => selectionSummary(s));
  const { updateLoad, removeLoad, setActiveItem } = useStore.getState();

  return (
    <Section title="Loads">
      <div className="row">
        <Btn variant="primary" onClick={() => addLoadFromSelection()} disabled={!hasSel} testId="add-load">
          Add from selection
        </Btn>
        <span className="dim">{summary}</span>
      </div>
      {loads.length === 0 && <p className="dim">No loads. Select faces, run a query or add a primitive, then add.</p>}
      <ul className="items">
        {loads.map((l) => (
          <li
            key={l.id}
            className={`item ${active?.kind === 'load' && active.id === l.id ? 'active' : ''}`}
            data-testid="load-row"
            onClick={() => setActiveItem({ kind: 'load', id: l.id })}
          >
            <div className="row">
              <Swatch color={caseColor(l.case)} />
              <input
                type="text"
                className="text grow"
                value={l.name}
                data-testid="load-name"
                onChange={(e) => updateLoad(l.id, { name: e.target.value })}
              />
              <span className="dim">{selectionLabel(l.selection)}</span>
            </div>
            <div className="row">
              {(['x', 'y', 'z'] as const).map((axis, k) => (
                <Field key={axis} label={`F${axis}`}>
                  <NumField
                    value={l.force[k] ?? 0}
                    testId={`force-${axis}`}
                    onChange={(v) => updateLoad(l.id, { force: l.force.map((f, i) => (i === k ? v : f)) })}
                  />
                </Field>
              ))}
              <Field label="Case">
                <NumField value={l.case} min={0} step={1} testId="load-case" onChange={(v) => updateLoad(l.id, { case: Math.round(v) })} />
              </Field>
            </div>
            <ItemActions
              selection={l.selection}
              label={l.name || l.id}
              onReplace={(selection) => updateLoad(l.id, { selection })}
              onDelete={() => removeLoad(l.id)}
            />
          </li>
        ))}
      </ul>
    </Section>
  );
}
