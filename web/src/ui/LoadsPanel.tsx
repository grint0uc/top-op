import type { CSSProperties } from 'react';
import type { LoadSpec } from '../api/client';
import { addLoadFromSelection } from '../state/actions';
import { cssHex, hasSelection, selectionSummary } from '../state/derived';
import { useStore } from '../state/store';
import { Btn, Field, Group, NumField } from './controls';
import { Icon } from './icons';
import { ItemActions, Swatch, caseColor, selectionLabel } from './ItemRows';

/** Loads group: the list (click to edit) and the command to create one from the current selection. */
export function LoadsPage() {
  const loads = useStore((s) => s.project.loads);
  const hasSel = useStore((s) => hasSelection(s));
  const summary = useStore((s) => selectionSummary(s));
  const { setActiveItem } = useStore.getState();
  return (
    <>
      <Group title="New load" icon="force">
        <p className="dim">Select faces (pick / paint), a query or a region primitive, then create the load from it.</p>
        <div className="row">
          <Btn variant="primary" icon="force" onClick={() => addLoadFromSelection()} disabled={!hasSel} testId="pm-add-load">
            Force from selection
          </Btn>
          <span className="dim ellipsis grow">{summary}</span>
        </div>
      </Group>
      <Group title="Loads" icon="list" aside={<span className="count">{loads.length}</span>}>
        {loads.length === 0 && <p className="dim">No loads yet.</p>}
        <ul className="pm-list">
          {loads.map((l) => (
            <li key={l.id}>
              <button type="button" className="pm-list-row" onClick={() => setActiveItem({ kind: 'load', id: l.id })}>
                <Swatch color={caseColor(l.case)} />
                <span className="grow ellipsis">{l.name || l.id}</span>
                <span className="dim num-out">{forceLabel(l)}</span>
              </button>
            </li>
          ))}
        </ul>
      </Group>
    </>
  );
}

export const forceLabel = (l: LoadSpec): string => `|F| ${+Math.hypot(...l.force).toPrecision(4)}`;

/** One load: name, load case, force vector and its selection. */
export function LoadPage({ load: l }: { load: LoadSpec }) {
  const { updateLoad, removeLoad } = useStore.getState();
  return (
    <>
      <Group title="Load" icon="force">
        <Field label="Name">
          <input type="text" className="text" value={l.name} data-testid="load-name" onChange={(e) => updateLoad(l.id, { name: e.target.value })} />
        </Field>
        <div className="row">
          <Field label="Load case" hint="Loads with the same case act together; cases are optimized for the sum of compliances">
            <NumField value={l.case} min={0} step={1} testId="load-case" onChange={(v) => updateLoad(l.id, { case: Math.round(v) })} />
          </Field>
          {/* tinted chip with the case colour as its border: dark text reads on every case colour (WCAG 1.4.3) */}
          <span className="case-chip" style={{ '--case': cssHex(caseColor(l.case)) } as CSSProperties} title="Case colour in the viewport">
            C{l.case}
          </span>
        </div>
      </Group>
      <Group title="Force" icon="force" aside={<span className="dim num-out">{forceLabel(l)}</span>}>
        <div className="vec3 labelled">
          {(['x', 'y', 'z'] as const).map((axis, k) => (
            <Field key={axis} label={`F${axis}`}>
              <NumField
                value={l.force[k] ?? 0}
                testId={`force-${axis}`}
                onChange={(v) => updateLoad(l.id, { force: l.force.map((f, i) => (i === k ? v : f)) })}
              />
            </Field>
          ))}
        </div>
        <p className="dim">Total force on the region, spread over its grid nodes.</p>
      </Group>
      <Group title="Region" icon="target">
        <p className="region-label">
          <Icon name="target" size={14} /> {selectionLabel(l.selection)}
        </p>
        <ItemActions
          selection={l.selection}
          label={l.name || l.id}
          onReplace={(selection) => updateLoad(l.id, { selection })}
          onDelete={() => removeLoad(l.id)}
        />
      </Group>
    </>
  );
}
