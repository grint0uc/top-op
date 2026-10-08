import type { Selection } from '../api/client';
import { reselect, resolvePreview } from '../state/actions';
import { caseColor, cssHex, SUPPORT_COLOR } from '../state/derived';
import { queryLabel } from '../state/query';
import { currentSelectionSpec, useStore } from '../state/store';
import { Btn } from './controls';

/** One-line description of a selection for load/support rows ("normal +Z \u00b110\u00b0 in box", "plane z=60", "facet #0 (5\u00b0)"). */
export function selectionLabel(sel: Selection): string {
  switch (sel.kind) {
    case 'faces':
      return `${sel.face_ids.length} face${sel.face_ids.length === 1 ? '' : 's'}`;
    case 'facets':
    case 'normal':
    case 'plane':
      return queryLabel(sel);
    default:
      return `${sel.kind} primitive`;
  }
}

export function Swatch({ color }: { color: number }) {
  return <span className="swatch" style={{ background: cssHex(color) }} />;
}

export { SUPPORT_COLOR, caseColor };

/** Shared buttons of a load/support row: select, preview, replace selection, delete. */
export function ItemActions({
  selection,
  label,
  onReplace,
  onDelete,
}: {
  selection: Selection;
  /** the item's name, shown next to the preview points */
  label?: string;
  onReplace: (sel: Selection) => void;
  onDelete: () => void;
}) {
  const hasCurrent = useStore((s) => currentSelectionSpec(s) !== null);
  return (
    <div className="row wrap" onClick={(e) => e.stopPropagation()}>
      <Btn onClick={() => reselect(selection)} testId="item-select" title="Show this selection in the viewport (query selections open the Query tab)">
        Select
      </Btn>
      <Btn onClick={() => void resolvePreview(selection, label)} testId="item-preview" title="Resolve to grid nodes and show them as points">
        Preview
      </Btn>
      <Btn
        onClick={() => {
          const sel = currentSelectionSpec(useStore.getState());
          if (sel) onReplace(sel);
        }}
        disabled={!hasCurrent}
        testId="item-replace"
        title="Replace this item's selection with the current selection"
      >
        Use current
      </Btn>
      <Btn onClick={onDelete} variant="danger" testId="item-delete">
        Delete
      </Btn>
    </div>
  );
}
