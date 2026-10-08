import type { Selection } from '../api/client';
import { reselect, resolvePreview } from '../state/actions';
import { caseColor, cssHex, SUPPORT_COLOR } from '../state/derived';
import { currentSelectionSpec, useStore } from '../state/store';
import { Btn } from './controls';

export function selectionLabel(sel: Selection): string {
  switch (sel.kind) {
    case 'faces':
      return `${sel.face_ids.length} face${sel.face_ids.length === 1 ? '' : 's'}`;
    case 'facets':
      return `${sel.facet_ids.length} facet(s)`;
    case 'normal':
      return `normal [${sel.direction.map((v) => +v.toFixed(2)).join(', ')}]`;
    case 'plane':
      return 'plane';
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
  onReplace,
  onDelete,
}: {
  selection: Selection;
  onReplace: (sel: Selection) => void;
  onDelete: () => void;
}) {
  const hasCurrent = useStore((s) => currentSelectionSpec(s) !== null);
  return (
    <div className="row wrap" onClick={(e) => e.stopPropagation()}>
      <Btn onClick={() => reselect(selection)} testId="item-select" title="Show this selection in the viewport">
        Select
      </Btn>
      <Btn onClick={() => void resolvePreview(selection)} testId="item-preview" title="Resolve to grid nodes and show them as points">
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
