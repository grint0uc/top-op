import { useEffect } from 'react';
import { deleteActive } from '../state/actions';
import { type ToolMode, useStore } from '../state/store';
import { getViewport } from '../viewport/mount';
import { useUi } from './uiStore';

const MODE_KEYS: Record<string, ToolMode> = { '1': 'orbit', '2': 'pick', '3': 'paint', '4': 'gizmo', '5': 'query' };
const GIZMO_KEYS = { g: 'translate', r: 'rotate', s: 'scale' } as const;
const TEXT_INPUT = new Set(['text', 'number', 'search', 'email', 'url', 'password', 'tel']);

function isTyping(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  if (t.isContentEditable || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT') return true;
  return t instanceof HTMLInputElement && TEXT_INPUT.has(t.type);
}

/** 1-5 modes, g/r/s gizmo mode, f fit all, Esc clears the selection, Delete removes the active load/support/reference. */
export function useHotkeys(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target)) return;
      const s = useStore.getState();
      if (e.key === 'Escape') {
        if (useUi.getState().menu) return; // the context menu takes this Esc
        s.clearSelection();
        s.setPreview(null);
        s.setActiveItem(null);
        return;
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (s.activeItem) {
          e.preventDefault();
          deleteActive();
        }
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const mode = MODE_KEYS[e.key];
      if (mode) {
        s.setTool(mode);
        return;
      }
      const g = GIZMO_KEYS[e.key.toLowerCase() as keyof typeof GIZMO_KEYS];
      if (g) s.setGizmoMode(g);
      else if (e.key.toLowerCase() === 'f') getViewport()?.fitAll();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}
