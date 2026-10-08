// Project document <-> localStorage. Mesh buffers are never stored: after a reload the user re-uploads.
import { DEFAULT_GRID, DEFAULT_MATERIAL, DEFAULT_PARAMS } from './defaults';
import type { ProjectDoc, State } from './store';

const KEY = 'topop.project.v1';

export interface Persisted {
  project: ProjectDoc;
  meshMemo: State['meshMemo'];
}

export function loadPersisted(): Persisted | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const p = JSON.parse(raw) as Partial<Persisted>;
    const doc = p.project;
    if (!doc || typeof doc !== 'object') return null;
    return {
      project: {
        name: typeof doc.name === 'string' ? doc.name : 'untitled',
        design_mesh: doc.design_mesh ?? null,
        ref_models: Array.isArray(doc.ref_models) ? doc.ref_models : [],
        grid: { ...DEFAULT_GRID, ...doc.grid },
        material: { ...DEFAULT_MATERIAL, ...doc.material },
        params: { ...DEFAULT_PARAMS, ...doc.params },
        loads: Array.isArray(doc.loads) ? doc.loads : [],
        supports: Array.isArray(doc.supports) ? doc.supports : [],
      },
      meshMemo: p.meshMemo ?? {},
    };
  } catch {
    return null; // storage blocked or corrupt: start fresh
  }
}

interface Subscribable {
  getState: () => State;
  subscribe: (l: (s: State, prev: State) => void) => () => void;
}

/** Writes the document on change (debounced). Returns an unsubscribe function. */
export function installPersistence(store: Subscribable): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const write = () => {
    try {
      const { project, meshMemo } = store.getState();
      localStorage.setItem(KEY, JSON.stringify({ project, meshMemo }));
    } catch {
      // quota/blocked: persistence is best-effort
    }
  };
  const unsub = store.subscribe((s, prev) => {
    if (s.project === prev.project && s.meshMemo === prev.meshMemo) return;
    clearTimeout(timer);
    timer = setTimeout(write, 250);
  });
  return () => {
    clearTimeout(timer);
    unsub();
  };
}

export function clearPersisted(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}
