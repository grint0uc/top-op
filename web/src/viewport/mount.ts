// Creates the single Viewport for the canvas and binds it to the store. Idempotent per canvas so React
// StrictMode's double effect does not tear down a live WebGL context.
import { ensureAdjacency } from '../state/actions';
import { useStore } from '../state/store';
import { bindViewport } from './bind';
import { Viewport } from './Viewport';

let current: { canvas: HTMLCanvasElement; vp: Viewport; unbind: () => void } | null = null;

export function mountViewport(canvas: HTMLCanvasElement): Viewport {
  if (current?.canvas === canvas) return current.vp;
  teardown();
  const get = useStore.getState;
  const vp = new Viewport(canvas, {
    picker: {
      getSelection: () => get().selection.faceIds,
      setSelection: (ids) => get().setFaceSelection(ids),
      getGrowAngle: () => get().growAngleDeg,
      ensureAdjacency,
    },
    painter: {
      getSelection: () => get().selection.faceIds,
      setSelection: (ids) => get().setFaceSelection(ids),
      getRadius: () => get().brushRadius,
    },
  });
  const unbind = bindViewport(vp);
  current = { canvas, vp, unbind };
  if (import.meta.env.DEV || import.meta.env.VITE_MOCK === '1') window.__topopViewport = vp;
  return vp;
}

function teardown(): void {
  if (!current) return;
  current.unbind();
  current.vp.dispose();
  current = null;
}

if (import.meta.hot) import.meta.hot.dispose(teardown);
