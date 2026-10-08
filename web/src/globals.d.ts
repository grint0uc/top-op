import type { useStore } from './state/store';
import type { Viewport } from './viewport/Viewport';

declare global {
  interface Window {
    /** zustand store, exposed in dev/mock builds so Playwright can assert state. */
    __topop?: typeof useStore;
    /** the live Viewport instance (dev/mock builds only). */
    __topopViewport?: Viewport;
  }
}
