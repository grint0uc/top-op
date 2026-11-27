// UI-only state (theme, ribbon tab, which property page is open, viewport display options). Nothing here is part
// of the project document or reaches the server; the theme preference is a per-browser convenience.
import { create } from 'zustand';
import { type ActiveItem, useStore } from '../state/store';

export type ThemePref = 'system' | 'light' | 'dark';
export type RibbonTab = 'model' | 'bc' | 'domain' | 'optimize' | 'results';
/** Feature-tree groups; each opens one property page. An active item (load/support/ref/design) shows its own page. */
export type PageId = 'part' | 'refs' | 'loads' | 'supports' | 'domain' | 'study' | 'material' | 'results';
export type DisplayStyle = 'shaded' | 'edges' | 'wireframe';
export type Axis = 'x' | 'y' | 'z';

export interface SectionState {
  on: boolean;
  axis: Axis;
  /** plane position as a fraction of the content box along `axis` */
  t: number;
}

export interface ContextMenuState {
  x: number;
  y: number;
  items: { label: string; testId?: string; danger?: boolean; disabled?: boolean; run: () => void }[];
}

interface UiState {
  themePref: ThemePref;
  dark: boolean;
  tab: RibbonTab;
  page: PageId;
  display: DisplayStyle;
  section: SectionState;
  /** tree groups the user folded */
  folded: Partial<Record<PageId, boolean>>;
  /** item id being renamed inline in the tree */
  renaming: string | null;
  menu: ContextMenuState | null;
  setThemePref: (t: ThemePref) => void;
  setTab: (t: RibbonTab) => void;
  /** Open a property page: clears the active item (its page would win otherwise) and follows with the ribbon tab. */
  openPage: (p: PageId) => void;
  setDisplay: (d: DisplayStyle) => void;
  setSection: (patch: Partial<SectionState>) => void;
  toggleFold: (p: PageId) => void;
  setRenaming: (id: string | null) => void;
  setMenu: (m: ContextMenuState | null) => void;
}

const THEME_KEY = 'topop.ui.theme';

function readThemePref(): ThemePref {
  try {
    const v = localStorage.getItem(THEME_KEY);
    return v === 'light' || v === 'dark' ? v : 'system';
  } catch {
    return 'system';
  }
}

const media = typeof window !== 'undefined' && window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
const resolveDark = (pref: ThemePref): boolean => (pref === 'system' ? !!media?.matches : pref === 'dark');

export const PAGE_TAB: Record<PageId, RibbonTab> = {
  part: 'model',
  refs: 'domain',
  loads: 'bc',
  supports: 'bc',
  domain: 'domain',
  study: 'optimize',
  material: 'optimize',
  results: 'results',
};

export const TAB_PAGE: Record<RibbonTab, PageId> = {
  model: 'part',
  bc: 'loads',
  domain: 'domain',
  optimize: 'study',
  results: 'results',
};

export const itemPage = (a: NonNullable<ActiveItem>): PageId =>
  a.kind === 'load' ? 'loads' : a.kind === 'support' ? 'supports' : a.kind === 'ref' ? 'refs' : 'part';

const initialPref = readThemePref();

export const useUi = create<UiState>()((set) => ({
  themePref: initialPref,
  dark: resolveDark(initialPref),
  tab: 'model',
  page: 'part',
  display: 'edges',
  section: { on: false, axis: 'x', t: 0.5 },
  folded: {},
  renaming: null,
  menu: null,
  setThemePref: (themePref) => {
    try {
      if (themePref === 'system') localStorage.removeItem(THEME_KEY);
      else localStorage.setItem(THEME_KEY, themePref);
    } catch {
      // storage blocked: the choice lasts for this page only
    }
    set({ themePref, dark: resolveDark(themePref) });
  },
  setTab: (tab) => set({ tab }),
  openPage: (page) => {
    if (useStore.getState().activeItem) useStore.getState().setActiveItem(null);
    set({ page, tab: PAGE_TAB[page] });
  },
  setDisplay: (display) => set({ display }),
  setSection: (patch) => set((s) => ({ section: { ...s.section, ...patch } })),
  toggleFold: (p) => set((s) => ({ folded: { ...s.folded, [p]: !s.folded[p] } })),
  setRenaming: (renaming) => set({ renaming }),
  setMenu: (menu) => set({ menu }),
}));

media?.addEventListener?.('change', () => {
  const s = useUi.getState();
  if (s.themePref === 'system') useUi.setState({ dark: media.matches });
});

// An item that becomes active (added, clicked in the tree, picked for the gizmo) opens its page and the ribbon tab of
// its stage; when it is released (Esc, deleted) the page of its group stays, so the user lands on the list it belonged to.
useStore.subscribe((s, prev) => {
  if (s.activeItem && s.activeItem !== prev.activeItem) {
    const page = itemPage(s.activeItem);
    const ui = useUi.getState();
    if (ui.page !== page || ui.tab !== PAGE_TAB[page]) useUi.setState({ page, tab: PAGE_TAB[page] });
  }
});

/** Applies the resolved theme to <html data-theme>, so the CSS tokens switch. */
export function installTheme(): void {
  const apply = (dark: boolean) => {
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  };
  apply(useUi.getState().dark);
  useUi.subscribe((s, p) => {
    if (s.dark !== p.dark) apply(s.dark);
  });
}
