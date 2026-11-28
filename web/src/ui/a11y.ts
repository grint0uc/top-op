// Keyboard helpers shared by the menus, popovers and composite widgets (WCAG 2.1.1 / 2.4.3).
import { type KeyboardEvent, type RefObject, useEffect, useRef } from 'react';

const FOCUSABLE = 'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex]';

/** Focusable descendants in DOM order (menu items with tabindex=-1 included, hidden ones left out). */
export function focusables(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((el) => !el.hidden && el.getClientRects().length > 0);
}

/** Tab / Shift+Tab cycle inside `root` instead of leaving it (Esc is the way out). Returns true when handled. */
export function trapTab(e: KeyboardEvent, root: HTMLElement | null): boolean {
  if (e.key !== 'Tab' || !root) return false;
  const els = focusables(root);
  if (els.length === 0) return false;
  const i = els.indexOf(document.activeElement as HTMLElement);
  const next = e.shiftKey ? (i <= 0 ? els.length - 1 : i - 1) : i === els.length - 1 ? 0 : i + 1;
  e.preventDefault();
  els[next]!.focus();
  return true;
}

/** Arrow keys / Home / End move between the menu items of `root` (role=menuitem, enabled). Returns true when handled. */
export function menuKeys(e: KeyboardEvent, root: HTMLElement | null, horizontal = false): boolean {
  if (!root) return false;
  const items = [...root.querySelectorAll<HTMLElement>('[role="menuitem"]:not(:disabled)')];
  if (items.length === 0) return false;
  const i = items.indexOf(document.activeElement as HTMLElement);
  const fwd = horizontal ? ['ArrowDown', 'ArrowRight'] : ['ArrowDown'];
  const back = horizontal ? ['ArrowUp', 'ArrowLeft'] : ['ArrowUp'];
  let next: number;
  if (fwd.includes(e.key)) next = (i + 1) % items.length;
  else if (back.includes(e.key)) next = (i - 1 + items.length) % items.length;
  else if (e.key === 'Home') next = 0;
  else if (e.key === 'End') next = items.length - 1;
  else return false;
  e.preventDefault();
  items[next]!.focus();
  return true;
}

/**
 * Run / Stop pairs: the pressed button turns disabled and the browser drops its focus to <body> (2.4.3). When `swap`
 * flips and the focus was on the pair, it moves to the button that is enabled now. Nothing happens on mount.
 */
export function useFocusHandoff(swap: boolean, from: RefObject<HTMLElement | null>, to: RefObject<HTMLElement | null>): void {
  const had = useRef(false);
  const prev = useRef(swap);
  useEffect(() => {
    const els = [from.current, to.current].filter((e): e is HTMLElement => !!e);
    const on = () => (had.current = true);
    // focus going to another element is a choice; going nowhere (disabled, a click on the canvas) is not
    const off = (e: FocusEvent) => {
      if (e.relatedTarget) had.current = false;
    };
    els.forEach((e) => {
      e.addEventListener('focusin', on);
      e.addEventListener('focusout', off);
    });
    return () =>
      els.forEach((e) => {
        e.removeEventListener('focusin', on);
        e.removeEventListener('focusout', off);
      });
  }, [from, to]);
  useEffect(() => {
    if (prev.current === swap) return;
    prev.current = swap;
    const [lost, next] = swap ? [from.current, to.current] : [to.current, from.current];
    const a = document.activeElement;
    if (next && had.current && (a === lost || a === document.body || a === null)) next.focus();
  }, [swap, from, to]);
}
