// Heads-up view toolbar at the bottom of the viewport: fit, standard views, display style, section plane, ghosting
// of the design body and the density threshold.
import { type KeyboardEvent, type Ref, useEffect, useRef, useState } from 'react';
import { useStore } from '../state/store';
import type { StandardView, Viewport } from '../viewport/Viewport';
import { menuKeys, trapTab } from './a11y';
import { Icon, type IconName } from './icons';
import { type DisplayStyle, useUi } from './uiStore';

const VIEWS: { id: StandardView; label: string; key?: string }[] = [
  { id: 'iso', label: 'Isometric' },
  { id: 'front', label: 'Front' },
  { id: 'back', label: 'Back' },
  { id: 'left', label: 'Left' },
  { id: 'right', label: 'Right' },
  { id: 'top', label: 'Top' },
  { id: 'bottom', label: 'Bottom' },
];

const STYLES: { id: DisplayStyle; icon: IconName; label: string }[] = [
  { id: 'shaded', icon: 'shaded', label: 'Shaded' },
  { id: 'edges', icon: 'edges', label: 'Shaded with edges' },
  { id: 'wireframe', icon: 'wireframe', label: 'Wireframe' },
];

function HudBtn({
  icon,
  label,
  onClick,
  active,
  disabled,
  testId,
  popup,
  expanded,
  btnRef,
}: {
  icon: IconName;
  label: string;
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
  testId?: string;
  /** a button that opens a menu / dialog reports aria-haspopup + aria-expanded */
  popup?: 'menu' | 'dialog';
  expanded?: boolean;
  btnRef?: Ref<HTMLButtonElement>;
}) {
  return (
    <button
      ref={btnRef}
      type="button"
      className={`hud-btn ${active ? 'on' : ''}`}
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      aria-pressed={popup === 'menu' ? undefined : active}
      aria-haspopup={popup}
      aria-expanded={popup ? !!expanded : undefined}
      data-testid={testId}
    >
      <Icon name={icon} size={18} />
    </button>
  );
}

export function ViewToolbar({ vp }: { vp: Viewport }) {
  const display = useUi((s) => s.display);
  const section = useUi((s) => s.section);
  const { setDisplay, setSection } = useUi.getState();
  const ghost = useStore((s) => s.ghostDesign);
  const threshold = useStore((s) => s.threshold);
  const hasDensity = useStore((s) => !!s.run.densityFrame || !!s.resultStl);
  const hasDesign = useStore((s) => !!(s.project.design_mesh?.mesh_id && s.meshes[s.project.design_mesh.mesh_id]));
  const [open, setOpen] = useState<'views' | 'section' | null>(null);
  const viewsBtn = useRef<HTMLButtonElement>(null);
  const sectionBtn = useRef<HTMLButtonElement>(null);
  const pop = useRef<HTMLDivElement>(null);

  // focus goes into the menu / popover when it opens (first item, or the active axis)
  useEffect(() => {
    if (!open) return;
    const el = pop.current;
    (el?.querySelector<HTMLElement>('[role="menuitem"], .seg-btn.on') ?? el?.querySelector<HTMLElement>('button, input'))?.focus();
  }, [open]);
  /** Esc closes and returns focus to the button that opened it; Tab stays inside (arrows too in the menu). */
  const popKeys = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      const back = open === 'views' ? viewsBtn.current : sectionBtn.current;
      setOpen(null);
      back?.focus();
      return;
    }
    if ((open === 'views' && menuKeys(e, pop.current, true)) || trapTab(e, pop.current)) e.stopPropagation();
  };

  useEffect(() => {
    if (!open) return;
    const close = (e: PointerEvent) => {
      if (!(e.target instanceof Element) || !e.target.closest('.hud')) setOpen(null);
    };
    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [open]);

  return (
    <div className="hud" data-testid="view-toolbar" role="toolbar" aria-label="View">
      <HudBtn icon="fit" label="Fit all (F)" onClick={() => vp.fitAll()} testId="hud-fit" />
      <div className="hud-pop-wrap">
        <HudBtn
          icon="iso"
          label="Standard views"
          onClick={() => setOpen(open === 'views' ? null : 'views')}
          active={open === 'views'}
          popup="menu"
          expanded={open === 'views'}
          btnRef={viewsBtn}
          testId="hud-views"
        />
        {open === 'views' && (
          <div className="hud-pop hud-views" role="menu" aria-label="Standard views" ref={pop} onKeyDown={popKeys}>
            {VIEWS.map((v) => (
              <button
                key={v.id}
                type="button"
                role="menuitem"
                tabIndex={-1}
                className="hud-pop-item"
                data-testid={`hud-view-${v.id}`}
                onClick={() => {
                  vp.setView(v.id);
                  setOpen(null);
                  viewsBtn.current?.focus();
                }}
              >
                {v.label}
              </button>
            ))}
          </div>
        )}
      </div>
      <span className="hud-sep" />
      {STYLES.map((s) => (
        <HudBtn key={s.id} icon={s.icon} label={s.label} onClick={() => setDisplay(s.id)} active={display === s.id} testId={`hud-style-${s.id}`} />
      ))}
      <span className="hud-sep" />
      <div className="hud-pop-wrap">
        <HudBtn
          icon="section"
          label="Section plane"
          onClick={() => {
            if (!section.on) {
              setSection({ on: true });
              setOpen('section');
            } else setOpen(open === 'section' ? null : 'section');
          }}
          active={section.on}
          disabled={!hasDesign}
          popup="dialog"
          expanded={open === 'section'}
          btnRef={sectionBtn}
          testId="hud-section"
        />
        {open === 'section' && (
          <div className="hud-pop hud-section" data-testid="section-pop" role="dialog" aria-label="Section plane" ref={pop} onKeyDown={popKeys}>
            <div className="seg" role="group" aria-label="Section axis">
              {(['x', 'y', 'z'] as const).map((a) => (
                <button
                  key={a}
                  type="button"
                  className={`seg-btn ${section.axis === a ? 'on' : ''}`}
                  aria-pressed={section.axis === a}
                  aria-label={`${a.toUpperCase()} axis`}
                  onClick={() => setSection({ axis: a })}
                  data-testid={`section-axis-${a}`}
                >
                  {a.toUpperCase()}
                </button>
              ))}
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={section.t}
              aria-label="Section position"
              data-testid="section-pos"
              onChange={(e) => setSection({ t: e.target.valueAsNumber })}
            />
            <button
              type="button"
              className="seg-btn"
              onClick={() => {
                setSection({ on: false });
                setOpen(null);
                sectionBtn.current?.focus();
              }}
              data-testid="section-off"
            >
              Off
            </button>
          </div>
        )}
      </div>
      <HudBtn
        icon="ghost"
        label={hasDensity ? 'Ghost the design body over the results' : 'Ghost the design body (while results are shown)'}
        onClick={() => useStore.getState().setGhostDesign(!ghost)}
        active={ghost}
        testId="hud-ghost"
      />
      {hasDensity && (
        <label className="hud-threshold" title="Density threshold">
          <Icon name="density" size={16} />
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={threshold}
            aria-label="Density threshold"
            data-testid="hud-threshold"
            onChange={(e) => useStore.getState().setThreshold(e.target.valueAsNumber)}
          />
          <span className="num-out">{threshold.toFixed(2)}</span>
        </label>
      )}
    </div>
  );
}
