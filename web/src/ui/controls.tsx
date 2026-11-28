import { type ReactNode, type Ref, useEffect, useRef, useState } from 'react';
import { Icon, type IconName } from './icons';

/** Property-manager group box (collapsible). Children stay mounted when folded so local input state survives. */
export function Group({
  title,
  children,
  defaultOpen = true,
  testId,
  icon,
  aside,
}: {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
  testId?: string;
  icon?: IconName;
  /** extra content on the right of the header (a count, a tag) */
  aside?: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="group" data-testid={testId}>
      <h3 className="group-head">
        <button type="button" className="group-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
          <Icon name="chevron" size={12} className="caret" />
          {icon && <Icon name={icon} size={14} />}
          <span className="grow">{title}</span>
        </button>
        {aside}
      </h3>
      <div className="group-body" hidden={!open}>
        {children}
      </div>
    </section>
  );
}

const fmt = (v: number): string => (Number.isFinite(v) ? String(Math.round(v * 1e6) / 1e6) : '');

interface NumFieldProps {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number | 'any';
  disabled?: boolean;
  testId?: string;
  title?: string;
  className?: string;
  /** accessible name when the field has no wrapping <Field> label */
  label?: string;
}

/** Number input that keeps its own text while focused, so typing "0." or "-" does not fight the store. */
export function NumField({ value, onChange, min, max, step = 'any', disabled, testId, title, className, label }: NumFieldProps) {
  const [text, setText] = useState(fmt(value));
  const focused = useRef(false);
  useEffect(() => {
    if (!focused.current) setText(fmt(value));
  }, [value]);
  return (
    <input
      type="number"
      className={`num ${className ?? ''}`}
      value={text}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      title={title}
      aria-label={label}
      data-testid={testId}
      onFocus={() => (focused.current = true)}
      onBlur={() => {
        focused.current = false;
        setText(fmt(value));
      }}
      onChange={(e) => {
        setText(e.target.value);
        const n = e.target.valueAsNumber;
        if (!Number.isFinite(n)) return;
        if ((min !== undefined && n < min) || (max !== undefined && n > max)) return;
        onChange(n);
      }}
    />
  );
}

interface OptNumFieldProps {
  value: number | null;
  onChange: (v: number | null) => void;
  placeholder?: string;
  testId?: string;
  title?: string;
}

/** Positive number where an empty box means "off" (null), e.g. the stress limit. */
export function OptNumField({ value, onChange, placeholder, testId, title }: OptNumFieldProps) {
  const show = (v: number | null) => (v === null ? '' : fmt(v));
  const [text, setText] = useState(show(value));
  const focused = useRef(false);
  useEffect(() => {
    if (!focused.current) setText(show(value));
  }, [value]);
  return (
    <input
      type="number"
      className="num"
      value={text}
      min={0}
      step="any"
      placeholder={placeholder}
      title={title}
      data-testid={testId}
      onFocus={() => (focused.current = true)}
      onBlur={() => {
        focused.current = false;
        setText(show(value));
      }}
      onChange={(e) => {
        setText(e.target.value);
        if (e.target.value.trim() === '') {
          onChange(null);
          return;
        }
        const n = e.target.valueAsNumber;
        if (Number.isFinite(n) && n > 0) onChange(n);
      }}
    />
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="field" title={hint}>
      <span className="field-label">{label}</span>
      {children}
    </label>
  );
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  testId?: string;
  disabled?: boolean;
}

export function Slider({ label, value, min, max, step, onChange, format, testId, disabled }: SliderProps) {
  return (
    <label className="field slider">
      <span className="field-label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={Math.min(max, Math.max(min, value))}
        disabled={disabled}
        data-testid={testId}
        onChange={(e) => onChange(e.target.valueAsNumber)}
      />
      <span className="slider-value">{format ? format(value) : fmt(value)}</span>
    </label>
  );
}

/** `live={false}` when the text is announced elsewhere (the top notice goes through the Announcer). */
export function Banner({ kind, children, testId, live = true }: { kind: 'info' | 'warn' | 'error'; children: ReactNode; testId?: string; live?: boolean }) {
  return (
    <div className={`banner banner-${kind}`} role={live ? (kind === 'error' ? 'alert' : 'status') : undefined} data-testid={testId}>
      {children}
    </div>
  );
}

export function Btn({
  children,
  onClick,
  disabled,
  testId,
  variant,
  title,
  active,
  icon,
  btnRef,
}: {
  btnRef?: Ref<HTMLButtonElement>;
  children?: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  testId?: string;
  variant?: 'primary' | 'danger' | 'ghost';
  title?: string;
  active?: boolean;
  icon?: IconName;
}) {
  return (
    <button
      ref={btnRef}
      type="button"
      className={`btn ${variant ?? ''} ${active ? 'active' : ''} ${children ? '' : 'icon-only'}`}
      onClick={onClick}
      disabled={disabled}
      data-testid={testId}
      title={title}
      aria-label={children ? undefined : title}
      aria-pressed={active}
    >
      {icon && <Icon name={icon} size={14} />}
      {children}
    </button>
  );
}

/** Read-only key/value rows of a property page. */
export function Props({ children, testId, className }: { children: ReactNode; testId?: string; className?: string }) {
  return (
    <dl className={`kv ${className ?? ''}`} data-testid={testId}>
      {children}
    </dl>
  );
}
