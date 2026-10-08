import { type ReactNode, useEffect, useRef, useState } from 'react';

/** Collapsible sidebar section. Children stay mounted when collapsed so local input state survives. */
export function Section({ title, children, defaultOpen = true }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="panel" data-testid={`panel-${title.toLowerCase().replace(/\s+/g, '-')}`}>
      <h2>
        <button type="button" className="panel-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
          {title}
        </button>
      </h2>
      <div className="panel-body" hidden={!open}>
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
}

/** Number input that keeps its own text while focused, so typing "0." or "-" does not fight the store. */
export function NumField({ value, onChange, min, max, step = 'any', disabled, testId, title, className }: NumFieldProps) {
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

export function Banner({ kind, children, testId }: { kind: 'info' | 'warn' | 'error'; children: ReactNode; testId?: string }) {
  return (
    <div className={`banner banner-${kind}`} role={kind === 'error' ? 'alert' : 'status'} data-testid={testId}>
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
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  testId?: string;
  variant?: 'primary' | 'danger';
  title?: string;
  active?: boolean;
}) {
  return (
    <button
      type="button"
      className={`btn ${variant ?? ''} ${active ? 'active' : ''}`}
      onClick={onClick}
      disabled={disabled}
      data-testid={testId}
      title={title}
      aria-pressed={active}
    >
      {children}
    </button>
  );
}
