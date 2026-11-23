// Các thành phần giao diện nhỏ dùng ở nhiều trang.
import { useState, type ReactNode } from "react";
import { Check, Copy, X, type LucideIcon } from "lucide-react";
import { copyText } from "../lib/browser.ts";
import { clamp01 } from "../lib/format.ts";
import { useApp } from "../store.tsx";

export function Card({ title, actions, className = "", children }: { title?: ReactNode; actions?: ReactNode; className?: string; children?: ReactNode }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <div className="card-head">
          {title && <h2>{title}</h2>}
          {actions && <div className="card-actions">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Chip({ kind = "", title, children }: { kind?: string; title?: string; children?: ReactNode }) {
  return (
    <span className={`chip ${kind}`} title={title}>
      {children}
    </span>
  );
}

export function StatusDot({ state }: { state: string }) {
  return <span className={`dot ${state}`} />;
}

export function Bar({ fraction, hot = 1.1, kind = "" }: { fraction: number | null | undefined; hot?: number; kind?: string }) {
  const f = clamp01(fraction);
  const className = f >= hot ? "hot" : kind;
  return (
    <div className="bar">
      <div className={className} style={{ width: `${f * 100}%` }} />
    </div>
  );
}

export function Meter({ label, value, fraction, hot = 0.85 }: { label: ReactNode; value: ReactNode; fraction: number | null | undefined; hot?: number }) {
  return (
    <div className="meter">
      <div className="meter-top">
        <span>{label}</span>
        <span className="num">{value}</span>
      </div>
      <Bar fraction={fraction} hot={hot} />
    </div>
  );
}

export function Sparkline({ values, max, height = 36, className = "" }: { values: number[]; max?: number; height?: number; className?: string }) {
  if (!values.length) return <div className="spark-empty" style={{ height }} />;
  const top = max ?? Math.max(...values, 1);
  const width = 100;
  const step = values.length > 1 ? width / (values.length - 1) : width;
  const points = values.map((v, i) => `${(i * step).toFixed(2)},${(height - (clamp01(v / top) * (height - 2) + 1)).toFixed(2)}`);
  const line = points.join(" ");
  const area = `0,${height} ${line} ${width},${height}`;
  return (
    <svg className={`spark ${className}`} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ height }}>
      <polygon points={area} className="spark-area" />
      <polyline points={line} className="spark-line" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function CopyButton({ text, label = "Sao chép", small = true }: { text: string; label?: string; small?: boolean }) {
  const [done, setDone] = useState(false);
  async function onClick() {
    if (await copyText(text)) {
      setDone(true);
      setTimeout(() => setDone(false), 1500);
    }
  }
  return (
    <button type="button" className={`btn ghost ${small ? "sm" : ""}`} onClick={onClick} title={label}>
      {done ? <Check size={14} /> : <Copy size={14} />}
      {label && <span>{done ? "Đã chép" : label}</span>}
    </button>
  );
}

export function Empty({ icon: Icon, title, children }: { icon?: LucideIcon; title: ReactNode; children?: ReactNode }) {
  return (
    <div className="empty">
      {Icon && <Icon size={28} strokeWidth={1.5} />}
      <div className="empty-title">{title}</div>
      {children && <div className="muted">{children}</div>}
    </div>
  );
}

export function Field({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export interface SegmentOption<T extends string> {
  id: T;
  label: ReactNode;
}

export function Segmented<T extends string>({ value, options, onChange }: { value: T; options: readonly SegmentOption<T>[]; onChange: (id: T) => void }) {
  return (
    <div className="segmented" role="tablist">
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          role="tab"
          aria-selected={value === o.id}
          className={value === o.id ? "active" : ""}
          onClick={() => onChange(o.id)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toasts() {
  const { toasts, dismiss } = useApp();
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`}>
          <span>{t.text}</span>
          <button type="button" className="icon-btn" onClick={() => dismiss(t.id)} aria-label="Đóng">
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}

export function ConfirmDialog() {
  const { confirmState, closeConfirm } = useApp();
  if (!confirmState) return null;
  const { title, message, confirmLabel = "Đồng ý", danger } = confirmState;
  return (
    <div className="modal-backdrop" onClick={() => closeConfirm(false)}>
      <div className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <h3>{title}</h3>
        {message && <p className="muted">{message}</p>}
        <div className="row end">
          <button type="button" className="btn" onClick={() => closeConfirm(false)}>
            Huỷ
          </button>
          <button type="button" className={`btn ${danger ? "danger-solid" : "primary"}`} onClick={() => closeConfirm(true)} autoFocus>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

/** Danh sách nhãn kèm thanh giá trị; `top` được tô màu nổi bật. `format` đổi giá trị thành chữ. */
export function ScoreBars({ rows, top, max = 1, format }: { rows: [string, number][]; top?: string; max?: number; format: (value: number) => string }) {
  return (
    <div className="score-bars">
      {rows.map(([label, value]) => (
        <div key={label} className={`score-row ${label === top ? "top" : ""}`}>
          <span className="score-label" title={label}>
            {label}
          </span>
          <Bar fraction={value / max} kind={label === top ? "ok" : ""} />
          <span className="num muted">{format(value)}</span>
        </div>
      ))}
    </div>
  );
}

export function Spinner() {
  return <span className="spinner" aria-hidden="true" />;
}
