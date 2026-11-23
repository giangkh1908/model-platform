// Các thành phần giao diện nhỏ dùng ở nhiều trang.
import { useState } from "react";
import { Check, Copy, X } from "lucide-react";
import { clamp01, copyText } from "../lib.js";
import { useApp } from "../store.jsx";

export function Card({ title, actions, className = "", children }) {
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

export function Chip({ kind = "", title, children }) {
  return (
    <span className={`chip ${kind}`} title={title}>
      {children}
    </span>
  );
}

export function StatusDot({ state }) {
  return <span className={`dot ${state}`} />;
}

export function Bar({ fraction, hot = 1.1, kind = "" }) {
  const f = clamp01(fraction);
  const className = f >= hot ? "hot" : kind;
  return (
    <div className="bar">
      <div className={className} style={{ width: `${f * 100}%` }} />
    </div>
  );
}

export function Meter({ label, value, fraction, hot = 0.85 }) {
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

export function Sparkline({ values, max, height = 36, className = "" }) {
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

export function CopyButton({ text, label = "Sao chép", small = true }) {
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

export function Empty({ icon: Icon, title, children }) {
  return (
    <div className="empty">
      {Icon && <Icon size={28} strokeWidth={1.5} />}
      <div className="empty-title">{title}</div>
      {children && <div className="muted">{children}</div>}
    </div>
  );
}

export function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export function Segmented({ value, options, onChange }) {
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

// Danh sách nhãn kèm thanh giá trị; `top` được tô màu nổi bật. `format` đổi giá trị thành chữ.
export function ScoreBars({ rows, top, max = 1, format }) {
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
