// Biểu đồ đường dùng chung cho mọi chỉ số fine-tune: tối đa 8 chuỗi, một trục y, rê chuột để xem giá trị.
import { useEffect, useRef, useState } from "react";
import { formatMetric } from "../lib.js";

const HEIGHT = 180;
const PAD = { top: 10, right: 12, bottom: 24, left: 48 };
// Tối đa 8 chuỗi, màu theo thứ tự cố định của bảng màu (không xoay vòng): chuỗi thứ 9 trở đi không vẽ.
const MAX_SERIES = 8;
const SERIES_CLASS = Array.from({ length: MAX_SERIES }, (_, i) => `s${i + 1}`);
const SERIES_COLOR = SERIES_CLASS.map((_, i) => `var(--series-${i + 1})`);
// Chuỗi ít điểm (vd. eval vài lần) vẽ thêm chấm để thấy từng lần đo.
const DOT_MAX_POINTS = 40;

function useWidth() {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!ref.current) return undefined;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

// Các mốc "đẹp" (1, 2, 5 × 10^n) phủ khoảng [lo, hi].
function niceTicks(lo, hi, count) {
  if (hi <= lo) return [lo];
  const raw = (hi - lo) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw);
  // Mốc cuối làm tròn lên quá giá trị lớn nhất, để đỉnh của đường luôn có mốc phía trên.
  const top = Math.ceil(hi / step - 1e-9) * step;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= top + step * 1e-9; v += step) ticks.push(Number(v.toPrecision(12)));
  return ticks;
}

function domainOf(series) {
  const xs = [];
  const ys = [];
  for (const s of series) {
    for (const [x, y] of s.points) {
      xs.push(x);
      ys.push(y);
    }
  }
  const yMax = Math.max(...ys);
  const yMin = Math.min(...ys);
  // Bắt đầu từ 0 khi mọi giá trị không âm, để độ cao đường phản ánh đúng độ lớn.
  const lo = yMin >= 0 ? 0 : yMin;
  const hi = yMax === lo ? lo + 1 : yMax;
  return { x0: Math.min(...xs), x1: Math.max(...xs), y0: lo, y1: hi };
}

// Điểm có x gần `x` nhất của một chuỗi.
function nearest(points, x) {
  let best = null;
  for (const p of points) {
    if (best === null || Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p;
  }
  return best;
}

// `dots`: vẽ chấm tại từng điểm (khi chuỗi ít điểm); tắt cho đường cong dày như đường ngưỡng.
// Một chuỗi có thể tự đặt `dots` riêng (vd. eval có chấm, train thì không).
export default function LineChart({ title, hint, series: allSeries, unit = "", xLabel = "Bước", formatX, dots = true }) {
  const series = allSeries.slice(0, MAX_SERIES);
  const [ref, width] = useWidth();
  const [hoverX, setHoverX] = useState(null);
  const visible = series.filter((s) => s.points.length > 0);
  const showX = formatX || ((x) => String(x));

  let body;
  if (!visible.length || !width) {
    body = <div className="chart-empty">Chưa có số liệu</div>;
  } else {
    const d = domainOf(visible);
    const yTicks = niceTicks(d.y0, d.y1, 4);
    const y1 = Math.max(d.y1, yTicks[yTicks.length - 1]);
    const innerW = width - PAD.left - PAD.right;
    const innerH = HEIGHT - PAD.top - PAD.bottom;
    const spanX = d.x1 - d.x0 || 1;
    const sx = (x) => PAD.left + ((x - d.x0) / spanX) * innerW;
    const sy = (y) => PAD.top + innerH - ((y - d.y0) / (y1 - d.y0)) * innerH;
    const xTicks = niceTicks(d.x0, d.x1, Math.max(2, Math.floor(innerW / 60))).filter((t) => t <= d.x1 + 1e-9);

    function onMove(event) {
      const box = event.currentTarget.getBoundingClientRect();
      const x = d.x0 + ((event.clientX - box.left - PAD.left) / innerW) * spanX;
      setHoverX(Math.max(d.x0, Math.min(d.x1, x)));
    }

    let hover = null;
    if (hoverX !== null) {
      const hits = visible.map((s) => ({ s, p: nearest(s.points, hoverX) }));
      const anchor = hits.reduce((a, b) => (Math.abs(b.p[0] - hoverX) < Math.abs(a.p[0] - hoverX) ? b : a)).p[0];
      const left = sx(anchor);
      hover = { hits, anchor, left, flip: left > width * 0.6 };
    }

    body = (
      <>
        <svg viewBox={`0 0 ${width} ${HEIGHT}`} height={HEIGHT} onMouseMove={onMove} onMouseLeave={() => setHoverX(null)}>
          {yTicks.map((t) => (
            <g key={`y${t}`}>
              <line className="chart-grid" x1={PAD.left} x2={width - PAD.right} y1={sy(t)} y2={sy(t)} />
              <text className="chart-axis" x={PAD.left - 6} y={sy(t) + 4} textAnchor="end">
                {formatMetric(t, unit)}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={`x${t}`} className="chart-axis" x={sx(t)} y={HEIGHT - 6} textAnchor="middle">
              {showX(t)}
            </text>
          ))}
          {visible.map((s) => {
            const color = SERIES_COLOR[series.indexOf(s)];
            const path = s.points.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(" ");
            return (
              <g key={s.label}>
                <polyline className="chart-line" points={path} stroke={color} />
                {(s.dots ?? dots) && s.points.length <= DOT_MAX_POINTS &&
                  s.points.map(([x, y]) => <circle key={x} className="chart-dot" cx={sx(x)} cy={sy(y)} r={4} fill={color} />)}
              </g>
            );
          })}
          {hover && (
            <g>
              <line className="chart-cross" x1={hover.left} x2={hover.left} y1={PAD.top} y2={PAD.top + innerH} />
              {hover.hits.map(({ s, p }) => (
                <circle key={s.label} className="chart-dot" cx={sx(p[0])} cy={sy(p[1])} r={5} fill={SERIES_COLOR[series.indexOf(s)]} />
              ))}
            </g>
          )}
          <rect x={PAD.left} y={PAD.top} width={innerW} height={innerH} fill="transparent" />
        </svg>
        {hover && (
          <div className="chart-tip" style={hover.flip ? { right: width - hover.left + 10 } : { left: hover.left + 10 }}>
            <span className="muted">
              {xLabel} {showX(hover.anchor)}
            </span>
            {hover.hits.map(({ s, p }) => (
              <span key={s.label} className="row">
                <span className={`swatch ${SERIES_CLASS[series.indexOf(s)]}`} />
                <span className="grow">{s.label}</span>
                <b className="num">{formatMetric(p[1], unit)}</b>
              </span>
            ))}
          </div>
        )}
      </>
    );
  }

  return (
    <div className="chart">
      <div className="row between wrap">
        <span className="chart-title">{title}</span>
        {series.length > 1 && (
          <span className="legend">
            {series.map((s, i) => (
              <span key={s.label} className="row">
                <span className={`swatch ${SERIES_CLASS[i]}`} />
                {s.label}
              </span>
            ))}
          </span>
        )}
      </div>
      <div className="chart-wrap" ref={ref}>
        {body}
      </div>
      {hint && <span className="small muted">{hint}</span>}
    </div>
  );
}
