import { Boxes, Download, FlaskConical, Play, Power, Square } from "lucide-react";
import { Bar, Card, Chip, CopyButton, Empty, Meter, Sparkline, StatusDot } from "../components/ui.jsx";
import { logLevel, useGpuHistory, useLogs } from "../hooks.js";
import { MODE, RT, duration, endpointOf, gb, modeOf, percent, speed } from "../lib.js";
import { isActive, useModelActions } from "../modelActions.js";
import { go, useApp } from "../store.jsx";

const STATE_TITLE = { running: "Đang chạy", starting: "Đang khởi động…", stopped: "Không có model nào chạy" };

function stateDescription(status) {
  if (status.state === "running") {
    const parts = [RT[status.runtime] || "", MODE[status.mode] || ""];
    if (status.uptime) parts.push(`đã chạy ${duration(status.uptime)}`);
    if (status.load_seconds) parts.push(`nạp mất ${duration(status.load_seconds)}`);
    if (!status.managed) parts.push("bật từ ngoài nền tảng");
    return parts.filter(Boolean).join(" · ");
  }
  if (status.state === "starting") {
    const elapsed = status.uptime ? ` (${duration(status.uptime)})` : "";
    return `Đang nạp model vào GPU${elapsed}. Xem tiến trình ở trang Log.`;
  }
  return "GPU rảnh, không tốn VRAM.";
}

function RunningCard() {
  const { status, busy } = useApp();
  const { stop, shutdownWsl } = useModelActions();
  const endpoint = status.state === "running" ? status.model_url + endpointOf(status.runtime, status.mode) : null;

  return (
    <Card title="Model đang chạy" className="hero">
      <div className="hero-state">
        <StatusDot state={status.state} />
        <div>
          <div className="hero-title">{status.current || STATE_TITLE[status.state]}</div>
          <div className="muted">{stateDescription(status)}</div>
        </div>
      </div>
      {endpoint && (
        <div className="endpoint">
          <code>POST {endpoint}</code>
          <CopyButton text={endpoint} />
        </div>
      )}
      <div className="row wrap">
        {status.state === "running" && (
          <button type="button" className="btn primary" onClick={() => go("playground")}>
            <FlaskConical size={16} /> Thử model
          </button>
        )}
        <button type="button" className="btn" onClick={stop} disabled={busy || status.state === "stopped"}>
          <Square size={16} /> Tắt model
        </button>
        <button type="button" className="btn danger" onClick={shutdownWsl} disabled={busy || !status.wsl}>
          <Power size={16} /> Tắt hẳn WSL
        </button>
        <span className="muted small">{status.wsl ? "WSL đang bật" : "WSL đang tắt"}</span>
      </div>
    </Card>
  );
}

function GpuCard() {
  const { status } = useApp();
  const history = useGpuHistory();
  const gpu = status.gpu;
  if (!gpu) {
    return (
      <Card title="GPU">
        <Empty title="Không đọc được GPU">nvidia-smi không trả về dữ liệu.</Empty>
      </Card>
    );
  }
  const memTotal = gpu.mem_total || 1;
  const charts = [
    { label: "Tải GPU", value: `${gpu.util}%`, fraction: gpu.util / 100, values: history.map((s) => s.util), max: 100 },
    {
      label: "VRAM",
      value: `${(gpu.mem_used / 1024).toFixed(1)} / ${(memTotal / 1024).toFixed(0)} GB`,
      fraction: gpu.mem_used / memTotal,
      values: history.map((s) => s.mem),
      max: memTotal,
      hot: 0.92,
    },
    {
      label: "Điện năng",
      value: `${gpu.power?.toFixed(0)} / ${gpu.power_limit?.toFixed(0)} W`,
      fraction: gpu.power / gpu.power_limit,
      values: history.map((s) => s.power),
      max: gpu.power_limit,
    },
    { label: "Nhiệt độ", value: `${gpu.temp}°C`, fraction: gpu.temp / 90, values: history.map((s) => s.temp), max: 90, hot: 0.83 },
  ];
  return (
    <Card title="GPU" actions={<span className="muted small">{gpu.name}</span>}>
      <div className="gpu-grid">
        {charts.map((c) => (
          <div key={c.label} className="gpu-cell">
            <Meter label={c.label} value={c.value} fraction={c.fraction} hot={c.hot} />
            <Sparkline values={c.values} max={c.max} />
          </div>
        ))}
      </div>
      <div className="muted small">Biểu đồ: 10 phút gần nhất</div>
    </Card>
  );
}

function QuickLaunch() {
  const { status, busy } = useApp();
  const { start } = useModelActions();
  const ready = status.models.filter((m) => m.status === "ready");
  return (
    <Card
      title="Bật nhanh"
      className="span2"
      actions={
        <button type="button" className="btn ghost sm" onClick={() => go("models")}>
          Quản lý model
        </button>
      }
    >
      {ready.length === 0 && (
        <Empty icon={Boxes} title="Chưa có model sẵn sàng">
          Thêm model ở trang Model.
        </Empty>
      )}
      <div className="list">
        {ready.map((m) => {
          const active = isActive(status, m);
          return (
            <div key={m.id} className={`list-row ${active ? "active" : ""}`}>
              <div className="grow">
                <div className="strong">{m.id}</div>
                <div className="muted small">
                  {RT[m.runtime]} · {MODE[modeOf(m)]} · {gb(m.size)}
                </div>
              </div>
              {active ? (
                <Chip kind="ok">{status.state === "running" ? "Đang chạy" : "Đang khởi động"}</Chip>
              ) : (
                <button type="button" className="btn sm primary" disabled={busy} onClick={() => start(m)}>
                  <Play size={14} /> Bật
                </button>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function Downloads() {
  const { status } = useApp();
  const downloading = status.models.filter((m) => m.status === "downloading");
  if (!downloading.length) return null;
  return (
    <Card title="Đang tải" className="span2">
      <div className="list">
        {downloading.map((m) => (
          <div key={m.id} className="list-row column">
            <div className="row between">
              <span className="strong">
                <Download size={14} /> {m.id}
              </span>
              <span className="muted small num">
                {percent(m.progress || 0, 0)}
                {m.speed ? ` · ${speed(m.speed)}` : ""}
              </span>
            </div>
            <Bar fraction={m.progress} />
          </div>
        ))}
      </div>
    </Card>
  );
}

function RecentLogs() {
  const { lines } = useLogs();
  const recent = lines.slice(-14);
  return (
    <Card
      title="Log gần đây"
      className="span2"
      actions={
        <button type="button" className="btn ghost sm" onClick={() => go("logs")}>
          Xem tất cả
        </button>
      }
    >
      <pre className="logs short">
        {recent.length === 0 && <span className="muted">Chưa có log.</span>}
        {recent.map((l) => (
          <div key={l.seq} className={`log-line ${logLevel(l.text)}`}>
            <span className="log-src">[{l.source}]</span> {l.text}
          </div>
        ))}
      </pre>
    </Card>
  );
}

export default function Dashboard() {
  const { status } = useApp();
  if (!status) return <div className="muted">Đang tải…</div>;
  return (
    <div className="grid">
      <RunningCard />
      <GpuCard />
      <Downloads />
      <QuickLaunch />
      <RecentLogs />
    </div>
  );
}
