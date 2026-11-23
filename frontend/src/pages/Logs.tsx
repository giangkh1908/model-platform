import { useEffect, useMemo, useRef, useState } from "react";
import { Download, Pause, Play, Trash2 } from "lucide-react";
import { clearLogs, LOGS_DOWNLOAD_URL } from "../api/system.ts";
import { Card } from "../components/ui.tsx";
import { logLevel, useLogs } from "../hooks.ts";
import { useApp } from "../store.tsx";

function timeOf(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString("vi-VN", { hour12: false });
}

export default function Logs() {
  const { run, confirm } = useApp();
  const [paused, setPaused] = useState(false);
  const [follow, setFollow] = useState(true);
  const [source, setSource] = useState("");
  const [text, setText] = useState("");
  const [onlyProblems, setOnlyProblems] = useState(false);
  const { lines, reset } = useLogs({ paused });
  const boxRef = useRef<HTMLPreElement>(null);

  const sources = useMemo(() => [...new Set(lines.map((l) => l.source))].sort(), [lines]);

  const visible = useMemo(() => {
    const needle = text.trim().toLowerCase();
    return lines.filter((l) => {
      if (source && l.source !== source) return false;
      if (onlyProblems && !["error", "warn"].includes(logLevel(l.text))) return false;
      return !needle || l.text.toLowerCase().includes(needle);
    });
  }, [lines, source, text, onlyProblems]);

  useEffect(() => {
    if (follow && boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [visible, follow]);

  function onScroll() {
    const box = boxRef.current;
    if (!box) return;
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
    if (atBottom !== follow) setFollow(atBottom);
  }

  async function clear() {
    const ok = await confirm({ title: "Xoá toàn bộ log?", confirmLabel: "Xoá", danger: true });
    if (!ok) return;
    if (await run(clearLogs)) reset();
  }

  return (
    <Card
      title={`Log (${visible.length}/${lines.length} dòng)`}
      actions={
        <div className="row wrap">
          <button type="button" className="btn sm" onClick={() => setPaused(!paused)}>
            {paused ? <Play size={14} /> : <Pause size={14} />} {paused ? "Tiếp tục" : "Tạm dừng"}
          </button>
          <a className="btn sm" href={LOGS_DOWNLOAD_URL}>
            <Download size={14} /> Tải về
          </a>
          <button type="button" className="btn sm danger" onClick={clear}>
            <Trash2 size={14} /> Xoá
          </button>
        </div>
      }
    >
      <div className="log-filters">
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Tìm trong log…" />
        <select value={source} onChange={(e) => setSource(e.target.value)}>
          <option value="">Mọi nguồn</option>
          {sources.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <label className="check">
          <input type="checkbox" checked={onlyProblems} onChange={(e) => setOnlyProblems(e.target.checked)} />
          Chỉ lỗi/cảnh báo
        </label>
        <label className="check">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          Tự cuộn
        </label>
      </div>
      <pre className="logs tall" ref={boxRef} onScroll={onScroll}>
        {visible.length === 0 && <span className="muted">Không có dòng nào.</span>}
        {visible.map((l) => (
          <div key={l.seq} className={`log-line ${logLevel(l.text)}`}>
            <span className="log-time">{timeOf(l.ts)}</span> <span className="log-src">[{l.source}]</span> {l.text}
          </div>
        ))}
      </pre>
    </Card>
  );
}
