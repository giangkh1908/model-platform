// Hook lấy dữ liệu định kỳ: log và lịch sử GPU.
import { useCallback, useRef, useState } from "react";
import { api } from "./lib.js";
import { useInterval } from "./store.jsx";

const MAX_LOG_LINES = 5000;

export function useLogs({ paused = false, interval = 1500 } = {}) {
  const [lines, setLines] = useState([]);
  const lastSeq = useRef(0);

  const pull = useCallback(async () => {
    try {
      const { lines: fresh, seq } = await api(`/api/logs?since=${lastSeq.current}`);
      // Backend đã xoá log hoặc khởi động lại: tải lại từ đầu.
      if (seq < lastSeq.current) {
        lastSeq.current = 0;
        setLines([]);
        return;
      }
      if (!fresh.length) return;
      lastSeq.current = fresh[fresh.length - 1].seq;
      setLines((old) => [...old, ...fresh].slice(-MAX_LOG_LINES));
    } catch {
      // mất kết nối tạm thời: lần sau thử lại
    }
  }, []);

  useInterval(pull, paused ? null : interval);

  const reset = useCallback(() => {
    setLines([]);
  }, []);

  return { lines, reset };
}

export function useGpuHistory(interval = 4000) {
  const [samples, setSamples] = useState([]);
  useInterval(async () => {
    try {
      const data = await api("/api/gpu/history");
      setSamples(data.samples);
    } catch {
      // bỏ qua
    }
  }, interval);
  return samples;
}

export function logLevel(text) {
  if (/error|traceback|exception|failed|thất bại|lỗi/i.test(text)) return "error";
  if (/warn/i.test(text)) return "warn";
  if (text.startsWith("===")) return "event";
  return "";
}
