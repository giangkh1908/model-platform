// Hook lấy dữ liệu định kỳ dùng chung.
import { useCallback, useRef, useState } from "react";
import { getGpuHistory, getLogs } from "./api/system.ts";
import { useInterval } from "./store.tsx";
import type { GpuSample, LogLine } from "./types/api.ts";

const MAX_LOG_LINES = 5000;

export function useLogs({ paused = false, interval = 1500 }: { paused?: boolean; interval?: number } = {}) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const lastSeq = useRef(0);

  const pull = useCallback(async () => {
    try {
      const { lines: fresh, seq } = await getLogs(lastSeq.current);
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

export function useGpuHistory(interval = 4000): GpuSample[] {
  const [samples, setSamples] = useState<GpuSample[]>([]);
  useInterval(async () => {
    try {
      setSamples((await getGpuHistory()).samples);
    } catch {
      // bỏ qua
    }
  }, interval);
  return samples;
}

/** Lấy dữ liệu bằng `fetcher` ngay và mỗi `ms`; lỗi tạm thời thì giữ dữ liệu cũ. `items` null khi chưa tải xong. */
export function usePolling<T>(fetcher: () => Promise<T>, ms: number, deps: readonly unknown[] = []) {
  const [items, setItems] = useState<T | null>(null);
  const key = JSON.stringify(deps);
  const current = useRef({ fetcher, key });
  current.current = { fetcher, key };

  const pull = useCallback(async () => {
    const requested = current.current.key;
    try {
      const data = await current.current.fetcher();
      if (requested === current.current.key) setItems(data);
    } catch {
      // mất kết nối tạm thời: lần sau thử lại
    }
  }, []);

  useInterval(pull, ms);
  return { items, reload: pull };
}

export type LogLevel = "error" | "warn" | "event" | "";

export function logLevel(text: string): LogLevel {
  if (/error|traceback|exception|failed|thất bại|lỗi/i.test(text)) return "error";
  if (/warn/i.test(text)) return "warn";
  if (text.startsWith("===")) return "event";
  return "";
}
