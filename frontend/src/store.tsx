// Trạng thái dùng chung: status (làm mới mỗi 2 giây), toast, hộp xác nhận, hàm chạy lệnh, điều hướng theo hash.
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { getStatus } from "./api/system.ts";
import { STATUS_POLL_MS } from "./config/app.ts";
import type { Status } from "./types/api.ts";

export type ToastKind = "info" | "ok" | "error";

export interface Toast {
  id: string;
  text: string;
  kind: ToastKind;
}

export interface ConfirmOptions {
  title: string;
  message?: string;
  confirmLabel?: string;
  danger?: boolean;
}

interface ConfirmState extends ConfirmOptions {
  resolve: (ok: boolean) => void;
}

export interface AppContextValue {
  status: Status | null;
  offline: boolean;
  busy: boolean;
  refresh: () => Promise<void>;
  toast: (text: string, kind?: ToastKind) => void;
  toasts: Toast[];
  dismiss: (id: string) => void;
  confirm: (options: ConfirmOptions) => Promise<boolean>;
  confirmState: ConfirmState | null;
  closeConfirm: (ok: boolean) => void;
  /** Chạy một lệnh thay đổi trạng thái: báo kết quả bằng toast, làm mới status. Lỗi thì trả về null. */
  run: <T>(task: () => Promise<T>, pendingText?: string) => Promise<T | null>;
}

const AppContext = createContext<AppContextValue | null>(null);

export function useApp(): AppContextValue {
  const value = useContext(AppContext);
  if (!value) throw new Error("useApp phải nằm trong <AppProvider>");
  return value;
}

/** Gọi `callback` ngay, rồi mỗi `ms` (bỏ qua khi tab ẩn). `ms` null thì dừng. */
export function useInterval(callback: () => void, ms: number | null): void {
  const saved = useRef(callback);
  saved.current = callback;
  useEffect(() => {
    if (ms == null) return undefined;
    saved.current();
    const id = setInterval(() => {
      if (!document.hidden) saved.current();
    }, ms);
    return () => clearInterval(id);
  }, [ms]);
}

function messageOf(result: unknown): string | null {
  if (result && typeof result === "object" && "message" in result && typeof result.message === "string") return result.message;
  return null;
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status | null>(null);
  const [offline, setOffline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [confirmState, setConfirmState] = useState<ConfirmState | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await getStatus());
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);

  useInterval(refresh, STATUS_POLL_MS);

  const dismiss = useCallback((id: string) => setToasts((list) => list.filter((t) => t.id !== id)), []);

  const toast = useCallback(
    (text: string, kind: ToastKind = "info") => {
      const id = Math.random().toString(36).slice(2);
      setToasts((list) => [...list.slice(-4), { id, text, kind }]);
      setTimeout(() => dismiss(id), kind === "error" ? 8000 : 4000);
    },
    [dismiss],
  );

  const confirm = useCallback((options: ConfirmOptions) => new Promise<boolean>((resolve) => setConfirmState({ ...options, resolve })), []);

  function closeConfirm(ok: boolean) {
    confirmState?.resolve(ok);
    setConfirmState(null);
  }

  const run = useCallback(
    async <T,>(task: () => Promise<T>, pendingText?: string): Promise<T | null> => {
      setBusy(true);
      if (pendingText) toast(pendingText);
      try {
        const result = await task();
        const message = messageOf(result);
        if (message) toast(message, "ok");
        return result;
      } catch (e) {
        toast(e instanceof Error ? e.message : String(e), "error");
        return null;
      } finally {
        setBusy(false);
        refresh();
      }
    },
    [refresh, toast],
  );

  const value: AppContextValue = { status, offline, busy, refresh, toast, toasts, dismiss, confirm, confirmState, closeConfirm, run };
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

function currentRoute(): string {
  return window.location.hash.replace(/^#\/?/, "") || "dashboard";
}

export function useRoute(): string {
  const [route, setRoute] = useState(currentRoute);
  useEffect(() => {
    function onChange() {
      setRoute(currentRoute());
    }
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export function go(route: string): void {
  window.location.hash = "#/" + route;
}
