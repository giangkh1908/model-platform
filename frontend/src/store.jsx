// Trạng thái dùng chung: status (làm mới mỗi 2 giây), toast, hộp xác nhận và hàm chạy lệnh.
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { api } from "./lib.js";

const AppContext = createContext(null);

export function useApp() {
  return useContext(AppContext);
}

export function useInterval(callback, ms) {
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

export function AppProvider({ children }) {
  const [status, setStatus] = useState(null);
  const [offline, setOffline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [toasts, setToasts] = useState([]);
  const [confirmState, setConfirmState] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api("/api/status"));
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);

  useInterval(refresh, 2000);

  const dismiss = useCallback((id) => setToasts((list) => list.filter((t) => t.id !== id)), []);

  const toast = useCallback(
    (text, kind = "info") => {
      const id = Math.random().toString(36).slice(2);
      setToasts((list) => [...list.slice(-4), { id, text, kind }]);
      setTimeout(() => dismiss(id), kind === "error" ? 8000 : 4000);
    },
    [dismiss],
  );

  const confirm = useCallback((options) => new Promise((resolve) => setConfirmState({ ...options, resolve })), []);

  function closeConfirm(ok) {
    confirmState?.resolve(ok);
    setConfirmState(null);
  }

  // Chạy một lệnh thay đổi trạng thái, báo kết quả bằng toast rồi làm mới status.
  const run = useCallback(
    async (path, options = {}, pendingText = null) => {
      setBusy(true);
      if (pendingText) toast(pendingText);
      try {
        const result = await api(path, options);
        if (result.message) toast(result.message, "ok");
        return result;
      } catch (e) {
        toast(e.message, "error");
        return null;
      } finally {
        setBusy(false);
        refresh();
      }
    },
    [refresh, toast],
  );

  const value = { status, offline, busy, refresh, toast, toasts, dismiss, confirm, confirmState, closeConfirm, run };
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

function currentRoute() {
  return window.location.hash.replace(/^#\/?/, "") || "dashboard";
}

export function useRoute() {
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

export function go(route) {
  window.location.hash = "#/" + route;
}
