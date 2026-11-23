import { useEffect, useState } from "react";
import { Moon, Sun, WifiOff } from "lucide-react";
import { presenceLeave, presencePing } from "./api/system.ts";
import { ConfirmDialog, StatusDot, Toasts } from "./components/ui.tsx";
import { PRESENCE_PING_MS, STORAGE_KEYS } from "./config/app.ts";
import { PAGES } from "./config/navigation.tsx";
import { loadLocal, saveLocal } from "./lib/browser.ts";
import { go, useApp, useRoute } from "./store.tsx";
import type { ModelState } from "./types/api.ts";

type Theme = "auto" | "light" | "dark";

const STATE_LABEL: Record<ModelState, string> = { running: "Đang chạy", starting: "Đang khởi động", stopped: "Không chạy model" };

function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => loadLocal<Theme>(STORAGE_KEYS.theme, "auto"));
  useEffect(() => {
    if (theme === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    saveLocal(STORAGE_KEYS.theme, theme);
  }, [theme]);
  function toggle() {
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDark = theme === "dark" || (theme === "auto" && prefersDark);
    setTheme(isDark ? "light" : "dark");
  }
  return { theme, toggle };
}

// Báo cho server biết tab còn mở; khi tab đóng thì báo "đã rời" để server tự tắt khi không còn tab nào.
function usePresence() {
  useEffect(() => {
    const client = crypto.randomUUID();
    function ping() {
      presencePing(client);
    }
    function leave() {
      presenceLeave(client);
    }
    ping();
    const timer = setInterval(ping, PRESENCE_PING_MS);
    window.addEventListener("pagehide", leave);
    return () => {
      clearInterval(timer);
      window.removeEventListener("pagehide", leave);
    };
  }, []);
}

function StatusPill() {
  const { status, offline } = useApp();
  if (offline) {
    return (
      <div className="pill bad">
        <WifiOff size={14} /> Mất kết nối với nền tảng
      </div>
    );
  }
  if (!status) return <div className="pill">Đang kiểm tra…</div>;
  const gpu = status.gpu;
  return (
    <button type="button" className="pill" onClick={() => go("dashboard")} title="Xem tổng quan">
      <StatusDot state={status.state} />
      <span>{STATE_LABEL[status.state]}</span>
      {status.current && <b className="pill-model">{status.current}</b>}
      {gpu && gpu.mem_used != null && gpu.mem_total != null && (
        <span className="muted num">
          · VRAM {(gpu.mem_used / 1024).toFixed(1)}/{(gpu.mem_total / 1024).toFixed(0)} GB
        </span>
      )}
    </button>
  );
}

export default function App() {
  const route = useRoute();
  const { theme, toggle } = useTheme();
  usePresence();
  const page = PAGES.find((p) => p.id === route) ?? PAGES[0];
  const Page = page.component;

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">MP</span>
          <div>
            <div className="brand-name">Model Platform</div>
            <div className="brand-sub">RTX 3060 12GB · WSL</div>
          </div>
        </div>
        <nav>
          {PAGES.map(({ id, label, icon: Icon }) => (
            <a key={id} href={`#/${id}`} className={id === page.id ? "active" : ""}>
              <Icon size={18} />
              <span>{label}</span>
            </a>
          ))}
        </nav>
      </aside>
      <div className="main">
        <header className="topbar">
          <h1>{page.label}</h1>
          <div className="topbar-right">
            <StatusPill />
            <button type="button" className="icon-btn" onClick={toggle} title="Đổi giao diện sáng/tối">
              {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </header>
        <main className="content">
          <Page />
        </main>
      </div>
      <Toasts />
      <ConfirmDialog />
    </div>
  );
}
