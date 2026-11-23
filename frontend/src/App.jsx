import { useEffect, useState } from "react";
import { Boxes, ClipboardCheck, Database, FileText, FlaskConical, GraduationCap, Layers, LayoutDashboard, Moon, ScrollText, Settings as SettingsIcon, Sun, WifiOff } from "lucide-react";
import { ConfirmDialog, StatusDot, Toasts } from "./components/ui.jsx";
import { loadLocal, saveLocal } from "./lib.js";
import Builds from "./pages/Builds.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Datasets from "./pages/Datasets.jsx";
import Eval from "./pages/Eval.jsx";
import Finetune from "./pages/Finetune.jsx";
import Logs from "./pages/Logs.jsx";
import Models from "./pages/Models.jsx";
import Playground from "./pages/Playground.jsx";
import Prompts from "./pages/Prompts.jsx";
import Settings from "./pages/Settings.jsx";
import { go, useApp, useRoute } from "./store.jsx";

const PAGES = [
  { id: "dashboard", label: "Tổng quan", icon: LayoutDashboard, component: Dashboard },
  { id: "models", label: "Model", icon: Boxes, component: Models },
  { id: "playground", label: "Thử model", icon: FlaskConical, component: Playground },
  { id: "datasets", label: "Dataset", icon: Database, component: Datasets },
  { id: "prompts", label: "Prompt", icon: FileText, component: Prompts },
  { id: "builds", label: "Build dataset", icon: Layers, component: Builds },
  { id: "finetune", label: "Finetune", icon: GraduationCap, component: Finetune },
  { id: "eval", label: "Eval", icon: ClipboardCheck, component: Eval },
  { id: "logs", label: "Log", icon: ScrollText, component: Logs },
  { id: "settings", label: "Cài đặt", icon: SettingsIcon, component: Settings },
];

const STATE_LABEL = { running: "Đang chạy", starting: "Đang khởi động", stopped: "Không chạy model" };

function useTheme() {
  const [theme, setTheme] = useState(() => loadLocal("theme", "auto"));
  useEffect(() => {
    if (theme === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    saveLocal("theme", theme);
  }, [theme]);
  function toggle() {
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDark = theme === "dark" || (theme === "auto" && prefersDark);
    setTheme(isDark ? "light" : "dark");
  }
  return { theme, toggle };
}

const PING_SECONDS = 10;

// Báo cho server biết tab còn mở; khi tab đóng thì báo "đã rời" để server tự tắt khi không còn tab nào.
function usePresence() {
  useEffect(() => {
    const client = crypto.randomUUID();
    function ping() {
      fetch(`/api/presence?client=${client}`, { method: "POST" }).catch(() => {});
    }
    function leave() {
      navigator.sendBeacon(`/api/presence/leave?client=${client}`);
    }
    ping();
    const timer = setInterval(ping, PING_SECONDS * 1000);
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
      {gpu && (
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
  const page = PAGES.find((p) => p.id === route) || PAGES[0];
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
