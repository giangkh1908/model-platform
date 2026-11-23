// Khai báo các trang: đường dẫn (hash), tên, icon, component. Thêm trang mới: thêm một mục ở đây.
import type { ComponentType } from "react";
import {
  Boxes,
  ClipboardCheck,
  Database,
  FileText,
  FlaskConical,
  GraduationCap,
  Layers,
  LayoutDashboard,
  Package,
  ScrollText,
  Settings as SettingsIcon,
  type LucideIcon,
} from "lucide-react";
import Builds from "../pages/Builds.tsx";
import Dashboard from "../pages/Dashboard.tsx";
import Datasets from "../pages/Datasets.tsx";
import Eval from "../pages/Eval.tsx";
import Finetune from "../pages/Finetune.tsx";
import Logs from "../pages/Logs.tsx";
import Models from "../pages/Models.tsx";
import Playground from "../pages/Playground.tsx";
import Prompts from "../pages/Prompts.tsx";
import Registry from "../pages/Registry.tsx";
import Settings from "../pages/Settings.tsx";

export interface PageDef {
  id: string;
  label: string;
  icon: LucideIcon;
  component: ComponentType;
}

export const PAGES: PageDef[] = [
  { id: "dashboard", label: "Tổng quan", icon: LayoutDashboard, component: Dashboard },
  { id: "models", label: "Model", icon: Boxes, component: Models },
  { id: "playground", label: "Thử model", icon: FlaskConical, component: Playground },
  { id: "datasets", label: "Dataset", icon: Database, component: Datasets },
  { id: "prompts", label: "Prompt", icon: FileText, component: Prompts },
  { id: "builds", label: "Build dataset", icon: Layers, component: Builds },
  { id: "finetune", label: "Finetune", icon: GraduationCap, component: Finetune },
  { id: "eval", label: "Eval", icon: ClipboardCheck, component: Eval },
  { id: "registry", label: "Registry", icon: Package, component: Registry },
  { id: "logs", label: "Log", icon: ScrollText, component: Logs },
  { id: "settings", label: "Cài đặt", icon: SettingsIcon, component: Settings },
];
