// Thành phần dùng chung của các trang Finetune và Eval: trạng thái hàng đợi, danh sách bên trái, lấy dữ liệu định kỳ.
import { useState, type ReactNode } from "react";
import { Plus, type LucideIcon } from "lucide-react";
import { listTasks } from "../api/finetune.ts";
import { Card, Chip, Empty } from "../components/ui.tsx";
import { usePolling } from "../hooks.ts";
import { loadLocal, saveLocal } from "../lib/browser.ts";
import { useInterval } from "../store.tsx";
import type { RunState, Task } from "../types/api.ts";

export const STATE_LABEL: Record<RunState, string> = {
  queued: "Đang chờ",
  running: "Đang chạy",
  done: "Xong",
  stopped: "Đã dừng",
  error: "Lỗi",
  cancelled: "Đã huỷ",
};

export function isActive(state: RunState): boolean {
  return state === "queued" || state === "running";
}

export function StateChip({ state }: { state: RunState }) {
  return <Chip kind={state}>{STATE_LABEL[state]}</Chip>;
}

/** Lấy danh sách loại fine-tune (thử lại đến khi được). */
export function useTasks() {
  const [tasks, setTasks] = useState<Task[] | null>(null);
  const [error, setError] = useState("");
  useInterval(
    async () => {
      try {
        setTasks(await listTasks());
        setError("");
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    tasks ? null : 3000,
  );
  return { tasks, error };
}

/** Lấy một danh sách định kỳ bằng một hàm trong src/api/. */
export function useList<T>(fetcher: () => Promise<T[]>, ms = 3000, deps: readonly unknown[] = []) {
  return usePolling(fetcher, ms, deps);
}

/** Mục đang chọn (nhớ qua lần mở trang sau); NEW = đang tạo mới. */
export const NEW = "new";
export function useSelected(storageKey: string): [string, (id: string) => void] {
  const [selected, setSelected] = useState<string>(() => loadLocal(storageKey, NEW));
  function select(id: string) {
    setSelected(id);
    saveLocal(storageKey, id);
  }
  return [selected, select];
}

interface SideListProps<T extends { id: string }> {
  title: ReactNode;
  newLabel: ReactNode;
  items: T[];
  view: string;
  onSelect: (id: string) => void;
  renderRow: (item: T) => ReactNode;
  empty: ReactNode;
  footer?: ReactNode;
}

/** Cột trái: danh sách mục + nút tạo mới. `renderRow(item)` trả về nội dung một dòng. */
export function SideList<T extends { id: string }>({ title, newLabel, items, view, onSelect, renderRow, empty, footer }: SideListProps<T>) {
  return (
    <Card
      title={title}
      className="ft-jobs"
      actions={
        <button type="button" className={`btn sm ${view === NEW ? "primary" : ""}`} onClick={() => onSelect(NEW)}>
          <Plus size={14} /> {newLabel}
        </button>
      }
    >
      {items.length === 0 ? (
        <span className="small muted">{empty}</span>
      ) : (
        <div className="list">
          {items.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`list-row column clickable ft-job ${item.id === view ? "active" : ""}`}
              onClick={() => onSelect(item.id)}
            >
              {renderRow(item)}
            </button>
          ))}
        </div>
      )}
      {footer && <span className="small muted">{footer}</span>}
    </Card>
  );
}

export function Tile({ label, value, sub }: { label: ReactNode; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="tile">
      <span className="tile-label">{label}</span>
      <span className="tile-value">{value}</span>
      {sub && <span className="tile-sub">{sub}</span>}
    </div>
  );
}

export type LabelRow = { key: string } & Record<string, string | number>;

/** Bảng số câu theo nhãn: rows = [{key, ...cột}], columns = [[khoá, tiêu đề]]. */
export function LabelTable({ rows, columns }: { rows: LabelRow[]; columns: [string, string][] }) {
  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th>Nhãn</th>
            {columns.map(([key, label]) => (
              <th key={key} className="num">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <td>
                <code>{r.key}</code>
              </td>
              {columns.map(([key]) => (
                <td key={key} className={`num ${r[key] === 0 ? "warn-text" : ""}`}>
                  {r[key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Chờ tải danh sách loại fine-tune rồi mới vẽ trang: children là hàm nhận `tasks`. */
export function TasksGate({ icon, children }: { icon: LucideIcon; children: (tasks: Task[]) => ReactNode }) {
  const { tasks, error } = useTasks();
  if (!tasks) {
    return (
      <Card>
        <Empty icon={icon} title={error || "Đang tải…"} />
      </Card>
    );
  }
  return <div className="ft">{children(tasks)}</div>;
}
