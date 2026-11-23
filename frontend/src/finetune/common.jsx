// Thành phần dùng chung của các trang Finetune và Eval: trạng thái hàng đợi, danh sách bên trái, lấy dữ liệu định kỳ.
import { useState } from "react";
import { Plus } from "lucide-react";
import { Card, Chip, Empty } from "../components/ui.jsx";
import { api, loadLocal, saveLocal } from "../lib.js";
import { useInterval } from "../store.jsx";

export const STATE_LABEL = {
  queued: "Đang chờ",
  running: "Đang chạy",
  done: "Xong",
  stopped: "Đã dừng",
  error: "Lỗi",
  cancelled: "Đã huỷ",
};

export function isActive(state) {
  return state === "queued" || state === "running";
}

export function StateChip({ state }) {
  return <Chip kind={state}>{STATE_LABEL[state]}</Chip>;
}

// Lấy danh sách loại fine-tune một lần.
export function useTasks() {
  const [tasks, setTasks] = useState(null);
  const [error, setError] = useState("");
  useInterval(
    async () => {
      try {
        setTasks((await api("/api/finetune/tasks")).tasks);
        setError("");
      } catch (e) {
        setError(e.message);
      }
    },
    tasks ? null : 3000,
  );
  return { tasks, error };
}

// Lấy một danh sách định kỳ (`key` là khoá trong JSON trả về).
export function useList(path, key, ms = 3000) {
  const [items, setItems] = useState(null);
  async function pull() {
    try {
      setItems((await api(path))[key]);
    } catch {
      // mất kết nối tạm thời: lần sau thử lại
    }
  }
  useInterval(pull, ms);
  return { items, reload: pull };
}

// Mục đang chọn (nhớ qua lần mở trang sau); NEW = đang tạo mới.
export const NEW = "new";
export function useSelected(storageKey) {
  const [selected, setSelected] = useState(() => loadLocal(storageKey, NEW));
  function select(id) {
    setSelected(id);
    saveLocal(storageKey, id);
  }
  return [selected, select];
}

// Cột trái: danh sách mục + nút tạo mới. `renderRow(item)` trả về nội dung một dòng.
export function SideList({ title, newLabel, items, view, onSelect, renderRow, empty, footer }) {
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

export function Tile({ label, value, sub }) {
  return (
    <div className="tile">
      <span className="tile-label">{label}</span>
      <span className="tile-value">{value}</span>
      {sub && <span className="tile-sub">{sub}</span>}
    </div>
  );
}

// Bảng số câu theo nhãn: rows = [{key, ...cột}], columns = [[khoá, tiêu đề]]
export function LabelTable({ rows, columns }) {
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

// Chờ tải danh sách loại fine-tune rồi mới vẽ trang: children là hàm nhận `tasks`.
export function TasksGate({ icon, children }) {
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
