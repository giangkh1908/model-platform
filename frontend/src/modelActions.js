// Các thao tác trên model dùng chung cho trang Tổng quan và trang Model.
import { useApp } from "./store.jsx";

export function isActive(status, model) {
  return status?.current === model.id && status?.state !== "stopped";
}

export function useModelActions() {
  const { run, confirm, status } = useApp();

  function start(model, options = {}, mode = null) {
    const body = { options };
    if (mode) body.mode = mode;
    return run(`/api/models/${encodeURIComponent(model.id)}/start`, { body }, `Đang bật ${model.id}…`);
  }

  function stop() {
    return run("/api/stop", { body: {} }, "Đang tắt model…");
  }

  function save(model, options, mode = null) {
    const body = { options };
    if (mode) body.mode = mode;
    return run(`/api/models/${encodeURIComponent(model.id)}`, { method: "PATCH", body });
  }

  function retry(model) {
    return run(`/api/models/${encodeURIComponent(model.id)}/retry`, { body: {} });
  }

  async function remove(model) {
    const downloading = model.status === "downloading";
    let message = "Xoá khỏi danh sách. File gốc được giữ nguyên.";
    if (model.managed_path) message = "Xoá model và toàn bộ file đã tải. Không khôi phục được.";
    if (downloading) message = "Huỷ tải và xoá phần đã tải.";
    const ok = await confirm({
      title: `${downloading ? "Huỷ tải" : "Xoá"} ${model.id}?`,
      message,
      confirmLabel: downloading ? "Huỷ tải" : "Xoá",
      danger: true,
    });
    if (!ok) return null;
    return run(`/api/models/${encodeURIComponent(model.id)}`, { method: "DELETE" }, `Đang xoá ${model.id}…`);
  }

  async function shutdownWsl() {
    const ok = await confirm({
      title: "Tắt hẳn WSL?",
      message: "Tắt model đang chạy và toàn bộ WSL, trả RAM cho Windows.",
      confirmLabel: "Tắt WSL",
      danger: true,
    });
    if (!ok) return null;
    return run("/api/wsl/shutdown", { body: {} }, "Đang tắt WSL…");
  }

  return { start, stop, save, retry, remove, shutdownWsl, status };
}
