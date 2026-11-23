// Các thao tác trên model dùng chung cho trang Tổng quan và trang Model.
import * as modelsApi from "./api/models.ts";
import type { ModelOptions } from "./api/models.ts";
import { useApp } from "./store.tsx";
import type { ModelEntry, Status } from "./types/api.ts";

export function isActive(status: Status | null, model: ModelEntry): boolean {
  return status?.current === model.id && status?.state !== "stopped";
}

export function useModelActions() {
  const { run, confirm, status } = useApp();

  function start(model: ModelEntry, options: ModelOptions = {}, mode: string | null = null) {
    return run(() => modelsApi.startModel(model.id, options, mode), `Đang bật ${model.id}…`);
  }

  function stop() {
    return run(() => modelsApi.stopModel(), "Đang tắt model…");
  }

  function save(model: ModelEntry, options: ModelOptions, mode: string | null = null) {
    return run(() => modelsApi.updateModel(model.id, options, mode));
  }

  function retry(model: ModelEntry) {
    return run(() => modelsApi.retryDownload(model.id));
  }

  async function remove(model: ModelEntry) {
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
    return run(() => modelsApi.removeModel(model.id), `Đang xoá ${model.id}…`);
  }

  async function shutdownWsl() {
    const ok = await confirm({
      title: "Tắt hẳn WSL?",
      message: "Tắt model đang chạy và toàn bộ WSL, trả RAM cho Windows.",
      confirmLabel: "Tắt WSL",
      danger: true,
    });
    if (!ok) return null;
    return run(() => modelsApi.shutdownWsl(), "Đang tắt WSL…");
  }

  return { start, stop, save, retry, remove, shutdownWsl, status };
}
