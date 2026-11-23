// Giao diện riêng của từng loại fine-tune. Phần chung (chọn file, dữ liệu, tham số, biểu đồ, hàng đợi) không cần sửa.
// Thêm loại mới: viết một module cài đặt interface TaskUI (như decider.tsx) rồi đăng ký vào TASK_UI.
import type { ComponentType } from "react";
import type { EvalReportData } from "../types/api.ts";
import * as decider from "./decider.tsx";

export interface TaskUI {
  /** Báo cáo eval golden nhiều model (report.json của runtime): xếp hạng, chi tiết. */
  EvalReport: ComponentType<{ report: EvalReportData }>;
}

export const TASK_UI: Record<string, TaskUI> = {
  decider: decider satisfies TaskUI,
};

export function taskUiOf(taskId: string): TaskUI | null {
  return TASK_UI[taskId] ?? null;
}
