// Hằng số của giao diện: chu kỳ tải lại, nhãn phiên bản prompt, khoá localStorage dùng chung giữa các trang.

export const STATUS_POLL_MS = 2000;
export const PRESENCE_PING_MS = 10_000;
export const LIST_POLL_MS = 10_000;
export const RUN_POLL_MS = 3000; // job / eval đang chạy
export const PROMPT_POLL_MS = 5000; // trang Prompt: thấy ngay thay đổi từ Langfuse

export const PROMPT_LABELS = ["dev", "prod"] as const;

/** Khoá localStorage để một trang mở sẵn một mục ở trang khác. */
export const STORAGE_KEYS = {
  theme: "theme",
  /** Trang Build ghi build để trang Finetune mở form job mới với build đó. */
  trainBuild: "finetune.trainBuild",
  /** Job đang chọn ở trang Finetune. */
  selectedJob: "finetune.job",
  /** Trang job ghi adapter để trang Eval chọn sẵn ("Eval model này"). */
  evalPreselect: "eval.preselect",
} as const;
