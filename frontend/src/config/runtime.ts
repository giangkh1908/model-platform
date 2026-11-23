// Khai báo về runtime và chế độ chạy model: tên hiển thị, VRAM, tuỳ chọn của từng runtime, endpoint kiểu OpenAI.
import type { ModelEntry } from "../types/api.ts";

export type RuntimeId = "decider" | "vllm" | "llamacpp" | "diffusers" | "vieneu";
export type ModeId = "decide" | "chat" | "embed" | "rerank" | "image" | "tts";
export type FitLevel = "ok" | "tight" | "too_big";

export const RUNTIME_NAMES: Record<RuntimeId, string> = {
  decider: "Strands Decider",
  vllm: "vLLM",
  llamacpp: "llama.cpp",
  diffusers: "Diffusers (sinh ảnh)",
  vieneu: "VieNeu-TTS (giọng nói)",
};

export const MODE_NAMES: Record<ModeId, string> = {
  decide: "Ra quyết định",
  chat: "Chat",
  embed: "Embedding",
  rerank: "Reranker",
  image: "Sinh ảnh",
  tts: "Giọng nói",
};

export const VRAM_GB = 12;

export interface OptionField {
  key: string;
  label: string;
  step: number;
  min: number;
  max: number;
  hint: string;
}

/** Tuỳ chọn chỉnh được của từng runtime (khớp OPTION_LIMITS ở backend/runner.py). */
export const OPTION_FIELDS: Partial<Record<RuntimeId, OptionField[]>> = {
  decider: [{ key: "max_concurrency", label: "Song song", step: 1, min: 1, max: 40, hint: "Số request xử lý cùng lúc" }],
  vllm: [
    { key: "ctx", label: "Ngữ cảnh (token)", step: 256, min: 256, max: 262144, hint: "--max-model-len" },
    { key: "gpu_util", label: "Tỉ lệ VRAM", step: 0.05, min: 0.3, max: 0.95, hint: "--gpu-memory-utilization" },
  ],
  llamacpp: [
    { key: "ctx", label: "Ngữ cảnh / request", step: 256, min: 256, max: 262144, hint: "Tổng ngữ cảnh = ngữ cảnh × song song" },
    { key: "parallel", label: "Song song", step: 1, min: 1, max: 32, hint: "-np: số slot xử lý cùng lúc" },
  ],
};

export function runtimeName(runtime: string): string {
  return RUNTIME_NAMES[runtime as RuntimeId] ?? runtime;
}

export function modeName(mode: string): string {
  return MODE_NAMES[mode as ModeId] ?? mode;
}

/** Các chế độ có thể chọn cho một runtime (model sinh ảnh / giọng nói chỉ có một chế độ). */
export function modeChoicesOf(runtime: string): ModeId[] {
  if (runtime === "diffusers") return ["image"];
  if (runtime === "vieneu") return ["tts"];
  return ["chat", "embed", "rerank"];
}

export function modeOf(model: Pick<ModelEntry, "mode" | "runtime"> | null | undefined): string {
  if (model?.mode) return model.mode;
  return model?.runtime === "decider" ? "decide" : "chat";
}

function vramLimits(runtime: string): [number, number] {
  switch (runtime) {
    case "vllm":
      return [VRAM_GB * 0.85 - 1.5, VRAM_GB * 0.85];
    case "llamacpp":
      return [VRAM_GB - 2, VRAM_GB - 0.6];
    default:
      return [VRAM_GB - 3, VRAM_GB - 1];
  }
}

/** Giống hàm fit() trong backend/hf.py, để cập nhật ngay khi người dùng đổi lựa chọn. */
export function fitOf(bytes: number, runtime: string): [FitLevel, string] {
  const size = bytes / 1e9;
  const [ok, tight] = vramLimits(runtime);
  if (size <= ok) return ["ok", `Vừa VRAM (${size.toFixed(1)} GB trọng số / ${VRAM_GB} GB).`];
  if (size <= tight) return ["tight", `Sát giới hạn VRAM (${size.toFixed(1)} GB trọng số), phải giảm độ dài ngữ cảnh.`];
  if (runtime === "llamacpp") {
    return ["too_big", `Lớn hơn VRAM (${size.toFixed(1)} GB): chạy được nhưng một phần nằm trên CPU, chậm hơn nhiều.`];
  }
  return ["too_big", `Không vừa ${VRAM_GB} GB VRAM (${size.toFixed(1)} GB trọng số). Hãy tìm bản GGUF hoặc AWQ nhỏ hơn.`];
}

/** Endpoint của model đang chạy (cổng 8000) theo runtime và chế độ. */
export function endpointOf(runtime: string | null, mode: string | null): string {
  if (runtime === "decider") return "/v1/systemone";
  if (mode === "embed") return "/v1/embeddings";
  if (mode === "rerank") return "/rerank";
  if (mode === "image") return "/v1/images/generations";
  if (mode === "tts") return "/v1/audio/speech";
  return "/v1/chat/completions";
}
