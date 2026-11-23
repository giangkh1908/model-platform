// Gọi API, định dạng số liệu và các hằng số dùng chung.

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function errorMessage(json, response) {
  const detail = json.error || json.detail || response.statusText;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join("; ");
  if (typeof detail === "string") return detail;
  return JSON.stringify(detail);
}

export async function api(path, { method, body, signal } = {}) {
  const options = { method: method || (body === undefined ? "GET" : "POST"), signal };
  if (body !== undefined) {
    options.headers = { "content-type": "application/json" };
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(path, options);
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError("Mất kết nối với nền tảng", 0);
  }
  const json = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(errorMessage(json, response), response.status);
  return json;
}

export const RT = {
  decider: "Strands Decider",
  vllm: "vLLM",
  llamacpp: "llama.cpp",
  diffusers: "Diffusers (sinh ảnh)",
  vieneu: "VieNeu-TTS (giọng nói)",
};
export const MODE = { decide: "Ra quyết định", chat: "Chat", embed: "Embedding", rerank: "Reranker", image: "Sinh ảnh", tts: "Giọng nói" };
export const VRAM_GB = 12;

// Đóng gói mẫu âm thanh mono (-1..1) thành file WAV 16-bit
export function encodeWav(samples, sampleRate) {
  const dataSize = samples.length * 2;
  const view = new DataView(new ArrayBuffer(44 + dataSize));
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
}

// Các chế độ có thể chọn cho một runtime (model sinh ảnh/giọng nói chỉ có một chế độ)
export function modeChoicesOf(runtime) {
  if (runtime === "diffusers") return ["image"];
  if (runtime === "vieneu") return ["tts"];
  return ["chat", "embed", "rerank"];
}

export function modeOf(model) {
  if (model?.mode) return model.mode;
  return model?.runtime === "decider" ? "decide" : "chat";
}

export function gb(bytes) {
  if (bytes == null) return "–";
  if (bytes < 1e9) return (bytes / 1e6).toFixed(0) + " MB";
  return (bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 1) + " GB";
}

export function speed(bytesPerSecond) {
  if (bytesPerSecond >= 1e6) return (bytesPerSecond / 1e6).toFixed(1) + " MB/s";
  return (bytesPerSecond / 1e3).toFixed(0) + " KB/s";
}

export function duration(seconds) {
  if (seconds == null) return "–";
  const s = Math.max(0, Math.round(seconds));
  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  if (hours) return `${hours} giờ ${minutes} phút`;
  if (minutes) return `${minutes} phút ${s % 60} giây`;
  return `${s} giây`;
}

export function compact(n) {
  if (n == null) return "–";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

export function clamp01(x) {
  return Math.max(0, Math.min(1, x || 0));
}

function vramLimits(runtime) {
  switch (runtime) {
    case "vllm":
      return [VRAM_GB * 0.85 - 1.5, VRAM_GB * 0.85];
    case "llamacpp":
      return [VRAM_GB - 2, VRAM_GB - 0.6];
    default:
      return [VRAM_GB - 3, VRAM_GB - 1];
  }
}

// Giống hàm fit() trong backend/hf.py, để cập nhật ngay khi người dùng đổi lựa chọn.
export function fitOf(bytes, runtime) {
  const size = bytes / 1e9;
  const [ok, tight] = vramLimits(runtime);
  if (size <= ok) return ["ok", `Vừa VRAM (${size.toFixed(1)} GB trọng số / ${VRAM_GB} GB).`];
  if (size <= tight) return ["tight", `Sát giới hạn VRAM (${size.toFixed(1)} GB trọng số), phải giảm độ dài ngữ cảnh.`];
  if (runtime === "llamacpp") {
    return ["too_big", `Lớn hơn VRAM (${size.toFixed(1)} GB): chạy được nhưng một phần nằm trên CPU, chậm hơn nhiều.`];
  }
  return ["too_big", `Không vừa ${VRAM_GB} GB VRAM (${size.toFixed(1)} GB trọng số). Hãy tìm bản GGUF hoặc AWQ nhỏ hơn.`];
}

export function endpointOf(runtime, mode) {
  if (runtime === "decider") return "/v1/systemone";
  if (mode === "embed") return "/v1/embeddings";
  if (mode === "rerank") return "/rerank";
  if (mode === "image") return "/v1/images/generations";
  if (mode === "tts") return "/v1/audio/speech";
  return "/v1/chat/completions";
}

// [khoá, nhãn, bước, min, max, gợi ý]
export const OPTION_FIELDS = {
  decider: [["max_concurrency", "Song song", 1, 1, 40, "Số request xử lý cùng lúc"]],
  vllm: [
    ["ctx", "Ngữ cảnh (token)", 256, 256, 262144, "--max-model-len"],
    ["gpu_util", "Tỉ lệ VRAM", 0.05, 0.3, 0.95, "--gpu-memory-utilization"],
  ],
  llamacpp: [
    ["ctx", "Ngữ cảnh / request", 256, 256, 262144, "Tổng ngữ cảnh = ngữ cảnh × song song"],
    ["parallel", "Song song", 1, 1, 32, "-np: số slot xử lý cùng lúc"],
  ],
};

// localStorage có thể bị chặn (cửa sổ ẩn danh…), nên mọi lần đọc/ghi đều bọc try.
export function loadLocal(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value == null ? fallback : JSON.parse(value);
  } catch {
    return fallback;
  }
}

export function saveLocal(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // bỏ qua
  }
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

export function splitLines(text) {
  return text.split("\n").map((s) => s.trim()).filter(Boolean);
}

export function percent(fraction, digits = 1) {
  return (fraction * 100).toFixed(digits) + "%";
}

// Số liệu của chỉ số: unit "%" thì đổi tỉ lệ 0..1 thành phần trăm; còn lại giữ 3 chữ số có nghĩa.
export function formatMetric(value, unit = "") {
  if (value == null || Number.isNaN(value)) return "–";
  if (unit === "%") {
    const pct = value * 100;
    return pct.toFixed(Math.abs(pct) < 1 && pct !== 0 ? 2 : 1) + "%";
  }
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs >= 1000) return compact(value);
  if (abs >= 0.001) return String(Number(value.toPrecision(3)));
  return value.toExponential(1);
}

export function downloadText(filename, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function formatTime(seconds) {
  if (!seconds) return "–";
  return new Date(seconds * 1000).toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" });
}
