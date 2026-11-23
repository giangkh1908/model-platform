// Trạng thái nền tảng, cấu hình, log, cài đặt, tín hiệu còn tab giao diện.
import type { GpuSample, LogLine, Message, PlatformConfig, RuntimeInfo, Settings, Status } from "../types/api.ts";
import { query, request, signal } from "./client.ts";

export function getStatus(): Promise<Status> {
  return request("/api/status");
}

export function getGpuHistory(): Promise<{ samples: GpuSample[]; vram_gb: number }> {
  return request("/api/gpu/history");
}

export function getConfig(): Promise<PlatformConfig> {
  return request("/api/config");
}

export async function getRuntimes(): Promise<RuntimeInfo[]> {
  const data = await request<{ runtimes: RuntimeInfo[] }>("/api/runtimes");
  return data.runtimes;
}

/** `since`: seq của dòng cuối đã có. */
export function getLogs(since: number): Promise<{ lines: LogLine[]; seq: number }> {
  return request(`/api/logs${query({ since })}`);
}

export function clearLogs(): Promise<Message> {
  return request("/api/logs", { method: "DELETE" });
}

export const LOGS_DOWNLOAD_URL = "/api/logs/download";

export function getSettings(): Promise<Settings> {
  return request("/api/settings");
}

export function presencePing(client: string): void {
  signal(`/api/presence${query({ client })}`);
}

/** Báo tab đóng: sendBeacon vẫn gửi được khi trang đang tắt. */
export function presenceLeave(client: string): void {
  navigator.sendBeacon(`/api/presence/leave${query({ client })}`);
}
