// Tìm / tải model từ Hugging Face, danh sách model, bật / tắt model và WSL.
import type { HubResult, InspectResult, Message } from "../types/api.ts";
import { query, request, seg } from "./client.ts";

export type ModelOptions = Record<string, number>;

export function inspect(url: string): Promise<InspectResult> {
  return request("/api/inspect", { body: { url } });
}

export async function searchHub(q: string, kind: string): Promise<HubResult[]> {
  const data = await request<{ results: HubResult[] }>(`/api/hf/search${query({ q, kind })}`);
  return data.results;
}

export function addModel(body: { url: string; gguf: string | null; runtime: string; mode: string }): Promise<Message & { id: string }> {
  return request("/api/models", { body });
}

function withMode(options: ModelOptions, mode: string | null): { options: ModelOptions; mode?: string } {
  return mode ? { options, mode } : { options };
}

export function startModel(id: string, options: ModelOptions, mode: string | null): Promise<Message> {
  return request(`/api/models/${seg(id)}/start`, { body: withMode(options, mode) });
}

export function updateModel(id: string, options: ModelOptions, mode: string | null): Promise<Message> {
  return request(`/api/models/${seg(id)}`, { method: "PATCH", body: withMode(options, mode) });
}

export function retryDownload(id: string): Promise<Message> {
  return request(`/api/models/${seg(id)}/retry`, { body: {} });
}

export function removeModel(id: string): Promise<Message> {
  return request(`/api/models/${seg(id)}`, { method: "DELETE" });
}

export function stopModel(): Promise<Message> {
  return request("/api/stop", { body: {} });
}

export function shutdownWsl(): Promise<Message> {
  return request("/api/wsl/shutdown", { body: {} });
}
