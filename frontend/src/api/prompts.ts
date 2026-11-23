// Prompt của bộ quyết định (system / intent / gate): đọc ghi qua backend, backend đồng bộ với Langfuse.
import type {
  Message,
  PromptContent,
  PromptKind,
  PromptKindId,
  PromptLabel,
  PromptStoreStatus,
  PromptSummary,
  PromptVersion,
  PromptVersionInfo,
} from "../types/api.ts";
import { query, request, seg } from "./client.ts";

function path(kind: PromptKindId, id: string): string {
  return `/api/finetune/prompts/${kind}/${seg(id)}`;
}

/** Khai báo các loại prompt do backend định nghĩa. */
export function getKinds(): Promise<{ kinds: PromptKind[]; labels: PromptLabel[] }> {
  return request("/api/finetune/prompt-kinds");
}

export function getStoreStatus(): Promise<PromptStoreStatus> {
  return request("/api/finetune/prompts-status");
}

export async function listPrompts(kind: PromptKindId): Promise<PromptSummary[]> {
  const data = await request<{ items: PromptSummary[] }>(`/api/finetune/prompts/${kind}`);
  return data.items;
}

/** Một phiên bản: theo số, theo nhãn, hoặc (bỏ trống) bản mới nhất. */
export function getPrompt(kind: PromptKindId, id: string, select: { version?: number; label?: PromptLabel } = {}): Promise<PromptVersion> {
  return request(`${path(kind, id)}${query({ version: select.version, label: select.label })}`);
}

export async function listVersions(kind: PromptKindId, id: string): Promise<PromptVersionInfo[]> {
  const data = await request<{ versions: PromptVersionInfo[] }>(`${path(kind, id)}/versions`);
  return data.versions;
}

type Saved = Message & { id: string; version: number };

/** Tạo prompt mới (v1). */
export function createPrompt(kind: PromptKindId, name: string, content: PromptContent): Promise<Saved> {
  return request(`/api/finetune/prompts/${kind}`, { body: { name, content } });
}

/** Lưu phiên bản mới. */
export function saveVersion(kind: PromptKindId, id: string, content: PromptContent): Promise<Saved> {
  return request(path(kind, id), { body: { name: id, content } });
}

export function setLabel(kind: PromptKindId, id: string, version: number, label: PromptLabel): Promise<Message> {
  return request(`${path(kind, id)}/labels/${label}`, { method: "PUT", body: { version } });
}

export function deletePrompt(kind: PromptKindId, id: string): Promise<Message> {
  return request(path(kind, id), { method: "DELETE" });
}
