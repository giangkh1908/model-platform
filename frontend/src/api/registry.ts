// Model registry (MLflow): trạng thái, model đã xuất bản, alias.
import type { Message, RegistryModel, RegistryStatus } from "../types/api.ts";
import { request, seg } from "./client.ts";

export function getRegistryStatus(): Promise<RegistryStatus> {
  return request("/api/registry/status");
}

export async function listRegistryModels(): Promise<RegistryModel[]> {
  const data = await request<{ models: RegistryModel[] }>("/api/registry/models");
  return data.models;
}

export function setAlias(model: string, alias: string, version: string): Promise<Message> {
  return request(`/api/registry/models/${seg(model)}/aliases/${seg(alias)}`, { method: "PUT", body: { version } });
}

export function deleteAlias(model: string, alias: string): Promise<Message> {
  return request(`/api/registry/models/${seg(model)}/aliases/${seg(alias)}`, { method: "DELETE" });
}
