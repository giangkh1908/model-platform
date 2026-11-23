// Fine-tune: loại fine-tune, nguồn dữ liệu, file tải lên, dataset, build, job train, tham số YAML, eval.
import type {
  Build,
  BuildBody,
  BuildPreview,
  BuildRow,
  Dataset,
  EvalChoices,
  EvalDetail,
  EvalRun,
  Job,
  JobDetail,
  Message,
  ParamValues,
  PublishDefaults,
  Source,
  Task,
  Upload,
} from "../types/api.ts";
import { query, request, seg } from "./client.ts";

// ---------------------------------------------------------------- loại, nguồn, file

export async function listTasks(): Promise<Task[]> {
  const data = await request<{ tasks: Task[] }>("/api/finetune/tasks");
  return data.tasks;
}

export async function listSources(): Promise<Source[]> {
  const data = await request<{ sources: Source[] }>("/api/finetune/sources");
  return data.sources;
}

export function uploadFile(name: string, text: string): Promise<Upload> {
  return request("/api/finetune/uploads", { body: { name, text } });
}

// ---------------------------------------------------------------- dataset

export async function listDatasets(): Promise<Dataset[]> {
  const data = await request<{ datasets: Dataset[] }>("/api/finetune/datasets");
  return data.datasets;
}

export function getDataset(id: string): Promise<{ dataset: Dataset; preview: Record<string, unknown>[] }> {
  return request(`/api/finetune/datasets/${seg(id)}`);
}

export function createDataset(body: {
  task: string;
  source: string;
  name: string;
  upload_id: string;
  mapping: Record<string, string>;
}): Promise<Dataset & Message> {
  return request("/api/finetune/datasets", { body });
}

export function deleteDataset(id: string): Promise<Message> {
  return request(`/api/finetune/datasets/${seg(id)}`, { method: "DELETE" });
}

// ---------------------------------------------------------------- build

export async function listBuilds(): Promise<Build[]> {
  const data = await request<{ builds: Build[] }>("/api/finetune/builds");
  return data.builds;
}

export function getBuild(id: string): Promise<{ build: Build; examples: BuildRow[] }> {
  return request(`/api/finetune/builds/${seg(id)}`);
}

export function previewBuild(body: BuildBody, abortSignal?: AbortSignal): Promise<BuildPreview> {
  return request("/api/finetune/builds/preview", { body, signal: abortSignal });
}

export function createBuild(body: BuildBody): Promise<Build & Partial<Message>> {
  return request("/api/finetune/builds", { body });
}

export function deleteBuild(id: string): Promise<Message> {
  return request(`/api/finetune/builds/${seg(id)}`, { method: "DELETE" });
}

// ---------------------------------------------------------------- job train

export async function listJobs(): Promise<Job[]> {
  const data = await request<{ jobs: Job[] }>("/api/finetune/jobs");
  return data.jobs;
}

/** `since`: con trỏ chỉ số đã có, chỉ lấy dòng mới. */
export function getJob(id: string, since: number): Promise<JobDetail> {
  return request(`/api/finetune/jobs/${seg(id)}${query({ since })}`);
}

export function createJob(body: { name: string; build_id: string; base_model: string; params: ParamValues }): Promise<Job & Partial<Message>> {
  return request("/api/finetune/jobs", { body });
}

export function stopJob(id: string): Promise<Message> {
  return request(`/api/finetune/jobs/${seg(id)}/stop`, { method: "POST" });
}

export function deleteJob(id: string): Promise<Message> {
  return request(`/api/finetune/jobs/${seg(id)}`, { method: "DELETE" });
}

/** Thêm checkpoint vào danh sách Model. */
export function registerJob(id: string): Promise<{ id: string } & Partial<Message>> {
  return request(`/api/finetune/jobs/${seg(id)}/register`, { method: "POST" });
}

/** Ghi bù job lên MLflow. */
export function trackJob(id: string): Promise<Message & { run_id: string; url: string }> {
  return request(`/api/finetune/jobs/${seg(id)}/track`, { method: "POST" });
}

export function publishDefaults(id: string): Promise<PublishDefaults> {
  return request(`/api/finetune/jobs/${seg(id)}/publish`);
}

export function publishJob(id: string, name: string, description: string): Promise<Message & { name: string; version: string }> {
  return request(`/api/finetune/jobs/${seg(id)}/publish`, { method: "POST", body: { name, description } });
}

export async function paramsToYaml(task: string, params: ParamValues): Promise<string> {
  const data = await request<{ text: string }>("/api/finetune/params/yaml/dump", { body: { task, params } });
  return data.text;
}

export async function paramsFromYaml(task: string, text: string): Promise<ParamValues> {
  const data = await request<{ params: ParamValues }>("/api/finetune/params/yaml/parse", { body: { task, text } });
  return data.params;
}

// ---------------------------------------------------------------- eval

export async function listEvals(): Promise<EvalRun[]> {
  const data = await request<{ evals: EvalRun[] }>("/api/finetune/evals");
  return data.evals;
}

export function evalChoices(task: string): Promise<EvalChoices> {
  return request(`/api/finetune/evals/choices${query({ task })}`);
}

export function createEval(body: { name: string; build_id: string; base_model: string; adapters: string[] }): Promise<EvalRun & Partial<Message>> {
  return request("/api/finetune/evals", { body });
}

export function getEval(id: string): Promise<EvalDetail> {
  return request(`/api/finetune/evals/${seg(id)}`);
}

export function stopEval(id: string): Promise<Message> {
  return request(`/api/finetune/evals/${seg(id)}/stop`, { method: "POST" });
}

export function deleteEval(id: string): Promise<Message> {
  return request(`/api/finetune/evals/${seg(id)}`, { method: "DELETE" });
}
