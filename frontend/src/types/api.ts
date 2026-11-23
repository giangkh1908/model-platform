// Kiểu dữ liệu backend trả về / nhận vào (khai báo duy nhất, dùng chung cho src/api/ và component).
// Đổi API ở backend thì sửa ở đây trước, `npm run typecheck` sẽ chỉ ra mọi chỗ cần sửa theo.

export interface Message {
  message: string;
}

// ---------------------------------------------------------------- hệ thống, model

export type ModelState = "running" | "starting" | "stopped";

export interface ModelEntry {
  id: string;
  repo: string;
  runtime: string;
  mode?: string;
  path: string;
  file: string | null;
  size: number;
  status: "ready" | "downloading" | "error";
  progress?: number;
  /** Đang tải: số byte đã tải và tốc độ (byte/giây). */
  downloaded?: number;
  speed?: number;
  added?: string;
  error?: string;
  options: Record<string, number>;
  managed_path: boolean;
  finetune_job?: string;
  base_model?: string | null;
}

/** nvidia-smi: % / MiB / W / °C; null khi không đọc được. */
export interface GpuInfo {
  name: string;
  util: number | null;
  mem_used: number | null;
  mem_total: number | null;
  power: number | null;
  power_limit: number | null;
  temp: number | null;
  fan: number | null;
}

export interface Status {
  state: ModelState;
  current: string | null;
  runtime: string | null;
  mode: string | null;
  managed: boolean;
  uptime: number | null;
  load_seconds: number | null;
  gpu: GpuInfo | null;
  wsl: boolean;
  model_url: string;
  models: ModelEntry[];
  has_token: boolean;
}

export interface PlatformConfig {
  distro: string;
  home: string;
  models_dir: string;
  model_url: string;
  panel: string;
  vram_gb: number;
  runtime_names: Record<string, string>;
  default_options: Record<string, Record<string, number>>;
  option_limits: Record<string, { min: number; max: number; int: boolean }>;
}

export interface LogLine {
  seq: number;
  ts: number;
  source: string;
  text: string;
}

export interface GpuSample {
  t: number;
  util: number | null;
  mem: number | null;
  power: number | null;
  temp: number | null;
}

export interface RuntimeInfo {
  id: string;
  name: string;
  path: string;
  installed: boolean;
}

export interface HubResult {
  repo: string;
  downloads: number;
  likes: number;
  pipeline_tag: string | null;
  gguf: boolean;
  updated: string | null;
}

export interface Choice {
  id: string;
  name: string;
  note?: string;
}

export interface GgufFile {
  name: string;
  size: number;
  parts: number;
  pattern: string | null;
  recommended?: boolean;
  fit?: string;
  fit_text?: string;
}

export interface InspectResult {
  repo: string;
  pipeline_tag: string | null;
  gated: boolean;
  license: string | null;
  downloads: number | null;
  likes: number | null;
  updated: string | null;
  gguf_files: GgufFile[];
  warnings: string[];
  runtime: string;
  runtime_name: string;
  runtimes: Choice[];
  mode: string;
  modes: Choice[];
  size: number;
  weights: number;
  base_model?: string | null;
  has_safetensors?: boolean;
  fit: string;
  fit_text: string;
}

export interface Settings {
  has_token: boolean;
  token_masked: string | null;
  hf_user: string | null;
}

// ---------------------------------------------------------------- thử model

export interface NoulAnswer {
  type: "noul";
  noul: number;
}

export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
}

export type DecideAnswer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

/** Thời gian đo ở backend (ms), có trong mọi kết quả thử model. */
export interface Timed {
  _ms: number;
}

export interface DecideResponse extends Timed {
  model: string;
  answers: Record<string, DecideAnswer>;
  usage: { input_tokens: number; output_tokens: number };
}

export interface EmbedResponse extends Timed {
  dims: number;
  similarity: number[][];
  preview: number[][];
  usage?: Record<string, number>;
}

export interface RerankResponse extends Timed {
  results?: { index: number; relevance_score: number }[];
}

export interface ImageResponse extends Timed {
  image: string;
}

export interface TtsResponse extends Timed {
  audio: string;
}

export interface Voice {
  id: string;
  label: string;
}

// ---------------------------------------------------------------- prompt

export type PromptKindId = "system" | "intent" | "gate";
export type PromptLabel = "dev" | "prod";

export interface PromptField {
  key: string;
  label: string;
  hint: string;
  multiline: boolean;
  required: boolean;
}

export interface PromptKind {
  id: PromptKindId;
  label: string;
  help: string;
  fields: PromptField[];
}

export type PromptContent = Record<string, string> & { text: string };

export interface PromptSummary {
  id: string;
  kind: PromptKindId;
  name: string;
  latest: number;
  labels: Record<PromptLabel, number | null>;
  versions: { version: number }[];
}

export interface PromptVersion {
  id: string;
  kind: PromptKindId;
  name: string;
  version: number;
  content: PromptContent;
  labels: PromptLabel[];
  note: string;
}

export interface PromptVersionInfo {
  version: number;
  note: string;
  labels: PromptLabel[];
}

/** Chọn một prompt: theo số phiên bản, theo nhãn, hoặc (không có cả hai) bản mới nhất. */
export interface PromptRef {
  id: string;
  version?: number;
  label?: PromptLabel;
}

export interface PromptStoreStatus {
  source: "langfuse" | "local";
  url?: string;
}

// ---------------------------------------------------------------- dữ liệu, build

export interface Field {
  key: string;
  label: string;
  hint: string;
  required: boolean;
}

export interface Param {
  key: string;
  label: string;
  default: number | boolean;
  min: number;
  max: number;
  step: number;
  hint: string;
  kind: "number" | "bool";
  advanced: boolean;
}

export interface Chart {
  id: string;
  title: string;
  series: { source: "train" | "eval"; key: string; label: string }[];
  unit: string;
  hint: string;
}

export interface Task {
  id: string;
  name: string;
  description: string;
  fields: Field[];
  params: Param[];
  charts: Chart[];
  example_rows: Record<string, unknown>[];
  base_models: string[];
}

export interface Upload {
  upload_id: string;
  name: string;
  format: string;
  rows: number;
  columns: string[];
  preview: Record<string, unknown>[];
}

export interface Source {
  id: string;
  name: string;
  description: string;
  role: "train" | "eval";
}

export type ParamValues = Record<string, number | boolean | "">;

export interface Dataset {
  id: string;
  name: string;
  task: string;
  source: "sft" | "eval";
  created: number;
  file: string;
  format: string;
  mapping: Record<string, string>;
  rows: number;
  invalid: number;
  errors: { row: number; reason: string }[];
  labels: { key: string; count: number }[];
}

export interface Spec {
  system: { name: string; version: number; selected_by: string; text: string; intent_question: string };
  intents: { key: string; name: string; version: number; selected_by: string; policy: string }[];
  gates: { key: string; name: string; version: number; selected_by: string; policy: string; question: string }[];
}

export interface SpecTitle {
  system: string;
  intents: string;
  gates: string;
}

/** Một dòng "như thật" của build. */
export interface BuildRow {
  input: string;
  history: unknown;
  intent: string | null;
  gates: Record<string, boolean>;
  dataset?: string;
}

export interface BuildStats {
  train: number;
  val: number;
  total: number;
  val_from: "eval" | "random";
  val_fraction_actual: number;
  removed_duplicates: number;
  conflicts: number;
  leaked: number;
  dropped_no_label: number;
  missing_in_val: string[];
  strata: { key: string; total: number; train: number; val: number }[];
  gates: { key: string; train_true: number; train_false: number; val_true: number; val_false: number }[];
  max_chars: number;
  avg_chars: number;
}

export interface BuildBody {
  task: string;
  name?: string;
  note?: string;
  system: PromptRef | null;
  intents: PromptRef[];
  gates: PromptRef[];
  datasets: string[];
  eval_datasets: string[];
  dedupe: boolean;
  val_percent: number;
  split_seed: number;
}

export interface Build {
  id: string;
  name: string;
  version: number;
  task: string;
  created: number;
  manifest: { spec_title: SpecTitle; datasets: { id: string; name: string; rows: number }[]; eval_datasets: { id: string; name: string; rows: number }[] };
  task_config: Spec;
  dedupe: boolean;
  val_percent: number;
  split_seed: number;
  note: string;
  stats: BuildStats;
  has_eval: boolean;
  refs: { system: PromptRef; intents: PromptRef[]; gates: PromptRef[]; datasets: string[]; eval_datasets: string[] };
}

export interface BuildPreview extends BuildStats {
  spec: Spec;
  spec_title: SpecTitle;
  examples: BuildRow[];
}

// ---------------------------------------------------------------- job, eval, registry

export type RunState = "queued" | "running" | "done" | "stopped" | "error" | "cancelled";

/** status.json của runtime (train hoặc eval). */
export interface RunStatus {
  state?: string;
  step?: number;
  total_steps?: number;
  epoch?: number;
  eta?: number;
  message?: string;
  traceback?: string;
  done?: number;
  total?: number;
}

export interface Job {
  id: string;
  name: string;
  task: string;
  created: number;
  state: RunState;
  base_model: string;
  output_dir: string;
  params: Record<string, number | boolean>;
  task_config: Spec;
  build: { id: string; title: string; train: number; val: number; has_eval: boolean };
  registered_model: string | null;
  mlflow_run_id?: string;
  mlflow_url?: string;
  published?: { name: string; version: string; at: number }[];
  error?: string | null;
  queue_position?: number | null;
  status?: RunStatus;
}

export interface MetricRow {
  kind: "start" | "train" | "eval" | "final";
  step?: number;
  time: number;
  [key: string]: unknown;
}

export interface JobDetail {
  job: Job;
  status: RunStatus;
  metrics: MetricRow[];
  next: number;
  queue_position: number | null;
}

export interface EvalRun {
  id: string;
  name: string;
  task: string;
  created: number;
  state: RunState;
  source: { id: string; title: string };
  n: number;
  auto: boolean;
  models: { key: string; name: string; kind: "base" | "adapter"; threshold: number | null; leaked: number }[];
  queue_position?: number | null;
  status?: RunStatus;
  error?: string | null;
}

export interface ThresholdRates {
  threshold: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
  coverage: number;
}

export interface ClassSummary {
  n: number;
  loss: number;
  acc: number;
  macro_f1: number;
  ece: number;
  mean_confidence: number;
}

export interface LabelResult {
  label: string;
  n: number;
  predicted: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface GateResult {
  summary: ClassSummary;
  precision: number;
  recall: number;
  f1: number;
  positives: number;
  mistakes: { text: string; gold: "true" | "false"; p_true: number }[];
}

/** Kết quả của một model trong báo cáo eval (decider). Phần intent vắng khi bộ chỉ có gate. */
export interface EvalModelResult {
  key: string;
  name: string;
  kind: "base" | "adapter";
  threshold: number | null;
  leaked: number;
  summary: ClassSummary;
  per_label?: LabelResult[];
  confusions?: { gold: string; pred: string; count: number }[];
  at_threshold?: ThresholdRates | null;
  curve?: ThresholdRates[];
  mistakes?: { text: string; gold: string; pred: string; confidence: number }[];
  gates?: Record<string, GateResult>;
}

export interface EvalReportData {
  n: number;
  models: EvalModelResult[];
}

export interface EvalDetail {
  eval: EvalRun;
  status: RunStatus;
  report: EvalReportData | null;
  queue_position: number | null;
}

export interface EvalChoices {
  bases: { id: string; ref: string }[];
  adapters: { ref: string; name: string; job_name: string; root_base: string; base_model: string; build: string; created: number }[];
}

export interface PublishDefaults {
  name: string;
  eval: string | null;
  published: { name: string; version: string; at: number }[];
}

export interface RegistryStatus {
  enabled: boolean;
  online?: boolean;
  ui?: string;
  message: string;
  aliases: string[];
}

export interface RegistryVersion {
  version: string;
  created: number;
  description: string;
  tags: Record<string, string>;
  metrics: Record<string, number>;
  aliases: string[];
  run_id: string;
}

export interface RegistryModel {
  name: string;
  description: string;
  aliases: Record<string, string>;
  versions: RegistryVersion[];
}
