// Trang một job train: tiến độ, loss / eval loss, xuất bản lên registry. Mọi chỉ số khác và tham số xem trên MLflow
// (mỗi job kết thúc được ghi thành một run). Eval golden nằm ở trang Eval riêng.
import { useEffect, useRef, useState } from "react";
import { Boxes, ClipboardCheck, ExternalLink, Package, Square, Trash2 } from "lucide-react";
import * as ft from "../api/finetune.ts";
import { Bar, Card, Empty, Field, Spinner } from "../components/ui.tsx";
import { STORAGE_KEYS } from "../config/app.ts";
import { saveLocal } from "../lib/browser.ts";
import { duration, formatMetric, formatTime } from "../lib/format.ts";
import { go, useApp, useInterval } from "../store.tsx";
import type { Chart, Job, JobDetail as JobDetailData, MetricRow, PublishDefaults, RunStatus, Task } from "../types/api.ts";
import { StateChip, Tile, isActive } from "./common.tsx";
import LineChart, { type Point, type Series } from "./LineChart.tsx";
import { taskUiOf } from "./tasks.ts";

const POLL_MS = 2000;

/** Giá trị số của một chỉ số trong một dòng metrics.jsonl (không phải số thì null). */
function metricValue(row: MetricRow, key: string): number | null {
  const value = row[key];
  return typeof value === "number" ? value : null;
}

// Lấy job + chỉ số mới (theo con trỏ `next`) mỗi 2 giây khi job còn chờ hoặc đang chạy.
function useJob(jobId: string) {
  const [detail, setDetail] = useState<JobDetailData | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const cursor = useRef(0);
  const inflight = useRef(false);
  const jobRef = useRef(jobId);

  useEffect(() => {
    jobRef.current = jobId;
    cursor.current = 0;
    inflight.current = false;
    setDetail(null);
    setMetrics([]);
    pull();
    // pull() của lần render này đã dùng đúng jobId mới
  }, [jobId]);

  async function pull() {
    // Hai lần lấy song song cùng `since` sẽ nối chỉ số hai lần: mỗi lúc chỉ một lần.
    if (inflight.current) return;
    inflight.current = true;
    const requested = jobId;
    const since = cursor.current;
    try {
      const data = await ft.getJob(requested, since);
      // Đã chuyển sang job khác, hoặc con trỏ đã bị đặt lại trong lúc chờ: bỏ kết quả này.
      if (requested !== jobRef.current || since !== cursor.current) return;
      // Job chạy lại từ đầu (metrics.jsonl bị ghi mới): lấy lại toàn bộ.
      if (data.next < cursor.current) {
        cursor.current = 0;
        setMetrics([]);
        return;
      }
      cursor.current = data.next;
      if (data.metrics.length) setMetrics((old) => [...old, ...data.metrics]);
      setDetail(data);
    } catch {
      // mất kết nối tạm thời: lần sau thử lại
    } finally {
      inflight.current = false;
    }
  }

  const job = detail?.job;
  const live = !job || isActive(job.state);
  useInterval(pull, live ? POLL_MS : null);
  return { detail, metrics, reload: pull };
}


function seriesOf(metrics: MetricRow[], chart: Chart): Series[] {
  return chart.series.map((s) => ({
    label: s.label,
    dots: s.source === "eval",
    points: metrics.flatMap((m): Point[] => {
      const value = metricValue(m, s.key);
      if (m.kind !== s.source || value == null || m.step == null) return [];
      return [[m.step, value]];
    }),
  }));
}

function Progress({ job, status, position }: { job: Job; status: RunStatus; position: number | null }) {
  if (job.state === "queued") {
    return <div className="notice">Đang chờ trong hàng đợi{position ? `: thứ ${position}` : ""}. Job sẽ tự chạy khi GPU rảnh.</div>;
  }
  if (job.state !== "running") return null;
  const total = status.total_steps ?? 0;
  const step = status.step ?? 0;
  return (
    <div className="stack">
      <div className="row between small">
        <span>
          <Spinner /> {status.message ?? "Đang chạy…"}
        </span>
        <span className="num muted">
          {total ? `bước ${step}/${total}` : ""}
          {status.eta ? ` · còn khoảng ${duration(status.eta)}` : ""}
        </span>
      </div>
      <Bar fraction={total ? step / total : 0} />
    </div>
  );
}

interface HeaderProps {
  job: Job;
  status: RunStatus;
  position: number | null;
  onDeleted: () => void;
  reload: () => void;
}

function Header({ job, status, position, onDeleted, reload }: HeaderProps) {
  const { busy, run, confirm } = useApp();
  const published = job.published ?? [];
  const [publishing, setPublishing] = useState(false);

  async function stop() {
    if (job.state === "running") {
      const ok = await confirm({
        title: "Dừng train ngay?",
        message: "Job dừng ngay lập tức và không lưu checkpoint. Muốn có model thì phải train lại.",
        confirmLabel: "Dừng",
        danger: true,
      });
      if (!ok) return;
    }
    await run(() => ft.stopJob(job.id));
    reload();
  }

  function evaluate() {
    saveLocal(STORAGE_KEYS.evalPreselect, `job:${job.id}`);
    go("eval");
  }

  async function remove() {
    const ok = await confirm({
      title: `Xoá job "${job.name}"?`,
      message: "Xoá dữ liệu đã chia, chỉ số, các lần chấm golden và checkpoint. Không hoàn tác được.",
      confirmLabel: "Xoá",
      danger: true,
    });
    if (ok && (await run(() => ft.deleteJob(job.id)))) onDeleted();
  }

  async function register() {
    await run(() => ft.registerJob(job.id));
    reload();
  }

  function afterPublish() {
    setPublishing(false);
    reload();
  }

  return (
    <Card>
      <div className="row between wrap">
        <div className="job-title">{job.name}</div>
        <div className="row wrap">
          {job.state === "running" && (
            <button type="button" className="btn sm danger" disabled={busy} onClick={stop} title="Dừng ngay, không lưu checkpoint">
              <Square size={14} /> Dừng
            </button>
          )}
          {job.state === "queued" && (
            <button type="button" className="btn sm" disabled={busy} onClick={stop}>
              <Square size={14} /> Bỏ khỏi hàng đợi
            </button>
          )}
          {job.state === "done" && (
            <button type="button" className="btn sm" onClick={evaluate} title="Chấm model này trên dữ liệu golden ở trang Eval">
              <ClipboardCheck size={14} /> Eval model này
            </button>
          )}
          {job.state === "done" && !job.registered_model && (
            <button type="button" className="btn sm primary" disabled={busy} onClick={register} title="Thêm checkpoint vào danh sách Model để bật và thử">
              <Boxes size={14} /> Thêm vào danh sách model
            </button>
          )}
          {job.state === "done" && (
            <button type="button" className="btn sm" onClick={() => setPublishing(!publishing)} title="Đẩy checkpoint lên model registry (MLflow)">
              <Package size={14} /> Xuất bản
            </button>
          )}
          {job.state !== "running" && (
            <button type="button" className="btn sm danger" disabled={busy} onClick={remove}>
              <Trash2 size={14} /> Xoá
            </button>
          )}
        </div>
      </div>
      <div className="row wrap small">
        <StateChip state={job.state} />
        <span className="muted">
          build <b>{job.build.title}</b> · từ <b>{job.base_model}</b> · tạo lúc {formatTime(job.created)}
        </span>
        {job.registered_model && (
          <button type="button" className="btn ghost sm" onClick={() => go("models")}>
            Model: {job.registered_model}
          </button>
        )}
      </div>
      {publishing && <PublishForm job={job} onDone={afterPublish} />}
      {published.length > 0 && (
        <div className="row wrap small">
          <span className="muted">Đã xuất bản:</span>
          {published.map((p) => (
            <button key={`${p.name}-${p.version}`} type="button" className="btn ghost sm" onClick={() => go("registry")}>
              {p.name} v{p.version}
            </button>
          ))}
        </div>
      )}
      <Progress job={job} status={status} position={position} />
      {job.state === "stopped" && <div className="notice">Job đã bị dừng giữa chừng nên không có checkpoint.</div>}
      {job.state === "error" && <div className="notice bad">{job.error ?? status.message}</div>}
      {job.state === "error" && status.traceback && (
        <details>
          <summary className="small muted">Chi tiết lỗi</summary>
          <pre className="code">{status.traceback}</pre>
        </details>
      )}
    </Card>
  );
}

// Xuất bản lên registry: tên model (cùng tên = phiên bản mới) + mô tả; tự đính kèm lần eval gần nhất của job.
function PublishForm({ job, onDone }: { job: Job; onDone: () => void }) {
  const { busy, run } = useApp();
  const [info, setInfo] = useState<PublishDefaults | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  useEffect(() => {
    ft.publishDefaults(job.id)
      .then((data) => {
        setInfo(data);
        setName(data.name);
      })
      .catch(() => setInfo({ name: "", eval: null, published: [] }));
  }, [job.id]);

  async function submit() {
    const ok = await run(() => ft.publishJob(job.id, name, description), "Đang xuất bản…");
    if (ok) onDone();
  }

  if (!info) return <Spinner />;
  return (
    <div className="notice">
      <Field label="Tên model trên registry" hint="Chữ không dấu, số, . - _. Cùng tên với lần trước thì thành phiên bản mới.">
        <input value={name} onChange={(e) => setName(e.target.value)} />
      </Field>
      <Field label="Mô tả (tuỳ chọn)">
        <input value={description} onChange={(e) => setDescription(e.target.value)} />
      </Field>
      <div className="muted small">
        {info.eval ? `Đính kèm kết quả eval: ${info.eval}.` : "Chưa có lần eval nào chấm job này, gói sẽ không có kết quả eval."}
      </div>
      <div className="row">
        <button type="button" className="btn sm primary" disabled={busy || !name.trim()} onClick={submit}>
          {busy ? <Spinner /> : <Package size={14} />} Xuất bản
        </button>
      </div>
    </div>
  );
}

// Chỉ loss / eval loss để theo dõi lúc train; mọi chỉ số khác (learning rate, grad norm, VRAM, accuracy, ngưỡng…)
// và tham số xem trên MLflow: mỗi job kết thúc được ghi thành một run ở đó.
function Dashboard({ task, metrics }: { task: Task; metrics: MetricRow[] }) {
  const train = metrics.filter((m) => m.kind === "train");
  const evals = metrics.filter((m) => m.kind === "eval" && metricValue(m, "loss") != null);
  const lastTrain = train[train.length - 1];
  let best: MetricRow | null = null;
  for (const e of evals) {
    if (!best || (metricValue(e, "loss") ?? Infinity) < (metricValue(best, "loss") ?? Infinity)) best = e;
  }

  return (
    <Card title="Loss">
      <div className="tiles">
        <Tile label="Train loss" value={formatMetric(lastTrain ? metricValue(lastTrain, "loss") : null)} sub={lastTrain ? `bước ${lastTrain.step}` : "chưa có"} />
        <Tile label="Eval loss tốt nhất" value={formatMetric(best ? metricValue(best, "loss") : null)} sub={best ? `bước ${best.step}` : "chưa có"} />
      </div>
      <div className="charts">
        {task.charts.map((c) => (
          <LineChart key={c.id} title={c.title} hint={c.hint} unit={c.unit} series={seriesOf(metrics, c)} />
        ))}
      </div>
    </Card>
  );
}

// Link sang run trên MLflow; job kết thúc lúc MLflow tắt thì cho ghi bù.
function MlflowCard({ job, reload }: { job: Job; reload: () => void }) {
  const { busy, run } = useApp();

  async function track() {
    if (await run(() => ft.trackJob(job.id), "Đang ghi lên MLflow…")) reload();
  }

  if (job.mlflow_url) {
    return (
      <Card>
        <div className="row wrap small">
          <span className="muted">Learning rate, grad norm, VRAM, accuracy, ngưỡng, tham số…</span>
          <a href={job.mlflow_url} target="_blank" rel="noreferrer">
            Xem đầy đủ trên MLflow <ExternalLink size={12} />
          </a>
        </div>
      </Card>
    );
  }
  if (!["done", "error", "stopped"].includes(job.state)) return null;
  return (
    <Card>
      <div className="row wrap small">
        <span className="muted">Job này chưa có trên MLflow (lúc kết thúc MLflow chưa chạy).</span>
        <button type="button" className="btn sm" disabled={busy} onClick={track}>
          Ghi lên MLflow
        </button>
      </div>
    </Card>
  );
}

export default function JobDetail({ jobId, tasks, onDeleted }: { jobId: string; tasks: Task[]; onDeleted: () => void }) {
  const { detail, metrics, reload } = useJob(jobId);
  if (!detail) {
    return (
      <Card>
        <Empty title="Đang tải job…" />
      </Card>
    );
  }
  const { job, status, queue_position: position } = detail;
  const task = tasks.find((t) => t.id === job.task);
  const ui = taskUiOf(job.task);
  if (!task || !ui) {
    return (
      <Card>
        <Empty title={`Không hỗ trợ loại "${job.task}"`} />
      </Card>
    );
  }
  return (
    <div className="stack">
      <Header job={job} status={status} position={position} onDeleted={onDeleted} reload={reload} />
      <Dashboard task={task} metrics={metrics} />
      <MlflowCard job={job} reload={reload} />
    </div>
  );
}
