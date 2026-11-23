// Trang một job train: tiến độ, dashboard chỉ số (biểu đồ do backend khai báo, gồm cả eval loss chấm định kỳ trên val)
// và tóm tắt sau train. Eval golden nằm ở trang Eval riêng. Phần riêng của từng loại lấy từ TASK_UI.
import { useEffect, useRef, useState } from "react";
import { Boxes, ClipboardCheck, Package, Square, Trash2 } from "lucide-react";
import { Bar, Card, Chip, Empty, Field, Spinner } from "../components/ui.jsx";
import { api, duration, formatMetric, formatTime, saveLocal } from "../lib.js";
import { go, useApp, useInterval } from "../store.jsx";
import { StateChip, Tile, isActive } from "./common.jsx";
import LineChart from "./LineChart.jsx";
import { TASK_UI } from "./tasks.js";

const POLL_MS = 2000;
// Trang Eval đọc khoá này để chọn sẵn model khi bấm "Eval model này".
export const EVAL_PRESELECT_KEY = "eval.preselect";

// Lấy job + chỉ số mới (theo con trỏ `next`) mỗi 2 giây khi job còn chờ hoặc đang chạy.
function useJob(jobId) {
  const [detail, setDetail] = useState(null);
  const [metrics, setMetrics] = useState([]);
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
      const data = await api(`/api/finetune/jobs/${requested}?since=${since}`);
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


function seriesOf(metrics, chart) {
  return chart.series.map((s) => ({
    label: s.label,
    dots: s.source === "eval",
    points: metrics.filter((m) => m.kind === s.source && m[s.key] != null).map((m) => [m.step, m[s.key]]),
  }));
}

function Progress({ job, status, position }) {
  if (job.state === "queued") {
    return <div className="notice">Đang chờ trong hàng đợi{position ? `: thứ ${position}` : ""}. Job sẽ tự chạy khi GPU rảnh.</div>;
  }
  if (job.state !== "running") return null;
  const total = status.total_steps || 0;
  const step = status.step || 0;
  return (
    <div className="stack">
      <div className="row between small">
        <span>
          <Spinner /> {status.message || "Đang chạy…"}
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

function Header({ job, status, position, onDeleted, reload }) {
  const { busy, run, confirm } = useApp();
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
    await run(`/api/finetune/jobs/${job.id}/stop`, { method: "POST" });
    reload();
  }

  function evaluate() {
    saveLocal(EVAL_PRESELECT_KEY, `job:${job.id}`);
    go("eval");
  }

  async function remove() {
    const ok = await confirm({
      title: `Xoá job "${job.name}"?`,
      message: "Xoá dữ liệu đã chia, chỉ số, các lần chấm golden và checkpoint. Không hoàn tác được.",
      confirmLabel: "Xoá",
      danger: true,
    });
    if (ok && (await run(`/api/finetune/jobs/${job.id}`, { method: "DELETE" }))) onDeleted();
  }

  async function register() {
    await run(`/api/finetune/jobs/${job.id}/register`, { method: "POST" });
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
      {job.published?.length > 0 && (
        <div className="row wrap small">
          <span className="muted">Đã xuất bản:</span>
          {job.published.map((p) => (
            <button key={`${p.name}-${p.version}`} type="button" className="btn ghost sm" onClick={() => go("registry")}>
              {p.name} v{p.version}
            </button>
          ))}
        </div>
      )}
      <Progress job={job} status={status} position={position} />
      {job.state === "stopped" && <div className="notice">Job đã bị dừng giữa chừng nên không có checkpoint.</div>}
      {job.state === "error" && <div className="notice bad">{job.error || status.message}</div>}
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
function PublishForm({ job, onDone }) {
  const { busy, run } = useApp();
  const [info, setInfo] = useState(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  useEffect(() => {
    api(`/api/finetune/jobs/${job.id}/publish`)
      .then((data) => {
        setInfo(data);
        setName(data.name);
      })
      .catch(() => setInfo({ eval: null }));
  }, [job.id]);

  async function submit() {
    const ok = await run(`/api/finetune/jobs/${job.id}/publish`, { method: "POST", body: { name, description } }, "Đang xuất bản…");
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

function Dashboard({ task, ui, metrics }) {
  const train = metrics.filter((m) => m.kind === "train");
  const evals = metrics.filter((m) => m.kind === "eval");
  const lastTrain = train[train.length - 1];
  const lastEval = evals[evals.length - 1];
  const best = evals.reduce((a, b) => (a && a.loss <= b.loss ? a : b), null);

  return (
    <Card title="Chỉ số">
      <div className="tiles">
        <Tile label="Train loss" value={formatMetric(lastTrain?.loss)} sub={lastTrain ? `bước ${lastTrain.step}` : "chưa có"} />
        <Tile label="Eval loss tốt nhất" value={formatMetric(best?.loss)} sub={best ? `bước ${best.step}` : "chưa có"} />
        {ui.headlineTiles(lastEval).map((t) => (
          <Tile key={t.label} label={t.label} value={t.value} sub={lastEval ? `bước ${lastEval.step}` : ""} />
        ))}
        <Tile label="Tốc độ" value={lastTrain ? `${formatMetric(lastTrain.samples_per_s)}/s` : "–"} sub="mẫu mỗi giây" />
        <Tile label="VRAM cao nhất" value={lastTrain ? `${formatMetric(lastTrain.vram_gb)} GB` : "–"} />
      </div>
      <div className="charts">
        {task.charts.map((c) => (
          <LineChart key={c.id} title={c.title} hint={c.hint} unit={c.unit} series={seriesOf(metrics, c)} />
        ))}
      </div>
      {evals.length > 0 && (
        <details>
          <summary className="small muted">Bảng số liệu các lần chấm val ({evals.length})</summary>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th className="num">Bước</th>
                  <th className="num">Epoch</th>
                  {ui.EVAL_COLUMNS.map(([key, label]) => (
                    <th key={key} className="num">
                      {label}
                    </th>
                  ))}
                  <th />
                </tr>
              </thead>
              <tbody>
                {evals.map((e) => (
                  <tr key={e.step}>
                    <td className="num">{e.step}</td>
                    <td className="num">{e.epoch}</td>
                    {ui.EVAL_COLUMNS.map(([key, , unit]) => (
                      <td key={key} className="num">
                        {formatMetric(e[key], unit)}
                      </td>
                    ))}
                    <td>{e === best && <Chip kind="ok">tốt nhất</Chip>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </Card>
  );
}

function BuildCard({ job }) {
  return (
    <Card title="Dữ liệu và tham số">
      <div className="row wrap small">
        <span className="strong">{job.build.title}</span>
        <Chip>{job.build.train} train</Chip>
        <Chip>{job.build.val} val</Chip>
      </div>
      <details>
        <summary className="small muted">Tham số train</summary>
        <div className="row wrap small">
          {Object.entries(job.params).map(([k, v]) => (
            <Chip key={k}>
              {k} = {String(v)}
            </Chip>
          ))}
        </div>
      </details>
    </Card>
  );
}

export default function JobDetail({ jobId, tasks, onDeleted }) {
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
  const ui = TASK_UI[job.task];
  if (!task || !ui) {
    return (
      <Card>
        <Empty title={`Không hỗ trợ loại "${job.task}"`} />
      </Card>
    );
  }
  const final = metrics.find((m) => m.kind === "final");

  return (
    <div className="stack">
      <Header job={job} status={status} position={position} onDeleted={onDeleted} reload={reload} />
      <Dashboard task={task} ui={ui} metrics={metrics} />
      {final && (
        <Card title={`Sau train (checkpoint tốt nhất: bước ${final.best_step})`}>
          <ui.FinalSummary final={final} />
        </Card>
      )}
      <BuildCard job={job} />
    </div>
  );
}
