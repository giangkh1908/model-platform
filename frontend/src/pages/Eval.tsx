// Trang Eval: chấm nhiều model (model gốc và các adapter đã train) trên cùng một dataset golden rồi xếp hạng.
// Tách riêng với phần chấm val định kỳ trong lúc train. Báo cáo riêng của từng loại lấy từ TaskUI (finetune/tasks.ts).
import { useEffect, useState } from "react";
import { ClipboardCheck, ListPlus, Square, Trash2 } from "lucide-react";
import * as ft from "../api/finetune.ts";
import { Bar, Card, Empty, Field, Segmented, Spinner } from "../components/ui.tsx";
import { LIST_POLL_MS, RUN_POLL_MS, STORAGE_KEYS } from "../config/app.ts";
import { NEW, SideList, StateChip, isActive, useList, useSelected, useTasks } from "../finetune/common.tsx";
import { taskUiOf } from "../finetune/tasks.ts";
import { loadLocal, saveLocal } from "../lib/browser.ts";
import { formatTime } from "../lib/format.ts";
import { useApp, useInterval } from "../store.tsx";
import type { Build, EvalChoices, EvalDetail as EvalDetailData, Task } from "../types/api.ts";

const POLL_MS = 2000;

type Adapter = EvalChoices["adapters"][number];

function AdapterRow({ adapter, on, onToggle }: { adapter: Adapter; on: boolean; onToggle: () => void }) {
  return (
    <label className={`check-row ${on ? "on" : ""}`}>
      <input type="checkbox" checked={on} onChange={onToggle} />
      <span className="grow">
        <b>{adapter.name}</b>
        <span className="small muted" style={{ display: "block" }}>
          build {adapter.build} · {formatTime(adapter.created)}
          {adapter.base_model !== adapter.root_base ? ` · train tiếp từ ${adapter.base_model}` : ""}
        </span>
      </span>
    </label>
  );
}

function buildHint(build: Build | undefined): string {
  if (!build) return "Dữ liệu eval lấy từ build (bộ quyết định + dataset). Tạo ở mục Build dataset.";
  if (build.has_eval) return `${build.stats.val} mẫu từ phần eval của build.`;
  return `Build không có phần eval: dùng ${build.stats.val} mẫu val.`;
}

// Form theo một loại cụ thể (key = task id để tạo lại khi đổi loại).
function EvalForm({ task, onCreated }: { task: Task; onCreated: (id: string) => void }) {
  const { status, busy, run, confirm } = useApp();
  const { items: builds } = useList(ft.listBuilds, LIST_POLL_MS);
  const [choices, setChoices] = useState<EvalChoices | null>(null);
  const [baseId, setBaseId] = useState("");
  const [buildId, setBuildId] = useState("");
  const [chosen, setChosen] = useState<string[]>([]);
  const [name, setName] = useState("");

  useEffect(() => {
    ft.evalChoices(task.id)
      .then((data) => {
        setChoices(data);
        // Tới từ nút "Eval model này": chọn sẵn model gốc của adapter đó và tick adapter.
        const preselect = loadLocal<string | null>(STORAGE_KEYS.evalPreselect, null);
        saveLocal(STORAGE_KEYS.evalPreselect, null);
        const adapter = data.adapters.find((a) => a.ref === preselect);
        if (adapter) {
          setBaseId(adapter.root_base);
          setChosen([adapter.ref]);
        } else if (data.bases.length) {
          setBaseId(data.bases[0].id);
        }
      })
      .catch(() => setChoices({ bases: [], adapters: [] }));
  }, [task.id]);

  if (!builds || !choices) return <Card title="Đang tải…" />;
  const available = builds.filter((b) => b.task === task.id);
  const build = available.find((b) => b.id === buildId);
  const adapters = choices.adapters.filter((a) => a.root_base === baseId);

  function changeBase(id: string) {
    setBaseId(id);
    setChosen([]);
  }

  function toggle(ref: string) {
    setChosen((list) => (list.includes(ref) ? list.filter((x) => x !== ref) : [...list, ref]));
  }

  let problem = "";
  if (!baseId) problem = "Chưa có model gốc nào.";
  else if (!build) problem = "Chọn build có dữ liệu eval.";

  async function submit() {
    if (status?.state !== "stopped") {
      const ok = await confirm({
        title: "Model đang chạy sẽ bị tắt",
        message: "Khi tới lượt eval, nền tảng tắt model đang chạy để nhường GPU.",
        confirmLabel: "Thêm vào hàng đợi",
      });
      if (!ok) return;
    }
    const body = { name, build_id: buildId, base_model: baseId, adapters: chosen };
    const result = await run(() => ft.createEval(body));
    if (result) onCreated(result.id);
  }

  let summary = `Chỉ chấm ${baseId}.`;
  if (chosen.length) summary = `So ${chosen.length} adapter với ${baseId}.`;

  return (
    <Card title="Eval mới">
      <p className="small muted">
        Chọn model gốc, rồi chọn các adapter đã train từ nó để so sánh. Không chọn adapter nào thì chỉ chấm model gốc. Mọi model được chấm
        trên đúng dữ liệu và bộ quyết định của build (mỗi adapter dùng đúng phiên bản prompt lúc nó được train).
      </p>
      <div className="form-grid">
        <Field label="1. Model gốc">
          <select value={baseId} onChange={(e) => changeBase(e.target.value)}>
            {choices.bases.map((b) => (
              <option key={b.id} value={b.id}>
                {b.id}
              </option>
            ))}
          </select>
        </Field>
        <Field label="2. Build (dữ liệu eval)" hint={buildHint(build)}>
          <select value={buildId} onChange={(e) => setBuildId(e.target.value)}>
            <option value="">-- Chọn build --</option>
            {available.map((b) => (
              <option key={b.id} value={b.id}>
                {b.name} v{b.version} · {b.stats.val} mẫu {b.has_eval ? "eval" : "val"}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <div className="subhead">3. Adapter train từ {baseId || "model gốc"} (không bắt buộc)</div>
      {adapters.length === 0 && <span className="small muted">Chưa có adapter nào train xong từ model này.</span>}
      <div className="check-list">
        {adapters.map((a) => (
          <AdapterRow key={a.ref} adapter={a} on={chosen.includes(a.ref)} onToggle={() => toggle(a.ref)} />
        ))}
      </div>

      <Field label="Tên lần eval" hint="Để trống thì tự đặt.">
        <input value={name} maxLength={80} onChange={(e) => setName(e.target.value)} />
      </Field>

      <div className="row wrap submit-row">
        <button type="button" className="btn primary" disabled={busy || Boolean(problem)} onClick={submit}>
          <ListPlus size={14} /> Thêm vào hàng đợi
        </button>
        <span className="small muted">{problem || summary}</span>
      </div>
    </Card>
  );
}

function NewEval({ tasks, onCreated }: { tasks: Task[]; onCreated: (id: string) => void }) {
  const [taskId, setTaskId] = useState(tasks[0].id);
  const task = tasks.find((t) => t.id === taskId) ?? tasks[0];
  return (
    <div className="stack">
      {tasks.length > 1 && <Segmented value={taskId} onChange={setTaskId} options={tasks.map((t) => ({ id: t.id, label: t.name }))} />}
      <EvalForm key={task.id} task={task} onCreated={onCreated} />
    </div>
  );
}

function EvalDetail({ id, onDeleted }: { id: string; onDeleted: () => void }) {
  const { busy, run, confirm } = useApp();
  const [data, setData] = useState<EvalDetailData | null>(null);

  useEffect(() => {
    setData(null);
    pull();
    // pull() của lần render này đã dùng đúng id mới
  }, [id]);

  async function pull() {
    try {
      const fresh = await ft.getEval(id);
      if (fresh.eval.id === id) setData(fresh);
    } catch {
      // mất kết nối tạm thời: lần sau thử lại
    }
  }
  const live = !data || isActive(data.eval.state);
  useInterval(pull, live ? POLL_MS : null);

  if (!data) return <Card title="Đang tải…" />;
  const ev = data.eval;
  const status = data.status;
  const ui = taskUiOf(ev.task);

  async function stop() {
    await run(() => ft.stopEval(id));
    pull();
  }

  async function remove() {
    const ok = await confirm({ title: `Xoá lần eval "${ev.name}"?`, confirmLabel: "Xoá", danger: true });
    if (ok && (await run(() => ft.deleteEval(id)))) onDeleted();
  }

  return (
    <div className="stack">
      <Card>
        <div className="row between wrap">
          <div className="job-title">{ev.name}</div>
          <div className="row wrap">
            {isActive(ev.state) && (
              <button type="button" className="btn sm danger" disabled={busy} onClick={stop}>
                <Square size={14} /> {ev.state === "queued" ? "Bỏ khỏi hàng đợi" : "Dừng"}
              </button>
            )}
            {!isActive(ev.state) && (
              <button type="button" className="btn sm danger" disabled={busy} onClick={remove}>
                <Trash2 size={14} /> Xoá
              </button>
            )}
          </div>
        </div>
        <div className="row wrap small">
          <StateChip state={ev.state} />
          <span className="muted">
            dataset <b>{ev.source.title}</b> · {ev.n} câu · {ev.models.length} model · {formatTime(ev.created)}
          </span>
        </div>
        {ev.state === "queued" && (
          <div className="notice">Đang chờ trong hàng đợi{data.queue_position ? `: thứ ${data.queue_position}` : ""}.</div>
        )}
        {ev.state === "running" && (
          <div className="stack">
            <span className="small">
              <Spinner /> {status.message ?? "Đang chấm…"}
            </span>
            <Bar fraction={status.total ? (status.done ?? 0) / status.total : 0} />
          </div>
        )}
        {ev.state === "error" && <div className="notice bad">{ev.error}</div>}
        {ev.models
          .filter((m) => m.leaked > 0)
          .map((m) => (
            <div key={m.key} className="notice warn">
              {m.leaked}/{ev.n} câu golden có y hệt trong dữ liệu train của "{m.name}": điểm của model này bị thổi phồng.
            </div>
          ))}
      </Card>
      {data.report && ui && (
        <Card title="Kết quả">
          <ui.EvalReport report={data.report} />
        </Card>
      )}
    </div>
  );
}

export default function Eval() {
  const { tasks, error } = useTasks();
  const { items, reload } = useList(ft.listEvals, RUN_POLL_MS);
  const [selected, select] = useSelected("eval.selected");

  // Tới từ nút "Eval model này": mở form mới với model đó.
  useEffect(() => {
    if (loadLocal<string | null>(STORAGE_KEYS.evalPreselect, null)) select(NEW);
  }, []);

  if (!tasks || !items) {
    return (
      <Card>
        <Empty icon={ClipboardCheck} title={error || "Đang tải…"} />
      </Card>
    );
  }
  const view = items.some((e) => e.id === selected) ? selected : NEW;

  return (
    <div className="ft ft-layout">
      <SideList
        title="Lần eval"
        newLabel="Eval mới"
        items={items}
        view={view}
        onSelect={select}
        empty="Chưa có lần eval nào."
        renderRow={(e) => (
          <>
            <div className="row between">
              <span className="ft-job-name">{e.name}</span>
              <StateChip state={e.state} />
            </div>
            <span className="small muted">
              {e.n} câu · {e.models.length} model · {formatTime(e.created)}
            </span>
          </>
        )}
        footer="Eval chạy chung hàng đợi GPU với train."
      />
      <div className="stack">
        {view === NEW ? (
          <NewEval
            tasks={tasks}
            onCreated={(id) => {
              select(id);
              reload();
            }}
          />
        ) : (
          <EvalDetail
            id={view}
            onDeleted={() => {
              select(NEW);
              reload();
            }}
          />
        )}
      </div>
    </div>
  );
}
