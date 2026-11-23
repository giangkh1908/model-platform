// Tạo job train: chọn build + model + cấu hình riêng của loại + tham số, rồi thêm vào hàng đợi.
// Không biết gì về loại cụ thể: tham số do backend khai báo (Task.params), form tự vẽ theo đó.
import { useState } from "react";
import { ListPlus } from "lucide-react";
import * as ft from "../api/finetune.ts";
import { Card, Chip, Field, Segmented } from "../components/ui.tsx";
import { LIST_POLL_MS } from "../config/app.ts";
import { formatTime } from "../lib/format.ts";
import { useApp } from "../store.tsx";
import type { Build, Param, ParamValues, Task } from "../types/api.ts";
import { buildTitle } from "./Builds.tsx";
import { useList } from "./common.tsx";

type ParamValue = ParamValues[string];
type ParamMode = "form" | "yaml";

function defaultsOf(task: Task): ParamValues {
  return Object.fromEntries(task.params.map((p) => [p.key, p.default]));
}

function ParamInput({ spec, value, onChange }: { spec: Param; value: ParamValue; onChange: (value: ParamValue) => void }) {
  if (spec.kind === "bool") {
    return (
      <label className="check" title={spec.hint}>
        <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
        {spec.label}
      </label>
    );
  }
  return (
    <Field label={spec.label} hint={spec.hint}>
      <input
        type="number"
        min={spec.min}
        max={spec.max}
        step={spec.step}
        value={typeof value === "boolean" ? Number(value) : value}
        onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
      />
    </Field>
  );
}

function ParamFields({ specs, values, onChange }: { specs: Param[]; values: ParamValues; onChange: (values: ParamValues) => void }) {
  const basic = specs.filter((p) => !p.advanced);
  const advanced = specs.filter((p) => p.advanced);

  function set(key: string, value: ParamValue) {
    onChange({ ...values, [key]: value });
  }

  return (
    <div className="stack">
      <div className="form-grid three">
        {basic.map((p) => (
          <ParamInput key={p.key} spec={p} value={values[p.key]} onChange={(v) => set(p.key, v)} />
        ))}
      </div>
      <details>
        <summary className="small muted">Tham số nâng cao</summary>
        <div className="form-grid three">
          {advanced.map((p) => (
            <ParamInput key={p.key} spec={p} value={values[p.key]} onChange={(v) => set(p.key, v)} />
          ))}
        </div>
      </details>
    </div>
  );
}

const PARAM_MODES: { id: ParamMode; label: string }[] = [
  { id: "form", label: "Form" },
  { id: "yaml", label: "YAML" },
];

// Tham số dạng YAML: backend xuất YAML có chú thích từ giá trị trên form và đọc ngược lại (cùng luật kiểm tra với form).
function useParamsYaml(task: Task, params: ParamValues, setParams: (values: ParamValues) => void) {
  const [mode, setMode] = useState<ParamMode>("form");
  const [text, setText] = useState("");
  const [error, setError] = useState("");

  async function toParams() {
    const result = await ft.paramsFromYaml(task.id, text);
    setParams(result);
    return result;
  }

  async function switchMode(next: ParamMode) {
    setError("");
    try {
      if (next === "yaml") {
        setText(await ft.paramsToYaml(task.id, params));
      } else {
        await toParams();
      }
      setMode(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  // Tham số cuối cùng để gửi: ở chế độ YAML thì đọc từ YAML (lỗi thì hiện và trả về null).
  async function current(): Promise<ParamValues | null> {
    if (mode === "form") return params;
    try {
      setError("");
      return await toParams();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  }

  return { mode, switchMode, text, setText, error, current };
}

// Form theo một loại cụ thể; đổi sang build của loại khác thì form được tạo lại (key = task id).
interface JobFormProps {
  task: Task;
  build: Build | null;
  builds: Build[];
  setBuildId: (id: string) => void;
  onCreated: (jobId: string) => void;
}

function JobForm({ task, build, builds, setBuildId, onCreated }: JobFormProps) {
  const { status, busy, run, confirm } = useApp();
  const [params, setParams] = useState<ParamValues>(() => defaultsOf(task));
  const [baseModel, setBaseModel] = useState(task.base_models[0] ?? "");
  const [name, setName] = useState("");
  const paramsYaml = useParamsYaml(task, params, setParams);

  let problem = "";
  if (!build) problem = "Chọn build.";
  else if (!baseModel) problem = "Chưa có model nào train được bằng loại này.";

  async function submit() {
    const finalParams = await paramsYaml.current();
    if (!finalParams || !build) return;
    if (status?.state !== "stopped") {
      const ok = await confirm({
        title: "Model đang chạy sẽ bị tắt",
        message: "Khi tới lượt train, nền tảng tắt model đang chạy để nhường GPU. Trong lúc hàng đợi còn việc thì không bật được model.",
        confirmLabel: "Thêm vào hàng đợi",
      });
      if (!ok) return;
    }
    const body = { name, build_id: build.id, base_model: baseModel, params: finalParams };
    const result = await run(() => ft.createJob(body));
    if (result) onCreated(result.id);
  }

  let buildHint = "Tạo ở mục Build dataset.";
  if (build) buildHint = `${build.stats.train} câu train · ${build.stats.val} câu val`;

  return (
    <Card title="Job train mới">
      <div className="form-grid">
        <Field label="Build (dữ liệu)" hint={buildHint}>
          <select value={build?.id || ""} onChange={(e) => setBuildId(e.target.value)}>
            <option value="">-- Chọn build --</option>
            {builds.map((b) => (
              <option key={b.id} value={b.id}>
                {buildTitle(b)} · {formatTime(b.created)}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Model" hint="Model gốc, hoặc adapter đã thêm vào danh sách Model để train tiếp từ nó.">
          <select value={baseModel} onChange={(e) => setBaseModel(e.target.value)}>
            {task.base_models.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Tên job" hint="Để trống thì tự đặt theo build và model.">
          <input value={name} maxLength={80} onChange={(e) => setName(e.target.value)} />
        </Field>
      </div>
      {build && (
        <div className="notice small">
          Bộ quyết định từ build: system <b>{build.manifest.spec_title.system}</b>
          {build.manifest.spec_title.intents ? ` · intent ${build.manifest.spec_title.intents}` : ""}
          {build.manifest.spec_title.gates ? ` · gate ${build.manifest.spec_title.gates}` : ""}
          {build.has_eval ? " · có phần eval: train xong tự eval so với model gốc." : ""}
        </div>
      )}
      <div className="row between wrap">
        <div className="subhead">Tham số train</div>
        <Segmented value={paramsYaml.mode} options={PARAM_MODES} onChange={paramsYaml.switchMode} />
      </div>
      <p className="small muted">
        Trong lúc train, model được chấm trên phần val của build theo "Chấm val mỗi (bước)" để vẽ eval loss. Tiến độ và mọi chỉ số xem trực tiếp trên
        MLflow (link ở trang job).
      </p>
      {paramsYaml.mode === "form" && <ParamFields specs={task.params} values={params} onChange={setParams} />}
      {paramsYaml.mode === "yaml" && (
        <Field label="params.yaml" hint="Sửa giá trị; xoá một dòng thì dùng giá trị mặc định. Chuyển về Form để xem lại từng ô.">
          <textarea className="code-input" rows={22} spellCheck={false} value={paramsYaml.text} onChange={(e) => paramsYaml.setText(e.target.value)} />
        </Field>
      )}
      {paramsYaml.error && <div className="notice bad small">{paramsYaml.error}</div>}
      <div className="row wrap submit-row">
        <button type="button" className="btn primary" disabled={busy || Boolean(problem)} onClick={submit}>
          <ListPlus size={14} /> Thêm vào hàng đợi
        </button>
        {problem && <span className="small muted">{problem}</span>}
        {!problem && <Chip>{task.name}</Chip>}
      </div>
    </Card>
  );
}

interface NewJobProps {
  tasks: Task[];
  buildId: string;
  setBuildId: (id: string) => void;
  onCreated: (jobId: string) => void;
  onOpenBuilds: () => void;
}

export default function NewJob({ tasks, buildId, setBuildId, onCreated, onOpenBuilds }: NewJobProps) {
  const { items: builds } = useList(ft.listBuilds, LIST_POLL_MS);
  if (!builds) return <Card title="Đang tải…" />;
  if (!builds.length) {
    return (
      <Card title="Job train mới">
        <p className="small muted">Chưa có build nào. Train cần một build: thêm dataset ở mục Dataset rồi build ở mục Build dataset.</p>
        <div>
          <button type="button" className="btn primary" onClick={onOpenBuilds}>
            Mở mục Build dataset
          </button>
        </div>
      </Card>
    );
  }
  const build = builds.find((b) => b.id === buildId) ?? null;
  const task = tasks.find((t) => t.id === (build?.task ?? tasks[0].id)) ?? tasks[0];
  return <JobForm key={task.id} task={task} build={build} builds={builds} setBuildId={setBuildId} onCreated={onCreated} />;
}
