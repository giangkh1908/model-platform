// Tạo job train: chọn build + model + cấu hình riêng của loại + tham số, rồi thêm vào hàng đợi.
// Không biết gì về loại cụ thể: form riêng lấy từ TASK_UI, tham số lấy từ backend.
import { useState } from "react";
import { ListPlus } from "lucide-react";
import { Card, Chip, Field } from "../components/ui.jsx";
import { formatTime } from "../lib.js";
import { useApp } from "../store.jsx";
import { buildTitle } from "./Builds.jsx";
import { useList } from "./common.jsx";

// Trang Build dataset ghi khoá này để trang Finetune mở form job mới với build đó.
export const TRAIN_BUILD_KEY = "finetune.trainBuild";

function defaultsOf(task) {
  return Object.fromEntries(task.params.map((p) => [p.key, p.default]));
}

function ParamInput({ spec, value, onChange }) {
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
        value={value}
        onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
      />
    </Field>
  );
}

function ParamFields({ specs, values, onChange }) {
  const basic = specs.filter((p) => !p.advanced);
  const advanced = specs.filter((p) => p.advanced);

  function set(key, value) {
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

// Form theo một loại cụ thể; đổi sang build của loại khác thì form được tạo lại (key = task id).
function JobForm({ task, build, builds, setBuildId, onCreated }) {
  const { status, busy, run, confirm } = useApp();
  const [params, setParams] = useState(() => defaultsOf(task));
  const [baseModel, setBaseModel] = useState(task.base_models[0] || "");
  const [name, setName] = useState("");

  let problem = "";
  if (!build) problem = "Chọn build.";
  else if (!baseModel) problem = "Chưa có model nào train được bằng loại này.";

  async function submit() {
    if (status?.state !== "stopped") {
      const ok = await confirm({
        title: "Model đang chạy sẽ bị tắt",
        message: "Khi tới lượt train, nền tảng tắt model đang chạy để nhường GPU. Trong lúc hàng đợi còn việc thì không bật được model.",
        confirmLabel: "Thêm vào hàng đợi",
      });
      if (!ok) return;
    }
    const body = { name, build_id: build.id, base_model: baseModel, params };
    const result = await run("/api/finetune/jobs", { body });
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
          Prompt lấy từ build: template <b>{build.manifest.template.name} v{build.manifest.template.version}</b> · output "
          {build.task_config.instructions}" · nhãn {build.task_config.labels.join(", ")}
          {build.has_eval ? " · có phần eval: train xong tự eval so với model gốc." : ""}
        </div>
      )}
      <div className="subhead">Tham số train</div>
      <p className="small muted">Trong lúc train, model được chấm trên phần val của build theo "Chấm val mỗi (bước)" để vẽ eval loss.</p>
      <ParamFields specs={task.params} values={params} onChange={setParams} />
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

export default function NewJob({ tasks, buildId, setBuildId, onCreated, onOpenBuilds }) {
  const { items: builds } = useList("/api/finetune/builds", "builds", 10000);
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
  const build = builds.find((b) => b.id === buildId) || null;
  const task = tasks.find((t) => t.id === (build?.task || tasks[0].id));
  return <JobForm key={task.id} task={task} build={build} builds={builds} setBuildId={setBuildId} onCreated={onCreated} />;
}
