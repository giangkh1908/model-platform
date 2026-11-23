// Build dataset (manifest): template + các cặp (policy, dataset SFT) cho phần train + tuỳ chọn các cặp
// (policy, dataset Eval) cho phần eval. Mỗi dòng được ghép thành prompt hoàn chỉnh: 1 mẫu = 1 dòng.
import { useEffect, useState } from "react";
import { CopyPlus, Hammer, Play, Plus, Trash2, X } from "lucide-react";
import { Card, Chip, Field, Spinner } from "../components/ui.jsx";
import { api, formatTime, percent } from "../lib.js";
import { useApp } from "../store.jsx";
import { LabelTable, NEW, SideList, Tile, useList, useSelected } from "./common.jsx";

const VAL_MIN = 5;
const VAL_MAX = 30;
const VAL_DEFAULT = 10;
const SPLIT_SEED = 42;
const PREVIEW_DEBOUNCE_MS = 400;

export function buildTitle(b) {
  return `${b.name} v${b.version}`;
}

// Chọn một prompt theo phiên bản: value = {id, version}
function PromptPicker({ items, value, onChange, placeholder }) {
  const item = items.find((p) => p.id === value?.id);
  return (
    <div className="input-row">
      <select value={value?.id || ""} onChange={(e) => onChange(e.target.value ? { id: e.target.value, version: null } : null)}>
        <option value="">{placeholder}</option>
        {items.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>
      <select
        className="auto"
        value={value?.version || ""}
        disabled={!item}
        onChange={(e) => onChange({ id: value.id, version: Number(e.target.value) || null })}
      >
        <option value="">{item ? `mới nhất (v${item.latest})` : "phiên bản"}</option>
        {(item?.versions || []).map((v) => (
          <option key={v.version} value={v.version}>
            v{v.version}
          </option>
        ))}
      </select>
    </div>
  );
}

// Danh sách cặp (policy, dataset) của một phần.
function EntryList({ entries, onChange, datasets, policies, emptyText }) {
  function set(i, patch) {
    onChange(entries.map((e, j) => (j === i ? { ...e, ...patch } : e)));
  }
  return (
    <div className="stack">
      {entries.length === 0 && <span className="small muted">{emptyText}</span>}
      {entries.map((e, i) => (
        <div key={i} className="entry-row">
          <select value={e.dataset_id} onChange={(ev) => set(i, { dataset_id: ev.target.value })}>
            <option value="">-- Dataset --</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} · {d.rows} dòng
              </option>
            ))}
          </select>
          <PromptPicker items={policies} value={e.policy} placeholder="-- Policy --" onChange={(policy) => set(i, { policy })} />
          <button type="button" className="icon-btn" title="Bỏ cặp này" onClick={() => onChange(entries.filter((_, j) => j !== i))}>
            <X size={16} />
          </button>
        </div>
      ))}
      <div>
        <button type="button" className="btn sm" onClick={() => onChange([...entries, { dataset_id: "", policy: null }])}>
          <Plus size={14} /> Thêm cặp policy + dataset
        </button>
      </div>
    </div>
  );
}

function BuildStats({ stats }) {
  const warnings = [];
  if (stats.conflicts) warnings.push(`${stats.conflicts} prompt giống nhau nhưng khác nhãn: nên kiểm tra lại nhãn.`);
  if (stats.leaked) warnings.push(`${stats.leaked} mẫu eval có prompt y hệt trong phần train: điểm eval sẽ bị thổi phồng.`);
  if (stats.missing_in_val.length) warnings.push(`${stats.missing_in_val.length} nhãn không có mẫu nào trong val: ${stats.missing_in_val.join(", ")}.`);
  return (
    <div className="stack">
      <div className="tiles">
        <Tile label="Train" value={stats.train} sub="mẫu (prompt + dòng)" />
        <Tile label="Val" value={stats.val} sub={stats.val_from === "eval" ? "từ dataset Eval" : `ngẫu nhiên ${percent(stats.val_fraction_actual)}`} />
        <Tile label="Dòng trùng đã bỏ" value={stats.removed_duplicates} />
        <Tile label="Prompt dài nhất" value={`${stats.max_chars}`} sub={`ký tự · trung bình ${stats.avg_chars}`} />
      </div>
      {warnings.map((w) => (
        <div key={w} className="notice warn">
          {w}
        </div>
      ))}
      {stats.strata.length > 0 && (
        <details>
          <summary className="small muted">Phân bố {stats.strata.length} nhãn giữa train và val</summary>
          <LabelTable rows={stats.strata} columns={[["total", "Tổng"], ["train", "Train"], ["val", "Val"]]} />
        </details>
      )}
    </div>
  );
}

function Examples({ examples, config }) {
  if (!examples?.length) return null;
  return (
    <details>
      <summary className="small muted">Xem {examples.length} prompt đã ghép</summary>
      <div className="stack">
        {examples.map((e, i) => (
          <div key={i} className="stack">
            <span className="small muted">
              {e.dataset} · {e.policy} · nhãn <b>{e.label}</b>
            </span>
            <pre className="code">{e.text}</pre>
          </div>
        ))}
        {config && (
          <span className="small muted">
            Output: {config.instructions} · Nhãn: {config.labels.join(", ")}
          </span>
        )}
      </div>
    </details>
  );
}

function usePreview(body, ready) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const key = ready ? JSON.stringify(body) : "";
  useEffect(() => {
    setResult(null);
    setError("");
    if (!key) return undefined;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        setResult(await api("/api/finetune/builds/preview", { body: JSON.parse(key), signal: controller.signal }));
      } catch (e) {
        if (e.name !== "AbortError") setError(e.message);
      }
    }, PREVIEW_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [key]);
  return { result, error };
}

function entriesFromManifest(list) {
  return list.map((e) => ({ dataset_id: e.dataset.id, policy: { id: e.policy.id, version: e.policy.version } }));
}

function versionHint(name, sameName, nextVersion) {
  const trimmed = name.trim();
  if (!trimmed) return "Cùng tên với build cũ thì thành phiên bản tiếp theo.";
  if (!sameName.length) return `Sẽ tạo "${trimmed} v1".`;
  return `Đã có ${sameName.length} phiên bản: sẽ tạo "${trimmed} v${nextVersion}".`;
}

// `draft`: điền sẵn từ một build cũ khi bấm "Tạo phiên bản mới".
function NewBuild({ tasks, builds, draft, onCreated }) {
  const { busy, run } = useApp();
  const { items: datasets } = useList("/api/finetune/datasets", "datasets", 10000);
  const { items: templates } = useList("/api/finetune/prompts/template", "items", 10000);
  const { items: policies } = useList("/api/finetune/prompts/policy", "items", 10000);
  const taskId = draft?.task || tasks[0].id;
  const manifest = draft?.manifest;
  const [name, setName] = useState(draft?.name || "");
  const [note, setNote] = useState("");
  const [template, setTemplate] = useState(manifest ? { id: manifest.template.id, version: manifest.template.version } : null);
  const [entries, setEntries] = useState(manifest ? entriesFromManifest(manifest.entries) : [{ dataset_id: "", policy: null }]);
  const [evalEntries, setEvalEntries] = useState(manifest ? entriesFromManifest(manifest.eval_entries) : []);
  const [dedupe, setDedupe] = useState(draft ? draft.dedupe : true);
  const [valPercent, setValPercent] = useState(draft?.val_percent || VAL_DEFAULT);

  const complete = (list) => list.every((e) => e.dataset_id && e.policy?.id);
  const ready = Boolean(template?.id) && entries.length > 0 && complete(entries) && complete(evalEntries);
  const body = { task: taskId, template, entries, eval_entries: evalEntries, dedupe, val_percent: valPercent, split_seed: SPLIT_SEED };
  const { result, error } = usePreview(body, ready);
  if (!datasets || !templates || !policies) return <Card title="Đang tải…" />;

  const ofTask = datasets.filter((d) => d.task === taskId);
  const sft = ofTask.filter((d) => d.source === "sft");
  const evalSets = ofTask.filter((d) => d.source === "eval");
  const sameName = builds.filter((b) => b.task === taskId && b.name.toLowerCase() === name.trim().toLowerCase());
  const nextVersion = Math.max(0, ...sameName.map((b) => b.version)) + 1;

  let problem = "";
  if (!name.trim()) problem = "Đặt tên cho build.";
  else if (!template?.id) problem = "Chọn template.";
  else if (!ready) problem = "Chọn đủ dataset và policy cho mỗi cặp.";
  else if (error) problem = error;
  else if (!result) problem = "Đang ghép thử…";

  async function submit() {
    const created = await run("/api/finetune/builds", { body: { ...body, name, note } });
    if (created) onCreated(created.id);
  }

  return (
    <Card title={draft ? `Phiên bản mới của "${draft.name}"` : "Build mới"}>
      <p className="small muted">
        Mỗi mẫu = template + policy + một dòng dataset. Mỗi dataset đi với đúng một policy (nhãn của dataset là theo policy đó).
      </p>
      <div className="form-grid">
        <Field label="Tên build" hint={versionHint(name, sameName, nextVersion)}>
          <input value={name} maxLength={60} placeholder="vd. moderation" onChange={(e) => setName(e.target.value)} />
        </Field>
        <Field label="Ghi chú" hint="Phiên bản này thay đổi gì (không bắt buộc).">
          <input value={note} maxLength={300} onChange={(e) => setNote(e.target.value)} />
        </Field>
      </div>

      <div className="subhead">1. Template</div>
      {templates.length === 0 && <span className="small muted">Chưa có template: tạo ở mục Prompt.</span>}
      <PromptPicker items={templates} value={template} placeholder="-- Template --" onChange={setTemplate} />

      <div className="subhead">2. Phần train: policy + dataset SFT</div>
      <EntryList entries={entries} onChange={setEntries} datasets={sft} policies={policies} emptyText="Thêm ít nhất một cặp." />
      <label className="check">
        <input type="checkbox" checked={dedupe} onChange={(e) => setDedupe(e.target.checked)} />
        Bỏ mẫu trùng y hệt (giữ một bản)
      </label>

      <div className="subhead">3. Phần eval: policy + dataset Eval (không bắt buộc)</div>
      <p className="small muted">Có phần eval thì nó làm val khi train và tự được chấm sau khi train. Không có thì chia val ngẫu nhiên từ phần train.</p>
      <EntryList entries={evalEntries} onChange={setEvalEntries} datasets={evalSets} policies={policies} emptyText="Không có: val chia ngẫu nhiên." />
      {evalEntries.length === 0 && (
        <div className="split-slider">
          <span className="field-label">Tỉ lệ val</span>
          <input type="range" min={VAL_MIN} max={VAL_MAX} value={valPercent} onChange={(e) => setValPercent(Number(e.target.value))} />
          <b className="num">{valPercent}%</b>
          <span className="small muted">train {100 - valPercent}%</span>
        </div>
      )}

      {ready && !result && !error && (
        <span className="small muted">
          <Spinner /> Đang ghép thử…
        </span>
      )}
      {error && <div className="notice bad">{error}</div>}
      {result && <BuildStats stats={result} />}
      {result && <Examples examples={result.examples} config={result.config} />}

      <div className="row wrap submit-row">
        <button type="button" className="btn primary" disabled={busy || Boolean(problem)} onClick={submit}>
          <Hammer size={14} /> Build
        </button>
        {problem && <span className="small muted">{problem}</span>}
      </div>
    </Card>
  );
}

function ManifestView({ manifest }) {
  function part(list) {
    return list.map((e) => (
      <div key={`${e.dataset.id}:${e.policy.id}`} className="row wrap small">
        <Chip>{e.policy.name} v{e.policy.version}</Chip>+<Chip>{e.dataset.name}</Chip>
        <span className="muted">{e.dataset.rows} mẫu</span>
      </div>
    ));
  }
  return (
    <div className="stack">
      <div className="row wrap small">
        <span className="muted">Template:</span>
        <Chip>
          {manifest.template.name} v{manifest.template.version}
        </Chip>
      </div>
      <div className="small muted">Phần train</div>
      {part(manifest.entries)}
      {manifest.eval_entries.length > 0 && (
        <>
          <div className="small muted">Phần eval</div>
          {part(manifest.eval_entries)}
        </>
      )}
    </div>
  );
}

function BuildDetail({ id, onDeleted, onTrain, onNewVersion }) {
  const { busy, run, confirm } = useApp();
  const [data, setData] = useState(null);
  useEffect(() => {
    setData(null);
    api(`/api/finetune/builds/${id}`).then(setData).catch(() => {});
  }, [id]);
  if (!data) return <Card title="Đang tải…" />;
  const b = data.build;

  async function remove() {
    const ok = await confirm({
      title: `Xoá build "${buildTitle(b)}"?`,
      message: "Các job đã train từ build này vẫn giữ nguyên dữ liệu của chúng.",
      confirmLabel: "Xoá",
      danger: true,
    });
    if (ok && (await run(`/api/finetune/builds/${id}`, { method: "DELETE" }))) onDeleted();
  }

  return (
    <Card>
      <div className="row between wrap">
        <div className="job-title">{buildTitle(b)}</div>
        <div className="row wrap">
          <button type="button" className="btn sm" onClick={() => onNewVersion(b)} title="Điền sẵn manifest của build này">
            <CopyPlus size={14} /> Tạo phiên bản mới
          </button>
          <button type="button" className="btn sm primary" onClick={() => onTrain(b.id)}>
            <Play size={14} /> Train từ build này
          </button>
          <button type="button" className="btn sm danger" disabled={busy} onClick={remove}>
            <Trash2 size={14} /> Xoá
          </button>
        </div>
      </div>
      <div className="row wrap small muted">
        <span>tạo lúc {formatTime(b.created)}</span>
        <span>· {b.dedupe ? "đã bỏ mẫu trùng" : "giữ mẫu trùng"}</span>
        {b.has_eval && <Chip kind="ok">có phần eval: tự eval sau khi train</Chip>}
      </div>
      {b.note && <div className="notice">{b.note}</div>}
      <ManifestView manifest={b.manifest} />
      <BuildStats stats={b.stats} />
      <Examples examples={data.examples} config={b.task_config} />
    </Card>
  );
}

export default function Builds({ tasks, onTrain }) {
  const { items, reload } = useList("/api/finetune/builds", "builds", 10000);
  const [selected, select] = useSelected("finetune.build");
  const [draft, setDraft] = useState(null);
  if (!items) return <Card title="Đang tải…" />;
  const view = items.some((b) => b.id === selected) ? selected : NEW;

  return (
    <div className="ft-layout">
      <SideList
        title="Build"
        newLabel="Build mới"
        items={items}
        view={view}
        onSelect={(id) => {
          setDraft(null);
          select(id);
        }}
        empty="Chưa có build nào."
        renderRow={(b) => (
          <>
            <span className="ft-job-name">{buildTitle(b)}</span>
            <span className="small muted">
              {b.stats.train} train · {b.stats.val} val{b.has_eval ? " (eval)" : ""} · {formatTime(b.created)}
            </span>
          </>
        )}
      />
      <div className="stack">
        {view === NEW ? (
          <NewBuild
            key={draft?.id || "blank"}
            tasks={tasks}
            builds={items}
            draft={draft}
            onCreated={(id) => {
              setDraft(null);
              select(id);
              reload();
            }}
          />
        ) : (
          <BuildDetail
            id={view}
            onTrain={onTrain}
            onNewVersion={(b) => {
              setDraft(b);
              select(NEW);
            }}
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
