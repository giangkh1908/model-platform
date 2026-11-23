// Build dataset (manifest): một bộ quyết định (1 system + các intent + các gate, mỗi prompt theo phiên bản hoặc nhãn
// dev / prod) + các dataset SFT (train) và Eval (val / chấm). Mỗi câu của dataset thành một dòng "như thật": câu người
// dùng, lịch sử, nhãn intent, nhãn từng gate. Ghép thành prompt của model là việc của runtime.
import { useEffect, useState } from "react";
import { CopyPlus, Hammer, Play, Plus, Trash2, X } from "lucide-react";
import * as ft from "../api/finetune.ts";
import { listPrompts } from "../api/prompts.ts";
import { Card, Chip, Field, Spinner } from "../components/ui.tsx";
import { LIST_POLL_MS, PROMPT_LABELS } from "../config/app.ts";
import { formatTime, percent } from "../lib/format.ts";
import { useApp } from "../store.tsx";
import type { Build, BuildBody, BuildPreview, BuildRow, BuildStats, Dataset, PromptRef, PromptSummary, Spec, SpecTitle, Task } from "../types/api.ts";
import { LabelTable, NEW, SideList, Tile, useList, useSelected, type LabelRow } from "./common.tsx";

const VAL_MIN = 5;
const VAL_MAX = 30;
const VAL_DEFAULT = 10;
const SPLIT_SEED = 42;
const PREVIEW_DEBOUNCE_MS = 400;

export function buildTitle(b: Pick<Build, "name" | "version">): string {
  return `${b.name} v${b.version}`;
}

/** Giá trị của ô phiên bản: "" = mới nhất, "label:prod" = theo nhãn, "3" = v3. */
function refValue(ref: PromptRef | null): string {
  if (ref?.label) return `label:${ref.label}`;
  if (ref?.version) return String(ref.version);
  return "";
}

function refFromValue(id: string, value: string): PromptRef {
  if (value.startsWith("label:")) return { id, label: value.slice(6) as PromptRef["label"] };
  if (value) return { id, version: Number(value) };
  return { id };
}

interface PickerProps {
  items: PromptSummary[];
  value: PromptRef | null;
  onChange: (ref: PromptRef | null) => void;
  placeholder: string;
  exclude?: string[];
}

/** Chọn một prompt và phiên bản (mới nhất / theo nhãn dev, prod / số cụ thể). */
function PromptPicker({ items, value, onChange, placeholder, exclude = [] }: PickerProps) {
  const item = items.find((p) => p.id === value?.id);
  return (
    <div className="input-row">
      <select value={value?.id ?? ""} onChange={(e) => onChange(e.target.value ? { id: e.target.value } : null)}>
        <option value="">{placeholder}</option>
        {items
          .filter((p) => p.id === value?.id || !exclude.includes(p.id))
          .map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
      </select>
      <select className="auto" value={refValue(value)} disabled={!item} onChange={(e) => item && onChange(refFromValue(item.id, e.target.value))}>
        <option value="">{item ? `mới nhất (v${item.latest})` : "phiên bản"}</option>
        {item &&
          [...PROMPT_LABELS]
            .reverse()
            .filter((label) => item.labels[label])
            .map((label) => (
              <option key={label} value={`label:${label}`}>
                nhãn {label} (v{item.labels[label]})
              </option>
            ))}
        {(item?.versions ?? []).map((v) => (
          <option key={v.version} value={v.version}>
            v{v.version}
          </option>
        ))}
      </select>
    </div>
  );
}

interface PromptListProps {
  items: PromptSummary[];
  refs: (PromptRef | null)[];
  onChange: (refs: (PromptRef | null)[]) => void;
  addLabel: string;
  emptyText: string;
}

/** Danh sách prompt (intent hoặc gate) đã chọn, thêm / bỏ từng cái. */
function PromptList({ items, refs, onChange, addLabel, emptyText }: PromptListProps) {
  const chosen = refs.flatMap((r) => (r ? [r.id] : []));
  function set(i: number, ref: PromptRef | null) {
    onChange(refs.map((r, j) => (j === i ? ref : r)));
  }
  function chooseAll() {
    const rest = items.filter((p) => !chosen.includes(p.id)).map((p): PromptRef => ({ id: p.id }));
    onChange([...refs.filter((r): r is PromptRef => r !== null), ...rest]);
  }
  return (
    <div className="stack">
      {refs.length === 0 && <span className="small muted">{emptyText}</span>}
      {refs.map((r, i) => (
        <div key={i} className="row">
          <PromptPicker items={items} value={r} placeholder="-- chọn --" exclude={chosen} onChange={(ref) => set(i, ref)} />
          <button type="button" className="icon-btn" title="Bỏ" onClick={() => onChange(refs.filter((_, j) => j !== i))}>
            <X size={14} />
          </button>
        </div>
      ))}
      <div className="row wrap">
        <button type="button" className="btn sm" onClick={() => onChange([...refs, null])}>
          <Plus size={14} /> {addLabel}
        </button>
        {items.length > chosen.length && (
          <button type="button" className="btn sm ghost" onClick={chooseAll}>
            Chọn tất cả
          </button>
        )}
      </div>
    </div>
  );
}

function DatasetChecks({ datasets, selected, onChange, emptyText }: { datasets: Dataset[]; selected: string[]; onChange: (ids: string[]) => void; emptyText: string }) {
  if (!datasets.length) return <span className="small muted">{emptyText}</span>;
  function toggle(id: string) {
    onChange(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]);
  }
  return (
    <div className="stack">
      {datasets.map((d) => (
        <label key={d.id} className="check">
          <input type="checkbox" checked={selected.includes(d.id)} onChange={() => toggle(d.id)} />
          {d.name} <span className="muted small">· {d.rows} dòng · {formatTime(d.created)}</span>
        </label>
      ))}
    </div>
  );
}

function BuildStatsView({ stats }: { stats: BuildStats }) {
  const warnings: string[] = [];
  if (stats.conflicts) warnings.push(`${stats.conflicts} câu giống nhau nhưng khác nhãn intent: nên kiểm tra lại nhãn.`);
  if (stats.leaked) warnings.push(`${stats.leaked} câu eval có câu y hệt trong phần train: điểm eval sẽ bị thổi phồng.`);
  const missing = stats.missing_in_val.filter((k) => k);
  if (missing.length) warnings.push(`${missing.length} intent không có câu nào trong val: ${missing.join(", ")}.`);
  if (stats.dropped_no_label) warnings.push(`${stats.dropped_no_label} câu bị bỏ vì không còn nhãn nào thuộc các intent / gate đã chọn.`);
  const intents: LabelRow[] = stats.strata.filter((s) => s.key);
  const valSub = stats.val_from === "eval" ? "từ dataset Eval" : `ngẫu nhiên ${percent(stats.val_fraction_actual)}`;
  return (
    <div className="stack">
      <div className="tiles">
        <Tile label="Train" value={stats.train} sub="câu người dùng" />
        <Tile label="Val" value={stats.val} sub={valSub} />
        <Tile label="Câu trùng đã bỏ" value={stats.removed_duplicates} />
        <Tile label="Câu dài nhất" value={`${stats.max_chars}`} sub={`ký tự · trung bình ${stats.avg_chars}`} />
      </div>
      {warnings.map((w) => (
        <div key={w} className="notice warn">
          {w}
        </div>
      ))}
      {intents.length > 0 && (
        <details>
          <summary className="small muted">Phân bố {intents.length} intent giữa train và val</summary>
          <LabelTable
            rows={intents}
            columns={[
              ["total", "Tổng"],
              ["train", "Train"],
              ["val", "Val"],
            ]}
          />
        </details>
      )}
      {stats.gates.length > 0 && (
        <details>
          <summary className="small muted">Số câu dính / không dính từng gate</summary>
          <LabelTable
            rows={stats.gates}
            columns={[
              ["train_true", "Train dính"],
              ["train_false", "Train không"],
              ["val_true", "Val dính"],
              ["val_false", "Val không"],
            ]}
          />
        </details>
      )}
    </div>
  );
}

function ChipLine({ title, refs }: { title: string; refs: string }) {
  if (!refs) return null;
  return (
    <div className="row wrap">
      <span className="muted">{title}:</span>
      {refs.split(", ").map((t) => (
        <Chip key={t}>{t}</Chip>
      ))}
    </div>
  );
}

function SpecView({ title }: { title: SpecTitle }) {
  return (
    <div className="stack small">
      <ChipLine title="System" refs={title.system} />
      <ChipLine title="Intent" refs={title.intents} />
      <ChipLine title="Gate" refs={title.gates} />
    </div>
  );
}

/** Đúng thứ serving sẽ hỏi (câu, lịch sử, các câu hỏi theo spec) cộng đáp án. */
function requestOf(row: BuildRow, spec: Spec) {
  const questions: Record<string, string | string[]> = {};
  if (spec.intents.length) questions.intent = spec.intents.map((x) => x.key);
  for (const g of spec.gates) questions[`gate_${g.key}`] = "có / không";
  const labels: Record<string, string | boolean | null> = { intent: row.intent };
  for (const [key, value] of Object.entries(row.gates)) labels[`gate_${key}`] = value;
  return { input: row.input, history: row.history || undefined, questions, labels };
}

/** Vài dòng "như thật" gửi runtime. */
function Examples({ examples, spec }: { examples: BuildRow[]; spec: Spec }) {
  if (!examples.length) return null;
  return (
    <details>
      <summary className="small muted">Xem {examples.length} dòng gửi runtime</summary>
      <div className="stack">
        {examples.map((e, i) => (
          <pre key={i} className="code">
            {JSON.stringify(requestOf(e, spec), null, 2)}
          </pre>
        ))}
        <span className="small muted">
          Runtime ghép: system + policy các intent + policy các gate + câu người dùng thành state (phần dùng chung đặt đầu để serving cache được), câu
          hỏi intent chỉ ghi tên, mỗi gate một câu có/không.
        </span>
      </div>
    </details>
  );
}

function usePreview(body: BuildBody, ready: boolean) {
  const [result, setResult] = useState<BuildPreview | null>(null);
  const [error, setError] = useState("");
  const key = ready ? JSON.stringify(body) : "";
  useEffect(() => {
    setResult(null);
    setError("");
    if (!key) return undefined;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        setResult(await ft.previewBuild(JSON.parse(key) as BuildBody, controller.signal));
      } catch (e) {
        if (e instanceof Error && e.name !== "AbortError") setError(e.message);
      }
    }, PREVIEW_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [key]);
  return { result, error };
}

function versionHint(name: string, sameName: Build[], nextVersion: number): string {
  const trimmed = name.trim();
  if (!trimmed) return "Cùng tên với build cũ thì thành phiên bản tiếp theo.";
  if (!sameName.length) return `Sẽ tạo "${trimmed} v1".`;
  return `Đã có ${sameName.length} phiên bản: sẽ tạo "${trimmed} v${nextVersion}".`;
}

function definedRefs(refs: (PromptRef | null)[]): PromptRef[] {
  return refs.filter((r): r is PromptRef => r !== null);
}

interface NewBuildProps {
  tasks: Task[];
  builds: Build[];
  /** Điền sẵn từ một build cũ khi bấm "Tạo phiên bản mới". */
  draft: Build | null;
  onCreated: (id: string) => void;
}

function NewBuild({ tasks, builds, draft, onCreated }: NewBuildProps) {
  const { busy, run } = useApp();
  const { items: datasets } = useList(ft.listDatasets, LIST_POLL_MS);
  const { items: systems } = useList(() => listPrompts("system"), LIST_POLL_MS);
  const { items: intentItems } = useList(() => listPrompts("intent"), LIST_POLL_MS);
  const { items: gateItems } = useList(() => listPrompts("gate"), LIST_POLL_MS);
  const taskId = draft?.task ?? tasks[0].id;
  const refs = draft?.refs;
  const [name, setName] = useState(draft?.name ?? "");
  const [note, setNote] = useState("");
  const [system, setSystem] = useState<PromptRef | null>(refs?.system ?? null);
  const [intents, setIntents] = useState<(PromptRef | null)[]>(refs?.intents ?? []);
  const [gates, setGates] = useState<(PromptRef | null)[]>(refs?.gates ?? []);
  const [trainSets, setTrainSets] = useState<string[]>(refs?.datasets ?? []);
  const [evalSets, setEvalSets] = useState<string[]>(refs?.eval_datasets ?? []);
  const [dedupe, setDedupe] = useState(draft ? draft.dedupe : true);
  const [valPercent, setValPercent] = useState(draft?.val_percent ?? VAL_DEFAULT);

  const chosenIntents = definedRefs(intents);
  const chosenGates = definedRefs(gates);
  const ready = Boolean(system?.id) && (chosenIntents.length > 0 || chosenGates.length > 0) && trainSets.length > 0;
  const body: BuildBody = {
    task: taskId,
    system,
    intents: chosenIntents,
    gates: chosenGates,
    datasets: trainSets,
    eval_datasets: evalSets,
    dedupe,
    val_percent: valPercent,
    split_seed: SPLIT_SEED,
  };
  const { result, error } = usePreview(body, ready);
  if (!datasets || !systems || !intentItems || !gateItems) return <Card title="Đang tải…" />;

  const ofTask = datasets.filter((d) => d.task === taskId);
  const sameName = builds.filter((b) => b.task === taskId && b.name.toLowerCase() === name.trim().toLowerCase());
  const nextVersion = Math.max(0, ...sameName.map((b) => b.version)) + 1;

  let problem = "";
  if (!name.trim()) problem = "Đặt tên cho build.";
  else if (!system?.id) problem = "Chọn system.";
  else if (!chosenIntents.length && !chosenGates.length) problem = "Chọn ít nhất một intent hoặc một gate.";
  else if (!trainSets.length) problem = "Chọn ít nhất một dataset SFT.";
  else if (error) problem = error;
  else if (!result) problem = "Đang ghép thử…";

  async function submit() {
    const created = await run(() => ft.createBuild({ ...body, name, note }));
    if (created) onCreated(created.id);
  }

  return (
    <Card title={draft ? `Phiên bản mới của "${draft.name}"` : "Build mới"}>
      <p className="small muted">
        Một bộ quyết định = 1 system + các intent (chọn một) + các gate (có/không). Mỗi prompt chọn theo phiên bản, hoặc theo nhãn dev / prod để lấy
        phiên bản đang mang nhãn đó; build ghi lại số phiên bản cụ thể đã dùng.
      </p>
      <div className="form-grid">
        <Field label="Tên build" hint={versionHint(name, sameName, nextVersion)}>
          <input value={name} maxLength={60} placeholder="vd. cskh" onChange={(e) => setName(e.target.value)} />
        </Field>
        <Field label="Ghi chú" hint="Phiên bản này thay đổi gì (không bắt buộc).">
          <input value={note} maxLength={300} onChange={(e) => setNote(e.target.value)} />
        </Field>
      </div>

      <div className="subhead">1. System</div>
      {systems.length === 0 && <span className="small muted">Chưa có system: tạo ở mục Prompt.</span>}
      <PromptPicker items={systems} value={system} placeholder="-- System --" onChange={setSystem} />

      <div className="subhead">2. Intent (chọn một trong số này)</div>
      <PromptList items={intentItems} refs={intents} onChange={setIntents} addLabel="Thêm intent" emptyText="Không có: chỉ train gate." />

      <div className="subhead">3. Gate (mỗi gate một câu có/không)</div>
      <PromptList items={gateItems} refs={gates} onChange={setGates} addLabel="Thêm gate" emptyText="Không có: chỉ train intent." />

      <div className="subhead">4. Dataset SFT (train)</div>
      <DatasetChecks datasets={ofTask.filter((d) => d.source === "sft")} selected={trainSets} onChange={setTrainSets} emptyText="Chưa có dataset SFT." />
      <label className="check">
        <input type="checkbox" checked={dedupe} onChange={(e) => setDedupe(e.target.checked)} />
        Bỏ câu trùng y hệt (giữ một bản)
      </label>

      <div className="subhead">5. Dataset Eval (không bắt buộc)</div>
      <p className="small muted">Có dataset Eval thì nó làm val khi train và tự được chấm sau khi train. Không có thì chia val ngẫu nhiên từ phần train.</p>
      <DatasetChecks
        datasets={ofTask.filter((d) => d.source === "eval")}
        selected={evalSets}
        onChange={setEvalSets}
        emptyText="Chưa có dataset Eval: val chia ngẫu nhiên."
      />
      {evalSets.length === 0 && (
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
      {result && <SpecView title={result.spec_title} />}
      {result && <BuildStatsView stats={result} />}
      {result && <Examples examples={result.examples} spec={result.spec} />}

      <div className="row wrap submit-row">
        <button type="button" className="btn primary" disabled={busy || Boolean(problem)} onClick={submit}>
          <Hammer size={14} /> Build
        </button>
        {problem && <span className="small muted">{problem}</span>}
      </div>
    </Card>
  );
}

function DatasetList({ title, list }: { title: string; list: { id: string; name: string; rows: number }[] }) {
  if (!list.length) return null;
  return (
    <div className="row wrap small">
      <span className="muted">{title}:</span>
      {list.map((d) => (
        <Chip key={d.id}>
          {d.name} · {d.rows}
        </Chip>
      ))}
    </div>
  );
}

interface DetailProps {
  id: string;
  onDeleted: () => void;
  onTrain: (buildId: string) => void;
  onNewVersion: (build: Build) => void;
}

function BuildDetail({ id, onDeleted, onTrain, onNewVersion }: DetailProps) {
  const { busy, run, confirm } = useApp();
  const [data, setData] = useState<{ build: Build; examples: BuildRow[] } | null>(null);
  useEffect(() => {
    setData(null);
    ft.getBuild(id)
      .then(setData)
      .catch(() => {});
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
    if (ok && (await run(() => ft.deleteBuild(id)))) onDeleted();
  }

  return (
    <Card>
      <div className="row between wrap">
        <div className="job-title">{buildTitle(b)}</div>
        <div className="row wrap">
          <button type="button" className="btn sm" onClick={() => onNewVersion(b)} title="Điền sẵn lựa chọn của build này">
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
        <span>· {b.dedupe ? "đã bỏ câu trùng" : "giữ câu trùng"}</span>
        {b.has_eval && <Chip kind="ok">có phần eval: tự eval sau khi train</Chip>}
      </div>
      {b.note && <div className="notice">{b.note}</div>}
      <SpecView title={b.manifest.spec_title} />
      <DatasetList title="Train" list={b.manifest.datasets} />
      <DatasetList title="Eval" list={b.manifest.eval_datasets} />
      <BuildStatsView stats={b.stats} />
      <Examples examples={data.examples} spec={b.task_config} />
    </Card>
  );
}

export default function Builds({ tasks, onTrain }: { tasks: Task[]; onTrain: (buildId: string) => void }) {
  const { items, reload } = useList(ft.listBuilds, LIST_POLL_MS);
  const [selected, select] = useSelected("finetune.build");
  const [draft, setDraft] = useState<Build | null>(null);
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
            key={draft?.id ?? "blank"}
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
