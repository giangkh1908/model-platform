import { useEffect, useMemo, useState } from "react";
import { Boxes, Download, ExternalLink, Heart, Play, RotateCw, Save, Search, Square, Trash2 } from "lucide-react";
import { Bar, Card, Chip, CopyButton, Empty, Field, Segmented, Spinner } from "../components/ui.jsx";
import { MODE, OPTION_FIELDS, RT, api, compact, duration, fitOf, gb, modeChoicesOf, modeOf, percent, speed } from "../lib.js";
import { isActive, useModelActions } from "../modelActions.js";
import { useApp } from "../store.jsx";

const HF_URL = "https://huggingface.co/";

const SEARCH_KINDS = [
  { id: "", label: "Tất cả" },
  { id: "gguf", label: "GGUF" },
  { id: "text-generation", label: "Chat" },
  { id: "feature-extraction", label: "Embedding" },
  { id: "text-classification", label: "Phân loại / rerank" },
];

// ------------------------------------------------------------ thêm model

function InspectPanel({ info, onDone }) {
  const { run } = useApp();
  const ggufFiles = info.gguf_files || [];
  const runtimes = info.runtimes || [{ id: info.runtime, name: info.runtime_name, note: "" }];
  const modes = info.modes || [];
  const [file, setFile] = useState(ggufFiles.find((f) => f.recommended)?.name || "");
  const [runtime, setRuntime] = useState(info.runtime);
  const [mode, setMode] = useState(info.mode);
  const [sending, setSending] = useState(false);

  const selected = ggufFiles.find((f) => f.name === file);
  const isDecider = info.runtime === "decider";
  let weights = info.weights;
  if (selected && !isDecider) weights = selected.size;
  const downloadSize = selected && !isDecider ? selected.size : info.size;
  const [fitLevel, fitText] = fitOf(weights, runtime);

  let blockReason = "";
  if (fitLevel === "too_big" && runtime !== "llamacpp") blockReason = "Model quá lớn cho GPU này với runtime đã chọn.";
  if (runtime === "vllm" && selected?.parts > 1) blockReason = "vLLM không chạy được GGUF chia nhiều file.";

  async function download() {
    setSending(true);
    const result = await run("/api/models", { body: { url: info.repo, gguf: file || null, runtime, mode } });
    setSending(false);
    if (result) onDone();
  }

  return (
    <div className="inspect">
      <div className="row between wrap">
        <a className="inspect-title" href={HF_URL + info.repo} target="_blank" rel="noreferrer">
          {info.repo} <ExternalLink size={14} />
        </a>
        <div className="row muted small">
          {info.downloads != null && (
            <span>
              <Download size={12} /> {compact(info.downloads)}
            </span>
          )}
          {info.likes != null && (
            <span>
              <Heart size={12} /> {compact(info.likes)}
            </span>
          )}
        </div>
      </div>
      <div className="row wrap">
        <Chip kind={fitLevel}>{fitText}</Chip>
        {info.pipeline_tag && <Chip>{info.pipeline_tag}</Chip>}
        {info.gated && <Chip kind="tight">Gated</Chip>}
        {info.license && <span className="muted small">Giấy phép: {info.license}</span>}
      </div>
      <div className="muted small">
        Dung lượng tải: {gb(downloadSize)}
        {runtime === "vllm" && selected ? " + tokenizer của model gốc" : ""}
        {info.base_model ? ` · Model gốc: ${info.base_model}` : ""}
      </div>

      {ggufFiles.length > 0 && (
        <Field label="Bản nén (khuyên dùng Q4_K_M: nhẹ, chất lượng tốt)">
          <select value={file} onChange={(e) => setFile(e.target.value)}>
            {ggufFiles.map((f) => (
              <option key={f.name} value={f.name}>
                {f.name} · {gb(f.size)}
                {f.parts > 1 ? ` · ${f.parts} phần` : ""}
                {f.recommended ? " (khuyên dùng)" : ""}
              </option>
            ))}
          </select>
        </Field>
      )}

      <div className="form-grid">
        <Field label={runtimes.length > 1 ? `Runtime (${runtimes.length} lựa chọn)` : "Runtime"}>
          <select value={runtime} onChange={(e) => setRuntime(e.target.value)} disabled={runtimes.length < 2}>
            {runtimes.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Chế độ">
          <select value={mode} onChange={(e) => setMode(e.target.value)} disabled={modes.length < 2}>
            {modes.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
                {m.id === info.mode ? " (tự nhận)" : ""}
              </option>
            ))}
          </select>
        </Field>
      </div>
      <div className="muted small">{runtimes.find((r) => r.id === runtime)?.note}</div>

      {info.warnings?.length > 0 && (
        <ul className="warnings">
          {info.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}

      <div className="row wrap">
        <button type="button" className="btn primary" onClick={download} disabled={sending || Boolean(blockReason)}>
          {sending ? <Spinner /> : <Download size={16} />} Tải về
        </button>
        <button type="button" className="btn ghost" onClick={onDone}>
          Bỏ qua
        </button>
        {blockReason && <span className="error-text small">{blockReason}</span>}
      </div>
    </div>
  );
}

function SearchResults({ results, onPick }) {
  if (!results) return null;
  if (!results.length) return <div className="muted small">Không tìm thấy model nào.</div>;
  return (
    <div className="list search-results">
      {results.map((r) => (
        <button type="button" key={r.repo} className="list-row clickable" onClick={() => onPick(r.repo)}>
          <div className="grow">
            <div className="strong">{r.repo}</div>
            <div className="muted small">
              {r.pipeline_tag || "không rõ loại"}
              {r.gguf ? " · GGUF" : ""}
            </div>
          </div>
          <span className="muted small num">
            <Download size={12} /> {compact(r.downloads)} · <Heart size={12} /> {compact(r.likes)}
          </span>
        </button>
      ))}
    </div>
  );
}

function AddModel() {
  const [tab, setTab] = useState("link");
  const [url, setUrl] = useState("");
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState("");
  const [results, setResults] = useState(null);
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function inspect(target) {
    if (!target.trim()) return;
    setLoading(true);
    setError("");
    setInfo(null);
    try {
      setInfo(await api("/api/inspect", { body: { url: target } }));
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  }

  async function search() {
    if (query.trim().length < 2) return;
    setLoading(true);
    setError("");
    try {
      const data = await api(`/api/hf/search?q=${encodeURIComponent(query)}&kind=${kind}`);
      setResults(data.results);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  }

  function pick(repo) {
    setUrl(repo);
    inspect(repo);
  }

  function reset() {
    setInfo(null);
    setUrl("");
  }

  return (
    <Card
      title="Thêm model"
      className="span2"
      actions={
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { id: "link", label: "Dán link" },
            { id: "search", label: "Tìm trên Hugging Face" },
          ]}
        />
      }
    >
      {tab === "link" ? (
        <form
          className="input-row"
          onSubmit={(e) => {
            e.preventDefault();
            inspect(url);
          }}
        >
          <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="Dán link Hugging Face, ví dụ https://huggingface.co/BAAI/bge-m3" />
          <button type="submit" className="btn primary" disabled={loading || !url.trim()}>
            {loading ? <Spinner /> : <Search size={16} />} Kiểm tra
          </button>
        </form>
      ) : (
        <>
          <form
            className="input-row three"
            onSubmit={(e) => {
              e.preventDefault();
              search();
            }}
          >
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Tên model, ví dụ qwen3, bge, llama…" />
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              {SEARCH_KINDS.map((k) => (
                <option key={k.id} value={k.id}>
                  {k.label}
                </option>
              ))}
            </select>
            <button type="submit" className="btn primary" disabled={loading || query.trim().length < 2}>
              {loading ? <Spinner /> : <Search size={16} />} Tìm
            </button>
          </form>
          {!info && <SearchResults results={results} onPick={pick} />}
        </>
      )}
      {error && <div className="error-text">{error}</div>}
      {info && <InspectPanel key={info.repo} info={info} onDone={reset} />}
    </Card>
  );
}

// ------------------------------------------------------------ danh sách model

function statusChip(model, active, state) {
  if (model.status === "downloading") return <Chip kind="tight">Đang tải {percent(model.progress || 0, 0)}</Chip>;
  if (model.status === "error") return <Chip kind="bad">Lỗi</Chip>;
  if (!active) return null;
  return <Chip kind="ok">{state === "running" ? "Đang chạy" : "Đang khởi động"}</Chip>;
}

function downloadEta(model) {
  if (!model.speed || !model.size) return "";
  const remaining = Math.max(0, model.size - (model.downloaded || 0));
  return ` · còn khoảng ${duration(remaining / model.speed)}`;
}

function OptionsEditor({ model, active }) {
  const { busy } = useApp();
  const { start, save } = useModelActions();
  const [draft, setDraft] = useState({});
  const fields = OPTION_FIELDS[model.runtime] || [];
  const currentMode = draft.mode ?? modeOf(model);
  const modeChoices = modeChoicesOf(model.runtime);
  const changed = Object.keys(draft).length > 0;

  function setValue(key, value) {
    setDraft((d) => ({ ...d, [key]: value }));
  }

  function optionsOnly() {
    const { mode, ...options } = draft;
    return Object.fromEntries(Object.entries(options).filter(([, v]) => v !== ""));
  }

  async function onSave() {
    if (await save(model, optionsOnly(), draft.mode)) setDraft({});
  }

  async function onStart() {
    if (await start(model, optionsOnly(), draft.mode)) setDraft({});
  }

  let applyHint = "Áp dụng khi bật";
  if (changed) applyHint = active ? "Đã sửa: lưu rồi tắt và bật lại để áp dụng" : "Đã sửa: áp dụng khi bấm Bật";

  return (
    <div className="options">
      {model.runtime !== "decider" && (
        <label className="option">
          <span>Chế độ</span>
          <select value={currentMode} onChange={(e) => setValue("mode", e.target.value)}>
            {modeChoices.map((m) => (
              <option key={m} value={m}>
                {MODE[m]}
              </option>
            ))}
          </select>
        </label>
      )}
      {fields.map(([key, label, step, min, max, help]) => (
        <label key={key} className="option" title={help}>
          <span>{label}</span>
          <input
            type="number"
            step={step}
            min={min}
            max={max}
            value={draft[key] ?? model.options?.[key] ?? ""}
            onChange={(e) => setValue(key, e.target.value === "" ? "" : Number(e.target.value))}
          />
        </label>
      ))}
      <span className="muted small grow">{applyHint}</span>
      {changed && (
        <>
          <button type="button" className="btn sm ghost" onClick={() => setDraft({})}>
            Hoàn tác
          </button>
          <button type="button" className="btn sm" onClick={onSave} disabled={busy}>
            <Save size={14} /> Lưu
          </button>
        </>
      )}
      {!active && (
        <button type="button" className="btn sm primary" onClick={onStart} disabled={busy}>
          <Play size={14} /> Bật
        </button>
      )}
    </div>
  );
}

function ModelCard({ model }) {
  const { status, busy } = useApp();
  const { stop, retry, remove } = useModelActions();
  const active = isActive(status, model);
  const downloading = model.status === "downloading";

  return (
    <div className={`model ${active ? "active" : ""}`}>
      <div className="model-head">
        <div className="grow">
          <div className="model-name">
            <span>{model.id}</span>
            <Chip>{RT[model.runtime]}</Chip>
            <Chip>{MODE[modeOf(model)]}</Chip>
            {statusChip(model, active, status.state)}
          </div>
          <div className="muted small break">
            <a href={HF_URL + model.repo} target="_blank" rel="noreferrer">
              {model.repo}
            </a>
            {model.file ? ` · ${model.file}` : ""} · {gb(model.size)}
            {model.added ? ` · thêm ${model.added}` : ""}
          </div>
        </div>
        <div className="row">
          <CopyButton text={model.path} label="" />
          {active && (
            <button type="button" className="btn sm" onClick={stop} disabled={busy}>
              <Square size={14} /> Tắt
            </button>
          )}
          {model.status === "error" && model.managed_path && (
            <button type="button" className="btn sm" onClick={() => retry(model)} disabled={busy}>
              <RotateCw size={14} /> Tải lại
            </button>
          )}
          <button type="button" className="btn sm danger" onClick={() => remove(model)} disabled={busy}>
            <Trash2 size={14} /> {downloading ? "Huỷ" : "Xoá"}
          </button>
        </div>
      </div>
      {downloading && (
        <div className="progress">
          <Bar fraction={model.progress} />
          <span className="muted small num">
            {gb(model.downloaded || 0)} / {gb(model.size)}
            {model.speed ? ` · ${speed(model.speed)}` : ""}
            {downloadEta(model)}
          </span>
        </div>
      )}
      {model.status === "error" && <div className="error-text small">{model.error}</div>}
      {model.status === "ready" && <OptionsEditor model={model} active={active} />}
    </div>
  );
}

const FILTERS = [
  { id: "all", label: "Tất cả" },
  { id: "ready", label: "Sẵn sàng" },
  { id: "downloading", label: "Đang tải" },
  { id: "error", label: "Lỗi" },
];

function ModelList() {
  const { status } = useApp();
  const [text, setText] = useState("");
  const [filter, setFilter] = useState("all");
  const models = status.models;

  const visible = useMemo(() => {
    const needle = text.trim().toLowerCase();
    return models.filter((m) => {
      if (filter !== "all" && m.status !== filter) return false;
      if (!needle) return true;
      return `${m.id} ${m.repo} ${m.runtime}`.toLowerCase().includes(needle);
    });
  }, [models, text, filter]);

  const totalSize = models.reduce((sum, m) => sum + (m.size || 0), 0);

  return (
    <Card
      title={`Model của bạn (${models.length} · ${gb(totalSize)})`}
      className="span2"
      actions={<Segmented value={filter} onChange={setFilter} options={FILTERS} />}
    >
      {models.length > 3 && <input className="filter" value={text} onChange={(e) => setText(e.target.value)} placeholder="Lọc theo tên…" />}
      {models.length === 0 && (
        <Empty icon={Boxes} title="Chưa có model">
          Dán link hoặc tìm model ở mục Thêm model.
        </Empty>
      )}
      <div className="models">
        {visible.map((m) => (
          <ModelCard key={m.id} model={m} />
        ))}
      </div>
    </Card>
  );
}

export default function Models() {
  const { status, refresh } = useApp();
  useEffect(() => {
    refresh();
  }, [refresh]);
  if (!status) return <div className="muted">Đang tải…</div>;
  return (
    <div className="grid">
      <AddModel />
      <ModelList />
    </div>
  );
}
