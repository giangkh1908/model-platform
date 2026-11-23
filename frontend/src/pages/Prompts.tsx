// Trang Prompt: system, intent, gate của các bộ quyết định. Loại prompt và các trường do backend khai báo
// (/api/finetune/prompt-kinds); trang chỉ vẽ theo đó. Có key Langfuse thì backend đọc / ghi thẳng trên Langfuse: sửa ở
// đây thì Langfuse có ngay, sửa trên Langfuse thì trang tự cập nhật (tải lại mỗi vài giây).
import { useEffect, useState } from "react";
import { diffLines } from "diff";
import { ExternalLink, Save, Trash2 } from "lucide-react";
import * as promptsApi from "../api/prompts.ts";
import { Card, Chip, Empty, Field, Segmented } from "../components/ui.tsx";
import { PROMPT_POLL_MS } from "../config/app.ts";
import { NEW, SideList, useList, useSelected } from "../finetune/common.tsx";
import { useApp } from "../store.tsx";
import type {
  PromptContent,
  PromptKind,
  PromptLabel,
  PromptStoreStatus,
  PromptSummary,
  PromptVersion,
  PromptVersionInfo,
} from "../types/api.ts";

function emptyContent(kind: PromptKind): PromptContent {
  const content: PromptContent = { text: "" };
  for (const field of kind.fields) content[field.key] = "";
  return content;
}

function contentText(kind: PromptKind, content: PromptContent): string {
  return kind.fields
    .filter((f) => content[f.key])
    .map((f) => (f.key === "text" ? content.text : `[${f.label}] ${content[f.key]}`))
    .join("\n\n");
}

function orderedPair(a: PromptVersion, b: PromptVersion): [PromptVersion, PromptVersion] {
  if (a.version < b.version) return [a, b];
  return [b, a];
}

function LabelChips({ labels }: { labels: PromptLabel[] }) {
  return (
    <>
      {labels.map((label) => (
        <Chip key={label} kind={label === "prod" ? "ok" : ""}>
          {label}
        </Chip>
      ))}
    </>
  );
}

/** Diff theo dòng giữa hai phiên bản (cũ -> mới). */
function DiffView({ kind, before, after, title }: { kind: PromptKind; before: PromptVersion; after: PromptVersion; title: string }) {
  const parts = diffLines(contentText(kind, before.content), contentText(kind, after.content));
  return (
    <div className="stack">
      <div className="small muted">{title}</div>
      <pre className="code diff">
        {parts.map((part, i) => {
          let cls = "";
          let mark = "  ";
          if (part.added) {
            cls = "diff-add";
            mark = "+ ";
          } else if (part.removed) {
            cls = "diff-del";
            mark = "- ";
          }
          const lines = part.value.replace(/\n$/, "").split("\n");
          return (
            <span key={i} className={cls}>
              {lines.map((line) => `${mark}${line}`).join("\n")}
              {"\n"}
            </span>
          );
        })}
      </pre>
    </div>
  );
}

/** Các ô nhập vẽ theo khai báo của loại prompt. */
function ContentFields({ kind, content, onChange }: { kind: PromptKind; content: PromptContent; onChange: (content: PromptContent) => void }) {
  return (
    <>
      {kind.fields.map((field) => (
        <Field key={field.key} label={field.label} hint={field.hint || undefined}>
          {field.multiline ? (
            <textarea rows={kind.id === "system" ? 10 : 7} value={content[field.key] ?? ""} onChange={(e) => onChange({ ...content, [field.key]: e.target.value })} />
          ) : (
            <input value={content[field.key] ?? ""} onChange={(e) => onChange({ ...content, [field.key]: e.target.value })} />
          )}
        </Field>
      ))}
    </>
  );
}

function NewPrompt({ kind, onSaved }: { kind: PromptKind; onSaved: (id: string) => void }) {
  const { busy, run } = useApp();
  const [name, setName] = useState("");
  const [content, setContent] = useState<PromptContent>(() => emptyContent(kind));

  async function save() {
    const result = await run(() => promptsApi.createPrompt(kind.id, name, content));
    if (result) onSaved(result.id);
  }

  return (
    <Card title={`${kind.label} mới`}>
      <p className="small muted">{kind.help}</p>
      <Field label="Tên" hint="Chữ thường không dấu, số, _ và - (vd. thanh_toan). Không đổi được sau khi tạo.">
        <input value={name} maxLength={80} onChange={(e) => setName(e.target.value.toLowerCase())} />
      </Field>
      <ContentFields kind={kind} content={content} onChange={setContent} />
      <div className="row wrap submit-row">
        <button type="button" className="btn primary" disabled={busy || !name.trim()} onClick={save}>
          <Save size={14} /> Lưu thành v1
        </button>
      </div>
    </Card>
  );
}

interface EditorProps {
  kind: PromptKind;
  labels: readonly PromptLabel[];
  item: PromptSummary;
  reload: () => void;
  onDeleted: () => void;
}

function PromptEditor({ kind, labels, item, reload, onDeleted }: EditorProps) {
  const { busy, run, confirm } = useApp();
  const [versions, setVersions] = useState<PromptVersionInfo[]>([]);
  const [viewed, setViewed] = useState<PromptVersion | null>(null); // phiên bản đang xem (đủ nội dung)
  const [content, setContent] = useState<PromptContent>(() => emptyContent(kind));
  const [dirty, setDirty] = useState(false);
  const [compareTo, setCompareTo] = useState("");
  const [compared, setCompared] = useState<PromptVersion | null>(null);
  const [lastDiff, setLastDiff] = useState<{ before: PromptVersion; after: PromptVersion } | null>(null);

  async function loadVersion(version: number) {
    const p = await promptsApi.getPrompt(kind.id, item.id, { version });
    setViewed(p);
    setContent(p.content);
    setDirty(false);
    setCompareTo("");
    setCompared(null);
  }

  async function loadVersions() {
    setVersions(await promptsApi.listVersions(kind.id, item.id));
  }

  useEffect(() => {
    loadVersion(item.latest);
    loadVersions();
    // tải lại khi chọn prompt khác
  }, [kind.id, item.id]);

  // Danh sách (tải lại mỗi vài giây) báo có phiên bản mới hoặc nhãn đổi, vd. vừa sửa trên Langfuse:
  // cập nhật danh sách phiên bản; đang xem bản mới nhất mà chưa sửa gì thì tự mở bản mới.
  useEffect(() => {
    loadVersions();
    if (viewed && !dirty && viewed.version < item.latest && viewed.version === versions.length) loadVersion(item.latest);
    // chỉ phản ứng khi danh sách báo thay đổi
  }, [item.latest, item.labels.dev, item.labels.prod]);

  async function save() {
    if (!viewed) return;
    const before = viewed;
    const result = await run(() => promptsApi.saveVersion(kind.id, item.id, content));
    if (!result) return;
    const after = await promptsApi.getPrompt(kind.id, item.id, { version: result.version });
    setLastDiff({ before, after });
    await loadVersion(result.version);
    await loadVersions();
    reload();
  }

  async function setLabel(label: PromptLabel) {
    if (!viewed) return;
    const version = viewed.version;
    if (await run(() => promptsApi.setLabel(kind.id, item.id, version, label))) {
      await loadVersions();
      reload();
    }
  }

  async function compare(value: string) {
    setCompareTo(value);
    setCompared(value ? await promptsApi.getPrompt(kind.id, item.id, { version: Number(value) }) : null);
  }

  async function remove() {
    const ok = await confirm({
      title: `Xoá "${item.id}" và mọi phiên bản?`,
      message: "Xoá cả trên Langfuse (nếu đang đồng bộ). Build đã dùng nó vẫn giữ nội dung.",
      confirmLabel: "Xoá",
      danger: true,
    });
    if (ok && (await run(() => promptsApi.deletePrompt(kind.id, item.id)))) onDeleted();
  }

  if (!viewed) return <Card title="Đang tải…" />;
  const viewedLabels = versions.find((v) => v.version === viewed.version)?.labels ?? [];
  const newerExists = dirty && item.latest > viewed.version;
  const comparedPair = compared ? orderedPair(compared, viewed) : null;

  return (
    <Card>
      <div className="row between wrap">
        <div className="job-title">
          {item.id} · v{viewed.version}
        </div>
        <button type="button" className="btn sm danger" disabled={busy} onClick={remove}>
          <Trash2 size={14} /> Xoá
        </button>
      </div>
      <div className="row wrap small">
        <span className="muted">Phiên bản:</span>
        {versions.map((v) => (
          <button
            key={v.version}
            type="button"
            className={`btn sm ${viewed.version === v.version ? "primary" : ""}`}
            title={v.note}
            onClick={() => loadVersion(v.version)}
          >
            v{v.version} {v.labels.length > 0 && <span className="muted">({v.labels.join(", ")})</span>}
          </button>
        ))}
      </div>
      <div className="row wrap small">
        <span className="muted">Nhãn của v{viewed.version}:</span>
        <LabelChips labels={viewedLabels} />
        {labels
          .filter((label) => !viewedLabels.includes(label))
          .map((label) => (
            <button key={label} type="button" className="btn sm" disabled={busy} onClick={() => setLabel(label)}>
              Gắn {label}
            </button>
          ))}
        {viewed.note && <span className="muted">· {viewed.note}</span>}
      </div>
      {newerExists && (
        <div className="notice warn small">
          Đã có v{item.latest} mới hơn (có thể vừa sửa trên Langfuse). Lưu bây giờ sẽ tạo phiên bản sau v{item.latest}.
        </div>
      )}
      <ContentFields
        kind={kind}
        content={content}
        onChange={(c) => {
          setContent(c);
          setDirty(true);
        }}
      />
      <div className="row wrap submit-row">
        <button type="button" className="btn primary" disabled={busy || !dirty} onClick={save}>
          <Save size={14} /> Lưu thành v{item.latest + 1}
        </button>
        <span className="small muted">Phiên bản tự đánh số. Sửa không ghi đè: phiên bản cũ giữ nguyên.</span>
      </div>
      {lastDiff && (
        <DiffView
          kind={kind}
          before={lastDiff.before}
          after={lastDiff.after}
          title={`Vừa lưu: thay đổi từ v${lastDiff.before.version} sang v${lastDiff.after.version}`}
        />
      )}
      <div className="row wrap small">
        <span className="muted">So sánh v{viewed.version} với</span>
        <select className="auto" value={compareTo} onChange={(e) => compare(e.target.value)}>
          <option value="">-- chọn phiên bản --</option>
          {versions
            .filter((v) => v.version !== viewed.version)
            .map((v) => (
              <option key={v.version} value={v.version}>
                v{v.version}
              </option>
            ))}
        </select>
      </div>
      {comparedPair && (
        <DiffView kind={kind} before={comparedPair[0]} after={comparedPair[1]} title={`Diff v${comparedPair[0].version} → v${comparedPair[1].version}`} />
      )}
    </Card>
  );
}

function SourceNote({ status }: { status: PromptStoreStatus | null }) {
  if (!status) return null;
  if (status.source === "langfuse") {
    return (
      <span className="small muted">
        Đồng bộ với Langfuse · prompt tạo trên Langfuse cần tag system / intent / gate ·{" "}
        <a href={status.url} target="_blank" rel="noreferrer">
          mở Langfuse <ExternalLink size={12} />
        </a>
      </span>
    );
  }
  return <span className="small muted">Lưu trên máy (chưa có key Langfuse trong .env).</span>;
}

/** Khai báo loại prompt + nơi lưu, tải một lần. */
function usePromptSchema() {
  const [schema, setSchema] = useState<{ kinds: PromptKind[]; labels: PromptLabel[] } | null>(null);
  const [status, setStatus] = useState<PromptStoreStatus | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    promptsApi
      .getKinds()
      .then(setSchema)
      .catch((e: Error) => setError(e.message));
    promptsApi
      .getStoreStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);
  return { schema, status, error };
}

function PromptsOfKind({ kind, kinds, labels, onKind }: { kind: PromptKind; kinds: PromptKind[]; labels: PromptLabel[]; onKind: (id: string) => void }) {
  const { items, reload } = useList(() => promptsApi.listPrompts(kind.id), PROMPT_POLL_MS, [kind.id]);
  const [selected, select] = useSelected(`prompt.${kind.id}`);
  const list = items ?? [];
  const item = list.find((p) => p.id === selected);

  return (
    <>
      <div className="ft-tabs row between wrap">
        <Segmented value={kind.id} onChange={onKind} options={kinds.map((k) => ({ id: k.id, label: k.label }))} />
      </div>
      <div className="ft-layout">
        <SideList
          title={kind.label}
          newLabel="Tạo mới"
          items={list}
          view={item ? item.id : NEW}
          onSelect={select}
          empty="Chưa có gì."
          renderRow={(p) => (
            <>
              <span className="ft-job-name">{p.name}</span>
              <span className="small muted row wrap">
                <Chip>v{p.latest}</Chip>
                {p.labels.prod && <Chip kind="ok">prod v{p.labels.prod}</Chip>}
                {p.labels.dev && <Chip>dev v{p.labels.dev}</Chip>}
              </span>
            </>
          )}
        />
        {item ? (
          <PromptEditor
            key={`${kind.id}:${item.id}`}
            kind={kind}
            labels={labels}
            item={item}
            reload={reload}
            onDeleted={() => {
              select(NEW);
              reload();
            }}
          />
        ) : (
          <NewPrompt
            key={kind.id}
            kind={kind}
            onSaved={(id) => {
              select(id);
              reload();
            }}
          />
        )}
      </div>
    </>
  );
}

export default function Prompts() {
  const { schema, status, error } = usePromptSchema();
  const [kindId, setKindId] = useState("system");
  if (!schema) {
    return (
      <Card>
        <Empty title={error || "Đang tải…"} />
      </Card>
    );
  }
  const kind = schema.kinds.find((k) => k.id === kindId) ?? schema.kinds[0];
  return (
    <div className="ft">
      <SourceNote status={status} />
      <PromptsOfKind key={kind.id} kind={kind} kinds={schema.kinds} labels={schema.labels} onKind={setKindId} />
    </div>
  );
}
