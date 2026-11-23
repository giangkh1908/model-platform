// Trang Prompt: system, intent, gate của các bộ quyết định. Có key Langfuse thì đọc / ghi thẳng trên Langfuse:
// sửa ở đây thì Langfuse có ngay, sửa trên Langfuse thì trang tự cập nhật (tải lại mỗi vài giây).
// Mỗi lần lưu là một phiên bản mới (v1, v2…); gắn nhãn dev / prod cho phiên bản; xem diff giữa hai phiên bản.
import { useEffect, useState } from "react";
import { diffLines } from "diff";
import { ExternalLink, Save, Trash2 } from "lucide-react";
import { Card, Chip, Field, Segmented } from "../components/ui.jsx";
import { NEW, SideList, useList, useSelected } from "../finetune/common.jsx";
import { api } from "../lib.js";
import { useApp } from "../store.jsx";

const POLL_MS = 5000;
const LABELS = ["dev", "prod"];
const KINDS = [
  { id: "system", label: "System" },
  { id: "intent", label: "Intent" },
  { id: "gate", label: "Gate" },
];
const KIND_HELP = {
  system: "Bối cảnh chung của bộ quyết định (vd. \"Bạn là bộ phận CSKH của…\"). Runtime đặt nó đầu state, trước policy và câu người dùng.",
  intent: "Mỗi intent một prompt: viết policy của intent đó (cái gì thuộc, cái gì không). Tên prompt là nhãn intent trong dataset.",
  gate: "Mỗi gate một câu có/không (vd. lừa đảo, cần chuyển người): viết policy của gate. Tên prompt là nhãn gate trong dataset.",
};
const TEXT_LABEL = { system: "System prompt", intent: "Policy của intent", gate: "Policy của gate" };
// Trường tuỳ chọn của từng loại (lưu ở config trên Langfuse).
const EXTRA_FIELDS = {
  system: [["intent_question", "Câu hỏi intent", "Bỏ trống = câu mặc định của runtime (\"Người dùng đang cần gì?…\")."]],
  intent: [],
  gate: [["question", "Câu hỏi của gate", "Bỏ trống = câu mặc định (\"Câu của người dùng có thuộc trường hợp gate … không?\")."]],
};

function emptyContent(kind) {
  const content = { text: "" };
  for (const [key] of EXTRA_FIELDS[kind]) content[key] = "";
  return content;
}

function promptPath(kind, id) {
  return `/api/finetune/prompts/${kind}/${encodeURIComponent(id)}`;
}

function contentText(kind, content) {
  const parts = [content.text || ""];
  for (const [key, label] of EXTRA_FIELDS[kind]) {
    if (content[key]) parts.push(`[${label}] ${content[key]}`);
  }
  return parts.join("\n\n");
}

function LabelChips({ labels }) {
  return labels.map((label) => (
    <Chip key={label} kind={label === "prod" ? "ok" : ""}>
      {label}
    </Chip>
  ));
}

// Diff theo dòng giữa hai phiên bản (cũ -> mới).
function DiffView({ kind, before, after, title }) {
  const parts = diffLines(contentText(kind, before.content), contentText(kind, after.content));
  return (
    <div className="stack">
      <div className="small muted">{title}</div>
      <pre className="code diff">
        {parts.map((p, i) => {
          let cls = "";
          let mark = "  ";
          if (p.added) {
            cls = "diff-add";
            mark = "+ ";
          } else if (p.removed) {
            cls = "diff-del";
            mark = "- ";
          }
          const lines = p.value.replace(/\n$/, "").split("\n");
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

function ContentFields({ kind, content, onChange }) {
  return (
    <>
      <Field label={TEXT_LABEL[kind]}>
        <textarea rows={kind === "system" ? 10 : 7} value={content.text} onChange={(e) => onChange({ ...content, text: e.target.value })} />
      </Field>
      {EXTRA_FIELDS[kind].map(([key, label, hint]) => (
        <Field key={key} label={label} hint={hint}>
          <input value={content[key] || ""} onChange={(e) => onChange({ ...content, [key]: e.target.value })} />
        </Field>
      ))}
    </>
  );
}

function NewPrompt({ kind, onSaved }) {
  const { busy, run } = useApp();
  const [name, setName] = useState("");
  const [content, setContent] = useState(emptyContent(kind));

  async function save() {
    const result = await run(`/api/finetune/prompts/${kind}`, { body: { name, content } });
    if (result) onSaved(result.id);
  }

  return (
    <Card title={`${KINDS.find((k) => k.id === kind).label} mới`}>
      <p className="small muted">{KIND_HELP[kind]}</p>
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

function orderedPair(a, b) {
  if (a.version < b.version) return [a, b];
  return [b, a];
}

function PromptEditor({ kind, item, onDeleted, reload }) {
  const { busy, run, confirm } = useApp();
  const [versions, setVersions] = useState([]);
  const [viewed, setViewed] = useState(null); // phiên bản đang xem (đủ nội dung)
  const [content, setContent] = useState(emptyContent(kind));
  const [dirty, setDirty] = useState(false);
  const [compareTo, setCompareTo] = useState("");
  const [compared, setCompared] = useState(null);
  const [lastDiff, setLastDiff] = useState(null); // diff sau khi lưu

  async function loadVersion(version) {
    const p = await api(`${promptPath(kind, item.id)}?version=${version}`);
    setViewed(p);
    setContent(p.content);
    setDirty(false);
    setCompareTo("");
    setCompared(null);
  }

  async function loadVersions() {
    setVersions((await api(`${promptPath(kind, item.id)}/versions`)).versions);
  }

  useEffect(() => {
    loadVersion(item.latest);
    loadVersions();
    // tải lại khi chọn prompt khác
  }, [kind, item.id]);

  // Danh sách (tải lại mỗi vài giây) báo có phiên bản mới hoặc nhãn đổi, vd. vừa sửa trên Langfuse:
  // cập nhật danh sách phiên bản; đang xem bản mới nhất mà chưa sửa gì thì tự mở bản mới.
  useEffect(() => {
    loadVersions();
    if (viewed && !dirty && viewed.version < item.latest && viewed.version === versions.length) loadVersion(item.latest);
    // chỉ phản ứng khi danh sách báo thay đổi
  }, [item.latest, item.labels.dev, item.labels.prod]);

  async function save() {
    const before = viewed;
    const result = await run(promptPath(kind, item.id), { body: { name: item.id, content } });
    if (!result) return;
    const after = await api(`${promptPath(kind, item.id)}?version=${result.version}`);
    setLastDiff({ before, after });
    await loadVersion(result.version);
    await loadVersions();
    reload();
  }

  async function setLabel(label) {
    if (await run(`${promptPath(kind, item.id)}/labels/${label}`, { method: "PUT", body: { version: viewed.version } })) {
      await loadVersions();
      reload();
    }
  }

  async function compare(version) {
    setCompareTo(version);
    setCompared(version ? await api(`${promptPath(kind, item.id)}?version=${version}`) : null);
  }

  async function remove() {
    const ok = await confirm({
      title: `Xoá "${item.id}" và mọi phiên bản?`,
      message: "Xoá cả trên Langfuse (nếu đang đồng bộ). Build đã dùng nó vẫn giữ nội dung.",
      confirmLabel: "Xoá",
      danger: true,
    });
    if (ok && (await run(promptPath(kind, item.id), { method: "DELETE" }))) onDeleted();
  }

  if (!viewed) return <Card title="Đang tải…" />;
  const viewedLabels = versions.find((v) => v.version === viewed.version)?.labels || [];
  const newerExists = dirty && item.latest > viewed.version;
  let comparedPair = null;
  if (compared) comparedPair = orderedPair(compared, viewed);

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
            title={v.note || ""}
            onClick={() => loadVersion(v.version)}
          >
            v{v.version} {v.labels.length > 0 && <span className="muted">({v.labels.join(", ")})</span>}
          </button>
        ))}
      </div>
      <div className="row wrap small">
        <span className="muted">Nhãn của v{viewed.version}:</span>
        <LabelChips labels={viewedLabels} />
        {LABELS.filter((label) => !viewedLabels.includes(label)).map((label) => (
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

function SourceNote() {
  const [status, setStatus] = useState(null);
  useEffect(() => {
    api("/api/finetune/prompts-status")
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);
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

export default function Prompts() {
  const [kind, setKind] = useState("system");
  const { items, reload } = useList(`/api/finetune/prompts/${kind}`, "items", POLL_MS);
  const [selected, select] = useSelected(`prompt.${kind}`);
  const list = items || [];
  const item = list.find((p) => p.id === selected);

  return (
    <div className="ft">
      <div className="ft-tabs row between wrap">
        <Segmented value={kind} onChange={setKind} options={KINDS} />
        <SourceNote />
      </div>
      <div className="ft-layout">
        <SideList
          title={KINDS.find((k) => k.id === kind).label}
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
            key={`${kind}:${item.id}`}
            kind={kind}
            item={item}
            reload={reload}
            onDeleted={() => {
              select(NEW);
              reload();
            }}
          />
        ) : (
          <NewPrompt
            key={kind}
            kind={kind}
            onSaved={(id) => {
              select(id);
              reload();
            }}
          />
        )}
      </div>
    </div>
  );
}
