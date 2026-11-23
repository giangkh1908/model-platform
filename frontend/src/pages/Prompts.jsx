// Trang Prompt: template (khung có {policy}, {history}, {input} + phần output) và policy (đoạn luật).
// Sửa là lưu phiên bản mới; build ghi lại phiên bản đã dùng nên không bị ảnh hưởng.
import { useEffect, useState } from "react";
import { Save, Trash2 } from "lucide-react";
import { Card, Chip, Field, Segmented } from "../components/ui.jsx";
import { NEW, SideList, useList, useSelected } from "../finetune/common.jsx";
import { api, formatTime } from "../lib.js";
import { useApp } from "../store.jsx";

const KINDS = [
  { id: "template", label: "Template" },
  { id: "policy", label: "Policy" },
];
const TEMPLATE_EXAMPLE = {
  body: "Hãy phân tích câu dưới đây theo policy sau:\n{policy}\n<input>\n{input}\n</input>",
  output: "Câu trong thẻ <input> có vi phạm policy không? Trả 1 nếu vi phạm, 0 nếu không.",
};

function emptyContent(kind) {
  if (kind === "template") return { ...TEMPLATE_EXAMPLE };
  return { text: "" };
}

function ContentEditor({ kind, content, onChange }) {
  if (kind === "policy") {
    return (
      <Field label="Nội dung policy" hint="Chèn vào chỗ {policy} của template.">
        <textarea rows={8} value={content.text} onChange={(e) => onChange({ text: e.target.value })} />
      </Field>
    );
  }
  return (
    <>
      <Field label="Thân prompt" hint="Phải có {input} (câu trong dataset); thường có thêm {policy}. Đây là phần model đọc.">
        <textarea rows={10} value={content.body} onChange={(e) => onChange({ ...content, body: e.target.value })} />
      </Field>
      <Field label="Output" hint="Model phải trả lời gì (câu hỏi). Các nhãn trong dataset là các lựa chọn.">
        <textarea rows={3} value={content.output} onChange={(e) => onChange({ ...content, output: e.target.value })} />
      </Field>
    </>
  );
}

function PromptEditor({ kind, id, onSaved, onDeleted }) {
  const { busy, run, confirm } = useApp();
  const [item, setItem] = useState(null);
  const [version, setVersion] = useState(null);
  const [name, setName] = useState("");
  const [note, setNote] = useState("");
  const [content, setContent] = useState(emptyContent(kind));
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    setItem(null);
    setMeta(null);
    setNote("");
    if (id === NEW) {
      setName("");
      setContent(emptyContent(kind));
      return;
    }
    const versionQuery = version ? `?version=${version}` : "";
    api(`/api/finetune/prompts/${kind}/${id}${versionQuery}`).then((p) => {
      setItem(p);
      setName(p.name);
      setContent(p.content);
    });
    api(`/api/finetune/prompts/${kind}`).then(({ items }) => setMeta(items.find((m) => m.id === id) || null));
  }, [kind, id, version]);

  async function save() {
    const path = id === NEW ? `/api/finetune/prompts/${kind}` : `/api/finetune/prompts/${kind}/${id}`;
    const result = await run(path, { body: { name, note, content } });
    if (result) {
      setVersion(null);
      onSaved(result.id);
    }
  }

  async function remove() {
    const ok = await confirm({ title: `Xoá "${name}" và mọi phiên bản?`, message: "Build đã dùng nó vẫn giữ nội dung.", confirmLabel: "Xoá", danger: true });
    if (ok && (await run(`/api/finetune/prompts/${kind}/${id}`, { method: "DELETE" }))) onDeleted();
  }

  const isNew = id === NEW;
  let title = `${kind === "template" ? "Template" : "Policy"} mới`;
  if (!isNew && item) title = `${item.name} · v${item.version}`;

  return (
    <Card>
      <div className="row between wrap">
        <div className="job-title">{title}</div>
        {!isNew && (
          <button type="button" className="btn sm danger" disabled={busy} onClick={remove}>
            <Trash2 size={14} /> Xoá
          </button>
        )}
      </div>
      {meta && (
        <div className="row wrap small">
          <span className="muted">Phiên bản:</span>
          {meta.versions.map((v) => (
            <button
              key={v.version}
              type="button"
              className={`btn sm ${item?.version === v.version ? "primary" : ""}`}
              title={v.note || formatTime(v.created)}
              onClick={() => setVersion(v.version)}
            >
              v{v.version}
            </button>
          ))}
        </div>
      )}
      <div className="form-grid">
        <Field label="Tên">
          <input value={name} maxLength={80} onChange={(e) => setName(e.target.value)} />
        </Field>
        <Field label="Ghi chú phiên bản" hint="Lần sửa này thay đổi gì (không bắt buộc).">
          <input value={note} maxLength={200} onChange={(e) => setNote(e.target.value)} />
        </Field>
      </div>
      <ContentEditor kind={kind} content={content} onChange={setContent} />
      <div className="row wrap submit-row">
        <button type="button" className="btn primary" disabled={busy} onClick={save}>
          <Save size={14} /> {isNew ? "Lưu" : "Lưu thành phiên bản mới"}
        </button>
        {!isNew && <span className="small muted">Sửa không ghi đè: lưu ra phiên bản mới, phiên bản cũ giữ nguyên.</span>}
      </div>
    </Card>
  );
}

export default function Prompts() {
  const [kind, setKind] = useState("template");
  const { items, reload } = useList(`/api/finetune/prompts/${kind}`, "items", 10000);
  const [selected, select] = useSelected(`prompt.${kind}`);
  const list = items || [];
  const view = list.some((p) => p.id === selected) ? selected : NEW;

  return (
    <div className="ft">
      <div className="ft-tabs">
        <Segmented value={kind} onChange={setKind} options={KINDS} />
      </div>
      <div className="ft-layout">
        <SideList
          title={kind === "template" ? "Template" : "Policy"}
          newLabel="Tạo mới"
          items={list}
          view={view}
          onSelect={select}
          empty="Chưa có gì."
          renderRow={(p) => (
            <>
              <span className="ft-job-name">{p.name}</span>
              <span className="small muted">
                <Chip>v{p.latest}</Chip> {formatTime(p.versions[p.versions.length - 1].created)}
              </span>
            </>
          )}
        />
        <PromptEditor
          key={`${kind}:${view}`}
          kind={kind}
          id={view}
          onSaved={(id) => {
            select(id);
            reload();
          }}
          onDeleted={() => {
            select(NEW);
            reload();
          }}
        />
      </div>
    </div>
  );
}

