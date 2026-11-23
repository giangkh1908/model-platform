// Chọn file dữ liệu (JSONL / JSON / CSV / TSV), xem trước và chọn cột ứng với từng trường của loại fine-tune.
// Dùng chung cho file train và file golden.
import { useState } from "react";
import { Download, Upload } from "lucide-react";
import { Chip, Field, Segmented, Spinner } from "../components/ui.jsx";
import { api, downloadText } from "../lib.js";
import { useApp } from "../store.jsx";

const ACCEPT = ".jsonl,.ndjson,.json,.csv,.tsv,.txt";
const FORMAT_LABEL = { jsonl: "JSONL", json: "JSON", csv: "CSV", tsv: "TSV" };
// Tên cột hay gặp, để tự chọn sẵn cột cho từng trường.
const COLUMN_GUESSES = {
  input: ["text", "input", "user_input", "sentence", "utterance", "utt", "content", "query", "cau", "câu", "noi_dung", "nội dung"],
  intent: ["intent", "label", "class", "category", "nhan", "nhãn", "loai", "loại"],
  gates: ["gate", "gates", "flags", "co"],
  history: ["history", "context", "lich_su", "lịch sử"],
};

function guessMapping(fields, columns) {
  const mapping = {};
  for (const field of fields) {
    const names = [field.key, ...(COLUMN_GUESSES[field.key] || [])];
    const hit = columns.find((c) => names.includes(c.toLowerCase().trim()));
    mapping[field.key] = hit || "";
  }
  return mapping;
}

function cell(value) {
  if (value == null) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function toCsv(rows) {
  const columns = Object.keys(rows[0]);
  const quote = (v) => `"${cell(v).replaceAll('"', '""')}"`;
  return [columns.join(","), ...rows.map((r) => columns.map((c) => quote(r[c])).join(","))].join("\n");
}

// Ví dụ định dạng file cần chuẩn bị, kèm nút tải file mẫu.
export function FormatExample({ rows }) {
  const [format, setFormat] = useState("jsonl");
  if (!rows?.length) return null;
  let text;
  if (format === "jsonl") text = rows.map((r) => JSON.stringify(r)).join("\n");
  else if (format === "json") text = JSON.stringify(rows, null, 2);
  else text = toCsv(rows);
  const ext = format === "json" ? "json" : format;

  return (
    <details>
      <summary className="small muted">Xem định dạng file cần chuẩn bị</summary>
      <div className="stack" style={{ marginTop: 8 }}>
        <div className="row between wrap">
          <Segmented
            value={format}
            onChange={setFormat}
            options={[
              { id: "jsonl", label: "JSONL" },
              { id: "json", label: "JSON" },
              { id: "csv", label: "CSV" },
            ]}
          />
          <button type="button" className="btn ghost sm" onClick={() => downloadText(`mau.${ext}`, text)}>
            <Download size={14} /> Tải file mẫu
          </button>
        </div>
        <pre className="code">{text}</pre>
        <span className="small muted">Tên cột không bắt buộc giống mẫu: sau khi chọn file sẽ chọn cột tương ứng.</span>
      </div>
    </details>
  );
}

export function PreviewTable({ columns, rows }) {
  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c} className="text">
                  {cell(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// `value` = { upload, mapping } hoặc null; `onChange` nhận giá trị mới.
export default function DataFile({ fields, value, onChange, label = "Chọn file" }) {
  const { toast } = useApp();
  const [loading, setLoading] = useState(false);
  const upload = value?.upload;

  async function pick(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setLoading(true);
    try {
      const text = await file.text();
      const result = await api("/api/finetune/uploads", { body: { name: file.name, text } });
      onChange({ upload: result, mapping: guessMapping(fields, result.columns) });
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setLoading(false);
    }
  }

  function setColumn(key, column) {
    onChange({ upload, mapping: { ...value.mapping, [key]: column } });
  }

  return (
    <div className="stack">
      <div className="row wrap">
        <label className={`btn ${loading ? "disabled" : ""}`}>
          {loading ? <Spinner /> : <Upload size={14} />} {label}
          <input type="file" accept={ACCEPT} onChange={pick} hidden disabled={loading} />
        </label>
        {upload ? (
          <>
            <span className="small strong break">{upload.name}</span>
            <Chip>{FORMAT_LABEL[upload.format]}</Chip>
            <Chip>{upload.rows} dòng</Chip>
          </>
        ) : (
          <span className="small muted">JSONL, JSON, CSV hoặc TSV</span>
        )}
      </div>
      {upload && (
        <>
          <div className="form-grid">
            {fields.map((f) => (
              <Field key={f.key} label={`Cột "${f.label}"${f.required ? "" : " (không bắt buộc)"}`} hint={f.hint}>
                <select value={value.mapping[f.key] || ""} onChange={(e) => setColumn(f.key, e.target.value)}>
                  <option value="">{f.required ? "-- Chọn cột --" : "(không dùng)"}</option>
                  {upload.columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </Field>
            ))}
          </div>
          <PreviewTable columns={upload.columns} rows={upload.preview} />
        </>
      )}
    </div>
  );
}

export function mappingComplete(fields, value) {
  return Boolean(value?.upload) && fields.every((f) => !f.required || value.mapping[f.key]);
}
