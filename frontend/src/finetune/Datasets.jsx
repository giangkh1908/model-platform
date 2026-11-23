// Tab Dataset: đẩy file lên, chọn cột, lưu thành dataset dùng chung cho build (train) và eval golden.
import { useEffect, useState } from "react";
import { Save, Trash2 } from "lucide-react";
import { Card, Chip, Field, Segmented } from "../components/ui.jsx";
import { api, formatTime } from "../lib.js";
import { useApp } from "../store.jsx";
import { LabelTable, NEW, SideList, Tile, useList, useSelected } from "./common.jsx";
import DataFile, { FormatExample, PreviewTable, mappingComplete } from "./DataFile.jsx";

function NewDataset({ tasks, onCreated }) {
  const { busy, run } = useApp();
  const [taskId, setTaskId] = useState(tasks[0].id);
  const task = tasks.find((t) => t.id === taskId);
  const { items: sources } = useList("/api/finetune/sources", "sources", 60000);
  const [source, setSource] = useState("sft");
  const [file, setFile] = useState(null);
  const [name, setName] = useState("");
  const ready = mappingComplete(task.fields, file);

  async function save() {
    const body = { task: taskId, source, name: name || file.upload.name, upload_id: file.upload.upload_id, mapping: file.mapping };
    const result = await run("/api/finetune/datasets", { body });
    if (result) onCreated(result.id);
  }

  return (
    <Card
      title="Dataset mới"
      actions={
        tasks.length > 1 && (
          <Segmented
            value={taskId}
            onChange={(id) => {
              setTaskId(id);
              setFile(null);
            }}
            options={tasks.map((t) => ({ id: t.id, label: t.name }))}
          />
        )
      }
    >
      <p className="small muted">
        Dataset là dữ liệu gốc (lịch sử, câu người dùng, nhãn). Build ghép nó với template + policy: dataset SFT vào phần train,
        dataset Eval vào phần eval.
      </p>
      <div className="row wrap">
        <span className="field-label">Loại nguồn</span>
        <Segmented value={source} onChange={setSource} options={(sources || []).map((s) => ({ id: s.id, label: s.name }))} />
      </div>
      <span className="small muted">{(sources || []).find((s) => s.id === source)?.description}</span>
      <FormatExample rows={task.example_rows} />
      <DataFile fields={task.fields} value={file} onChange={setFile} label="Chọn file" />
      {file && (
        <div className="form-grid">
          <Field label="Tên dataset" hint="Để trống thì lấy tên file.">
            <input value={name} maxLength={80} placeholder={file.upload.name} onChange={(e) => setName(e.target.value)} />
          </Field>
        </div>
      )}
      <div className="row wrap submit-row">
        <button type="button" className="btn primary" disabled={busy || !ready} onClick={save}>
          <Save size={14} /> Lưu dataset
        </button>
        {!ready && <span className="small muted">Chọn file và chọn đủ cột.</span>}
      </div>
    </Card>
  );
}

function DatasetDetail({ id, tasks, onDeleted }) {
  const { busy, run, confirm } = useApp();
  const [data, setData] = useState(null);

  useEffect(() => {
    setData(null);
    api(`/api/finetune/datasets/${id}`).then(setData).catch(() => {});
  }, [id]);

  if (!data) return <Card title="Đang tải…" />;
  const d = data.dataset;
  const task = tasks.find((t) => t.id === d.task);
  const columns = task ? task.fields.map((f) => f.key) : Object.keys(data.preview[0] || {});

  async function remove() {
    const ok = await confirm({
      title: `Xoá dataset "${d.name}"?`,
      message: "Các build đã tạo từ dataset này vẫn giữ nguyên dữ liệu của chúng.",
      confirmLabel: "Xoá",
      danger: true,
    });
    if (ok && (await run(`/api/finetune/datasets/${id}`, { method: "DELETE" }))) onDeleted();
  }

  return (
    <Card>
      <div className="row between wrap">
        <div className="job-title">{d.name}</div>
        <button type="button" className="btn sm danger" disabled={busy} onClick={remove}>
          <Trash2 size={14} /> Xoá
        </button>
      </div>
      <div className="row wrap small muted">
        <Chip>{task?.name || d.task}</Chip>
        <Chip kind={d.source === "eval" ? "ok" : ""}>{d.source === "eval" ? "Eval" : "SFT"}</Chip>
        <span>
          từ file {d.file} · tạo lúc {formatTime(d.created)}
        </span>
      </div>
      <div className="tiles">
        <Tile label="Số dòng" value={d.rows} />
        {d.labels.length > 0 && <Tile label="Số nhãn" value={d.labels.length} />}
        {d.invalid > 0 && <Tile label="Dòng lỗi bị bỏ" value={d.invalid} />}
      </div>
      {d.errors?.length > 0 && (
        <details>
          <summary className="small muted">Xem dòng lỗi</summary>
          {d.errors.map((e) => (
            <div key={e.row} className="small error-text">
              Dòng {e.row}: {e.reason}
            </div>
          ))}
        </details>
      )}
      {d.labels.length > 0 && (
        <>
          <div className="subhead">Phân bố nhãn</div>
          <LabelTable rows={d.labels} columns={[["count", "Số câu"]]} />
        </>
      )}
      <div className="subhead">Vài dòng đầu</div>
      <PreviewTable columns={columns} rows={data.preview} />
    </Card>
  );
}

export default function Datasets({ tasks }) {
  const { items, reload } = useList("/api/finetune/datasets", "datasets", 10000);
  const [selected, select] = useSelected("finetune.dataset");
  if (!items) return <Card title="Đang tải…" />;
  const view = items.some((d) => d.id === selected) ? selected : NEW;

  return (
    <div className="ft-layout">
      <SideList
        title="Dataset"
        newLabel="Dataset mới"
        items={items}
        view={view}
        onSelect={select}
        empty="Chưa có dataset nào."
        renderRow={(d) => (
          <>
            <span className="ft-job-name">{d.name}</span>
            <span className="small muted">
              <Chip kind={d.source === "eval" ? "ok" : ""}>{d.source === "eval" ? "Eval" : "SFT"}</Chip> {d.rows} dòng ·{" "}
              {formatTime(d.created)}
            </span>
          </>
        )}
      />
      <div className="stack">
        {view === NEW ? (
          <NewDataset
            tasks={tasks}
            onCreated={(id) => {
              select(id);
              reload();
            }}
          />
        ) : (
          <DatasetDetail
            id={view}
            tasks={tasks}
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
