// Tab Dataset: đẩy file lên, chọn cột, lưu thành dataset dùng chung cho build (train) và eval golden.
import { useEffect, useState } from "react";
import { Save, Trash2 } from "lucide-react";
import * as ft from "../api/finetune.ts";
import { Card, Chip, Field, Segmented } from "../components/ui.tsx";
import { LIST_POLL_MS } from "../config/app.ts";
import { formatTime } from "../lib/format.ts";
import { useApp } from "../store.tsx";
import type { Dataset, Task } from "../types/api.ts";
import { LabelTable, NEW, SideList, Tile, useList, useSelected } from "./common.tsx";
import DataFile, { FormatExample, PreviewTable, mappingComplete, type DataFileValue } from "./DataFile.tsx";

const SOURCES_POLL_MS = 60_000;

function NewDataset({ tasks, onCreated }: { tasks: Task[]; onCreated: (id: string) => void }) {
  const { busy, run } = useApp();
  const [taskId, setTaskId] = useState(tasks[0].id);
  const task = tasks.find((t) => t.id === taskId) ?? tasks[0];
  const { items: sources } = useList(ft.listSources, SOURCES_POLL_MS);
  const [source, setSource] = useState("sft");
  const [file, setFile] = useState<DataFileValue | null>(null);
  const [name, setName] = useState("");
  const ready = mappingComplete(task.fields, file);

  async function save() {
    if (!file) return;
    const body = { task: taskId, source, name: name || file.upload.name, upload_id: file.upload.upload_id, mapping: file.mapping };
    const result = await run(() => ft.createDataset(body));
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
        Dataset là dữ liệu gốc (câu người dùng, lịch sử, nhãn intent, gate dính). Build ghép nó với một bộ quyết định (system + intent +
        gate): dataset SFT vào phần train, dataset Eval vào phần eval.
      </p>
      <div className="row wrap">
        <span className="field-label">Loại nguồn</span>
        <Segmented value={source} onChange={setSource} options={(sources ?? []).map((s) => ({ id: s.id, label: s.name }))} />
      </div>
      <span className="small muted">{(sources ?? []).find((s) => s.id === source)?.description}</span>
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

function DatasetDetail({ id, tasks, onDeleted }: { id: string; tasks: Task[]; onDeleted: () => void }) {
  const { busy, run, confirm } = useApp();
  const [data, setData] = useState<{ dataset: Dataset; preview: Record<string, unknown>[] } | null>(null);

  useEffect(() => {
    setData(null);
    ft.getDataset(id)
      .then(setData)
      .catch(() => {});
  }, [id]);

  if (!data) return <Card title="Đang tải…" />;
  const d = data.dataset;
  const task = tasks.find((t) => t.id === d.task);
  const columns = task ? task.fields.map((f) => f.key) : Object.keys(data.preview[0] ?? {});

  async function remove() {
    const ok = await confirm({
      title: `Xoá dataset "${d.name}"?`,
      message: "Các build đã tạo từ dataset này vẫn giữ nguyên dữ liệu của chúng.",
      confirmLabel: "Xoá",
      danger: true,
    });
    if (ok && (await run(() => ft.deleteDataset(id)))) onDeleted();
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
        <Chip>{task?.name ?? d.task}</Chip>
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
      {d.errors.length > 0 && (
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

export default function Datasets({ tasks }: { tasks: Task[] }) {
  const { items, reload } = useList(ft.listDatasets, LIST_POLL_MS);
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
