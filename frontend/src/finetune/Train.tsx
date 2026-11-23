// Tab Train: hàng đợi job bên trái, bên phải là form tạo job hoặc chi tiết job đang chọn.
import { listJobs } from "../api/finetune.ts";
import { Bar, Card } from "../components/ui.tsx";
import { RUN_POLL_MS } from "../config/app.ts";
import { formatTime } from "../lib/format.ts";
import type { Job, Task } from "../types/api.ts";
import { NEW, SideList, StateChip, isActive, useList } from "./common.tsx";
import JobDetail from "./JobDetail.tsx";
import NewJob from "./NewJob.tsx";

function JobRow({ job }: { job: Job }) {
  const status = job.status ?? {};
  const total = status.total_steps ?? 0;
  let sub = `${job.build.title} · ${formatTime(job.created)}`;
  if (job.state === "queued") sub = job.queue_position ? `thứ ${job.queue_position} trong hàng đợi` : "đang chờ";
  if (job.state === "running") sub = status.message ?? "đang chạy";
  return (
    <>
      <div className="row between">
        <span className="ft-job-name">{job.name}</span>
        <StateChip state={job.state} />
      </div>
      <span className="small muted">{sub}</span>
      {job.state === "running" && total > 0 && <Bar fraction={(status.step ?? 0) / total} />}
    </>
  );
}

interface TrainProps {
  tasks: Task[];
  selected: string;
  select: (id: string) => void;
  buildId: string;
  setBuildId: (id: string) => void;
  onOpenBuilds: () => void;
}

export default function Train({ tasks, selected, select, buildId, setBuildId, onOpenBuilds }: TrainProps) {
  const { items: jobs, reload } = useList(listJobs, RUN_POLL_MS);
  if (!jobs) return <Card title="Đang tải…" />;
  const view = jobs.some((j) => j.id === selected) ? selected : NEW;
  const active = jobs.filter((j) => isActive(j.state)).length;

  return (
    <div className="ft-layout">
      <SideList
        title={`Job${active ? ` · ${active} đang chờ/chạy` : ""}`}
        newLabel="Job mới"
        items={jobs}
        view={view}
        onSelect={select}
        empty="Chưa có job nào."
        renderRow={(j) => <JobRow job={j} />}
        footer="Mỗi lúc chỉ chạy một việc trên GPU (train hoặc eval), việc sau tự chạy khi việc trước xong. Còn việc trong hàng đợi thì nền tảng không tự tắt khi đóng tab."
      />
      <div className="stack">
        {view === NEW ? (
          <NewJob
            tasks={tasks}
            buildId={buildId}
            setBuildId={setBuildId}
            onOpenBuilds={onOpenBuilds}
            onCreated={(id) => {
              select(id);
              reload();
            }}
          />
        ) : (
          <JobDetail
            jobId={view}
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
