// Tab Train: hàng đợi job bên trái, bên phải là form tạo job hoặc chi tiết job đang chọn.
import { Bar, Card } from "../components/ui.jsx";
import { formatTime } from "../lib.js";
import { NEW, SideList, StateChip, isActive, useList } from "./common.jsx";
import JobDetail from "./JobDetail.jsx";
import NewJob from "./NewJob.jsx";

function JobRow({ job }) {
  const status = job.status || {};
  const total = status.total_steps || 0;
  let sub = `${job.build.title} · ${formatTime(job.created)}`;
  if (job.state === "queued") sub = job.queue_position ? `thứ ${job.queue_position} trong hàng đợi` : "đang chờ";
  if (job.state === "running") sub = status.message || "đang chạy";
  return (
    <>
      <div className="row between">
        <span className="ft-job-name">{job.name}</span>
        <StateChip state={job.state} />
      </div>
      <span className="small muted">{sub}</span>
      {job.state === "running" && total > 0 && <Bar fraction={(status.step || 0) / total} />}
    </>
  );
}

export default function Train({ tasks, selected, select, buildId, setBuildId, onOpenBuilds }) {
  const { items: jobs, reload } = useList("/api/finetune/jobs", "jobs", 3000);
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
