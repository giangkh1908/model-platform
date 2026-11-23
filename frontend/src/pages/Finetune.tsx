// Trang Finetune: hàng đợi job train, tạo job từ một build + model, dashboard chỉ số của từng job.
import { useState } from "react";
import { GraduationCap } from "lucide-react";
import { TasksGate, useSelected } from "../finetune/common.tsx";
import { STORAGE_KEYS } from "../config/app.ts";
import Train from "../finetune/Train.tsx";
import { loadLocal, saveLocal } from "../lib/browser.ts";
import { go } from "../store.tsx";

export default function Finetune() {
  const [job, selectJob] = useSelected(STORAGE_KEYS.selectedJob);
  // Build chọn sẵn khi tới từ nút "Train từ build này" (đọc một lần rồi xoá).
  const [buildId, setBuildId] = useState(() => {
    const id = loadLocal(STORAGE_KEYS.trainBuild, "");
    saveLocal(STORAGE_KEYS.trainBuild, "");
    return id;
  });

  return (
    <TasksGate icon={GraduationCap}>
      {(tasks) => (
        <Train tasks={tasks} selected={job} select={selectJob} buildId={buildId} setBuildId={setBuildId} onOpenBuilds={() => go("builds")} />
      )}
    </TasksGate>
  );
}
