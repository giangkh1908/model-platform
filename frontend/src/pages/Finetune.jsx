// Trang Finetune: hàng đợi job train, tạo job từ một build + model, dashboard chỉ số của từng job.
import { useState } from "react";
import { GraduationCap } from "lucide-react";
import { TasksGate, useSelected } from "../finetune/common.jsx";
import { TRAIN_BUILD_KEY } from "../finetune/NewJob.jsx";
import Train from "../finetune/Train.jsx";
import { loadLocal, saveLocal } from "../lib.js";
import { go } from "../store.jsx";

export default function Finetune() {
  const [job, selectJob] = useSelected("finetune.job");
  // Build chọn sẵn khi tới từ nút "Train từ build này" (đọc một lần rồi xoá).
  const [buildId, setBuildId] = useState(() => {
    const id = loadLocal(TRAIN_BUILD_KEY, "");
    saveLocal(TRAIN_BUILD_KEY, "");
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
