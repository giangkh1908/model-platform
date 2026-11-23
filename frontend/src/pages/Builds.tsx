// Trang Build dataset: ghép dataset thành các phiên bản dữ liệu để train.
import { Layers } from "lucide-react";
import Builds from "../finetune/Builds.tsx";
import { NEW, TasksGate } from "../finetune/common.tsx";
import { STORAGE_KEYS } from "../config/app.ts";
import { saveLocal } from "../lib/browser.ts";
import { go } from "../store.tsx";

// "Train từ build này": mở trang Finetune với form job mới đã chọn sẵn build.
function trainFrom(buildId: string) {
  saveLocal(STORAGE_KEYS.trainBuild, buildId);
  saveLocal(STORAGE_KEYS.selectedJob, NEW);
  go("finetune");
}

export default function BuildsPage() {
  return <TasksGate icon={Layers}>{(tasks) => <Builds tasks={tasks} onTrain={trainFrom} />}</TasksGate>;
}
