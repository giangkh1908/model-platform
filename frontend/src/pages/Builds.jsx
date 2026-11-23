// Trang Build dataset: ghép dataset thành các phiên bản dữ liệu để train.
import { Layers } from "lucide-react";
import Builds from "../finetune/Builds.jsx";
import { NEW, TasksGate } from "../finetune/common.jsx";
import { TRAIN_BUILD_KEY } from "../finetune/NewJob.jsx";
import { saveLocal } from "../lib.js";
import { go } from "../store.jsx";

// "Train từ build này": mở trang Finetune với form job mới đã chọn sẵn build.
function trainFrom(buildId) {
  saveLocal(TRAIN_BUILD_KEY, buildId);
  saveLocal("finetune.job", NEW);
  go("finetune");
}

export default function BuildsPage() {
  return <TasksGate icon={Layers}>{(tasks) => <Builds tasks={tasks} onTrain={trainFrom} />}</TasksGate>;
}
