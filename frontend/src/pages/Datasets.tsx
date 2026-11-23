// Trang Dataset: dữ liệu gốc dùng chung cho Build dataset (train) và Eval (golden).
import { Database } from "lucide-react";
import Datasets from "../finetune/Datasets.tsx";
import { TasksGate } from "../finetune/common.tsx";

export default function DatasetsPage() {
  return <TasksGate icon={Database}>{(tasks) => <Datasets tasks={tasks} />}</TasksGate>;
}
