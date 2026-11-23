// Giao diện riêng của từng loại fine-tune. Thêm loại mới: viết file như decider.jsx (cùng các export) rồi đăng ký ở đây.
// Phần chung (chọn file, chia dữ liệu, tham số, biểu đồ, hàng đợi) không cần sửa.
//
// Mỗi loại export:
//   EVAL_COLUMNS                   [khoá, nhãn, đơn vị] cho bảng chỉ số eval
//   headlineTiles(lastEval)        ô tóm tắt ở đầu trang job
//   FinalSummary({final})          tóm tắt sau train (dòng "final" của metrics.jsonl)
//   EvalReport({report})           báo cáo eval golden nhiều model (report.json): xếp hạng, chi tiết
import * as decider from "./decider.jsx";

export const TASK_UI = { decider };
