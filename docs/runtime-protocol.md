# Runtime protocol v1

Hợp đồng giữa **model-platform** (giao diện, dataset, prompt, build, hàng đợi) và **runtime fine-tune** (thuật toán
train và chấm). Hai bên chỉ trao đổi qua **một thư mục chạy** và **một lệnh**, không import code của nhau. Runtime nào
làm đúng tài liệu này thì platform dùng được, dù nằm trong project này (`runtime/finetune/`) hay ở repo
`runtime-platform`.

Phía platform: `backend/finetune/connector.py` (interface `RuntimeConnector`). Phía runtime hiện tại:
`runtime/finetune/common.py`.

## Lệnh

```
python run.py train <task> <run_dir>
python run.py eval  <task> <run_dir>
```

- `<task>`: loại fine-tune, vd. `decider`.
- `<run_dir>`: thư mục chạy (đường dẫn nhìn từ phía runtime). Platform ghi đầu vào trước khi gọi lệnh.
- Log in ra stdout/stderr, platform hiện ở trang Log.
- Kết quả thật (thành công hay lỗi) đọc từ `status.json`, không dựa vào mã thoát.

## Train

### Đầu vào (platform ghi)

| File | Nội dung |
| --- | --- |
| `job.json` | `{"protocol": 1, "id", "task", "base_path", "output_dir", "params", "task_config"}` |
| `train.jsonl` | Mẫu train, mỗi dòng một object JSON đã ghép prompt (vd. decider: `{"text", "label", "input"}`) |
| `val.jsonl` | Mẫu val, cùng định dạng, không trùng dòng với train |

- `base_path`: model gốc (hoặc checkpoint để train tiếp), đường dẫn phía runtime.
- `output_dir`: nơi runtime lưu checkpoint tốt nhất khi xong.
- `params`: tham số chung (`lr`, `epochs`, `batch_size`, `grad_accum`, `eval_every`, `warmup_ratio`, `weight_decay`,
  `max_grad_norm`, `log_every`, `seed`, `gradient_checkpointing`) cộng tham số riêng của loại fine-tune.
- `task_config`: cấu hình riêng của loại, rút từ build (vd. decider: `labels`, `instructions`, `descriptions`).

### Đầu ra (runtime ghi)

**`status.json`**: trạng thái hiện tại, ghi đè liên tục.

```json
{"state": "training", "step": 120, "total_steps": 400, "epoch": 0.6, "eta": 310.5, "message": "Epoch 1/2"}
```

`state` đi theo thứ tự: `starting` → `loading` → `training` / `evaluating` → `finalizing` → `saving` → `done`.
Có lỗi ở bất kỳ bước nào thì `state = "error"`, kèm `message` (và `traceback` nếu có).

**`metrics.jsonl`**: mỗi dòng một sự kiện, chỉ ghi thêm vào cuối file.

| `kind` | Khi nào | Khoá |
| --- | --- | --- |
| `start` | Bắt đầu train | `total_steps`, `steps_per_epoch`, `n_train`, `n_val` |
| `train` | Mỗi `log_every` bước | `step`, `epoch`, `loss`, `lr`, `grad_norm`, `samples_per_s`, `vram_gb` |
| `eval` | Mỗi `eval_every` bước (và bước 0) | `step`, `epoch`, `best`, `loss` + chỉ số riêng của loại (vd. `acc`, `macro_f1`, `ece`) |
| `final` | Sau khi lưu checkpoint | `best_step`, `best_eval_loss`, `train_seconds` + tóm tắt riêng (vd. decider: `threshold`) |

Mọi dòng có thêm `time` (epoch seconds). Platform chỉ vẽ `loss` của dòng `train` và `eval`; mọi khoá số khác được
chuyển sang MLflow với tiền tố theo `kind` (`train_lr`, `eval_acc`, `final_threshold_threshold`…), nên runtime
thêm chỉ số mới thì không phải sửa platform.

**Checkpoint** ở `output_dir`. Platform chạy thử checkpoint bằng runtime serving khai báo trong `model_entry` của
loại fine-tune (vd. decider: `strands-decider serve`).

## Eval

### Đầu vào

| File | Nội dung |
| --- | --- |
| `eval.json` | `{"protocol": 1, "targets": [...]}`: mỗi target `{"key", "name", "kind": "base" \| "adapter", "path", "config", "threshold", "leaked"}` |
| `golden.jsonl` | Tập chấm, cùng định dạng mẫu với `train.jsonl` |

### Đầu ra

- `status.json`: như train, kết thúc bằng `done` hoặc `error`.
- `report.json`: báo cáo do loại fine-tune tự định nghĩa, giao diện của loại đó biết cách vẽ
  (vd. decider: chỉ số từng model và kết quả từng câu).

## Dừng

Platform yêu cầu dừng bằng SIGINT, sau 5 giây chưa thoát thì SIGKILL. Runtime dừng ngay, ghi `status.json` với
`state = "error"` (platform tự đổi thành `stopped`), không cần lưu checkpoint. Platform xoá `output_dir` của lần train
dừng giữa chừng hoặc lỗi.

## Phiên bản

- `protocol` trong `job.json` / `eval.json` là số phiên bản của tài liệu này.
- Thêm khoá mới (không bắt buộc) thì giữ nguyên số. Bên đọc phải bỏ qua khoá không biết.
- Đổi tên, bỏ, hoặc đổi nghĩa một khoá thì tăng số phiên bản, và runtime nên từ chối phiên bản lớn hơn bản nó hỗ trợ.
