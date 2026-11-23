"""Phần chung của mọi loại fine-tune chạy trong WSL: file của job, ghi chỉ số, vòng train, eval golden.

Mỗi loại (decider, sau này chat…) chỉ cài đặt lớp `Task` bên dưới; vòng train, lịch learning rate, eval định kỳ,
giữ checkpoint tốt nhất (eval loss thấp nhất), lệnh dừng và ghi `metrics.jsonl` / `status.json` đều nằm ở đây.

Giao thức với backend (backend/finetune/jobs.py đọc các file này):
  job.json       cấu hình do backend ghi: params chung + task_config riêng của từng loại
  train.jsonl    mẫu train (đã chia), val.jsonl mẫu val
  metrics.jsonl  mỗi dòng một sự kiện: {"kind": "train"|"eval"|"final", "step": …, …chỉ số}
  status.json    trạng thái hiện tại: state, step, total_steps, eta, message
Dừng: backend gửi SIGINT (rồi SIGKILL) cho tiến trình, dừng ngay, không lưu checkpoint.
Eval golden (lệnh `eval`): eval.json liệt kê các model cần chấm, golden.jsonl là tập chấm, kết quả ở report.json.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


class RunFiles:
    """Thư mục của một lần chạy (job train hoặc lần chấm golden): status.json và metrics.jsonl."""

    def __init__(self, folder: Path) -> None:
        self.dir = folder
        self.status_data: dict = {"state": "starting", "started_at": time.time()}
        self.metrics_file = folder / "metrics.jsonl"

    def status(self, state: str, **extra) -> None:
        self.status_data.update(state=state, updated_at=time.time(), **extra)
        write_json(self.dir / "status.json", self.status_data)

    def metric(self, kind: str, **values) -> None:
        row = {"kind": kind, "time": time.time(), **values}
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Task:
    """Interface mỗi loại fine-tune phải cài đặt. `self.model` là nn.Module đang train."""

    model = None

    def load(self, path: str, config: dict, trainable: bool) -> None:
        """Nạp model từ `path` (model gốc hoặc checkpoint) lên GPU."""
        raise NotImplementedError

    def expand(self, rows: list[dict]) -> list[dict]:
        """Dòng của train.jsonl / val.jsonl -> mẫu train (vd. một câu người dùng thành mẫu intent + mẫu từng gate).
        Gọi sau `load`. Mặc định: mỗi dòng là một mẫu."""
        return rows

    def enable_gradient_checkpointing(self) -> None:
        raise NotImplementedError

    def param_groups(self, lr: float, params: dict) -> list[dict]:
        """Nhóm tham số cho AdamW (vd. head học với lr khác)."""
        raise NotImplementedError

    def loss(self, rows: list[dict], rng: random.Random):
        """Loss trung bình của một micro-batch (tensor có grad)."""
        raise NotImplementedError

    def evaluate(self, rows: list[dict]) -> dict:
        """Chỉ số trên val; bắt buộc có khoá "loss" (dùng để chọn checkpoint tốt nhất)."""
        raise NotImplementedError

    def finalize(self, val_rows: list[dict], params: dict) -> dict:
        """Sau train, trên checkpoint tốt nhất: hiệu chỉnh, chọn ngưỡng… Trả về tóm tắt để ghi vào metrics."""
        return {}

    def save(self, out_dir: Path, source_path: str) -> None:
        raise NotImplementedError

    def golden(self, rows: list[dict], targets: list[dict], files: RunFiles) -> dict:
        """Chấm từng model trong `targets` (path, config, threshold…) trên `rows`; trả về báo cáo (JSON)
        mà giao diện của loại này biết cách vẽ."""
        raise NotImplementedError

    # Giữ bản sao tham số đang train để quay lại checkpoint tốt nhất mà không phải ghi đĩa mỗi lần eval.
    def snapshot(self) -> dict:
        return {k: v.detach().to("cpu", copy=True) for k, v in self.model.named_parameters() if v.requires_grad}

    def restore(self, state: dict) -> None:
        params = dict(self.model.named_parameters())
        for k, v in state.items():
            params[k].data.copy_(v.to(params[k].device))


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Warmup tuyến tính rồi giảm theo cosine về 10%."""
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def vram_gb() -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0.0


def train(task: Task, job_dir: Path) -> None:
    import torch

    files = RunFiles(job_dir)
    cfg = json.loads((job_dir / "job.json").read_text("utf-8"))
    p = cfg["params"]
    random.seed(p["seed"])
    torch.manual_seed(p["seed"])
    rng = random.Random(p["seed"])
    files.status("loading", message="Đang nạp model…")
    task.load(cfg["base_path"], cfg, trainable=True)
    train_rows = task.expand(read_jsonl(job_dir / "train.jsonl"))
    val_rows = task.expand(read_jsonl(job_dir / "val.jsonl"))
    if p.get("gradient_checkpointing"):
        task.enable_gradient_checkpointing()
    task.model.train()

    bs, accum = p["batch_size"], p["grad_accum"]
    steps_per_epoch = math.ceil(len(train_rows) / (bs * accum))
    total = steps_per_epoch * p["epochs"]
    warmup = round(total * p["warmup_ratio"])
    eval_every = p["eval_every"] or steps_per_epoch
    opt = torch.optim.AdamW(task.param_groups(p["lr"], p), weight_decay=p["weight_decay"])
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step: int) -> float:
        f = lr_factor(step, total, warmup)
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f
        return opt.param_groups[0]["lr"]

    best = {"loss": math.inf, "step": 0, "state": None}

    def run_eval(step: int, epoch: float) -> None:
        files.status("evaluating", step=step, message="Đang chấm val…")
        task.model.eval()
        m = task.evaluate(val_rows)
        task.model.train()
        improved = m["loss"] < best["loss"]
        if improved:
            best.update(loss=m["loss"], step=step, state=task.snapshot())
        files.metric("eval", step=step, epoch=round(epoch, 3), best=improved, **m)

    files.status("training", step=0, total_steps=total, steps_per_epoch=steps_per_epoch, message="Chấm val trước khi train…")
    files.metric("start", total_steps=total, steps_per_epoch=steps_per_epoch, n_train=len(train_rows), n_val=len(val_rows))
    run_eval(0, 0.0)

    step = 0
    t0 = time.time()
    window = {"loss": 0.0, "n": 0, "samples": 0, "t": time.time()}
    for epoch in range(p["epochs"]):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        batches = [[train_rows[i] for i in order[s:s + bs]] for s in range(0, len(order), bs)]
        for i, batch in enumerate(batches):
            loss = task.loss(batch, rng)
            (loss / accum).backward()
            window["loss"] += loss.item()
            window["n"] += 1
            window["samples"] += len(batch)
            if (i + 1) % accum and i + 1 != len(batches):
                continue
            lr = set_lr(step)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [q for g in opt.param_groups for q in g["params"]], p["max_grad_norm"]).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            ep = epoch + (i + 1) / len(batches)
            if step % p["log_every"] == 0 or step == total:
                now = time.time()
                elapsed = now - t0
                files.metric("train", step=step, epoch=round(ep, 3), loss=window["loss"] / window["n"], lr=lr,
                             grad_norm=round(grad_norm, 4), samples_per_s=round(window["samples"] / (now - window["t"]), 2),
                             vram_gb=vram_gb())
                window = {"loss": 0.0, "n": 0, "samples": 0, "t": now}
                files.status("training", step=step, epoch=round(ep, 3),
                             eta=elapsed / step * (total - step), message=f"Epoch {epoch + 1}/{p['epochs']}")
            if step % eval_every == 0 or step == total:
                run_eval(step, ep)

    if best["step"] != step and best["state"] is not None:
        task.restore(best["state"])
    files.status("finalizing", step=step, message=f"Hiệu chỉnh trên val (checkpoint bước {best['step']})…")
    task.model.eval()
    summary = task.finalize(val_rows, p)
    files.status("saving", message="Đang lưu checkpoint…")
    task.save(Path(cfg["output_dir"]), cfg["base_path"])
    files.metric("final", best_step=best["step"], best_eval_loss=best["loss"],
                 train_seconds=round(time.time() - t0), **summary)
    files.status("done", step=step, eta=0, message=f"Đã lưu checkpoint tốt nhất (bước {best['step']}).")


def golden(task: Task, eval_dir: Path) -> None:
    files = RunFiles(eval_dir)
    eval_cfg = json.loads((eval_dir / "eval.json").read_text("utf-8"))
    rows = read_jsonl(eval_dir / "golden.jsonl")
    report = task.golden(rows, eval_cfg["targets"], files)
    write_json(eval_dir / "report.json", report)
    files.status("done", message="Đã chấm xong.")


def main(tasks: dict) -> None:
    command, task_id, run_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    task = tasks[task_id]()
    try:
        if command == "train":
            train(task, run_dir)
        else:
            golden(task, run_dir)
    except BaseException as e:  # gồm cả Ctrl-C / pkill -INT: vẫn ghi lại lý do dừng
        files = RunFiles(run_dir)
        prev = run_dir / "status.json"
        if prev.exists():
            files.status_data = json.loads(prev.read_text("utf-8"))
        tail = "".join(traceback.format_exception(e))[-3000:]
        files.status("error", message=f"{type(e).__name__}: {e}", traceback=tail)
        print(tail, file=sys.stderr, flush=True)
        os._exit(1)
