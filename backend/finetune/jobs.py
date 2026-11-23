"""Job train: một build + một model gốc + tham số. Chạy qua hàng đợi GPU (gpu_queue.py).
Câu hỏi và nhãn lấy từ build (phần output của template). Train xong, các hàm trong DONE_HOOKS được gọi
(evals.py đăng ký tự eval khi build có phần eval).

data/finetune/jobs/<id>/:
  meta.json      tên, build, model gốc, trạng thái hàng đợi
  job.json       cấu hình gửi cho runtime (runtime/finetune/common.py)
  train.jsonl, val.jsonl (chép từ build), metrics.jsonl, status.json
Checkpoint nằm trong WSL: FINETUNE_OUT/<id> (tên thư mục = tên model mà strands-decider báo ở /health).
"""

from __future__ import annotations

import json
import time

from ..config import DEFAULT_OPTIONS, FINETUNE_OUT, wsl_path
from ..core import LOG, load_json, q, save_json, wsl
from ..registry import REG
from . import builds
from .gpu_queue import QUEUE, RUNNERS, Runner
from .store import MetaStore, clean_name, make_id, read_jsonl, write_jsonl
from .tasks import task_of

JOBS = MetaStore("jobs", "job")
KIND = "train"
DONE_HOOKS: list = []  # hàm(job_id) gọi sau khi job train xong


def create(body: dict) -> dict:
    build = builds.BUILDS.get(body.get("build_id") or "")
    task = task_of(build["task"])
    train, val = builds.split_files(build["id"])
    base = REG.get(body.get("base_model") or "")
    if not base or base["id"] not in {m["id"] for m in task.base_models(REG.public())}:
        raise ValueError("Chọn model hợp lệ cho loại fine-tune này.")
    task_config = build["task_config"]
    params = task.clean_params(body.get("params") or {})

    name = clean_name(body.get("name"), f"{builds.title(build)} · {base['id']}")
    jid = make_id(name)
    output_dir = f"{FINETUNE_OUT}/{jid}"
    meta = JOBS.create({
        "id": jid, "name": name, "task": task.id, "created": time.time(), "state": "queued",
        "base_model": base["id"], "output_dir": output_dir, "params": params, "task_config": task_config,
        "build": {"id": build["id"], "title": builds.title(build), "train": len(train), "val": len(val),
                  "has_eval": build.get("has_eval", False)},
        "registered_model": None,
    })
    folder = JOBS.dir(jid)
    write_jsonl(folder / "train.jsonl", train)
    write_jsonl(folder / "val.jsonl", val)
    save_json(folder / "job.json", {"id": jid, "task": task.id, "base_path": base["path"], "output_dir": output_dir,
                                    "params": params, "task_config": task_config})
    QUEUE.enqueue(KIND, jid)
    LOG.add(f"=== Thêm job fine-tune {jid} vào hàng đợi ===")
    return meta


def read_metrics(jid: str, since: int = 0) -> tuple[list[dict], int]:
    path = JOBS.dir(jid) / "metrics.jsonl"
    if not path.exists():
        return [], 0
    rows = []
    for line in path.read_text("utf-8").splitlines()[since:]:
        try:
            rows.append(json.loads(line))
        except ValueError:  # dòng đang ghi dở
            break
    return rows, since + len(rows)


def final_metrics(jid: str) -> dict:
    rows, _ = read_metrics(jid)
    return next((r for r in reversed(rows) if r["kind"] == "final"), {})


def list_all() -> list[dict]:
    return [{**m, "queue_position": QUEUE.position(KIND, m["id"]), "status": load_json(JOBS.dir(m["id"]) / "status.json", {})}
            for m in JOBS.all()]


def detail(jid: str, since: int = 0) -> dict:
    m = JOBS.get(jid)
    metrics, next_since = read_metrics(jid, since)
    return {"job": m, "status": load_json(JOBS.dir(jid) / "status.json", {}), "metrics": metrics,
            "next": next_since, "queue_position": QUEUE.position(KIND, jid)}


def stop(jid: str) -> str:
    JOBS.get(jid)
    return QUEUE.stop(KIND, jid)


def register_model(jid: str) -> dict:
    m = JOBS.get(jid)
    if m["state"] != "done":
        raise ValueError("Job chưa train xong, chưa có checkpoint.")
    if m.get("registered_model") and REG.get(m["registered_model"]):
        raise ValueError(f"Đã có trong danh sách model: {m['registered_model']}.")
    entry = task_of(m["task"]).model_entry(m)
    base = REG.get(m["base_model"]) or {}
    with REG.lock:
        mid = REG.unique_id(f"ft-{m['name']}")
        REG.models.append({
            "id": mid, "repo": f"finetune/{jid}", "path": m["output_dir"], "file": None,
            "size": base.get("size", 0), "status": "ready", "options": dict(DEFAULT_OPTIONS[entry["runtime"]]),
            "managed_path": False, "finetune_job": jid, **entry,
        })
        REG.save()
    JOBS.update(jid, registered_model=mid)
    return {"id": mid}


def delete(jid: str) -> str:
    m = JOBS.get(jid)
    if m["state"] in ("queued", "running"):
        raise ValueError("Job đang chờ hoặc đang chạy, hãy dừng trước.")
    if m.get("registered_model") and REG.get(m["registered_model"]):
        raise ValueError(f"Checkpoint đang dùng cho model {m['registered_model']}: xoá model đó trong trang Model trước.")
    wsl(f"rm -rf {q(m['output_dir'])}", 60)
    JOBS.delete(jid)
    LOG.add(f"=== Xoá job fine-tune {jid} ===")
    return "Đã xoá job và checkpoint."


def train_rows(jid: str) -> list[dict]:
    return read_jsonl(JOBS.dir(jid) / "train.jsonl")


# ---------------------------------------------------------------- chạy trong hàng đợi


def _prepare(meta: dict):
    folder = JOBS.dir(meta["id"])
    (folder / "metrics.jsonl").unlink(missing_ok=True)
    (folder / "status.json").unlink(missing_ok=True)
    return folder, f"train {meta['task']} {q(wsl_path(folder))}", f"fine-tune {meta['id']}"


def _finish(meta: dict, state: str) -> None:
    if state != "done":  # dừng hoặc lỗi giữa chừng: không giữ checkpoint dở
        wsl(f"rm -rf {q(meta['output_dir'])}", 60)


def _on_done(jid: str) -> None:
    for hook in DONE_HOOKS:
        try:
            hook(jid)
        except Exception as e:  # tự eval lỗi không được làm hỏng job đã train xong
            LOG.add(f"=== Job {jid}: không tạo được lần tự eval: {e} ===")


RUNNERS[KIND] = Runner(store=JOBS, prepare=_prepare, python=lambda m: task_of(m["task"]).runtime_python,
                       finish=_finish, on_done=_on_done)
