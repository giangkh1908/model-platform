"""Ghi mỗi job train đã kết thúc (xong, lỗi, bị dừng) thành một run trên MLflow.

Run gồm: tham số, tag (build, model gốc, phiên bản prompt), lịch sử chỉ số theo bước từ metrics.jsonl (train_*, eval_*,
final_*) và thư mục platform/ (task_config.json, params.json, final.json). Giao diện platform chỉ vẽ loss / eval loss;
mọi chỉ số khác xem trên MLflow. MLflow tắt hay lỗi thì job vẫn kết thúc bình thường, chỉ ghi lại trong Log; ghi bù
bằng nút "Ghi lên MLflow" ở trang job.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ..core import LOG, load_json
from ..model_registry import REGISTRY, Point
from ..registry import REG
from . import builds, evals, jobs
from .connector import PROTOCOL_VERSION

RUN_STATUS = {"done": "FINISHED", "error": "FAILED", "stopped": "KILLED"}
SKIP_KEYS = {"kind", "time", "step", "epoch", "best"}  # khoá của metrics.jsonl không phải chỉ số


def latest_eval(job_id: str) -> tuple[dict, dict] | None:
    """Lần eval xong gần nhất có chấm job này: (meta lần eval, kết quả của job trong báo cáo)."""
    key = f"job:{job_id}"
    for meta in sorted(evals.EVALS.all(), key=lambda m: m["created"], reverse=True):
        if meta["state"] != "done":
            continue
        report = load_json(evals.EVALS.dir(meta["id"]) / "report.json", None) or {}
        entry = next((m for m in report.get("models", []) if m.get("key") == key), None)
        if entry:
            return meta, entry
    return None


def eval_metrics(entry: dict) -> dict:
    """Chỉ số chính của một model trong báo cáo eval: eval_acc, eval_macro_f1, eval_ece, *_at_threshold."""
    metrics = {}
    summary = entry.get("summary") or {}
    for key in ("acc", "macro_f1", "ece"):
        if key in summary:
            metrics[f"golden_{key}"] = summary[key]
    at = entry.get("at_threshold") or {}
    for key in ("precision", "coverage"):
        if key in at:
            metrics[f"golden_{key}_at_threshold"] = at[key]
    return metrics


def prompt_tags(build: dict | None) -> dict:
    """Tham chiếu prompt đã dùng: template một bản, policy có thể nhiều bản (mỗi cặp policy + dataset một bản)."""
    if not build:
        return {}
    manifest = build["manifest"]
    template = manifest["template"]
    policies = []
    for entry in manifest["entries"] + manifest["eval_entries"]:
        ref = f"{entry['policy']['name']}@v{entry['policy']['version']}"
        if ref not in policies:
            policies.append(ref)
    return {"prompt.template": f"{template['name']}@v{template['version']}", "prompt.policy": ", ".join(policies)}


def tags_of(job: dict) -> dict:
    build = None
    try:
        build = builds.BUILDS.get(job["build"]["id"])
    except ValueError:  # build đã xoá: vẫn ghi được, chỉ thiếu tag prompt
        pass
    base = REG.get(job["base_model"]) or {}
    return {
        "task": job["task"], "job_id": job["id"], "job_name": job["name"],
        "base_model": job["base_model"], "base_model_repo": base.get("repo", ""),
        "build": job["build"]["title"], "build_id": job["build"]["id"],
        "runtime_protocol": str(PROTOCOL_VERSION), "source": "model-platform",
        **prompt_tags(build),
    }


def params_of(job: dict) -> dict:
    return {**job["params"], "base_model": job["base_model"], "build": job["build"]["title"],
            "n_train": job["build"]["train"], "n_val": job["build"]["val"]}


def flatten(prefix: str, values: dict) -> dict:
    """Chỉ số số học, gồm cả một tầng lồng nhau: {"threshold": {"precision": 0.9}} -> final_threshold_precision."""
    out = {}
    for key, value in values.items():
        if key in SKIP_KEYS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[f"{prefix}{key}"] = value
        elif isinstance(value, dict):
            for sub, sub_value in value.items():
                if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                    out[f"{prefix}{key}_{sub}"] = sub_value
    return out


def history(job_id: str) -> list[Point]:
    """metrics.jsonl -> các điểm chỉ số: dòng train thành train_*, eval thành eval_*, final thành final_* (bước cuối)."""
    rows, _ = jobs.read_metrics(job_id)
    points: list[Point] = []
    last_step = 0
    for row in rows:
        kind = row.get("kind")
        step = int(row["step"]) if row.get("step") is not None else last_step  # bước 0 là bước thật, không phải "thiếu"
        last_step = max(last_step, step)
        ts = int((row.get("time") or 0) * 1000)
        if kind in ("train", "eval"):
            values = flatten(f"{kind}_", row)
        elif kind == "final":
            values = flatten("final_", row)
            step = last_step
        else:
            continue
        points.extend(Point(key, value, step, ts) for key, value in values.items())
    return points


def config_files(job: dict) -> dict:
    final = jobs.final_metrics(job["id"])
    return {
        "task_config.json": job["task_config"],
        "params.json": job["params"],
        "final.json": {k: v for k, v in final.items() if k not in ("kind", "time")},
    }


@contextmanager
def json_folder(files: dict):
    """Thư mục tạm chứa các file JSON, xoá khi xong."""
    with tempfile.TemporaryDirectory(prefix="mlflow-") as tmp:
        folder = Path(tmp)
        for filename, content in files.items():
            (folder / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2), "utf-8")
        yield folder


def log_job(job_id: str, state: str | None = None) -> dict:
    """Ghi job lên MLflow, lưu run_id / url vào meta của job. `state`: trạng thái cuối (mặc định lấy từ meta)."""
    job = jobs.JOBS.get(job_id)
    state = state or job["state"]
    if state not in RUN_STATUS:
        raise ValueError("Job chưa kết thúc, chưa ghi lên MLflow được.")
    with json_folder(config_files(job)) as platform_dir:
        run = REGISTRY.log_run(job["name"], params_of(job), history(job_id), tags_of(job), {"platform": platform_dir},
                               RUN_STATUS[state])
    jobs.JOBS.update(job_id, mlflow_run_id=run["run_id"], mlflow_url=run["url"])
    LOG.add(f"=== Đã ghi job {job_id} lên MLflow ===")
    return run


def on_finish(meta: dict, state: str) -> None:
    """Hook sau khi job kết thúc: không để lỗi MLflow làm hỏng hàng đợi."""
    if not REGISTRY.enabled:
        return
    try:
        log_job(meta["id"], state)
    except Exception as e:  # MLflow tắt, mạng lỗi…: job vẫn xong, ghi bù sau
        LOG.add(f"=== Job {meta['id']}: chưa ghi được lên MLflow ({e}). Bấm \"Ghi lên MLflow\" ở trang job để ghi bù. ===")


jobs.FINISH_HOOKS.append(on_finish)
