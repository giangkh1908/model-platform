"""Xuất bản một job train đã xong lên model registry.

Dùng lại run MLflow của job (tracking.py; chưa có thì ghi trước), thêm:
  model/                     checkpoint (đọc từ runtime qua connector)
  platform/eval_report.json  kết quả lần eval gần nhất có chấm job (nếu có), kèm chỉ số golden_*
rồi đăng ký thư mục model/ thành phiên bản mới của model.
Prompt (template, policy) không đóng gói: nguồn duy nhất là kho prompt, phiên bản chỉ ghi thành tag "<tên>@v<N>".
"""

from __future__ import annotations

import re
import time

from ..core import LOG
from ..model_registry import REGISTRY
from . import jobs, tracking
from .connector import CONNECTOR
from .store import ascii_fold

NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")


def default_name(job: dict) -> str:
    """Tên model mặc định: <loại>-<tên build không dấu>, vd. decider-intent."""
    build_name = (job["build"]["title"].rsplit(" v", 1)[0]).strip()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_fold(build_name).lower()).strip("-") or "model"
    return f"{job['task']}-{slug}"


def check_name(name: str) -> str:
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError("Tên model chỉ gồm chữ không dấu, số, dấu chấm, gạch ngang, gạch dưới (vd. decider-intent).")
    return name


def publish(job_id: str, name: str | None, description: str) -> dict:
    job = jobs.JOBS.get(job_id)
    if job["state"] != "done":
        raise ValueError("Job chưa train xong, chưa có checkpoint để xuất bản.")
    name = check_name((name or "").strip() or default_name(job))
    model_dir = CONNECTOR.output_path(job["output_dir"])
    if not model_dir.is_dir():
        raise ValueError(f"Không tìm thấy checkpoint ({job['output_dir']}). WSL có đang bật không?")

    run_id = job.get("mlflow_run_id")
    if not run_id or not REGISTRY.has_run(run_id):
        run_id = tracking.log_job(job_id)["run_id"]

    tags = tracking.tags_of(job)
    metrics = {}
    extra_files = {}
    found = tracking.latest_eval(job_id)
    if found:
        eval_meta, entry = found
        extra_files["eval_report.json"] = {"eval": {"id": eval_meta["id"], "name": eval_meta["name"],
                                                    "source": eval_meta["source"], "n": eval_meta["n"]}, **entry}
        metrics = tracking.eval_metrics(entry)
        tags["eval_id"] = eval_meta["id"]

    LOG.add(f"=== Đang xuất bản {job_id} lên registry thành \"{name}\"… ===")
    with tracking.json_folder(extra_files) as platform_dir:
        folders = {"model": model_dir}
        if extra_files:
            folders["platform"] = platform_dir
        REGISTRY.attach(run_id, folders, metrics)
    result = REGISTRY.register(name, run_id, tags, description.strip()[:500])

    published = [*(job.get("published") or []), {"name": result["name"], "version": result["version"], "at": time.time()}]
    jobs.JOBS.update(job_id, published=published)
    LOG.add(f"=== Đã xuất bản {name} phiên bản {result['version']} ===")
    return result
