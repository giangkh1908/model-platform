"""Xuất bản một job train đã xong lên model registry.

Gói gồm:
  model/     checkpoint (đọc từ runtime qua connector)
  platform/  task_config.json (nhãn, câu hỏi), params.json, final.json (ngưỡng…), eval_report.json (nếu đã eval)
Prompt (template, policy) không đóng gói: nguồn duy nhất là kho prompt, phiên bản chỉ ghi thành tag "<tên>@v<N>".
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path

from ..core import LOG, load_json
from ..model_registry import REGISTRY
from ..registry import REG
from . import builds, evals, jobs
from .connector import CONNECTOR, PROTOCOL_VERSION
from .store import ascii_fold

NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")


def default_name(job: dict) -> str:
    """Tên model mặc định: <loại>-<tên build>, vd. decider-intent."""
    build_name = (job["build"]["title"].rsplit(" v", 1)[0]).strip()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_fold(build_name).lower()).strip("-") or "model"
    return f"{job['task']}-{slug}"


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


def collect(job: dict) -> tuple[dict, dict, dict, dict]:
    """(file platform/, params, metrics, tags) của một job. Tách riêng để test không cần registry."""
    final = jobs.final_metrics(job["id"])
    threshold = (final.get("threshold") or {}).get("threshold")
    files = {
        "task_config.json": job["task_config"],
        "params.json": job["params"],
        "final.json": {k: v for k, v in final.items() if k not in ("val_curve", "kind", "time")},
    }
    metrics = {"best_eval_loss": final.get("best_eval_loss")}
    if threshold is not None:
        metrics["threshold"] = threshold

    found = latest_eval(job["id"])
    if found:
        eval_meta, entry = found
        files["eval_report.json"] = {"eval": {"id": eval_meta["id"], "name": eval_meta["name"],
                                              "source": eval_meta["source"], "n": eval_meta["n"]}, **entry}
        summary = entry.get("summary") or {}
        for key in ("acc", "macro_f1", "ece"):
            if key in summary:
                metrics[f"eval_{key}"] = summary[key]
        at = entry.get("at_threshold") or {}
        for key in ("precision", "coverage"):
            if key in at:
                metrics[f"eval_{key}_at_threshold"] = at[key]

    build = None
    try:
        build = builds.BUILDS.get(job["build"]["id"])
    except ValueError:  # build đã xoá: vẫn xuất bản được, chỉ thiếu tag prompt
        pass
    base = REG.get(job["base_model"]) or {}
    tags = {
        "task": job["task"], "job_id": job["id"], "job_name": job["name"],
        "base_model": job["base_model"], "base_model_repo": base.get("repo", ""),
        "build": job["build"]["title"], "build_id": job["build"]["id"],
        "runtime_protocol": str(PROTOCOL_VERSION), "published_from": "model-platform",
        **prompt_tags(build),
    }
    if found:
        tags["eval_id"] = found[0]["id"]
    params = {**job["params"], "base_model": job["base_model"], "build": job["build"]["title"],
              "n_train": job["build"]["train"], "n_val": job["build"]["val"]}
    metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    return files, params, metrics, tags


def publish(job_id: str, name: str | None, description: str) -> dict:
    job = jobs.JOBS.get(job_id)
    if job["state"] != "done":
        raise ValueError("Job chưa train xong, chưa có checkpoint để xuất bản.")
    name = (name or "").strip() or default_name(job)
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError("Tên model chỉ gồm chữ không dấu, số, dấu chấm, gạch ngang, gạch dưới (vd. decider-intent).")
    model_dir = CONNECTOR.output_path(job["output_dir"])
    if not model_dir.is_dir():
        raise ValueError(f"Không tìm thấy checkpoint ({job['output_dir']}). WSL có đang bật không?")

    files, params, metrics, tags = collect(job)
    with tempfile.TemporaryDirectory(prefix="publish-") as tmp:
        platform_dir = Path(tmp)
        for filename, content in files.items():
            (platform_dir / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2), "utf-8")
        LOG.add(f"=== Đang xuất bản {job_id} lên registry thành \"{name}\"… ===")
        result = REGISTRY.publish(name, {"model": model_dir, "platform": platform_dir}, params, metrics, tags,
                                  description.strip()[:500])

    published = [*(job.get("published") or []), {"name": result["name"], "version": result["version"], "at": time.time()}]
    jobs.JOBS.update(job_id, published=published)
    LOG.add(f"=== Đã xuất bản {name} phiên bản {result['version']} ===")
    return result
