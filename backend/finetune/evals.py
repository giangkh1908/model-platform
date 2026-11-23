"""Eval: chấm nhiều model (model gốc và các adapter đã train) trên phần eval của một build rồi xếp hạng.
Dữ liệu eval chỉ lấy từ build (đã ghép template + policy), nên mọi model được chấm trên đúng cùng một prompt.
Tách riêng với phần chấm val định kỳ trong lúc train (runtime/finetune/common.py). Chạy qua hàng đợi GPU.

data/finetune/evals/<id>/: meta.json, eval.json (gửi runtime), golden.jsonl, status.json, report.json
"""

from __future__ import annotations

import time

from ..core import load_json, save_json
from ..registry import REG
from . import builds, jobs
from .connector import PROTOCOL_VERSION
from .gpu_queue import QUEUE, RUNNERS, Runner
from .store import MetaStore, clean_name, make_id, write_jsonl
from .tasks import task_of

EVALS = MetaStore("evals", "lần eval")
KIND = "eval"


def root_base(model_id: str) -> str:
    """Model gốc của một model: adapter train tiếp từ adapter khác vẫn quy về model gốc ban đầu."""
    seen = set()
    while model_id not in seen:
        seen.add(model_id)
        m = REG.get(model_id)
        jid = m.get("finetune_job") if m else None
        if not jid:
            break
        try:
            model_id = jobs.JOBS.get(jid)["base_model"]
        except ValueError:  # job đã bị xoá
            break
    return model_id


def model_choices(task) -> dict:
    """Model gốc chọn được, và mọi adapter đã train xong kèm model gốc của nó (để lọc theo model gốc đang chọn)."""
    bases = [{"id": m["id"], "ref": f"model:{m['id']}"} for m in task.base_models(REG.public()) if not m.get("finetune_job")]
    adapters = []
    for j in jobs.JOBS.all():
        if j["task"] != task.id or j["state"] != "done":
            continue
        adapters.append({"ref": f"job:{j['id']}", "name": j.get("registered_model") or j["name"], "job_name": j["name"],
                         "root_base": root_base(j["base_model"]), "base_model": j["base_model"],
                         "build": j["build"]["title"], "created": j["created"]})
    return {"bases": bases, "adapters": adapters}


def _target(task, ref: str, rows: list[dict], config: dict) -> dict:
    """Mọi model chấm trên cùng prompt đã ghép và cùng câu hỏi / nhãn của build; adapter giữ ngưỡng riêng của nó."""
    kind, _, ident = ref.partition(":")
    if kind == "job":
        j = jobs.JOBS.get(ident)
        if j["state"] != "done":
            raise ValueError(f"Job \"{j['name']}\" chưa train xong.")
        task.check_eval(rows, j["task_config"], j["name"])
        train_keys = {task.group_key(s) for s in jobs.train_rows(ident)}
        threshold = (jobs.final_metrics(ident).get("threshold") or {}).get("threshold")
        return {"key": ref, "name": j.get("registered_model") or j["name"], "kind": "adapter", "path": j["output_dir"],
                "config": config, "threshold": threshold,
                "leaked": sum(task.group_key(s) in train_keys for s in rows)}
    m = REG.get(ident)
    if kind != "model" or not m:
        raise ValueError(f"Model không hợp lệ: {ref}.")
    return {"key": ref, "name": m["id"], "kind": "base", "path": m["path"], "config": config, "threshold": None, "leaked": 0}


def create(body: dict) -> dict:
    """Chấm model gốc + các adapter đã chọn trên phần eval của một build (không có phần eval thì phần val)."""
    build = builds.BUILDS.get(body.get("build_id") or "")
    task = task_of(build["task"])
    _, rows = builds.split_files(build["id"])
    base = REG.get(body.get("base_model") or "")
    if not base or base.get("finetune_job"):
        raise ValueError("Chọn model gốc.")
    adapters = list(dict.fromkeys(body.get("adapters") or []))
    for ref in adapters:
        kind, _, jid = ref.partition(":")
        if kind != "job" or root_base(jobs.JOBS.get(jid)["base_model"]) != base["id"]:
            raise ValueError(f"Adapter {ref} không được train từ {base['id']}.")
    # Model gốc luôn được chấm để làm mốc so sánh.
    targets = [_target(task, ref, rows, build["task_config"]) for ref in [f"model:{base['id']}", *adapters]]
    part = "phần eval" if build.get("has_eval") else "phần val"
    source = {"id": build["id"], "title": f"{builds.title(build)} ({part})"}
    name = clean_name(body.get("name"), f"{builds.title(build)} · {len(targets)} model")
    meta = EVALS.create({
        "id": make_id(name), "name": name, "task": task.id, "created": time.time(), "state": "queued",
        "source": source, "n": len(rows), "auto": bool(body.get("auto")),
        "models": [{k: t[k] for k in ("key", "name", "kind", "threshold", "leaked")} for t in targets],
    })
    folder = EVALS.dir(meta["id"])
    write_jsonl(folder / "golden.jsonl", rows)
    save_json(folder / "eval.json", {"protocol": PROTOCOL_VERSION, "targets": targets})
    QUEUE.enqueue(KIND, meta["id"])
    return meta


def auto_eval(jid: str) -> None:
    """Sau khi train xong: build có phần eval thì tự chấm adapter vừa train so với model gốc."""
    j = jobs.JOBS.get(jid)
    if not j["build"].get("has_eval"):
        return
    create({"build_id": j["build"]["id"], "base_model": root_base(j["base_model"]), "adapters": [f"job:{jid}"],
            "name": f"Tự eval · {j['name']}", "auto": True})


jobs.DONE_HOOKS.append(auto_eval)


def list_all() -> list[dict]:
    return [{**m, "queue_position": QUEUE.position(KIND, m["id"]), "status": load_json(EVALS.dir(m["id"]) / "status.json", {})}
            for m in EVALS.all()]


def detail(eval_id: str) -> dict:
    folder = EVALS.dir(eval_id)
    return {"eval": EVALS.get(eval_id), "status": load_json(folder / "status.json", {}),
            "report": load_json(folder / "report.json", None), "queue_position": QUEUE.position(KIND, eval_id)}


def stop(eval_id: str) -> str:
    EVALS.get(eval_id)
    return QUEUE.stop(KIND, eval_id)


def delete(eval_id: str) -> str:
    m = EVALS.get(eval_id)
    if m["state"] in ("queued", "running"):
        raise ValueError("Lần eval đang chờ hoặc đang chạy, hãy dừng trước.")
    EVALS.delete(eval_id)
    return "Đã xoá lần eval."


def _prepare(meta: dict):
    folder = EVALS.dir(meta["id"])
    (folder / "status.json").unlink(missing_ok=True)
    (folder / "report.json").unlink(missing_ok=True)
    return folder, f"eval {meta['id']}"


RUNNERS[KIND] = Runner(store=EVALS, command="eval", prepare=_prepare, python=lambda m: task_of(m["task"]).runtime_python)
