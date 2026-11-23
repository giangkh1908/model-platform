"""Build (manifest): một phiên bản dữ liệu để train, gồm một bộ quyết định (spec) + các dataset.

Spec = một system + các intent + các gate, mỗi prompt chọn theo phiên bản hoặc nhãn dev / prod, lưu lại đủ nội dung và
số phiên bản cụ thể đã dùng. Mỗi mẫu của dataset thành một dòng "như thật" (câu người dùng, lịch sử, nhãn intent, nhãn
từng gate theo spec): đúng thứ serving sẽ hỏi, cộng đáp án. Ghép spec + dòng thành prompt của model là việc của runtime.

Dataset SFT vào phần train; dataset Eval (nếu có) làm val khi train và được chấm tự động sau khi train; không có thì
chia val ngẫu nhiên từ train. Build giữ bản sao dòng và spec, nên train lại luôn ra đúng dữ liệu cũ.
Cùng tên thì đánh số phiên bản tăng dần (cskh v1, v2…).

data/finetune/builds/<id>/: meta.json, train.jsonl, val.jsonl
"""

from __future__ import annotations

import time
from collections import Counter

from . import data, datasets
from .prompts import PROMPTS, ref_title
from .sources import source_of
from .store import MetaStore, make_id, read_jsonl, write_jsonl
from .tasks import task_of

BUILDS = MetaStore("builds", "build")
PREVIEW_ROWS = 3


def _resolve(kind: str, ref: dict | None) -> dict:
    """{id, version} hoặc {id, label: dev|prod} -> phiên bản cụ thể, kèm cách đã chọn ("v3" hoặc "prod")."""
    if not ref or not ref.get("id"):
        raise ValueError(f"Chọn {kind}.")
    p = PROMPTS.get(kind, ref["id"], ref.get("version"), ref.get("label"))
    p["selected_by"] = ref.get("label") or f"v{p['version']}"
    return p


def assemble_spec(body: dict) -> dict:
    system = _resolve("system", body.get("system"))
    intents = [_resolve("intent", r) for r in body.get("intents") or []]
    gates = [_resolve("gate", r) for r in body.get("gates") or []]
    for kind, items in (("intent", intents), ("gate", gates)):
        names = [p["name"] for p in items]
        if len(names) != len(set(names)):
            raise ValueError(f"Mỗi {kind} chỉ chọn một lần.")
    return {
        "system": {"name": system["name"], "version": system["version"], "selected_by": system["selected_by"],
                   "text": system["content"]["text"], "intent_question": system["content"].get("intent_question", "")},
        "intents": [{"key": p["name"], "name": p["name"], "version": p["version"], "selected_by": p["selected_by"],
                     "policy": p["content"]["text"]} for p in intents],
        "gates": [{"key": p["name"], "name": p["name"], "version": p["version"], "selected_by": p["selected_by"],
                   "policy": p["content"]["text"], "question": p["content"].get("question", "")} for p in gates],
    }


def spec_title(spec: dict) -> dict:
    """Tham chiếu phiên bản để hiện và gắn tag: {system, intents, gates} dạng "<tên>@v<N>"."""
    return {
        "system": ref_title(spec["system"]),
        "intents": ", ".join(ref_title(i) for i in spec["intents"]),
        "gates": ", ".join(ref_title(g) for g in spec["gates"]),
    }


def _samples(task, dataset_ids: list[str], role: str) -> tuple[list[dict], list[dict]]:
    out, described = [], []
    for dataset_id in dict.fromkeys(dataset_ids):
        m = datasets.DATASETS.get(dataset_id)
        if m["task"] != task.id:
            raise ValueError(f"Dataset \"{m['name']}\" thuộc loại fine-tune khác.")
        source = source_of(m["source"])
        if source.role != role:
            raise ValueError(f"Dataset \"{m['name']}\" là loại {source.name}, không dùng cho phần này.")
        rows = [{**s, "dataset": m["name"]} for s in datasets.samples(dataset_id)]
        out.extend(rows)
        described.append({"id": m["id"], "name": m["name"], "rows": len(rows)})
    return out, described


def gate_stats(spec: dict, train: list[dict], val: list[dict]) -> list[dict]:
    """Số câu dính / không dính từng gate ở train và val (câu không gán gate thì không tính)."""
    out = []
    for g in spec["gates"]:
        key = g["key"]
        counts = {}
        for part, rows in (("train", train), ("val", val)):
            c = Counter(r["gates"][key] for r in rows if key in r["gates"])
            counts[f"{part}_true"], counts[f"{part}_false"] = c[True], c[False]
        out.append({"key": key, **counts})
    return out


def _assemble(body: dict) -> tuple:
    task = task_of(body.get("task", ""))
    spec = assemble_spec(body)
    train_samples, train_desc = _samples(task, body.get("datasets") or [], "train")
    if not train_samples:
        raise ValueError("Chọn ít nhất một dataset SFT cho phần train.")
    eval_samples, eval_desc = _samples(task, body.get("eval_datasets") or [], "eval")
    train_all, dropped = task.make_rows(spec, train_samples)
    eval_rows, eval_dropped = task.make_rows(spec, eval_samples)
    train_all, merge_stats = data.merge(task, [train_all], bool(body.get("dedupe")))
    seed = int(body.get("split_seed", 42))
    val_percent = float(body.get("val_percent", 10))
    if eval_rows:
        train, val = train_all, eval_rows
        split_stats = data.describe_split(task, train, val)
        train_keys = {task.group_key(s) for s in train}
        split_stats["leaked"] = sum(task.group_key(s) in train_keys for s in val)
    else:
        train, val, split_stats = data.split(task, train_all, val_percent / 100, seed)
        split_stats["leaked"] = 0
    config = task.build_config(spec, train + val)
    lengths = [len(s["input"]) for s in train + val]
    stats = {**merge_stats, **split_stats, "val_from": "eval" if eval_rows else "random",
             "dropped_no_label": dropped + eval_dropped, "gates": gate_stats(spec, train, val),
             "max_chars": max(lengths), "avg_chars": round(sum(lengths) / len(lengths))}
    manifest = {"spec_title": spec_title(spec), "datasets": train_desc, "eval_datasets": eval_desc}
    return task, manifest, config, train, val, stats, val_percent, seed


def preview(body: dict) -> dict:
    _, manifest, config, train, _, stats, _, _ = _assemble(body)
    return {**stats, "spec": config, "spec_title": manifest["spec_title"], "examples": train[:PREVIEW_ROWS]}


def next_version(name: str) -> int:
    versions = [b["version"] for b in BUILDS.all() if b["name"].lower() == name.lower()]
    return max(versions, default=0) + 1


def create(body: dict) -> dict:
    task, manifest, config, train, val, stats, val_percent, seed = _assemble(body)
    name = str(body.get("name") or "").strip()[:60]
    if not name:
        raise ValueError("Đặt tên cho build (vd. \"cskh\"). Cùng tên sẽ thành phiên bản mới.")
    version = next_version(name)
    meta = BUILDS.create({
        "id": make_id(f"{name} v{version}"), "name": name, "version": version, "task": task.id,
        "created": time.time(), "manifest": manifest, "task_config": config, "dedupe": bool(body.get("dedupe")),
        "val_percent": val_percent, "split_seed": seed, "note": str(body.get("note") or "").strip()[:300],
        "stats": stats, "has_eval": bool(manifest["eval_datasets"]),
        "refs": {"system": body.get("system"), "intents": body.get("intents") or [], "gates": body.get("gates") or [],
                 "datasets": body.get("datasets") or [], "eval_datasets": body.get("eval_datasets") or []},
    })
    folder = BUILDS.dir(meta["id"])
    write_jsonl(folder / "train.jsonl", train)
    write_jsonl(folder / "val.jsonl", val)
    return meta


def title(meta: dict) -> str:
    return f"{meta['name']} v{meta['version']}"


def split_files(build_id: str) -> tuple[list[dict], list[dict]]:
    BUILDS.get(build_id)
    folder = BUILDS.dir(build_id)
    return read_jsonl(folder / "train.jsonl"), read_jsonl(folder / "val.jsonl")


def detail(build_id: str) -> dict:
    train, _ = split_files(build_id)
    return {"build": BUILDS.get(build_id), "examples": train[:PREVIEW_ROWS]}


def delete(build_id: str) -> str:
    m = BUILDS.get(build_id)
    BUILDS.delete(build_id)
    return f"Đã xoá build \"{title(m)}\". Các job đã train từ nó vẫn giữ nguyên dữ liệu."
