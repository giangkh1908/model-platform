"""Build (manifest): một phiên bản dữ liệu để train, ghép từ dataset + prompt.

Manifest = một template + các cặp (policy, dataset SFT) cho phần train + tuỳ chọn các cặp (policy, dataset Eval)
cho phần eval. Mỗi dòng được ghép thành prompt hoàn chỉnh (template + policy + lịch sử + câu): 1 mẫu = 1 dòng.
Có phần eval thì nó làm val khi train và được chấm tự động sau khi train; không có thì chia val ngẫu nhiên từ train.
Build giữ bản sao mẫu đã ghép và nội dung prompt đã dùng, nên train lại luôn ra đúng dữ liệu cũ.
Cùng tên thì đánh số phiên bản tăng dần (moderation v1, v2…).

data/finetune/builds/<id>/: meta.json, train.jsonl, val.jsonl
"""

from __future__ import annotations

import time

from . import data, datasets
from .prompts import PROMPTS, ref_title
from .sources import source_of
from .store import MetaStore, make_id, read_jsonl, write_jsonl
from .tasks import task_of

BUILDS = MetaStore("builds", "build")
PREVIEW_PROMPTS = 3


def _prompt_ref(ref: dict | None, kind: str) -> dict:
    if not ref or not ref.get("id"):
        raise ValueError(f"Chọn {kind}.")
    return PROMPTS.get(kind, ref["id"], ref.get("version"))


def _entries(task, template: dict, entries: list[dict], role: str) -> tuple[list[dict], list[dict]]:
    """Ghép từng cặp (policy, dataset) của một phần. Trả về (mẫu đã ghép, mô tả các cặp)."""
    rendered, described = [], []
    for e in entries:
        m = datasets.DATASETS.get(e.get("dataset_id") or "")
        if m["task"] != task.id:
            raise ValueError(f"Dataset \"{m['name']}\" thuộc loại fine-tune khác.")
        source = source_of(m["source"])
        if source.role != role:
            raise ValueError(f"Dataset \"{m['name']}\" là loại {source.name}, không dùng cho phần này.")
        policy = _prompt_ref(e.get("policy"), "policy")
        rows = [{**task.render(template["content"], policy["content"]["text"], r), "dataset": m["name"],
                 "policy": ref_title(policy)} for r in datasets.samples(m["id"])]
        rendered.extend(rows)
        described.append({"dataset": {"id": m["id"], "name": m["name"], "rows": len(rows)},
                          "policy": {"id": policy["id"], "name": policy["name"], "version": policy["version"],
                                     "text": policy["content"]["text"]}})
    return rendered, described


def _assemble(body: dict) -> tuple:
    task = task_of(body.get("task", ""))
    template = _prompt_ref(body.get("template"), "template")
    train_entries = body.get("entries") or []
    if not train_entries:
        raise ValueError("Thêm ít nhất một cặp policy + dataset SFT.")
    train_all, train_desc = _entries(task, template, train_entries, "train")
    eval_rows, eval_desc = _entries(task, template, body.get("eval_entries") or [], "eval")
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
    config = task.build_config(template["content"], train + val)
    lengths = [len(s.get("text", "")) for s in train + val]
    stats = {**merge_stats, **split_stats, "val_from": "eval" if eval_rows else "random",
             "max_chars": max(lengths), "avg_chars": round(sum(lengths) / len(lengths))}
    manifest = {"template": {"id": template["id"], "name": template["name"], "version": template["version"],
                             "content": template["content"]},
                "entries": train_desc, "eval_entries": eval_desc}
    return task, manifest, config, train, val, stats, val_percent, seed


def preview(body: dict) -> dict:
    _, _, config, train, _, stats, _, _ = _assemble(body)
    return {**stats, "config": config, "examples": train[:PREVIEW_PROMPTS]}


def next_version(name: str) -> int:
    versions = [b["version"] for b in BUILDS.all() if b["name"].lower() == name.lower()]
    return max(versions, default=0) + 1


def create(body: dict) -> dict:
    task, manifest, config, train, val, stats, val_percent, seed = _assemble(body)
    name = str(body.get("name") or "").strip()[:60]
    if not name:
        raise ValueError("Đặt tên cho build (vd. \"moderation\"). Cùng tên sẽ thành phiên bản mới.")
    version = next_version(name)
    meta = BUILDS.create({
        "id": make_id(f"{name} v{version}"), "name": name, "version": version, "task": task.id,
        "created": time.time(), "manifest": manifest, "task_config": config, "dedupe": bool(body.get("dedupe")),
        "val_percent": val_percent, "split_seed": seed, "note": str(body.get("note") or "").strip()[:300],
        "stats": stats, "has_eval": bool(manifest["eval_entries"]),
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
    return {"build": BUILDS.get(build_id), "examples": train[:PREVIEW_PROMPTS]}


def delete(build_id: str) -> str:
    m = BUILDS.get(build_id)
    BUILDS.delete(build_id)
    return f"Đã xoá build \"{title(m)}\". Các job đã train từ nó vẫn giữ nguyên dữ liệu."
