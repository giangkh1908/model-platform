"""Dataset: dữ liệu gốc đã chọn cột, kèm loại nguồn (SFT để train, Eval để chấm, xem sources.py).
Build ghép dataset với template + policy; dataset không dùng trực tiếp để train hay eval.
data/finetune/datasets/<id>/: meta.json, rows.jsonl (mẫu đã chuẩn hoá theo FinetuneTask.fields)."""

from __future__ import annotations

import time
from collections import Counter

from . import data
from .sources import source_of
from .store import MetaStore, clean_name, load_upload, make_id, read_jsonl, write_jsonl
from .tasks import task_of

DATASETS = MetaStore("datasets", "dataset")
PREVIEW_ROWS = 20


def label_counts(task, samples: list[dict]) -> list[dict]:
    c = Counter(task.strata_key(s) for s in samples)
    if None in c:
        return []
    return [{"key": k, "count": n} for k, n in c.most_common()]


def create(body: dict) -> dict:
    task = task_of(body.get("task", ""))
    source = source_of(body.get("source", ""))
    up = load_upload(body.get("upload_id", ""))
    samples, errors = data.to_samples(task, up["rows"], body.get("mapping") or {})
    if not samples:
        first = errors[0]
        raise ValueError(f"Không có dòng hợp lệ nào. Lỗi đầu tiên: dòng {first['row']}: {first['reason']}.")
    meta = DATASETS.create({
        "id": make_id(body.get("name") or up["name"]), "name": clean_name(body.get("name"), up["name"]),
        "task": task.id, "source": source.id, "created": time.time(), "file": up["name"], "format": up["format"],
        "mapping": body.get("mapping"), "rows": len(samples), "invalid": len(errors),
        "errors": errors[:data.MAX_SHOWN_ERRORS], "labels": label_counts(task, samples),
    })
    write_jsonl(DATASETS.dir(meta["id"]) / "rows.jsonl", samples)
    return meta


def samples(dataset_id: str) -> list[dict]:
    DATASETS.get(dataset_id)
    return read_jsonl(DATASETS.dir(dataset_id) / "rows.jsonl")


def detail(dataset_id: str) -> dict:
    return {"dataset": DATASETS.get(dataset_id), "preview": samples(dataset_id)[:PREVIEW_ROWS]}


def delete(dataset_id: str) -> str:
    m = DATASETS.get(dataset_id)
    DATASETS.delete(dataset_id)
    return f"Đã xoá dataset \"{m['name']}\". Các build đã tạo từ nó vẫn giữ nguyên dữ liệu."
