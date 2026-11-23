"""Đọc file dữ liệu (JSONL / JSON / CSV / TSV) và chia train/val ngẫu nhiên. Không biết gì về loại fine-tune cụ thể:
nhãn để thống kê lấy từ FinetuneTask.strata_key."""

from __future__ import annotations

import csv
import io
import json
import random
from collections import Counter

from .base import FinetuneTask

MAX_BYTES = 50 * 1024 * 1024
PREVIEW_ROWS = 8
MAX_SHOWN_ERRORS = 20


def norm_text(value) -> str:
    """Chuẩn hoá để so trùng: chữ thường, gộp khoảng trắng."""
    return " ".join(str(value).lower().split())


def detect_format(name: str, text: str) -> str:
    lower = name.lower()
    for ext, fmt in ((".jsonl", "jsonl"), (".ndjson", "jsonl"), (".json", "json"), (".tsv", "tsv"), (".csv", "csv")):
        if lower.endswith(ext):
            return fmt
    head = text.lstrip()[:1]
    if head == "[":
        return "json"
    if head == "{":
        return "jsonl"
    return "csv"


def parse_table(name: str, text: str) -> tuple[str, list[dict]]:
    """Trả về (định dạng, các dòng dạng dict). Lỗi định dạng thì ValueError kèm số dòng."""
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError(f"File lớn hơn {MAX_BYTES // 1024 // 1024} MB.")
    text = text.lstrip("﻿")
    fmt = detect_format(name, text)
    if fmt == "jsonl":
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise ValueError(f"Dòng {i} không phải JSON hợp lệ.") from None
            if not isinstance(row, dict):
                raise ValueError(f"Dòng {i} phải là một object JSON ({{...}}).")
            rows.append(row)
    elif fmt == "json":
        try:
            data = json.loads(text)
        except ValueError as e:
            raise ValueError(f"File JSON không hợp lệ: {e}") from None
        if isinstance(data, dict):  # {"data": [...]} hoặc {"rows": [...]}
            data = next((v for v in data.values() if isinstance(v, list)), None)
        if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
            raise ValueError("File JSON phải là một mảng các object ([{...}, {...}]).")
        rows = data
    else:
        delimiter = "\t" if fmt == "tsv" else _sniff_delimiter(text)
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("File CSV không có dòng tiêu đề.")
        rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in reader]
    if not rows:
        raise ValueError("File không có dòng dữ liệu nào.")
    return fmt, rows


def _sniff_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:4096], delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def columns_of(rows: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for r in rows[:1000]:
        for k in r:
            seen.setdefault(k, None)
    return list(seen)


def to_samples(task: FinetuneTask, rows: list[dict], mapping: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Áp mapping cột cho mọi dòng. Trả về (mẫu hợp lệ, lỗi theo dòng)."""
    for f in task.fields:
        if f.required and not mapping.get(f.key):
            raise ValueError(f"Chưa chọn cột cho \"{f.label}\".")
    samples, errors = [], []
    for i, row in enumerate(rows, 1):
        try:
            samples.append(task.to_sample(row, mapping))
        except ValueError as e:
            errors.append({"row": i, "reason": str(e)})
    return samples, errors


def tvd(part: Counter, whole: Counter) -> float:
    """Khoảng cách giữa hai phân phối (0 = giống hệt, 1 = khác hoàn toàn)."""
    n_part, n_whole = sum(part.values()), sum(whole.values())
    if not n_part or not n_whole:
        return 0.0
    return 0.5 * sum(abs(part[k] / n_part - whole[k] / n_whole) for k in whole)


def split(task: FinetuneTask, samples: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict], dict]:
    """Lấy ngẫu nhiên `val_fraction` số dòng làm val, phần còn lại là train (hai tập không chung dòng nào).
    Cùng seed thì ra cùng cách chia. Trả về (train, val, thống kê)."""
    if not 0 < val_fraction < 1:
        raise ValueError("Tỉ lệ val phải lớn hơn 0% và nhỏ hơn 100%.")
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    n_val = round(len(samples) * val_fraction)
    if n_val < 1 or n_val >= len(samples):
        raise ValueError("Dữ liệu quá ít để chia train/val.")
    val = [samples[i] for i in order[:n_val]]
    train = [samples[i] for i in order[n_val:]]
    return train, val, describe_split(task, train, val)


def describe_split(task: FinetuneTask, train: list[dict], val: list[dict]) -> dict:
    """Thống kê train/val theo nhãn (strata_key) để biết nhãn nào không có mặt ở val."""
    samples = train + val
    whole = Counter(task.strata_key(s) or "" for s in samples)
    c_train = Counter(task.strata_key(s) or "" for s in train)
    c_val = Counter(task.strata_key(s) or "" for s in val)
    has_strata = task.strata_key(samples[0]) is not None
    return {
        "total": len(samples),
        "train": len(train),
        "val": len(val),
        "val_fraction_actual": len(val) / len(samples),
        "strata": [{"key": k, "total": whole[k], "train": c_train[k], "val": c_val[k]}
                   for k in sorted(whole, key=lambda k: -whole[k])] if has_strata else [],
        "missing_in_val": sorted(k for k in whole if not c_val[k]) if has_strata else [],
        "tvd_val": tvd(c_val, whole),
    }


def merge(task: FinetuneTask, parts: list[list[dict]], dedupe: bool) -> tuple[list[dict], dict]:
    """Ghép nhiều danh sách mẫu. `dedupe`: bỏ dòng giống hệt (giữ bản đầu tiên).
    Luôn đếm câu trùng nhưng khác nhãn để người dùng kiểm tra, không tự xử lý."""
    rows = [s for part in parts for s in part]
    removed = 0
    if dedupe:
        seen, kept = set(), []
        for s in rows:
            key = json.dumps(s, ensure_ascii=False, sort_keys=True)
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            kept.append(s)
        rows = kept
    strata_by_key: dict[str, set] = {}
    for s in rows:
        strata_by_key.setdefault(task.group_key(s), set()).add(task.strata_key(s))
    conflicts = sum(len(v) > 1 for v in strata_by_key.values())
    return rows, {"before": sum(len(p) for p in parts), "removed_duplicates": removed, "conflicts": conflicts}


def describe(rows: list[dict], fmt: str) -> dict:
    return {"format": fmt, "rows": len(rows), "columns": columns_of(rows), "preview": rows[:PREVIEW_ROWS]}
