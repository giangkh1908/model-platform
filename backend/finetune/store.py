"""Lưu trữ dùng chung cho dataset, build, job và eval: mỗi mục là một thư mục data/finetune/<loại>/<id>/ có meta.json.
Cùng chỗ với file tải lên tạm thời (chọn file xong mới chọn cột) và vài hàm tiện ích."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import unicodedata
import uuid
from pathlib import Path

from ..config import FINETUNE_DIR
from ..core import load_json, save_json
from . import data

UPLOADS_DIR = FINETUNE_DIR / "uploads"
UPLOAD_TTL_SECONDS = 24 * 3600
ID_PATTERN = re.compile(r"[a-z0-9-]{1,80}")


def ascii_fold(text: str) -> str:
    """Bỏ dấu tiếng Việt để làm mã: "Giao diện đẹp" -> "Giao dien dep"."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def make_id(name: str) -> str:
    """Mã thư mục: thời điểm tạo + tên không dấu, vd. 261123-005643-intent-v2."""
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_fold(name).lower()).strip("-")[:40] or "muc"
    return f"{time.strftime('%y%m%d-%H%M%S')}-{slug}"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_name(value, fallback: str) -> str:
    return str(value or "").strip()[:80] or fallback


class MetaStore:
    """Các mục cùng loại trong một thư mục; meta.json ghi dưới khoá để nhiều luồng không ghi đè nhau."""

    def __init__(self, kind: str, label: str) -> None:
        self.root = FINETUNE_DIR / kind
        self.label = label  # tên hiển thị trong thông báo lỗi: "dataset", "build"…
        self.lock = threading.RLock()

    def dir(self, item_id: str) -> Path:
        if not ID_PATTERN.fullmatch(item_id or ""):
            raise ValueError(f"Mã {self.label} không hợp lệ.")
        return self.root / item_id

    def get(self, item_id: str) -> dict:
        m = load_json(self.dir(item_id) / "meta.json", None)
        if m is None:
            raise ValueError(f"Không có {self.label} này.")
        return m

    def create(self, meta: dict) -> dict:
        folder = self.dir(meta["id"])
        folder.mkdir(parents=True)
        self.save(meta)
        return meta

    def save(self, meta: dict) -> None:
        with self.lock:
            save_json(self.dir(meta["id"]) / "meta.json", meta)

    def update(self, item_id: str, **changes) -> dict:
        with self.lock:
            m = self.get(item_id)
            m.update(changes)
            self.save(m)
            return m

    def all(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        metas = [load_json(d / "meta.json", None) for d in self.root.iterdir() if d.is_dir()]
        return sorted((m for m in metas if m), key=lambda m: m["created"], reverse=True)

    def delete(self, item_id: str) -> None:
        shutil.rmtree(self.dir(item_id), ignore_errors=True)


# ---------------------------------------------------------------- file tải lên tạm thời


def save_upload(name: str, text: str) -> dict:
    fmt, rows = data.parse_table(name, text)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for old in UPLOADS_DIR.glob("*.json"):
        if now - old.stat().st_mtime > UPLOAD_TTL_SECONDS:
            old.unlink(missing_ok=True)
    uid = uuid.uuid4().hex
    save_json(UPLOADS_DIR / f"{uid}.json", {"name": name, "format": fmt, "rows": rows})
    return {"upload_id": uid, "name": name, **data.describe(rows, fmt)}


def load_upload(uid: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", uid or ""):
        raise ValueError("Mã file không hợp lệ.")
    up = load_json(UPLOADS_DIR / f"{uid}.json", None)
    if up is None:
        raise ValueError("File đã hết hạn hoặc không tồn tại, hãy chọn lại file.")
    return up
