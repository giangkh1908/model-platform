"""Danh sách model (lưu ở data/models.json)."""

from __future__ import annotations

import re
import threading

from .config import DEFAULT_OPTIONS, HOME, REGISTRY_FILE
from .core import load_json, save_json


class Registry:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.models: list[dict] = load_json(REGISTRY_FILE, None)
        if self.models is None:
            # Model đã cài từ trước
            self.models = [{
                "id": "strands-decider-2b",
                "repo": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
                "runtime": "decider",
                "path": f"{HOME}/strands-decider/models/strands-decider-2B-qwen3.5-v1-2610",
                "file": None,
                "size": 4_800_000_000,
                "status": "ready",
                "options": dict(DEFAULT_OPTIONS["decider"]),
                "managed_path": False,
            }]
            self.save()
        for m in self.models:  # tải dở khi panel tắt
            if m.get("status") == "downloading":
                m["status"] = "error"
                m["error"] = "Tải bị gián đoạn khi nền tảng tắt. Bấm \"Tải lại\" để tiếp tục."

    def save(self) -> None:
        with self.lock:
            save_json(REGISTRY_FILE, self.public())

    def get(self, mid: str) -> dict | None:
        with self.lock:
            return next((m for m in self.models if m["id"] == mid), None)

    def require(self, mid: str) -> dict:
        m = self.get(mid)
        if not m:
            raise ValueError("Không có model này.")
        return m

    def public(self) -> list[dict]:
        with self.lock:
            return [{k: v for k, v in m.items() if not k.startswith("_")} for m in self.models]

    def unique_id(self, base: str) -> str:
        base = re.sub(r"[^a-z0-9._-]+", "-", base.lower()).strip("-")[:60] or "model"
        mid, i = base, 2
        while self.get(mid):
            mid, i = f"{base}-{i}", i + 1
        return mid


REG = Registry()


def mode_of(m: dict) -> str:
    return m.get("mode") or ("decide" if m["runtime"] == "decider" else "chat")
