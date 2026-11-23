"""Prompt: template (khung prompt có {policy}, {input} + phần output) và policy (đoạn luật).
({history} vẫn được nhận nếu sau này dataset có lịch sử hội thoại; hiện để trống.)

Mỗi lần sửa là một phiên bản mới, phiên bản cũ giữ nguyên, nên build cũ luôn ghép lại ra đúng prompt cũ.
Lưu trữ đi qua interface PromptStore: bây giờ là LocalPromptStore (data/finetune/prompts/), sau này thêm một lớp
lấy từ Langfuse theo version mà không phải sửa build hay giao diện.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from abc import ABC, abstractmethod

from ..config import FINETUNE_DIR
from ..core import load_json, save_json
from .store import ID_PATTERN, make_id

KINDS = ("template", "policy")
PLACEHOLDERS = ("{policy}", "{history}", "{input}")
ROLE_NAMES = {"user": "user", "human": "user", "assistant": "AI", "ai": "AI", "bot": "AI", "system": "system"}


def check_content(kind: str, content: dict) -> dict:
    """Kiểm tra và chuẩn hoá nội dung một phiên bản. Sai thì ValueError."""
    if kind == "template":
        body = str(content.get("body") or "").strip()
        output = str(content.get("output") or "").strip()
        if "{input}" not in body:
            raise ValueError("Template phải có chỗ {input} để chèn câu của người dùng.")
        if not output:
            raise ValueError("Template cần phần output (model phải trả lời gì).")
        return {"body": body, "output": output}
    text = str(content.get("text") or "").strip()
    if not text:
        raise ValueError("Policy không được để trống.")
    return {"text": text}


def format_history(history) -> str:
    """Lịch sử hội thoại -> dòng "user: …" / "AI: …". Nhận chuỗi sẵn, hoặc danh sách {role, content}."""
    if not history:
        return ""
    if isinstance(history, str):
        stripped = history.strip()
        if stripped.startswith("["):
            try:
                history = json.loads(stripped)
            except ValueError:
                return stripped
        else:
            return stripped
    lines = []
    for turn in history:
        if isinstance(turn, dict):
            role = ROLE_NAMES.get(str(turn.get("role", "")).lower(), turn.get("role", "user"))
            lines.append(f"{role}: {turn.get('content', '')}")
        else:
            lines.append(str(turn))
    return "\n".join(lines)


def render_body(template: dict, policy: str, history, user_input: str) -> str:
    """Thay đúng ba chỗ trống, không dùng str.format (template có thể chứa dấu ngoặc nhọn khác)."""
    values = {"{policy}": policy, "{history}": format_history(history), "{input}": user_input}
    return re.sub(r"\{policy\}|\{history\}|\{input\}", lambda m: values[m.group(0)], template["body"])


class PromptStore(ABC):
    """Nơi lấy template và policy theo phiên bản."""

    @abstractmethod
    def list(self, kind: str) -> list[dict]:
        """Danh sách mục: {id, kind, name, latest, versions: [{version, created, note}]}."""

    @abstractmethod
    def get(self, kind: str, item_id: str, version: int | None = None) -> dict:
        """Một phiên bản (mới nhất nếu không chỉ định): {id, kind, name, version, content}."""

    @abstractmethod
    def save(self, kind: str, item_id: str | None, name: str, content: dict, note: str) -> dict:
        """Tạo mục mới (item_id None) hoặc thêm phiên bản mới cho mục có sẵn. Trả về phiên bản vừa lưu."""

    @abstractmethod
    def delete(self, kind: str, item_id: str) -> None: ...


class LocalPromptStore(PromptStore):
    """data/finetune/prompts/<kind>/<id>/meta.json + v<N>.json"""

    def __init__(self) -> None:
        self.root = FINETUNE_DIR / "prompts"

    def _dir(self, kind: str, item_id: str):
        if kind not in KINDS:
            raise ValueError("Loại prompt không hợp lệ.")
        if not ID_PATTERN.fullmatch(item_id or ""):
            raise ValueError("Mã prompt không hợp lệ.")
        return self.root / kind / item_id

    def _meta(self, kind: str, item_id: str) -> dict:
        m = load_json(self._dir(kind, item_id) / "meta.json", None)
        if m is None:
            raise ValueError("Không có prompt này.")
        return m

    def list(self, kind: str) -> list[dict]:
        folder = self.root / kind
        if kind not in KINDS or not folder.is_dir():
            return []
        metas = [load_json(d / "meta.json", None) for d in folder.iterdir() if d.is_dir()]
        return sorted((m for m in metas if m), key=lambda m: m["name"].lower())

    def get(self, kind: str, item_id: str, version: int | None = None) -> dict:
        m = self._meta(kind, item_id)
        v = int(version or m["latest"])
        data = load_json(self._dir(kind, item_id) / f"v{v}.json", None)
        if data is None:
            raise ValueError(f"Không có phiên bản v{v} của \"{m['name']}\".")
        return {"id": item_id, "kind": kind, "name": m["name"], "version": v, "content": data["content"]}

    def save(self, kind: str, item_id: str | None, name: str, content: dict, note: str) -> dict:
        content = check_content(kind, content)
        name = str(name or "").strip()[:80]
        if item_id:
            m = self._meta(kind, item_id)
            if self.get(kind, item_id)["content"] == content:
                raise ValueError("Nội dung không đổi so với phiên bản mới nhất.")
            m["name"] = name or m["name"]
        else:
            if not name:
                raise ValueError("Đặt tên cho prompt.")
            item_id = make_id(name)
            m = {"id": item_id, "kind": kind, "name": name, "latest": 0, "versions": []}
            self._dir(kind, item_id).mkdir(parents=True)
        version = m["latest"] + 1
        save_json(self._dir(kind, item_id) / f"v{version}.json", {"content": content})
        m["latest"] = version
        m["versions"].append({"version": version, "created": time.time(), "note": str(note or "").strip()[:200]})
        save_json(self._dir(kind, item_id) / "meta.json", m)
        return self.get(kind, item_id, version)

    def delete(self, kind: str, item_id: str) -> None:
        self._meta(kind, item_id)
        shutil.rmtree(self._dir(kind, item_id), ignore_errors=True)


PROMPTS: PromptStore = LocalPromptStore()


def ref_title(p: dict) -> str:
    return f"{p['name']} v{p['version']}"
