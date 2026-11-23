"""Prompt của một bộ quyết định: system, intent, gate.

  system  bối cảnh chung (vd. "Bạn là bộ phận CSKH…"). config.intent_question: câu hỏi intent (bỏ trống = câu mặc định
          của runtime).
  intent  mỗi intent một prompt: văn bản là policy của intent đó. Tên prompt là nhãn intent trong dataset.
  gate    mỗi gate một prompt (câu có/không, vd. lừa đảo, cần chuyển người): văn bản là policy của gate.
          config.question: câu hỏi riêng của gate (bỏ trống = câu mặc định). Tên prompt là nhãn gate trong dataset.

Ghép các prompt thành prompt của model là việc của runtime (docs/runtime-protocol.md); ở đây chỉ lưu nội dung.

Mỗi lần sửa là một phiên bản mới (v1, v2… tự đánh số), phiên bản cũ giữ nguyên, nên build cũ luôn ghép lại ra đúng nội
dung cũ. Mỗi phiên bản có thể mang nhãn dev / prod (mỗi nhãn chỉ nằm ở một phiên bản: gắn cho bản khác là dời sang).

Lưu trữ đi qua interface PromptStore:
  LangfusePromptStore  có key Langfuse trong .env: đọc / ghi thẳng trên Langfuse (nguồn duy nhất: sửa ở Langfuse hay ở
                       đây đều thấy ngay ở bên kia). Loại ghi ở tag (system / intent / gate); nhãn prod = "production"
                       của Langfuse, dev = "dev".
  LocalPromptStore     không có key: data/finetune/prompts/.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from ..config import FINETUNE_DIR
from ..core import LOG, load_json, save_json


@dataclass(frozen=True)
class PromptField:
    """Một trường nội dung. "text" là văn bản chính (prompt trên Langfuse), các trường khác nằm ở config."""

    key: str
    label: str
    hint: str = ""
    multiline: bool = False
    required: bool = False


@dataclass(frozen=True)
class PromptKind:
    id: str
    label: str
    help: str
    fields: tuple[PromptField, ...]

    def public(self) -> dict:
        return asdict(self)


# Khai báo duy nhất của các loại prompt: backend kiểm tra theo đây, giao diện vẽ form theo đây (/api/finetune/prompt-kinds).
PROMPT_KINDS = (
    PromptKind("system", "System",
               "Bối cảnh chung của bộ quyết định (vd. \"Bạn là bộ phận CSKH của…\"). Runtime đặt nó đầu state, trước policy "
               "và câu người dùng.",
               (PromptField("text", "System prompt", multiline=True, required=True),
                PromptField("intent_question", "Câu hỏi intent",
                            "Bỏ trống = câu mặc định của runtime (\"Người dùng đang cần gì?…\")."))),
    PromptKind("intent", "Intent",
               "Mỗi intent một prompt: viết policy của intent đó (cái gì thuộc, cái gì không). Tên prompt là nhãn intent "
               "trong dataset.",
               (PromptField("text", "Policy của intent", multiline=True, required=True),)),
    PromptKind("gate", "Gate",
               "Mỗi gate một câu có/không (vd. lừa đảo, cần chuyển người): viết policy của gate. Tên prompt là nhãn gate "
               "trong dataset.",
               (PromptField("text", "Policy của gate", multiline=True, required=True),
                PromptField("question", "Câu hỏi của gate",
                            "Bỏ trống = câu mặc định (\"Câu của người dùng có thuộc trường hợp gate … không?\")."))),
)
KIND_BY_ID = {k.id: k for k in PROMPT_KINDS}
KINDS = tuple(KIND_BY_ID)
LABELS = ("dev", "prod")
# Tên prompt cũng là nhãn trong dataset: chữ thường không dấu, số, "_" "-".
NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}")
OPTIONAL_FIELDS = {k.id: tuple(f.key for f in k.fields if f.key != "text") for k in PROMPT_KINDS}


def check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError("Loại prompt phải là system, intent hoặc gate.")
    return kind


def check_label(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"Nhãn phải là một trong: {', '.join(LABELS)}.")
    return label


def check_name(name: str) -> str:
    name = str(name or "").strip()
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError("Tên prompt (cũng là nhãn trong dataset) chỉ gồm chữ thường không dấu, số, \"_\", \"-\", "
                         "vd. thanh_toan.")
    return name


def check_content(kind: str, content: dict) -> dict:
    """Chuẩn hoá nội dung một phiên bản theo khai báo của loại. Thiếu trường bắt buộc thì ValueError."""
    out = {}
    for field in KIND_BY_ID[check_kind(kind)].fields:
        value = str(content.get(field.key) or "").strip()
        if field.required and not value:
            raise ValueError(f"{field.label} không được để trống.")
        out[field.key] = value
    return out


class PromptStore(ABC):
    """Nơi lấy prompt theo phiên bản hoặc nhãn. Mã (id) của một prompt là tên của nó."""

    source = "local"

    @abstractmethod
    def list(self, kind: str) -> list[dict]:
        """Danh sách: {id, kind, name, latest, labels: {dev: v|None, prod: v|None}, versions: [{version}]}."""

    @abstractmethod
    def get(self, kind: str, item_id: str, version: int | None = None, label: str | None = None) -> dict:
        """Một phiên bản (theo số, theo nhãn, mặc định mới nhất): {id, kind, name, version, content, labels, note}."""

    @abstractmethod
    def versions(self, kind: str, item_id: str) -> list[dict]:
        """Mọi phiên bản (cũ trước): {version, note, labels}."""

    @abstractmethod
    def save(self, kind: str, item_id: str | None, name: str, content: dict, note: str) -> dict:
        """Tạo prompt mới (item_id None) hoặc thêm phiên bản mới. Trả về phiên bản vừa lưu."""

    @abstractmethod
    def set_label(self, kind: str, item_id: str, version: int, label: str) -> None:
        """Gắn nhãn cho một phiên bản (nhãn rời khỏi phiên bản đang giữ nó)."""

    @abstractmethod
    def delete(self, kind: str, item_id: str) -> None: ...

    def status(self) -> dict:
        return {"source": self.source}


class LocalPromptStore(PromptStore):
    """data/finetune/prompts/<kind>/<name>/meta.json + v<N>.json"""

    def __init__(self) -> None:
        self.root = FINETUNE_DIR / "prompts"

    def _dir(self, kind: str, item_id: str):
        check_kind(kind)
        if not NAME_PATTERN.fullmatch(item_id or ""):
            raise ValueError("Mã prompt không hợp lệ.")
        return self.root / kind / item_id

    def _meta(self, kind: str, item_id: str) -> dict:
        m = load_json(self._dir(kind, item_id) / "meta.json", None)
        if m is None:
            raise ValueError(f"Không có {kind} \"{item_id}\".")
        m.setdefault("labels", {})
        return m

    def _labels_of(self, meta: dict, version: int) -> list[str]:
        return sorted(label for label, v in meta["labels"].items() if v == version)

    def _public(self, meta: dict) -> dict:
        return {"id": meta["id"], "kind": meta["kind"], "name": meta["name"], "latest": meta["latest"],
                "labels": {label: meta["labels"].get(label) for label in LABELS},
                "versions": [{"version": v["version"]} for v in meta["versions"]]}

    def list(self, kind: str) -> list[dict]:
        check_kind(kind)
        folder = self.root / kind
        if not folder.is_dir():
            return []
        items = []
        for d in folder.iterdir():
            m = load_json(d / "meta.json", None) if d.is_dir() else None
            if m:
                m.setdefault("labels", {})
                items.append(self._public(m))
        return sorted(items, key=lambda m: m["name"])

    def get(self, kind: str, item_id: str, version: int | None = None, label: str | None = None) -> dict:
        m = self._meta(kind, item_id)
        if label:
            version = m["labels"].get(check_label(label))
            if not version:
                raise ValueError(f"\"{item_id}\" chưa có phiên bản nào mang nhãn {label}.")
        v = int(version or m["latest"])
        data = load_json(self._dir(kind, item_id) / f"v{v}.json", None)
        if data is None:
            raise ValueError(f"Không có phiên bản v{v} của \"{item_id}\".")
        note = next((x.get("note", "") for x in m["versions"] if x["version"] == v), "")
        return {"id": item_id, "kind": kind, "name": item_id, "version": v, "content": data["content"],
                "labels": self._labels_of(m, v), "note": note}

    def versions(self, kind: str, item_id: str) -> list[dict]:
        m = self._meta(kind, item_id)
        return [{"version": x["version"], "note": x.get("note", ""), "labels": self._labels_of(m, x["version"])}
                for x in m["versions"]]

    def save(self, kind: str, item_id: str | None, name: str, content: dict, note: str) -> dict:
        content = check_content(kind, content)
        if item_id:
            m = self._meta(kind, item_id)
            if self.get(kind, item_id)["content"] == content:
                raise ValueError("Nội dung không đổi so với phiên bản mới nhất.")
        else:
            item_id = check_name(name)
            if any((self.root / k / item_id).is_dir() for k in KINDS):
                raise ValueError(f"Đã có prompt tên \"{item_id}\", đặt tên khác.")
            m = {"id": item_id, "kind": kind, "name": item_id, "latest": 0, "versions": [], "labels": {}}
            self._dir(kind, item_id).mkdir(parents=True)
        version = m["latest"] + 1
        save_json(self._dir(kind, item_id) / f"v{version}.json", {"content": content})
        m["latest"] = version
        m["versions"].append({"version": version, "created": time.time(), "note": str(note or "").strip()[:200]})
        save_json(self._dir(kind, item_id) / "meta.json", m)
        return self.get(kind, item_id, version)

    def set_label(self, kind: str, item_id: str, version: int, label: str) -> None:
        m = self._meta(kind, item_id)
        if not any(x["version"] == version for x in m["versions"]):
            raise ValueError(f"Không có phiên bản v{version}.")
        m["labels"][check_label(label)] = version
        save_json(self._dir(kind, item_id) / "meta.json", m)

    def delete(self, kind: str, item_id: str) -> None:
        self._meta(kind, item_id)
        shutil.rmtree(self._dir(kind, item_id), ignore_errors=True)


class LangfusePromptStore(PromptStore):
    """Prompt dạng text trên Langfuse: prompt = text, config = các trường tuỳ chọn, tag = loại.
    Prompt tạo thẳng trên Langfuse phải có tag system / intent / gate thì mới hiện ở đây."""

    source = "langfuse"
    LANGFUSE_LABEL = {"dev": "dev", "prod": "production"}

    def __init__(self, client, base_url: str) -> None:
        self.client = client
        self.api = client.api
        self.base_url = base_url
        # Nội dung một phiên bản không bao giờ đổi: giữ lại. Nhãn đổi được: chỉ giữ theo mốc last_updated_at của prompt.
        self._versions: dict[tuple[str, int], object] = {}
        self._labels: dict[str, tuple[object, dict]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _quote(name: str) -> str:
        # SDK không mã hoá tên trong đường dẫn: tên có "/" hay dấu cách sẽ ra 404 nếu không tự mã hoá.
        return urllib.parse.quote(name, safe="")

    @staticmethod
    def _kind_of(tags) -> str | None:
        return next((kind for kind in KINDS if kind in (tags or [])), None)

    def _fetch(self, name: str, version: int):
        key = (name, version)
        with self._lock:
            cached = self._versions.get(key)
        if cached is None:
            cached = self.api.prompts.get(self._quote(name), version=version)
            with self._lock:
                self._versions[key] = cached
        return cached

    def _metas(self) -> list:
        metas, page = [], 1
        while True:
            res = self.api.prompts.list(page=page, limit=100)
            metas.extend(m for m in res.data if getattr(m, "type", "text") == "text")
            if page >= (getattr(res.meta, "total_pages", 1) or 1):
                return metas
            page += 1

    def _meta(self, kind: str, name: str):
        check_kind(kind)
        res = self.api.prompts.list(name=name, limit=1)
        meta = next((m for m in res.data if m.name == name), None)
        if meta is None or self._kind_of(meta.tags) != kind:
            raise ValueError(f"Không có {kind} \"{name}\" trên Langfuse.")
        return meta

    def _label_map(self, meta) -> dict:
        """{dev: v|None, prod: v|None} của một prompt, tra theo nhãn, giữ đến khi prompt đổi."""
        with self._lock:
            cached = self._labels.get(meta.name)
        if cached and cached[0] == meta.last_updated_at:
            return cached[1]
        mapping = {}
        for label, lf_label in self.LANGFUSE_LABEL.items():
            mapping[label] = None
            if lf_label in (meta.labels or []):
                mapping[label] = self.api.prompts.get(self._quote(meta.name), label=lf_label).version
        with self._lock:
            self._labels[meta.name] = (meta.last_updated_at, mapping)
        return mapping

    def _public(self, kind: str, meta) -> dict:
        versions = sorted(meta.versions)
        return {"id": meta.name, "kind": kind, "name": meta.name, "latest": versions[-1],
                "labels": self._label_map(meta), "versions": [{"version": v} for v in versions]}

    def _content(self, kind: str, prompt) -> dict:
        config = prompt.config if isinstance(prompt.config, dict) else {}
        out = {"text": prompt.prompt}
        for field in OPTIONAL_FIELDS[kind]:
            out[field] = str(config.get(field) or "")
        return out

    def _labels_of(self, mapping: dict, version: int) -> list[str]:
        return sorted(label for label, v in mapping.items() if v == version)

    def list(self, kind: str) -> list[dict]:
        check_kind(kind)
        items = [self._public(kind, m) for m in self._metas() if self._kind_of(m.tags) == kind]
        return sorted(items, key=lambda m: m["name"])

    def get(self, kind: str, item_id: str, version: int | None = None, label: str | None = None) -> dict:
        meta = self._meta(kind, item_id)
        mapping = self._label_map(meta)
        if label:
            version = mapping.get(check_label(label))
            if not version:
                raise ValueError(f"\"{item_id}\" chưa có phiên bản nào mang nhãn {label}.")
        v = int(version or max(meta.versions))
        if v not in meta.versions:
            raise ValueError(f"Không có phiên bản v{v} của \"{item_id}\".")
        prompt = self._fetch(item_id, v)
        return {"id": item_id, "kind": kind, "name": item_id, "version": v, "content": self._content(kind, prompt),
                "labels": self._labels_of(mapping, v), "note": prompt.commit_message or ""}

    def versions(self, kind: str, item_id: str) -> list[dict]:
        meta = self._meta(kind, item_id)
        mapping = self._label_map(meta)
        return [{"version": v, "note": self._fetch(item_id, v).commit_message or "", "labels": self._labels_of(mapping, v)}
                for v in sorted(meta.versions)]

    def save(self, kind: str, item_id: str | None, name: str, content: dict, note: str) -> dict:
        content = check_content(kind, content)
        if item_id:
            self._meta(kind, item_id)
            if self.get(kind, item_id)["content"] == content:
                raise ValueError("Nội dung không đổi so với phiên bản mới nhất.")
            name = item_id
        else:
            name = check_name(name)
            if self.api.prompts.list(name=name, limit=1).data:
                raise ValueError(f"Đã có prompt tên \"{name}\" trên Langfuse, đặt tên khác.")
        config = {field: content[field] for field in OPTIONAL_FIELDS[kind] if content[field]}
        created = self.client.create_prompt(name=name, prompt=content["text"], config=config, tags=[kind], labels=[],
                                            commit_message=str(note or "").strip()[:200] or None)
        LOG.add(f"Langfuse: đã lưu {kind} \"{name}\" v{created.version}")
        return self.get(kind, name, created.version)

    def set_label(self, kind: str, item_id: str, version: int, label: str) -> None:
        meta = self._meta(kind, item_id)
        if version not in meta.versions:
            raise ValueError(f"Không có phiên bản v{version}.")
        self.api.prompt_version.update(self._quote(item_id), version, new_labels=[self.LANGFUSE_LABEL[check_label(label)]])

    def delete(self, kind: str, item_id: str) -> None:
        self._meta(kind, item_id)
        self.api.prompts.delete(self._quote(item_id))
        with self._lock:
            self._labels.pop(item_id, None)
            for key in [k for k in self._versions if k[0] == item_id]:
                del self._versions[key]

    def status(self) -> dict:
        return {"source": self.source, "url": self.base_url}


def make_prompt_store() -> PromptStore:
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    if not (public_key and secret_key):
        return LocalPromptStore()
    base_url = os.environ.get("LANGFUSE_BASE_URL", "").strip() or "https://cloud.langfuse.com"
    try:
        from langfuse import Langfuse

        # Chỉ dùng quản lý prompt: tắt tracing để không gửi trace nào.
        client = Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url, tracing_enabled=False)
    except Exception as e:  # thiếu thư viện, key sai định dạng…: vẫn chạy được với kho trên máy
        LOG.add(f"Langfuse: không khởi tạo được ({e}), dùng kho prompt trên máy.")
        return LocalPromptStore()
    return LangfusePromptStore(client, base_url)


PROMPTS: PromptStore = make_prompt_store()


def ref_title(p: dict) -> str:
    return f"{p['name']}@v{p['version']}"
