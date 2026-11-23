"""Tìm/tải model từ Hugging Face, danh sách model, bật/tắt model và WSL."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import hf
from ..core import LOG, run
from ..downloads import delete_model, retry_download, start_download
from ..finetune.gpu_queue import QUEUE as FT_QUEUE
from ..monitor import MON
from ..registry import REG
from ..runner import SERVER, update_model

router = APIRouter()


# ---------------------------------------------------------------- Hugging Face / tải model


class UrlBody(BaseModel):
    url: str


class DownloadBody(BaseModel):
    url: str
    gguf: str | None = None
    runtime: str | None = None
    mode: str | None = None


@router.post("/api/inspect")
def post_inspect(body: UrlBody):
    return hf.inspect(body.url)


@router.get("/api/hf/search")
def hf_search(q: str = "", kind: str = "", limit: int = 20):
    if len(q.strip()) < 2:
        return {"results": []}
    return {"results": hf.search(q.strip(), kind, max(1, min(50, limit)))}


@router.post("/api/models")
def create_model(body: DownloadBody):
    info = hf.inspect(body.url)
    m = start_download(info, body.gguf, body.runtime, body.mode)
    return {"message": f"Đang tải {m['id']}…", "id": m["id"]}


# ---------------------------------------------------------------- model


class ModelPatch(BaseModel):
    options: dict | None = None
    mode: str | None = None


class StartBody(BaseModel):
    options: dict = Field(default_factory=dict)
    mode: str | None = None


@router.get("/api/models")
def list_models():
    return {"models": REG.public()}


@router.patch("/api/models/{mid}")
def patch_model(mid: str, body: ModelPatch):
    m = update_model(mid, body.options, body.mode)
    note = " Đang chạy: tắt rồi bật lại để áp dụng." if SERVER.current == mid else ""
    return {"message": f"Đã lưu tuỳ chọn cho {mid}.{note}", "model": {k: v for k, v in m.items() if not k.startswith("_")}}


@router.post("/api/models/{mid}/start")
def start_model(mid: str, body: StartBody):
    if FT_QUEUE.busy():
        raise ValueError("GPU đang dùng cho fine-tune. Đợi hàng đợi chạy xong hoặc dừng job trong trang Finetune.")
    return {"message": SERVER.start(mid, body.options, body.mode)}


@router.post("/api/models/{mid}/retry")
def retry_model(mid: str):
    m = retry_download(mid)
    return {"message": f"Đang tải lại {m['id']}…"}


@router.delete("/api/models/{mid}")
def remove_model(mid: str):
    return {"message": delete_model(mid, SERVER.stop)}


@router.post("/api/stop")
def stop_model():
    return {"message": SERVER.stop()}


@router.post("/api/wsl/shutdown")
def wsl_shutdown():
    SERVER.stop()
    run(["wsl.exe", "--shutdown"], 30)
    MON.refresh_wsl()
    LOG.add("=== Đã tắt toàn bộ WSL ===")
    return {"message": "Đã tắt WSL, RAM đã trả lại cho Windows."}
