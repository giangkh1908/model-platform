"""Trạng thái, cấu hình, log, cài đặt và tín hiệu còn tab giao diện."""

from __future__ import annotations

import time

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from .. import hf
from ..config import (
    DEFAULT_OPTIONS,
    DISTRO,
    GPU_VRAM_GB,
    HOME,
    MODEL_URL,
    MODELS_DIR,
    PANEL_HOST,
    PANEL_PORT,
    RUNTIME_NAMES,
)
from ..core import LOG, health, hf_token
from ..monitor import MON
from ..presence import PRESENCE
from ..registry import REG, mode_of
from ..runner import OPTION_LIMITS, SERVER, running_model, runtimes_info

router = APIRouter()


# ---------------------------------------------------------------- trạng thái


@router.get("/api/status")
def get_status():
    h = health()
    alive = SERVER.alive()
    state = "running" if h is not None else ("starting" if alive else "stopped")
    if state == "running" and SERVER.started_at and SERVER.ready_at is None:
        SERVER.ready_at = time.time()
        LOG.add(f"=== {SERVER.current} đã sẵn sàng sau {SERVER.ready_at - SERVER.started_at:.0f} giây ===")
    cur = running_model() if state != "stopped" else None
    m = REG.get(cur) if cur else None
    return {
        "state": state,
        "current": cur,
        "runtime": m["runtime"] if m else None,
        "mode": mode_of(m) if m else None,
        "managed": alive,
        "uptime": (time.time() - SERVER.started_at) if (SERVER.started_at and alive) else None,
        "load_seconds": (SERVER.ready_at - SERVER.started_at) if (SERVER.ready_at and SERVER.started_at) else None,
        "gpu": MON.gpu,
        "wsl": MON.wsl,
        "model_url": MODEL_URL,
        "models": REG.public(),
        "has_token": bool(hf_token()),
    }


@router.get("/api/gpu/history")
def gpu_history():
    return {"samples": list(MON.history), "vram_gb": GPU_VRAM_GB}


@router.get("/api/config")
def get_config():
    return {
        "distro": DISTRO, "home": HOME, "models_dir": MODELS_DIR, "model_url": MODEL_URL,
        "panel": f"http://{PANEL_HOST}:{PANEL_PORT}", "vram_gb": GPU_VRAM_GB,
        "runtime_names": RUNTIME_NAMES, "default_options": DEFAULT_OPTIONS,
        "option_limits": {k: {"min": lo, "max": hi, "int": typ is int} for k, (typ, lo, hi) in OPTION_LIMITS.items()},
    }


@router.get("/api/runtimes")
def get_runtimes():
    return {"runtimes": runtimes_info()}


# ---------------------------------------------------------------- log


@router.get("/api/logs")
def get_logs(since: int = 0, limit: int = 2000):
    return {"lines": LOG.since(since, limit), "seq": LOG.seq}


@router.delete("/api/logs")
def clear_logs():
    LOG.clear()
    return {"message": "Đã xoá log."}


@router.get("/api/logs/download", response_class=PlainTextResponse)
def download_logs():
    text = "\n".join(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r['ts']))} [{r['source']}] {r['text']}"
                     for r in LOG.since(0, 100000))
    return PlainTextResponse(text, headers={"content-disposition": 'attachment; filename="model-platform.log"'})


# ---------------------------------------------------------------- cài đặt


def _mask(tok: str | None) -> str | None:
    return f"{tok[:5]}…{tok[-4:]}" if tok and len(tok) > 12 else ("đã đặt" if tok else None)


@router.get("/api/settings")
def get_settings():
    tok = hf_token()
    user = None
    if tok:
        try:
            user = hf.whoami(tok).get("name")
        except OSError:  # gồm cả lỗi HTTP (token sai) và lỗi mạng
            user = None
    return {"has_token": bool(tok), "token_masked": _mask(tok), "hf_user": user}


@router.post("/api/presence")
def presence_ping(client: str = Query(min_length=1, max_length=64)):
    PRESENCE.touch(client)
    return {"ok": True}


@router.post("/api/presence/leave")
def presence_leave(client: str = Query(min_length=1, max_length=64)):
    PRESENCE.leave(client)
    return {"ok": True}
