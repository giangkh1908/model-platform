"""Ứng dụng FastAPI: REST API cho giao diện React và phục vụ bản build của frontend."""

from __future__ import annotations

import os
import subprocess
import threading
import urllib.error
import urllib.request
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import presence
from .api import finetune, models, playground, system
from .config import DEV_ORIGINS, FRONTEND_DIST, NO_WINDOW, PANEL_HOST, PANEL_PORT, ROOT_PID_ENV
from .core import LOG, health
from .finetune.gpu_queue import QUEUE as FT_QUEUE
from .monitor import MON
from .runner import SERVER


def quit_platform() -> bool:
    """Hết tab giao diện: tắt model rồi dừng cả cây tiến trình (Vite, và tiến trình giám sát khi --reload).
    uvicorn không dừng được từ luồng này nên dùng taskkill /T trên tiến trình gốc.
    Còn job fine-tune đang chạy hoặc chờ thì chưa tắt (trả về False), để train qua đêm không cần mở tab."""
    if FT_QUEUE.busy():
        return False
    LOG.add("Không còn tab giao diện nào mở, đang tắt nền tảng…")
    if SERVER.alive() or health() is not None:
        SERVER.stop()
    root = os.environ.get(ROOT_PID_ENV) or str(os.getpid())
    subprocess.run(["taskkill", "/F", "/T", "/PID", root], creationflags=NO_WINDOW, capture_output=True)
    return True


@asynccontextmanager
async def lifespan(_: FastAPI):
    MON.start()
    FT_QUEUE.start()
    threading.Thread(target=presence.watch, args=(quit_platform,), daemon=True).start()
    yield
    # Tắt bảng điều khiển thì tắt luôn model và job fine-tune, để không còn gì chạy ngầm chiếm VRAM.
    await run_in_threadpool(FT_QUEUE.shutdown)
    if SERVER.alive() or health() is not None:
        await run_in_threadpool(SERVER.stop)


app = FastAPI(title="Model Platform", version="2.0", lifespan=lifespan)

ALLOWED_ORIGINS = {f"http://{PANEL_HOST}:{PANEL_PORT}", f"http://localhost:{PANEL_PORT}", *DEV_ORIGINS}


@app.middleware("http")
async def origin_guard(request: Request, call_next):
    """Chặn trang web lạ gửi lệnh tới panel (chống CSRF từ trình duyệt)."""
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin not in ALLOWED_ORIGINS:
            return JSONResponse({"error": "forbidden"}, 403)
    return await call_next(request)


@app.exception_handler(ValueError)
async def _value_error(_: Request, e: ValueError):
    return JSONResponse({"error": str(e)}, 400)


@app.exception_handler(urllib.error.HTTPError)
async def _http_error(_: Request, e: urllib.error.HTTPError):
    return JSONResponse({"error": e.read().decode("utf-8", "replace")[:1000] or str(e)}, e.code)


@app.exception_handler(OSError)
async def _os_error(_: Request, e: OSError):
    return JSONResponse({"error": f"Không gọi được model: {e}"}, 502)


@app.exception_handler(httpx.HTTPError)
async def _httpx_error(_: Request, e: httpx.HTTPError):
    return JSONResponse({"error": f"Không gọi được model: {e}"}, 502)


# ---------------------------------------------------------------- API

for _router in (system.router, models.router, playground.router, finetune.router):
    app.include_router(_router)


# ---------------------------------------------------------------- frontend

if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/", response_class=PlainTextResponse)
    def no_frontend():
        return ("Chưa build giao diện. Chạy:\n  cd frontend\n  npm install\n  npm run build\n"
                "rồi mở lại trang này.")
