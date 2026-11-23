"""Ứng dụng FastAPI: REST API cho giao diện React và phục vụ bản build của frontend."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import hf, presence
from .config import (DEFAULT_OPTIONS, DEV_ORIGINS, DISTRO, FRONTEND_DIST, GPU_VRAM_GB, HOME, MODEL_URL,
                     MODELS_DIR, NO_WINDOW, PANEL_HOST, PANEL_PORT, ROOT_PID_ENV, RUNTIME_NAMES)
from .core import LOG, health, hf_token, http_json, run
from .downloads import delete_model, retry_download, start_download
from .finetune import builds as ft_builds
from .finetune import datasets as ft_datasets
from .finetune import evals as ft_evals
from .finetune import jobs as ft_jobs
from .finetune.gpu_queue import QUEUE as FT_QUEUE
from .finetune.prompts import PROMPTS as FT_PROMPTS
from .finetune.sources import SOURCES as FT_SOURCES
from .finetune.store import save_upload
from .finetune.tasks import TASKS as FT_TASKS
from .finetune.tasks import task_of
from .monitor import MON
from .presence import PRESENCE
from .registry import REG, mode_of
from .runner import SERVER, OPTION_LIMITS, running_model, runtimes_info, update_model

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


# ---------------------------------------------------------------- trạng thái


@app.get("/api/status")
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


@app.get("/api/gpu/history")
def gpu_history():
    return {"samples": list(MON.history), "vram_gb": GPU_VRAM_GB}


@app.get("/api/config")
def get_config():
    return {
        "distro": DISTRO, "home": HOME, "models_dir": MODELS_DIR, "model_url": MODEL_URL,
        "panel": f"http://{PANEL_HOST}:{PANEL_PORT}", "vram_gb": GPU_VRAM_GB,
        "runtime_names": RUNTIME_NAMES, "default_options": DEFAULT_OPTIONS,
        "option_limits": {k: {"min": lo, "max": hi, "int": typ is int} for k, (typ, lo, hi) in OPTION_LIMITS.items()},
    }


@app.get("/api/runtimes")
def get_runtimes():
    return {"runtimes": runtimes_info()}


# ---------------------------------------------------------------- log


@app.get("/api/logs")
def get_logs(since: int = 0, limit: int = 2000):
    return {"lines": LOG.since(since, limit), "seq": LOG.seq}


@app.delete("/api/logs")
def clear_logs():
    LOG.clear()
    return {"message": "Đã xoá log."}


@app.get("/api/logs/download", response_class=PlainTextResponse)
def download_logs():
    text = "\n".join(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r['ts']))} [{r['source']}] {r['text']}"
                     for r in LOG.since(0, 100000))
    return PlainTextResponse(text, headers={"content-disposition": 'attachment; filename="model-platform.log"'})


# ---------------------------------------------------------------- Hugging Face / tải model


class UrlBody(BaseModel):
    url: str


class DownloadBody(BaseModel):
    url: str
    gguf: str | None = None
    runtime: str | None = None
    mode: str | None = None


@app.post("/api/inspect")
def post_inspect(body: UrlBody):
    return hf.inspect(body.url)


@app.get("/api/hf/search")
def hf_search(q: str = "", kind: str = "", limit: int = 20):
    if len(q.strip()) < 2:
        return {"results": []}
    return {"results": hf.search(q.strip(), kind, max(1, min(50, limit)))}


@app.post("/api/models")
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


@app.get("/api/models")
def list_models():
    return {"models": REG.public()}


@app.patch("/api/models/{mid}")
def patch_model(mid: str, body: ModelPatch):
    m = update_model(mid, body.options, body.mode)
    note = " Đang chạy: tắt rồi bật lại để áp dụng." if SERVER.current == mid else ""
    return {"message": f"Đã lưu tuỳ chọn cho {mid}.{note}", "model": {k: v for k, v in m.items() if not k.startswith("_")}}


@app.post("/api/models/{mid}/start")
def start_model(mid: str, body: StartBody):
    if FT_QUEUE.busy():
        raise ValueError("GPU đang dùng cho fine-tune. Đợi hàng đợi chạy xong hoặc dừng job trong trang Finetune.")
    return {"message": SERVER.start(mid, body.options, body.mode)}


@app.post("/api/models/{mid}/retry")
def retry_model(mid: str):
    m = retry_download(mid)
    return {"message": f"Đang tải lại {m['id']}…"}


@app.delete("/api/models/{mid}")
def remove_model(mid: str):
    return {"message": delete_model(mid, SERVER.stop)}


@app.post("/api/stop")
def stop_model():
    return {"message": SERVER.stop()}


@app.post("/api/wsl/shutdown")
def wsl_shutdown():
    SERVER.stop()
    run(["wsl.exe", "--shutdown"], 30)
    MON.refresh_wsl()
    LOG.add("=== Đã tắt toàn bộ WSL ===")
    return {"message": "Đã tắt WSL, RAM đã trả lại cho Windows."}


# ---------------------------------------------------------------- cài đặt


def _mask(tok: str | None) -> str | None:
    return f"{tok[:5]}…{tok[-4:]}" if tok and len(tok) > 12 else ("đã đặt" if tok else None)


@app.get("/api/settings")
def get_settings():
    tok = hf_token()
    user = None
    if tok:
        try:
            user = hf.whoami(tok).get("name")
        except OSError:  # gồm cả lỗi HTTP (token sai) và lỗi mạng
            user = None
    return {"has_token": bool(tok), "token_masked": _mask(tok), "hf_user": user}


# ---------------------------------------------------------------- thử model


class ChatBody(BaseModel):
    messages: list[dict]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float | None = None
    thinking: bool | None = None  # None: để mặc định của model


def _chat_payload(body: ChatBody, stream: bool) -> dict:
    p = {"model": running_model(), "messages": body.messages, "max_tokens": max(1, body.max_tokens),
         "temperature": body.temperature, "stream": stream}
    if body.top_p is not None:
        p["top_p"] = body.top_p
    if body.thinking is not None:
        # vLLM và llama.cpp (--jinja) đều chuyển biến này vào chat template (Qwen3, DeepSeek-R1, GLM…).
        p["chat_template_kwargs"] = {"enable_thinking": body.thinking}
    if stream:
        p["stream_options"] = {"include_usage": True}
    return p


@app.post("/api/chat")
def chat(body: ChatBody):
    t = time.perf_counter()
    res = http_json(MODEL_URL + "/v1/chat/completions", _chat_payload(body, False), timeout=600)
    ms = (time.perf_counter() - t) * 1000
    toks = (res.get("usage") or {}).get("completion_tokens") or 0
    res["_ms"] = round(ms)
    res["_tps"] = round(toks / (ms / 1000), 1) if ms else None
    return res


@app.post("/api/chat/stream")
async def chat_stream(body: ChatBody):
    """Chuyển tiếp luồng SSE từ vLLM / llama.cpp tới trình duyệt."""
    payload = await run_in_threadpool(_chat_payload, body, True)
    client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=5))
    req = client.build_request("POST", MODEL_URL + "/v1/chat/completions", json=payload)
    try:
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError:
        await client.aclose()
        raise
    if upstream.status_code != 200:
        text = (await upstream.aread()).decode("utf-8", "replace")[:1000]
        await upstream.aclose()
        await client.aclose()
        return JSONResponse({"error": text or f"Model trả lỗi {upstream.status_code}"}, upstream.status_code)

    async def gen():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        except httpx.HTTPError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"cache-control": "no-cache"})


@app.post("/api/ask")
def ask(body: dict):
    """Strands Decider: /v1/systemone."""
    t = time.perf_counter()
    res = http_json(MODEL_URL + "/v1/systemone", body, timeout=120)
    res["_ms"] = round((time.perf_counter() - t) * 1000)
    return res


class EmbedBody(BaseModel):
    input: list[str]


@app.post("/api/embed")
def embed(body: EmbedBody):
    t = time.perf_counter()
    res = http_json(MODEL_URL + "/v1/embeddings", {"model": running_model(), "input": body.input}, timeout=120)
    vecs = [d["embedding"] for d in sorted(res.get("data", []), key=lambda d: d.get("index", 0))]

    def cos(a, b):
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0

    return {"_ms": round((time.perf_counter() - t) * 1000), "dims": len(vecs[0]) if vecs else 0,
            "similarity": [[round(cos(a, b), 4) for b in vecs] for a in vecs],
            "preview": [[round(x, 4) for x in v[:8]] for v in vecs],
            "usage": res.get("usage")}


class RerankBody(BaseModel):
    query: str
    documents: list[str]


@app.post("/api/rerank")
def rerank(body: RerankBody):
    t = time.perf_counter()
    res = http_json(MODEL_URL + "/rerank", {"model": running_model(), "query": body.query,
                                            "documents": body.documents}, timeout=120)
    res["_ms"] = round((time.perf_counter() - t) * 1000)
    return res


class ImageBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    size: str = "1024x1024"
    steps: int | None = Field(default=None, ge=1, le=100)
    seed: int | None = None


@app.post("/api/image")
def image(body: ImageBody):
    t = time.perf_counter()
    payload = {"model": running_model(), "prompt": body.prompt, "size": body.size, "n": 1}
    if body.steps is not None:
        payload["steps"] = body.steps
    if body.seed is not None:
        payload["seed"] = body.seed
    res = http_json(MODEL_URL + "/v1/images/generations", payload, timeout=600)
    return {"_ms": round((time.perf_counter() - t) * 1000), "image": res["data"][0]["b64_json"]}


@app.get("/api/tts/voices")
def tts_voices():
    return http_json(MODEL_URL + "/v1/voices", timeout=10)


class TTSBody(BaseModel):
    input: str = Field(min_length=1, max_length=1000)
    voice: str | None = None
    ref_audio: str | None = None  # WAV mã base64, dùng để clone giọng


@app.post("/api/tts")
def tts(body: TTSBody):
    t = time.perf_counter()
    payload = {"input": body.input}
    if body.ref_audio:
        payload["ref_audio"] = body.ref_audio
    elif body.voice:
        payload["voice"] = body.voice
    req = urllib.request.Request(MODEL_URL + "/v1/audio/speech", json.dumps(payload).encode(),
                                 {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            wav = r.read()
    except urllib.error.HTTPError as e:  # máy chủ TTS báo lỗi có nội dung (ví dụ văn bản rỗng): hiện đúng thông báo
        try:
            message = json.loads(e.read())["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = None
        if message is None:
            raise
        raise ValueError(message)
    return {"_ms": round((time.perf_counter() - t) * 1000), "audio": base64.b64encode(wav).decode("ascii")}


# ---------------------------------------------------------------- fine-tune


class UploadBody(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    text: str


class DatasetBody(BaseModel):
    task: str
    source: str
    name: str = ""
    upload_id: str
    mapping: dict[str, str]


class BuildBody(BaseModel):
    task: str
    name: str = ""
    note: str = ""
    template: dict
    entries: list[dict]
    eval_entries: list[dict] = Field(default_factory=list)
    dedupe: bool = True
    val_percent: float = 10
    split_seed: int = 42


class JobBody(BaseModel):
    name: str = ""
    build_id: str
    base_model: str
    params: dict = Field(default_factory=dict)


class EvalBody(BaseModel):
    name: str = ""
    build_id: str
    base_model: str
    adapters: list[str] = Field(default_factory=list)


class PromptBody(BaseModel):
    name: str = ""
    note: str = ""
    content: dict


@app.get("/api/finetune/tasks")
def finetune_tasks():
    models = REG.public()
    return {"tasks": [t.public(models) for t in FT_TASKS.values()]}


@app.post("/api/finetune/uploads")
def finetune_upload(body: UploadBody):
    return save_upload(body.name, body.text)


@app.get("/api/finetune/queue")
def finetune_queue():
    return FT_QUEUE.summary()


@app.get("/api/finetune/sources")
def finetune_sources():
    return {"sources": [s.public() for s in FT_SOURCES.values()]}


@app.get("/api/finetune/prompts/{kind}")
def finetune_prompts(kind: str):
    return {"items": FT_PROMPTS.list(kind)}


@app.post("/api/finetune/prompts/{kind}")
def finetune_prompt_create(kind: str, body: PromptBody):
    p = FT_PROMPTS.save(kind, None, body.name, body.content, body.note)
    return {"message": f"Đã lưu \"{p['name']}\" v{p['version']}.", "id": p["id"], "version": p["version"]}


@app.get("/api/finetune/prompts/{kind}/{item_id}")
def finetune_prompt(kind: str, item_id: str, version: int | None = None):
    return FT_PROMPTS.get(kind, item_id, version)


@app.post("/api/finetune/prompts/{kind}/{item_id}")
def finetune_prompt_version(kind: str, item_id: str, body: PromptBody):
    p = FT_PROMPTS.save(kind, item_id, body.name, body.content, body.note)
    return {"message": f"Đã lưu phiên bản mới: \"{p['name']}\" v{p['version']}.", "id": p["id"], "version": p["version"]}


@app.delete("/api/finetune/prompts/{kind}/{item_id}")
def finetune_prompt_delete(kind: str, item_id: str):
    FT_PROMPTS.delete(kind, item_id)
    return {"message": "Đã xoá. Các build đã dùng nó vẫn giữ nguyên nội dung prompt."}


@app.get("/api/finetune/datasets")
def finetune_datasets():
    return {"datasets": ft_datasets.DATASETS.all()}


@app.post("/api/finetune/datasets")
def finetune_dataset_create(body: DatasetBody):
    m = ft_datasets.create(body.model_dump())
    return {"message": f"Đã lưu dataset \"{m['name']}\" ({m['rows']} dòng).", "id": m["id"]}


@app.get("/api/finetune/datasets/{item_id}")
def finetune_dataset(item_id: str):
    return ft_datasets.detail(item_id)


@app.delete("/api/finetune/datasets/{item_id}")
def finetune_dataset_delete(item_id: str):
    return {"message": ft_datasets.delete(item_id)}


@app.get("/api/finetune/builds")
def finetune_builds():
    return {"builds": ft_builds.BUILDS.all()}


@app.post("/api/finetune/builds/preview")
def finetune_build_preview(body: BuildBody):
    return ft_builds.preview(body.model_dump())


@app.post("/api/finetune/builds")
def finetune_build_create(body: BuildBody):
    m = ft_builds.create(body.model_dump())
    return {"message": f"Đã build \"{ft_builds.title(m)}\".", "id": m["id"]}


@app.get("/api/finetune/builds/{item_id}")
def finetune_build(item_id: str):
    return ft_builds.detail(item_id)


@app.delete("/api/finetune/builds/{item_id}")
def finetune_build_delete(item_id: str):
    return {"message": ft_builds.delete(item_id)}


@app.get("/api/finetune/jobs")
def finetune_jobs():
    return {"jobs": ft_jobs.list_all()}


@app.post("/api/finetune/jobs")
def finetune_job_create(body: JobBody):
    m = ft_jobs.create(body.model_dump())
    note = " Model đang chạy sẽ được tắt khi tới lượt." if health() is not None else ""
    return {"message": f"Đã thêm \"{m['name']}\" vào hàng đợi.{note}", "id": m["id"]}


@app.get("/api/finetune/jobs/{item_id}")
def finetune_job(item_id: str, since: int = 0):
    return ft_jobs.detail(item_id, max(0, since))


@app.post("/api/finetune/jobs/{item_id}/stop")
def finetune_job_stop(item_id: str):
    return {"message": ft_jobs.stop(item_id)}


@app.post("/api/finetune/jobs/{item_id}/register")
def finetune_job_register(item_id: str):
    res = ft_jobs.register_model(item_id)
    return {"message": f"Đã thêm {res['id']} vào danh sách model.", **res}


@app.delete("/api/finetune/jobs/{item_id}")
def finetune_job_delete(item_id: str):
    return {"message": ft_jobs.delete(item_id)}


@app.get("/api/finetune/evals")
def finetune_evals():
    return {"evals": ft_evals.list_all()}


@app.get("/api/finetune/evals/choices")
def finetune_eval_choices(task: str):
    return ft_evals.model_choices(task_of(task))


@app.post("/api/finetune/evals")
def finetune_eval_create(body: EvalBody):
    m = ft_evals.create(body.model_dump())
    return {"message": f"Đã thêm lần eval \"{m['name']}\" vào hàng đợi.", "id": m["id"]}


@app.get("/api/finetune/evals/{item_id}")
def finetune_eval(item_id: str):
    return ft_evals.detail(item_id)


@app.post("/api/finetune/evals/{item_id}/stop")
def finetune_eval_stop(item_id: str):
    return {"message": ft_evals.stop(item_id)}


@app.delete("/api/finetune/evals/{item_id}")
def finetune_eval_delete(item_id: str):
    return {"message": ft_evals.delete(item_id)}


@app.post("/api/presence")
def presence_ping(client: str = Query(min_length=1, max_length=64)):
    PRESENCE.touch(client)
    return {"ok": True}


@app.post("/api/presence/leave")
def presence_leave(client: str = Query(min_length=1, max_length=64)):
    PRESENCE.leave(client)
    return {"ok": True}


# ---------------------------------------------------------------- frontend

if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/", response_class=PlainTextResponse)
    def no_frontend():
        return ("Chưa build giao diện. Chạy:\n  cd frontend\n  npm install\n  npm run build\n"
                "rồi mở lại trang này.")
