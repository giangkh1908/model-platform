"""Thử model đang chạy: chat, decider, embedding, rerank, sinh ảnh, giọng nói."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request

import httpx
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import MODEL_URL
from ..core import http_json
from ..runner import running_model

router = APIRouter()


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


@router.post("/api/chat")
def chat(body: ChatBody):
    t = time.perf_counter()
    res = http_json(MODEL_URL + "/v1/chat/completions", _chat_payload(body, False), timeout=600)
    ms = (time.perf_counter() - t) * 1000
    toks = (res.get("usage") or {}).get("completion_tokens") or 0
    res["_ms"] = round(ms)
    res["_tps"] = round(toks / (ms / 1000), 1) if ms else None
    return res


@router.post("/api/chat/stream")
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


@router.post("/api/ask")
def ask(body: dict):
    """Strands Decider: /v1/systemone."""
    t = time.perf_counter()
    res = http_json(MODEL_URL + "/v1/systemone", body, timeout=120)
    res["_ms"] = round((time.perf_counter() - t) * 1000)
    return res


class EmbedBody(BaseModel):
    input: list[str]


@router.post("/api/embed")
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


@router.post("/api/rerank")
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


@router.post("/api/image")
def image(body: ImageBody):
    t = time.perf_counter()
    payload = {"model": running_model(), "prompt": body.prompt, "size": body.size, "n": 1}
    if body.steps is not None:
        payload["steps"] = body.steps
    if body.seed is not None:
        payload["seed"] = body.seed
    res = http_json(MODEL_URL + "/v1/images/generations", payload, timeout=600)
    return {"_ms": round((time.perf_counter() - t) * 1000), "image": res["data"][0]["b64_json"]}


@router.get("/api/tts/voices")
def tts_voices():
    return http_json(MODEL_URL + "/v1/voices", timeout=10)


class TTSBody(BaseModel):
    input: str = Field(min_length=1, max_length=1000)
    voice: str | None = None
    ref_audio: str | None = None  # WAV mã base64, dùng để clone giọng


@router.post("/api/tts")
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
        raise ValueError(message) from None
    return {"_ms": round((time.perf_counter() - t) * 1000), "audio": base64.b64encode(wav).decode("ascii")}
