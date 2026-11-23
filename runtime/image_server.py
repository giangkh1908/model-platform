"""Máy chủ sinh ảnh cho Model Platform (chạy trong WSL bằng venv diffusers).

API kiểu OpenAI:
  GET  /health                    200 khi model đã nạp xong, 503 khi đang nạp
  GET  /v1/models                 tên model đang phục vụ
  POST /v1/images/generations     {"prompt", "n", "size", "steps", "guidance_scale", "seed"} -> {"data": [{"b64_json"}]}

Chạy: python image_server.py --model <thư mục model> --name <id> --port 8000
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from diffusers import AutoPipelineForText2Image

PIPE = None
NAME = ""
GEN_LOCK = threading.Lock()  # pipeline dùng chung một GPU, chỉ sinh từng ảnh một
MIN_SIDE, MAX_SIDE = 256, 1536


def is_turbo(name: str) -> bool:
    return "turbo" in name.lower() or "lightning" in name.lower()


def default_steps(name: str) -> int:
    return 4 if is_turbo(name) else 30


def default_guidance(name: str) -> float:
    return 0.0 if is_turbo(name) else 7.5


def parse_size(size: str) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in size.lower().split("x"))
    except ValueError:
        raise ValueError(f"Kích thước không hợp lệ: {size!r} (dạng 1024x1024)")
    for side in (w, h):
        if not MIN_SIDE <= side <= MAX_SIDE:
            raise ValueError(f"Mỗi cạnh phải từ {MIN_SIDE} đến {MAX_SIDE}.")
    # Pipeline yêu cầu cạnh chia hết cho 8
    return w - w % 8, h - h % 8


def load(model_dir: str) -> None:
    global PIPE
    print(f"Đang nạp {NAME} từ {model_dir}", flush=True)
    has_fp16 = bool(glob.glob(os.path.join(model_dir, "**", "*.fp16.safetensors"), recursive=True))
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_dir, torch_dtype=torch.float16, variant="fp16" if has_fp16 else None, local_files_only=True)
    pipe.to("cuda")
    pipe.vae.enable_slicing()  # giải mã ảnh lớn theo từng lát, đỡ tốn VRAM
    PIPE = pipe
    print(f"Đã nạp xong {NAME}", flush=True)


def generate(body: dict) -> dict:
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Thiếu prompt.")
    n = max(1, min(4, int(body.get("n", 1))))
    width, height = parse_size(str(body.get("size", "1024x1024")))
    steps = int(body.get("steps") or default_steps(NAME))
    guidance = float(body.get("guidance_scale", default_guidance(NAME)))
    seed = body.get("seed")
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    with GEN_LOCK:
        started = time.time()
        result = PIPE(prompt=prompt, num_images_per_prompt=n, width=width, height=height,
                      num_inference_steps=steps, guidance_scale=guidance, generator=generator)
    data = []
    for image in result.images:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})
    print(f"Sinh {n} ảnh {width}x{height}, {steps} bước, {time.time() - started:.1f} s", flush=True)
    return {"created": int(time.time()), "data": data}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            if PIPE is None:
                self._send(503, {"status": "loading"})
            else:
                self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": NAME, "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "Không có đường dẫn này."}})

    def do_POST(self) -> None:
        if self.path != "/v1/images/generations":
            self._send(404, {"error": {"message": "Không có đường dẫn này."}})
            return
        if PIPE is None:
            self._send(503, {"error": {"message": "Model chưa nạp xong."}})
            return
        try:
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, generate(body))
        except ValueError as e:
            self._send(400, {"error": {"message": str(e)}})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._send(500, {"error": {"message": "Hết VRAM khi sinh ảnh. Thử kích thước nhỏ hơn."}})

    def log_message(self, fmt: str, *args) -> None:
        return  # log truy cập không cần thiết; lỗi đã in ra stdout


def main() -> None:
    global NAME
    ap = argparse.ArgumentParser(description="Máy chủ sinh ảnh diffusers")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    NAME = args.name

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Máy chủ sinh ảnh ở http://{args.host}:{args.port}", flush=True)
    threading.Thread(target=load, args=(args.model,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
