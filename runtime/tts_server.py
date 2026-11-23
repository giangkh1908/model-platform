"""Máy chủ giọng nói VieNeu-TTS cho Model Platform (chạy trong WSL bằng venv có gói vieneu).

API:
  GET  /health              200 khi model đã nạp xong, 503 khi đang nạp
  GET  /v1/voices           danh sách giọng dựng sẵn [{"id", "label"}]
  POST /v1/audio/speech     {"input", "voice"} hoặc {"input", "ref_audio"} (WAV base64, để clone) -> audio/wav

Chạy: python tts_server.py --name <id> --port 8000 [--device auto|cuda|cpu]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import soundfile as sf
from vieneu import Vieneu

TTS = None
NAME = ""
INFER_LOCK = threading.Lock()  # một luồng sinh tiếng nói tại một thời điểm
MAX_CHARS = 1000


def load(device: str) -> None:
    global TTS
    print(f"Đang nạp {NAME} (device={device})", flush=True)
    TTS = Vieneu(mode="v3turbo", device=device)
    print(f"Đã nạp xong {NAME}", flush=True)


def list_voices() -> list[dict]:
    return [{"id": name, "label": label} for label, name in TTS.list_preset_voices()]


def synthesize(body: dict) -> bytes:
    text = str(body.get("input", "")).strip()
    if not text:
        raise ValueError("Thiếu nội dung cần đọc.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Văn bản quá dài (tối đa {MAX_CHARS} ký tự).")
    ref = body.get("ref_audio")
    voice = body.get("voice") or None

    ref_path = None
    if ref:
        try:
            raw = base64.b64decode(ref, validate=True)
        except ValueError:
            raise ValueError("File giọng mẫu không đọc được (cần file WAV).")
        fd, ref_path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)

    started = time.time()
    try:
        with INFER_LOCK:
            if ref_path:
                audio = TTS.infer(text, ref_audio=ref_path)
            else:
                audio = TTS.infer(text, voice=voice)
    except Exception as e:  # lỗi từ thư viện (ví dụ file giọng mẫu hỏng) gửi lại cho người dùng
        raise ValueError(f"Không sinh được giọng nói: {e}")
    finally:
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)

    buf = io.BytesIO()
    sf.write(buf, audio, TTS.sample_rate, format="WAV")
    print(f"Đọc {len(text)} ký tự, {len(audio) / TTS.sample_rate:.1f} s âm thanh, mất {time.time() - started:.1f} s", flush=True)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            if TTS is None:
                self._json(503, {"status": "loading"})
            else:
                self._json(200, {"status": "ok"})
        elif self.path == "/v1/voices":
            if TTS is None:
                self._json(503, {"error": {"message": "Model chưa nạp xong."}})
            else:
                self._json(200, {"data": list_voices()})
        else:
            self._json(404, {"error": {"message": "Không có đường dẫn này."}})

    def do_POST(self) -> None:
        if self.path != "/v1/audio/speech":
            self._json(404, {"error": {"message": "Không có đường dẫn này."}})
            return
        if TTS is None:
            self._json(503, {"error": {"message": "Model chưa nạp xong."}})
            return
        try:
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            wav = synthesize(body)
        except ValueError as e:
            self._json(400, {"error": {"message": str(e)}})
            return
        self.send_response(200)
        self.send_header("content-type", "audio/wav")
        self.send_header("content-length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)

    def log_message(self, fmt: str, *args) -> None:
        return  # lỗi và tiến trình đã in ra stdout


def main() -> None:
    global NAME
    ap = argparse.ArgumentParser(description="Máy chủ VieNeu-TTS")
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    NAME = args.name

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Máy chủ giọng nói ở http://{args.host}:{args.port}", flush=True)
    threading.Thread(target=load, args=(args.device,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
