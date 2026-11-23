"""Tiện ích: JSON, cài đặt, gọi WSL, log dùng chung."""

from __future__ import annotations

import collections
import json
import os
import shlex
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from .config import DISTRO, MODEL_URL, NO_WINDOW, SETTINGS_FILE


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


def settings() -> dict:
    return load_json(SETTINGS_FILE, {})


def save_settings(s: dict) -> None:
    save_json(SETTINGS_FILE, s)


def hf_token() -> str | None:
    """Token Hugging Face lấy từ biến môi trường HF_TOKEN (không lưu trong data/)."""
    return os.environ.get("HF_TOKEN", "").strip() or None


def wsl_env() -> dict:
    """Biến môi trường cho wsl.exe; HF_TOKEN được chuyển vào Linux qua WSLENV."""
    env = dict(os.environ)
    token = hf_token()
    if token:
        env["HF_TOKEN"] = token
        env["WSLENV"] = ":".join(filter(None, [env.get("WSLENV"), "HF_TOKEN/u"]))
    return env


def wsl_cmd(script: str) -> list[str]:
    # -e: chạy thẳng bash, không qua shell mặc định (shell đó sẽ đọc lại và làm hỏng $1, dấu nháy…)
    return ["wsl.exe", "-d", DISTRO, "-e", "bash", "-c", script]


def run(cmd: list[str], timeout: float = 30) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=NO_WINDOW,
                             env=wsl_env(), stdin=subprocess.DEVNULL)
        text = (out.stdout + out.stderr).decode("utf-8", "replace").replace("\x00", "")
        return out.returncode, text
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def wsl(script: str, timeout: float = 30) -> tuple[int, str]:
    return run(wsl_cmd(script), timeout)


def popen_wsl(script: str) -> subprocess.Popen:
    return subprocess.Popen(wsl_cmd(script), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=NO_WINDOW, env=wsl_env())


def q(s: str) -> str:
    return shlex.quote(s)


def wsl_running() -> bool:
    _, out = run(["wsl.exe", "-l", "--running", "-q"], 5)
    return DISTRO.lower() in out.lower()


def http_json(url: str, data=None, timeout: float = 10, headers: dict | None = None):
    h = {"content-type": "application/json", **(headers or {})}
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, body, h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def health() -> dict | None:
    """Server model đã sẵn sàng chưa (llama.cpp trả 503 khi đang nạp)."""
    try:
        with urllib.request.urlopen(MODEL_URL + "/health", timeout=1) as r:
            if r.status != 200:
                return None
            raw = r.read()
            try:
                return json.loads(raw) if raw.strip() else {}
            except ValueError:
                return {}
    except (OSError, ValueError):
        return None


class Log:
    """Bộ đệm log dạng vòng, mỗi dòng có số thứ tự để client lấy phần mới."""

    def __init__(self, maxlen: int = 5000) -> None:
        self.lines: collections.deque[tuple[int, float, str, str]] = collections.deque(maxlen=maxlen)
        self.seq = 0
        self.lock = threading.Lock()

    def add(self, line: str, source: str = "system") -> None:
        line = line.rstrip()
        if not line or "GET /health" in line:
            return
        with self.lock:
            self.seq += 1
            self.lines.append((self.seq, time.time(), source, line))

    def since(self, n: int, limit: int = 2000) -> list[dict]:
        with self.lock:
            rows = [r for r in self.lines if r[0] > n][-limit:]
        return [{"seq": s, "ts": t, "source": src, "text": txt} for s, t, src, txt in rows]

    def clear(self) -> None:
        with self.lock:
            self.lines.clear()


LOG = Log()


def pump(stream, source: str) -> None:
    """Đọc stdout của tiến trình (kể cả dòng kết thúc bằng \r) vào LOG."""
    buf = b""
    while True:
        chunk = stream.read1(4096)
        if not chunk:
            break
        buf += chunk
        *lines, buf = buf.replace(b"\r", b"\n").split(b"\n")
        for raw in lines:
            LOG.add(raw.decode("utf-8", "replace"), source)
    if buf.strip():
        LOG.add(buf.decode("utf-8", "replace"), source)
