"""Bật/tắt server model trong WSL (mỗi lúc một model, cổng 8000)."""

from __future__ import annotations

import subprocess
import threading
import time

from .config import (
    DECIDER_BIN,
    DEFAULT_OPTIONS,
    DIFFUSERS_PY,
    IMAGE_SERVER,
    LLAMA_DIR,
    MODEL_PORT,
    MODEL_URL,
    RUNTIME_BINS,
    RUNTIME_NAMES,
    SERVER_PATTERNS,
    TTS_SERVER,
    VIENEU_PY,
    VLLM_BIN,
)
from .core import LOG, health, http_json, popen_wsl, pump, q, wsl
from .registry import REG, mode_of

OPTION_LIMITS = {  # khoá: (kiểu, min, max)
    "max_concurrency": (int, 1, 40),
    "ctx": (int, 256, 262144),
    "gpu_util": (float, 0.3, 0.95),
    "parallel": (int, 1, 32),
}


def clean_options(runtime: str, opt: dict) -> dict:
    out = {}
    for k, v in (opt or {}).items():
        if k not in DEFAULT_OPTIONS[runtime] or v in ("", None):
            continue
        typ, lo, hi = OPTION_LIMITS[k]
        try:
            out[k] = max(lo, min(hi, typ(v)))
        except (TypeError, ValueError):
            raise ValueError(f"Giá trị không hợp lệ cho {k}: {v!r}") from None
    return out


def runtime_ready(runtime: str) -> tuple[bool, str]:
    code, _ = wsl(f"test -x {q(RUNTIME_BINS[runtime])}", 30)
    if code != 0:
        return False, f"Runtime {RUNTIME_NAMES[runtime]} chưa được cài."
    return True, ""


def runtimes_info() -> list[dict]:
    """Kiểm tra cả ba runtime trong một lần gọi WSL."""
    checks = " ; ".join(f"test -x {q(p)} && echo {k}=1 || echo {k}=0" for k, p in RUNTIME_BINS.items())
    code, out = wsl(checks, 30)
    found = dict(line.strip().split("=", 1) for line in out.splitlines() if "=" in line)
    return [{"id": k, "name": RUNTIME_NAMES[k], "path": p, "installed": found.get(k) == "1"}
            for k, p in RUNTIME_BINS.items()]


def update_model(mid: str, options: dict | None = None, mode: str | None = None) -> dict:
    m = REG.require(mid)
    if options is not None:
        m["options"] = {**DEFAULT_OPTIONS[m["runtime"]], **(m.get("options") or {}),
                        **clean_options(m["runtime"], options)}
    if mode in ("chat", "embed", "rerank") and m["runtime"] != "decider":
        m["mode"] = mode
    REG.save()
    return m


def launch_script(m: dict, opt: dict) -> str:
    path, rt = m["path"], m["runtime"]
    if rt == "decider":
        mc = int(opt.get("max_concurrency", 1))
        return (f"cd {q(path)} && exec env TRITON_DEFAULT_FP_FUSION=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 "
                f"{DECIDER_BIN} serve {q(path)} --device cuda --port {MODEL_PORT} --max-concurrency {mc}")
    if rt == "vllm":
        ctx = int(opt.get("ctx", 8192))
        gu = float(opt.get("gpu_util", 0.9))
        extra = {"embed": " --runner pooling --convert embed",
                 "rerank": " --runner pooling --convert classify"}.get(mode_of(m), "")
        target = path
        if m.get("file"):  # GGUF: vLLM cần tokenizer và config của model gốc
            target = f"{path}/{m['file']}"
            extra += f" --tokenizer {q(path + '/hf')} --hf-config-path {q(path + '/hf')}"
        return (f"exec env PYTHONUNBUFFERED=1 TRITON_DEFAULT_FP_FUSION=1 {VLLM_BIN} serve {q(target)} "
                f"--served-model-name {q(m['id'])} --host 127.0.0.1 --port {MODEL_PORT} "
                f"--gpu-memory-utilization {gu} --max-model-len {ctx}{extra}")
    if rt == "vieneu":
        return (f"exec env PYTHONUNBUFFERED=1 HF_HUB_CACHE={q(path + '/hf')} {VIENEU_PY} {TTS_SERVER} "
                f"--name {q(m['id'])} --port {MODEL_PORT} --device auto")
    if rt == "diffusers":
        return (f"exec env PYTHONUNBUFFERED=1 {DIFFUSERS_PY} {IMAGE_SERVER} --model {q(path)} "
                f"--name {q(m['id'])} --host 127.0.0.1 --port {MODEL_PORT}")
    ctx = int(opt.get("ctx", 8192))
    par = int(opt.get("parallel", 4))
    mode_flag = {"embed": "--embeddings", "rerank": "--reranking"}.get(mode_of(m), "--jinja")
    return (f"cd {LLAMA_DIR} && exec env LD_LIBRARY_PATH=. ./llama-server -m {q(path + '/' + m['file'])} "
            f"--alias {q(m['id'])} --host 127.0.0.1 --port {MODEL_PORT} -ngl 999 -c {ctx * par} -np {par} {mode_flag}")


class Server:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.current: str | None = None
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self.lock = threading.Lock()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, mid: str, opt: dict | None = None, mode: str | None = None) -> str:
        m = REG.require(mid)
        if m["status"] != "ready":
            raise ValueError("Model chưa sẵn sàng.")
        ok, why = runtime_ready(m["runtime"])
        if not ok:
            raise ValueError(why)
        with self.lock:
            if self.alive() or health() is not None:
                self._stop()
            update_model(mid, opt or {}, mode)
            LOG.add(f"=== Bật {mid} ({RUNTIME_NAMES[m['runtime']]}, {mode_of(m)}) với {m['options']} ===")
            self.proc = popen_wsl(launch_script(m, m["options"]))
            self.current, self.started_at, self.ready_at = mid, time.time(), None
            threading.Thread(target=self._watch, args=(self.proc, mid), daemon=True).start()
        return f"Đang bật {mid}…"

    def _watch(self, proc: subprocess.Popen, mid: str) -> None:
        pump(proc.stdout, mid)
        LOG.add(f"=== Tiến trình model đã thoát (mã {proc.wait()}) ===")

    def stop(self, only: str | None = None) -> str:
        """Tắt model; nếu có `only` thì chỉ tắt khi model đó đang chạy."""
        if only is not None and self.current != only:
            return ""
        with self.lock:
            return self._stop()

    def _stop(self) -> str:
        if not self.alive() and health() is None:
            self.current = self.started_at = self.ready_at = None
            return "Không có model nào đang chạy."
        LOG.add("=== Tắt model ===")
        pat = "|".join(SERVER_PATTERNS)
        wsl(f"pkill -INT -f {q(pat)}", 15)
        for _ in range(40):
            if not self.alive() and health() is None:
                break
            time.sleep(0.5)
        else:
            wsl(f"pkill -KILL -f {q(pat)}", 15)
            if self.alive():
                self.proc.kill()
        self.current = self.started_at = self.ready_at = None
        return "Đã tắt model, VRAM đã được giải phóng."


SERVER = Server()


def running_model() -> str | None:
    """Model nào đang chạy, kể cả khi được bật từ ngoài panel."""
    if SERVER.current:
        return SERVER.current
    try:
        data = http_json(MODEL_URL + "/v1/models", timeout=1)
        ids = [x.get("id") for x in data.get("data", [])]
        return next((i for i in ids if REG.get(i)), ids[0] if ids else None)
    except (OSError, ValueError):
        pass
    h = health() or {}
    name = h.get("model")
    for m in REG.public():
        if name and (m["repo"].endswith(name) or m["id"] == name):
            return m["id"]
    return name
