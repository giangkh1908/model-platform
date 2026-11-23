"""Lấy mẫu GPU và trạng thái WSL ở nền, giữ lịch sử để vẽ biểu đồ."""

from __future__ import annotations

import collections
import threading
import time

from .core import run, wsl_running

HISTORY_SECONDS = 600
INTERVAL = 2.0


def read_gpu() -> dict | None:
    _, out = run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                  "power.draw,power.limit,temperature.gpu,fan.speed", "--format=csv,noheader,nounits"], 5)
    parts = [p.strip() for p in out.strip().splitlines()[0].split(",")] if out.strip() else []
    if len(parts) < 7:
        return None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None

    keys = ["util", "mem_used", "mem_total", "power", "power_limit", "temp", "fan"]
    return {"name": parts[0], **{k: num(v) for k, v in zip(keys, parts[1:])}}


class Monitor:
    def __init__(self) -> None:
        self.gpu: dict | None = None
        self.wsl = False
        self.history: collections.deque[dict] = collections.deque(maxlen=int(HISTORY_SECONDS / INTERVAL))
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        threading.Thread(target=self._gpu_loop, daemon=True).start()
        threading.Thread(target=self._wsl_loop, daemon=True).start()

    def _gpu_loop(self) -> None:
        while True:
            g = read_gpu()
            self.gpu = g
            if g:
                self.history.append({"t": time.time(), "util": g["util"], "mem": g["mem_used"],
                                     "power": g["power"], "temp": g["temp"]})
            time.sleep(INTERVAL)

    def _wsl_loop(self) -> None:
        while True:
            self.wsl = wsl_running()
            time.sleep(5)

    def refresh_wsl(self) -> None:
        self.wsl = wsl_running()


MON = Monitor()
