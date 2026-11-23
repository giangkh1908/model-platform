"""Hàng đợi GPU: mỗi lúc một việc (train hoặc eval), việc sau tự chạy khi việc trước xong.

Mỗi loại việc đăng ký một `Runner` (jobs.py, evals.py). Trạng thái hàng đợi nằm trong meta.json của từng mục
(queued / running / done / stopped / error / cancelled) nên còn nguyên sau khi tắt nền tảng.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import FINETUNE_RUN, wsl_path
from ..core import LOG, health, load_json, popen_wsl, pump, q, wsl
from ..runner import SERVER
from .store import MetaStore

KILL_GRACE_SECONDS = 5  # sau tín hiệu dừng (SIGINT) mà tiến trình chưa thoát thì SIGKILL


@dataclass
class Runner:
    store: MetaStore
    prepare: Callable[[dict], tuple[Path, str, str]]  # meta -> (thư mục chạy, tham số cho run.py, tên trong log)
    python: Callable[[dict], str]  # meta -> python trong WSL
    finish: Callable[[dict, str], None] = lambda meta, state: None  # dọn dẹp sau khi chạy xong (state cuối)
    on_done: Callable[[str], None] = lambda item_id: None  # sau khi đã ghi trạng thái "done" (vd. tự eval)


RUNNERS: dict[str, Runner] = {}


class Queue:
    def __init__(self) -> None:
        self.cv = threading.Condition(threading.RLock())
        self.items: deque[tuple[str, str]] = deque()  # (loại việc, mã)
        self.current: tuple[str, str] | None = None
        self.current_dir: Path | None = None
        self.stopping = False

    def start(self) -> None:
        pending = []
        for kind, r in RUNNERS.items():
            for m in r.store.all():
                if m["state"] == "running":  # nền tảng tắt giữa chừng
                    r.store.update(m["id"], state="error", error="Bị gián đoạn khi nền tảng tắt.")
                elif m["state"] == "queued":
                    pending.append((m["created"], kind, m["id"]))
        with self.cv:
            self.items.extend((kind, item_id) for _, kind, item_id in sorted(pending))
        threading.Thread(target=self._loop, daemon=True).start()

    def enqueue(self, kind: str, item_id: str) -> None:
        with self.cv:
            self.items.append((kind, item_id))
            self.cv.notify()

    def position(self, kind: str, item_id: str) -> int | None:
        with self.cv:
            for n, item in enumerate(self.items, 1):
                if item == (kind, item_id):
                    return n
        return None

    def busy(self) -> bool:
        with self.cv:
            return self.current is not None or bool(self.items)

    def summary(self) -> dict:
        with self.cv:
            return {"current": self.current, "waiting": list(self.items)}

    def stop(self, kind: str, item_id: str) -> str:
        """Đang chờ thì bỏ khỏi hàng đợi; đang chạy thì dừng ngay (không lưu gì thêm)."""
        r = RUNNERS[kind]
        with self.cv:
            if (kind, item_id) in self.items:
                self.items = deque(i for i in self.items if i != (kind, item_id))
                r.store.update(item_id, state="cancelled")
                return "Đã bỏ khỏi hàng đợi."
            if self.current != (kind, item_id):
                return "Không chạy."
            self.stopping = True
            target = wsl_path(self.current_dir)
        wsl(f"pkill -INT -f {q(target)}", 15)
        threading.Thread(target=self._force_kill, args=(target,), daemon=True).start()
        return "Đã dừng."

    def _force_kill(self, target: str) -> None:
        time.sleep(KILL_GRACE_SECONDS)
        wsl(f"pkill -KILL -f {q(target)}", 15)

    def shutdown(self) -> None:
        if self.current:
            wsl(f"pkill -INT -f {q(FINETUNE_RUN)}", 15)

    def _loop(self) -> None:
        while True:
            with self.cv:
                while not self.items:
                    self.cv.wait()
                self.current = self.items.popleft()
                self.stopping = False
            kind, item_id = self.current
            r = RUNNERS[kind]
            try:
                self._run(r, item_id)
            except Exception as e:  # một việc lỗi không được làm chết hàng đợi
                LOG.add(f"=== Fine-tune {item_id}: lỗi nền tảng: {e} ===")
                r.store.update(item_id, state="error", error=str(e))
            finally:
                with self.cv:
                    self.current = None
                    self.current_dir = None

    def _run(self, r: Runner, item_id: str) -> None:
        meta = r.store.get(item_id)
        run_dir, args, label = r.prepare(meta)
        with self.cv:
            self.current_dir = run_dir
        r.store.update(item_id, state="running", started=time.time(), error=None)
        if SERVER.alive() or health() is not None:
            LOG.add(f"=== Tắt model đang chạy để nhường GPU cho {label} ===")
            SERVER.stop()
        LOG.add(f"=== Bắt đầu {label} ===")
        # cd /tmp: python thêm thư mục hiện tại vào sys.path; HF_HUB_OFFLINE: model gốc đã có trong cache
        script = (f"cd /tmp && exec env HF_HUB_OFFLINE=1 TRITON_DEFAULT_FP_FUSION=1 PYTHONUNBUFFERED=1 "
                  f"{r.python(meta)} {q(FINETUNE_RUN)} {args}")
        proc = popen_wsl(script)
        pump(proc.stdout, label)
        code = proc.wait()
        status = load_json(run_dir / "status.json", {})
        error = None
        if self.stopping:
            state = "stopped"
        elif status.get("state") == "done":
            state = "done"
        else:
            state = "error"
            error = status.get("message") or f"Tiến trình thoát với mã {code}, xem trang Log."
        r.finish(meta, state)
        r.store.update(item_id, state=state, finished=time.time(), error=error)
        LOG.add(f"=== Kết thúc {label}: {state} ===")
        if state == "done":
            r.on_done(item_id)


QUEUE = Queue()
