"""Cách platform nói chuyện với runtime fine-tune (xem docs/runtime-protocol.md).

Hàng đợi GPU, job và eval chỉ gọi qua interface RuntimeConnector, không tự chạy lệnh WSL. Bây giờ runtime nằm ngay
trong project (runtime/finetune/run.py) và chạy trong WSL cùng máy: LocalWslConnector. Khi runtime tách sang repo
runtime-platform hay chạy ở máy khác, chỉ cần thêm một connector mới (vd. gọi HTTP), phần còn lại không đổi.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import DISTRO, FINETUNE_RUN, wsl_path
from ..core import popen_wsl, pump, q, wsl

PROTOCOL_VERSION = 1
KILL_GRACE_SECONDS = 5  # sau tín hiệu dừng (SIGINT) mà tiến trình chưa thoát thì SIGKILL


class RuntimeConnector(ABC):
    @abstractmethod
    def run(self, command: str, task_id: str, run_dir: Path, python: str, label: str) -> int:
        """Chạy `command` ("train" | "eval") cho thư mục `run_dir` đến khi xong; log đẩy vào LOG với nhãn `label`.
        Trả về mã thoát. Kết quả thật (done / error) đọc từ status.json trong run_dir."""

    @abstractmethod
    def stop(self, run_dir: Path) -> None:
        """Yêu cầu dừng lần chạy đang dùng `run_dir` (không chờ)."""

    @abstractmethod
    def stop_all(self) -> None:
        """Dừng mọi lần chạy (khi tắt nền tảng)."""

    @abstractmethod
    def remove_output(self, output_dir: str) -> None:
        """Xoá checkpoint ở `output_dir` (đường dẫn phía runtime)."""

    @abstractmethod
    def output_path(self, output_dir: str) -> Path:
        """Thư mục checkpoint đọc được từ phía platform (vd. để xuất bản lên registry)."""


class LocalWslConnector(RuntimeConnector):
    def run(self, command: str, task_id: str, run_dir: Path, python: str, label: str) -> int:
        # cd /tmp: python thêm thư mục hiện tại vào sys.path; HF_HUB_OFFLINE: model gốc đã có trong cache
        script = (f"cd /tmp && exec env HF_HUB_OFFLINE=1 TRITON_DEFAULT_FP_FUSION=1 PYTHONUNBUFFERED=1 "
                  f"{python} {q(FINETUNE_RUN)} {command} {task_id} {q(wsl_path(run_dir))}")
        proc = popen_wsl(script)
        pump(proc.stdout, label)
        return proc.wait()

    def stop(self, run_dir: Path) -> None:
        target = wsl_path(run_dir)
        wsl(f"pkill -INT -f {q(target)}", 15)
        threading.Thread(target=self._force_kill, args=(target,), daemon=True).start()

    def _force_kill(self, target: str) -> None:
        time.sleep(KILL_GRACE_SECONDS)
        wsl(f"pkill -KILL -f {q(target)}", 15)

    def stop_all(self) -> None:
        wsl(f"pkill -INT -f {q(FINETUNE_RUN)}", 15)

    def remove_output(self, output_dir: str) -> None:
        wsl(f"rm -rf {q(output_dir)}", 60)

    def output_path(self, output_dir: str) -> Path:
        # Windows đọc thẳng file trong WSL qua \\wsl.localhost\<distro>\...
        return Path(rf"\\wsl.localhost\{DISTRO}") / output_dir.lstrip("/")


CONNECTOR: RuntimeConnector = LocalWslConnector()
