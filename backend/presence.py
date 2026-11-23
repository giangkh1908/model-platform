"""Theo dõi các tab giao diện đang mở; hết tab thì nền tảng tự tắt.

Mỗi tab gửi nhịp "còn mở" định kỳ và gửi "đã rời" khi đóng hoặc tải lại. Tải lại (F5) tạo tab mới
trong vài giây, nên server chờ GRACE_SECONDS trước khi tắt để không tắt nhầm.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

GRACE_SECONDS = 5  # thời gian chờ sau khi tab cuối cùng rời đi (đủ để tải lại trang)
STALE_SECONDS = 90  # tab không còn nhịp trong khoảng này coi như đã đóng (trình duyệt bị tắt đột ngột)


class Presence:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seen: dict[str, float] = {}
        self.had_page = False
        self.empty_since: float | None = None

    def touch(self, client: str) -> None:
        with self.lock:
            self.seen[client] = time.monotonic()
            self.had_page = True
            self.empty_since = None

    def leave(self, client: str) -> None:
        with self.lock:
            self.seen.pop(client, None)
            if not self.seen:
                self.empty_since = time.monotonic()

    def should_quit(self) -> bool:
        """Đã từng có tab và không còn tab nào đủ lâu."""
        with self.lock:
            now = time.monotonic()
            self.seen = {c: t for c, t in self.seen.items() if now - t < STALE_SECONDS}
            if not self.had_page:
                return False
            if self.seen:
                self.empty_since = None
                return False
            if self.empty_since is None:
                self.empty_since = now
            return now - self.empty_since >= GRACE_SECONDS


PRESENCE = Presence()


def watch(on_empty: Callable[[], bool]) -> None:
    """`on_empty` trả về False khi chưa tắt được (vd. còn job fine-tune): tiếp tục chờ, hỏi lại sau."""
    while True:
        time.sleep(1)
        if PRESENCE.should_quit() and on_empty():
            return
