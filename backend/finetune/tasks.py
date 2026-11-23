"""Các loại fine-tune đã đăng ký. Thêm loại mới: viết lớp con của base.FinetuneTask rồi thêm vào TASKS."""

from __future__ import annotations

from .base import FinetuneTask
from .decider import DeciderTask

TASKS: dict[str, FinetuneTask] = {t.id: t for t in (DeciderTask(),)}


def task_of(task_id: str) -> FinetuneTask:
    task = TASKS.get(task_id)
    if not task:
        raise ValueError(f"Không có loại fine-tune \"{task_id}\".")
    return task
