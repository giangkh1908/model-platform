"""Loại nguồn của dataset: dữ liệu SFT (để train) hay Eval (để chấm). Build đọc vai trò của từng loại để biết
dataset nào được ghép vào phần train, dataset nào vào phần eval. Thêm loại mới: một lớp con + đăng ký vào SOURCES."""

from __future__ import annotations

from abc import ABC
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SourceType(ABC):
    id: str
    name: str
    description: str
    role: str  # "train" | "eval": phần nào của build nhận dataset loại này

    def public(self) -> dict:
        return asdict(self)


class SftSource(SourceType):
    def __init__(self) -> None:
        super().__init__("sft", "SFT (train)", "Dữ liệu để train: ghép vào phần train của build.", "train")


class EvalSource(SourceType):
    def __init__(self) -> None:
        super().__init__("eval", "Eval (chấm)", "Dữ liệu để chấm: ghép vào phần eval của build, dùng làm val khi train và để eval.", "eval")


SOURCES: dict[str, SourceType] = {s.id: s for s in (SftSource(), EvalSource())}


def source_of(source_id: str) -> SourceType:
    source = SOURCES.get(source_id)
    if not source:
        raise ValueError(f"Không có loại nguồn \"{source_id}\".")
    return source
