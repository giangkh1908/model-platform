"""Tham số train dạng YAML: xuất từ form (kèm chú thích từng tham số) và đọc ngược lại.

Kiểm tra giá trị dùng chung FinetuneTask.clean_params với form, nên YAML và form luôn ra cùng kết quả.
Khoá thiếu trong YAML thì lấy giá trị mặc định; khoá lạ thì báo lỗi (thường là gõ sai tên).
"""

from __future__ import annotations

import yaml

from .base import FinetuneTask

COMMENT_COLUMN = 28


def _line(key: str, value, comment: str) -> str:
    rendered = yaml.safe_dump({key: value}, allow_unicode=True, default_flow_style=False).strip()
    return f"{rendered.ljust(COMMENT_COLUMN)} # {comment}" if comment else rendered


def dump(task: FinetuneTask, params: dict) -> str:
    """params (đã hoặc chưa kiểm tra) -> YAML có chú thích, tham số cơ bản trước, nâng cao sau."""
    values = task.clean_params(params)
    lines = [f"# Tham số train · {task.name}", "# Xoá một dòng = dùng giá trị mặc định.", ""]
    specs = task.all_params()
    for title, group in (("", [p for p in specs if not p.advanced]), ("# Nâng cao", [p for p in specs if p.advanced])):
        if title:
            lines += ["", title]
        for p in group:
            comment = p.label + (f". {p.hint}" if p.hint else "")
            lines.append(_line(p.key, values[p.key], comment))
    return "\n".join(lines) + "\n"


def parse(task: FinetuneTask, text: str) -> dict:
    """YAML -> params đã kiểm tra. Sai cú pháp, sai kiểu hay khoá lạ thì ValueError."""
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" (dòng {mark.line + 1})" if mark else ""
        raise ValueError(f"YAML không hợp lệ{where}.") from None
    if not isinstance(data, dict):
        raise ValueError("YAML phải là các dòng dạng `tên: giá trị`.")
    known = {p.key for p in task.all_params()}
    unknown = sorted(str(k) for k in data if k not in known)
    if unknown:
        raise ValueError(f"Tham số không có: {', '.join(unknown)}. Các tham số hợp lệ: {', '.join(sorted(known))}.")
    return task.clean_params(data)
