"""Fine-tune Strands Decider với một bộ quyết định: system + các intent + các gate (prompt ở prompts.py).

Mỗi câu của người dùng có thể mang nhãn intent (một trong các intent của bộ) và nhãn gate (gate nào dính). Runtime
(runtime/finetune/decider.py) ghép: state = system + policy các intent + policy các gate + câu người dùng; câu hỏi
intent là `choice`, mỗi gate một câu `noul`; mỗi câu người dùng thành một mẫu intent + một mẫu cho mỗi gate.
"""

from __future__ import annotations

from ..config import DECIDER_PY
from .base import Field, FinetuneTask, Param
from .data import norm_text

MAX_INTENTS = 255  # giới hạn số lựa chọn của câu hỏi choice trong strands-decider
MAX_GATES = 32


def split_keys(value) -> list[str]:
    """Cột gates: "lua_dao, chuyen_nguoi" hoặc danh sách JSON -> ["chuyen_nguoi", "lua_dao"]."""
    if isinstance(value, list):
        items = value
    else:
        items = str(value or "").replace(";", ",").replace("|", ",").split(",")
    return sorted({str(x).strip() for x in items if str(x).strip()})


def spec_keys(spec: dict) -> tuple[list[str], list[str]]:
    return [i["key"] for i in spec.get("intents", [])], [g["key"] for g in spec.get("gates", [])]


class DeciderTask(FinetuneTask):
    id = "decider"
    name = "Strands Decider (intent + gate)"
    description = ("Dạy decider một bộ quyết định: system + các intent (chọn một) + các gate (có/không). "
                   "Mỗi câu trả lời kèm độ tin cậy, dùng ngưỡng để quyết định tự động hay chuyển người.")
    runtime_python = DECIDER_PY
    fields = (
        Field("input", "Câu của người dùng", "Nội dung model phải xét."),
        Field("intent", "Intent", "Tên intent đúng (trùng tên prompt intent, vd. thanh_toan). Ô trống = không gán.",
              required=False),
        Field("gates", "Gate dính", "Các gate đúng với câu này, cách nhau bằng dấu phẩy (vd. lua_dao, chuyen_nguoi); "
              "ô trống = không dính gate nào.", required=False),
        Field("history", "Lịch sử hội thoại", "Các lượt trước (chuỗi, hoặc JSON [{role, content}]).", required=False),
    )
    params = (
        Param("head_lr_ratio", "Tỉ lệ learning rate của head", 10.0, 0.1, 100, 0.5,
              "Head nhỏ, học nhanh hơn LoRA. Công thức gốc của decider dùng 10.", advanced=True),
        Param("label_smoothing", "Label smoothing", 0.0, 0, 0.5, 0.01,
              "Giảm độ tự tin thái quá; thử 0.05 nếu ECE cao.", advanced=True),
        Param("target_precision", "Precision mục tiêu", 0.9, 0.5, 1, 0.01,
              "Ngưỡng tự động của intent được chọn trên val sao cho phần tự động đúng ít nhất từng này."),
        Param("max_length", "Độ dài tối đa (token)", 4096, 256, 8192, 256,
              "State (system + policy + lịch sử + câu) + câu hỏi. Dài hơn sẽ báo lỗi.", advanced=True),
        Param("eval_batch_size", "Batch size khi chấm", 16, 1, 128, 1, advanced=True),
    )
    defaults = {"lr": 1e-4, "epochs": 2, "batch_size": 4, "grad_accum": 4}
    example_rows = (
        {"input": "Tôi chuyển khoản rồi mà đơn vẫn báo chưa thanh toán", "intent": "thanh_toan", "gates": ""},
        {"input": "Shipper không gọi cho tôi, tôi muốn gặp người thật", "intent": "van_chuyen", "gates": "chuyen_nguoi"},
        {"input": "Gửi mã OTP vào số này để nhận quà nhé", "intent": "khac", "gates": "lua_dao"},
    )

    def to_sample(self, row: dict, mapping: dict[str, str]) -> dict:
        if not mapping.get("intent") and not mapping.get("gates"):
            raise ValueError("chọn cột Intent hoặc cột Gate dính (ít nhất một)")
        text = str(row.get(mapping["input"]) or "").strip()
        if not text:
            raise ValueError("thiếu câu của người dùng")
        intent = str(row.get(mapping["intent"]) or "").strip() if mapping.get("intent") else ""
        history = row.get(mapping["history"]) or "" if mapping.get("history") else ""
        gates = split_keys(row.get(mapping["gates"])) if mapping.get("gates") else None
        if not intent and gates is None:
            raise ValueError("thiếu nhãn intent")
        return {"input": text, "history": history, "intent": intent or None, "gates": gates}

    def group_key(self, sample: dict) -> str:
        return norm_text(sample["input"])

    def strata_key(self, sample: dict) -> str | None:
        return sample.get("intent") or ""

    def base_models(self, models: list[dict]) -> list[dict]:
        return [m for m in models if m["runtime"] == "decider" and m.get("status") == "ready"]

    def make_rows(self, spec: dict, samples: list[dict]) -> tuple[list[dict], int]:
        intents, gates = spec_keys(spec)
        unknown_intents = sorted({s["intent"] for s in samples if s["intent"] and s["intent"] not in intents})
        if unknown_intents and intents:
            raise ValueError(f"Dataset có intent không nằm trong bộ đã chọn: {', '.join(unknown_intents[:10])}. "
                             "Thêm prompt intent cùng tên hoặc sửa nhãn trong dataset.")
        unknown_gates = sorted({g for s in samples for g in (s["gates"] or []) if g not in gates})
        if unknown_gates and gates:
            raise ValueError(f"Dataset có gate không nằm trong bộ đã chọn: {', '.join(unknown_gates[:10])}.")
        rows, dropped = [], 0
        for s in samples:
            intent = s["intent"] if s["intent"] in intents else None
            labels = {g: g in s["gates"] for g in gates} if s["gates"] is not None else {}
            if intent is None and not labels:
                dropped += 1
                continue
            row = {"input": s["input"], "history": s.get("history") or "", "intent": intent, "gates": labels}
            if "dataset" in s:
                row["dataset"] = s["dataset"]
            rows.append(row)
        return rows, dropped

    def build_config(self, spec: dict, rows: list[dict]) -> dict:
        intents, gates = spec_keys(spec)
        if not intents and not gates:
            raise ValueError("Chọn ít nhất một intent hoặc một gate.")
        if len(intents) == 1:
            raise ValueError("Câu hỏi intent cần ít nhất 2 intent để chọn (hoặc bỏ hẳn intent, chỉ train gate).")
        if len(intents) > MAX_INTENTS:
            raise ValueError(f"Decider chỉ nhận tối đa {MAX_INTENTS} intent.")
        if len(gates) > MAX_GATES:
            raise ValueError(f"Tối đa {MAX_GATES} gate.")
        if not rows:
            raise ValueError("Không còn dòng nào có nhãn khớp với các intent / gate đã chọn.")
        return spec

    def check_eval(self, samples: list[dict], config: dict, model_name: str) -> None:
        intents, gates = spec_keys(config)
        unknown = sorted({s["intent"] for s in samples if s.get("intent") and s["intent"] not in intents})
        unknown += sorted({g for s in samples for g in (s.get("gates") or {}) if g not in gates})
        if unknown:
            shown = ", ".join(unknown[:10]) + ("…" if len(unknown) > 10 else "")
            raise ValueError(f"Tập chấm có {len(unknown)} intent / gate mà \"{model_name}\" không được train: {shown}.")

    def model_entry(self, job: dict) -> dict:
        return {"runtime": "decider", "mode": "decide"}
