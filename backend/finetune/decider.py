"""Fine-tune Strands Decider: phân loại {text, label}, train tiếp LoRA + head của một model decider có sẵn.
Phía WSL: runtime/finetune/decider.py."""

from __future__ import annotations

from ..config import DECIDER_PY
from .base import Chart, Field, FinetuneTask, Param, Series
from .data import norm_text
from .prompts import render_body

MAX_LABELS = 255  # giới hạn số lựa chọn của câu hỏi choice trong strands-decider


class DeciderTask(FinetuneTask):
    id = "decider"
    name = "Strands Decider (phân loại)"
    description = ("Dạy decider áp policy vào câu của người dùng: prompt = template + policy + lịch sử + câu, "
                   "model chọn một nhãn kèm độ tin cậy, dùng ngưỡng để quyết định tự động hay chuyển người.")
    runtime_python = DECIDER_PY
    fields = (
        Field("text", "Câu / nội dung", "Chèn vào chỗ {input} của template."),
        Field("label", "Nhãn", "Đáp án đúng theo policy đi kèm dataset (vd. intent, hoặc 0 / 1)."),
    )
    params = (
        Param("head_lr_ratio", "Tỉ lệ learning rate của head", 10.0, 0.1, 100, 0.5,
              "Head nhỏ, học nhanh hơn LoRA. Công thức gốc của decider dùng 10.", advanced=True),
        Param("label_smoothing", "Label smoothing", 0.0, 0, 0.5, 0.01,
              "Giảm độ tự tin thái quá; thử 0.05 nếu ECE cao.", advanced=True),
        Param("target_precision", "Precision mục tiêu", 0.9, 0.5, 1, 0.01,
              "Ngưỡng tự động được chọn trên val sao cho phần tự động đúng ít nhất từng này."),
        Param("max_length", "Độ dài tối đa (token)", 4096, 256, 8192, 256,
              "Prompt đã ghép (template + policy + lịch sử + câu) + phần output. Dài hơn sẽ báo lỗi.", advanced=True),
        Param("eval_batch_size", "Batch size khi chấm", 16, 1, 128, 1, advanced=True),
    )
    defaults = {"lr": 1e-4, "epochs": 2, "batch_size": 4, "grad_accum": 4}
    charts = (
        Chart("eval_acc", "Độ chính xác trên val",
              (Series("eval", "acc", "Accuracy"), Series("eval", "macro_f1", "Macro-F1")), unit="%"),
        Chart("eval_ece", "Sai lệch độ tin cậy (ECE) trên val", (Series("eval", "ece", "ECE"),), unit="%",
              hint="Càng thấp càng tốt: độ tin cậy model báo khớp với tỉ lệ đúng thực tế."),
    )
    example_rows = (
        {"text": "Tôi chuyển khoản rồi mà đơn vẫn báo chưa thanh toán", "label": "thanh_toan"},
        {"text": "Shipper không gọi cho tôi", "label": "van_chuyen"},
        {"text": "App bị lỗi không đăng nhập được", "label": "ky_thuat"},
    )

    def to_sample(self, row: dict, mapping: dict[str, str]) -> dict:
        text = str(row.get(mapping["text"]) or "").strip()
        label = str(row.get(mapping["label"]) or "").strip()
        if not text:
            raise ValueError("thiếu nội dung câu")
        if not label:
            raise ValueError("thiếu nhãn")
        return {"text": text, "label": label}

    def render(self, template: dict, policy: str, sample: dict) -> dict:
        # Runtime đọc "text" (prompt đã ghép) làm state; phần output của template thành câu hỏi (build_config).
        return {"text": render_body(template, policy, "", sample["text"]), "label": sample["label"], "input": sample["text"]}

    def group_key(self, sample: dict) -> str:
        return norm_text(sample["text"])

    def strata_key(self, sample: dict) -> str | None:
        return sample["label"]

    def base_models(self, models: list[dict]) -> list[dict]:
        return [m for m in models if m["runtime"] == "decider" and m.get("status") == "ready"]

    def build_config(self, template: dict, rendered: list[dict]) -> dict:
        labels = sorted({s["label"] for s in rendered})
        if len(labels) < 2:
            raise ValueError("Cần ít nhất 2 nhãn khác nhau.")
        if len(labels) > MAX_LABELS:
            raise ValueError(f"Decider chỉ nhận tối đa {MAX_LABELS} nhãn (dữ liệu có {len(labels)}).")
        return {"labels": labels, "instructions": template["output"], "descriptions": {}}

    def check_eval(self, samples: list[dict], config: dict, model_name: str) -> None:
        unknown = sorted({s["label"] for s in samples} - set(config["labels"]))
        if unknown:
            shown = ", ".join(unknown[:10]) + ("…" if len(unknown) > 10 else "")
            raise ValueError(f"Tập chấm có {len(unknown)} nhãn mà \"{model_name}\" không được train: {shown}.")

    def model_entry(self, job: dict) -> dict:
        return {"runtime": "decider", "mode": "decide"}
