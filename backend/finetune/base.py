"""Interface của một loại fine-tune. Phần chung (dataset, prompt, build, hàng đợi job, chỉ số, eval) chỉ gọi
qua lớp này; thêm loại mới = viết một lớp con + một file runtime/finetune/<id>.py + một mục trong frontend.

Dòng dữ liệu (dataset) -> `to_sample`. Lúc build, mỗi dòng được ghép với template + policy thành prompt hoàn chỉnh
bằng `render`; runtime chỉ thấy mẫu đã ghép và cấu hình từ `build_config`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Field:
    """Một cột dữ liệu loại này cần; người dùng chọn cột nào trong file ứng với nó."""

    key: str
    label: str
    hint: str = ""
    required: bool = True


@dataclass(frozen=True)
class Param:
    """Một tham số train. Giao diện tự vẽ ô nhập từ danh sách này."""

    key: str
    label: str
    default: float | int | bool
    min: float = 0
    max: float = 0
    step: float = 1
    hint: str = ""
    kind: str = "number"  # number | bool
    advanced: bool = False


@dataclass(frozen=True)
class Series:
    source: str  # "train" | "eval": loại dòng trong metrics.jsonl
    key: str
    label: str


@dataclass(frozen=True)
class Chart:
    id: str
    title: str
    series: tuple[Series, ...]
    unit: str = ""  # "%" thì nhân 100 khi hiển thị
    hint: str = ""


COMMON_PARAMS = (
    Param("lr", "Learning rate", 1e-4, 1e-6, 1e-2, 1e-5, "LoRA thường 5e-5 đến 3e-4."),
    Param("epochs", "Số epoch", 2, 1, 20),
    Param("batch_size", "Batch size", 4, 1, 64, 1, "Số mẫu mỗi lần chạy qua GPU."),
    Param("grad_accum", "Gradient accumulation", 4, 1, 128, 1, "Batch hiệu dụng = batch size × số này."),
    Param("eval_every", "Chấm val mỗi (bước)", 0, 0, 100000, 1, "0 = mỗi epoch một lần."),
    Param("warmup_ratio", "Warmup", 0.03, 0, 0.5, 0.01, "Tỉ lệ số bước tăng dần learning rate.", advanced=True),
    Param("weight_decay", "Weight decay", 0.0, 0, 1, 0.01, advanced=True),
    Param("max_grad_norm", "Cắt gradient", 1.0, 0.1, 100, 0.1, advanced=True),
    Param("log_every", "Ghi chỉ số mỗi (bước)", 5, 1, 1000, 1, advanced=True),
    Param("seed", "Seed", 0, 0, 1_000_000, 1, "Cùng seed, cùng dữ liệu thì chạy lại ra cùng kết quả.", advanced=True),
    Param("gradient_checkpointing", "Gradient checkpointing", True, kind="bool",
          hint="Tiết kiệm VRAM, chậm hơn khoảng 30%.", advanced=True),
)

# Platform chỉ vẽ loss / eval loss; mọi chỉ số khác (learning rate, grad norm, VRAM, accuracy…) xem trên MLflow.
COMMON_CHARTS = (
    Chart("loss", "Loss", (Series("train", "loss", "Train loss"), Series("eval", "loss", "Eval loss")),
          hint="Eval loss tăng trong khi train loss vẫn giảm: đang overfit."),
)


class FinetuneTask(ABC):
    id: str
    name: str
    description: str
    runtime_python: str  # python trong WSL chạy runtime/finetune/run.py cho loại này
    fields: tuple[Field, ...]
    params: tuple[Param, ...] = ()  # tham số riêng, thêm vào COMMON_PARAMS
    defaults: dict = {}  # ghi đè giá trị mặc định của tham số chung (lớp con gán dict mới, không sửa tại chỗ)
    charts: tuple[Chart, ...] = ()
    example_rows: tuple[dict, ...] = ()  # vài dòng mẫu để người dùng xem định dạng cần chuẩn bị

    @abstractmethod
    def to_sample(self, row: dict, mapping: dict[str, str]) -> dict:
        """Dòng thô của file -> mẫu chuẩn theo `fields`. Dòng không hợp lệ thì ném ValueError."""

    @abstractmethod
    def group_key(self, sample: dict) -> str:
        """Khoá so trùng: golden có câu cùng khoá với train thì cảnh báo (điểm trên câu đó bị thổi phồng)."""

    def strata_key(self, sample: dict) -> str | None:
        """Nhóm để thống kê phân bố train/val (vd. nhãn). None = không thống kê."""
        return None

    @abstractmethod
    def base_models(self, models: list[dict]) -> list[dict]:
        """Những model trong danh sách train tiếp được bằng loại này."""

    @abstractmethod
    def render(self, template: dict, policy: str, sample: dict) -> dict:
        """Dòng dữ liệu + template + policy -> mẫu đã ghép mà runtime train / chấm."""

    @abstractmethod
    def build_config(self, template: dict, rendered: list[dict]) -> dict:
        """Cấu hình gửi cho runtime, rút từ template và các mẫu đã ghép (vd. câu hỏi, danh sách nhãn). Sai thì ValueError."""

    def check_eval(self, samples: list[dict], config: dict, model_name: str) -> None:  # noqa: B027 — tuỳ chọn, mặc định không kiểm tra gì
        """Tập chấm có dùng được với model train bằng `config` không (vd. nhãn lạ). Sai thì ValueError."""

    @abstractmethod
    def model_entry(self, job: dict) -> dict:
        """Mục thêm vào danh sách Model để chạy thử checkpoint (runtime, mode…)."""

    # ------------------------------------------------------------ dùng chung

    def all_params(self) -> list[Param]:
        own = {p.key for p in self.params}
        return [p for p in COMMON_PARAMS if p.key not in own] + list(self.params)

    def clean_params(self, raw: dict) -> dict:
        out = {}
        for p in self.all_params():
            value = raw.get(p.key, self.defaults.get(p.key, p.default))
            if p.kind == "bool":
                out[p.key] = bool(value)
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{p.label}: giá trị không hợp lệ ({value!r}).") from None
            if not p.min <= number <= p.max:
                raise ValueError(f"{p.label} phải trong khoảng {p.min} – {p.max}.")
            out[p.key] = int(number) if isinstance(p.default, int) and not isinstance(p.default, bool) else number
        return out

    def public(self, models: list[dict]) -> dict:
        params = []
        for p in self.all_params():
            d = asdict(p)
            d["default"] = self.defaults.get(p.key, p.default)
            params.append(d)
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "fields": [asdict(f) for f in self.fields],
            "params": params,
            "charts": [asdict(c) for c in (*COMMON_CHARTS, *self.charts)],
            "example_rows": list(self.example_rows),
            "base_models": [m["id"] for m in self.base_models(models)],
        }
