"""Mỗi job train là một run trên MLflow, cập nhật trực tiếp trong lúc train.

Job bắt đầu: mở run (tham số, tag: build, model gốc, phiên bản prompt). Trong lúc train: cứ SYNC_SECONDS giây đọc các
dòng mới của metrics.jsonl và đẩy sang (train_*, eval_*, final_* theo bước). Job kết thúc (xong, lỗi, bị dừng): đẩy nốt,
thêm thư mục platform/ (task_config.json, params.json, params.yaml, final.json) rồi đóng run với trạng thái tương ứng.
Runtime không biết gì về MLflow, chỉ ghi file theo runtime protocol.

Giao diện platform chỉ vẽ loss / eval loss; mọi chỉ số khác xem trên MLflow. MLflow tắt hay lỗi thì job vẫn chạy và kết
thúc bình thường, chỉ ghi lại trong Log; ghi bù bằng nút "Ghi lên MLflow" ở trang job.
"""

from __future__ import annotations

import json
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

from ..core import LOG, load_json
from ..model_registry import REGISTRY, Point
from ..registry import REG
from . import builds, evals, jobs, params_yaml
from .connector import PROTOCOL_VERSION
from .tasks import task_of

RUN_STATUS = {"done": "FINISHED", "error": "FAILED", "stopped": "KILLED"}
SYNC_SECONDS = 5
SKIP_KEYS = {"kind", "time", "step", "epoch", "best"}  # khoá của metrics.jsonl không phải chỉ số


def latest_eval(job_id: str) -> tuple[dict, dict] | None:
    """Lần eval xong gần nhất có chấm job này: (meta lần eval, kết quả của job trong báo cáo)."""
    key = f"job:{job_id}"
    for meta in sorted(evals.EVALS.all(), key=lambda m: m["created"], reverse=True):
        if meta["state"] != "done":
            continue
        report = load_json(evals.EVALS.dir(meta["id"]) / "report.json", None) or {}
        entry = next((m for m in report.get("models", []) if m.get("key") == key), None)
        if entry:
            return meta, entry
    return None


def eval_metrics(entry: dict) -> dict:
    """Chỉ số chính của một model trong báo cáo eval: eval_acc, eval_macro_f1, eval_ece, *_at_threshold."""
    metrics = {}
    summary = entry.get("summary") or {}
    for key in ("acc", "macro_f1", "ece"):
        if key in summary:
            metrics[f"golden_{key}"] = summary[key]
    at = entry.get("at_threshold") or {}
    for key in ("precision", "coverage"):
        if key in at:
            metrics[f"golden_{key}_at_threshold"] = at[key]
    return metrics


def prompt_tags(build: dict | None) -> dict:
    """Tham chiếu prompt đã dùng: template một bản, policy có thể nhiều bản (mỗi cặp policy + dataset một bản)."""
    if not build:
        return {}
    manifest = build["manifest"]
    template = manifest["template"]
    policies = []
    for entry in manifest["entries"] + manifest["eval_entries"]:
        ref = f"{entry['policy']['name']}@v{entry['policy']['version']}"
        if ref not in policies:
            policies.append(ref)
    return {"prompt.template": f"{template['name']}@v{template['version']}", "prompt.policy": ", ".join(policies)}


def tags_of(job: dict) -> dict:
    build = None
    try:
        build = builds.BUILDS.get(job["build"]["id"])
    except ValueError:  # build đã xoá: vẫn ghi được, chỉ thiếu tag prompt
        pass
    base = REG.get(job["base_model"]) or {}
    return {
        "task": job["task"], "job_id": job["id"], "job_name": job["name"],
        "base_model": job["base_model"], "base_model_repo": base.get("repo", ""),
        "build": job["build"]["title"], "build_id": job["build"]["id"],
        "runtime_protocol": str(PROTOCOL_VERSION), "source": "model-platform",
        **prompt_tags(build),
    }


def params_of(job: dict) -> dict:
    return {**job["params"], "base_model": job["base_model"], "build": job["build"]["title"],
            "n_train": job["build"]["train"], "n_val": job["build"]["val"]}


def flatten(prefix: str, values: dict) -> dict:
    """Chỉ số số học, gồm cả một tầng lồng nhau: {"threshold": {"precision": 0.9}} -> final_threshold_precision."""
    out = {}
    for key, value in values.items():
        if key in SKIP_KEYS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[f"{prefix}{key}"] = value
        elif isinstance(value, dict):
            for sub, sub_value in value.items():
                if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                    out[f"{prefix}{key}_{sub}"] = sub_value
    return out


def points_of(rows: list[dict], last_step: int = 0) -> tuple[list[Point], int]:
    """Dòng metrics.jsonl -> điểm chỉ số: train thành train_*, eval thành eval_*, final thành final_* (ở bước cuối).
    `last_step`: bước lớn nhất đã gặp ở các lần đọc trước. Trả về (điểm, bước lớn nhất mới)."""
    points: list[Point] = []
    for row in rows:
        kind = row.get("kind")
        step = int(row["step"]) if row.get("step") is not None else last_step  # bước 0 là bước thật, không phải "thiếu"
        last_step = max(last_step, step)
        ts = int((row.get("time") or 0) * 1000)
        if kind in ("train", "eval"):
            values = flatten(f"{kind}_", row)
        elif kind == "final":
            values = flatten("final_", row)
            step = last_step
        else:
            continue
        points.extend(Point(key, value, step, ts) for key, value in values.items())
    return points, last_step


def history(job_id: str) -> list[Point]:
    rows, _ = jobs.read_metrics(job_id)
    return points_of(rows)[0]


def config_files(job: dict) -> dict:
    final = jobs.final_metrics(job["id"])
    return {
        "task_config.json": job["task_config"],
        "params.json": job["params"],
        "params.yaml": params_yaml.dump(task_of(job["task"]), job["params"]),
        "final.json": {k: v for k, v in final.items() if k not in ("kind", "time")},
    }


@contextmanager
def json_folder(files: dict):
    """Thư mục tạm chứa các file (chuỗi ghi nguyên văn, còn lại ghi JSON), xoá khi xong."""
    with tempfile.TemporaryDirectory(prefix="mlflow-") as tmp:
        folder = Path(tmp)
        for filename, content in files.items():
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
            (folder / filename).write_text(text, "utf-8")
        yield folder


def log_job(job_id: str, state: str | None = None) -> dict:
    """Ghi job lên MLflow, lưu run_id / url vào meta của job. `state`: trạng thái cuối (mặc định lấy từ meta)."""
    job = jobs.JOBS.get(job_id)
    state = state or job["state"]
    if state not in RUN_STATUS:
        raise ValueError("Job chưa kết thúc, chưa ghi lên MLflow được.")
    with json_folder(config_files(job)) as platform_dir:
        run = REGISTRY.log_run(job["name"], params_of(job), history(job_id), tags_of(job), {"platform": platform_dir},
                               RUN_STATUS[state])
    jobs.JOBS.update(job_id, mlflow_run_id=run["run_id"], mlflow_url=run["url"])
    LOG.add(f"=== Đã ghi job {job_id} lên MLflow ===")
    return run


class LiveRun:
    """Đẩy các dòng mới của metrics.jsonl sang run MLflow trong lúc job train (luồng nền)."""

    def __init__(self, job_id: str, run_id: str) -> None:
        self.job_id = job_id
        self.run_id = run_id
        self.cursor = 0
        self.last_step = 0
        self.warned = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> LiveRun:
        self.thread.start()
        return self

    def sync(self) -> None:
        with self.lock:  # luồng nền và lần đẩy cuối không được đọc cùng một đoạn hai lần
            rows, cursor = jobs.read_metrics(self.job_id, self.cursor)
            points, last_step = points_of(rows, self.last_step)
            if points:
                REGISTRY.log_points(self.run_id, points)
            self.cursor, self.last_step = cursor, last_step

    def _loop(self) -> None:
        while not self.stop_event.wait(SYNC_SECONDS):
            try:
                self.sync()
            except Exception as e:  # MLflow tạm lỗi: thử lại lần sau, dòng chưa đẩy vẫn còn trong file
                if not self.warned:
                    LOG.add(f"=== Job {self.job_id}: chưa đẩy được chỉ số lên MLflow ({e}), sẽ thử lại. ===")
                    self.warned = True

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=SYNC_SECONDS * 2)
        self.sync()


LIVE: dict[str, LiveRun] = {}


def on_start(meta: dict) -> None:
    """Job bắt đầu train: mở run và bắt đầu đẩy chỉ số. Lỗi thì job vẫn chạy, cuối job sẽ ghi trọn một lần."""
    if not REGISTRY.enabled:
        return
    try:
        run = REGISTRY.start_run(meta["name"], params_of(meta), tags_of(meta))
    except Exception as e:
        LOG.add(f"=== Job {meta['id']}: không mở được run MLflow ({e}), sẽ thử ghi lúc job kết thúc. ===")
        return
    jobs.JOBS.update(meta["id"], mlflow_run_id=run["run_id"], mlflow_url=run["url"])
    LIVE[meta["id"]] = LiveRun(meta["id"], run["run_id"]).start()


def _close_live(live: LiveRun, state: str) -> None:
    status = RUN_STATUS[state]
    try:
        live.close()
        with json_folder(config_files(jobs.JOBS.get(live.job_id))) as platform_dir:
            REGISTRY.finish_run(live.run_id, {"platform": platform_dir}, status)
    except Exception:
        REGISTRY.finish_run(live.run_id, {}, status)  # ít nhất cũng đóng run, không để "đang chạy" mãi
        raise
    LOG.add(f"=== Đã ghi job {live.job_id} lên MLflow ===")


def on_finish(meta: dict, state: str) -> None:
    """Hook sau khi job kết thúc: không để lỗi MLflow làm hỏng hàng đợi."""
    if not REGISTRY.enabled:
        return
    live = LIVE.pop(meta["id"], None)
    try:
        if live:
            _close_live(live, state)
        else:
            log_job(meta["id"], state)
    except Exception as e:  # MLflow tắt, mạng lỗi…: job vẫn xong, ghi bù sau
        LOG.add(f"=== Job {meta['id']}: chưa ghi được lên MLflow ({e}). Bấm \"Ghi lên MLflow\" ở trang job để ghi bù. ===")


jobs.START_HOOKS.append(on_start)
jobs.FINISH_HOOKS.append(on_finish)
