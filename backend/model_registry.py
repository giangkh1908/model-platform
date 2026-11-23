"""MLflow cho platform: theo dõi mọi lần train (tracking) và xuất bản model (registry).

Đi qua interface ModelRegistry. Bây giờ là MlflowRegistry: mỗi job train là một run (tham số, tag, lịch sử chỉ số
theo bước, file cấu hình); xuất bản = thêm checkpoint vào run đó rồi đăng ký thành một phiên bản model. File do MLflow
tự chuyển tới nơi lưu (thư mục trên máy, sau này Cloudflare R2), platform không cần biết. Chưa đặt
MLFLOW_TRACKING_URI thì là NullRegistry: mọi thao tác báo lỗi dễ hiểu, phần còn lại của platform vẫn chạy.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple

from .config import MLFLOW_TRACKING_URI

EXPERIMENT = "finetune"
ALIASES = ("champion", "challenger")  # champion: bản serving đang dùng; challenger: bản đang thử
RUN_STATUSES = ("FINISHED", "FAILED", "KILLED")
BATCH = 1000  # MLflow nhận tối đa 1000 chỉ số mỗi lần gửi


class Point(NamedTuple):
    """Một giá trị chỉ số tại một bước train."""

    key: str
    value: float
    step: int
    timestamp_ms: int


class ModelRegistry(ABC):
    enabled = False

    @abstractmethod
    def status(self) -> dict: ...

    @abstractmethod
    def log_run(self, name: str, params: dict, history: list[Point], tags: dict, folders: dict[str, Path],
                status: str) -> dict:
        """Ghi một lần train đã kết thúc. `folders`: {tên trên MLflow: thư mục trên máy}. Trả về {run_id, url}."""

    @abstractmethod
    def has_run(self, run_id: str) -> bool:
        """Run còn tồn tại (chưa bị xoá trên MLflow)."""

    @abstractmethod
    def attach(self, run_id: str, folders: dict[str, Path], metrics: dict) -> None:
        """Thêm file / chỉ số vào một run đã có (vd. checkpoint và kết quả eval lúc xuất bản)."""

    @abstractmethod
    def register(self, name: str, run_id: str, tags: dict, description: str) -> dict:
        """Đăng ký thư mục `model/` của run thành phiên bản mới của model `name`. Trả về {name, version}."""

    @abstractmethod
    def models(self) -> list[dict]:
        """Mọi model kèm các phiên bản (mới nhất trước) và alias."""

    @abstractmethod
    def set_alias(self, name: str, alias: str, version: str) -> None: ...

    @abstractmethod
    def delete_alias(self, name: str, alias: str) -> None: ...


def check_alias(alias: str) -> str:
    if alias not in ALIASES:
        raise ValueError(f"Alias phải là một trong: {', '.join(ALIASES)}.")
    return alias


class NullRegistry(ModelRegistry):
    def status(self) -> dict:
        return {"enabled": False, "message": "Chưa đặt MLFLOW_TRACKING_URI trong .env."}

    def _off(self):
        raise ValueError("Chưa cấu hình MLflow (MLFLOW_TRACKING_URI trong .env).")

    def log_run(self, name, params, history, tags, folders, status):
        self._off()

    def has_run(self, run_id):
        return False

    def attach(self, run_id, folders, metrics):
        self._off()

    def register(self, name, run_id, tags, description):
        self._off()

    def models(self):
        self._off()

    def set_alias(self, name, alias, version):
        self._off()

    def delete_alias(self, name, alias):
        self._off()


class MlflowRegistry(ModelRegistry):
    enabled = True

    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri.rstrip("/")
        self._client = None

    @property
    def client(self):
        # Import muộn: mlflow nạp khá lâu, chỉ cần khi thật sự dùng.
        if self._client is None:
            from mlflow import MlflowClient

            self._client = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.tracking_uri)
        return self._client

    def status(self) -> dict:
        import httpx

        try:
            ok = httpx.get(f"{self.tracking_uri}/health", timeout=3).status_code == 200
        except httpx.HTTPError:
            ok = False
        message = "" if ok else "Không kết nối được MLflow server. Chạy `docker compose up -d` ở thư mục project."
        return {"enabled": True, "online": ok, "ui": self.tracking_uri, "message": message}

    def _experiment_id(self) -> str:
        exp = self.client.get_experiment_by_name(EXPERIMENT)
        return exp.experiment_id if exp else self.client.create_experiment(EXPERIMENT)

    def _log_metrics(self, run_id: str, points: list[Point]) -> None:
        from mlflow.entities import Metric

        metrics = [Metric(p.key, float(p.value), p.timestamp_ms, p.step) for p in points]
        for start in range(0, len(metrics), BATCH):
            self.client.log_batch(run_id, metrics=metrics[start:start + BATCH])

    def _log_folders(self, run_id: str, folders: dict[str, Path]) -> None:
        for artifact_path, folder in folders.items():
            self.client.log_artifacts(run_id, str(folder), artifact_path)

    def log_run(self, name: str, params: dict, history: list[Point], tags: dict, folders: dict[str, Path],
                status: str) -> dict:
        from mlflow.entities import Param

        if status not in RUN_STATUSES:
            raise ValueError(f"Trạng thái run không hợp lệ: {status}.")
        experiment_id = self._experiment_id()
        run = self.client.create_run(experiment_id, run_name=name, tags=tags)
        run_id = run.info.run_id
        try:
            self.client.log_batch(run_id, params=[Param(k, str(v)[:6000]) for k, v in params.items()])
            self._log_metrics(run_id, history)
            self._log_folders(run_id, folders)
        except Exception:
            self.client.set_terminated(run_id, status="FAILED")
            raise
        self.client.set_terminated(run_id, status=status)
        return {"run_id": run_id, "url": f"{self.tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"}

    def has_run(self, run_id: str) -> bool:
        from mlflow.exceptions import MlflowException

        try:
            return self.client.get_run(run_id).info.lifecycle_stage == "active"
        except MlflowException:
            return False

    def attach(self, run_id: str, folders: dict[str, Path], metrics: dict) -> None:
        now = int(time.time() * 1000)
        self._log_metrics(run_id, [Point(k, v, 0, now) for k, v in metrics.items()])
        self._log_folders(run_id, folders)

    def register(self, name: str, run_id: str, tags: dict, description: str) -> dict:
        from mlflow.exceptions import MlflowException

        run = self.client.get_run(run_id)
        try:
            self.client.get_registered_model(name)
        except MlflowException:
            self.client.create_registered_model(name)
        version = self.client.create_model_version(name, source=f"{run.info.artifact_uri}/model", run_id=run_id,
                                                   tags=tags, description=description)
        return {"name": name, "version": version.version}

    def models(self) -> list[dict]:
        out = []
        for rm in self.client.search_registered_models(max_results=1000):
            aliases_by_version: dict[str, list[str]] = {}
            for alias, version in (rm.aliases or {}).items():
                aliases_by_version.setdefault(str(version), []).append(alias)
            versions = []
            for v in self.client.search_model_versions(f"name = '{rm.name}'", order_by=["version_number DESC"]):
                metrics = self.client.get_run(v.run_id).data.metrics if v.run_id else {}
                versions.append({
                    "version": v.version, "created": v.creation_timestamp / 1000, "description": v.description or "",
                    "tags": dict(v.tags or {}), "metrics": metrics,
                    "aliases": sorted(aliases_by_version.get(str(v.version), [])), "run_id": v.run_id,
                })
            out.append({"name": rm.name, "description": rm.description or "", "aliases": dict(rm.aliases or {}),
                        "versions": versions})
        return sorted(out, key=lambda m: m["name"].lower())

    def set_alias(self, name: str, alias: str, version: str) -> None:
        self.client.set_registered_model_alias(name, check_alias(alias), str(version))

    def delete_alias(self, name: str, alias: str) -> None:
        self.client.delete_registered_model_alias(name, check_alias(alias))


def make_registry() -> ModelRegistry:
    return MlflowRegistry(MLFLOW_TRACKING_URI) if MLFLOW_TRACKING_URI else NullRegistry()


REGISTRY: ModelRegistry = make_registry()
