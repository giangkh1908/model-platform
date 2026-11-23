"""Model registry: nơi xuất bản model đã train để serving-platform lấy về chạy.

Đi qua interface ModelRegistry. Bây giờ là MlflowRegistry: metadata (tên, phiên bản, alias, chỉ số, tag) trong
MLflow server, file model trên Cloudflare R2 (MLflow tự chuyển file, platform không giữ key R2). Chưa đặt
MLFLOW_TRACKING_URI thì là NullRegistry: mọi thao tác báo lỗi dễ hiểu, phần còn lại của platform vẫn chạy.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

from .config import MLFLOW_TRACKING_URI

EXPERIMENT = "model-platform"
ALIASES = ("champion", "challenger")  # champion: bản serving đang dùng; challenger: bản đang thử


class ModelRegistry(ABC):
    enabled = False

    @abstractmethod
    def status(self) -> dict: ...

    @abstractmethod
    def publish(self, name: str, folders: dict[str, Path], params: dict, metrics: dict, tags: dict,
                description: str) -> dict:
        """Đẩy các thư mục (`{tên trên registry: thư mục trên máy}`) thành một phiên bản mới của model `name`.
        Trả về {name, version}."""

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
        raise ValueError("Chưa cấu hình model registry (MLFLOW_TRACKING_URI trong .env).")

    def publish(self, name, folders, params, metrics, tags, description):
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
        # Import muộn: mlflow nạp khá lâu, chỉ cần khi thật sự dùng registry.
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

    def publish(self, name: str, folders: dict[str, Path], params: dict, metrics: dict, tags: dict,
                description: str) -> dict:
        from mlflow.entities import Metric, Param
        from mlflow.exceptions import MlflowException

        client = self.client
        run = client.create_run(self._experiment_id(), run_name=f"publish {name}", tags=tags)
        run_id = run.info.run_id
        try:
            now = int(time.time() * 1000)
            client.log_batch(
                run_id,
                metrics=[Metric(k, float(v), now, 0) for k, v in metrics.items()],
                params=[Param(k, str(v)[:6000]) for k, v in params.items()],
            )
            for artifact_path, folder in folders.items():
                client.log_artifacts(run_id, str(folder), artifact_path)
            client.set_terminated(run_id)
        except Exception:
            client.set_terminated(run_id, status="FAILED")
            raise
        try:
            client.get_registered_model(name)
        except MlflowException:
            client.create_registered_model(name)
        version = client.create_model_version(name, source=f"{run.info.artifact_uri}/model", run_id=run_id,
                                              tags=tags, description=description)
        return {"name": name, "version": version.version, "run_id": run_id}

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
                    "tags": dict(v.tags or {}), "metrics": metrics, "aliases": sorted(aliases_by_version.get(str(v.version), [])),
                    "run_id": v.run_id,
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
