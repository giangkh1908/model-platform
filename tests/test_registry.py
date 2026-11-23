"""Xuất bản lên registry với registry giả (không cần MLflow server)."""

import json

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.finetune import builds, evals, jobs, publish
from backend.finetune import connector as conn
from backend.model_registry import ModelRegistry, NullRegistry, check_alias


class FakeRegistry(ModelRegistry):
    enabled = True

    def __init__(self) -> None:
        self.published = []
        self.aliases = {}

    def status(self):
        return {"enabled": True, "online": True}

    def publish(self, name, folders, params, metrics, tags, description):
        files = {key: sorted(p.name for p in folder.iterdir()) for key, folder in folders.items()}
        contents = {p.name: json.loads(p.read_text("utf-8")) for p in folders["platform"].iterdir()}
        self.published.append({"name": name, "files": files, "contents": contents, "params": params,
                               "metrics": metrics, "tags": tags, "description": description})
        return {"name": name, "version": str(len(self.published)), "run_id": "r"}

    def models(self):
        return []

    def set_alias(self, name, alias, version):
        self.aliases[(name, check_alias(alias))] = version

    def delete_alias(self, name, alias):
        self.aliases.pop((name, check_alias(alias)), None)


TASK_CONFIG = {"labels": ["a", "b"], "instructions": "Chọn", "descriptions": {}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for store, name in ((builds.BUILDS, "builds"), (jobs.JOBS, "jobs"), (evals.EVALS, "evals")):
        monkeypatch.setattr(store, "root", tmp_path / name)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "head.safetensors").write_bytes(b"x")
    monkeypatch.setattr(conn.CONNECTOR, "output_path", lambda output_dir: ckpt)
    fake = FakeRegistry()
    monkeypatch.setattr(publish, "REGISTRY", fake)

    builds.BUILDS.create({
        "id": "b1", "name": "Intent mới", "version": 2, "task": "decider", "created": 1, "task_config": TASK_CONFIG,
        "manifest": {"template": {"id": "t", "name": "khung", "version": 3, "content": {}},
                     "entries": [{"policy": {"name": "luat", "version": 5}}, {"policy": {"name": "luat", "version": 5}}],
                     "eval_entries": [{"policy": {"name": "luat-eval", "version": 1}}]},
    })
    jobs.JOBS.create({
        "id": "j1", "name": "job 1", "task": "decider", "created": 1, "state": "done", "base_model": "base",
        "output_dir": "/x", "params": {"lr": 0.0001}, "task_config": TASK_CONFIG,
        "build": {"id": "b1", "title": "Intent mới v2", "train": 9, "val": 1, "has_eval": True},
    })
    final = {"kind": "final", "best_eval_loss": 0.3, "threshold": {"threshold": 0.8}, "val_curve": [1, 2]}
    (jobs.JOBS.dir("j1") / "metrics.jsonl").write_text(json.dumps(final) + "\n", "utf-8")
    return fake


def add_eval(eval_id, created, state="done", key="job:j1"):
    evals.EVALS.create({"id": eval_id, "name": eval_id, "task": "decider", "created": created, "state": state,
                        "source": {"id": "b1", "title": "Intent mới v2"}, "n": 10})
    report = {"models": [{"key": key, "summary": {"acc": 0.9, "macro_f1": 0.85, "ece": 0.05},
                          "at_threshold": {"precision": 0.97, "coverage": 0.6}}]}
    (evals.EVALS.dir(eval_id) / "report.json").write_text(json.dumps(report), "utf-8")


def test_default_name():
    assert publish.default_name({"task": "decider", "build": {"title": "Intent mới v2"}}) == "decider-intent-moi"


def test_publish_bundle_without_eval(setup):
    result = publish.publish("j1", None, " mô tả ")
    assert result == {"name": "decider-intent-moi", "version": "1", "run_id": "r"}
    sent = setup.published[0]
    assert sent["files"] == {"model": ["head.safetensors"], "platform": ["final.json", "params.json", "task_config.json"]}
    assert "val_curve" not in sent["contents"]["final.json"]
    assert sent["contents"]["task_config.json"] == TASK_CONFIG
    assert sent["metrics"] == {"best_eval_loss": 0.3, "threshold": 0.8}
    assert sent["tags"]["prompt.template"] == "khung@v3"
    assert sent["tags"]["prompt.policy"] == "luat@v5, luat-eval@v1"
    assert sent["description"] == "mô tả"
    assert jobs.JOBS.get("j1")["published"][0]["version"] == "1"


def test_publish_attaches_latest_done_eval(setup):
    add_eval("old", created=1)
    add_eval("new", created=5)
    add_eval("running", created=9, state="running")
    add_eval("other", created=10, key="job:khac")
    publish.publish("j1", "decider-intent", "")
    sent = setup.published[0]
    assert sent["contents"]["eval_report.json"]["eval"]["id"] == "new"
    assert sent["tags"]["eval_id"] == "new"
    assert sent["metrics"]["eval_acc"] == 0.9
    assert sent["metrics"]["eval_precision_at_threshold"] == 0.97


@pytest.mark.parametrize("name", ["có dấu", "-bat-dau-gach", "a/b"])
def test_publish_rejects_bad_names(setup, name):
    with pytest.raises(ValueError, match="Tên model"):
        publish.publish("j1", name, "")


def test_publish_requires_done_job(setup):
    jobs.JOBS.update("j1", state="running")
    with pytest.raises(ValueError, match="chưa train xong"):
        publish.publish("j1", None, "")


def test_null_registry_explains():
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        NullRegistry().models()


def test_alias_api(setup, monkeypatch):
    from backend.api import registry as api

    monkeypatch.setattr(api, "REGISTRY", setup)
    client = TestClient(app)
    assert client.put("/api/registry/models/m/aliases/champion", json={"version": "2"}).status_code == 200
    assert setup.aliases == {("m", "champion"): "2"}
    assert client.put("/api/registry/models/m/aliases/prod", json={"version": "2"}).status_code == 400
    assert client.delete("/api/registry/models/m/aliases/champion").status_code == 200
    assert setup.aliases == {}
