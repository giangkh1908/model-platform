"""Tracking (ghi job lên MLflow) và xuất bản lên registry, với registry giả (không cần MLflow server)."""

import json

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.finetune import builds, evals, jobs, publish, tracking
from backend.finetune import connector as conn
from backend.model_registry import ModelRegistry, NullRegistry, check_alias


def read_folder(folder):
    return {p.name: json.loads(p.read_text("utf-8")) if p.suffix == ".json" else p.read_bytes() for p in folder.iterdir()}


class FakeRegistry(ModelRegistry):
    enabled = True

    def __init__(self) -> None:
        self.runs = {}
        self.registered = []
        self.aliases = {}

    def status(self):
        return {"enabled": True, "online": True}

    def log_run(self, name, params, history, tags, folders, status):
        run_id = f"run{len(self.runs) + 1}"
        self.runs[run_id] = {"name": name, "params": params, "history": history, "tags": tags, "status": status,
                             "files": {k: read_folder(f) for k, f in folders.items()}, "metrics": {}}
        return {"run_id": run_id, "url": f"http://mlflow/{run_id}"}

    def has_run(self, run_id):
        return run_id in self.runs

    def attach(self, run_id, folders, metrics):
        run = self.runs[run_id]
        run["metrics"].update(metrics)
        for key, folder in folders.items():
            run["files"].setdefault(key, {}).update(read_folder(folder))

    def register(self, name, run_id, tags, description):
        self.registered.append({"name": name, "run_id": run_id, "tags": tags, "description": description})
        return {"name": name, "version": str(len(self.registered))}

    def models(self):
        return []

    def set_alias(self, name, alias, version):
        self.aliases[(name, check_alias(alias))] = version

    def delete_alias(self, name, alias):
        self.aliases.pop((name, check_alias(alias)), None)


TASK_CONFIG = {"labels": ["a", "b"], "instructions": "Chọn", "descriptions": {}}
METRICS = [
    {"kind": "start", "total_steps": 4, "time": 1},
    {"kind": "eval", "step": 0, "epoch": 0, "best": True, "loss": 0.9, "acc": 0.5, "time": 2},
    {"kind": "train", "step": 2, "epoch": 0.5, "loss": 0.7, "lr": 1e-4, "grad_norm": 1.5, "time": 3},
    {"kind": "eval", "step": 4, "epoch": 1, "best": True, "loss": 0.3, "acc": 0.9, "time": 4},
    {"kind": "final", "best_step": 4, "best_eval_loss": 0.3, "time": 5,
     "threshold": {"threshold": 0.8, "precision": 0.95, "meets_target": True}, "val_curve": [{"coverage": 1}]},
]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    for store, name in ((builds.BUILDS, "builds"), (jobs.JOBS, "jobs"), (evals.EVALS, "evals")):
        monkeypatch.setattr(store, "root", tmp_path / name)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "head.safetensors").write_bytes(b"x")
    monkeypatch.setattr(conn.CONNECTOR, "output_path", lambda output_dir: ckpt)
    registry = FakeRegistry()
    monkeypatch.setattr(publish, "REGISTRY", registry)
    monkeypatch.setattr(tracking, "REGISTRY", registry)

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
    (jobs.JOBS.dir("j1") / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in METRICS), "utf-8")
    return registry


def add_eval(eval_id, created, state="done", key="job:j1"):
    evals.EVALS.create({"id": eval_id, "name": eval_id, "task": "decider", "created": created, "state": state,
                        "source": {"id": "b1", "title": "Intent mới v2"}, "n": 10})
    report = {"models": [{"key": key, "summary": {"acc": 0.9, "macro_f1": 0.85, "ece": 0.05},
                          "at_threshold": {"precision": 0.97, "coverage": 0.6}}]}
    (evals.EVALS.dir(eval_id) / "report.json").write_text(json.dumps(report), "utf-8")


# ---------------------------------------------------------------- tracking


def test_history_maps_metrics_by_kind_and_step(fake):
    points = {(p.key, p.step): p.value for p in tracking.history("j1")}
    assert points[("eval_loss", 0)] == 0.9
    assert points[("eval_acc", 4)] == 0.9
    assert points[("train_loss", 2)] == 0.7
    assert points[("train_grad_norm", 2)] == 1.5
    assert points[("final_threshold_threshold", 4)] == 0.8  # final đặt ở bước cuối
    assert points[("final_best_step", 4)] == 4
    keys = {key for key, _ in points}
    assert not any(k.endswith(("_epoch", "_best", "_time", "_meets_target")) for k in keys)
    assert "final_val_curve" not in keys


def test_history_keeps_step_zero_after_later_rows(fake):
    rows = [{"kind": "train", "step": 5, "loss": 0.5, "time": 1}, {"kind": "eval", "step": 0, "loss": 0.9, "time": 2}]
    (jobs.JOBS.dir("j1") / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    assert {(p.key, p.step) for p in tracking.history("j1")} == {("train_loss", 5), ("eval_loss", 0)}


def test_log_job_records_run_and_saves_link(fake):
    run = tracking.log_job("j1")
    logged = fake.runs[run["run_id"]]
    assert logged["status"] == "FINISHED"
    assert logged["name"] == "job 1"
    assert logged["tags"]["prompt.template"] == "khung@v3"
    assert logged["tags"]["prompt.policy"] == "luat@v5, luat-eval@v1"
    assert logged["params"]["n_train"] == 9
    assert sorted(logged["files"]["platform"]) == ["final.json", "params.json", "task_config.json"]
    job = jobs.JOBS.get("j1")
    assert (job["mlflow_run_id"], job["mlflow_url"]) == (run["run_id"], run["url"])


@pytest.mark.parametrize(("state", "status"), [("done", "FINISHED"), ("error", "FAILED"), ("stopped", "KILLED")])
def test_finish_hook_logs_every_end_state(fake, monkeypatch, state, status):
    monkeypatch.setattr(conn.CONNECTOR, "remove_output", lambda output_dir: None)
    jobs._finish(jobs.JOBS.get("j1"), state)
    assert [r["status"] for r in fake.runs.values()] == [status]


def test_finish_hook_survives_mlflow_errors(fake, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("MLflow tắt")

    monkeypatch.setattr(fake, "log_run", broken)
    jobs._finish(jobs.JOBS.get("j1"), "done")  # không được ném lỗi ra hàng đợi
    assert "mlflow_run_id" not in jobs.JOBS.get("j1")


def test_log_job_rejects_unfinished(fake):
    jobs.JOBS.update("j1", state="running")
    with pytest.raises(ValueError, match="chưa kết thúc"):
        tracking.log_job("j1")


# ---------------------------------------------------------------- xuất bản


def test_default_name():
    assert publish.default_name({"task": "decider", "build": {"title": "Intent mới v2"}}) == "decider-intent-moi"


def test_publish_logs_run_first_then_registers(fake):
    result = publish.publish("j1", None, " mô tả ")
    assert result == {"name": "decider-intent-moi", "version": "1"}
    (run_id,) = fake.runs
    run = fake.runs[run_id]
    assert run["files"]["model"] == {"head.safetensors": b"x"}
    assert "eval_report.json" not in run["files"]["platform"]
    reg = fake.registered[0]
    assert reg["run_id"] == run_id and reg["description"] == "mô tả"
    assert jobs.JOBS.get("j1")["published"][0]["version"] == "1"


def test_publish_reuses_existing_run(fake):
    run_id = tracking.log_job("j1")["run_id"]
    publish.publish("j1", "decider-intent", "")
    assert list(fake.runs) == [run_id]
    assert fake.registered[0]["run_id"] == run_id


def test_publish_attaches_latest_done_eval(fake):
    add_eval("old", created=1)
    add_eval("new", created=5)
    add_eval("running", created=9, state="running")
    add_eval("other", created=10, key="job:khac")
    publish.publish("j1", "decider-intent", "")
    run = next(iter(fake.runs.values()))
    assert run["files"]["platform"]["eval_report.json"]["eval"]["id"] == "new"
    assert run["metrics"]["golden_acc"] == 0.9
    assert run["metrics"]["golden_precision_at_threshold"] == 0.97
    assert fake.registered[0]["tags"]["eval_id"] == "new"


@pytest.mark.parametrize("name", ["có dấu", "-bat-dau-gach", "a/b"])
def test_publish_rejects_bad_names(fake, name):
    with pytest.raises(ValueError, match="Tên model"):
        publish.publish("j1", name, "")


def test_publish_requires_done_job(fake):
    jobs.JOBS.update("j1", state="running")
    with pytest.raises(ValueError, match="chưa train xong"):
        publish.publish("j1", None, "")


def test_null_registry_explains():
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        NullRegistry().models()


def test_alias_and_track_api(fake, monkeypatch):
    from backend.api import registry as api

    monkeypatch.setattr(api, "REGISTRY", fake)
    client = TestClient(app)
    assert client.put("/api/registry/models/m/aliases/champion", json={"version": "2"}).status_code == 200
    assert fake.aliases == {("m", "champion"): "2"}
    assert client.put("/api/registry/models/m/aliases/prod", json={"version": "2"}).status_code == 400
    assert client.delete("/api/registry/models/m/aliases/champion").status_code == 200
    assert fake.aliases == {}
    res = client.post("/api/finetune/jobs/j1/track")
    assert res.status_code == 200 and res.json()["run_id"] == "run1"
