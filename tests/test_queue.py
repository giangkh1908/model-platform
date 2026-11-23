"""Hàng đợi GPU với connector giả: kiểm tra trạng thái sau khi chạy mà không cần WSL/GPU."""

import json

import pytest

from backend.finetune import gpu_queue
from backend.finetune.connector import RuntimeConnector
from backend.finetune.store import MetaStore


class FakeConnector(RuntimeConnector):
    def __init__(self, status: dict | None, code: int = 0) -> None:
        self.status = status
        self.code = code
        self.calls = []
        self.stopped = []

    def run(self, command, task_id, run_dir, python, label):
        self.calls.append((command, task_id, run_dir, python, label))
        if self.status is not None:
            (run_dir / "status.json").write_text(json.dumps(self.status), "utf-8")
        return self.code

    def stop(self, run_dir):
        self.stopped.append(run_dir)

    def stop_all(self):
        pass

    def remove_output(self, output_dir):
        pass


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = MetaStore("queue-test", "việc")
    monkeypatch.setattr(store, "root", tmp_path)
    monkeypatch.setattr(gpu_queue.SERVER, "alive", lambda: False)
    monkeypatch.setattr(gpu_queue, "health", lambda: None)
    finished = []
    runner = gpu_queue.Runner(store=store, command="train", prepare=lambda m: (store.dir(m["id"]), f"việc {m['id']}"),
                              python=lambda m: "/py", finish=lambda m, state: finished.append(state))
    store.create({"id": "j1", "name": "j1", "task": "decider", "created": 0, "state": "queued"})
    return store, runner, finished


def run_with(monkeypatch, connector, runner, stopping=False):
    monkeypatch.setattr(gpu_queue, "CONNECTOR", connector)
    queue = gpu_queue.Queue()
    queue.stopping = stopping
    queue._run(runner, "j1")


@pytest.mark.parametrize(
    ("status", "code", "stopping", "expected"),
    [
        ({"state": "done"}, 0, False, "done"),
        ({"state": "error", "message": "hết VRAM"}, 1, False, "error"),
        (None, 137, False, "error"),
        ({"state": "training"}, 1, True, "stopped"),
    ],
)
def test_run_states(setup, monkeypatch, status, code, stopping, expected):
    store, runner, finished = setup
    connector = FakeConnector(status, code)
    run_with(monkeypatch, connector, runner, stopping)

    meta = store.get("j1")
    assert meta["state"] == expected
    assert finished == [expected]
    command, task_id, run_dir, python, _ = connector.calls[0]
    assert (command, task_id, python) == ("train", "decider", "/py")
    assert run_dir == store.dir("j1")
    if expected == "error":
        assert meta["error"] == ("hết VRAM" if status else "Tiến trình thoát với mã 137, xem trang Log.")


def test_stop_running_item_uses_connector(setup, monkeypatch):
    store, runner, _ = setup
    connector = FakeConnector(None)
    monkeypatch.setattr(gpu_queue, "CONNECTOR", connector)
    monkeypatch.setitem(gpu_queue.RUNNERS, "train-test", runner)
    queue = gpu_queue.Queue()
    queue.current = ("train-test", "j1")
    queue.current_dir = store.dir("j1")

    assert queue.stop("train-test", "j1") == "Đã dừng."
    assert queue.stopping is True
    assert connector.stopped == [store.dir("j1")]
