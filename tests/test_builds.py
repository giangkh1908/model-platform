"""Build từ đầu đến cuối: prompt (kho trên máy) + dataset -> dòng "như thật" + spec đủ nội dung và phiên bản."""

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.finetune import builds, datasets, prompts
from backend.finetune.store import write_jsonl


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = prompts.LocalPromptStore()
    monkeypatch.setattr(store, "root", tmp_path / "prompts")
    monkeypatch.setattr(builds, "PROMPTS", store)
    monkeypatch.setattr(builds.BUILDS, "root", tmp_path / "builds")
    monkeypatch.setattr(datasets.DATASETS, "root", tmp_path / "datasets")

    store.save("system", None, "cskh", {"text": "Bạn là CSKH.", "intent_question": "Khách cần gì?"}, "")
    store.save("intent", None, "thanh_toan", {"text": "lỗi chuyển khoản"}, "")
    store.save("intent", "thanh_toan", "", {"text": "lỗi chuyển khoản, hoá đơn"}, "")
    store.set_label("intent", "thanh_toan", 1, "prod")
    store.save("intent", None, "van_chuyen", {"text": "giao hàng, shipper"}, "")
    store.save("gate", None, "lua_dao", {"text": "đòi OTP, mật khẩu", "question": "Có dấu hiệu lừa đảo không?"}, "")

    def add_dataset(dataset_id, source, rows):
        datasets.DATASETS.create({"id": dataset_id, "name": dataset_id, "task": "decider", "source": source,
                                  "created": 1, "rows": len(rows)})
        write_jsonl(datasets.DATASETS.dir(dataset_id) / "rows.jsonl", rows)

    train = [{"input": f"chuyển khoản lỗi {i}", "history": "", "intent": "thanh_toan", "gates": []} for i in range(8)]
    train += [{"input": f"shipper chưa tới {i}", "history": "", "intent": "van_chuyen", "gates": []} for i in range(8)]
    train += [{"input": "gửi OTP để nhận quà", "history": "", "intent": None, "gates": ["lua_dao"]}]
    add_dataset("sft", "sft", train)
    add_dataset("golden", "eval", [{"input": "đơn chưa giao", "history": "", "intent": "van_chuyen", "gates": None}])
    return store


def body(**extra):
    return {"task": "decider", "name": "cskh", "system": {"id": "cskh"},
            "intents": [{"id": "thanh_toan", "label": "prod"}, {"id": "van_chuyen"}],
            "gates": [{"id": "lua_dao", "version": 1}], "datasets": ["sft"], "val_percent": 20, **extra}


def test_spec_resolves_versions_and_labels(env):
    spec = builds.assemble_spec(body())
    assert spec["system"] == {"name": "cskh", "version": 1, "selected_by": "v1", "text": "Bạn là CSKH.",
                              "intent_question": "Khách cần gì?"}
    thanh_toan = spec["intents"][0]
    assert (thanh_toan["key"], thanh_toan["version"], thanh_toan["selected_by"], thanh_toan["policy"]) == (
        "thanh_toan", 1, "prod", "lỗi chuyển khoản")
    assert spec["gates"][0]["question"] == "Có dấu hiệu lừa đảo không?"
    assert builds.spec_title(spec) == {"system": "cskh@v1", "intents": "thanh_toan@v1, van_chuyen@v1",
                                       "gates": "lua_dao@v1"}


def test_create_build_rows_like_real_requests(env):
    meta = builds.create(body())
    train, val = builds.split_files(meta["id"])
    rows = train + val
    assert len(rows) == 17
    otp = next(r for r in rows if r["input"] == "gửi OTP để nhận quà")
    assert otp == {"input": "gửi OTP để nhận quà", "history": "", "intent": None, "gates": {"lua_dao": True},
                   "dataset": "sft"}
    assert all(r["gates"] == {"lua_dao": False} for r in rows if r["intent"])
    assert meta["task_config"]["intents"][1]["policy"] == "giao hàng, shipper"
    assert meta["stats"]["gates"][0]["key"] == "lua_dao"
    assert meta["stats"]["gates"][0]["train_true"] + meta["stats"]["gates"][0]["val_true"] == 1
    assert meta["refs"]["intents"][0] == {"id": "thanh_toan", "label": "prod"}
    assert builds.create(body())["version"] == 2


def test_eval_dataset_becomes_val(env):
    meta = builds.create(body(eval_datasets=["golden"]))
    _, val = builds.split_files(meta["id"])
    assert val == [{"input": "đơn chưa giao", "history": "", "intent": "van_chuyen", "gates": {}, "dataset": "golden"}]
    assert meta["has_eval"] and meta["stats"]["val_from"] == "eval"


def test_build_errors(env):
    with pytest.raises(ValueError, match="intent không nằm trong bộ"):
        builds.create(body(intents=[{"id": "thanh_toan"}]))
    with pytest.raises(ValueError, match="chỉ chọn một lần"):
        builds.create(body(intents=[{"id": "thanh_toan"}, {"id": "thanh_toan", "label": "prod"}]))
    with pytest.raises(ValueError, match="Chọn system"):
        builds.preview(body(system=None))
    with pytest.raises(ValueError, match="dataset SFT"):
        builds.preview(body(datasets=[]))
    with pytest.raises(ValueError, match="Eval"):
        builds.preview(body(datasets=["golden"]))


def test_preview_api(env):
    res = TestClient(app).post("/api/finetune/builds/preview", json=body())
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["spec_title"]["gates"] == "lua_dao@v1"
    assert len(data["examples"]) == 3 and "gates" in data["examples"][0]
