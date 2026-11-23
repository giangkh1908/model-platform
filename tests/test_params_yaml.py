import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.finetune import params_yaml
from backend.finetune.decider import DeciderTask

TASK = DeciderTask()


def test_dump_parse_round_trip_with_comments():
    text = params_yaml.dump(TASK, {"seed": 7, "weight_decay": 0.01})
    assert "seed: 7" in text and "weight_decay: 0.01" in text
    assert "# Nâng cao" in text and "# Learning rate" in text
    assert params_yaml.parse(TASK, text) == TASK.clean_params({"seed": 7, "weight_decay": 0.01})


def test_missing_keys_use_defaults():
    assert params_yaml.parse(TASK, "epochs: 3\n")["epochs"] == 3
    assert params_yaml.parse(TASK, "") == TASK.clean_params({})


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("epochs: [1, 2\n", "dòng"),
        ("- 1\n- 2\n", "tên: giá trị"),
        ("epoch: 3\n", "epoch"),
        ("epochs: 0\n", "Số epoch"),
        ("lr: abc\n", "không hợp lệ"),
    ],
)
def test_parse_errors(text, message):
    with pytest.raises(ValueError, match=message):
        params_yaml.parse(TASK, text)


def test_yaml_api():
    client = TestClient(app)
    dumped = client.post("/api/finetune/params/yaml/dump", json={"task": "decider", "params": {"seed": 3}})
    assert dumped.status_code == 200 and "seed: 3" in dumped.json()["text"]
    parsed = client.post("/api/finetune/params/yaml/parse", json={"task": "decider", "text": "seed: 4\n"})
    assert parsed.json()["params"]["seed"] == 4
    bad = client.post("/api/finetune/params/yaml/parse", json={"task": "decider", "text": "khong_co: 1\n"})
    assert bad.status_code == 400
