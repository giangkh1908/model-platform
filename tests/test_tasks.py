import pytest

from backend.finetune.decider import MAX_LABELS, DeciderTask
from backend.hf import parse_repo
from backend.runner import clean_options

TASK = DeciderTask()


def test_clean_params_defaults_types_and_ranges():
    params = TASK.clean_params({})
    assert params["lr"] == 1e-4
    assert params["epochs"] == 2 and isinstance(params["epochs"], int)
    assert params["gradient_checkpointing"] is True

    assert TASK.clean_params({"epochs": "3"})["epochs"] == 3
    with pytest.raises(ValueError, match="Số epoch"):
        TASK.clean_params({"epochs": 0})
    with pytest.raises(ValueError, match="không hợp lệ"):
        TASK.clean_params({"lr": "abc"})


def test_decider_to_sample():
    mapping = {"text": "t", "label": "l"}
    assert TASK.to_sample({"t": " x ", "l": " a "}, mapping) == {"text": "x", "label": "a"}
    with pytest.raises(ValueError, match="nhãn"):
        TASK.to_sample({"t": "x", "l": ""}, mapping)


def test_decider_build_config_and_check_eval():
    rendered = [{"label": "b"}, {"label": "a"}, {"label": "a"}]
    config = TASK.build_config({"output": "Chọn nhãn"}, rendered)
    assert config == {"labels": ["a", "b"], "instructions": "Chọn nhãn", "descriptions": {}}

    with pytest.raises(ValueError, match="2 nhãn"):
        TASK.build_config({"output": "x"}, [{"label": "a"}])
    with pytest.raises(ValueError, match=str(MAX_LABELS)):
        TASK.build_config({"output": "x"}, [{"label": str(i)} for i in range(MAX_LABELS + 1)])

    TASK.check_eval([{"label": "a"}], config, "m")
    with pytest.raises(ValueError, match="1 nhãn"):
        TASK.check_eval([{"label": "c"}], config, "m")


def test_decider_render_fills_template():
    out = TASK.render({"body": "{policy}|{input}", "output": "o"}, "P", {"text": "Q", "label": "a"})
    assert out == {"text": "P|Q", "label": "a", "input": "Q"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://huggingface.co/BAAI/bge-m3", "BAAI/bge-m3"),
        ("huggingface.co/models/org/name.v1", "org/name.v1"),
        ("  org/model-x  ", "org/model-x"),
        ("không phải link", None),
    ],
)
def test_parse_repo(text, expected):
    assert parse_repo(text) == expected


def test_clean_options_clamps_and_drops_unknown():
    assert clean_options("vllm", {"ctx": 10, "gpu_util": 2, "parallel": 4, "x": 1}) == {"ctx": 256, "gpu_util": 0.95}
    assert clean_options("llamacpp", {"ctx": "", "parallel": "8"}) == {"parallel": 8}
    with pytest.raises(ValueError):
        clean_options("vllm", {"ctx": "abc"})
