import pytest

from backend.finetune.decider import MAX_INTENTS, DeciderTask, split_keys
from backend.hf import parse_repo
from backend.runner import clean_options

TASK = DeciderTask()
MAPPING = {"input": "t", "intent": "i", "gates": "g", "history": "h"}
SPEC = {"system": {"name": "cskh", "version": 1, "text": "Bạn là CSKH"},
        "intents": [{"key": "thanh_toan"}, {"key": "van_chuyen"}],
        "gates": [{"key": "lua_dao"}, {"key": "chuyen_nguoi"}]}


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [("lua_dao, chuyen_nguoi", ["chuyen_nguoi", "lua_dao"]), ("a;b|a", ["a", "b"]), ("", []), (None, []),
     (["x", " y "], ["x", "y"])],
)
def test_split_keys(value, expected):
    assert split_keys(value) == expected


def test_to_sample():
    row = {"t": " Gửi OTP đi ", "i": "", "g": "lua_dao", "h": "user: chào"}
    assert TASK.to_sample(row, MAPPING) == {"input": "Gửi OTP đi", "history": "user: chào", "intent": None,
                                             "gates": ["lua_dao"]}
    assert TASK.to_sample({"t": "x", "i": "thanh_toan"}, {"input": "t", "intent": "i"}) == {
        "input": "x", "history": "", "intent": "thanh_toan", "gates": None}
    with pytest.raises(ValueError, match="Intent hoặc cột Gate"):
        TASK.to_sample({"t": "x"}, {"input": "t"})
    with pytest.raises(ValueError, match="thiếu nhãn intent"):
        TASK.to_sample({"t": "x", "i": ""}, {"input": "t", "intent": "i"})
    with pytest.raises(ValueError, match="thiếu câu"):
        TASK.to_sample({"t": " ", "i": "a"}, {"input": "t", "intent": "i"})


def test_make_rows_labels_follow_spec():
    samples = [
        {"input": "a", "intent": "thanh_toan", "gates": ["lua_dao"]},
        {"input": "b", "intent": None, "gates": []},              # gán gate: không dính gate nào
        {"input": "c", "intent": "van_chuyen", "gates": None},    # không gán gate
    ]
    rows, dropped = TASK.make_rows(SPEC, samples)
    assert dropped == 0
    assert rows[0]["gates"] == {"lua_dao": True, "chuyen_nguoi": False}
    assert rows[1] == {"input": "b", "history": "", "intent": None, "gates": {"lua_dao": False, "chuyen_nguoi": False}}
    assert rows[2]["gates"] == {}


def test_make_rows_rejects_unknown_labels_and_drops_unlabelled():
    with pytest.raises(ValueError, match="intent không nằm trong bộ"):
        TASK.make_rows(SPEC, [{"input": "a", "intent": "doi_tra", "gates": None}])
    with pytest.raises(ValueError, match="gate không nằm trong bộ"):
        TASK.make_rows(SPEC, [{"input": "a", "intent": None, "gates": ["spam"]}])
    gate_only = {**SPEC, "intents": []}
    rows, dropped = TASK.make_rows(gate_only, [{"input": "a", "intent": "thanh_toan", "gates": None}])
    assert (rows, dropped) == ([], 1)


def test_build_config_checks():
    assert TASK.build_config(SPEC, [{"input": "a"}]) is SPEC
    with pytest.raises(ValueError, match="ít nhất 2 intent"):
        TASK.build_config({**SPEC, "intents": [{"key": "a"}]}, [{}])
    with pytest.raises(ValueError, match="ít nhất một intent hoặc một gate"):
        TASK.build_config({**SPEC, "intents": [], "gates": []}, [{}])
    with pytest.raises(ValueError, match=str(MAX_INTENTS)):
        TASK.build_config({**SPEC, "intents": [{"key": str(i)} for i in range(MAX_INTENTS + 1)]}, [{}])
    with pytest.raises(ValueError, match="Không còn dòng"):
        TASK.build_config(SPEC, [])


def test_check_eval():
    TASK.check_eval([{"intent": "thanh_toan", "gates": {"lua_dao": True}}], SPEC, "m")
    with pytest.raises(ValueError, match="2 intent / gate"):
        TASK.check_eval([{"intent": "doi_tra", "gates": {"spam": False}}], SPEC, "m")


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
