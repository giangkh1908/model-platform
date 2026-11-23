"""Phần ghép prompt của runtime decider (không cần torch): bố cục state, tách mẫu intent / gate, request tương đương."""

import importlib.util
import sys
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent.parent / "runtime" / "finetune"
sys.path.insert(0, str(RUNTIME))  # runtime import common, classify_metrics theo tên trần
_spec = importlib.util.spec_from_file_location("runtime_decider", RUNTIME / "decider.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)

SPEC = {
    "system": {"text": "Bạn là CSKH.", "intent_question": ""},
    "intents": [{"key": "thanh_toan", "policy": "chuyển khoản, hoá đơn"}, {"key": "van_chuyen", "policy": "giao hàng"}],
    "gates": [{"key": "lua_dao", "policy": "đòi OTP", "question": "Có lừa đảo không?"},
              {"key": "chuyen_nguoi", "policy": "khách đòi gặp người", "question": ""}],
}


def test_state_puts_shared_part_first_then_user():
    row = {"input": " Gửi OTP đi ", "history": [{"role": "assistant", "content": "Chào bạn"}]}
    text = rd.render_state_text(SPEC, row)
    shared = rd.shared_state(SPEC)
    assert text.startswith(shared)  # phần dùng chung đứng đầu, giống hệt mọi request -> cache được
    assert text.endswith("## Người dùng\nGửi OTP đi")
    assert "## Lịch sử hội thoại\nAI: Chào bạn" in text
    assert shared.index("### thanh_toan\nchuyển khoản, hoá đơn") < shared.index("## Gate") < shared.index("### lua_dao")
    other = rd.render_state_text(SPEC, {"input": "câu khác"})
    assert other.startswith(shared)


def test_expand_rows_one_example_per_question():
    rows = [
        {"input": "a", "intent": "thanh_toan", "gates": {"lua_dao": False, "chuyen_nguoi": True}},
        {"input": "b", "intent": None, "gates": {"lua_dao": True}},
        {"input": "c", "intent": "khong_co", "gates": {"khong_co": True}},
    ]
    examples = rd.expand_rows(SPEC, rows)
    assert [(e["row"]["input"], e["q"], e["key"], e["gold"]) for e in examples] == [
        ("a", "intent", "intent", "thanh_toan"),
        ("a", "gate", "lua_dao", "false"),
        ("a", "gate", "chuyen_nguoi", "true"),
        ("b", "gate", "lua_dao", "true"),
    ]


def test_questions_and_request():
    req = rd.request_of(SPEC, {"input": "x"})
    assert set(req["questions"]) == {"intent", "gate_lua_dao", "gate_chuyen_nguoi"}
    assert req["questions"]["intent"] == {"type": "choice", "instructions": rd.DEFAULT_INTENT_QUESTION,
                                          "criteria": {"thanh_toan": None, "van_chuyen": None}}
    assert req["questions"]["gate_lua_dao"] == {"type": "noul", "instructions": "Có lừa đảo không?"}
    assert "chuyen_nguoi" in req["questions"]["gate_chuyen_nguoi"]["instructions"]
    gate_only = rd.request_of({**SPEC, "intents": []}, {"input": "x"})
    assert "intent" not in gate_only["questions"]
    assert "## Intent" not in gate_only["state"]


def test_format_history():
    assert rd.format_history("") == ""
    assert rd.format_history(" sẵn ") == "sẵn"
    assert rd.format_history('[{"role": "bot", "content": "hi"}]') == "AI: hi"
    assert rd.format_history("[hỏng") == "[hỏng"


def test_confidence():
    assert rd.confidence([0.5, 0.5]) == 0
    assert rd.confidence([1.0, 0.0, 0.0]) == 1
