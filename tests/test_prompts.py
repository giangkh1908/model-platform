import pytest

from backend.finetune import prompts


def test_render_body_only_replaces_placeholders():
    template = {"body": "Luật: {policy}\nLịch sử: {history}\nCâu: {input}\nJSON mẫu: {\"a\": {b}}"}
    out = prompts.render_body(template, "P", [{"role": "human", "content": "hi"}], "Q")
    assert out == "Luật: P\nLịch sử: user: hi\nCâu: Q\nJSON mẫu: {\"a\": {b}}"


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        (None, ""),
        ("", ""),
        ("  sẵn  ", "sẵn"),
        ('[{"role": "bot", "content": "chào"}]', "AI: chào"),
        ("[không phải json", "[không phải json"),
        ([{"role": "assistant", "content": "x"}, "dòng thô"], "AI: x\ndòng thô"),
    ],
)
def test_format_history(history, expected):
    assert prompts.format_history(history) == expected


def test_check_content_template_needs_input_and_output():
    with pytest.raises(ValueError, match="input"):
        prompts.check_content("template", {"body": "không có chỗ trống", "output": "x"})
    with pytest.raises(ValueError, match="output"):
        prompts.check_content("template", {"body": "{input}", "output": " "})
    assert prompts.check_content("template", {"body": " {input} ", "output": " o "}) == {"body": "{input}", "output": "o"}


def test_check_content_policy():
    with pytest.raises(ValueError):
        prompts.check_content("policy", {"text": ""})
    assert prompts.check_content("policy", {"text": " luật "}) == {"text": "luật"}


def test_local_store_keeps_old_versions(tmp_path, monkeypatch):
    store = prompts.LocalPromptStore()
    monkeypatch.setattr(store, "root", tmp_path)

    v1 = store.save("policy", None, "An toàn", {"text": "luật 1"}, "đầu tiên")
    v2 = store.save("policy", v1["id"], "", {"text": "luật 2"}, "sửa")
    assert (v1["version"], v2["version"]) == (1, 2)
    assert store.get("policy", v1["id"], 1)["content"] == {"text": "luật 1"}
    assert store.get("policy", v1["id"])["content"] == {"text": "luật 2"}
    assert store.list("policy")[0]["latest"] == 2

    with pytest.raises(ValueError, match="không đổi"):
        store.save("policy", v1["id"], "", {"text": "luật 2"}, "")
    with pytest.raises(ValueError, match="v9"):
        store.get("policy", v1["id"], 9)

    store.delete("policy", v1["id"])
    assert store.list("policy") == []


def test_local_store_rejects_bad_kind_and_id(tmp_path, monkeypatch):
    store = prompts.LocalPromptStore()
    monkeypatch.setattr(store, "root", tmp_path)
    with pytest.raises(ValueError):
        store.get("khac", "x")
    with pytest.raises(ValueError):
        store.get("policy", "../ngoai")
