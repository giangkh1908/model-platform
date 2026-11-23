import pytest

from backend.finetune import prompts


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = prompts.LocalPromptStore()
    monkeypatch.setattr(s, "root", tmp_path)
    return s


def test_check_content_per_kind():
    assert prompts.check_content("system", {"text": " Bạn là CSKH ", "intent_question": " Khách cần gì? "}) == {
        "text": "Bạn là CSKH", "intent_question": "Khách cần gì?"}
    assert prompts.check_content("intent", {"text": "lỗi thanh toán", "khac": 1}) == {"text": "lỗi thanh toán"}
    assert prompts.check_content("gate", {"text": "lừa đảo"}) == {"text": "lừa đảo", "question": ""}
    with pytest.raises(ValueError, match="Policy của gate"):
        prompts.check_content("gate", {"text": "  "})
    with pytest.raises(ValueError, match="system, intent hoặc gate"):
        prompts.check_content("template", {"text": "x"})


@pytest.mark.parametrize("name", ["Thanh Toan", "thanh toán", "-abc", "", "a" * 81, "a/b"])
def test_names_must_be_dataset_labels(name):
    with pytest.raises(ValueError, match="Tên prompt"):
        prompts.check_name(name)


def test_versions_labels_and_lookup(store):
    v1 = store.save("intent", None, "thanh_toan", {"text": "chuyển khoản, hoá đơn"}, "tạo")
    v2 = store.save("intent", "thanh_toan", "", {"text": "chuyển khoản, hoá đơn, hoàn tiền"}, "")
    assert (v1["id"], v1["version"], v2["version"]) == ("thanh_toan", 1, 2)

    store.set_label("intent", "thanh_toan", 1, "prod")
    store.set_label("intent", "thanh_toan", 2, "dev")
    assert store.get("intent", "thanh_toan", label="prod")["content"] == {"text": "chuyển khoản, hoá đơn"}
    assert store.get("intent", "thanh_toan", label="dev")["version"] == 2
    store.set_label("intent", "thanh_toan", 2, "prod")  # dời nhãn sang bản khác
    assert store.list("intent")[0]["labels"] == {"dev": 2, "prod": 2}
    assert [v["labels"] for v in store.versions("intent", "thanh_toan")] == [[], ["dev", "prod"]]
    assert store.versions("intent", "thanh_toan")[0]["note"] == "tạo"

    with pytest.raises(ValueError, match="không đổi"):
        store.save("intent", "thanh_toan", "", {"text": "chuyển khoản, hoá đơn, hoàn tiền"}, "")
    with pytest.raises(ValueError, match="v9"):
        store.get("intent", "thanh_toan", 9)
    with pytest.raises(ValueError, match="Nhãn phải"):
        store.set_label("intent", "thanh_toan", 1, "staging")


def test_label_missing_and_names_unique_across_kinds(store):
    store.save("gate", None, "lua_dao", {"text": "đòi OTP"}, "")
    with pytest.raises(ValueError, match="chưa có phiên bản nào mang nhãn prod"):
        store.get("gate", "lua_dao", label="prod")
    with pytest.raises(ValueError, match="Đã có prompt"):
        store.save("intent", None, "lua_dao", {"text": "x"}, "")
    store.delete("gate", "lua_dao")
    assert store.list("gate") == []


def test_wrong_kind_and_bad_id(store):
    store.save("system", None, "cskh", {"text": "Bạn là CSKH"}, "")
    with pytest.raises(ValueError, match="Không có intent"):
        store.get("intent", "cskh")
    with pytest.raises(ValueError):
        store.get("system", "../ngoai")
