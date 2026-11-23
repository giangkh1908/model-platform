"""Kiểm tra lớp API (không cần WSL hay GPU): danh sách route, bảo vệ origin, đổi ValueError thành 400."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import app

ROUTES_FILE = Path(__file__).with_name("routes.txt")


def current_routes() -> list[str]:
    """Lấy từ OpenAPI schema: đủ cả route trong các router đã include."""
    lines = []
    for path, operations in app.openapi()["paths"].items():
        if path == "/":  # chỉ có khi chưa build frontend (vd. trên CI), không phải API
            continue
        for method in operations:
            lines.append(f"{method.upper()} {path}")
    return sorted(lines)


@pytest.fixture
def client():
    # Không dùng "with": không chạy lifespan (giám sát GPU, hàng đợi), chỉ thử route.
    return TestClient(app)


def test_routes_match_snapshot():
    """Đổi API (thêm/bớt/đổi đường dẫn) thì cập nhật tests/routes.txt cho khớp — để không ai lỡ tay làm mất route."""
    expected = ROUTES_FILE.read_text("utf-8").split("\n")
    assert current_routes() == sorted(line for line in expected if line)


def test_foreign_origin_is_blocked(client):
    res = client.post("/api/stop", headers={"origin": "https://evil.example"})
    assert res.status_code == 403


def test_value_error_becomes_400(client):
    res = client.get("/api/finetune/prompts/khong-co/abc")
    assert res.status_code == 400
    assert "error" in res.json()


def test_finetune_tasks_lists_decider(client):
    res = client.get("/api/finetune/tasks")
    assert res.status_code == 200
    assert any(t["id"] == "decider" for t in res.json()["tasks"])


def test_prompt_crud_via_api(client):
    created = client.post("/api/finetune/prompts/gate", json={"name": "thu_api", "content": {"text": "luật"}, "note": ""})
    assert created.status_code == 200, created.text
    item = created.json()
    assert item["version"] == 1

    listed = client.get("/api/finetune/prompts/gate").json()
    assert any(p["id"] == item["id"] for p in listed["items"])

    assert client.put(f"/api/finetune/prompts/gate/{item['id']}/labels/prod", json={"version": 1}).status_code == 200
    versions = client.get(f"/api/finetune/prompts/gate/{item['id']}/versions").json()["versions"]
    assert versions == [{"version": 1, "note": "", "labels": ["prod"]}]
    assert client.get(f"/api/finetune/prompts/gate/{item['id']}?label=prod").json()["version"] == 1
    assert client.delete(f"/api/finetune/prompts/gate/{item['id']}").status_code == 200
