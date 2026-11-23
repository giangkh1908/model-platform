"""Model registry: xem model đã xuất bản, gán alias, xuất bản job train."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..finetune import publish as ft_publish
from ..model_registry import ALIASES, REGISTRY

router = APIRouter()


class PublishBody(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    description: str = Field(default="", max_length=500)


class AliasBody(BaseModel):
    version: str = Field(min_length=1, max_length=20)


@router.get("/api/registry/status")
def registry_status():
    return {**REGISTRY.status(), "aliases": list(ALIASES)}


@router.get("/api/registry/models")
def registry_models():
    return {"models": REGISTRY.models()}


@router.put("/api/registry/models/{name}/aliases/{alias}")
def registry_set_alias(name: str, alias: str, body: AliasBody):
    REGISTRY.set_alias(name, alias, body.version)
    return {"message": f"Đã gán {alias} cho {name} phiên bản {body.version}."}


@router.delete("/api/registry/models/{name}/aliases/{alias}")
def registry_delete_alias(name: str, alias: str):
    REGISTRY.delete_alias(name, alias)
    return {"message": f"Đã bỏ {alias} của {name}."}


@router.get("/api/finetune/jobs/{item_id}/publish")
def publish_defaults(item_id: str):
    """Tên mặc định và lần eval sẽ đính kèm, để hiện trước khi bấm xuất bản."""
    job = ft_publish.jobs.JOBS.get(item_id)
    found = ft_publish.latest_eval(item_id)
    return {"name": ft_publish.default_name(job), "eval": found[0]["name"] if found else None,
            "published": job.get("published") or []}


@router.post("/api/finetune/jobs/{item_id}/publish")
def publish_job(item_id: str, body: PublishBody):
    result = ft_publish.publish(item_id, body.name, body.description)
    return {"message": f"Đã xuất bản {result['name']} phiên bản {result['version']}.", **result}
