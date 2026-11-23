"""Fine-tune: prompt, dataset, build, job train, eval."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..core import health
from ..finetune import builds as ft_builds
from ..finetune import datasets as ft_datasets
from ..finetune import evals as ft_evals
from ..finetune import jobs as ft_jobs
from ..finetune import params_yaml as ft_params_yaml
from ..finetune.gpu_queue import QUEUE as FT_QUEUE
from ..finetune.prompts import PROMPTS as FT_PROMPTS
from ..finetune.sources import SOURCES as FT_SOURCES
from ..finetune.store import save_upload
from ..finetune.tasks import TASKS as FT_TASKS
from ..finetune.tasks import task_of
from ..registry import REG

router = APIRouter()


# ---------------------------------------------------------------- fine-tune


class UploadBody(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    text: str


class DatasetBody(BaseModel):
    task: str
    source: str
    name: str = ""
    upload_id: str
    mapping: dict[str, str]


class BuildBody(BaseModel):
    task: str
    name: str = ""
    note: str = ""
    system: dict | None = None  # {id, version} hoặc {id, label}
    intents: list[dict] = Field(default_factory=list)
    gates: list[dict] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    eval_datasets: list[str] = Field(default_factory=list)
    dedupe: bool = True
    val_percent: float = 10
    split_seed: int = 42


class JobBody(BaseModel):
    name: str = ""
    build_id: str
    base_model: str
    params: dict = Field(default_factory=dict)


class EvalBody(BaseModel):
    name: str = ""
    build_id: str
    base_model: str
    adapters: list[str] = Field(default_factory=list)


class ParamsBody(BaseModel):
    task: str
    params: dict = Field(default_factory=dict)


class ParamsYamlBody(BaseModel):
    task: str
    text: str = Field(default="", max_length=20000)


class LabelBody(BaseModel):
    version: int = Field(ge=1)


class PromptBody(BaseModel):
    name: str = ""
    note: str = ""
    content: dict


@router.get("/api/finetune/tasks")
def finetune_tasks():
    models = REG.public()
    return {"tasks": [t.public(models) for t in FT_TASKS.values()]}


@router.post("/api/finetune/uploads")
def finetune_upload(body: UploadBody):
    return save_upload(body.name, body.text)


@router.get("/api/finetune/queue")
def finetune_queue():
    return FT_QUEUE.summary()


@router.get("/api/finetune/sources")
def finetune_sources():
    return {"sources": [s.public() for s in FT_SOURCES.values()]}


@router.get("/api/finetune/prompts-status")
def finetune_prompts_status():
    """Kho prompt đang dùng: langfuse (đồng bộ hai chiều) hay local."""
    return FT_PROMPTS.status()


@router.get("/api/finetune/prompts/{kind}")
def finetune_prompts(kind: str):
    return {"items": FT_PROMPTS.list(kind)}


@router.post("/api/finetune/prompts/{kind}")
def finetune_prompt_create(kind: str, body: PromptBody):
    p = FT_PROMPTS.save(kind, None, body.name, body.content, body.note)
    return {"message": f"Đã lưu \"{p['name']}\" v{p['version']}.", "id": p["id"], "version": p["version"]}


@router.get("/api/finetune/prompts/{kind}/{item_id}")
def finetune_prompt(kind: str, item_id: str, version: int | None = None, label: str | None = None):
    return FT_PROMPTS.get(kind, item_id, version, label)


@router.get("/api/finetune/prompts/{kind}/{item_id}/versions")
def finetune_prompt_versions(kind: str, item_id: str):
    return {"versions": FT_PROMPTS.versions(kind, item_id)}


@router.put("/api/finetune/prompts/{kind}/{item_id}/labels/{label}")
def finetune_prompt_label(kind: str, item_id: str, label: str, body: LabelBody):
    FT_PROMPTS.set_label(kind, item_id, body.version, label)
    return {"message": f"Đã gắn nhãn {label} cho v{body.version}."}


@router.post("/api/finetune/prompts/{kind}/{item_id}")
def finetune_prompt_version(kind: str, item_id: str, body: PromptBody):
    p = FT_PROMPTS.save(kind, item_id, body.name, body.content, body.note)
    return {"message": f"Đã lưu phiên bản mới: \"{p['name']}\" v{p['version']}.", "id": p["id"], "version": p["version"]}


@router.delete("/api/finetune/prompts/{kind}/{item_id}")
def finetune_prompt_delete(kind: str, item_id: str):
    FT_PROMPTS.delete(kind, item_id)
    return {"message": "Đã xoá. Các build đã dùng nó vẫn giữ nguyên nội dung prompt."}


@router.get("/api/finetune/datasets")
def finetune_datasets():
    return {"datasets": ft_datasets.DATASETS.all()}


@router.post("/api/finetune/datasets")
def finetune_dataset_create(body: DatasetBody):
    m = ft_datasets.create(body.model_dump())
    return {"message": f"Đã lưu dataset \"{m['name']}\" ({m['rows']} dòng).", "id": m["id"]}


@router.get("/api/finetune/datasets/{item_id}")
def finetune_dataset(item_id: str):
    return ft_datasets.detail(item_id)


@router.delete("/api/finetune/datasets/{item_id}")
def finetune_dataset_delete(item_id: str):
    return {"message": ft_datasets.delete(item_id)}


@router.get("/api/finetune/builds")
def finetune_builds():
    return {"builds": ft_builds.BUILDS.all()}


@router.post("/api/finetune/builds/preview")
def finetune_build_preview(body: BuildBody):
    return ft_builds.preview(body.model_dump())


@router.post("/api/finetune/builds")
def finetune_build_create(body: BuildBody):
    m = ft_builds.create(body.model_dump())
    return {"message": f"Đã build \"{ft_builds.title(m)}\".", "id": m["id"]}


@router.get("/api/finetune/builds/{item_id}")
def finetune_build(item_id: str):
    return ft_builds.detail(item_id)


@router.delete("/api/finetune/builds/{item_id}")
def finetune_build_delete(item_id: str):
    return {"message": ft_builds.delete(item_id)}


@router.get("/api/finetune/jobs")
def finetune_jobs():
    return {"jobs": ft_jobs.list_all()}


@router.post("/api/finetune/jobs")
def finetune_job_create(body: JobBody):
    m = ft_jobs.create(body.model_dump())
    note = " Model đang chạy sẽ được tắt khi tới lượt." if health() is not None else ""
    return {"message": f"Đã thêm \"{m['name']}\" vào hàng đợi.{note}", "id": m["id"]}


@router.get("/api/finetune/jobs/{item_id}")
def finetune_job(item_id: str, since: int = 0):
    return ft_jobs.detail(item_id, max(0, since))


@router.post("/api/finetune/jobs/{item_id}/stop")
def finetune_job_stop(item_id: str):
    return {"message": ft_jobs.stop(item_id)}


@router.post("/api/finetune/jobs/{item_id}/register")
def finetune_job_register(item_id: str):
    res = ft_jobs.register_model(item_id)
    return {"message": f"Đã thêm {res['id']} vào danh sách model.", **res}


@router.delete("/api/finetune/jobs/{item_id}")
def finetune_job_delete(item_id: str):
    return {"message": ft_jobs.delete(item_id)}


@router.get("/api/finetune/evals")
def finetune_evals():
    return {"evals": ft_evals.list_all()}


@router.get("/api/finetune/evals/choices")
def finetune_eval_choices(task: str):
    return ft_evals.model_choices(task_of(task))


@router.post("/api/finetune/evals")
def finetune_eval_create(body: EvalBody):
    m = ft_evals.create(body.model_dump())
    return {"message": f"Đã thêm lần eval \"{m['name']}\" vào hàng đợi.", "id": m["id"]}


@router.get("/api/finetune/evals/{item_id}")
def finetune_eval(item_id: str):
    return ft_evals.detail(item_id)


@router.post("/api/finetune/evals/{item_id}/stop")
def finetune_eval_stop(item_id: str):
    return {"message": ft_evals.stop(item_id)}


@router.delete("/api/finetune/evals/{item_id}")
def finetune_eval_delete(item_id: str):
    return {"message": ft_evals.delete(item_id)}


@router.post("/api/finetune/params/yaml/dump")
def finetune_params_dump(body: ParamsBody):
    """Tham số trên form -> YAML có chú thích."""
    return {"text": ft_params_yaml.dump(task_of(body.task), body.params)}


@router.post("/api/finetune/params/yaml/parse")
def finetune_params_parse(body: ParamsYamlBody):
    """YAML -> tham số đã kiểm tra (cùng luật với form)."""
    return {"params": ft_params_yaml.parse(task_of(body.task), body.text)}
