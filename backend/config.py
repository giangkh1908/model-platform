"""Cấu hình chung: đường dẫn, cổng, runtime."""

from __future__ import annotations

import subprocess
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
# Nạp ngay khi import config (module được import đầu tiên), trước khi hf_token đọc biến.
# Biến đã đặt sẵn trong Windows được ưu tiên hơn .env.
load_dotenv(ENV_FILE, override=False)
DATA = ROOT / "data"
REGISTRY_FILE = DATA / "models.json"
SETTINGS_FILE = DATA / "settings.json"
FRONTEND_DIST = ROOT / "frontend" / "dist"

PANEL_HOST, PANEL_PORT = "127.0.0.1", 8100
DEV_ORIGINS = ["http://127.0.0.1:5173", "http://localhost:5173"]
MODEL_PORT = 8000
MODEL_URL = f"http://127.0.0.1:{MODEL_PORT}"
GPU_VRAM_GB = 12.0

DISTRO = "Ubuntu"
HOME = "/home/giang"
MODELS_DIR = f"{HOME}/model-platform/models"
HF = f"{HOME}/strands-decider/.venv/bin/hf"
DECIDER_BIN = f"{HOME}/strands-decider/.venv/bin/strands-decider"
VLLM_BIN = f"{HOME}/model-platform/runtimes/vllm/.venv/bin/vllm"
LLAMA_DIR = f"{HOME}/model-platform/runtimes/llama.cpp/current/llama-b12600"
DIFFUSERS_PY = f"{HOME}/model-platform/runtimes/diffusers/.venv/bin/python"


def wsl_path(path: Path) -> str:
    """Đường dẫn Windows (ổ C:) nhìn từ trong WSL (/mnt/c/...)."""
    drive, rest = path.resolve().as_posix().split(":", 1)
    return f"/mnt/{drive.lower()}{rest}"


IMAGE_SERVER = wsl_path(ROOT / "runtime" / "image_server.py")  # đường dẫn trong WSL tới máy chủ sinh ảnh
TTS_SERVER = wsl_path(ROOT / "runtime" / "tts_server.py")  # máy chủ giọng nói VieNeu-TTS
FINETUNE_DIR = DATA / "finetune"  # job, dữ liệu đã chia, chỉ số (phía Windows, backend đọc trực tiếp)
FINETUNE_RUN = wsl_path(ROOT / "runtime" / "finetune" / "run.py")
FINETUNE_OUT = f"{HOME}/model-platform/finetune"  # checkpoint sau fine-tune (phía WSL, đọc ghi nhanh hơn /mnt/c)
DECIDER_PY = f"{HOME}/strands-decider/.venv/bin/python"
VIENEU_PY = f"{HOME}/model-platform/runtimes/tts/.venv/bin/python"
SERVER_PATTERNS = ["strands-decider serve", "vllm serve", "llama-server", "VLLM::EngineCore", "image_server.py", "tts_server.py"]

RUNTIME_NAMES = {"decider": "Strands Decider", "vllm": "vLLM", "llamacpp": "llama.cpp", "diffusers": "Diffusers (sinh ảnh)",
                 "vieneu": "VieNeu-TTS (giọng nói)"}
RUNTIME_BINS = {"decider": DECIDER_BIN, "vllm": VLLM_BIN, "llamacpp": f"{LLAMA_DIR}/llama-server", "diffusers": DIFFUSERS_PY,
                "vieneu": VIENEU_PY}
DEFAULT_OPTIONS = {
    "decider": {"max_concurrency": 1},
    "vllm": {"ctx": 8192, "gpu_util": 0.9},
    "llamacpp": {"ctx": 8192, "parallel": 4},
    "diffusers": {},
    "vieneu": {},
}
MODES = [
    {"id": "chat", "name": "Chat (sinh chữ)"},
    {"id": "embed", "name": "Embedding (vector)"},
    {"id": "rerank", "name": "Reranker (xếp hạng tài liệu)"},
]
IMAGE_MODES = [{"id": "image", "name": "Sinh ảnh (text → ảnh)"}]
TTS_MODES = [{"id": "tts", "name": "Giọng nói (văn bản → tiếng nói)"}]

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ROOT_PID_ENV = "MODEL_PLATFORM_ROOT_PID"  # PID của tiến trình chạy platform_server.py (tiến trình giám sát khi --reload)
