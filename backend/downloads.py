"""Tải model từ Hugging Face vào WSL, theo dõi tiến trình, huỷ, tải lại, xoá."""

from __future__ import annotations

import re
import threading
import time

from .config import DEFAULT_OPTIONS, HF, HOME, MODELS_DIR
from .core import LOG, popen_wsl, q, wsl
from .hf import fit
from .registry import REG


def start_download(info: dict, gguf_name: str | None, runtime: str | None = None, mode: str | None = None) -> dict:
    allowed = [r["id"] for r in info.get("runtimes", [])] or [info["runtime"]]
    runtime = runtime if runtime in allowed else info["runtime"]
    modes = [x["id"] for x in info.get("modes", [])]
    mode = mode if mode in modes else info.get("mode", "chat")
    repo = info["repo"]
    base_id = repo.split("/")[-1]
    entry_file = None
    size = info["size"]
    include: list[str] = []
    if info.get("gguf_files"):
        files = info["gguf_files"]
        f = next((g for g in files if g["name"] == gguf_name), None) or next(g for g in files if g.get("recommended"))
        if runtime == "vllm" and f["parts"] > 1:
            raise ValueError("vLLM không chạy được GGUF chia nhiều file. Chọn bản khác hoặc dùng llama.cpp.")
        entry_file, size = f["name"], f["size"]
        include = [f["pattern"] or f["name"]]
        quant = re.search(r"(I?Q\d[\w]*|F16|BF16|F32)", f["name"], re.I)
        if quant:
            base_id += "-" + quant.group(1)
        if runtime == "vllm":
            base_id += "-vllm"
        fit_level, fit_text = fit(size, runtime)
        if fit_level == "too_big" and runtime != "llamacpp":
            raise ValueError(fit_text)
    elif info["fit"] == "too_big" and info["runtime"] != "llamacpp":
        raise ValueError(info["fit_text"])
    with REG.lock:
        if any(m["repo"] == repo and m.get("file") == entry_file and m["runtime"] == runtime for m in REG.models):
            raise ValueError("Model này (cùng bản, cùng runtime) đã có trong danh sách.")
        mid = REG.unique_id(base_id)
        m = {"id": mid, "repo": repo, "runtime": runtime, "path": f"{MODELS_DIR}/{mid}", "file": entry_file,
             "include": include, "size": size, "status": "downloading", "progress": 0.0,
             "options": dict(DEFAULT_OPTIONS[runtime]), "managed_path": True, "base_model": info.get("base_model"),
             "mode": mode, "has_safetensors": info.get("has_safetensors", True),
             "added": time.strftime("%Y-%m-%d %H:%M")}
        REG.models.append(m)
        REG.save()
    _spawn(m)
    return m


def retry_download(mid: str) -> dict:
    m = REG.require(mid)
    if m["status"] != "error" or not m.get("managed_path"):
        raise ValueError("Chỉ tải lại được model đang lỗi do nền tảng tải về.")
    m.update(status="downloading", progress=0.0)
    m.pop("error", None)
    m.pop("_cancelled", None)
    REG.save()
    _spawn(m)
    return m


def _spawn(m: dict) -> None:
    threading.Thread(target=_worker, args=(m,), daemon=True).start()


def _script(m: dict) -> str:
    path, repo = m["path"], m["repo"]
    steps = []
    if m.get("file"):  # GGUF (llama.cpp, hoặc vLLM kèm tokenizer của model gốc)
        include = m.get("include") or [m["file"]]
        steps.append(f"{HF} download {q(repo)} {' '.join('--include ' + q(i) for i in include)} --local-dir {q(path)}")
        steps.append(f"test -f {q(path + '/' + m['file'])} || {{ echo 'Thiếu file GGUF sau khi tải'; exit 1; }}")
        if m["runtime"] == "vllm":
            pats = ["*.json", "*.jinja", "tokenizer*", "*.model", "*.txt", "*.tiktoken"]
            steps.append(f"{HF} download {q(m['base_model'])} {' '.join('--include ' + q(p) for p in pats)} "
                         f"--local-dir {q(path + '/hf')}")
            steps.append(f"test -f {q(path + '/hf/config.json')} || "
                         f"{{ echo 'Không tải được config/tokenizer của model gốc'; exit 1; }}")
    elif m["runtime"] == "vllm":
        skip = ["*.pth", "*.pt", "original/*", "onnx/*", "openvino/*", "imgs/*", "*.jpg", "*.png", "*.webp",
                "*.gguf", "*.onnx", "*.msgpack", "*.h5", "*.ot"]
        if m.get("has_safetensors", True):
            skip.append("*.bin")
        steps.append(f"{HF} download {q(repo)} --local-dir {q(path)} {' '.join('--exclude ' + q(s) for s in skip)}")
        steps.append(f"test -f {q(path + '/config.json')} || {{ echo 'Thiếu config.json sau khi tải'; exit 1; }}")
    elif m["runtime"] == "vieneu":  # cache của thư viện nằm trong thư mục model, để xóa model là dọn sạch
        steps.append(f"env HF_HUB_CACHE={q(path + '/hf')} {HF} download {q(repo)}")
        steps.append(f"test -d {q(path + '/hf')} || {{ echo 'Không tải được model giọng nói'; exit 1; }}")
    elif m["runtime"] == "diffusers":  # chỉ lấy bản fp16 và file cấu hình, bỏ bản fp32 để đỡ tốn dung lượng
        inc = ["*.json", "*.txt", "*.model", "*/*fp16.safetensors"]
        steps.append(f"{HF} download {q(repo)} --local-dir {q(path)} {' '.join('--include ' + q(p) for p in inc)}")
        steps.append(f"test -f {q(path + '/model_index.json')} || {{ echo 'Thiếu model_index.json sau khi tải'; exit 1; }}")
    else:  # decider: adapter + model gốc vào cache của Hugging Face
        steps.append(f"{HF} download {q(repo)} --local-dir {q(path)} --exclude 'eval/*' --exclude 'training/*'")
        base = m.get("base_model")
        if base:
            rev = (f"$(python3 -c \"import json;print(json.load(open('{path}/provenance.json'))"
                   f".get('base_model_revision') or 'main')\" 2>/dev/null || echo main)")
            steps.append(f"{HF} download {q(base)} --revision {rev} >/dev/null")
    # Tắt Xet: Xet tải vào bộ nhớ đệm riêng rồi mới ghi file khi xong, nên không đo được tiến trình.
    return (f"set -e; export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1; "
            f"mkdir -p {q(path)}; " + "; ".join(steps) + "; echo __DONE__")


def _worker(m: dict) -> None:
    path, repo, src = m["path"], m["repo"], f"tải {m['id']}"
    LOG.add(f"=== Tải {repo} → {m['id']} ===", src)
    proc = popen_wsl(_script(m))
    m["_proc"] = proc
    out_lines: list[str] = []

    def reader():
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            out_lines.append(line)
            if line and not line.startswith("Hint") and "__DONE__" not in line:
                LOG.add(line, src)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    base_cache = ""
    if m["runtime"] == "decider" and m.get("base_model"):
        base_cache = f" {HOME}/.cache/huggingface/hub/models--{m['base_model'].replace('/', '--')}"
    last_bytes, last_t = None, time.time()
    while proc.poll() is None:
        time.sleep(2)
        _, out = wsl(f"du -sbL {q(path)}{base_cache} 2>/dev/null | awk '{{s+=$1}} END {{print s}}'", 10)
        try:
            done = int(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            continue
        now = time.time()
        if last_bytes is not None and now > last_t:
            speed = max(0.0, (done - last_bytes) / (now - last_t))
            # làm mượt tốc độ
            m["speed"] = speed if not m.get("speed") else 0.7 * m["speed"] + 0.3 * speed
        last_bytes, last_t = done, now
        m["downloaded"] = done
        m["progress"] = min(0.99, done / max(1, m["size"]))
    t.join(5)
    m.pop("_proc", None)
    m.pop("speed", None)
    if m.get("_cancelled"):
        return
    if proc.returncode == 0 and any("__DONE__" in x for x in out_lines):
        m["status"], m["progress"] = "ready", 1.0
        m.pop("downloaded", None)
        LOG.add(f"=== Đã tải xong {m['id']} ===", src)
    else:
        tail = [x for x in out_lines if x.strip()][-3:]
        m["status"] = "error"
        m["error"] = " ".join(tail)[-400:] or f"Lỗi tải (mã {proc.returncode})"
        if any("401" in x or "403" in x or "gated" in x.lower() for x in out_lines):
            m["error"] = "Không có quyền tải: model gated, cần token Hugging Face và đồng ý điều khoản trên trang model."
        LOG.add(f"=== Tải {m['id']} thất bại ===", src)
    REG.save()


def delete_model(mid: str, stop_server) -> str:
    m = REG.require(mid)
    stop_server(mid)
    proc = m.get("_proc")
    if proc and proc.poll() is None:
        m["_cancelled"] = True
        proc.kill()
        wsl(f"pkill -f {q('download ' + m['repo'])}", 10)
    msg = "Đã xoá khỏi danh sách (file gốc được giữ nguyên)."
    if m.get("managed_path") and m["path"].startswith(MODELS_DIR + "/"):
        wsl(f"rm -rf {q(m['path'])}", 120)
        msg = "Đã xoá model và file của nó."
    with REG.lock:
        REG.models.remove(m)
        REG.save()
    LOG.add(f"=== Xoá {mid} ===")
    return msg
