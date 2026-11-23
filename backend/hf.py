"""Đọc thông tin model trên Hugging Face và đánh giá có vừa GPU không."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

from .config import GPU_VRAM_GB, IMAGE_MODES, MODES, RUNTIME_NAMES, TTS_MODES
from .core import hf_token


def parse_repo(text: str) -> str | None:
    text = text.strip()
    m = re.search(r"huggingface\.co/(?:models/)?([\w.-]+/[\w.-]+)", text)
    if m:
        return m.group(1)
    m = re.fullmatch(r"([\w.-]+/[\w.-]+)", text)
    return m.group(1) if m else None


def _hf_get(url: str, timeout: float = 20):
    token = hf_token()
    headers = {"authorization": f"Bearer {token}"} if token else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return json.load(r)


def hf_api(repo: str) -> dict:
    return _hf_get(f"https://huggingface.co/api/models/{repo}?blobs=true")


def whoami(token: str) -> dict:
    req = urllib.request.Request("https://huggingface.co/api/whoami-v2", headers={"authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


SEARCH_KINDS = {"text-generation", "feature-extraction", "sentence-similarity", "text-classification"}


def search(query: str, kind: str = "", limit: int = 20) -> list[dict]:
    """Tìm model trên Hugging Face (sắp theo lượt tải)."""
    params = {"search": query, "limit": str(limit), "sort": "downloads", "direction": "-1"}
    if kind == "gguf":
        params["filter"] = "gguf"
    elif kind in SEARCH_KINDS:
        params["pipeline_tag"] = kind
    try:
        data = _hf_get("https://huggingface.co/api/models?" + urllib.parse.urlencode(params))
    except urllib.error.HTTPError as e:
        raise ValueError(f"Hugging Face báo lỗi {e.code}.")
    except OSError as e:
        raise ValueError(f"Không kết nối được Hugging Face: {e}")
    return [{"repo": d.get("id") or d.get("modelId"), "downloads": d.get("downloads", 0), "likes": d.get("likes", 0),
             "pipeline_tag": d.get("pipeline_tag"), "gguf": "gguf" in (d.get("tags") or []),
             "updated": d.get("lastModified")} for d in data]


QUANT_PREF = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "Q6_K", "Q8_0"]


def fit(weights: int, runtime: str) -> tuple[str, str]:
    gb = weights / 1e9
    if runtime == "vllm":
        ok, tight = GPU_VRAM_GB * 0.85 - 1.5, GPU_VRAM_GB * 0.85
    elif runtime == "llamacpp":
        ok, tight = GPU_VRAM_GB - 2, GPU_VRAM_GB - 0.6
    else:
        ok, tight = GPU_VRAM_GB - 3, GPU_VRAM_GB - 1
    if gb <= ok:
        return "ok", f"Vừa VRAM ({gb:.1f} GB trọng số / {GPU_VRAM_GB:.0f} GB)."
    if gb <= tight:
        return "tight", f"Sát giới hạn VRAM ({gb:.1f} GB trọng số), phải giảm độ dài ngữ cảnh."
    if runtime == "llamacpp":
        return "too_big", f"Lớn hơn VRAM ({gb:.1f} GB): chạy được nhưng một phần nằm trên CPU, chậm hơn nhiều."
    return "too_big", f"Không vừa {GPU_VRAM_GB:.0f} GB VRAM ({gb:.1f} GB trọng số). Hãy tìm bản GGUF hoặc AWQ nhỏ hơn."


def gguf_files(siblings: list[dict]) -> list[dict]:
    """Các file GGUF (gộp file chia nhỏ -0000N-of-0000M thành một lựa chọn)."""
    groups: dict[str, dict] = {}
    for s in siblings:
        name = s["rfilename"]
        if not name.lower().endswith(".gguf") or "mmproj" in name.lower():
            continue
        m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", name)
        key = m.group(1) if m else name
        g = groups.setdefault(key, {"name": name, "size": 0, "parts": 0, "pattern": None})
        g["size"] += s.get("size") or 0
        g["parts"] += 1
        if m:
            g["pattern"] = m.group(1) + "-*-of-*.gguf"
            if m.group(2) == "00001":
                g["name"] = name
    files = sorted(groups.values(), key=lambda g: g["name"])

    def rank(g):
        up = g["name"].upper()
        return next((i for i, p in enumerate(QUANT_PREF) if p in up), len(QUANT_PREF))

    if files:
        best = min(files, key=rank)
        for g in files:
            g["recommended"] = g is best
            g["fit"], g["fit_text"] = fit(g["size"], "llamacpp")
    return files


def guess_mode(repo: str, tag: str | None, tags: list[str], lib: str) -> str:
    low = repo.lower()
    if "rerank" in low or (tag == "text-classification" and "rerank" in " ".join(tags).lower()):
        return "rerank"
    if (tag in ("feature-extraction", "sentence-similarity") or lib == "sentence-transformers"
            or re.search(r"embed|(^|[/-])(bge|e5|gte)-", low)):
        return "embed"
    return "chat"


def _base(card: dict) -> str | None:
    base = card.get("base_model")
    return base[0] if isinstance(base, list) else base


def inspect(text: str) -> dict:
    repo = parse_repo(text)
    if not repo:
        raise ValueError("Không nhận ra link. Dán link dạng huggingface.co/<tổ chức>/<tên model>.")
    try:
        info = hf_api(repo)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise ValueError("Model này cần token Hugging Face (gated). Dán token vào trang Cài đặt rồi thử lại.")
        if e.code == 404:
            raise ValueError(f"Không tìm thấy model {repo} trên Hugging Face.")
        raise ValueError(f"Hugging Face báo lỗi {e.code}.")
    except OSError as e:
        raise ValueError(f"Không kết nối được Hugging Face: {e}")

    repo = info.get("id", repo)
    sib = info.get("siblings", [])
    names = [s["rfilename"] for s in sib]
    tags = info.get("tags", [])
    lib = info.get("library_name") or ""
    card = info.get("cardData") or {}
    res = {"repo": repo, "pipeline_tag": info.get("pipeline_tag"), "gated": bool(info.get("gated")),
           "license": card.get("license"), "downloads": info.get("downloads"), "likes": info.get("likes"),
           "updated": info.get("lastModified"), "gguf_files": [], "warnings": []}

    if lib == "strands-decider" or "strands-decider" in tags:
        base = _base(card)
        base_size = 0
        if base:
            try:
                bi = hf_api(base)
                base_size = sum(s.get("size") or 0 for s in bi.get("siblings", []) if s["rfilename"].endswith(".safetensors"))
            except (OSError, ValueError):
                pass
        own = sum(s.get("size") or 0 for s in sib if not s["rfilename"].startswith(("eval/", "training/")))
        res.update(runtime="decider", base_model=base, size=own + base_size, weights=base_size)
    elif res["pipeline_tag"] == "text-to-speech":  # kiểm tra trước GGUF: repo giọng nói có thể kèm file .gguf
        total = sum(s.get("size") or 0 for s in sib)
        res.update(runtime="vieneu", mode="tts", size=total, weights=total, has_safetensors=True)
        res["runtimes"] = [{"id": "vieneu", "name": "VieNeu-TTS (giọng nói)",
                            "note": "Runtime duy nhất chạy model giọng nói (VieNeu-TTS). Hỗ trợ clone giọng từ file mẫu."}]
        res["modes"] = TTS_MODES
    elif files := gguf_files(sib):
        rec = next(f for f in files if f.get("recommended"))
        res.update(runtime="llamacpp", mode=guess_mode(repo, res["pipeline_tag"], tags, lib), gguf_files=files,
                   size=rec["size"], weights=rec["size"], base_model=_base(card))
        # vLLM 0.34 đã bỏ bộ nạp GGUF, nên GGUF chỉ chạy bằng llama.cpp.
        res["runtimes"] = [{"id": "llamacpp", "name": "llama.cpp",
                            "note": "Runtime duy nhất chạy GGUF (vLLM 0.34 đã bỏ hỗ trợ GGUF). "
                                    "Muốn chạy bằng vLLM thì dán link bản safetensors của model."}]
    elif "model_index.json" in names:  # pipeline diffusers (model sinh ảnh)
        # Chỉ lấy file trong thư mục con (unet/, vae/…); file gộp ở thư mục gốc là bản khác, không dùng chung
        fp16 = [s for s in sib if "/" in s["rfilename"] and s["rfilename"].endswith("fp16.safetensors")]
        if not fp16:
            raise ValueError("Repo sinh ảnh này không có bản fp16 (*fp16.safetensors): nền tảng chưa hỗ trợ.")
        weights = sum(s.get("size") or 0 for s in fp16)
        res.update(runtime="diffusers", mode="image", size=weights, weights=weights, has_safetensors=True)
        res["runtimes"] = [{"id": "diffusers", "name": "Diffusers (sinh ảnh)",
                            "note": "Runtime duy nhất chạy model sinh ảnh (diffusers, bản fp16)."}]
        res["modes"] = IMAGE_MODES
    elif any(n.endswith((".safetensors", ".bin")) and "/" not in n for n in names):
        has_st = any(n.endswith(".safetensors") for n in names)
        ext = ".safetensors" if has_st else ".bin"
        if "adapter_config.json" in names and not any(n.startswith(("model", "pytorch_model")) for n in names):
            raise ValueError("Đây là adapter LoRA, cần model gốc đi kèm; nền tảng chưa hỗ trợ loại này.")
        weights = sum(s.get("size") or 0 for s in sib if s["rfilename"].endswith(ext) and "/" not in s["rfilename"])
        mode = guess_mode(repo, res["pipeline_tag"], tags, lib)
        res.update(runtime="vllm", mode=mode, size=weights, weights=weights, has_safetensors=has_st)
        res["runtimes"] = [{"id": "vllm", "name": "vLLM", "note": "Runtime duy nhất đọc được safetensors/bin. "
                                                                   "Muốn chạy bằng llama.cpp thì dán link bản GGUF của model."}]
        tag = res["pipeline_tag"]
        if mode == "chat" and tag not in (None, "text-generation", "image-text-to-text", "conversational"):
            res["warnings"].append(f"Loại model là \"{tag}\", có thể không chat được.")
    else:
        raise ValueError("Không có file safetensors, .bin hoặc GGUF: nền tảng chưa chạy được model này.")
    res.setdefault("mode", "decide" if res["runtime"] == "decider" else "chat")
    if res["runtime"] == "decider":
        res["runtimes"] = [{"id": "decider", "name": "Strands Decider", "note": "Chỉ runtime này chạy được model ra quyết định."}]
        res["modes"] = [{"id": "decide", "name": "Ra quyết định"}]
    else:
        res.setdefault("modes", MODES)

    res["fit"], res["fit_text"] = fit(res["weights"], res["runtime"])
    if res["gated"] and not hf_token():
        res["warnings"].append("Model gated: cần token Hugging Face và đã bấm đồng ý điều khoản trên trang model.")
    res["runtime_name"] = RUNTIME_NAMES[res["runtime"]]
    return res
