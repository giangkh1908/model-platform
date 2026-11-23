"""Fine-tune Strands Decider: train tiếp LoRA + head của checkpoint có sẵn trên dữ liệu {text, label}.

Mỗi mẫu thành một câu hỏi `choice`: state = text, lựa chọn = toàn bộ nhãn của job (kèm mô tả nếu có).
Lúc train xáo thứ tự lựa chọn (như công thức gốc của decider) để head không học thuộc vị trí.
Kết quả lưu đúng định dạng `strands-decider serve` đọc được, kèm finetune_meta.json (nhãn, câu hỏi, ngưỡng).
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import classify_metrics as cm
from common import RunFiles, Task, write_json

META_FILE = "finetune_meta.json"
COPY_FROM_SOURCE = ("chat_template.jinja", "LICENSE.md")


class DeciderTask(Task):
    def load(self, path: str, config: dict, trainable: bool) -> None:
        import torch
        from strands_decider.modeling import StrandsDeciderModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = StrandsDeciderModel.load(path, trainable=trainable).to(self.device)
        self.tok = self.model.tokenizer
        tc = config["task_config"]
        self.labels: list[str] = tc["labels"]
        self.instructions: str = tc["instructions"]
        self.descriptions: dict = tc.get("descriptions") or {}
        self.max_length = int(config["params"].get("max_length") or self.model.config.max_length)
        self.eval_batch = int(config["params"].get("eval_batch_size", 16))
        self.label_smoothing = float(config["params"].get("label_smoothing", 0.0))

    def enable_gradient_checkpointing(self) -> None:
        self.model.torso.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def param_groups(self, lr: float, params: dict) -> list[dict]:
        head = [p for p in self.model.head.parameters() if p.requires_grad]
        torso = [p for p in self.model.torso.parameters() if p.requires_grad]
        return [{"params": torso, "lr": lr}, {"params": head, "lr": lr * params["head_lr_ratio"]}]

    # ------------------------------------------------------------ mã hoá

    def _encode(self, text: str, order: list[int]) -> tuple[list[int], list[int], list[str]]:
        from strands_decider.prompting import fit_window, option_token_index, render_question, render_state
        from strands_decider.schema import ChoiceQuestion

        question = ChoiceQuestion(instructions=self.instructions,
                                  criteria={k: self.descriptions.get(k) or None for k in self.labels})
        rq = render_question(question, option_order=order)
        s, q, offsets = fit_window(self.tok.backend_tokenizer, render_state(text), [rq.text], self.max_length)
        opt = [len(s) + i for i in option_token_index(offsets[0], rq.option_spans, 0)]
        return s + q[0], opt, list(rq.slot_labels)

    def _batch(self, encoded: list[tuple[list[int], list[int]]]):
        import torch

        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        width = max(len(ids) for ids, _ in encoded)
        ids = torch.tensor([x + [pad] * (width - len(x)) for x, _ in encoded], device=self.device)
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in encoded], device=self.device)
        opt_idx = torch.tensor([o for _, o in encoded], device=self.device)
        n_slots = torch.full((len(encoded),), len(self.labels), device=self.device)
        return ids, mask, opt_idx, n_slots

    # ------------------------------------------------------------ train

    def loss(self, rows: list[dict], rng: random.Random):
        import torch

        encoded, gold = [], []
        for r in rows:
            order = list(range(len(self.labels)))
            rng.shuffle(order)
            ids, opt, slots = self._encode(r["text"], order)
            encoded.append((ids, opt))
            gold.append(slots.index(r["label"]))
        ids, mask, opt_idx, n_slots = self._batch(encoded)
        labels = torch.tensor(gold, device=self.device)
        ones = torch.ones(len(rows), device=self.device)
        kwargs = {"labels": labels}
        if self.label_smoothing > 0:
            n = len(self.labels)
            dist = torch.full((len(rows), n), self.label_smoothing / n, device=self.device)
            dist[torch.arange(len(rows)), labels] += 1 - self.label_smoothing
            kwargs = {"label_dist": dist}
        out = self.model(input_ids=ids, attention_mask=mask, n_slots=n_slots, opt_idx=opt_idx,
                         temperature=ones, **kwargs)
        return out["loss"]

    def log_probs(self, rows: list[dict]) -> list[list[float]]:
        """Log-xác suất (temperature 1) theo thứ tự nhãn của job, không xáo."""
        import torch

        canonical = list(range(len(self.labels)))
        encoded = [self._encode(r["text"], canonical)[:2] for r in rows]
        order = sorted(range(len(rows)), key=lambda i: len(encoded[i][0]))  # gom câu dài gần nhau, ít padding
        out: list[list[float]] = [[] for _ in rows]
        with torch.inference_mode():
            for s in range(0, len(order), self.eval_batch):
                idx = order[s:s + self.eval_batch]
                ids, mask, opt_idx, n_slots = self._batch([encoded[i] for i in idx])
                res = self.model(input_ids=ids, attention_mask=mask, n_slots=n_slots, opt_idx=opt_idx,
                                 temperature=torch.ones(len(idx), device=self.device))
                for i, lp in zip(idx, res["log_probs"].float().cpu().tolist()):
                    out[i] = lp
        return out

    def _gold(self, rows: list[dict]) -> list[int]:
        index = {k: i for i, k in enumerate(self.labels)}
        return [index[r["label"]] for r in rows]

    def evaluate(self, rows: list[dict]) -> dict:
        lps = self.log_probs(rows)
        probs = [cm.softmax(lp, 1.0) for lp in lps]
        conf = [confidence(p) for p in probs]
        return cm.summarize(self._gold(rows), probs, conf)

    def finalize(self, val_rows: list[dict], params: dict) -> dict:
        """Temperature cho câu hỏi choice và ngưỡng tự động / chuyển người, đều chọn trên val."""
        gold = self._gold(val_rows)
        lps = self.log_probs(val_rows)
        t = cm.fit_temperature(lps, gold)
        cfg = self.model.config
        cfg.temperature_by_kind = {**(cfg.temperature_by_kind or {}), "choice": t}
        probs = [cm.softmax(lp, t) for lp in lps]
        conf = [confidence(p) for p in probs]
        correct = [max(range(len(p)), key=p.__getitem__) == g for p, g in zip(probs, gold)]
        curve = cm.threshold_curve(conf, correct)
        chosen = cm.choose_threshold(curve, params["target_precision"])
        self.meta = {"labels": self.labels, "instructions": self.instructions, "descriptions": self.descriptions,
                     "temperature": t, "threshold": chosen["threshold"], "val_at_threshold": chosen}
        calibrated = cm.summarize(gold, probs, conf)
        return {"temperature": t, "threshold": chosen, "val_calibrated": calibrated, "val_curve": cm.thin(curve)}

    def save(self, out_dir: Path, source_path: str) -> None:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        self.model.save_pretrained(str(out_dir))
        src = Path(source_path)
        for name in COPY_FROM_SOURCE:
            if (src / name).exists() and not (out_dir / name).exists():
                shutil.copy2(src / name, out_dir / name)
        prov_src = src / "provenance.json"
        if prov_src.exists():
            prov = json.loads(prov_src.read_text("utf-8"))
            keep = {k: prov[k] for k in ("base_model", "base_model_revision", "base_model_revision_note") if k in prov}
            write_json(out_dir / "provenance.json", {**keep, "role": "finetune", "finetuned_from": prov.get("name")})
        write_json(out_dir / META_FILE, self.meta)

    # ------------------------------------------------------------ golden

    def golden(self, rows: list[dict], targets: list[dict], files: RunFiles) -> dict:
        import torch

        report = {"n": len(rows), "models": []}
        for i, target in enumerate(targets):
            files.status("running", message=f"Đang chấm {target['name']} ({i + 1}/{len(targets)})…",
                         done=i, total=len(targets))
            # Mỗi model dùng đúng nhãn, câu hỏi và mô tả lúc nó được train (model gốc: cấu hình của lần eval).
            self.load(target["path"], {"task_config": target["config"], "params": {}}, trainable=False)
            self.model.eval()
            t = (self.model.config.temperature_by_kind or {}).get("choice", self.model.config.temperature)
            lps = self.log_probs(rows)
            gold = self._gold(rows)
            probs = [cm.softmax(lp, t) for lp in lps]
            conf = [confidence(p) for p in probs]
            pred = [max(range(len(p)), key=p.__getitem__) for p in probs]
            correct = [p == g for p, g in zip(pred, gold)]
            mistakes = sorted((j for j in range(len(rows)) if not correct[j]), key=lambda j: -conf[j])[:50]
            threshold = target.get("threshold")
            report["models"].append({
                "key": target["key"], "name": target["name"], "kind": target["kind"], "threshold": threshold,
                "leaked": target.get("leaked", 0),
                "summary": cm.summarize(gold, probs, conf),
                "per_label": cm.per_label(gold, pred, self.labels),
                "confusions": cm.confusions(gold, pred, self.labels),
                "at_threshold": cm.at_threshold(conf, correct, threshold) if threshold is not None else None,
                "curve": cm.thin(cm.threshold_curve(conf, correct)),
                "mistakes": [{"text": rows[j]["text"], "gold": self.labels[gold[j]], "pred": self.labels[pred[j]],
                              "confidence": conf[j]} for j in mistakes],
            })
            del self.model
            torch.cuda.empty_cache()
        return report


def confidence(probs: list[float]) -> float:
    """Cùng công thức với strands_decider.confidence.derive_confidence: phân bố đều -> 0, chắc chắn -> 1."""
    n = len(probs)
    if n <= 1:
        return 1.0
    return max(0.0, min(1.0, (n * max(probs) - 1) / (n - 1)))
