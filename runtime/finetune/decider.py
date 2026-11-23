"""Fine-tune / chấm Strands Decider với một bộ quyết định (runtime protocol v2, docs/runtime-protocol.md).

Đầu vào: task_config = spec {system, intents[{key, policy}], gates[{key, policy, question}]} và các dòng
{input, history, intent, gates: {key: bool}}. Runtime tự ghép prompt của model:

  state  = system + policy các intent + policy các gate      <- giống hệt mọi request: đặt đầu để serving cache được
           + lịch sử + câu người dùng                         <- phần riêng của mỗi request
  câu hỏi intent = choice, lựa chọn chỉ ghi tên intent (policy đã ở state; decider đọc state kỹ hơn câu hỏi)
  câu hỏi gate   = mỗi gate một noul

Mỗi dòng thành một mẫu intent (nếu có nhãn intent) + một mẫu cho mỗi gate có nhãn. Lúc train xáo thứ tự lựa chọn
(như công thức gốc của decider) để head không học thuộc vị trí. Kết quả lưu đúng định dạng `strands-decider serve` đọc
được, kèm finetune_meta.json (spec, temperature, ngưỡng).
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
DEFAULT_INTENT_QUESTION = "Người dùng đang cần gì? Chọn đúng một intent theo các policy ở trên."
NOUL_SLOTS = ("false", "true")
ROLE_NAMES = {"user": "user", "human": "user", "assistant": "AI", "ai": "AI", "bot": "AI", "system": "system"}
MAX_MISTAKES = 50


# ---------------------------------------------------------------- ghép prompt (không cần torch, test được)


def format_history(history) -> str:
    """Lịch sử hội thoại -> dòng "user: …" / "AI: …". Nhận chuỗi sẵn, hoặc danh sách {role, content}."""
    if not history:
        return ""
    if isinstance(history, str):
        stripped = history.strip()
        if not stripped.startswith("["):
            return stripped
        try:
            history = json.loads(stripped)
        except ValueError:
            return stripped
    lines = []
    for turn in history:
        if isinstance(turn, dict):
            role = ROLE_NAMES.get(str(turn.get("role", "")).lower(), turn.get("role", "user"))
            lines.append(f"{role}: {turn.get('content', '')}")
        else:
            lines.append(str(turn))
    return "\n".join(lines)


def shared_state(spec: dict) -> str:
    """Phần đầu state giống nhau ở mọi request của cùng một spec (serving cache phần này)."""
    parts = [spec["system"]["text"].strip()]
    if spec.get("intents"):
        parts.append("## Intent\n" + "\n\n".join(f"### {i['key']}\n{i['policy'].strip()}" for i in spec["intents"]))
    if spec.get("gates"):
        parts.append("## Gate\n" + "\n\n".join(f"### {g['key']}\n{g['policy'].strip()}" for g in spec["gates"]))
    return "\n\n".join(parts)


def render_state_text(spec: dict, row: dict) -> str:
    text = shared_state(spec)
    history = format_history(row.get("history"))
    if history:
        text += f"\n\n## Lịch sử hội thoại\n{history}"
    return text + f"\n\n## Người dùng\n{row['input'].strip()}"


def intent_question(spec: dict) -> dict:
    return {"type": "choice", "instructions": spec["system"].get("intent_question") or DEFAULT_INTENT_QUESTION,
            "criteria": {i["key"]: None for i in spec["intents"]}}


def gate_question(gate: dict) -> dict:
    question = gate.get("question") or f"Câu của người dùng có thuộc trường hợp gate \"{gate['key']}\" ở trên không?"
    return {"type": "noul", "instructions": question}


def expand_rows(spec: dict, rows: list[dict]) -> list[dict]:
    """Mỗi dòng -> mẫu {row, q, key, gold}: q "intent" (gold = tên intent) hoặc "gate" (gold = "true"/"false")."""
    gates = {g["key"] for g in spec.get("gates", [])}
    intents = {i["key"] for i in spec.get("intents", [])}
    out = []
    for r in rows:
        if r.get("intent") and r["intent"] in intents:
            out.append({"row": r, "q": "intent", "key": "intent", "gold": r["intent"]})
        for key, value in (r.get("gates") or {}).items():
            if key in gates:
                out.append({"row": r, "q": "gate", "key": key, "gold": NOUL_SLOTS[int(bool(value))]})
    return out


def request_of(spec: dict, row: dict) -> dict:
    """Request /v1/systemone tương đương (để xem / để serving dùng): một state, intent + mọi gate."""
    questions = {}
    if spec.get("intents"):
        questions["intent"] = intent_question(spec)
    for g in spec.get("gates", []):
        questions[f"gate_{g['key']}"] = gate_question(g)
    return {"state": render_state_text(spec, row), "questions": questions}


# ---------------------------------------------------------------- task


class DeciderTask(Task):
    def load(self, path: str, config: dict, trainable: bool) -> None:
        import torch
        from strands_decider.modeling import StrandsDeciderModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = StrandsDeciderModel.load(path, trainable=trainable).to(self.device)
        self.tok = self.model.tokenizer
        self.spec: dict = config["task_config"]
        self.gates = {g["key"]: g for g in self.spec.get("gates", [])}
        self.intent_keys = [i["key"] for i in self.spec.get("intents", [])]
        self.max_length = int(config["params"].get("max_length") or self.model.config.max_length)
        self.eval_batch = int(config["params"].get("eval_batch_size", 16))
        self.label_smoothing = float(config["params"].get("label_smoothing", 0.0))

    def expand(self, rows: list[dict]) -> list[dict]:
        return expand_rows(self.spec, rows)

    def enable_gradient_checkpointing(self) -> None:
        self.model.torso.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def param_groups(self, lr: float, params: dict) -> list[dict]:
        head = [p for p in self.model.head.parameters() if p.requires_grad]
        torso = [p for p in self.model.torso.parameters() if p.requires_grad]
        return [{"params": torso, "lr": lr}, {"params": head, "lr": lr * params["head_lr_ratio"]}]

    # ------------------------------------------------------------ mã hoá

    def _question(self, example: dict):
        from strands_decider.schema import ChoiceQuestion, NoulQuestion

        if example["q"] == "intent":
            return ChoiceQuestion(**intent_question(self.spec))
        return NoulQuestion(**gate_question(self.gates[example["key"]]))

    def _slots(self, example: dict) -> list[str]:
        return list(self.intent_keys) if example["q"] == "intent" else list(NOUL_SLOTS)

    def _encode(self, example: dict, order: list[int]) -> tuple[list[int], list[int], list[str]]:
        from strands_decider.prompting import fit_window, option_token_index, render_question, render_state

        rq = render_question(self._question(example), option_order=order)
        state = render_state(render_state_text(self.spec, example["row"]))
        s, q, offsets = fit_window(self.tok.backend_tokenizer, state, [rq.text], self.max_length)
        opt = [len(s) + i for i in option_token_index(offsets[0], rq.option_spans, 0)]
        return s + q[0], opt, list(rq.slot_labels)

    def _batch(self, encoded: list[tuple[list[int], list[int]]]):
        """Câu hỏi có số lựa chọn khác nhau (intent K, gate 2): chỉ số lựa chọn đệm bằng lựa chọn cuối, phần dư bị
        che bởi n_slots trước softmax."""
        import torch

        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        width = max(len(ids) for ids, _ in encoded)
        k = max(len(opt) for _, opt in encoded)
        ids = torch.tensor([x + [pad] * (width - len(x)) for x, _ in encoded], device=self.device)
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in encoded], device=self.device)
        opt_idx = torch.tensor([o + [o[-1]] * (k - len(o)) for _, o in encoded], device=self.device)
        n_slots = torch.tensor([len(o) for _, o in encoded], device=self.device)
        return ids, mask, opt_idx, n_slots

    # ------------------------------------------------------------ train

    def loss(self, rows: list[dict], rng: random.Random):
        import torch

        encoded, gold, sizes = [], [], []
        for ex in rows:
            order = list(range(len(self._slots(ex))))
            rng.shuffle(order)
            ids, opt, slots = self._encode(ex, order)
            encoded.append((ids, opt))
            gold.append(slots.index(ex["gold"]))
            sizes.append(len(slots))
        ids, mask, opt_idx, n_slots = self._batch(encoded)
        labels = torch.tensor(gold, device=self.device)
        ones = torch.ones(len(rows), device=self.device)
        kwargs = {"labels": labels}
        if self.label_smoothing > 0:
            width = opt_idx.shape[1]
            dist = torch.zeros((len(rows), width), device=self.device)
            for i, n in enumerate(sizes):
                dist[i, :n] = self.label_smoothing / n
                dist[i, gold[i]] += 1 - self.label_smoothing
            kwargs = {"label_dist": dist}
        out = self.model(input_ids=ids, attention_mask=mask, n_slots=n_slots, opt_idx=opt_idx,
                         temperature=ones, **kwargs)
        return out["loss"]

    def log_probs(self, examples: list[dict]) -> list[list[float]]:
        """Log-xác suất (temperature 1) theo thứ tự lựa chọn chuẩn (intent theo spec, gate false/true), không xáo."""
        import torch

        encoded = [self._encode(ex, list(range(len(self._slots(ex)))))[:2] for ex in examples]
        order = sorted(range(len(examples)), key=lambda i: len(encoded[i][0]))  # gom câu dài gần nhau, ít padding
        out: list[list[float]] = [[] for _ in examples]
        with torch.inference_mode():
            for s in range(0, len(order), self.eval_batch):
                idx = order[s:s + self.eval_batch]
                ids, mask, opt_idx, n_slots = self._batch([encoded[i] for i in idx])
                res = self.model(input_ids=ids, attention_mask=mask, n_slots=n_slots, opt_idx=opt_idx,
                                 temperature=torch.ones(len(idx), device=self.device))
                for i, lp in zip(idx, res["log_probs"].float().cpu().tolist()):
                    out[i] = lp[:len(self._slots(examples[i]))]
        return out

    def _gold(self, examples: list[dict]) -> list[int]:
        return [self._slots(ex).index(ex["gold"]) for ex in examples]

    def _split(self, examples: list[dict]) -> tuple[list[int], list[int]]:
        intent = [i for i, ex in enumerate(examples) if ex["q"] == "intent"]
        gate = [i for i, ex in enumerate(examples) if ex["q"] == "gate"]
        return intent, gate

    def evaluate(self, rows: list[dict]) -> dict:
        lps = self.log_probs(rows)
        gold = self._gold(rows)
        probs = [cm.softmax(lp, 1.0) for lp in lps]
        intent, gate = self._split(rows)
        out = {"loss": cm.summarize(gold, probs, [confidence(p) for p in probs])["loss"]}
        for prefix, idx in (("", intent), ("gate_", gate)):
            if idx:
                part = cm.summarize([gold[i] for i in idx], [probs[i] for i in idx], [confidence(probs[i]) for i in idx])
                out.update({f"{prefix}{k}": v for k, v in part.items() if k in ("acc", "macro_f1", "ece")})
        return out

    def finalize(self, val_rows: list[dict], params: dict) -> dict:
        """Temperature riêng cho intent (choice) và gate (noul); ngưỡng tự động của intent; chỉ số từng gate. Chọn trên val."""
        gold = self._gold(val_rows)
        lps = self.log_probs(val_rows)
        intent, gate = self._split(val_rows)
        cfg = self.model.config
        temps = dict(cfg.temperature_by_kind or {})
        summary: dict = {}
        threshold = None
        if intent:
            t = cm.fit_temperature([lps[i] for i in intent], [gold[i] for i in intent])
            temps["choice"] = t
            probs = [cm.softmax(lps[i], t) for i in intent]
            conf = [confidence(p) for p in probs]
            g = [gold[i] for i in intent]
            correct = [max(range(len(p)), key=p.__getitem__) == x for p, x in zip(probs, g)]
            curve = cm.threshold_curve(conf, correct)
            threshold = cm.choose_threshold(curve, params["target_precision"])
            summary.update({"temperature": t, "threshold": threshold, "val_calibrated": cm.summarize(g, probs, conf),
                            "val_curve": cm.thin(curve)})
        if gate:
            t = cm.fit_temperature([lps[i] for i in gate], [gold[i] for i in gate])
            temps["noul"] = t
            summary["temperature_noul"] = t
            for key in self.gates:
                idx = [i for i in gate if val_rows[i]["key"] == key]
                if not idx:
                    continue
                probs = [cm.softmax(lps[i], t) for i in idx]
                g = [gold[i] for i in idx]
                pred = [max(range(2), key=p.__getitem__) for p in probs]
                true = cm.per_label(g, pred, list(NOUL_SLOTS))[1]
                summary[f"gate_{key}"] = {"n": len(idx), "precision": true["precision"], "recall": true["recall"],
                                          "f1": true["f1"], "ece": cm.summarize(g, probs, [max(p) for p in probs])["ece"]}
        cfg.temperature_by_kind = temps
        self.meta = {"spec": self.spec, "temperature_by_kind": temps,
                     "intent_threshold": threshold["threshold"] if threshold else None}
        return summary

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

    def _intent_report(self, examples: list[dict], lps: list[list[float]], t: float, threshold) -> dict:
        gold = self._gold(examples)
        probs = [cm.softmax(lp, t) for lp in lps]
        conf = [confidence(p) for p in probs]
        pred = [max(range(len(p)), key=p.__getitem__) for p in probs]
        correct = [p == g for p, g in zip(pred, gold)]
        mistakes = sorted((j for j in range(len(examples)) if not correct[j]), key=lambda j: -conf[j])[:MAX_MISTAKES]
        return {
            "summary": cm.summarize(gold, probs, conf),
            "per_label": cm.per_label(gold, pred, self.intent_keys),
            "confusions": cm.confusions(gold, pred, self.intent_keys),
            "at_threshold": cm.at_threshold(conf, correct, threshold) if threshold is not None else None,
            "curve": cm.thin(cm.threshold_curve(conf, correct)),
            "mistakes": [{"text": examples[j]["row"]["input"], "gold": self.intent_keys[gold[j]],
                          "pred": self.intent_keys[pred[j]], "confidence": conf[j]} for j in mistakes],
        }

    def _gate_report(self, examples: list[dict], lps: list[list[float]], t: float) -> dict:
        out = {}
        for key in self.gates:
            idx = [i for i, ex in enumerate(examples) if ex["key"] == key]
            if not idx:
                continue
            gold = [self._gold([examples[i]])[0] for i in idx]
            probs = [cm.softmax(lps[i], t) for i in idx]
            pred = [max(range(2), key=p.__getitem__) for p in probs]
            true = cm.per_label(gold, pred, list(NOUL_SLOTS))[1]
            wrong = sorted((j for j in range(len(idx)) if pred[j] != gold[j]), key=lambda j: -max(probs[j]))
            out[key] = {
                "summary": cm.summarize(gold, probs, [max(p) for p in probs]),
                "precision": true["precision"], "recall": true["recall"], "f1": true["f1"], "positives": true["n"],
                "mistakes": [{"text": examples[idx[j]]["row"]["input"], "gold": NOUL_SLOTS[gold[j]],
                              "p_true": probs[j][1]} for j in wrong[:MAX_MISTAKES]],
            }
        return out

    def golden(self, rows: list[dict], targets: list[dict], files: RunFiles) -> dict:
        import torch

        report = {"n": len(rows), "models": []}
        for i, target in enumerate(targets):
            files.status("running", message=f"Đang chấm {target['name']} ({i + 1}/{len(targets)})…",
                         done=i, total=len(targets))
            # Mỗi model dùng đúng spec lúc nó được train (model gốc: spec của lần eval).
            self.load(target["path"], {"task_config": target["config"], "params": {}}, trainable=False)
            self.model.eval()
            examples = self.expand(rows)
            lps = self.log_probs(examples)
            kinds = self.model.config.temperature_by_kind or {}
            base_t = self.model.config.temperature
            intent = [j for j, ex in enumerate(examples) if ex["q"] == "intent"]
            gate = [j for j, ex in enumerate(examples) if ex["q"] == "gate"]
            entry = {"key": target["key"], "name": target["name"], "kind": target["kind"],
                     "threshold": target.get("threshold"), "leaked": target.get("leaked", 0), "summary": None}
            if intent:
                entry.update(self._intent_report([examples[j] for j in intent], [lps[j] for j in intent],
                                                 kinds.get("choice", base_t), target.get("threshold")))
            if gate:
                entry["gates"] = self._gate_report([examples[j] for j in gate], [lps[j] for j in gate],
                                                   kinds.get("noul", base_t))
                if entry["summary"] is None:  # bộ chỉ có gate: xếp hạng theo trung bình các gate
                    parts = [g["summary"] for g in entry["gates"].values()]
                    entry["summary"] = {k: sum(p[k] for p in parts) / len(parts)
                                        for k in ("acc", "macro_f1", "ece", "loss", "mean_confidence")}
                    entry["summary"]["n"] = sum(p["n"] for p in parts)
            report["models"].append(entry)
            del self.model
            torch.cuda.empty_cache()
        return report


def confidence(probs: list[float]) -> float:
    """Cùng công thức với strands_decider.confidence.derive_confidence: phân bố đều -> 0, chắc chắn -> 1."""
    n = len(probs)
    if n <= 1:
        return 1.0
    return max(0.0, min(1.0, (n * max(probs) - 1) / (n - 1)))
