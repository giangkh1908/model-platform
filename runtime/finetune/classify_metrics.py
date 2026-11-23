"""Chỉ số cho mọi loại fine-tune dạng phân loại (chọn 1 nhãn kèm xác suất). Thuần Python, không cần sklearn.

Ngưỡng tự động / chuyển người: confidence >= t thì tự xử lý, ngược lại chuyển người. Positive = model trả lời đúng:
  TP tự động và đúng · FP tự động nhưng sai (lỗi lọt ra ngoài) · FN chuyển người dù đúng · TN chuyển người, chặn được lỗi
"""

from __future__ import annotations

import math
from collections import Counter

ECE_BINS = 15
CURVE_POINTS = 200  # số điểm tối đa của đường ngưỡng gửi lên giao diện
MIN_TEMPERATURE = 0.5


def summarize(gold: list[int], probs: list[list[float]], confidence: list[float]) -> dict:
    n = len(gold)
    pred = [max(range(len(p)), key=p.__getitem__) for p in probs]
    correct = [g == q for g, q in zip(gold, pred)]
    nll = -sum(math.log(max(p[g], 1e-12)) for g, p in zip(gold, probs)) / n
    return {
        "n": n,
        "loss": nll,
        "acc": sum(correct) / n,
        "macro_f1": macro_f1(gold, pred, len(probs[0])),
        "ece": ece([max(p) for p in probs], correct),
        "mean_confidence": sum(confidence) / n,
    }


def macro_f1(gold: list[int], pred: list[int], n_labels: int) -> float:
    """Trung bình F1 trên các nhãn có mặt trong gold."""
    rows = per_label(gold, pred, list(range(n_labels)))
    present = [r["f1"] for r in rows if r["n"]]
    return sum(present) / len(present) if present else 0.0


def per_label(gold: list[int], pred: list[int], labels: list) -> list[dict]:
    tp, n_gold, n_pred = Counter(), Counter(gold), Counter(pred)
    for g, q in zip(gold, pred):
        if g == q:
            tp[g] += 1
    out = []
    for i, name in enumerate(labels):
        precision = tp[i] / n_pred[i] if n_pred[i] else 0.0
        recall = tp[i] / n_gold[i] if n_gold[i] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append({"label": name, "n": n_gold[i], "predicted": n_pred[i],
                    "precision": precision, "recall": recall, "f1": f1})
    return out


def confusions(gold: list[int], pred: list[int], labels: list, k: int = 15) -> list[dict]:
    c = Counter((g, q) for g, q in zip(gold, pred) if g != q)
    return [{"gold": labels[g], "pred": labels[q], "count": n} for (g, q), n in c.most_common(k)]


def ece(max_probs: list[float], correct: list[bool]) -> float:
    """Expected calibration error: độ lệch giữa độ tự tin và tỉ lệ đúng thực tế, chia 15 khoảng."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(ECE_BINS)]
    for p, ok in zip(max_probs, correct):
        bins[min(ECE_BINS - 1, int(p * ECE_BINS))].append((p, ok))
    n = len(max_probs)
    total = 0.0
    for b in bins:
        if b:
            conf = sum(p for p, _ in b) / len(b)
            acc = sum(ok for _, ok in b) / len(b)
            total += len(b) / n * abs(conf - acc)
    return total


def rates(threshold: float, tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "coverage": (tp + fp) / max(1, tp + fp + fn + tn),
    }


def at_threshold(confidence: list[float], correct: list[bool], threshold: float) -> dict:
    tp = sum(c >= threshold and ok for c, ok in zip(confidence, correct))
    fp = sum(c >= threshold and not ok for c, ok in zip(confidence, correct))
    fn = sum(c < threshold and ok for c, ok in zip(confidence, correct))
    tn = len(correct) - tp - fp - fn
    return rates(threshold, tp, fp, fn, tn)


def threshold_curve(confidence: list[float], correct: list[bool]) -> list[dict]:
    """Mọi ngưỡng = mọi giá trị confidence khác nhau, từ thấp (tự động hết) lên cao."""
    pairs = sorted(zip(confidence, correct), reverse=True)
    pos = sum(correct)
    neg = len(correct) - pos
    curve, tp, fp, i = [], 0, 0, 0
    while i < len(pairs):
        t = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == t:
            tp += pairs[i][1]
            fp += not pairs[i][1]
            i += 1
        curve.append(rates(t, tp, fp, pos - tp, neg - fp))
    curve.reverse()
    return curve


def choose_threshold(curve: list[dict], target_precision: float) -> dict:
    """Ngưỡng tự động được nhiều nhất mà precision vẫn >= mục tiêu; không ngưỡng nào đạt thì lấy F1 lớn nhất.
    Bằng nhau thì lấy ngưỡng cao hơn (thận trọng hơn)."""
    ok = [c for c in curve if c["precision"] >= target_precision and c["tp"] + c["fp"] > 0]
    if ok:
        best = max(ok, key=lambda c: (c["coverage"], c["threshold"]))
    else:
        best = max(curve, key=lambda c: (round(c["f1"], 6), c["threshold"]))
    return {**best, "meets_target": bool(ok), "target_precision": target_precision}


def thin(curve: list[dict]) -> list[dict]:
    """Rút gọn đường ngưỡng còn tối đa CURVE_POINTS điểm để vẽ."""
    if len(curve) <= CURVE_POINTS:
        return curve
    stride = len(curve) / CURVE_POINTS
    return [curve[int(i * stride)] for i in range(CURVE_POINTS)] + [curve[-1]]


def fit_temperature(log_probs: list[list[float]], gold: list[int]) -> float:
    """Temperature làm NLL trên val nhỏ nhất (dò lưới theo log, rồi dò mịn quanh điểm tốt nhất)."""
    def nll(t: float) -> float:
        total = 0.0
        for lp, g in zip(log_probs, gold):
            scaled = [x / t for x in lp]
            m = max(scaled)
            total += m + math.log(sum(math.exp(x - m) for x in scaled)) - scaled[g]
        return total / len(gold)

    # Chặn dưới 0.5: val đúng hết thì NLL giảm mãi khi t -> 0, sẽ làm model tự tin quá mức trên dữ liệu thật.
    grid = [math.exp(x / 20) for x in range(-14, 41)]  # 0.5 … 7.4
    best = min(grid, key=nll)
    fine = [t for t in (best * math.exp(x / 200) for x in range(-10, 11)) if t >= MIN_TEMPERATURE]
    return min(fine, key=nll)


def softmax(log_probs: list[float], temperature: float) -> list[float]:
    scaled = [x / temperature for x in log_probs]
    m = max(scaled)
    e = [math.exp(x - m) for x in scaled]
    s = sum(e)
    return [x / s for x in e]
