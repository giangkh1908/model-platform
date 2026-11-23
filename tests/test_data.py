import pytest

from backend.finetune import data
from backend.finetune.decider import DeciderTask

TASK = DeciderTask()


def samples(n: int, labels=("a", "b")) -> list[dict]:
    return [{"text": f"câu {i}", "label": labels[i % len(labels)]} for i in range(n)]


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("x.jsonl", "", "jsonl"),
        ("x.ndjson", "", "jsonl"),
        ("x.json", "", "json"),
        ("x.tsv", "", "tsv"),
        ("x.CSV", "", "csv"),
        ("noext", "[{}]", "json"),
        ("noext", '{"a": 1}', "jsonl"),
        ("noext", "a,b", "csv"),
    ],
)
def test_detect_format(name, text, expected):
    assert data.detect_format(name, text) == expected


def test_parse_jsonl_skips_blank_lines_and_bom():
    fmt, rows = data.parse_table("d.jsonl", '﻿{"text": "x", "label": "a"}\n\n{"text": "y", "label": "b"}\n')
    assert fmt == "jsonl"
    assert rows == [{"text": "x", "label": "a"}, {"text": "y", "label": "b"}]


def test_parse_jsonl_reports_bad_line_number():
    with pytest.raises(ValueError, match="Dòng 2"):
        data.parse_table("d.jsonl", '{"a": 1}\nnot json\n')


def test_parse_csv():
    fmt, rows = data.parse_table("d.csv", "text,label\nxin chào,a\n")
    assert fmt == "csv"
    assert rows == [{"text": "xin chào", "label": "a"}]


def test_to_samples_requires_mapping_and_collects_row_errors():
    with pytest.raises(ValueError, match="Chưa chọn cột"):
        data.to_samples(TASK, [], {"text": "t"})
    good, errors = data.to_samples(TASK, [{"t": "x", "l": "a"}, {"t": "", "l": "a"}], {"text": "t", "label": "l"})
    assert good == [{"text": "x", "label": "a"}]
    assert errors == [{"row": 2, "reason": "thiếu nội dung câu"}]


def test_split_is_deterministic_and_disjoint():
    rows = samples(50)
    train1, val1, stats = data.split(TASK, rows, 0.2, seed=7)
    train2, val2, _ = data.split(TASK, rows, 0.2, seed=7)
    assert (train1, val1) == (train2, val2)
    assert len(val1) == 10 and len(train1) == 40
    assert not {r["text"] for r in train1} & {r["text"] for r in val1}
    assert stats["total"] == 50 and stats["missing_in_val"] == []


@pytest.mark.parametrize("fraction", [0, 1, 1.5])
def test_split_rejects_bad_fraction(fraction):
    with pytest.raises(ValueError):
        data.split(TASK, samples(10), fraction, seed=0)


def test_split_rejects_too_little_data():
    with pytest.raises(ValueError, match="quá ít"):
        data.split(TASK, samples(1), 0.2, seed=0)


def test_merge_dedupes_and_counts_label_conflicts():
    part1 = [{"text": "Xin  chào", "label": "a"}, {"text": "x", "label": "a"}]
    part2 = [{"text": "x", "label": "a"}, {"text": "xin chào", "label": "b"}]
    rows, stats = data.merge(TASK, [part1, part2], dedupe=True)
    assert len(rows) == 3
    assert stats == {"before": 4, "removed_duplicates": 1, "conflicts": 1}


def test_tvd():
    from collections import Counter

    assert data.tvd(Counter(a=1, b=1), Counter(a=5, b=5)) == 0
    assert data.tvd(Counter(a=2), Counter(a=1, b=1)) == pytest.approx(0.5)
