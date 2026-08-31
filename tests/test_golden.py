"""Golden set schema v2: loading and validation (pure-function layer).

Schema (issue #3): the walking-skeleton {"id","prompt","key_points"} is
extended with load_type and source provenance:

    {"id": "...", "load_type": "rag", "prompt": "...", "key_points": [...],
     "source": {"dataset": "...", "subset": "...", "license": "...",
                "split": "eval", ...}}

Discipline encoded in the loader:
  - golden sets are the exam: every entry must carry split="eval"
  - MeetingBank is the base checkpoint's training set — hard-excluded
    (docs/data-strategy.md 硬约束 1)
"""

import json
from pathlib import Path

import pytest

from trillic.golden import GoldenError, GoldenItem, collect_golden_errors, load_golden

VALID = {
    "id": "rag-qasper-0001",
    "load_type": "rag",
    "prompt": "paper text... question?",
    "key_points": ["answer fact one", "answer fact two"],
    "source": {
        "dataset": "LongBench",
        "subset": "qasper",
        "license": "CC BY-NC 4.0",
        "split": "eval",
    },
}


def write_golden(tmp_path: Path, rows: list) -> Path:
    path = tmp_path / "golden.jsonl"
    lines = []
    for row in rows:
        lines.append(row if isinstance(row, str) else json.dumps(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_loads_valid_entry(tmp_path):
    items = load_golden(write_golden(tmp_path, [VALID]))
    assert items == [
        GoldenItem(
            id="rag-qasper-0001",
            load_type="rag",
            prompt="paper text... question?",
            key_points=("answer fact one", "answer fact two"),
            source={
                "dataset": "LongBench",
                "subset": "qasper",
                "license": "CC BY-NC 4.0",
                "split": "eval",
            },
        )
    ]


def test_source_may_carry_extra_provenance_fields(tmp_path):
    entry = json.loads(json.dumps(VALID))
    entry["source"]["content_sha1"] = "0" * 40
    entry["source"]["answers"] = ["dataset answer"]
    entry["source"]["url"] = "https://example.org"
    assert len(load_golden(write_golden(tmp_path, [entry]))) == 1


def test_blank_lines_are_skipped(tmp_path):
    a = dict(VALID, id="a")
    b = dict(VALID, id="b")
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps(a) + "\n\n" + json.dumps(b) + "\n"
    )
    assert [i.id for i in load_golden(path)] == ["a", "b"]


def test_file_with_no_entries_is_rejected(tmp_path):
    """An empty golden set is a mistake (wrong path, truncated file), not a
    valid exam — reject rather than produce an empty report."""
    path = tmp_path / "golden.jsonl"
    path.write_text("\n\n")
    with pytest.raises(GoldenError, match="no entries"):
        load_golden(path)
    errors = collect_golden_errors(path)
    assert len(errors) == 1 and "no entries" in errors[0]


def test_missing_file_reports_path(tmp_path):
    with pytest.raises(GoldenError, match="no such file"):
        load_golden(tmp_path / "missing.jsonl")


@pytest.mark.parametrize(
    "row,expect_in_message",
    [
        # v1 schema is no longer complete: load_type and source are required
        ({"id": "x-1", "prompt": "p", "key_points": ["k"]}, "load_type"),
        ({"id": "x-1", "prompt": "p", "key_points": ["k"], "load_type": "rag"}, "source"),
        ({"id": "x-1", "prompt": "p", "key_points": ["k"], "load_type": "bogus",
          "source": VALID["source"]}, "load_type"),
        ({"id": "x-1", "prompt": "p", "key_points": [], "load_type": "rag",
          "source": VALID["source"]}, "key_points"),
        ({"id": "x-1", "prompt": "p", "key_points": ["k"], "load_type": "rag",
          "source": {"dataset": "LongBench", "subset": "qasper", "license": "x"}}, "split"),
        ({"id": "x-1", "prompt": "p", "key_points": ["k"], "load_type": "rag",
          "source": {"subset": "qasper", "license": "x", "split": "eval"}}, "dataset"),
        ({"id": "x-1", "prompt": "p", "key_points": ["k"], "load_type": "rag",
          "source": {"dataset": "LongBench", "subset": "qasper", "license": "x",
                     "split": "train"}}, "split"),
        # MeetingBank exclusion: base checkpoint's training set, hard constraint
        ({"id": "x-1", "prompt": "p", "key_points": ["k"], "load_type": "rag",
          "source": {"dataset": "MeetingBank", "subset": "any", "license": "x",
                     "split": "eval"}}, "MeetingBank"),
        ({"id": "x-1", "prompt": "p", "key_points": ["k"], "load_type": "rag",
          "source": {"dataset": "LongBench", "subset": "meetingbank-llmcompressed",
                     "license": "x", "split": "eval"}}, "MeetingBank"),
        # base-field regressions keep their messages
        ({"load_type": "rag", "prompt": "p", "key_points": ["k"], "source": VALID["source"]}, "id"),
        ({"id": "x-1", "load_type": "rag", "key_points": ["k"], "source": VALID["source"]}, "prompt"),
        ({"id": "x-1", "prompt": "p", "load_type": "rag", "key_points": ["ok", "  "],
          "source": VALID["source"]}, "key_points"),
        (["not", "an", "object"], "object"),
        ("not json at all {", "parse"),
    ],
)
def test_invalid_rows_are_rejected_with_line_and_id(tmp_path, row, expect_in_message):
    good = dict(VALID, id="ok-0")
    path = write_golden(tmp_path, [good, row])
    with pytest.raises(GoldenError) as exc_info:
        load_golden(path)
    message = str(exc_info.value)
    assert expect_in_message in message
    assert "line 2" in message  # row position, 1-indexed over file lines


def test_duplicate_ids_are_rejected(tmp_path):
    path = write_golden(tmp_path, [VALID, dict(VALID, key_points=["k"])])
    with pytest.raises(GoldenError, match=r"line 2.*duplicate.*'rag-qasper-0001'"):
        load_golden(path)


def test_load_permissive_mode_for_non_golden_files(tmp_path):
    """Split=train rows are legal data in general — just never in a golden set.
    load_golden(strict=False) is the escape hatch for non-exam use."""
    train_entry = json.loads(json.dumps(VALID))
    train_entry["source"]["split"] = "train"
    items = load_golden(write_golden(tmp_path, [train_entry]), require_eval_split=False)
    assert items[0].source["split"] == "train"


def test_fixture_golden_loads(fixture_golden_path):
    items = load_golden(fixture_golden_path)
    assert [i.id for i in items] == [
        "fixture-rag-001",
        "fixture-sys-002",
        "fixture-chat-003",
    ]
    assert [i.load_type for i in items] == ["rag", "system_prompt", "dialogue"]
    assert all(i.source["split"] == "eval" for i in items)
