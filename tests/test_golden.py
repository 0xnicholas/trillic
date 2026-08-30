"""Golden set loading and validation (pure-function layer).

External behavior under test: a jsonl file goes in, validated items come out,
and every invalid row is rejected with a message that names the line and the
entry id so fixing data never requires guessing.
"""

import json
from pathlib import Path

import pytest

from trillic.golden import GoldenError, GoldenItem, load_golden


def write_golden(tmp_path: Path, rows: list) -> Path:
    path = tmp_path / "golden.jsonl"
    lines = []
    for row in rows:
        lines.append(row if isinstance(row, str) else json.dumps(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_loads_valid_entries(tmp_path):
    path = write_golden(
        tmp_path,
        [
            {
                "id": "a-001",
                "prompt": "hello",
                "key_points": ["fact one", "fact two"],
            },
            {
                "id": "a-002",
                "prompt": "world",
                "key_points": ["fact"],
                # Forward compatibility: schema extension fields (load_type,
                # source) arrive with issue #3 and must not break loading.
                "load_type": "system_prompt",
                "source": {"dataset": "handwritten", "license": "CC0", "split": "eval"},
            },
        ],
    )
    items = load_golden(path)
    assert items == [
        GoldenItem(id="a-001", prompt="hello", key_points=("fact one", "fact two")),
        GoldenItem(id="a-002", prompt="world", key_points=("fact",)),
    ]


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps({"id": "a", "prompt": "p", "key_points": ["k"]}) + "\n\n"
        + json.dumps({"id": "b", "prompt": "q", "key_points": ["k"]}) + "\n"
    )
    assert [i.id for i in load_golden(path)] == ["a", "b"]


def test_missing_file_reports_path(tmp_path):
    with pytest.raises(GoldenError, match="no such file"):
        load_golden(tmp_path / "missing.jsonl")


@pytest.mark.parametrize(
    "row,expect_in_message",
    [
        ({"id": "x-1", "key_points": ["k"]}, "prompt"),
        ({"prompt": "p", "key_points": ["k"]}, "id"),
        ({"id": "x-1", "prompt": "p"}, "key_points"),
        ({"id": "x-1", "prompt": "p", "key_points": []}, "key_points"),
        ({"id": "x-1", "prompt": "p", "key_points": ["ok", "  "]}, "key_points"),
        ({"id": "", "prompt": "p", "key_points": ["k"]}, "id"),
        ({"id": "x-1", "prompt": "  ", "key_points": ["k"]}, "prompt"),
        (["not", "an", "object"], "object"),
        ("not json at all {", "parse"),
    ],
)
def test_invalid_rows_are_rejected_with_line_and_id(tmp_path, row, expect_in_message):
    good = {"id": "ok-0", "prompt": "p", "key_points": ["k"]}
    path = write_golden(tmp_path, [good, row])
    with pytest.raises(GoldenError) as exc_info:
        load_golden(path)
    message = str(exc_info.value)
    assert expect_in_message in message
    assert "line 2" in message  # row position, 1-indexed over file lines


def test_duplicate_ids_are_rejected(tmp_path):
    path = write_golden(
        tmp_path,
        [
            {"id": "dup", "prompt": "p", "key_points": ["k"]},
            {"id": "dup", "prompt": "q", "key_points": ["k"]},
        ],
    )
    with pytest.raises(GoldenError, match=r"line 2.*duplicate.*'dup'"):
        load_golden(path)


def test_fixture_golden_loads(fixture_golden_path):
    items = load_golden(fixture_golden_path)
    assert [i.id for i in items] == [
        "fixture-rag-001",
        "fixture-sys-002",
        "fixture-chat-003",
    ]
    assert all(item.prompt.strip() and item.key_points for item in items)
