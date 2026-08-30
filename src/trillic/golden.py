"""Golden set loading and validation.

A golden file is jsonl: one JSON object per non-blank line. Required fields
(issue #1 schema, minimal for the walking skeleton):

    {"id": "...", "prompt": "...", "key_points": ["...", ...]}

Unknown extra fields are allowed — load_type / source provenance arrive with
issue #3 and must not break this loader.
"""

import json
from dataclasses import dataclass
from pathlib import Path


class GoldenError(Exception):
    """A golden file could not be read or validated."""


@dataclass(frozen=True)
class GoldenItem:
    id: str
    prompt: str
    key_points: tuple[str, ...]


def load_golden(path: Path) -> list[GoldenItem]:
    """Load and validate a golden jsonl file.

    Raises GoldenError whose message names the file line (1-indexed) and the
    entry id whenever possible, so fixing bad data never requires guessing.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise GoldenError(f"{path}: no such file") from e

    items: list[GoldenItem] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = _parse_row(path, line_number, line)
        item = _validate_row(path, line_number, row)
        if item.id in seen_ids:
            raise GoldenError(
                f"{path}: line {line_number}: duplicate id {item.id!r} "
                "(ids must be unique across the golden set)"
            )
        seen_ids.add(item.id)
        items.append(item)
    return items


def _parse_row(path: Path, line_number: int, line: str) -> dict:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as e:
        raise GoldenError(f"{path}: line {line_number}: cannot parse JSON: {e}") from e
    if not isinstance(row, dict):
        raise GoldenError(f"{path}: line {line_number}: each row must be a JSON object")
    return row


def _validate_row(path: Path, line_number: int, row: dict) -> GoldenItem:
    entry_id = row.get("id")
    where = f"{path}: line {line_number}"
    if entry_id is not None:
        where += f" (id={entry_id!r})"

    def fail(field: str, reason: str) -> None:
        raise GoldenError(f"{where}: {field} {reason}")

    if not isinstance(entry_id, str) or not entry_id.strip():
        fail("id", "must be a non-empty string")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        fail("prompt", "must be a non-empty string")
    key_points = row.get("key_points")
    if (
        not isinstance(key_points, list)
        or not key_points
        or not all(isinstance(k, str) and k.strip() for k in key_points)
    ):
        fail("key_points", "must be a non-empty list of non-empty strings")
    return GoldenItem(id=entry_id, prompt=prompt, key_points=tuple(key_points))
