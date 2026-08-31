"""Golden set schema v2: loading and validation.

A golden file is jsonl: one JSON object per non-blank line. Schema (issue #3):

    {"id": "...",                      # unique, non-empty
     "load_type": "rag" | "system_prompt" | "dialogue",
     "prompt": "...",                  # the text under compression
     "key_points": ["...", ...],       # must-survive facts (non-empty)
     "source": {                       # provenance, per-entry license trail
       "dataset": "...",               # e.g. "LongBench" or "handwritten"
       "subset": "...",                # e.g. "qasper"
       "license": "...",               # e.g. "CC BY-NC 4.0"
       "split": "eval" | "train",      # golden sets carry eval-half entries only
       ...                             # extra provenance (sha1, answers, url) allowed
     }}

Discipline encoded here, not in reviewers' memory:
  - MeetingBank is hard-excluded anywhere in source (docs/data-strategy.md
    硬约束 1: it is the base checkpoint's training set);
  - golden sets are the exam, so entries must come from the eval half
    (require_eval_split=False relaxes this for non-golden consumers).
"""

import json
from dataclasses import dataclass
from pathlib import Path

LOAD_TYPES = ("rag", "system_prompt", "dialogue")
SPLITS = ("eval", "train")
_SOURCE_REQUIRED = ("dataset", "subset", "license", "split")


class GoldenError(Exception):
    """A golden file could not be read or validated."""


@dataclass(frozen=True)
class GoldenItem:
    id: str
    load_type: str
    prompt: str
    key_points: tuple[str, ...]
    source: dict


def collect_golden_errors(
    path: Path, *, require_eval_split: bool = True
) -> list[str]:
    """Validate every row and return ALL error messages (empty = valid).

    Unlike load_golden, which fails fast, this reports every bad row with its
    line number so `golden validate` fixes the whole file in one pass.
    """
    return _scan_golden(path, require_eval_split=require_eval_split)[1]


def load_golden(path: Path, *, require_eval_split: bool = True) -> list[GoldenItem]:
    """Load and validate a golden jsonl file (single pass).

    Raises GoldenError whose message names the file line (1-indexed) and the
    entry id whenever possible, so fixing bad data never requires guessing.
    """
    items, errors = _scan_golden(path, require_eval_split=require_eval_split)
    if errors:
        suffix = f" (+{len(errors) - 1} more errors)" if len(errors) > 1 else ""
        raise GoldenError(f"{errors[0]}{suffix}")
    return items


def _scan_golden(
    path: Path, *, require_eval_split: bool
) -> tuple[list[GoldenItem], list[str]]:
    """One pass over the file: (valid items, error messages)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], [f"{path}: no such file"]

    items: list[GoldenItem] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = _parse_row(path, line_number, line)
        try:
            item = _validate_row(
                path, line_number, row, require_eval_split=require_eval_split
            )
        except GoldenError as e:
            errors.append(str(e))
            continue
        if item.id in seen_ids:
            errors.append(
                f"{path}: line {line_number}: duplicate id {item.id!r} "
                "(ids must be unique across the golden set)"
            )
        else:
            seen_ids.add(item.id)
            items.append(item)
    if not items and not errors:
        errors.append(
            f"{path}: no entries found — an empty golden set is a mistake "
            "(wrong path or truncated file), not a valid exam"
        )
    return items, errors


def _parse_row(path: Path, line_number: int, line: str) -> dict:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as e:
        raise GoldenError(f"{path}: line {line_number}: cannot parse JSON: {e}") from e
    if not isinstance(row, dict):
        raise GoldenError(f"{path}: line {line_number}: each row must be a JSON object")
    return row


def _validate_row(
    path: Path, line_number: int, row: dict, *, require_eval_split: bool
) -> GoldenItem:
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
    load_type = row.get("load_type")
    if load_type not in LOAD_TYPES:
        fail("load_type", f"must be one of {list(LOAD_TYPES)}, got {load_type!r}")
    key_points = row.get("key_points")
    if (
        not isinstance(key_points, list)
        or not key_points
        or not all(isinstance(k, str) and k.strip() for k in key_points)
    ):
        fail("key_points", "must be a non-empty list of non-empty strings")

    source = row.get("source")
    if not isinstance(source, dict):
        fail("source", "must be an object with dataset/subset/license/split")
    for field in _SOURCE_REQUIRED:
        if not isinstance(source.get(field), str) or not source[field].strip():
            fail("source", f"field {field!r} must be a non-empty string")
    if "meetingbank" in f"{source.get('dataset', '')}/{source.get('subset', '')}".lower():
        fail(
            "source",
            "MeetingBank is excluded by hard constraint (base checkpoint "
            "training set — see docs/data-strategy.md 硬约束 1)",
        )
    if source["split"] not in SPLITS:
        fail("source", f"split must be one of {list(SPLITS)}, got {source['split']!r}")
    if require_eval_split and source["split"] != "eval":
        fail(
            "source",
            "split must be 'eval' in a golden set (golden draws only from "
            "the eval half; train-half entries are reserved for phase-2 training)",
        )
    return GoldenItem(
        id=entry_id,
        load_type=load_type,
        prompt=prompt,
        key_points=tuple(key_points),
        source=dict(source),
    )
