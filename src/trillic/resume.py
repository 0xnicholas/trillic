"""Interrupt-resume ledger (issue #8): replay recorded gateway results.

The billing surface is the gateway (answers + judges); compression through
the sidecar is local compute. A resumed run replays the prior run's
recorded results instead of re-billing them, under content-addressed
conditions:

- originals: reusable per item id — the payload is the golden prompt, so
  golden sha256 + quality pins (answer model, judge model, rubric sha)
  fully determine the calls;
- compressed payloads: reusable per (level, item id) only when the fresh
  recompression is byte-identical to the prior run's recorded
  compressed_text. A drift in compression invalidates the recorded answer;
  the item is honestly re-billed.

The prior run's metrics.json IS the ledger — no second format to keep in
sync, and run immutability is preserved (a resume writes a fresh run dir).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


class ResumeError(ValueError):
    """A prior run cannot serve as a resume source."""


@dataclass
class Replay:
    """Content-addressed gateway results extracted from a prior run."""

    golden_sha256: str
    answer_model: str
    judge_model: str
    rubric_sha256: str
    # item id -> recorded original-side result
    originals: dict[str, dict] = field(default_factory=dict)
    # (aggressiveness, item id) -> recorded compressed-side result
    compressed: dict[tuple[float, str], dict] = field(default_factory=dict)

    @property
    def original_ids(self) -> set[str]:
        return set(self.originals)

    @property
    def levels(self) -> set[float]:
        return {level for level, _ in self.compressed}

    def compressed_payload(self, level: float, item_id: str) -> str | None:
        entry = self.compressed.get((level, item_id))
        return None if entry is None else entry["payload"]

    def check_pins(self, *, answer_model: str, judge_model: str, rubric_sha256: str) -> None:
        """Refuse replay when the quality pins changed (silent reuse across
        a different configuration would poison the baseline)."""
        if answer_model != self.answer_model:
            raise ResumeError(
                f"cannot resume: answer_model changed (prior {self.answer_model!r} "
                f"vs {answer_model!r})"
            )
        if judge_model != self.judge_model:
            raise ResumeError(
                f"cannot resume: judge_model changed (prior {self.judge_model!r} "
                f"vs {judge_model!r})"
            )
        if rubric_sha256 != self.rubric_sha256:
            raise ResumeError(
                "cannot resume: judge rubric changed "
                f"(prior sha256 {self.rubric_sha256[:12]}… vs {rubric_sha256[:12]}…) — "
                "scores would not be comparable"
            )

    def check_golden(self, golden_sha256: str) -> None:
        """Refuse replay when the exam itself changed."""
        if golden_sha256 != self.golden_sha256:
            raise ResumeError(
                "cannot resume: golden set changed "
                f"(prior sha256 {self.golden_sha256[:12]}… vs {golden_sha256[:12]}…) — "
                "the prior answers grade a different exam"
            )


def load_replay(metrics: dict) -> Replay:
    """Extract a Replay from a prior run's metrics.json.

    Original-side results come from the task-quality rows; compressed-side
    payloads come from the SAME level's compression rows (compressed_text),
    matched by item id.
    """
    schema = metrics.get("schema_version")
    if not isinstance(schema, int) or schema < 3:
        raise ResumeError(
            f"cannot resume from schema_version {schema!r} — task-quality "
            "ledger requires schema 3+"
        )
    task_quality = metrics.get("task_quality")
    if not task_quality or not task_quality.get("enabled", False):
        raise ResumeError(
            "cannot resume: the prior run has no task_quality ledger "
            "(quality.task_quality was disabled)"
        )

    replay = Replay(
        golden_sha256=metrics["golden"]["sha256"],
        answer_model=task_quality["answer_model"],
        judge_model=task_quality["judge_model"],
        rubric_sha256=task_quality["judge_rubric_sha256"],
    )
    for level in metrics.get("metrics", {}).get("levels", []):
        payloads = {
            row["id"]: row.get("compressed_text")
            for row in level.get("items", [])
        }
        block = level.get("aggregate", {}).get("task_quality")
        if not block:
            continue
        for row in block.get("items", []):
            original = {
                "answer": row["original_answer"],
                "scores": list(row["original_judge_scores"]),
            }
            existing = replay.originals.get(row["id"])
            if existing is not None and existing != original:
                raise ResumeError(
                    f"cannot resume: prior run records conflicting originals "
                    f"for item {row['id']!r}"
                )
            replay.originals[row["id"]] = original
            replay.compressed[(level["aggressiveness"], row["id"])] = {
                "payload": payloads.get(row["id"]),
                "answer": row["compressed_answer"],
                "scores": list(row["compressed_judge_scores"]),
            }
    return replay


def load_replay_from_run_dir(run_dir: Path) -> Replay:
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.is_file():
        raise ResumeError(f"cannot resume: no metrics.json under {run_dir}")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ResumeError(f"cannot resume: cannot read {metrics_path}: {e}") from e
    return load_replay(metrics)
