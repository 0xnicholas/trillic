"""LongBench acquisition and seeded drafting of RAG golden entries.

Pipeline (issue #3):
  1. Curated subset registry (eval/manifests/subsets.toml, committed) holds
     the human research output: per-subset license and train-use verdict.
  2. `build_manifest` merges that curation with computed split counts over
     the downloaded data files (gitignored _downloads/, per old-repo
     precedent) into a deterministic manifest JSON.
  3. `build_rag_entries` samples seeded, ONLY from the eval half, and emits
     schema-v2 golden entries with full provenance. Its key_points are a
     deterministic DRAFT derived from dataset answers — final golden sets
     are hand-pruned from this draft (人工修剪), with provenance preserved
     via source.answers.
"""

import hashlib
import json
import random
import re
import tomllib
from pathlib import Path

from trillic.splits import assign_splits, row_fingerprint, split_half_rows
from trillic.tokens import TokenCounter

SPLIT_METHOD = (
    "sha1(context + NUL + input) sorted ascending; "
    "first ceil(n/2) -> eval half, remainder -> train half"
)

_PROMPT_TEMPLATES = {
    "qasper": (
        "Read the following research paper excerpt and answer the question "
        "based only on it.\n\nPaper: {context}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given documents.\n\n"
        "Documents: {context}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Summarize the report, "
        "covering its main points and key facts.\n\nReport: {context}\n\nSummary:"
    ),
}

_MAX_POINT_CHARS = 160
_MAX_POINTS = 6
_SHORT_ANSWER_CHARS = 80


class LongBenchError(ValueError):
    """LongBench subset data could not be loaded or does not match the manifest."""


def load_subset_rows(path: Path) -> list[dict]:
    """Load a LongBench subset jsonl ({input, context, answers, ...} rows)."""
    path = Path(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise LongBenchError(f"{path}: line {line_number}: cannot parse JSON: {e}") from e
        for field in ("input", "context", "answers"):
            if field not in row:
                raise LongBenchError(f"{path}: line {line_number}: missing field {field!r}")
        rows.append(row)
    return rows


def render_prompt(subset: str, context: str, question: str) -> str:
    try:
        template = _PROMPT_TEMPLATES[subset]
    except KeyError:
        raise ValueError(f"no prompt template for subset {subset!r}") from None
    return template.format(context=context, question=question)


def derive_key_points(answers: list[str]) -> list[str]:
    """Deterministic draft of must-survive facts from dataset annotations.

    Short answers are the point itself. Long answers contribute their first
    sentence plus every sentence carrying a digit (numeric facts are the
    known failure mode this project measures). Hand-pruning refines this.
    """
    answer = next((a.strip() for a in answers if isinstance(a, str) and a.strip()), "")
    if not answer:
        return []
    if len(answer) <= _SHORT_ANSWER_CHARS:
        return [answer]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    chosen: list[str] = []
    if sentences:
        chosen.append(sentences[0])
    chosen.extend(s for s in sentences if re.search(r"\d", s))
    deduped: list[str] = []
    for s in chosen:
        if s not in deduped:
            deduped.append(s)
    points = [s[:_MAX_POINT_CHARS] for s in deduped[:_MAX_POINTS]]
    return points or [answer[:_MAX_POINT_CHARS]]


def build_manifest(subsets_toml: Path, data_dir: Path) -> dict:
    """Merge curated subset registry with computed split counts."""
    subsets_toml = Path(subsets_toml)
    data_dir = Path(data_dir)
    curation = tomllib.loads(subsets_toml.read_text(encoding="utf-8"))

    subsets = []
    for entry in curation.get("subset", []):
        path = data_dir / entry["file"]
        rows = load_subset_rows(path)
        eval_rows, train_rows = split_half_rows(rows)
        fingerprints = sorted({row_fingerprint(r) for r in rows})
        subsets.append(
            {
                "name": entry["name"],
                "file": entry["file"],
                "file_sha256": _file_sha256(path),
                "entries": len(rows),
                "eval_half": len(eval_rows),
                "train_half": len(train_rows),
                # boundary = first fingerprint of the train half; None when the
                # subset is so small everything lands in the eval half
                "train_boundary_fingerprint": (
                    fingerprints[len(eval_rows)] if train_rows else None
                ),
                "license": entry["license"],
                "train_use": entry["train_use"],
                "note": entry.get("note", ""),
            }
        )
    return {
        "schema_version": 1,
        "split_method": SPLIT_METHOD,
        "source_url": curation.get("source_url", ""),
        "subsets": subsets,
    }


def build_rag_entries(
    manifest: dict,
    *,
    data_dir: Path,
    counts: dict[str, int],
    seed: int,
    min_tokens: int = 500,
    max_tokens: int = 4000,
) -> list[dict]:
    """Seeded drafting of RAG golden entries from the EVAL half only."""
    data_dir = Path(data_dir)
    counter = TokenCounter("cl100k_base")
    manifest_subsets = {s["name"]: s for s in manifest.get("subsets", [])}

    entries: list[dict] = []
    for subset_name in manifest_subsets:
        if subset_name not in counts:
            continue
        wanted = counts[subset_name]
        info = manifest_subsets[subset_name]
        rows = load_subset_rows(data_dir / info["file"])
        _check_against_manifest(subset_name, info, data_dir / info["file"], rows)

        eval_rows, train_rows = split_half_rows(rows)
        eval_fps = {row_fingerprint(r) for r in eval_rows}
        train_fps = {row_fingerprint(r) for r in train_rows}

        candidates = []
        for row in eval_rows:
            if not any(isinstance(a, str) and a.strip() for a in row["answers"]):
                continue  # no dataset annotation -> nothing to derive key points from
            prompt = render_prompt(
                subset_name, context=row["context"], question=row["input"]
            )
            tokens = counter.count(prompt)
            if min_tokens <= tokens <= max_tokens:
                candidates.append((row_fingerprint(row), prompt, row))
        candidates.sort(key=lambda c: c[0])
        if len(candidates) < wanted:
            raise ValueError(
                f"{subset_name}: only {len(candidates)} in-band eval-half candidates, "
                f"need {wanted} (adjust band, counts, or seed)"
            )
        rng = random.Random(seed)
        sampled = rng.sample(candidates, wanted)
        assert {fp for fp, _, _ in sampled} <= eval_fps and not (
            {fp for fp, _, _ in sampled} & train_fps
        ), f"{subset_name}: split discipline violated"
        for fingerprint, prompt, row in sampled:
            entries.append(
                {
                    "id": f"rag-{subset_name}-{fingerprint[:8]}",
                    "load_type": "rag",
                    "prompt": prompt,
                    "key_points": derive_key_points(row["answers"]),
                    "source": {
                        "dataset": "LongBench",
                        "subset": subset_name,
                        "license": info["license"],
                        "split": "eval",
                        "content_sha1": fingerprint,
                        "answers": [a for a in row["answers"] if isinstance(a, str) and a.strip()],
                    },
                }
            )
    missing = set(counts) - set(manifest_subsets)
    if missing:
        raise ValueError(f"counts reference subsets not in the manifest: {sorted(missing)}")
    return entries


def _check_against_manifest(subset_name: str, info: dict, path: Path, rows: list[dict]) -> None:
    """Refuse to mix splits against drifted data: the manifest is the split
    record, and it must describe the exact bytes being split."""
    if len(rows) != info["entries"]:
        raise ValueError(
            f"{subset_name}: data file has {len(rows)} entries but the manifest "
            f"records {info['entries']} — regenerate the manifest (splits must "
            "never be recomputed against drifted data)"
        )
    actual_sha = _file_sha256(path)
    if actual_sha != info["file_sha256"]:
        raise ValueError(
            f"{subset_name}: data file sha256 {actual_sha[:12]}… does not match "
            f"the manifest's {info['file_sha256'][:12]}… — same entry count but "
            "different content; regenerate the manifest"
        )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
