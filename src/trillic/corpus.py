"""Training corpus schema v1: train-half extraction, manifest, validation.

A training corpus file is jsonl: one JSON object per non-blank line. Schema
(issue #16; the golden schema's mirror on the training side):

    {"id": "train-<subset>-<fp8>",     # unique, non-empty
     "load_type": "rag" | "system_prompt" | "dialogue",
     "prompt": "...",                  # the text under compression
     "question": "",                   # query-aware reserved field (硬约束 3)
     "task": "",                       # query-aware reserved field (硬约束 3)
     "source": {                       # provenance, per-entry license trail
       "dataset": "...",               # e.g. "LongBench" or "synthetic"
       "subset": "...",
       "license": "...",
       "split": "train",               # training draws only from the train half
       "content_sha1": "...",          # REQUIRED here: the split-discipline key
     }}

question/task stay empty in v1 (task-agnostic line); v2's query-aware
extension (`[task; context]` samples, labels on context only) fills them —
the schema reserves the fields from day one so that extension never
rewrites the corpus format.

Discipline encoded here, not in reviewers' memory (docs/data-strategy.md):
  - entries come ONLY from the train half — the eval half is golden's
    (硬约束 4), re-proved OFFLINE by the registry's train boundary;
  - MeetingBank is hard-excluded anywhere in source (硬约束 1: it is the
    base checkpoint's training set);
  - subsets without train-use permission (qasper) never enter the corpus
    (硬约束 5) — they are recorded as excluded in the manifest instead;
  - zero fingerprint overlap with golden is asserted by the validator.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from trillic.golden import GoldenError, load_golden
from trillic.longbench import (
    LongBenchError,
    load_verified_split,
    render_prompt,
)
from trillic.splits import row_fingerprint

LOAD_TYPES = ("rag", "system_prompt", "dialogue")
_SOURCE_REQUIRED = ("dataset", "subset", "license", "split", "content_sha1")
_V1_QUERY_POLICY = (
    "question/task are reserved query-aware fields and are always empty in "
    "v1 (task-agnostic line); v2 fills them — schema fixed from day one "
    "(data-strategy 硬约束 3)"
)


class CorpusError(ValueError):
    """Training corpus construction or validation failed."""


@dataclass(frozen=True)
class CorpusItem:
    id: str
    load_type: str
    prompt: str
    question: str
    task: str
    source: dict


def build_training_entries(manifest: dict, *, data_dir: Path) -> list[dict]:
    """Extract the TRAIN half of every train-permitted LongBench subset.

    Mirrors trillic.longbench.build_rag_entries (golden side): the manifest
    is the split record and must describe the exact bytes being split;
    entries are emitted sorted by fingerprint; subsets with train_use=false
    are skipped here and recorded as excluded by build_training_manifest.
    """
    data_dir = Path(data_dir)
    permitted = [s for s in manifest.get("subsets", []) if s.get("train_use")]
    if not permitted:
        raise CorpusError(
            "no train-permitted subsets in the manifest — the training corpus "
            "would be empty (check train_use verdicts in the subset registry)"
        )
    entries: list[dict] = []
    for info in permitted:
        subset = info["name"]
        try:
            eval_rows, train_rows = load_verified_split(subset, info, data_dir)
        except LongBenchError as e:
            raise CorpusError(str(e)) from e
        eval_fps = {row_fingerprint(row) for row in eval_rows}
        emitted: set[str] = set()
        for row in train_rows:
            fingerprint = row_fingerprint(row)
            if fingerprint in emitted:
                continue  # duplicate source content: train once
            emitted.add(fingerprint)
            entries.append(
                {
                    "id": f"train-{subset}-{fingerprint[:8]}",
                    "load_type": "rag",
                    "prompt": render_prompt(
                        subset, context=row["context"], question=row["input"]
                    ),
                    "question": "",
                    "task": "",
                    "source": {
                        "dataset": "LongBench",
                        "subset": subset,
                        "license": info["license"],
                        "split": "train",
                        "content_sha1": fingerprint,
                    },
                }
            )
        assert not emitted & eval_fps, f"{subset}: split discipline violated"
        assert emitted == {row_fingerprint(r) for r in train_rows}
    return entries


def write_training_artifacts(
    manifest: dict,
    *,
    data_dir: Path,
    corpus_path: Path,
    manifest_path: Path,
    repo_commit: str | None = None,
    repo_dirty: bool | None = None,
    record_file: str | None = None,
) -> dict:
    """Build + write the corpus jsonl and its manifest in one step.

    Shared by `trillic corpus build` and scripts/build_training_corpus.py
    so the CLI and the pinned invocation can never drift apart.
    record_file overrides the path string recorded in the manifest (the
    script records the repo-relative identity instead of an absolute path).
    Returns the training manifest dict.
    """
    entries = build_training_entries(manifest, data_dir=data_dir)
    corpus_path = Path(corpus_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
        encoding="utf-8",
    )
    training_manifest = build_training_manifest(
        manifest,
        corpus_path,
        repo_commit=repo_commit,
        repo_dirty=repo_dirty,
    )
    if record_file is not None:
        training_manifest["corpus"]["file"] = record_file
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(training_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return training_manifest


def golden_fingerprints(paths: list[Path]) -> dict[str, str]:
    """Map content fingerprint -> golden entry id over golden jsonl files.

    Shared by `corpus validate --golden` and the build script's zero-overlap
    self-check; entries without source.content_sha1 are skipped (nothing to
    intersect).
    """
    fingerprints: dict[str, str] = {}
    try:
        for path in paths:
            for item in load_golden(path):
                fingerprint = item.source.get("content_sha1")
                if isinstance(fingerprint, str) and fingerprint.strip():
                    fingerprints[fingerprint] = item.id
    except GoldenError as e:
        raise CorpusError(str(e)) from e
    return fingerprints


def build_training_manifest(
    manifest: dict,
    corpus_path: Path,
    *,
    repo_commit: str | None = None,
    repo_dirty: bool | None = None,
) -> dict:
    """Manifest for a built corpus file: identity, licenses, exclusions.

    The manifest is the training-side split record: it pins the corpus
    bytes (sha256), the per-subset train boundary (offline re-proof of
    half membership), the license + evidence trail, and the exclusion
    statements MeetingBank/customer-prompt audits expect (same standard
    as golden's freeze record).
    """
    corpus_path = Path(corpus_path)
    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_subset: dict[str, int] = {}
    load_types: dict[str, int] = {}
    for row in rows:
        per_subset[row["source"]["subset"]] = per_subset.get(row["source"]["subset"], 0) + 1
        load_types[row["load_type"]] = load_types.get(row["load_type"], 0) + 1

    subsets = []
    excluded = []
    for info in manifest.get("subsets", []):
        record = {
            "name": info["name"],
            "file": info["file"],
            "file_sha256": info["file_sha256"],
            "eval_half": info["eval_half"],
            "train_half": info["train_half"],
            "train_boundary_fingerprint": info["train_boundary_fingerprint"],
            "license": info["license"],
            "train_use": info["train_use"],
            "train_use_evidence": list(info.get("train_use_evidence", [])),
            "note": info.get("note", ""),
        }
        if info.get("train_use"):
            record["entries"] = per_subset.get(info["name"], 0)
            subsets.append(record)
        else:
            record["reason"] = (
                f"train_use=false — excluded from training; eval-only "
                "(data-strategy 硬约束 5, see license + evidence)"
            )
            excluded.append(record)
    return {
        "schema_version": 1,
        "split_method": manifest.get("split_method", ""),
        "generated_by": {
            "tool": "trillic.corpus",
            "repo_commit": repo_commit,
            "repo_dirty": repo_dirty,
        },
        "corpus": {
            "file": str(corpus_path),
            "sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "entries": len(rows),
            "load_types": load_types,
        },
        "query_aware": {
            "fields": ["question", "task"],
            "v1_empty": True,
            "v1_policy": _V1_QUERY_POLICY,
        },
        "subsets": subsets,
        "excluded_subsets": excluded,
        "exclusions": {
            "meetingbank": "hard-rejected by the training-corpus validator "
            "(docs/data-strategy.md 硬约束 1: base-checkpoint training corpus)",
            "customer_prompts": "never used for training or eval (硬约束 2)",
            "non_train_permitted": "subsets with train_use=false are recorded "
            "under excluded_subsets and hard-rejected by the validator",
        },
        "split_discipline": {
            "train_half": "every entry asserts split=='train' and is re-proved "
            "offline: content_sha1 >= train_boundary_fingerprint",
            "golden_zero_overlap": "content_sha1 intersection with golden "
            "asserted empty by `trillic corpus validate --golden`",
        },
    }


def collect_corpus_errors(
    path: Path,
    *,
    registry: dict | None = None,
    golden_fingerprints: dict[str, str] | None = None,
) -> list[str]:
    """Validate a training corpus file; ALL errors, empty list = valid.

    Layers:
      1. per-entry schema (errors name the entry id);
      2. MeetingBank hard reject + split=='train' + duplicate ids/fingerprints;
      3. registry (training manifest): frozen-bytes sha256, per-subset counts,
         offline train-boundary re-proof, license trail, exclusion of
         train_use=false subsets;
      4. golden zero-overlap by content fingerprint (硬约束 4).
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [f"{path}: no such file"]

    subsets_by_name = {s["name"]: s for s in (registry or {}).get("subsets", [])}
    excluded_by_name = {s["name"]: s for s in (registry or {}).get("excluded_subsets", [])}
    # v1 manifests pin the "question/task always empty" policy (硬约束 3);
    # a v2 manifest drops/flips the flag and the same validator accepts it.
    require_empty_query = bool((registry or {}).get("query_aware", {}).get("v1_empty"))
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_fps: dict[str, str] = {}
    actual_counts: dict[str, int] = {}

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"{path}: line {line_number}: cannot parse JSON: {e}")
            continue
        if not isinstance(row, dict):
            errors.append(f"{path}: line {line_number}: each row must be a JSON object")
            continue

        entry_id = row.get("id")
        where = f"{path}: line {line_number}"
        if isinstance(entry_id, str) and entry_id.strip():
            where += f" (id={entry_id!r})"

        def fail(field: str, reason: str) -> None:
            errors.append(f"{where}: {field} {reason}")

        item = _validate_row(row, fail)
        if item is None:
            continue

        if require_empty_query and (item.question.strip() or item.task.strip()):
            fail(
                "question/task",
                "must be empty in a v1 corpus (query-aware reserved fields; "
                "only the v2 extension fills them — see manifest query_aware)",
            )

        if item.id in seen_ids:
            fail("id", f"duplicate id {item.id!r} (ids must be unique in the corpus)")
        else:
            seen_ids.add(item.id)
        if item.source["content_sha1"] in seen_fps:
            fail(
                "source.content_sha1",
                f"duplicate content fingerprint {item.source['content_sha1']} "
                f"(already emitted as {seen_fps[item.source['content_sha1']]!r} — "
                "each source row trains once)",
            )
        else:
            seen_fps[item.source["content_sha1"]] = item.id

        subset = item.source["subset"]
        actual_counts[subset] = actual_counts.get(subset, 0) + 1

        if registry is not None:
            _registry_checks(item, fail, subsets_by_name, excluded_by_name)

        if golden_fingerprints and item.source["content_sha1"] in golden_fingerprints:
            fail(
                "source.content_sha1",
                f"overlap with golden entry {golden_fingerprints[item.source['content_sha1']]!r} "
                "(training data must never share content with the exam — "
                "data-strategy 硬约束 4)",
            )

    if not seen_ids and not errors:
        errors.append(
            f"{path}: no entries found — an empty training corpus is a mistake "
            "(wrong path or truncated file)"
        )

    if registry is not None:
        _registry_totals(path, registry, actual_counts, errors)
    return errors


def _validate_row(row: dict, fail) -> CorpusItem | None:
    """Schema layer; collects every field error, returns the item only when
    the row is fully usable downstream (fingerprint/id/count bookkeeping)."""
    usable = True
    entry_id = row.get("id")
    if not isinstance(entry_id, str) or not entry_id.strip():
        fail("id", "must be a non-empty string")
        return None
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        fail("prompt", "must be a non-empty string")
        usable = False
    load_type = row.get("load_type")
    if load_type not in LOAD_TYPES:
        fail("load_type", f"must be one of {list(LOAD_TYPES)}, got {load_type!r}")
        usable = False
    for field in ("question", "task"):
        value = row.get(field)
        if not isinstance(value, str):
            fail(
                field,
                "must be a string (query-aware reserved field: empty in v1, "
                "filled by the v2 extension — never absent)",
            )
            usable = False
    source = row.get("source")
    if not isinstance(source, dict):
        fail("source", "must be an object with dataset/subset/license/split/content_sha1")
        return None
    usable = True
    for field in _SOURCE_REQUIRED:
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            fail("source", f"field {field!r} must be a non-empty string")
            usable = False
    if "meetingbank" in f"{source.get('dataset', '')}/{source.get('subset', '')}".lower():
        fail(
            "source",
            "MeetingBank is excluded by hard constraint (base checkpoint "
            "training set — see docs/data-strategy.md 硬约束 1)",
        )
        usable = False
    if source.get("split") != "train":
        fail(
            "source",
            f"split must be 'train' in a training corpus, got "
            f"{source.get('split')!r} (the eval half belongs to golden)",
        )
        usable = False
    if not usable:
        return None
    assert isinstance(prompt, str) and isinstance(load_type, str)  # usable implies checked
    return CorpusItem(
        id=entry_id,
        load_type=load_type,
        prompt=prompt,
        question=row.get("question", ""),
        task=row.get("task", ""),
        source=dict(source),
    )


def _registry_checks(item: CorpusItem, fail, subsets_by_name: dict, excluded_by_name: dict) -> None:
    """Per-entry registry layer: boundary re-proof + license/exclusion trail."""
    subset = item.source["subset"]
    if subset in excluded_by_name:
        info = excluded_by_name[subset]
        fail(
            "source.subset",
            f"subset {subset!r} is excluded from training (license: "
            f"{info.get('license', '?')}; train_use=false — data-strategy "
            "硬约束 5)",
        )
        return
    info = subsets_by_name.get(subset)
    if info is None:
        fail(
            "source.subset",
            f"subset {subset!r} is not in the training registry — every entry "
            "needs a registered, train-permitted source",
        )
        return
    if item.source.get("license") != info.get("license"):
        fail(
            "source.license",
            f"license trail drift: entry says {item.source.get('license')!r}, "
            f"registry says {info.get('license')!r} (regenerate corpus or "
            "manifest)",
        )
    boundary = info.get("train_boundary_fingerprint")
    fingerprint = item.source["content_sha1"]
    if boundary is None or fingerprint < boundary:
        fail(
            "source.content_sha1",
            f"fingerprint {fingerprint[:12]}… is below the train boundary "
            f"{str(boundary)[:12] if boundary else '∅'}… — this content "
            "belongs to the EVAL half (golden's side of the split)",
        )


def _registry_totals(path: Path, registry: dict, actual_counts: dict, errors: list[str]) -> None:
    """Corpus-level registry layer: frozen bytes + per-subset counts."""
    corpus = registry.get("corpus", {})
    expected_sha = corpus.get("sha256")
    if expected_sha:
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            errors.append(
                f"{path}: corpus sha256 {actual_sha[:12]}… does not match the "
                f"manifest's {expected_sha[:12]}… — the manifest is the split "
                "record; regenerate it (never edit the corpus by hand)"
            )
    for info in registry.get("subsets", []):
        expected = info.get("entries")
        if expected is None:
            continue
        actual = actual_counts.get(info["name"], 0)
        if actual != expected:
            errors.append(
                f"{path}: subset {info['name']!r} has {actual} entries but the "
                f"manifest records {expected} — regenerate the manifest"
            )
