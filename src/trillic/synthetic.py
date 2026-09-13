"""Synthetic training layers (issue #17): the training side of the family/
seed diversion.

docs/training-mix.md fixes the v1 mix at rag 30% / system_prompt 35% /
dialogue 35%; the rag base (LongBench train half, 300 entries) is frozen by
issue #16. This module generates the two synthetic layers on top:

  1. `select_train_seed_plan` — deterministic seed selection per class:
     round-robin over families in registry order, per family scanning the
     TRAIN seed range ascending. A seed is accepted only when its rendered
     prompt is content-distinct within the corpus AND absent from the
     golden fingerprints — small slot pools mean different seeds can render
     byte-identical prompts, so this is what keeps "zero overlap with
     golden" true at the CONTENT level, above the structural seed-domain
     split (hard constraint 4: 场景族/种子分流, both sides' guards raise on
     the other's seeds).
  2. `write_synthetic_artifacts` — corpus jsonl (training schema v1:
     question/task reserved and empty) + manifest (sha256, per-family seed
     plans, targets, achieved mix, exclusion statements). Same input,
     same bytes — the pinned script re-runs this and relies on it.
  3. `collect_synthetic_errors` — the training-side validator. Layers the
     generic corpus checks (schema, MeetingBank hard reject, split=='train',
     duplicate ids/fingerprints, golden content overlap) with the synthetic
     registry layer: family known, train_use, seed re-proved inside the
     family's TRAIN range (never eval), id rule, and full regeneration of
     every entry from (family, seed) — the manifest never blesses a
     hand-edited corpus.

Zero gateway calls by construction: everything below is deterministic
template rendering over committed files; the teacher-distillation spend
lives in later issues.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from trillic.corpus import collect_corpus_errors
from trillic.dialogue import DialogueError, synthesize_training_entry as _dlg_train
from trillic.sysprompt import (
    SysPromptError,
    synthesize_training_entry as _sys_train,
)

RATIO_TARGET_PCT = {"rag": 30.0, "system_prompt": 35.0, "dialogue": 35.0}
_RATIO_SOURCE_DOC = "docs/training-mix.md"

SELECTION_RULE = (
    "round-robin over families in registry order; per family, train seeds are "
    "scanned in ascending order from the family's train_seed_range and a seed "
    "is accepted only when sha1(prompt) is distinct within the corpus and "
    "absent from the golden fingerprints; selection stops at the class target; "
    "low-diversity families (small slot pools) simply contribute fewer entries"
)

_ZERO_GATEWAY = (
    "deterministic template rendering only — no network I/O exists in the "
    "generation path; teacher distillation spend belongs to later issues"
)


class SyntheticError(ValueError):
    """Synthetic-layer construction or validation failed."""


@dataclass(frozen=True)
class _ClassSpec:
    key: str  # manifest/plan key == corpus load_type
    id_prefix: str
    families: list[dict]
    synthesize: Callable[[dict, int], dict]  # (family, seed) -> training entry


def _class_specs(
    sysprompt_families: list[dict], dialogue_families: list[dict]
) -> list[_ClassSpec]:
    return [
        _ClassSpec("system_prompt", "sys", sysprompt_families, _sys_train),
        _ClassSpec("dialogue", "dlg", dialogue_families, _dlg_train),
    ]


def select_train_seed_plan(
    sysprompt_families: list[dict],
    dialogue_families: list[dict],
    *,
    targets: dict[str, int],
    golden_fingerprints: dict[str, str] | None = None,
) -> dict[str, dict[str, list[int]]]:
    """Select the per-class {family: [seeds]} plan (train seeds only).

    Deterministic for (registries, targets, golden fingerprints): same
    inputs, same plan, byte for byte. Families with train_use=false are
    never selected (硬约束 5 mirrors the subset-level exclusion).
    """
    specs = _class_specs(sysprompt_families, dialogue_families)
    known = {spec.key for spec in specs}
    unknown = sorted(set(targets) - known)
    if unknown:
        raise SyntheticError(
            f"targets reference unknown classes {unknown} — known: {sorted(known)}"
        )
    for key, target in targets.items():
        if not isinstance(target, int) or target < 0:
            raise SyntheticError(
                f"target for {key!r} must be a non-negative int, got {target!r}"
            )

    forbidden = set(golden_fingerprints or {})
    plan: dict[str, dict[str, list[int]]] = {}
    seen: set[str] = set(forbidden)  # shared across classes: content is
    # distinct across the WHOLE corpus, not just per class
    for spec in specs:
        plan[spec.key] = _select_class(spec, targets[spec.key], seen)
    return plan


def _select_class(
    spec: _ClassSpec, target: int, seen: set[str]
) -> dict[str, list[int]]:
    """Round-robin content-distinct selection for one class.

    `seen` accumulates accepted content fingerprints (seeded with the
    golden ones by the caller) and is shared across classes, so the
    whole corpus stays content-distinct, not just each class.
    """
    eligible = [f for f in spec.families if f.get("train_use")]
    if not eligible:
        raise SyntheticError(
            f"class {spec.key!r}: no train_use families in the registry — "
            "nothing to synthesize from"
        )
    by_name = {f["name"]: f for f in eligible}
    plan: dict[str, list[int]] = {f["name"]: [] for f in eligible}
    cursor = {f["name"]: f["train_seed_range"][0] for f in eligible}
    active = [f["name"] for f in eligible]
    accepted = 0
    while accepted < target and active:
        progressed = False
        for name in list(active):
            family = by_name[name]
            seed = _next_distinct_seed(spec, family, cursor[name], seen)
            if seed is None:
                active.remove(name)
                continue
            cursor[name] = seed + 1
            plan[name].append(seed)
            accepted += 1
            progressed = True
            if accepted >= target:
                break
        if not progressed:
            break
    if accepted < target:
        raise SyntheticError(
            f"target {target} for {spec.key!r} is unreachable: train seed "
            f"ranges exhausted at {accepted} content-distinct entries "
            f"(slot pools cap per-family diversity) — widen the "
            "train_seed_range in the family registry or lower the target"
        )
    return plan


def _next_distinct_seed(
    spec: _ClassSpec, family: dict, start: int, seen: set[str]
) -> int | None:
    """First seed >= start whose rendered content is unseen, else None."""
    lo, hi = family["train_seed_range"]
    for seed in range(max(start, lo), hi + 1):
        fingerprint = spec.synthesize(family, seed)["source"]["content_sha1"]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        return seed
    return None


def build_synthetic_entries(
    sysprompt_families: list[dict],
    dialogue_families: list[dict],
    seed_plan: dict[str, dict[str, list[int]]],
) -> list[dict]:
    """Render the corpus entries for a seed plan, canonical order:
    class (system_prompt, then dialogue) -> registry family order -> seed."""
    entries: list[dict] = []
    for spec in _class_specs(sysprompt_families, dialogue_families):
        by_name = {f["name"]: f for f in spec.families}
        class_plan = seed_plan.get(spec.key, {})
        unknown = sorted(set(class_plan) - set(by_name))
        if unknown:
            raise SyntheticError(
                f"seed plan for {spec.key!r} references families not in the "
                f"registry: {unknown}"
            )
        for family in spec.families:
            for seed in sorted(class_plan.get(family["name"], [])):
                entries.append(spec.synthesize(family, seed))
    return entries


def write_synthetic_artifacts(
    sysprompt_families: list[dict],
    dialogue_families: list[dict],
    *,
    targets: dict[str, int],
    golden_fingerprints: dict[str, str] | None,
    corpus_path: Path,
    manifest_path: Path,
    rag_base: int | None = None,
    repo_commit: str | None = None,
    repo_dirty: bool | None = None,
    record_file: str | None = None,
) -> dict:
    """Select, render, and write the synthetic corpus + manifest in one step.

    Shared by `trillic corpus build-synthetic` and the pinned
    scripts/build_synthetic_training.py so the CLI and the frozen
    invocation can never drift apart. record_file overrides the path
    string recorded in the manifest (repo-relative identity, not an
    absolute machine path). Returns the manifest dict.
    """
    seed_plan = select_train_seed_plan(
        sysprompt_families,
        dialogue_families,
        targets=targets,
        golden_fingerprints=golden_fingerprints,
    )
    entries = build_synthetic_entries(sysprompt_families, dialogue_families, seed_plan)

    corpus_path = Path(corpus_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
        encoding="utf-8",
    )
    manifest = build_synthetic_manifest(
        sysprompt_families,
        dialogue_families,
        seed_plan=seed_plan,
        targets=targets,
        corpus_path=corpus_path,
        rag_base=rag_base,
        repo_commit=repo_commit,
        repo_dirty=repo_dirty,
    )
    if record_file is not None:
        manifest["corpus"]["file"] = record_file
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_synthetic_manifest(
    sysprompt_families: list[dict],
    dialogue_families: list[dict],
    *,
    seed_plan: dict[str, dict[str, list[int]]],
    targets: dict[str, int],
    corpus_path: Path,
    rag_base: int | None = None,
    repo_commit: str | None = None,
    repo_dirty: bool | None = None,
) -> dict:
    """Manifest for a built synthetic corpus: identity, seed plans, mix."""
    corpus_path = Path(corpus_path)
    rows = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    load_types: dict[str, int] = {}
    for row in rows:
        load_types[row["load_type"]] = load_types.get(row["load_type"], 0) + 1

    families: dict[str, list[dict]] = {}
    excluded_families: dict[str, list[dict]] = {}
    for spec in _class_specs(sysprompt_families, dialogue_families):
        class_families = []
        class_excluded = []
        class_plan = seed_plan.get(spec.key, {})
        for family in spec.families:
            record = {
                "name": family["name"],
                "license": family["license"],
                "train_use": family["train_use"],
                "train_seed_range": family["train_seed_range"],
                "eval_seed_range": family["eval_seed_range"],
            }
            if family["train_use"]:
                seeds = sorted(class_plan.get(family["name"], []))
                record["seed_plan"] = seeds
                record["entries"] = len(seeds)
                class_families.append(record)
            else:
                record["reason"] = (
                    "train_use=false — family excluded from training synthesis "
                    "(same standard as subset-level 硬约束 5)"
                )
                class_excluded.append(record)
        families[spec.key] = class_families
        excluded_families[spec.key] = class_excluded

    mix: dict = {
        "source_doc": _RATIO_SOURCE_DOC,
        "ratio_target_pct": dict(RATIO_TARGET_PCT),
        "rag_base": rag_base,
    }
    if rag_base is not None:
        if rag_base < 0:
            raise SyntheticError(f"rag_base must be non-negative, got {rag_base!r}")
        total = rag_base + len(rows)
        mix["achieved_pct"] = {
            "rag": 100.0 * rag_base / total,
            "system_prompt": 100.0 * load_types.get("system_prompt", 0) / total,
            "dialogue": 100.0 * load_types.get("dialogue", 0) / total,
        }

    return {
        "schema_version": 1,
        "method": (
            "shared scenario-family generators (src/trillic/sysprompt.py + "
            "src/trillic/dialogue.py), train-seed side of the split: same "
            "templates and seeded slot machinery as golden, train seeds only "
            "(硬约束 4: 场景族/种子分流 — both sides' guards raise on the "
            "other's seeds)"
        ),
        "entry_id_rule": "sys-<family>-s<seed> / dlg-<family>-s<seed> (seed embeds the train-domain seed)",
        "rng": "random.Random(int.from_bytes(sha256('<family>:<seed>')[:8]))",
        "content_addressing": "sha1(prompt)",
        "selection_rule": SELECTION_RULE,
        "targets": dict(targets),
        "generated_by": {
            "tool": "trillic.synthetic",
            "repo_commit": repo_commit,
            "repo_dirty": repo_dirty,
        },
        "corpus": {
            "file": str(corpus_path),
            "sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "entries": len(rows),
            "load_types": load_types,
        },
        "mix": mix,
        "query_aware": {
            "fields": ["question", "task"],
            "v1_empty": True,
            "v1_policy": (
                "question/task are reserved query-aware fields and are always "
                "empty in v1 (task-agnostic line); v2 fills them — schema "
                "fixed from day one (data-strategy 硬约束 3)"
            ),
        },
        "families": families,
        "excluded_families": excluded_families,
        "exclusions": {
            "meetingbank": (
                "hard-rejected by the training-corpus validator "
                "(docs/data-strategy.md 硬约束 1: base-checkpoint training corpus)"
            ),
            "customer_prompts": "never used for training or eval (硬约束 2)",
            "golden_seeds": (
                "train seed ranges are disjoint from eval ranges in the family "
                "registries (the split authority); validators on both sides "
                "raise on the other's seeds"
            ),
        },
        "split_discipline": {
            "seed_domain": (
                "every entry's seed is re-proved offline to sit inside its "
                "family's train_seed_range — never the eval range (硬约束 4)"
            ),
            "zero_overlap": (
                "zero content overlap with golden asserted two ways: the "
                "selection stage skips golden-identical prompts, and the "
                "validator intersects content_sha1 with golden fingerprints"
            ),
            "reproducible": (
                "same (registries, targets, golden fingerprints) reproduces "
                "the same corpus bytes; the manifest pins them with sha256"
            ),
        },
        "zero_gateway_calls": _ZERO_GATEWAY,
    }


def collect_synthetic_errors(
    path: Path,
    *,
    sysprompt_families: list[dict] | None = None,
    dialogue_families: list[dict] | None = None,
    manifest: dict | None = None,
    golden_fingerprints: dict[str, str] | None = None,
) -> list[str]:
    """Validate a synthetic training corpus; ALL errors, empty list = valid.

    Layers:
      1. generic training-corpus checks via trillic.corpus (schema v1,
         MeetingBank hard reject, split=='train', duplicate ids and content
         fingerprints, golden content overlap);
      2. synthetic registry layer: family known + train_use, seed re-proved
         inside the family's TRAIN range, id rule, regeneration match —
         the corpus is never trusted over the (registry, generator);
      3. manifest layer: frozen-bytes sha256, per-family entries, corpus
         totals, and achieved-mix consistency.
    """
    path = Path(path)
    errors = collect_corpus_errors(path, golden_fingerprints=golden_fingerprints)
    if sysprompt_families is None and dialogue_families is None and manifest is None:
        return errors

    specs: list[_ClassSpec] = []
    if sysprompt_families is not None:
        specs.append(_ClassSpec("system_prompt", "sys", sysprompt_families, _sys_train))
    if dialogue_families is not None:
        specs.append(_ClassSpec("dialogue", "dlg", dialogue_families, _dlg_train))

    require_empty_query = bool((manifest or {}).get("query_aware", {}).get("v1_empty"))
    actual_family_counts: dict[str, dict[str, int]] = {
        spec.key: {} for spec in specs
    }
    actual_totals: dict[str, int] = {spec.key: 0 for spec in specs}

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return errors  # the generic layer already reported the unreadable file

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # already reported by the generic layer
        if not isinstance(row, dict):
            continue
        entry_id = row.get("id")
        where = f"{path}: line {line_number}"
        if isinstance(entry_id, str) and entry_id.strip():
            where += f" (id={entry_id!r})"

        spec = None
        for candidate in specs:
            if row.get("load_type") == candidate.key:
                spec = candidate
                break
        if spec is None:
            continue  # load_type errors already reported by the generic layer

        source = row.get("source", {})
        family_name = source.get("family", source.get("subset", ""))
        family = next((f for f in spec.families if f["name"] == family_name), None)
        if family is None:
            errors.append(
                f"{where}: source.family {family_name!r} is not in the "
                f"{spec.key} family registry — every synthetic entry needs a "
                "registered family"
            )
            continue
        if not family["train_use"]:
            errors.append(
                f"{where}: source.family {family_name!r} has train_use=false — "
                "excluded from training synthesis (硬约束 5)"
            )
            continue

        seed = source.get("seed")
        error = _seed_domain_error(spec, family, seed)
        if error is not None:
            errors.append(f"{where}: source.seed {error}")
            continue

        assert isinstance(seed, int)
        try:
            regenerated = spec.synthesize(family, seed)
        except (SysPromptError, DialogueError) as e:
            errors.append(f"{where}: regeneration refused: {e}")
            continue
        if row.get("prompt") != regenerated["prompt"] or source.get(
            "content_sha1"
        ) != regenerated["source"]["content_sha1"]:
            errors.append(
                f"{where}: content no longer matches regeneration from "
                "(registry, family, seed) — the corpus was hand-edited or the "
                "generator drifted; regenerate deliberately, never silently"
            )
        if entry_id != regenerated["id"]:
            errors.append(
                f"{where}: id must be {regenerated['id']!r} "
                f"(rule: {spec.id_prefix}-<family>-s<seed>)"
            )
        if source.get("license") != family["license"]:
            errors.append(
                f"{where}: source.license drift: entry says "
                f"{source.get('license')!r}, registry says {family['license']!r}"
            )
        if require_empty_query and (
            str(row.get("question", "")).strip() or str(row.get("task", "")).strip()
        ):
            errors.append(
                f"{where}: question/task must be empty in a v1 corpus "
                "(query-aware reserved fields; only the v2 extension fills "
                "them — see manifest query_aware)"
            )

        actual_family_counts[spec.key][family_name] = (
            actual_family_counts[spec.key].get(family_name, 0) + 1
        )
        actual_totals[spec.key] += 1

    if manifest is not None:
        _manifest_checks(path, manifest, specs, actual_family_counts, actual_totals, errors)
    return errors


def _seed_domain_error(spec: _ClassSpec, family: dict, seed) -> str | None:
    """Precise seed-domain reason, or None when the seed is train-side."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        return f"{seed!r} must be an int"
    name = family["name"]
    eval_lo, eval_hi = family["eval_seed_range"]
    if eval_lo <= seed <= eval_hi:
        return (
            f"{seed} is an EVAL seed (family {name!r}) — training synthesis "
            "draws from train seeds only (硬约束 4: 种子分流)"
        )
    train_lo, train_hi = family["train_seed_range"]
    if not (train_lo <= seed <= train_hi):
        return (
            f"{seed} is outside both seed pools (family {name!r}: eval "
            f"{family['eval_seed_range']}, train {family['train_seed_range']})"
        )
    return None


def _manifest_checks(
    path: Path,
    manifest: dict,
    specs: list[_ClassSpec],
    actual_family_counts: dict[str, dict[str, int]],
    actual_totals: dict[str, int],
    errors: list[str],
) -> None:
    """Corpus-level manifest layer: frozen bytes + counts + mix consistency."""
    corpus = manifest.get("corpus", {})
    expected_sha = corpus.get("sha256")
    if expected_sha:
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            errors.append(
                f"{path}: corpus sha256 {actual_sha[:12]}… does not match the "
                f"manifest's {expected_sha[:12]}… — the manifest is the split "
                "record; regenerate it (never edit the corpus by hand)"
            )
    for spec in specs:
        for record in manifest.get("families", {}).get(spec.key, []):
            expected = record.get("entries")
            if expected is None:
                continue
            actual = actual_family_counts[spec.key].get(record["name"], 0)
            if actual != expected:
                errors.append(
                    f"{path}: family {record['name']!r} ({spec.key}) has "
                    f"{actual} entries but the manifest records {expected} — "
                    "regenerate the manifest"
                )
    if "entries" in corpus:
        actual_rows = sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if actual_rows != corpus["entries"]:
            errors.append(
                f"{path}: corpus has {actual_rows} entries but the manifest "
                f"records {corpus['entries']} — regenerate the manifest"
            )
    mix = manifest.get("mix", {})
    achieved = mix.get("achieved_pct")
    rag_base = mix.get("rag_base")
    if achieved is not None and isinstance(rag_base, int):
        total = rag_base + sum(actual_totals.values())
        recomputed = {
            "rag": 100.0 * rag_base / total if total else 0.0,
            "system_prompt": 100.0 * actual_totals.get("system_prompt", 0) / total
            if total
            else 0.0,
            "dialogue": 100.0 * actual_totals.get("dialogue", 0) / total if total else 0.0,
        }
        if any(abs(recomputed[k] - achieved.get(k, -1.0)) > 1e-9 for k in recomputed):
            errors.append(
                f"{path}: manifest mix.achieved_pct {achieved} does not match "
                f"the corpus ({recomputed}) — regenerate the manifest"
            )
