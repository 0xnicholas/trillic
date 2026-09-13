"""Build the v1 synthetic training layers (issue #17).

Pinned invocation for `trillic corpus build-synthetic` + immediate full
validation:

  - synthesizes the system_prompt + dialogue layers from TRAIN seeds only
    (per-family eval/train seed diversion; the committed family registries
    are the split authority);
  - scale per docs/training-mix.md: rag 30% / system_prompt 35% /
    dialogue 35%; with the frozen rag base (300 entries, issue #16) the
    anchor sizing is 350 + 350 synthetic entries;
  - writes training/corpus/synthetic-train-v1.jsonl and
    training/manifests/synthetic-training-v1.json (corpus sha256, per-family
    seed plans, mix record);
  - validates: schema, MeetingBank hard-reject, seed-domain re-proof,
    full regeneration of every entry, frozen-bytes sha256, and ZERO content
    overlap with the frozen golden files (whose list is read from the
    golden freeze record — the canonical golden identity).

Zero gateway calls by construction: deterministic template rendering only;
teacher-distillation spend belongs to later issues.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SYS_FAMILIES = REPO / "eval" / "manifests" / "sysprompt_families.toml"
DLG_FAMILIES = REPO / "eval" / "manifests" / "dialogue_families.toml"
GOLDEN_FREEZE = REPO / "eval" / "manifests" / "golden-freeze.json"
RAG_MANIFEST = REPO / "training" / "manifests" / "training-corpus-v1.json"
CORPUS_OUT = REPO / "training" / "corpus" / "synthetic-train-v1.jsonl"
MANIFEST_OUT = REPO / "training" / "manifests" / "synthetic-training-v1.json"

# docs/training-mix.md: rag 30 / system_prompt 35 / dialogue 35 (±5pt at
# sizing time). The shares live in trillic.synthetic.RATIO_TARGET_PCT
# (single authority, recorded in the manifest); this script derives the
# per-class anchor from the FROZEN rag base, not hard-coded blindly:
# 300 rag -> total 1000 -> 350 + 350.
RATIO_TOLERANCE_PCT = 5.0


def _repo_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        return commit, dirty
    except subprocess.SubprocessError:
        return None, None


def _targets(rag_base: int, ratio: dict[str, float]) -> dict[str, int]:
    """Anchor sizing from the mix doc, derived from the frozen rag base."""
    total = round(rag_base / (ratio["rag"] / 100.0))
    target = round(total * ratio["system_prompt"] / 100.0)
    if (rag_base, target) != (300, 350):
        raise SystemExit(
            f"rag base moved to {rag_base} — the 350/350 anchor no longer "
            f"holds (derived total {total}, per-class {target}); re-derive "
            "the targets from docs/training-mix.md and record the change as "
            "a new corpus version"
        )
    return {"system_prompt": target, "dialogue": target}


def main() -> int:
    from trillic.corpus import golden_fingerprints
    from trillic.dialogue import load_families as load_dialogue_families
    from trillic.synthetic import (
        RATIO_TARGET_PCT,
        collect_synthetic_errors,
        write_synthetic_artifacts,
    )
    from trillic.sysprompt import load_families as load_sysprompt_families

    commit, dirty = _repo_state()
    rag_manifest = json.loads(RAG_MANIFEST.read_text(encoding="utf-8"))
    rag_base = rag_manifest["corpus"]["entries"]

    freeze = json.loads(GOLDEN_FREEZE.read_text(encoding="utf-8"))
    golden_paths = [REPO / f for f in freeze["golden"]["files"]]
    fingerprints = golden_fingerprints(golden_paths)

    manifest = write_synthetic_artifacts(
        load_sysprompt_families(SYS_FAMILIES),
        load_dialogue_families(DLG_FAMILIES),
        targets=_targets(rag_base, RATIO_TARGET_PCT),
        golden_fingerprints=fingerprints,
        corpus_path=CORPUS_OUT,
        manifest_path=MANIFEST_OUT,
        rag_base=rag_base,
        repo_commit=commit,
        repo_dirty=dirty,
        # record the repo-relative identity, not the machine-specific path
        record_file=CORPUS_OUT.relative_to(REPO).as_posix(),
    )

    achieved = manifest["mix"]["achieved_pct"]
    for load_type, share in RATIO_TARGET_PCT.items():
        if abs(achieved[load_type] - share) > RATIO_TOLERANCE_PCT:
            print(
                f"error: achieved {load_type} share {achieved[load_type]:.1f}% "
                f"misses the {share:.1f}%±{RATIO_TOLERANCE_PCT:g}pt target",
                file=sys.stderr,
            )
            return 1

    errors = collect_synthetic_errors(
        CORPUS_OUT,
        sysprompt_families=load_sysprompt_families(SYS_FAMILIES),
        dialogue_families=load_dialogue_families(DLG_FAMILIES),
        manifest=manifest,
        golden_fingerprints=fingerprints,
    )
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1

    per_class_family_counts = {
        class_key: {f["name"]: f["entries"] for f in records}
        for class_key, records in manifest["families"].items()
    }
    summary = {
        "corpus": CORPUS_OUT.relative_to(REPO).as_posix(),
        "entries": manifest["corpus"]["entries"],
        "load_types": manifest["corpus"]["load_types"],
        "mix": {k: round(v, 2) for k, v in achieved.items()},
        "families": per_class_family_counts,
        "golden_fingerprints_checked": len(fingerprints),
        "manifest": MANIFEST_OUT.relative_to(REPO).as_posix(),
        "repo_commit": commit,
        "repo_dirty": dirty,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"written: {CORPUS_OUT}\nwritten: {MANIFEST_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
