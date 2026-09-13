"""Build the v1 training-corpus base layer (issue #16).

Pinned invocation for `trillic corpus build` + immediate full validation:

  - extracts the TRAIN half of every train-permitted LongBench subset
    (hotpotqa + gov_report; qasper is license-excluded, eval-only);
  - writes training/corpus/longbench-train-v1.jsonl and
    training/manifests/training-corpus-v1.json (corpus sha256, per-subset
    train boundary, license + evidence trail, exclusions);
  - validates: schema, MeetingBank hard-reject, offline train-boundary
    re-proof, frozen-bytes sha256, and ZERO content overlap with the
    frozen golden files (the exam's eval half).

Zero gateway calls by construction: pure file transforms over _downloads/.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

LB_MANIFEST = REPO / "eval" / "manifests" / "longbench.json"
DATA_DIR = REPO / "_downloads" / "data"
CORPUS_OUT = REPO / "training" / "corpus" / "longbench-train-v1.jsonl"
MANIFEST_OUT = REPO / "training" / "manifests" / "training-corpus-v1.json"
GOLDEN_FILES = [
    REPO / "eval" / "golden" / "rag_pilot.jsonl",
    REPO / "eval" / "golden" / "rag_scaled.jsonl",
    REPO / "eval" / "golden" / "sysprompt_pilot.jsonl",
    REPO / "eval" / "golden" / "sysprompt_scaled.jsonl",
    REPO / "eval" / "golden" / "dialogue_pilot.jsonl",
    REPO / "eval" / "golden" / "dialogue_scaled.jsonl",
]


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


def main() -> int:
    from trillic.corpus import (
        build_training_entries,
        build_training_manifest,
        collect_corpus_errors,
    )
    from trillic.golden import load_golden

    commit, dirty = _repo_state()
    lb_manifest = json.loads(LB_MANIFEST.read_text(encoding="utf-8"))
    entries = build_training_entries(lb_manifest, data_dir=DATA_DIR)
    CORPUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_OUT.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
        encoding="utf-8",
    )
    training_manifest = build_training_manifest(
        lb_manifest, CORPUS_OUT, repo_commit=commit, repo_dirty=dirty
    )
    # record the repo-relative identity, not the machine-specific absolute path
    training_manifest["corpus"]["file"] = CORPUS_OUT.relative_to(REPO).as_posix()
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(
        json.dumps(training_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    golden_fingerprints: dict[str, str] = {}
    for path in GOLDEN_FILES:
        for item in load_golden(path):
            fingerprint = item.source.get("content_sha1")
            if isinstance(fingerprint, str) and fingerprint.strip():
                golden_fingerprints[fingerprint] = item.id
    errors = collect_corpus_errors(
        CORPUS_OUT, registry=training_manifest, golden_fingerprints=golden_fingerprints
    )
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1

    summary = {
        "corpus": str(CORPUS_OUT.relative_to(REPO)),
        "entries": training_manifest["corpus"]["entries"],
        "subsets": {s["name"]: s["entries"] for s in training_manifest["subsets"]},
        "excluded": [s["name"] for s in training_manifest["excluded_subsets"]],
        "golden_fingerprints_checked": len(golden_fingerprints),
        "manifest": str(MANIFEST_OUT.relative_to(REPO)),
        "repo_commit": commit,
        "repo_dirty": dirty,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"written: {CORPUS_OUT}\nwritten: {MANIFEST_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
