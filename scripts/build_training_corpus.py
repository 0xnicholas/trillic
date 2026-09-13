"""Build the v1 training-corpus base layer (issue #16).

Pinned invocation for `trillic corpus build` + immediate full validation:

  - extracts the TRAIN half of every train-permitted LongBench subset
    (hotpotqa + gov_report; qasper is license-excluded, eval-only);
  - writes training/corpus/longbench-train-v1.jsonl and
    training/manifests/training-corpus-v1.json (corpus sha256, per-subset
    train boundary, license + evidence trail, exclusions);
  - validates: schema, MeetingBank hard-reject, offline train-boundary
    re-proof, frozen-bytes sha256, and ZERO content overlap with the
    frozen golden files (the exam's eval half; file list read from the
    golden freeze record — the canonical golden identity).

Zero gateway calls by construction: pure file transforms over _downloads/.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

LB_MANIFEST = REPO / "eval" / "manifests" / "longbench.json"
GOLDEN_FREEZE = REPO / "eval" / "manifests" / "golden-freeze.json"
DATA_DIR = REPO / "_downloads" / "data"
CORPUS_OUT = REPO / "training" / "corpus" / "longbench-train-v1.jsonl"
MANIFEST_OUT = REPO / "training" / "manifests" / "training-corpus-v1.json"


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
    from trillic.corpus import collect_corpus_errors, golden_fingerprints, write_training_artifacts

    commit, dirty = _repo_state()
    lb_manifest = json.loads(LB_MANIFEST.read_text(encoding="utf-8"))
    training_manifest = write_training_artifacts(
        lb_manifest,
        data_dir=DATA_DIR,
        corpus_path=CORPUS_OUT,
        manifest_path=MANIFEST_OUT,
        repo_commit=commit,
        repo_dirty=dirty,
        # record the repo-relative identity, not the machine-specific path
        record_file=CORPUS_OUT.relative_to(REPO).as_posix(),
    )

    freeze = json.loads(GOLDEN_FREEZE.read_text(encoding="utf-8"))
    golden_paths = [REPO / f for f in freeze["golden"]["files"]]
    fingerprints = golden_fingerprints(golden_paths)
    errors = collect_corpus_errors(
        CORPUS_OUT, registry=training_manifest, golden_fingerprints=fingerprints
    )
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1

    summary = {
        "corpus": CORPUS_OUT.relative_to(REPO).as_posix(),
        "entries": training_manifest["corpus"]["entries"],
        "subsets": {s["name"]: s["entries"] for s in training_manifest["subsets"]},
        "excluded": [s["name"] for s in training_manifest["excluded_subsets"]],
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
