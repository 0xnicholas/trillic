"""Freeze the golden exam (issue #9): content-addressed freeze record.

Writes eval/manifests/golden-freeze.json pinning:
- the golden file list (argv order) + combined sha256 (the exact caliber
  run reports record as golden.sha256),
- the repo commit whose reports produced the baseline numbers,
- the quality pins (answer/judge models, rubric sha).

Freeze discipline (roadmap 阶段 1): after the baseline numbers exist, the
exam may only be EXTENDED (+ rerun of the baseline), never edited —
`eval run --expect-golden-sha` enforces this for host pinned refs.
"""

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: freeze_golden.py <run_config.toml> <golden file>...", file=sys.stderr)
        return 2
    config_path = Path(argv[1]).resolve()
    golden_paths = [Path(p).resolve() for p in argv[2:]]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    combined = hashlib.sha256()
    for path in golden_paths:
        combined.update(path.read_bytes())

    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    except subprocess.SubprocessError:
        commit, dirty = None, None

    from trillic.judge import RUBRIC_VERSION, rubric_sha256

    record = {
        "schema_version": 1,
        "golden": {
            "files": [p.relative_to(REPO).as_posix() for p in golden_paths],
            "item_count": sum(
                1
                for p in golden_paths
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
            "sha256": combined.hexdigest(),
            "byte_caliber": "argv-order concatenation of the listed files "
            "(identical to run reports' golden.sha256)",
        },
        "repo_commit": commit,
        "repo_dirty_at_freeze": dirty,
        "quality_pins": {
            "answer_model": config["quality"]["answer_model"],
            "judge_model": config["quality"]["judge_model"],
            "rubric_version": RUBRIC_VERSION,
            "rubric_sha256": rubric_sha256(),
        },
        "exclusions": {
            "meetingbank": "hard-rejected by the golden validator "
            "(docs/data-strategy.md 硬约束 1: base-checkpoint training corpus)",
            "customer_prompts": "never used for eval or training (硬约束 2)",
        },
        "license_trail": "per-entry source{dataset,subset,license,split}; "
        "subset registries under eval/manifests/",
    }
    out = REPO / "eval" / "manifests" / "golden-freeze.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
