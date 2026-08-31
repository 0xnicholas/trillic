"""trillic CLI.

Subcommand families (issue #1): `eval *` now; `golden *` covers schema
validation, split manifests, and the three pilot generators (issues
#3/#4/#5). All errors surface as `error: ...` on stderr with exit code 1.
"""

import argparse
import json
import sys
from pathlib import Path

from trillic import __version__
from trillic.clients.gateway import GatewayError
from trillic.clients.sidecar import RefineError
from trillic.config import ConfigError, load_config
from trillic.dialogue import (
    DialogueError,
    build_dialogue_entries,
    build_dialogue_manifest,
    load_families as load_dialogue_families,
    make_review as make_dialogue_review,
)
from trillic.golden import GoldenError, collect_golden_errors
from trillic.judge import JudgeError
from trillic.longbench import LongBenchError, build_manifest, build_rag_entries
from trillic.report import ReportWriterError
from trillic.resume import ResumeError
from trillic.runner import run_eval
from trillic.sizing import sizing_report
from trillic.sysprompt import (
    SysPromptError,
    build_sysprompt_entries,
    build_sysprompt_manifest,
    load_families,
    make_review,
)
from trillic.task_quality import TaskQualityError

_ERROR_EXIT_CODE = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trillic",
        description="Trillic evaluation harness (prompt-compression yardstick).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="evaluation runs")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)

    run_parser = eval_sub.add_parser("run", help="run an evaluation over a golden set")
    run_parser.add_argument(
        "--config", required=True, type=Path, help="run configuration (TOML)"
    )
    run_parser.add_argument(
        "--golden", required=True, type=Path, nargs="+",
        help="golden set (jsonl); multiple files combine into one exam (argv order)",
    )
    run_parser.add_argument(
        "--out", type=Path, default=Path("runs"), help="runs root (default: ./runs)"
    )
    run_parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="prior run dir whose gateway ledger to replay (content-addressed; "
        "identical work is never re-billed)",
    )

    sizing_parser = eval_sub.add_parser(
        "sizing", help="judge variance + required-n report from a completed run"
    )
    sizing_parser.add_argument(
        "--run", required=True, type=Path, help="run directory (with metrics.json)"
    )
    sizing_parser.add_argument(
        "--level", type=float, default=None,
        help="restrict sizing to one sweep level (recommended on multi-level "
        "runs: pooling levels mixes heterogeneous effects)",
    )
    sizing_parser.add_argument(
        "--out", type=Path, default=None, help="optional JSON output path"
    )

    golden_parser = subparsers.add_parser(
        "golden", help="golden set tooling (schema, splits, provenance)"
    )
    golden_sub = golden_parser.add_subparsers(dest="golden_command", required=True)

    validate_parser = golden_sub.add_parser(
        "validate", help="validate golden jsonl files against the schema"
    )
    validate_parser.add_argument("files", nargs="+", type=Path, help="golden jsonl files")

    manifest_parser = golden_sub.add_parser(
        "manifest", help="generate the LongBench split/license manifest"
    )
    manifest_parser.add_argument(
        "--subsets", required=True, type=Path,
        help="curated subset registry (TOML: license + train_use per subset)",
    )
    manifest_parser.add_argument(
        "--data-dir", required=True, type=Path, help="downloaded LongBench data directory"
    )
    manifest_parser.add_argument(
        "--out", required=True, type=Path, help="output manifest JSON path"
    )

    build_parser = golden_sub.add_parser(
        "build-rag", help="seeded drafting of RAG golden entries (eval half only)"
    )
    build_parser.add_argument("--manifest", required=True, type=Path)
    build_parser.add_argument("--data-dir", required=True, type=Path)
    build_parser.add_argument(
        "--counts", required=True,
        help="entries per subset, e.g. qasper=4,hotpotqa=4,gov_report=2",
    )
    build_parser.add_argument("--seed", type=int, default=0)
    build_parser.add_argument("--min-tokens", type=int, default=500)
    build_parser.add_argument("--max-tokens", type=int, default=4000)
    build_parser.add_argument("--out", required=True, type=Path)

    sysprompt_manifest_parser = golden_sub.add_parser(
        "manifest-sysprompt",
        help="generate the system-prompt family seed-split manifest",
    )
    sysprompt_manifest_parser.add_argument(
        "--families", required=True, type=Path,
        help="scenario-family registry (TOML: seed pools + licenses per family)",
    )
    sysprompt_manifest_parser.add_argument(
        "--golden", type=Path, default=None,
        help="frozen golden jsonl to verify against regeneration and record",
    )
    sysprompt_manifest_parser.add_argument(
        "--review-status", choices=("pending", "approved"), default=None,
        help="owner sign-off record for the pilot (requires --golden)",
    )
    sysprompt_manifest_parser.add_argument(
        "--reviewer", default=None,
        help="who performed the review (requires --review-status)",
    )
    sysprompt_manifest_parser.add_argument(
        "--review-notes", default=None,
        help="free-text review notes (requires --review-status)",
    )
    sysprompt_manifest_parser.add_argument(
        "--out", required=True, type=Path, help="output manifest JSON path"
    )

    sysprompt_build_parser = golden_sub.add_parser(
        "build-sysprompt",
        help="seeded synthesis of system_prompt golden entries (eval seeds only)",
    )
    sysprompt_build_parser.add_argument(
        "--families", required=True, type=Path,
        help="scenario-family registry (TOML)",
    )
    sysprompt_build_parser.add_argument(
        "--seeds", required=True,
        help="family=seed pairs, e.g. support_logistics=101,finance_analyst=101",
    )
    sysprompt_build_parser.add_argument("--out", required=True, type=Path)

    dialogue_manifest_parser = golden_sub.add_parser(
        "manifest-dialogue",
        help="generate the dialogue family seed-split manifest",
    )
    dialogue_manifest_parser.add_argument(
        "--families", required=True, type=Path,
        help="scenario-family registry (TOML: seed pools + licenses per family)",
    )
    dialogue_manifest_parser.add_argument(
        "--golden", type=Path, default=None,
        help="frozen golden jsonl to verify against regeneration and record",
    )
    dialogue_manifest_parser.add_argument(
        "--review-status", choices=("pending", "approved"), default=None,
        help="owner sign-off record for the pilot (requires --golden)",
    )
    dialogue_manifest_parser.add_argument(
        "--reviewer", default=None,
        help="who performed the review (requires --review-status)",
    )
    dialogue_manifest_parser.add_argument(
        "--review-notes", default=None,
        help="free-text review notes (requires --review-status)",
    )
    dialogue_manifest_parser.add_argument(
        "--out", required=True, type=Path, help="output manifest JSON path",
    )

    dialogue_build_parser = golden_sub.add_parser(
        "build-dialogue",
        help="seeded synthesis of dialogue golden entries (eval seeds only)",
    )
    dialogue_build_parser.add_argument(
        "--families", required=True, type=Path,
        help="scenario-family registry (TOML)",
    )
    dialogue_build_parser.add_argument(
        "--seeds", required=True,
        help="family=seed pairs, e.g. support_ticket=101,vendor_procurement=101",
    )
    dialogue_build_parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval" and args.eval_command == "run":
        return _eval_run(args)
    if args.command == "eval" and args.eval_command == "sizing":
        return _eval_sizing(args)
    if args.command == "golden":
        try:
            if args.golden_command == "validate":
                return _golden_validate(args)
            if args.golden_command == "manifest":
                return _golden_manifest(args)
            if args.golden_command == "build-rag":
                return _golden_build_rag(args)
            if args.golden_command == "manifest-sysprompt":
                return _golden_manifest_sysprompt(args)
            if args.golden_command == "build-sysprompt":
                return _golden_build_sysprompt(args)
            if args.golden_command == "manifest-dialogue":
                return _golden_manifest_dialogue(args)
            if args.golden_command == "build-dialogue":
                return _golden_build_dialogue(args)
        except (
            GoldenError,
            LongBenchError,
            SysPromptError,
            DialogueError,
            ValueError,
            OSError,
        ) as e:
            print(f"error: {e}", file=sys.stderr)
            return _ERROR_EXIT_CODE

    parser.error(f"unknown command: {args.command}")
    return _ERROR_EXIT_CODE  # unreachable


def _eval_run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        run_dir = run_eval(
            config=config,
            config_path=args.config,
            golden_path=list(args.golden),
            out_root=args.out,
            resume_from=args.resume_from,
        )
    except (
        ConfigError,
        GoldenError,
        RefineError,
        GatewayError,
        JudgeError,
        TaskQualityError,
        ReportWriterError,
        ValueError,
        OSError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return _ERROR_EXIT_CODE
    print(str(run_dir))
    return 0


def _eval_sizing(args: argparse.Namespace) -> int:
    """Judge variance + effect size + required-n from a run's task-quality rows."""
    metrics_path = Path(args.run) / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"error: no metrics.json under {args.run}", file=sys.stderr)
        return _ERROR_EXIT_CODE
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: cannot read {metrics_path}: {e}", file=sys.stderr)
        return _ERROR_EXIT_CODE
    task_quality = metrics.get("task_quality") or {}
    if not task_quality.get("enabled"):
        print(
            "error: the run has no task_quality data (quality.task_quality "
            "was disabled) — sizing needs judged deltas",
            file=sys.stderr,
        )
        return _ERROR_EXIT_CODE
    levels = metrics.get("metrics", {}).get("levels", [])
    if args.level is not None:
        levels = [lv for lv in levels if lv.get("aggressiveness") == args.level]
        if not levels:
            print(
                f"error: the run has no level {args.level} — "
                f"levels present: {[lv.get('aggressiveness') for lv in metrics.get('metrics', {}).get('levels', [])]}",
                file=sys.stderr,
            )
            return _ERROR_EXIT_CODE
    rows = [
        row
        for level in levels
        for row in level.get("aggregate", {}).get("task_quality", {}).get("items", [])
    ]
    if not rows:
        print("error: the run has no task-quality rows to size on", file=sys.stderr)
        return _ERROR_EXIT_CODE
    report = sizing_report(rows)
    if args.out is not None:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    _print_sizing(report)
    return 0


def _print_sizing(report: dict) -> None:
    overall = report["overall"]
    if overall["cohens_d"] is not None:
        print(
            f"overall: n={overall['n_observed']}, "
            f"mean delta {overall['mean_delta']:+.4f}, "
            f"std {overall['std_delta']:.4f}, "
            f"Cohen's d {overall['cohens_d']:+.4f}"
        )
    else:
        print(
            f"overall: n={overall['n_observed']}, "
            f"mean delta {overall['mean_delta']:+.4f}, "
            f"std {overall['std_delta']:.4f}, Cohen's d undefined"
        )
    required = (
        f"required n per class ≈ {overall['n_required']} "
        f"(raw {overall['n_required_raw']:.2f})"
        if overall["n_required_raw"] is not None
        else "required n: degenerate (zero variance or zero effect)"
    )
    print(required)
    for name, stats in report["by_load_type"].items():
        need = (
            f"required n ≈ {stats['n_required']}"
            if stats["n_required_raw"] is not None
            else "degenerate"
        )
        print(
            f"  {name}: n={stats['n_observed']}, "
            f"mean delta {stats['mean_delta']:+.4f}, {need}"
        )


def _golden_validate(args: argparse.Namespace) -> int:
    failed = False
    for path in args.files:
        errors = collect_golden_errors(path)
        if errors:
            for error in errors:
                print(f"error: {error}", file=sys.stderr)
            failed = True
        else:
            count = sum(
                1
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            print(f"{path}: ok ({count} entries)")
    return _ERROR_EXIT_CODE if failed else 0


def _golden_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest(args.subsets, args.data_dir)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(manifest['subsets'])} subsets")
    return 0


def _golden_build_rag(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    counts = {}
    for part in str(args.counts).split(","):
        name, _, value = part.strip().partition("=")
        if not name or not value:
            raise ValueError(f"bad --counts item {part!r} (expected subset=N)")
        counts[name] = int(value)
    entries = build_rag_entries(
        manifest,
        data_dir=args.data_dir,
        counts=counts,
        seed=args.seed,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    args.out.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8"
    )
    print(f"{args.out}: {len(entries)} entries (draft — hand-review before freezing)")
    return 0


def _golden_manifest_sysprompt(args: argparse.Namespace) -> int:
    families = load_families(args.families)
    review = None
    if args.review_status is not None:
        if args.golden is None:
            raise ValueError("--review-status requires --golden (no pilot to review)")
        review = make_review(
            args.review_status, reviewer=args.reviewer, notes=args.review_notes or ""
        )
    manifest = build_sysprompt_manifest(
        families, golden_path=args.golden, review=review
    )
    args.out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pilot = f", pilot: {manifest['pilot']['entries']} entries verified" if "pilot" in manifest else ""
    print(f"{args.out}: {len(manifest['families'])} families{pilot}")
    return 0


def _golden_build_sysprompt(args: argparse.Namespace) -> int:
    families = load_families(args.families)
    seed_plan: dict[str, list[int]] = {}
    for part in str(args.seeds).split(","):
        name, _, value = part.strip().partition("=")
        if not name or not value:
            raise ValueError(f"bad --seeds item {part!r} (expected family=seed)")
        seed_plan.setdefault(name, []).append(int(value))
    entries = build_sysprompt_entries(families, seed_plan)
    args.out.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
        encoding="utf-8",
    )
    print(
        f"{args.out}: {len(entries)} entries (eval seeds only — "
        "hand-review, then freeze via manifest-sysprompt --golden)"
    )
    return 0


def _golden_manifest_dialogue(args: argparse.Namespace) -> int:
    families = load_dialogue_families(args.families)
    review = None
    if args.review_status is not None:
        if args.golden is None:
            raise ValueError("--review-status requires --golden (no pilot to review)")
        review = make_dialogue_review(
            args.review_status, reviewer=args.reviewer, notes=args.review_notes or ""
        )
    manifest = build_dialogue_manifest(
        families, golden_path=args.golden, review=review
    )
    args.out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pilot = f", pilot: {manifest['pilot']['entries']} entries verified" if "pilot" in manifest else ""
    print(f"{args.out}: {len(manifest['families'])} families{pilot}")
    return 0


def _golden_build_dialogue(args: argparse.Namespace) -> int:
    families = load_dialogue_families(args.families)
    seed_plan: dict[str, list[int]] = {}
    for part in str(args.seeds).split(","):
        name, _, value = part.strip().partition("=")
        if not name or not value:
            raise ValueError(f"bad --seeds item {part!r} (expected family=seed)")
        seed_plan.setdefault(name, []).append(int(value))
    entries = build_dialogue_entries(families, seed_plan)
    args.out.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
        encoding="utf-8",
    )
    print(
        f"{args.out}: {len(entries)} entries (eval seeds only — "
        "hand-review, then freeze via manifest-dialogue --golden)"
    )
    return 0
