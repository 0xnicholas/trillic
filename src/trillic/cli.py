"""trillic CLI.

Subcommand families (issue #1): `eval *` now; `golden *` arrives with
issue #3. All errors surface as `error: ...` on stderr with exit code 1.
"""

import argparse
import json
import sys
from pathlib import Path

from trillic import __version__
from trillic.clients.gateway import GatewayError
from trillic.clients.sidecar import RefineError
from trillic.config import ConfigError, load_config
from trillic.golden import GoldenError, collect_golden_errors
from trillic.longbench import LongBenchError, build_manifest, build_rag_entries
from trillic.report import ReportWriterError
from trillic.runner import run_eval
from trillic.sysprompt import (
    SysPromptError,
    build_sysprompt_entries,
    build_sysprompt_manifest,
    load_families,
    make_review,
)

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
        "--golden", required=True, type=Path, help="golden set (jsonl)"
    )
    run_parser.add_argument(
        "--out", type=Path, default=Path("runs"), help="runs root (default: ./runs)"
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval" and args.eval_command == "run":
        return _eval_run(args)
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
        except (
            GoldenError,
            LongBenchError,
            SysPromptError,
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
            golden_path=args.golden,
            out_root=args.out,
        )
    except (
        ConfigError,
        GoldenError,
        RefineError,
        GatewayError,
        ReportWriterError,
        ValueError,
        OSError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return _ERROR_EXIT_CODE
    print(str(run_dir))
    return 0


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
