"""trillic CLI.

Subcommand families (issue #1): `eval *` now; `golden *` arrives with
issue #3. All errors surface as `error: ...` on stderr with exit code 1.
"""

import argparse
import sys
from pathlib import Path

from trillic import __version__
from trillic.clients.gateway import GatewayError
from trillic.clients.sidecar import RefineError
from trillic.config import ConfigError, load_config
from trillic.golden import GoldenError
from trillic.report import ReportWriterError
from trillic.runner import run_eval
from trillic.tokens import TokenCounter

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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval" and args.eval_command == "run":
        return _eval_run(args)

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
