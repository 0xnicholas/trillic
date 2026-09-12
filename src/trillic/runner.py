"""`eval run` orchestration: golden + config in, immutable run dir out.

Scope (issues #2 + #6 + #7): compress every golden item through the
(injectable) refine-sidecar client at every configured sweep level, measure
compression in both token calibers (tiktoken billing + parameterized
model-native), fact recall and token-alignment F0.5 (calibers shared with
the runtime guardrail — see trillic.quality), per-item refine latency with
p50/p95 per level, and — since issue #7 — task-level quality: each load
type's downstream task is answered through the gateway for BOTH the
original and the compressed prompt, an LLM judge grades both answers
against the golden key_points, and the level block carries the quality
delta with a seeded paired-bootstrap 95% CI. metrics.json + report.md out.
"""

import hashlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from trillic import __version__
from trillic.bootstrap import DEFAULT_CONFIDENCE, METHOD
from trillic.clients.gateway import GatewayClient, HttpGatewayClient, StubGatewayClient
from trillic.clients.sidecar import (
    HttpRefineClient,
    RefineClient,
    RefineResult,
    StubRefineClient,
)
from trillic.config import RunConfig
from trillic.golden import GoldenError, GoldenItem, load_golden_set
from trillic.judge import RUBRIC_VERSION, rubric_sha256
from trillic.native_tokens import NativeCounter, build_native_counter
from trillic.quality import fact_recall, percentile, rounded_mean, text_token_f05
from trillic.report import render_report_md, write_run_dir
from trillic.resume import (
    CallJournal,
    Replay,
    ledger_id,
    load_replay_from_run_dir,
)
from trillic.task_quality import TaskQualityError, TaskQualityLoop
from trillic.tasks import task_templates_sha256
from trillic.tokens import TokenCounter, compression_ratio, kept_ratio

METRICS_SCHEMA_VERSION = 3


def build_refine_client(config: RunConfig) -> RefineClient:
    if config.sidecar_mode == "stub":
        return StubRefineClient()
    return HttpRefineClient(base_url=config.sidecar_url)


def build_gateway_client(config: RunConfig) -> GatewayClient:
    if config.gateway_mode == "stub":
        return StubGatewayClient()
    return HttpGatewayClient(base_url=config.gateway_url)


def resolve_native_vocab(config: RunConfig, config_path: Path) -> Path | None:
    """Resolve metrics.native_vocab against the config file's directory
    (relative vocab paths live next to the config that names them)."""
    if config.native_vocab is None:
        return None
    vocab = Path(config.native_vocab)
    if not vocab.is_absolute():
        vocab = Path(config_path).resolve().parent / vocab
    return vocab


def run_eval(
    config: RunConfig,
    config_path: Path,
    golden_path: Path | list[Path],
    out_root: Path,
    resume_from: Path | None = None,
    expect_golden_sha: str | None = None,
) -> Path:
    """Run the evaluation sweep and write the run directory.

    golden_path takes one file or several (issue #8: the pilot combines
    the per-type golden files; combined bytes in argv order are the run's
    golden identity). resume_from replays a prior run's gateway ledger —
    zero duplicate billing for content-identical work (issue #8).
    expect_golden_sha (issue #12) is the frozen-reference assertion: it
    must match the golden digest (same single source as the report's
    golden.sha256) and is checked BEFORE any sidecar/gateway call, so a
    drifted exam fails without spending money or leaving a run dir.
    """
    golden_paths = [golden_path] if isinstance(golden_path, Path) else list(golden_path)
    items, golden_bytes = load_golden_set(golden_paths)
    golden_sha = hashlib.sha256(golden_bytes).hexdigest()
    _assert_expected_golden_sha(expect_golden_sha, golden_sha)
    counter = TokenCounter(config.tiktoken_encoding)
    native = build_native_counter(
        config.native_flavor, resolve_native_vocab(config, config_path)
    )
    sidecar = build_refine_client(config)
    levels = config.effective_levels()

    replay = None
    if resume_from is not None:
        if not config.task_quality:
            raise TaskQualityError(
                "--resume-from requires quality.task_quality = true: the "
                "ledger being replayed IS the task-quality record"
            )
        replay = load_replay_from_run_dir(resume_from)

    journal = None
    task_loop = None
    if config.task_quality:
        # Crash journal (issue #8: kill-and-rerun must not re-bill): same
        # exam + pins + sweep -> same journal file; a rerun replays it
        # automatically, no flag needed.
        lid = ledger_id(
            golden_sha256=golden_sha,
            answer_model=config.answer_model,
            judge_model=config.judge_model,
            rubric_sha256=rubric_sha256(),
            task_templates_sha256=task_templates_sha256(),
        )
        journal = CallJournal(Path(out_root) / ".ledger" / f"{lid}.jsonl")
        if replay is None:
            pins = {
                "golden_sha256": golden_sha,
                "answer_model": config.answer_model,
                "judge_model": config.judge_model,
                "rubric_sha256": rubric_sha256(),
                "task_templates_sha256": task_templates_sha256(),
            }
            journaled = CallJournal.load_replay(journal.path, pins)
            if journaled is not None:
                replay = journaled
        task_loop = TaskQualityLoop(
            build_gateway_client(config),
            answer_model=config.answer_model,
            judge_model=config.judge_model,
            seed=config.seed,
            n_resamples=config.bootstrap_samples,
            replay=replay,
            journal=journal,
            concurrency=config.concurrency,
        )
        # Originals are level-independent: answered + judged exactly once,
        # so a sweep pays for them one time only.
        task_loop.prime_originals(items, golden_sha256=golden_sha)

    level_blocks = []
    refine_meta: list[tuple[GoldenItem, RefineResult, float]] = []
    for level in levels:
        refine_results = _refine_all(sidecar, items, config, level)
        task_quality_block = (
            task_loop.evaluate_level(
                level, items, [result.refined_text for _, result, _ in refine_results]
            )
            if task_loop is not None
            else None
        )
        block, meta = _build_level_block(
            level=level,
            refine_results=refine_results,
            counter=counter,
            native=native,
            task_quality=task_quality_block,
        )
        level_blocks.append(block)
        if not refine_meta:
            refine_meta = meta

    metrics = _build_metrics(
        config=config,
        config_path=config_path,
        golden_paths=golden_paths,
        golden_bytes=golden_bytes,
        golden_sha=golden_sha,
        refine_results=refine_meta,
        level_blocks=level_blocks,
        counter=counter,
        native_caliber=native.caliber_name,
        levels=levels,
        task_quality_meta=_task_quality_meta(
            config,
            served_models=task_loop.served_models() if task_loop is not None else None,
            gateway_calls=task_loop.call_stats() if task_loop is not None else None,
            resumed_from=resume_from,
        ),
    )
    report_md = render_report_md(metrics)
    run_dir = write_run_dir(out_root, metrics, report_md)
    if journal is not None:
        journal.close()
    return run_dir


def _assert_expected_golden_sha(expected: str | None, actual: str) -> None:
    """Frozen-reference gate (issue #12): same caliber as the report's
    golden.sha256; mismatch = the exam drifted, refuse to run."""
    if expected is None:
        return
    normalized = expected.strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized[len("sha256:"):]
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise GoldenError(
            f"--expect-golden-sha must be a 64-hex sha256 (got {expected!r}); "
            f"the run report's golden.sha256 is the source of truth"
        )
    if normalized != actual:
        raise GoldenError(
            "golden set drifted from the frozen reference — refusing to run "
            f"before any gateway call: expected sha256 {normalized}, actual {actual}; "
            "either restore the frozen golden set or re-pin the reference "
            "deliberately (no run directory was created)"
        )


def _refine_all(
    sidecar: RefineClient,
    items: list[GoldenItem],
    config: RunConfig,
    level: float,
) -> list[tuple[GoldenItem, RefineResult, float]]:
    """Refine every item at one sweep level, timing each call."""
    results: list[tuple[GoldenItem, RefineResult, float]] = []
    for item in items:
        started = time.perf_counter()
        result = sidecar.refine(
            item.prompt,
            rewrite=config.sidecar_rewrite,
            compress=config.sidecar_compress,
            aggressiveness=level,
        )
        elapsed = time.perf_counter() - started
        results.append((item, result, elapsed))
    return results


def _build_level_block(
    level: float,
    refine_results: list[tuple[GoldenItem, RefineResult, float]],
    counter: TokenCounter,
    native: NativeCounter,
    task_quality: dict | None,
) -> tuple[dict, list[tuple[GoldenItem, RefineResult, float]]]:
    """One sweep level: per-item rows + aggregate (four metric slots).

    Returns (block, refine_results) so per-item rows and the caller's
    sidecar metadata stay separate — the block is exactly what gets
    serialized into metrics.json.
    """
    item_rows = []
    for item, result, latency in refine_results:
        original_tokens = counter.count(item.prompt)
        compressed_tokens = counter.count(result.refined_text)
        native_original = native.count(item.prompt)
        native_compressed = native.count(result.refined_text)
        item_rows.append(
            {
                "id": item.id,
                "load_type": item.load_type,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "kept_ratio": kept_ratio(original_tokens, compressed_tokens),
                "compression_ratio": compression_ratio(original_tokens, compressed_tokens),
                "native_original_tokens": native_original,
                "native_compressed_tokens": native_compressed,
                "native_kept_ratio": kept_ratio(native_original, native_compressed),
                "native_compression_ratio": compression_ratio(
                    native_original, native_compressed
                ),
                "fact_recall": round(fact_recall(item.prompt, result.refined_text), 4),
                "token_f05": round(text_token_f05(item.prompt, result.refined_text), 4),
                "latency_seconds": round(latency, 6),
                "sidecar_original_tokens": result.original_tokens,
                "sidecar_compressed_tokens": result.refined_tokens,
                "compressed_text": result.refined_text,
            }
        )

    total_original = sum(row["original_tokens"] for row in item_rows)
    total_compressed = sum(row["compressed_tokens"] for row in item_rows)
    total_native_original = sum(row["native_original_tokens"] for row in item_rows)
    total_native_compressed = sum(row["native_compressed_tokens"] for row in item_rows)
    latencies = [row["latency_seconds"] for row in item_rows]

    aggregate = {
        "item_count": len(item_rows),
        "total_original_tokens": total_original,
        "total_compressed_tokens": total_compressed,
        "corpus_kept_ratio": kept_ratio(total_original, total_compressed),
        "corpus_compression_ratio": compression_ratio(total_original, total_compressed),
        "mean_kept_ratio": rounded_mean([r["kept_ratio"] for r in item_rows]),
        "mean_compression_ratio": rounded_mean([r["compression_ratio"] for r in item_rows]),
        "total_native_original_tokens": total_native_original,
        "total_native_compressed_tokens": total_native_compressed,
        "corpus_native_kept_ratio": kept_ratio(total_native_original, total_native_compressed),
        "corpus_native_compression_ratio": compression_ratio(
            total_native_original, total_native_compressed
        ),
        "mean_fact_recall": rounded_mean([r["fact_recall"] for r in item_rows]),
        "mean_token_f05": rounded_mean([r["token_f05"] for r in item_rows]),
        "latency": {
            "mean_seconds": rounded_mean(latencies),
            "p50_seconds": _nullable_round(percentile(latencies, 50)),
            "p95_seconds": _nullable_round(percentile(latencies, 95)),
        },
        # Task-level quality (LLM judge on key_points, issue #7): the
        # numbers land here when quality.task_quality = true; None marks
        # the loop disabled in config (structural slot — never missing).
        "task_quality": task_quality,
    }
    return {
        "aggressiveness": level,
        "items": item_rows,
        "aggregate": aggregate,
    }, refine_results


def _git_identity() -> tuple[str | None, bool | None]:
    """(commit, dirty) of the repo the harness runs from, best-effort.

    Freeze discipline (issue #9): a baseline report must be auditable to
    the exact code state that produced it — the exam's content hash
    (golden.sha256) plus the harness revision. None when not run from a
    git checkout (nothing to pin, honestly reported as such).
    """
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return commit, bool(status.strip())
    except (subprocess.SubprocessError, OSError):
        return None, None


def _build_metrics(
    config: RunConfig,
    config_path: Path,
    golden_paths: list[Path],
    golden_bytes: bytes,
    golden_sha: str,
    refine_results: list[tuple[GoldenItem, RefineResult, float]],
    level_blocks: list[dict],
    counter: TokenCounter,
    native_caliber: str,
    levels: list[float],
    task_quality_meta: dict,
) -> dict:
    config_raw = Path(config_path).read_text(encoding="utf-8")
    golden_raw_bytes = golden_bytes
    git_commit, git_dirty = _git_identity()

    refine_models = sorted(
        {
            result.refine_model
            for _, result, _ in refine_results
        }
    )
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run_id": _run_id(config, config_raw, golden_raw_bytes, created_at),
        "created_at": created_at,
        "harness": {
            "name": "trillic",
            "version": __version__,
            "python": sys.version.split()[0],
            "tiktoken": _tiktoken_version(),
            # Freeze pins (issue #9): code revision behind the numbers.
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
        "config": {
            "source_path": str(config_path),
            "sha256": hashlib.sha256(config_raw.encode("utf-8")).hexdigest(),
            "raw": config_raw,
            "parsed": config.snapshot(),
        },
        "golden": {
            "source_path": ",".join(str(path) for path in golden_paths),
            "sha256": golden_sha,
            "item_count": len(level_blocks[0]["items"]) if level_blocks else 0,
        },
        "sidecar": {
            "mode": config.sidecar_mode,
            "refine_model": refine_models[0] if len(refine_models) == 1 else refine_models,
            "rewrite": config.sidecar_rewrite,
            "compress": config.sidecar_compress,
            # Single level: the exact aggressiveness. Sweep: the levels live
            # in metrics.levels — no misleading "primary" here.
            "aggressiveness": levels[0] if len(levels) == 1 else None,
        },
        "task_quality": task_quality_meta,
        "metrics": {
            "tiktoken_encoding": counter.encoding_name,
            "native_caliber": native_caliber,
            "levels": level_blocks,
        },
    }
    return metrics


def _task_quality_meta(
    config: RunConfig,
    served_models: dict[str, list[str]] | None = None,
    gateway_calls: dict[str, int] | None = None,
    resumed_from: Path | None = None,
) -> dict:
    """Run-level task-quality provenance (issue #7): the pinned judge +
    answer models and the versioned rubric identity, so two runs can never
    be silently compared across grading behavior. `served_models` records
    what the gateway actually served per role — the version of record
    when a gateway resolves pinned ids to concrete snapshots."""
    if not config.task_quality:
        return {"enabled": False}
    meta = {
        "enabled": True,
        "answer_model": config.answer_model,
        "judge_model": config.judge_model,
        "judge_rubric_version": RUBRIC_VERSION,
        "judge_rubric_sha256": rubric_sha256(),
        "task_templates_sha256": task_templates_sha256(),
        "bootstrap": {
            "method": METHOD,
            "n_resamples": config.bootstrap_samples,
            "seed": config.seed,
            "confidence": DEFAULT_CONFIDENCE,
        },
    }
    if served_models is not None:
        meta["served_answer_models"] = served_models["answer"]
        meta["served_judge_models"] = served_models["judge"]
    if gateway_calls is not None:
        meta["gateway_calls"] = gateway_calls
    if resumed_from is not None:
        meta["resumed_from"] = {
            "run_id": _prior_run_id(resumed_from),
            "path": str(resumed_from),
        }
    return meta


def _prior_run_id(run_dir: Path) -> str:
    """Best-effort prior run id for the resume trail (dir name is the id)."""
    return Path(run_dir).name


def _run_id(config: RunConfig, config_raw: str, golden_bytes: bytes, created_at: str) -> str:
    """Timestamp-first for sortability, content-hash suffix for identity:
    identical inputs (config + golden + harness versions) map to the same
    hash, so a rerun either lands in a fresh timestamped dir or is caught by
    the immutability guard."""
    digest = hashlib.sha256()
    digest.update(__version__.encode())
    digest.update(_tiktoken_version().encode())
    digest.update(config_raw.encode("utf-8"))
    digest.update(golden_bytes)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{config.name}-{digest.hexdigest()[:8]}"



def _nullable_round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _tiktoken_version() -> str:
    import tiktoken

    return tiktoken.__version__
