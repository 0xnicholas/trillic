"""`eval run` orchestration: golden + config in, immutable run dir out.

Walking-skeleton scope (issue #2): compress every golden item through the
(injectable) refine-sidecar client, measure the tiktoken-caliber compression
ratio, and write metrics.json + report.md. Downstream tasks, judges, and
bootstrap CIs arrive with issues #6/#7 on the same client seams.
"""

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from trillic import __version__
from trillic.clients.gateway import GatewayClient, HttpGatewayClient, StubGatewayClient
from trillic.clients.sidecar import (
    HttpRefineClient,
    RefineClient,
    RefineResult,
    StubRefineClient,
)
from trillic.config import RunConfig
from trillic.golden import GoldenItem, load_golden
from trillic.report import render_report_md, write_run_dir
from trillic.tokens import TokenCounter, compression_ratio, kept_ratio

METRICS_SCHEMA_VERSION = 1


def build_refine_client(config: RunConfig) -> RefineClient:
    if config.sidecar_mode == "stub":
        return StubRefineClient()
    return HttpRefineClient(base_url=config.sidecar_url)


def build_gateway_client(config: RunConfig) -> GatewayClient:
    if config.gateway_mode == "stub":
        return StubGatewayClient()
    return HttpGatewayClient(base_url=config.gateway_url)


def run_eval(config: RunConfig, config_path: Path, golden_path: Path, out_root: Path) -> Path:
    """Run the walking-skeleton evaluation and write the run directory."""
    items = load_golden(golden_path)
    counter = TokenCounter(config.tiktoken_encoding)
    sidecar = build_refine_client(config)

    refine_results: list[tuple[GoldenItem, RefineResult]] = []
    for item in items:
        result = sidecar.refine(
            item.prompt,
            rewrite=config.sidecar_rewrite,
            compress=config.sidecar_compress,
            aggressiveness=config.aggressiveness,
        )
        refine_results.append((item, result))

    metrics = _build_metrics(
        config=config,
        config_path=config_path,
        golden_path=golden_path,
        refine_results=refine_results,
        counter=counter,
    )
    report_md = render_report_md(metrics)
    return write_run_dir(out_root, metrics, report_md)


def _build_metrics(
    config: RunConfig,
    config_path: Path,
    golden_path: Path,
    refine_results: list[tuple[GoldenItem, RefineResult]],
    counter: TokenCounter,
) -> dict:
    config_raw = Path(config_path).read_text(encoding="utf-8")
    golden_raw_bytes = Path(golden_path).read_bytes()

    item_rows = []
    for item, result in refine_results:
        original_tokens = counter.count(item.prompt)
        compressed_tokens = counter.count(result.refined_text)
        item_rows.append(
            {
                "id": item.id,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "kept_ratio": kept_ratio(original_tokens, compressed_tokens),
                "compression_ratio": compression_ratio(original_tokens, compressed_tokens),
                "sidecar_original_tokens": result.original_tokens,
                "sidecar_compressed_tokens": result.refined_tokens,
                "compressed_text": result.refined_text,
            }
        )

    total_original = sum(row["original_tokens"] for row in item_rows)
    total_compressed = sum(row["compressed_tokens"] for row in item_rows)
    refine_models = sorted({result.refine_model for _, result in refine_results})
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
        },
        "config": {
            "source_path": str(config_path),
            "sha256": hashlib.sha256(config_raw.encode("utf-8")).hexdigest(),
            "raw": config_raw,
            "parsed": config.snapshot(),
        },
        "golden": {
            "source_path": str(golden_path),
            "sha256": hashlib.sha256(golden_raw_bytes).hexdigest(),
            "item_count": len(refine_results),
        },
        "sidecar": {
            "mode": config.sidecar_mode,
            "refine_model": refine_models[0] if len(refine_models) == 1 else refine_models,
            "rewrite": config.sidecar_rewrite,
            "compress": config.sidecar_compress,
            "aggressiveness": config.aggressiveness,
        },
        "metrics": {
            "tiktoken_encoding": counter.encoding_name,
            "items": item_rows,
            "aggregate": {
                "item_count": len(item_rows),
                "total_original_tokens": total_original,
                "total_compressed_tokens": total_compressed,
                "corpus_kept_ratio": kept_ratio(total_original, total_compressed),
                "corpus_compression_ratio": compression_ratio(total_original, total_compressed),
                "mean_kept_ratio": _mean([r["kept_ratio"] for r in item_rows]),
                "mean_compression_ratio": _mean([r["compression_ratio"] for r in item_rows]),
            },
        },
    }
    return metrics


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


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _tiktoken_version() -> str:
    import tiktoken

    return tiktoken.__version__
