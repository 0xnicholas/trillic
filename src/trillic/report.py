"""Immutable run-directory reporting: metrics.json + report.md.

Every run writes its own directory under the runs root; existing run
directories are never overwritten (issue #1: "旧报告永不改写").
"""

import json
from pathlib import Path


class ReportWriterError(Exception):
    """A run directory could not be written (e.g. it already exists)."""


def write_run_dir(out_root: Path, metrics: dict, report_md: str) -> Path:
    """Write metrics.json + report.md into a fresh <out_root>/<run_id> dir."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / metrics["run_id"]
    if run_dir.exists():
        raise ReportWriterError(
            f"run directory {run_dir} already exists — runs are immutable and "
            "are never overwritten"
        )
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "report.md").write_text(report_md, encoding="utf-8")
    return run_dir


def render_report_md(metrics: dict) -> str:
    """Human-readable summary of a run (markdown)."""
    m = metrics["metrics"]
    agg = m["aggregate"]
    sidecar = metrics["sidecar"]
    lines = [
        f"# Eval run {metrics['run_id']}",
        "",
        f"- created: {metrics['created_at']}",
        f"- harness: {metrics['harness']['name']} v{metrics['harness']['version']}"
        f" (python {metrics['harness'].get('python', '?')},"
        f" tiktoken {metrics['harness'].get('tiktoken', '?')})",
        f"- golden: {metrics['golden']['item_count']} items,"
        f" sha256 `{metrics['golden']['sha256'][:12]}…`",
        f"- sidecar: mode={sidecar['mode']}, refine_model={sidecar['refine_model']},"
        f" rewrite={sidecar.get('rewrite')}, compress={sidecar.get('compress')},"
        f" aggressiveness={sidecar.get('aggressiveness')}",
        f"- token caliber: tiktoken {m['tiktoken_encoding']}"
        " (compression_ratio = fraction removed; kept_ratio = retention)",
        "",
        "## Per-item results",
        "",
        "| id | original | compressed | kept_ratio | compression_ratio |",
        "|---|---|---|---|---|",
    ]
    for item in m["items"]:
        lines.append(
            f"| {item['id']} | {item['original_tokens']} | {item['compressed_tokens']}"
            f" | {item['kept_ratio']:.4f} | {item['compression_ratio']:.4f} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- items: {agg['item_count']}",
        f"- tokens: {agg['total_original_tokens']} -> {agg['total_compressed_tokens']}"
        f" (corpus kept {agg['corpus_kept_ratio']:.4f},"
        f" compression {agg['corpus_compression_ratio']:.4f})",
        f"- mean kept_ratio: {agg['mean_kept_ratio']:.4f},"
        f" mean compression_ratio: {agg['mean_compression_ratio']:.4f}",
        "",
        "## Config snapshot",
        "",
        "```toml",
        metrics["config"]["raw"].rstrip("\n"),
        "```",
        "",
        f"config sha256: `{metrics['config']['sha256']}`",
        f"golden sha256: `{metrics['golden']['sha256']}`",
        "",
    ]
    return "\n".join(lines)
