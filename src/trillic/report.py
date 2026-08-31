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
    """Human-readable summary of a run (markdown, one section per sweep
    level, all four metric slots: dual-caliber compression, fact recall,
    token F0.5, latency — plus the task-quality placeholder)."""
    m = metrics["metrics"]
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
        f"- token calibers: tiktoken {m['tiktoken_encoding']} (billing)"
        f" + native `{m['native_caliber']}`"
        " (compression_ratio = fraction removed; kept_ratio = retention)",
        f"- sweep levels: {', '.join(str(lv['aggressiveness']) for lv in m['levels'])}",
        "",
    ]
    for level in m["levels"]:
        lines += _render_level_section(level, sidecar)
    lines += [
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


def _render_level_section(level: dict, sidecar: dict) -> list[str]:
    agg = level["aggregate"]
    latency = agg["latency"]
    lines = [
        f"## Level {level['aggressiveness']} — refine_model {sidecar['refine_model']}",
        "",
        "| id | tiktoken o→c | kept | removed | native o→c | native kept"
        " | fact recall | F0.5 | latency (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in level["items"]:
        lines.append(
            f"| {item['id']}"
            f" | {item['original_tokens']}→{item['compressed_tokens']}"
            f" | {item['kept_ratio']:.4f} | {item['compression_ratio']:.4f}"
            f" | {item['native_original_tokens']}→{item['native_compressed_tokens']}"
            f" | {item['native_kept_ratio']:.4f}"
            f" | {item['fact_recall']:.4f} | {item['token_f05']:.4f}"
            f" | {item['latency_seconds'] * 1000:.2f} |"
        )
    lines += [
        "",
        f"- items: {agg['item_count']}",
        f"- corpus (tiktoken): {agg['total_original_tokens']} →"
        f" {agg['total_compressed_tokens']}"
        f" (kept {agg['corpus_kept_ratio']:.4f},"
        f" removed {agg['corpus_compression_ratio']:.4f})",
        f"- corpus (native): {agg['total_native_original_tokens']} →"
        f" {agg['total_native_compressed_tokens']}"
        f" (kept {agg['corpus_native_kept_ratio']:.4f},"
        f" removed {agg['corpus_native_compression_ratio']:.4f})",
        f"- mean fact recall: {agg['mean_fact_recall']:.4f},"
        f" mean token F0.5: {agg['mean_token_f05']:.4f}",
        f"- latency: p50 {latency['p50_seconds']:.4f}s /"
        f" p95 {latency['p95_seconds']:.4f}s"
        f" (mean {latency['mean_seconds']:.4f}s, n={agg['item_count']})",
        "- task-level quality: pending (issue #7 — LLM judge on key_points)",
        "",
    ]
    return lines
