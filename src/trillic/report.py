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
    token F0.5, latency — plus task-level quality: judged scores, delta,
    and the bootstrap CI)."""
    m = metrics["metrics"]
    sidecar = metrics["sidecar"]
    task_quality = metrics.get("task_quality", {"enabled": False})
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
    ]
    lines += _task_quality_header_lines(task_quality)
    lines.append("")
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


def _task_quality_header_lines(task_quality: dict) -> list[str]:
    """Provenance lines for the judge loop (issue #7): pinned models and
    the versioned rubric hash go in every report that graded answers."""
    if not task_quality.get("enabled"):
        return ["- task quality: disabled (quality.task_quality = false)"]
    bootstrap = task_quality["bootstrap"]
    return [
        f"- task quality: judge {task_quality['judge_model']}"
        f" (rubric v{task_quality['judge_rubric_version']}"
        f" sha256 `{task_quality['judge_rubric_sha256'][:12]}…`),"
        f" answer model {task_quality['answer_model']}",
        f"- significance: {bootstrap['method']}, {bootstrap['n_resamples']}"
        f" resamples, seed {bootstrap['seed']},"
        f" {bootstrap['confidence']:.0%} CI",
    ]


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
        f"- latency: {_latency_line(latency, agg['item_count'])}",
    ]
    lines += _render_task_quality(agg["task_quality"])
    lines.append("")
    return lines


def _render_task_quality(task_quality: dict | None) -> list[str]:
    """Task-level quality block: per-item judged scores + delta table,
    means, and the bootstrap CI the acceptance criterion reads."""
    if task_quality is None:
        return ["- task-level quality: disabled (quality.task_quality = false)"]
    lines = [
        "",
        "### Task quality (LLM judge on key_points)",
        "",
        "| id | load | original | compressed | delta |",
        "|---|---|---|---|---|",
    ]
    for item in task_quality["items"]:
        lines.append(
            f"| {item['id']} | {item['load_type']}"
            f" | {item['original_score']:.4f}"
            f" | {item['compressed_score']:.4f}"
            f" | {item['delta']:+.4f} |"
        )
    ci = task_quality["delta_ci95"]
    verdict = "yes" if task_quality["ci_lower_bound_not_negative"] else "no"
    lines += [
        "",
        f"- mean original {task_quality['mean_original_score']:.4f} →"
        f" compressed {task_quality['mean_compressed_score']:.4f}"
        f" (mean delta {task_quality['mean_delta']:+.4f})",
        f"- delta 95% CI"
        f" [{ci['low']:+.4f}, {ci['high']:+.4f}]"
        f" (paired bootstrap, resample unit = prompt)",
        f"- CI lower bound not negative: {verdict}",
    ]
    return lines


def _latency_line(latency: dict, item_count: int) -> str:
    """Latency summary; percentiles are None for an empty item list."""
    if latency["p50_seconds"] is None or latency["p95_seconds"] is None:
        return f"no samples (n={item_count})"
    return (
        f"p50 {latency['p50_seconds']:.4f}s /"
        f" p95 {latency['p95_seconds']:.4f}s"
        f" (mean {latency['mean_seconds']:.4f}s, n={item_count})"
    )
