"""Dual-checkpoint baseline summary (issue #9).

Reads two completed run directories (mBERT-base + XLM-R-large sweeps over
the SAME golden set), computes the checkpoint-vs-checkpoint comparison
the 后手牌 activation criteria need, and emits a markdown summary:

- per level: dual-caliber compression, fact recall, token F0.5,
  task-quality delta (mean + bootstrap CI), refine latency p50/p95;
- paired-by-item quality difference (XLM delta − mBERT delta) with a
  seeded bootstrap 95% CI — "XLM-R-large 显著更强" needs the CI lower
  bound above zero;
- latency comparison — "延迟可接受" evidence (p95 ratio + absolute ms);
- interface compatibility note (decisions.md §6: .bert vs .roberta).

Pure stdlib, reads metrics.json only; no network, no models.
"""

import hashlib
import json
import random
import sys
from pathlib import Path

CONFIDENCE = 0.95
N_RESAMPLES = 10_000
SEED = 20260912


def load_run(run_dir: Path) -> dict:
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
    if not metrics.get("task_quality", {}).get("enabled"):
        raise SystemExit(f"{run_dir}: run has no task-quality data")
    return metrics


def rows_by_key(metrics: dict) -> dict[tuple[float, str], dict]:
    rows = {}
    for level in metrics["metrics"]["levels"]:
        for row in level.get("aggregate", {}).get("task_quality", {}).get("items", []):
            rows[(level["aggressiveness"], row["id"])] = row
    return rows


def bootstrap_ci(values: list[float]) -> tuple[float, float, float]:
    """(mean, lo, hi) via seeded percentile bootstrap."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(SEED)
    n = len(values)
    means = []
    for _ in range(N_RESAMPLES):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((1 - CONFIDENCE) / 2 * N_RESAMPLES)]
    hi = means[int((1 + CONFIDENCE) / 2 * N_RESAMPLES) - 1]
    return sum(values) / n, lo, hi


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    mbert = load_run(Path(argv[1]))
    xlmr = load_run(Path(argv[2]))
    out_path = Path(argv[2]).parent / "dual-checkpoint-summary.md"

    golden_a, golden_b = mbert["golden"]["sha256"], xlmr["golden"]["sha256"]
    if golden_a != golden_b:
        raise SystemExit(
            f"golden mismatch: {argv[1]} {golden_a[:12]} vs {argv[2]} {golden_b[:12]} — "
            "the two baselines must run the SAME exam"
        )

    lines = [
        "# 双 checkpoint 全量基线对比(issue #9)",
        "",
        f"- mBERT-base run: `{mbert['run_id']}`",
        f"- XLM-R-large run: `{xlmr['run_id']}`",
        f"- golden: {mbert['golden']['item_count']} items, sha256 `{golden_a[:12]}…`",
        f"- harness: trillic v{mbert['harness']['version']}, "
        f"commit `{str(mbert['harness'].get('git_commit'))[:12]}…`"
        + ("(dirty tree)" if mbert["harness"].get("git_dirty") else "(clean tree)"),
        f"- judge/answer: {xlmr['task_quality']['answer_model']} answers, "
        f"{xlmr['task_quality']['judge_model']} judges, "
        f"rubric {xlmr['task_quality']['judge_rubric_version']} "
        f"({xlmr['task_quality']['judge_rubric_sha256'][:12]}…)",
        "",
        "MeetingBank 排除声明:golden 校验器硬拒 MeetingBank 源(decisions/data-strategy "
        "硬约束 1);本考卷逐条许可溯源见各 golden 条目 source 字段与 "
        "`eval/manifests/`(LongBench 子集)。",
        "",
        "## 分档对照",
        "",
        "| 档位 | 模型 | 压缩率(计费) | 压缩率(原生) | fact recall | token F0.5 | 质量 delta [95% CI] | 延迟 p95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for metrics, label in ((mbert, "mBERT"), (xlmr, "XLM-R")):
        for level in metrics["metrics"]["levels"]:
            a = level["aggregate"]
            tq = a["task_quality"]
            ci = tq["delta_ci95"]
            lines.append(
                f"| {level['aggressiveness']} | {label} "
                f"| {a['corpus_compression_ratio']:.4f} "
                f"| {a['corpus_native_compression_ratio']:.4f} "
                f"| {a['mean_fact_recall']:.4f} "
                f"| {a['mean_token_f05']:.4f} "
                f"| {tq['mean_delta']:+.4f} [{ci['low']:+.4f}, "
                f"{ci['high']:+.4f}] "
                f"| {a['latency']['p95_seconds']:.4f}s |"
            )

    rows_a, rows_b = rows_by_key(mbert), rows_by_key(xlmr)
    lines += ["", "## 后手牌判据(XLM-R-large vs mBERT-base,逐档配对)", ""]
    quality_verdicts = []
    for level in [l["aggressiveness"] for l in mbert["metrics"]["levels"]]:
        paired = []
        for key, row_a in rows_a.items():
            if key[0] != level or key not in rows_b:
                continue
            paired.append(
                rows_b[key]["delta"] - row_a["delta"]
            )
        mean, lo, hi = bootstrap_ci(paired)
        lines.append(
            f"- 档 {level}:质量 delta 差(XLM−mBERT){mean:+.4f},"
            f"95% CI [{lo:+.4f}, {hi:+.4f}],n={len(paired)}"
            f" {'— **显著更高**' if lo > 0 else '— 不显著' if hi >= 0 else '— 显著更低'}"
        )
        quality_verdicts.append(lo > 0)

    lat_a = [l["aggregate"]["latency"]["p95_seconds"] for l in mbert["metrics"]["levels"]]
    lat_b = [l["aggregate"]["latency"]["p95_seconds"] for l in xlmr["metrics"]["levels"]]
    ratio = max(lat_b) / max(lat_a) if max(lat_a) > 0 else float("inf")
    lines += [
        "",
        "## 延迟对比",
        "",
        f"- mBERT p95(最差档){max(lat_a):.4f}s;XLM-R p95(最差档){max(lat_b):.4f}s;"
        f"比值 {ratio:.2f}×(含 HTTP 开销的生产形态测量)。",
        "",
        "## 判据结论",
        "",
        f"- 质量显著更高(全部档位 CI 下界 > 0):{'是' if all(quality_verdicts) else '否'}"
        + ("" if all(quality_verdicts) else f"(显著档位:{sum(quality_verdicts)}/{len(quality_verdicts)})"),
        f"- 延迟可接受:XLM-R p95 = {max(lat_b)*1000:.0f}ms(比值 {ratio:.2f}×)",
        "- 接口相容:**否** — XLM-R-large 类属性为 `.roberta`,非 drop-in"
        "(decisions.md §6;宿主需配合改 `refiner.py` 才可用)",
        "",
        "> 后手牌激活需三判据同时成立(decisions.md §6 / roadmap 阶段 1)。",
    ]

    summary = "\n".join(lines) + "\n"
    out_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
