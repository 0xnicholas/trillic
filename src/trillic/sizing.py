"""Sizing math (issue #8): judge variance, effect size, required n.

Pilot-ramp sizing (docs/evaluation.md 规模定案): run the pilot, measure the
judge's score variance and the compression-harm effect size, then invert a
one-sided z-test on the paired deltas to get the per-class sample count
that separates the mean delta from zero:

    n = ((z_{1-alpha} + z_{1-power_target}) * sigma / delta)^2

with alpha = 0.05 (z = 1.6449) and a 0.8 power target (z = 0.8416),
matching the bootstrap CI's one-sided lower-bound criterion. All pure.
"""

import math
from collections import defaultdict

Z_ALPHA_95_ONE_SIDED = 1.6449
Z_POWER_80 = 0.8416


def judge_score_variance(rows: list[dict]) -> dict[str, float]:
    """Sample variance (n-1) of the original and compressed judge scores.

    Single-row inputs report 0.0 (variance is undefined; the pilot needs
    more data, not a crash).
    """
    return {
        "original": _sample_variance([r["original_score"] for r in rows]),
        "compressed": _sample_variance([r["compressed_score"] for r in rows]),
    }


def effect_size(deltas: list[float]) -> dict:
    """Mean/std of the paired deltas plus Cohen's d (std-standardized).

    Constant deltas give std 0 and cohens_d None — the caller reports the
    degeneracy instead of inventing an effect.
    """
    if not deltas:
        raise ValueError("deltas must be non-empty")
    mean = sum(deltas) / len(deltas)
    std = _sample_variance(deltas) ** 0.5
    return {
        "mean_delta": mean,
        "std_delta": std,
        "cohens_d": None if std == 0 else mean / std,
    }


def required_n(mean_delta: float, std_delta: float) -> float:
    """Raw (pre-ceil) sample count per class for the one-sided criterion.

    Sign-insensitive: harm magnitude is what sizing needs. Zero effect is
    rejected — an effect of zero cannot be powered for at any n.
    """
    if mean_delta == 0:
        raise ValueError(
            f"mean_delta is exactly 0 — no n separates it from zero; the "
            "effect (not the sample) is the problem"
        )
    if std_delta < 0:
        raise ValueError(f"std_delta must be non-negative, got {std_delta}")
    if std_delta == 0:
        return 0.0  # deterministic effect: one observation suffices
    return ((Z_ALPHA_95_ONE_SIDED + Z_POWER_80) * std_delta / abs(mean_delta)) ** 2


def sizing_report(rows: list[dict]) -> dict:
    """Sizing summary from task-quality item rows (deltas + scores).

    Overall plus per-load-type breakdown; degenerate cases (zero std)
    report n_required_raw None and degenerate True rather than crashing.
    """

    def summarize(subset: list[dict]) -> dict:
        deltas = [r["delta"] for r in subset]
        es = effect_size(deltas)
        degenerate = es["std_delta"] == 0 or es["mean_delta"] == 0
        raw = None if degenerate else required_n(es["mean_delta"], es["std_delta"])
        return {
            "n_observed": len(subset),
            **es,
            "n_required_raw": raw,
            "n_required": None if raw is None else math.ceil(raw),
            "degenerate": degenerate,
            "variance": judge_score_variance(subset),
        }

    by_load: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_load[row["load_type"]].append(row)
    return {
        "overall": summarize(rows),
        "by_load_type": {name: summarize(subset) for name, subset in sorted(by_load.items())},
    }


def _sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)
