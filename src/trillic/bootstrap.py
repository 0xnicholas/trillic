"""Paired bootstrap CI (issue #7): seeded, pure, no I/O.

The acceptance criterion (docs/evaluation.md 验收标准) compares a fine-tuned
checkpoint against the public one on task-level quality and calls the delta
significant when the 95% CI lower bound of the paired bootstrap is not
negative. "Paired" = the two conditions answer the SAME golden prompts, so
the resample unit is the prompt: draw item indices with replacement, recompute
the mean delta each draw, read CI bounds off the resampled means.

Calibers, pinned deliberately:
- RNG: random.Random(seed) with one random() draw per index — random() is
  the one generator CPython documents as stable across versions, so seeded
  runs reproduce bit-for-bit (the pinned literals in test_bootstrap.py
  exist to catch drift);
- percentile bounds: the SAME linear-interpolation percentile as latency
  p95 (trillic.quality.percentile — the old benchmark's caliber);
- point estimate: the plain mean of the observed paired deltas (bootstrap
  distributions center on it; the report never quotes a resampled mean as
  the estimate).
"""

import random
from dataclasses import dataclass

from trillic.quality import percentile

METHOD = "paired percentile bootstrap (resample prompts with replacement)"
DEFAULT_CONFIDENCE = 0.95  # the 95% CI the acceptance criterion reads


class BootstrapError(ValueError):
    """The bootstrap inputs were invalid (no pairs, bad params)."""


@dataclass(frozen=True)
class BootstrapResult:
    mean_delta: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    n_items: int
    seed: int

    @property
    def ci_lower_bound_not_negative(self) -> bool:
        """The acceptance judgment (docs/evaluation.md): the quality delta's
        CI lower bound is not negative."""
        return self.ci_low >= 0.0


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BootstrapResult:
    """95% (by default) CI for the mean of paired deltas, resampling the
    PAIRS (one pair = one prompt) with replacement.

    deltas[i] = compressed_score[i] - original_score[i] for prompt i; each
    resample keeps the pair intact, which is exactly what "按 prompt 重采样"
    requires.
    """
    if not deltas:
        raise BootstrapError("deltas must be non-empty (one delta per prompt)")
    if n_resamples < 1:
        raise BootstrapError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < confidence < 1.0:
        raise BootstrapError(f"confidence must be within (0, 1), got {confidence}")

    n = len(deltas)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[int(rng.random() * n)]
        means.append(total / n)

    alpha = (1.0 - confidence) / 2.0 * 100.0
    ci_low = percentile(means, alpha)
    ci_high = percentile(means, 100.0 - alpha)
    if ci_low is None or ci_high is None:
        # Unreachable (means is non-empty for n_resamples >= 1); the guard
        # documents the percentile contract instead of silencing it.
        raise BootstrapError("resampled means unexpectedly empty")
    return BootstrapResult(
        mean_delta=sum(deltas) / n,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n_resamples=n_resamples,
        n_items=n,
        seed=seed,
    )
