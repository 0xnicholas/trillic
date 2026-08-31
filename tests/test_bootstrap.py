"""Paired bootstrap CI: pure function, seeded assertions.

The pinned literals below were derived by running the implementation once
(seed fixed) and copying the numbers — the same hand-check style as
test_cli_eval_run.HAND_CHECKED. They exist to catch ANY drift in the RNG
draws, the mean algebra, or the percentile caliber (each is load-bearing:
the acceptance criterion reads the CI lower bound).
"""

import random

import pytest

from trillic.bootstrap import (
    METHOD,
    BootstrapError,
    BootstrapResult,
    paired_bootstrap_ci,
)

# Pinned literals (seed=7, defaults otherwise). Re-derive deliberately if
# the resampling or percentile caliber ever changes — never silently.
PINNED_SEED7_3ITEMS = {
    "deltas": [-0.60, 0.00, 0.25],
    "mean_delta": -0.11666666666666667,
    "ci_low": -0.60,
    "ci_high": 0.25,
    "ci_low_100resamples": -0.6,
    "ci_high_100resamples": 0.25,
}


def resampled_means_by_hand(deltas, n_resamples, seed):
    """Independent reimplementation of the resampling loop (same documented
    caliber: one random() draw per index, floor via int())."""
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[int(rng.random() * n)]
        means.append(total / n)
    return means


class TestPairedBootstrapCi:
    def test_pinned_literal_values(self):
        deltas = PINNED_SEED7_3ITEMS["deltas"]
        result = paired_bootstrap_ci(deltas, n_resamples=10_000, seed=7)
        assert result.mean_delta == pytest.approx(PINNED_SEED7_3ITEMS["mean_delta"])
        assert result.ci_low == pytest.approx(PINNED_SEED7_3ITEMS["ci_low"])
        assert result.ci_high == pytest.approx(PINNED_SEED7_3ITEMS["ci_high"])

    def test_pinned_literals_scale_down_to_100_resamples(self):
        deltas = PINNED_SEED7_3ITEMS["deltas"]
        result = paired_bootstrap_ci(deltas, n_resamples=100, seed=7)
        assert result.ci_low == pytest.approx(PINNED_SEED7_3ITEMS["ci_low_100resamples"])
        assert result.ci_high == pytest.approx(
            PINNED_SEED7_3ITEMS["ci_high_100resamples"]
        )

    def test_matches_independent_reimplementation(self):
        deltas = [-0.5, -0.2, 0.0, 0.1, 0.4, -0.3]
        result = paired_bootstrap_ci(deltas, n_resamples=2_000, seed=42)
        means = sorted(resampled_means_by_hand(deltas, 2_000, 42))
        # linear-interpolation percentile, computed by hand here
        def pct(ordered, p):
            rank = (p / 100) * (len(ordered) - 1)
            lower = int(rank)
            upper = min(lower + 1, len(ordered) - 1)
            frac = rank - lower
            return ordered[lower] + frac * (ordered[upper] - ordered[lower])

        assert result.mean_delta == pytest.approx(sum(deltas) / len(deltas))
        assert result.ci_low == pytest.approx(pct(means, 2.5))
        assert result.ci_high == pytest.approx(pct(means, 97.5))

    def test_same_seed_is_bit_identical(self):
        deltas = [0.1, -0.4, 0.2, 0.0]
        a = paired_bootstrap_ci(deltas, n_resamples=500, seed=3)
        b = paired_bootstrap_ci(deltas, n_resamples=500, seed=3)
        assert a == b

    def test_different_seed_moves_the_bounds(self):
        deltas = [0.1, -0.4, 0.2, 0.0, -0.1, 0.3]
        a = paired_bootstrap_ci(deltas, n_resamples=500, seed=1)
        b = paired_bootstrap_ci(deltas, n_resamples=500, seed=2)
        assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)

    def test_constant_deltas_degenerate_to_a_point(self):
        """All pairs share one delta: every resample mean is that delta, so
        the CI collapses onto the point estimate."""
        result = paired_bootstrap_ci([-0.5] * 4, n_resamples=1_000, seed=0)
        assert result.mean_delta == pytest.approx(-0.5)
        assert result.ci_low == pytest.approx(-0.5)
        assert result.ci_high == pytest.approx(-0.5)

    def test_brackets_the_point_estimate(self):
        deltas = [-0.2, 0.3, -0.1, 0.05, -0.35, 0.2, 0.0]
        for seed in range(5):
            result = paired_bootstrap_ci(deltas, n_resamples=1_000, seed=seed)
            assert result.ci_low <= result.mean_delta <= result.ci_high

    def test_uniformly_negative_deltas_separate_from_zero(self):
        """The acceptance-relevant case: consistent harm -> CI upper bound
        below zero."""
        result = paired_bootstrap_ci(
            [-0.6, -0.5, -0.7, -0.4, -0.8, -0.55, -0.65, -0.5],
            n_resamples=10_000,
            seed=0,
        )
        assert result.mean_delta < 0
        assert result.ci_high < 0

    def test_uniformly_positive_deltas_separate_from_zero(self):
        result = paired_bootstrap_ci(
            [0.6, 0.5, 0.7, 0.4, 0.8, 0.55, 0.65, 0.5],
            n_resamples=10_000,
            seed=0,
        )
        assert result.ci_low > 0
        assert result.ci_lower_bound_not_negative is True

    def test_result_records_run_parameters(self):
        result = paired_bootstrap_ci([0.1, -0.1], n_resamples=250, seed=9)
        assert isinstance(result, BootstrapResult)
        assert result.n_resamples == 250
        assert result.n_items == 2
        assert result.seed == 9
        assert result.confidence == 0.95
        assert METHOD  # method description is a non-empty constant

    def test_confidence_is_configurable(self):
        deltas = [-0.6, 0.0, 0.25, -0.1, 0.2]
        wide = paired_bootstrap_ci(deltas, n_resamples=10_000, seed=5, confidence=0.90)
        narrow = paired_bootstrap_ci(deltas, n_resamples=10_000, seed=5, confidence=0.99)
        assert wide.confidence == 0.90
        assert narrow.confidence == 0.99
        assert wide.ci_low >= narrow.ci_low
        assert wide.ci_high <= narrow.ci_high


class TestPairedBootstrapCiErrors:
    def test_empty_deltas_rejected(self):
        with pytest.raises(BootstrapError, match="non-empty"):
            paired_bootstrap_ci([])

    def test_zero_resamples_rejected(self):
        with pytest.raises(BootstrapError, match="n_resamples"):
            paired_bootstrap_ci([0.1], n_resamples=0)

    def test_bad_confidence_rejected(self):
        with pytest.raises(BootstrapError, match="confidence"):
            paired_bootstrap_ci([0.1], confidence=0.0)
        with pytest.raises(BootstrapError, match="confidence"):
            paired_bootstrap_ci([0.1], confidence=1.0)
