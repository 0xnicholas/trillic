"""Sizing math (issue #8): judge variance, effect size, required n.

Pilot-ramp sizing (docs/evaluation.md 规模定案): run the pilot, measure the
judge's score variance and the compression-harm effect size, then invert a
one-sided z-test to get the per-class sample count that would separate the
mean delta from zero. All pure functions.
"""

import pytest

from trillic.sizing import (
    effect_size,
    judge_score_variance,
    required_n,
    sizing_report,
)


class TestJudgeScoreVariance:
    def test_variance_of_original_and_compressed_scores(self):
        rows = [
            {"original_score": 1.0, "compressed_score": 0.5},
            {"original_score": 0.5, "compressed_score": 0.5},
            {"original_score": 0.75, "compressed_score": 0.25},
        ]
        var = judge_score_variance(rows)
        assert var["original"] == pytest.approx(
            ((1.0 - 0.75) ** 2 + (0.5 - 0.75) ** 2 + (0.75 - 0.75) ** 2) / 2
        )  # sample variance (n-1)
        assert var["compressed"] == pytest.approx(((0.5 - 5 / 12) ** 2 * 2 + (0.25 - 5 / 12) ** 2) / 2)

    def test_single_row_is_zero_variance(self):
        var = judge_score_variance([{"original_score": 0.5, "compressed_score": 0.5}])
        assert var["original"] == 0.0
        assert var["compressed"] == 0.0


class TestEffectSize:
    def test_mean_std_and_cohens_d(self):
        rows = [
            {"delta": -0.2},
            {"delta": -0.2},
            {"delta": -0.2},
            {"delta": -0.6},
        ]
        es = effect_size([r["delta"] for r in rows])
        assert es["mean_delta"] == pytest.approx(-0.3)
        # sample std (n-1): mean -.3, devs .01/.01/.01/.09 -> var .12/3 = .04
        assert es["std_delta"] == pytest.approx(0.2)
        assert es["cohens_d"] == pytest.approx(-1.5)

    def test_constant_deltas_degenerate(self):
        es = effect_size([-0.25, -0.25, -0.25])
        assert es["std_delta"] == 0.0
        assert es["cohens_d"] is None  # undefined; sizing must handle


class TestRequiredN:
    def test_hand_computed_one_sided_z_formula(self):
        # n = ((z_{0.95} + z_{0.80}) * sigma / delta)^2, z=1.6449/0.8416
        # sigma=0.10, delta=-0.05 -> ((2.4865)*2)^2 = 24.73 -> 25
        n = required_n(mean_delta=-0.05, std_delta=0.10)
        assert n == pytest.approx(24.74, abs=0.5)  # raw value before ceil
        import math

        assert math.ceil(n) == 25

    def test_returns_raw_float_for_the_record(self):
        n = required_n(mean_delta=-0.10, std_delta=0.10)
        assert n == pytest.approx((1.6449 + 0.8416) ** 2, rel=1e-3)

    def test_zero_effect_is_rejected(self):
        with pytest.raises(ValueError, match="delta"):
            required_n(mean_delta=0.0, std_delta=0.1)


class TestSizingReport:
    def test_report_from_level_task_quality_rows(self):
        rows = []
        for load_type, delta in (("rag", -0.1), ("rag", -0.3), ("dialogue", -0.2)):
            rows.append(
                {
                    "id": f"x-{load_type}",
                    "load_type": load_type,
                    "delta": delta,
                    "original_score": 0.8,
                    "compressed_score": 0.8 - delta,
                }
            )
        report = sizing_report(rows)
        assert set(report["by_load_type"]) == {"rag", "dialogue"}
        assert report["overall"]["mean_delta"] == pytest.approx(-0.2)
        assert report["overall"]["n_required_raw"] >= 0
        # per-class n uses only that class's deltas
        assert report["by_load_type"]["rag"]["mean_delta"] == pytest.approx(-0.2)

    def test_degenerate_zero_std_reports_none(self):
        rows = [{"delta": -0.25, "original_score": 1.0, "compressed_score": 0.75, "load_type": "rag", "id": "x"}]
        report = sizing_report(rows)
        assert report["overall"]["n_required_raw"] is None
        assert report["overall"]["degenerate"] is True
