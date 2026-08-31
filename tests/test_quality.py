"""Pure quality metrics: fact recall, token-alignment F0.5, percentiles.

Calibers are shared with the refine runtime guardrail
(tokencamp-pro/sidecars/refine/refiner.py) so harness numbers compare
against runtime thresholds on the same algebra:

- fact recall: verbatim survival of numeric facts/codes under the SAME
  regex as refiner._FACT_PATTERN (references.md: _FACT_PATTERN 两侧复用);
- token F0.5: same F0.5 algebra as refiner._token_f1 (recall = mean over
  original tokens of best-match similarity, precision = mean over rewrite
  tokens, beta^2 = 0.25), with token-identity similarity instead of
  embedding cosine — the harness carries no model runtime (issue #1:
  依赖不含训练栈). For extractive compression (strict subsequence) identity
  alignment is exact;
- percentile: linear interpolation, the old benchmark's caliber.
"""

import pytest

from trillic.quality import (
    FACT_PATTERN,
    extract_facts,
    fact_recall,
    percentile,
    text_token_f05,
    token_f05,
)


class TestFactPattern:
    def test_extracts_ticket_codes_dates_percentages_versions(self):
        text = (
            "Ticket TK-88123 ref #A-12345, opened 2026-07-19 at 09:14, "
            "v2.14.3, up 12.5%, $84k owed, ratio 3/4, phone 555-0100"
        )
        facts = extract_facts(text)
        assert "2026-07-19" in facts
        assert "12.5%" in facts
        assert "2.14.3" in facts
        assert "3/4" in facts
        assert "555-0100" in facts
        assert "#A-12345" in facts  # hash-prefixed code form
        assert "84" in facts  # bare digits inside $84k

    def test_hash_token_alone_is_a_fact(self):
        assert extract_facts("see #TK-88123 for details") == ["#TK-88123"]

    def test_no_digits_no_facts(self):
        assert extract_facts("no numeric facts here at all") == []

    def test_pattern_is_verbatim_runtime_copy(self):
        """Caliber continuity: the regex must equal refiner._FACT_PATTERN
        (both sides must drift together, never apart)."""
        assert FACT_PATTERN.pattern == r"#[\w-]+|\d+(?:[.,:/-]\d+)*%?"


class TestFactRecall:
    def test_all_facts_kept_is_one(self):
        original = "issued 2026-07-19 refund 12.5% ticket TK-88123"
        compressed = "refund 12.5% issued 2026-07-19 ticket TK-88123 kept"
        assert fact_recall(original, compressed) == 1.0

    def test_no_facts_in_original_is_one(self):
        """Runtime semantics: fact recall is 1.0 when the original carries
        no facts (nothing that could be lost)."""
        assert fact_recall("plain words only", "plain only") == 1.0

    def test_all_facts_dropped_is_zero(self):
        original = "invoice 1001 dated 2026-07-19 total 84"
        compressed = "invoice dated total"
        assert fact_recall(original, compressed) == 0.0

    def test_partial_survival_fraction(self):
        original = "one 11 two 22 three 33 four 44"
        compressed = "one 11 three 33 four 44"
        assert fact_recall(original, compressed) == 0.75

    def test_fact_must_survive_verbatim(self):
        """A truncated fact (2026-07 dropped from 2026-07-19) does NOT
        count as survival."""
        original = "meeting on 2026-07-19"
        compressed = "meeting on 2026-07"
        assert fact_recall(original, compressed) == 0.0

    def test_empty_original_facts_checked_not_compressed_facts(self):
        """Facts are extracted from the ORIGINAL only (the exam is: did the
        original's facts survive); compressed-side facts are irrelevant."""
        original = "a 1 b"
        compressed = "a b 999"
        assert fact_recall(original, compressed) == 0.0


class TestTokenF05:
    def test_identical_texts_score_one(self):
        tokens = "the quick brown fox".split()
        assert token_f05(tokens, list(tokens)) == 1.0

    def test_empty_compressed_scores_zero(self):
        """All-deleted edge: recall 0, precision degenerate -> F0.5 = 0."""
        assert token_f05(["a", "b"], []) == 0.0

    def test_extractive_subsequence_has_precision_one(self):
        """Strict-subsequence compression: every kept token is grounded,
        so precision = 1 and F0.5 is recall-weighted."""
        original = ["keep", "this", "important", "fact", "and", "drop", "filler"]
        compressed = ["keep", "this", "fact"]
        # recall = 3/7, precision = 1.0
        expected = (1 + 0.25) * 1.0 * (3 / 7) / (0.25 * 1.0 + 3 / 7)
        assert token_f05(original, compressed) == pytest.approx(expected)

    def test_repeated_tokens_are_matched_one_to_one(self):
        """Greedy one-to-one alignment: a token consumed once cannot match
        a second copy — duplicates must exist on both sides."""
        original = ["a", "a", "b"]
        compressed = ["a", "a", "a"]
        # matched pairs = 2 (a, a); recall = 2/3, precision = 2/3
        expected = (1.25 * (2 / 3) * (2 / 3)) / (0.25 * (2 / 3) + 2 / 3)
        assert token_f05(original, compressed) == pytest.approx(expected)

    def test_invented_tokens_cut_precision(self):
        """Non-extractive rewrite (invented token) is punished by
        precision, the side F0.5 weights twice."""
        original = ["real", "facts"]
        compressed = ["real", "hallucinated"]
        # matched = 1; precision = 1/2, recall = 1/2
        expected = (1.25 * 0.5 * 0.5) / (0.25 * 0.5 + 0.5)
        assert token_f05(original, compressed) == pytest.approx(expected)

    def test_empty_original_is_rejected(self):
        with pytest.raises(ValueError, match="original"):
            token_f05([], ["a"])

    def test_text_wrapper_splits_on_whitespace(self):
        assert text_token_f05("a b c", "a b c") == 1.0
        assert text_token_f05("a b c", "") == 0.0

    def test_f05_weights_precision_over_recall(self):
        """Same matched mass, two shapes: the more precise one must win
        (F0.5, not F1)."""
        # shape A: P=1.0, R=0.5 (drop half)
        # shape B: P=0.5, R=1.0 (half invented)
        a = token_f05(["x", "y"], ["x"])
        b = token_f05(["x"], ["x", "z"])
        assert a > b


class TestPercentile:
    def test_p95_of_known_values(self):
        values = [1.0] * 95 + [10.0] * 5  # 100 samples
        assert percentile(values, 95) == pytest.approx(1.45)

    def test_interpolates_between_order_statistics(self):
        values = [10.0, 20.0, 30.0, 40.0]
        # linear interpolation: index = 0.95 * 3 = 2.85 -> 30 + 0.85*10
        assert percentile(values, 95) == pytest.approx(38.5)

    def test_single_value(self):
        assert percentile([7.0], 95) == 7.0

    def test_empty_returns_none(self):
        assert percentile([], 95) is None
