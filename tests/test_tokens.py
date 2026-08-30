"""Token counting and compression-ratio math (pure-function layer).

Caliber convention (matches docs/evaluation.md reference numbers):
  kept_ratio        = compressed_tokens / original_tokens   (retention)
  compression_ratio = 1 - kept_ratio                        (fraction removed;
                      the aggressiveness-sweep metric)
The pinned token counts below were verified by hand against cl100k_base.
"""

import pytest

from trillic.tokens import TokenCounter, compression_ratio, kept_ratio


def test_counter_pinned_hand_checked_counts():
    counter = TokenCounter("cl100k_base")
    assert counter.count("hello world") == 2
    assert counter.count("Hello, world!") == 4


def test_counter_empty_string_is_zero():
    assert TokenCounter("cl100k_base").count("") == 0


def test_counter_rejects_unknown_encoding():
    with pytest.raises(ValueError, match="no-such-encoding"):
        TokenCounter("no-such-encoding")


def test_ratios_basic_math():
    assert kept_ratio(100, 25) == 0.25
    assert compression_ratio(100, 25) == 0.75
    assert kept_ratio(100, 100) == 1.0
    assert compression_ratio(100, 100) == 0.0


def test_ratios_rounded_to_four_decimals():
    assert kept_ratio(3, 1) == 0.3333
    assert compression_ratio(3, 1) == 0.6667


def test_ratios_reject_non_positive_original():
    with pytest.raises(ValueError, match="original"):
        kept_ratio(0, 0)
    with pytest.raises(ValueError, match="original"):
        compression_ratio(-5, 0)
