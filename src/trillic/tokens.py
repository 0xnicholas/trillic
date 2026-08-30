"""Token counting and compression-ratio math.

Caliber convention (matches docs/evaluation.md and the 2026-07-28 reference
numbers): compression_ratio is the fraction of tokens REMOVED, i.e.
1 - compressed/original. kept_ratio is the retention counterpart.
"""

import tiktoken

_ROUND_DIGITS = 4


class TokenCounter:
    """tiktoken-backed token counter (billing caliber).

    The encoding name is configurable so the harness can follow whatever the
    gateway bills with; cl100k_base is the default for continuity with the
    reference benchmark numbers.
    """

    def __init__(self, encoding_name: str) -> None:
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except ValueError as e:
            raise ValueError(
                f"unknown tiktoken encoding {encoding_name!r}: {e}"
            ) from e
        self.encoding_name = encoding_name

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))


def kept_ratio(original_tokens: int, compressed_tokens: int) -> float:
    """Retention: compressed_tokens / original_tokens, rounded to 4 decimals."""
    _check_counts(original_tokens, compressed_tokens)
    return round(compressed_tokens / original_tokens, _ROUND_DIGITS)


def compression_ratio(original_tokens: int, compressed_tokens: int) -> float:
    """Fraction removed: 1 - compressed/original, rounded to 4 decimals."""
    _check_counts(original_tokens, compressed_tokens)
    return round(1 - compressed_tokens / original_tokens, _ROUND_DIGITS)


def _check_counts(original_tokens: int, compressed_tokens: int) -> None:
    if original_tokens <= 0:
        raise ValueError(
            f"original_tokens must be positive, got {original_tokens}"
        )
    if compressed_tokens < 0:
        raise ValueError(
            f"compressed_tokens must be non-negative, got {compressed_tokens}"
        )
