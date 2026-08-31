"""Pure quality metrics: fact recall, token-alignment F0.5, percentiles.

Calibers are shared with the refine runtime guardrail
(tokencamp-pro/sidecars/refine/refiner.py) so harness numbers compare
against runtime thresholds on the same algebra — the regex below is the
reused-on-both-sides pattern that references.md pins:

- fact recall: verbatim survival of numeric facts/codes under the SAME
  regex as refiner._FACT_PATTERN, same semantics (1.0 when the original
  carries no facts — nothing that could be lost);
- token F0.5: the F0.5 algebra of refiner._token_f1 (recall = mean over
  original tokens of best-match similarity, precision = mean over rewrite
  tokens, beta^2 = 0.25 so precision weighs twice), computed with
  token-IDENTITY similarity instead of embedding cosine — the harness
  carries no model runtime (issue #1: dependencies exclude the training
  stack). For extractive compression (a strict subsequence) identity
  alignment is exact, so harness F0.5 is a lower bound on the embedding
  caliber, in the same direction as the runtime threshold;
- percentile: linear interpolation between order statistics — the old
  benchmark's latency caliber.

All functions here are pure: no I/O, no network, no model loading.
"""

import re
from collections import Counter

# Verbatim copy of tokencamp-pro/sidecars/refine/refiner.py::_FACT_PATTERN
# (numeric facts / codes that must survive verbatim: order numbers
# (#A-12345), dates, percentages, measurements, versions, prices...).
# If the runtime pattern ever changes, this copy must change with it —
# the two calibers exist to be compared.
FACT_PATTERN = re.compile(r"#[\w-]+|\d+(?:[.,:/-]\d+)*%?")

_BETA2 = 0.25  # beta = 0.5: F0.5 weights precision twice


def extract_facts(text: str) -> list[str]:
    """Numeric facts/codes in `text`, in order of appearance."""
    return FACT_PATTERN.findall(text)


def fact_recall(original: str, compressed: str) -> float:
    """Share of the ORIGINAL's numeric facts surviving verbatim in
    `compressed` (runtime semantics: 1.0 when the original carries no
    facts — the exam is whether the original's facts survive, never
    whether the compression adds new ones)."""
    facts = extract_facts(original)
    if not facts:
        return 1.0
    kept = sum(1 for fact in facts if fact in compressed)
    return kept / len(facts)


def token_f05(original_tokens: list[str], compressed_tokens: list[str]) -> float:
    """BERTScore-style greedy token-alignment F0.5 over token lists.

    recall    = matched / |original|   (mean over original tokens of the
                                      best identity similarity)
    precision = matched / |compressed| (mean over compressed tokens)

    where `matched` counts greedy one-to-one token matches (multiset
    intersection). F0.5 weights precision twice: dropping filler is the
    point of compression (recall falls by design), while invented or
    drifted content shows up as ungrounded compressed tokens (precision).
    Empty compressed text scores 0.0 (nothing survived).
    """
    if not original_tokens:
        raise ValueError("original_tokens must be non-empty")
    if not compressed_tokens:
        return 0.0
    matched = sum((Counter(original_tokens) & Counter(compressed_tokens)).values())
    recall = matched / len(original_tokens)
    precision = matched / len(compressed_tokens)
    denominator = _BETA2 * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + _BETA2) * precision * recall / denominator


def text_token_f05(original: str, compressed: str) -> float:
    """token_f05 over whitespace-split words (the harness word caliber)."""
    return token_f05(original.split(), compressed.split())


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (p in [0, 100]).

    Between order statistics, interpolates linearly (the old benchmark's
    caliber). Returns None for an empty list.
    """
    if not values:
        return None
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"p must be within [0, 100], got {p}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])
