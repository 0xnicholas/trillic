"""Split discipline: content-addressed half-splits over LongBench subsets.

The rule (docs/data-strategy.md 硬约束 4): training corpus and eval golden
set must be split — each subset is explicitly halved, and golden entries are
drawn ONLY from the eval half. Content addressing (sha1 over the entry's
context + question) makes the split independent of file line order, so the
same entry always lands on the same side — forever, and auditable.
"""

import pytest

from trillic.splits import assign_splits, entry_fingerprint, split_half_rows


def make_rows(n: int, prefix: str = "doc") -> list[dict]:
    return [
        {"input": f"question {i}?", "context": f"{prefix} context number {i} body", "answers": [f"a{i}"]}
        for i in range(n)
    ]


class TestEntryFingerprint:
    def test_deterministic_and_content_addressed(self):
        a = entry_fingerprint("some context", "some question")
        assert a == entry_fingerprint("some context", "some question")
        assert a != entry_fingerprint("some context ", "some question")  # byte-exact
        assert a != entry_fingerprint("other context", "some question")

    def test_context_and_question_are_delimited(self):
        # no ambiguity: (ctx="ab", q="c") vs (ctx="a", q="bc") differ
        assert entry_fingerprint("ab", "c") != entry_fingerprint("a", "bc")


class TestAssignSplits:
    def test_partition_is_disjoint_and_exhaustive(self):
        fps = [entry_fingerprint(f"ctx {i}", f"q {i}") for i in range(10)]
        splits = assign_splits(fps)
        assert set(splits) == set(fps)
        eval_half = {fp for fp, side in splits.items() if side == "eval"}
        train_half = {fp for fp, side in splits.items() if side == "train"}
        assert not eval_half & train_half
        assert eval_half | train_half == set(fps)

    def test_eval_half_is_the_larger_half_on_odd_counts(self):
        for n in (10, 11):
            fps = [entry_fingerprint(f"ctx {i}", f"q {i}") for i in range(n)]
            splits = assign_splits(fps)
            eval_count = sum(1 for side in splits.values() if side == "eval")
            assert eval_count == (n + 1) // 2

    def test_deterministic_across_calls_and_orderings(self):
        fps = [entry_fingerprint(f"ctx {i}", f"q {i}") for i in range(20)]
        first = assign_splits(fps)
        assert first == assign_splits(fps)
        shuffled = list(fps)
        shuffled.reverse()
        assert first == assign_splits(shuffled)  # file line order never matters

    def test_same_fingerprint_same_side_forever(self):
        """The letter of the discipline: a given entry can never appear on
        both sides — one fingerprint, one split, order-independent."""
        fp = entry_fingerprint("the one entry", "the question")
        for _ in range(5):
            others = [entry_fingerprint(f"noise {i}", f"q{i}") for i in range(9)]
            assert assign_splits(others + [fp])[fp] == assign_splits(others + [fp])[fp]


class TestSplitHalfRows:
    def test_union_is_all_rows_intersection_empty(self):
        rows = make_rows(11)
        eval_rows, train_rows = split_half_rows(rows)
        assert len(eval_rows) == 6 and len(train_rows) == 5
        eval_fps = {entry_fingerprint(r["context"], r["input"]) for r in eval_rows}
        train_fps = {entry_fingerprint(r["context"], r["input"]) for r in train_rows}
        assert not eval_fps & train_fps
        assert len(eval_fps | train_fps) == 11

    def test_row_order_in_file_does_not_change_the_partition(self):
        rows = make_rows(12)
        e1, t1 = split_half_rows(rows)
        e2, t2 = split_half_rows(list(reversed(rows)))
        assert {r["context"] for r in e1} == {r["context"] for r in e2}
        assert {r["context"] for r in t1} == {r["context"] for r in t2}
