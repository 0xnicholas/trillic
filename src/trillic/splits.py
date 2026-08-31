"""Content-addressed half-splits over LongBench subsets.

Method (recorded verbatim in every generated manifest):

    fingerprint = sha1(context + "\\x00" + input)   # content addressing
    sort unique fingerprints ascending (hex string);
    first ceil(n/2) -> eval half, remainder -> train half.

Sorting before halving makes the assignment independent of row order in the
source file, so a given entry always lands on the same side. Golden sets are
built exclusively from the eval half (enforced in trillic.longbench and
re-asserted by trillic.golden's split check).
"""

import hashlib
import math


def entry_fingerprint(context: str, question: str) -> str:
    """sha1 over context and question, NUL-delimited (byte-exact)."""
    digest = hashlib.sha1()
    digest.update(context.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(question.encode("utf-8"))
    return digest.hexdigest()


def row_fingerprint(row: dict) -> str:
    """Fingerprint a LongBench-style row ({context, input, ...})."""
    return entry_fingerprint(row["context"], row["input"])


def assign_splits(fingerprints: list[str]) -> dict[str, str]:
    """Deterministically assign each unique fingerprint to 'eval' or 'train'.

    The eval half is the larger half on odd counts: golden (the exam) never
    loses entries to training.
    """
    unique = sorted(set(fingerprints))
    boundary = math.ceil(len(unique) / 2)
    return {
        fp: ("eval" if index < boundary else "train")
        for index, fp in enumerate(unique)
    }


def split_half_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split LongBench-style rows ({context, input, answers, ...}).

    Returns (eval_rows, train_rows), each sorted by fingerprint so output is
    stable regardless of input order.
    """
    keyed = sorted(((row_fingerprint(row), row) for row in rows), key=lambda pair: pair[0])
    assignments = assign_splits([fp for fp, _ in keyed])
    eval_rows = [row for fp, row in keyed if assignments[fp] == "eval"]
    train_rows = [row for fp, row in keyed if assignments[fp] == "train"]
    return eval_rows, train_rows
