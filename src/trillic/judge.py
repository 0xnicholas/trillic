"""LLM judge over golden key_points (issue #7).

The judge grades ONE answer against the golden entry's key_points on a
binary per-point scale; the item score is the fraction of points covered
(1.0 = the answer satisfies the exam). The rubric below is the versioned
artifact: RUBRIC_VERSION is the human label, rubric_sha256() the content
identity that lands in every report — any wording change must bump the
version and will change the hash, so two runs with different grading
behavior can never be silently compared.

Prompt protocol: the first line is a machine-readable envelope
(TRILLIC-JUDGE/1 + one-line JSON with the key_points), followed by the
rubric and the answer inside stable fences. The envelope lets the
deterministic stub gateway (and any audit tooling) recover the semantic
inputs without parsing prose — the stub judge scores mechanically via
mechanical_coverage_score.
"""

import hashlib
import json
import re

RUBRIC_VERSION = "1"

RUBRIC_TEXT = """\
You are a strict grader. An answer to a task was produced from a prompt;
grade how well the answer preserves the required key points.

For each key point (listed as a JSON array on the envelope line), judge
independently:
- score 1 only if the answer alone fully satisfies the point — a reader
  who sees only the answer, not the original prompt, could confirm it;
- score 0 if the point is missing, only partially addressed, hedged, or
  contradicted.

Grade only what the answer states. Do not credit the prompt, do not infer
facts the answer does not state, and do not reward length or style.

Respond with ONLY a JSON object and no prose:
{"scores": [<0 or 1>, ...]} — exactly one score per key point, in order.
"""

JUDGE_PROMPT_MARKER = "TRILLIC-JUDGE/1 "
ANSWER_FENCE_START = "<<<ANSWER"
ANSWER_FENCE_END = "ANSWER>>>"

_WORD_RE = re.compile(r"[a-z0-9]+")
_MIN_WORD_LEN = 3


class JudgeError(Exception):
    """The judge model's output could not be parsed into per-point scores."""


def rubric_sha256() -> str:
    """Content identity of the rubric in force (report field)."""
    return hashlib.sha256(RUBRIC_TEXT.encode("utf-8")).hexdigest()


def judge_prompt(key_points: list[str], answer: str) -> str:
    envelope = json.dumps({"key_points": list(key_points)}, ensure_ascii=False)
    return (
        f"{JUDGE_PROMPT_MARKER}{envelope}\n\n"
        f"{RUBRIC_TEXT}\n"
        f"Answer to grade:\n{ANSWER_FENCE_START}\n{answer}\n{ANSWER_FENCE_END}\n"
    )


def split_judge_prompt(prompt: str) -> tuple[list[str], str] | None:
    """Recover (key_points, answer) from a judge prompt; None if `prompt`
    is not on the judge protocol (used by the stub gateway to route)."""
    if not prompt.startswith(JUDGE_PROMPT_MARKER):
        return None
    envelope_line, _, _rest = prompt.partition("\n")
    try:
        payload = json.loads(envelope_line[len(JUDGE_PROMPT_MARKER):])
        key_points = payload["key_points"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise JudgeError(f"malformed judge envelope: {e}") from e
    if not isinstance(key_points, list) or not all(
        isinstance(k, str) for k in key_points
    ):
        raise JudgeError("judge envelope key_points must be a list of strings")
    answer = _extract_fenced_answer(prompt)
    return list(key_points), answer


def _extract_fenced_answer(prompt: str) -> str:
    if f"{ANSWER_FENCE_START}\n" not in prompt:
        raise JudgeError("judge prompt is missing the answer fence")
    _, _, tail = prompt.partition(f"{ANSWER_FENCE_START}\n")
    body, _, _ = tail.partition(f"\n{ANSWER_FENCE_END}")
    if not body:
        raise JudgeError("judge prompt has an empty fenced answer")
    return body


def parse_judge_scores(raw: str, n_expected: int) -> list[int]:
    """Strictly parse the judge's reply into n_expected binary scores.

    Tolerates JSON wrapped in prose (finds the outermost braces) but
    nothing else: values must be 0/1 (1.0 tolerated as float noise), the
    count must match the key points, and the verdict must be a list.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise JudgeError(f"judge reply contains no JSON object: {raw[:120]!r}")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise JudgeError(f"judge reply is not valid JSON: {e}: {raw[:120]!r}") from e
    if not isinstance(data, dict) or "scores" not in data:
        raise JudgeError(f"judge reply missing 'scores': {raw[:120]!r}")
    scores = data["scores"]
    if not isinstance(scores, list):
        raise JudgeError(f"judge 'scores' must be a list, got {type(scores).__name__}")
    if len(scores) != n_expected:
        raise JudgeError(
            f"judge returned {len(scores)} scores, expected {n_expected} "
            f"(one per key point): {scores!r}"
        )
    parsed: list[int] = []
    for value in scores:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JudgeError(f"judge scores must be 0 or 1, got {value!r}")
        if value not in (0, 1, 0.0, 1.0):
            raise JudgeError(f"judge scores must be 0 or 1, got {value!r}")
        parsed.append(int(value))
    return parsed


def score_of(scores: list[int] | tuple[int, ...]) -> float:
    """Item score = fraction of key points covered (0.0 for no points)."""
    return sum(scores) / len(scores) if scores else 0.0


def significant_words(text: str) -> list[str]:
    """Lowercased alphanumeric runs of length >= 3 ('the'/'of'/punctuation
    carry no grading signal at this caliber)."""
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) >= _MIN_WORD_LEN]


def mechanical_coverage_score(key_point: str, answer: str) -> int:
    """Deterministic baseline judge: a key point is covered iff every one
    of its significant words appears in the answer.

    This is the stub gateway's judge — the behavior that makes the
    task-quality loop runnable (and directionally checkable) with zero
    network. It is NOT the report's quality number: real runs grade with
    the configured judge model.
    """
    words = significant_words(key_point)
    if not words:
        return 1
    answer_words = set(significant_words(answer))
    return 1 if all(word in answer_words for word in words) else 0


def mechanical_judge_response(key_points: list[str], answer: str) -> str:
    """Judge-protocol JSON a deterministic stub returns for (points,
    answer) — always parseable by parse_judge_scores."""
    scores = [mechanical_coverage_score(point, answer) for point in key_points]
    return json.dumps({"scores": scores})
