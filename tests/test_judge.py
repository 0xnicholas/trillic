"""LLM judge: versioned rubric, prompt protocol, strict score parsing, and
the deterministic mechanical scorer the stub gateway uses.

The rubric hash is the versioning contract (issue #7: judge rubric 版本化,
哈希进报告): RUBRIC_TEXT is exactly what every judge call carries, so its
sha256 identifies the grading behavior in force for a run.
"""

import hashlib
import json

import pytest

from trillic.judge import (
    ANSWER_FENCE_END,
    ANSWER_FENCE_START,
    JUDGE_PROMPT_MARKER,
    RUBRIC_TEXT,
    RUBRIC_VERSION,
    JudgeError,
    judge_prompt,
    mechanical_coverage_score,
    mechanical_judge_response,
    parse_judge_scores,
    rubric_sha256,
    score_of,
    split_judge_prompt,
    significant_words,
)

KEY_POINTS = ["refund window is 14 business days", "late fee is 1.5% per month"]


class TestRubricVersioning:
    def test_version_is_pinned(self):
        assert RUBRIC_VERSION == "1"

    def test_hash_is_sha256_of_rubric_text(self):
        assert rubric_sha256() == hashlib.sha256(RUBRIC_TEXT.encode("utf-8")).hexdigest()

    def test_rubric_states_the_binary_rule_and_the_response_contract(self):
        text = RUBRIC_TEXT
        assert "0" in text and "1" in text  # binary scale
        assert "scores" in text  # response shape
        assert "only the answer" in text.lower() or "answer alone" in text.lower()


class TestJudgePrompt:
    def test_starts_with_protocol_marker_and_carries_key_points_json(self):
        prompt = judge_prompt(KEY_POINTS, "some answer")
        assert prompt.startswith(JUDGE_PROMPT_MARKER)
        envelope_line = prompt.splitlines()[0]
        payload = json.loads(envelope_line[len(JUDGE_PROMPT_MARKER):])
        assert payload == {"key_points": KEY_POINTS}

    def test_carries_rubric_and_fenced_answer(self):
        answer = "Refunds process in 14 business days.\nSecond line."
        prompt = judge_prompt(KEY_POINTS, answer)
        assert RUBRIC_TEXT in prompt
        assert f"{ANSWER_FENCE_START}\n{answer}\n{ANSWER_FENCE_END}" in prompt

    def test_round_trips_through_split_judge_prompt(self):
        prompt = judge_prompt(KEY_POINTS, "the answer text")
        parsed = split_judge_prompt(prompt)
        assert parsed == (KEY_POINTS, "the answer text")

    def test_split_returns_none_for_non_judge_prompts(self):
        assert split_judge_prompt("just a chat prompt") is None
        assert split_judge_prompt(f"x{JUDGE_PROMPT_MARKER}{{}}") is None

    def test_malformed_envelope_raises(self):
        with pytest.raises(JudgeError, match="envelope"):
            split_judge_prompt(f"{JUDGE_PROMPT_MARKER}not json at all\nrest")

    def test_missing_answer_fence_raises(self):
        prompt = f"{JUDGE_PROMPT_MARKER}{{\"key_points\": [\"a\"]}}\n\nrubric but no fence\n"
        with pytest.raises(JudgeError, match="fence"):
            split_judge_prompt(prompt)


class TestParseJudgeScores:
    def test_bare_json(self):
        assert parse_judge_scores('{"scores": [1, 0]}', 2) == [1, 0]

    def test_json_wrapped_in_prose(self):
        raw = 'Here is my grading:\n{"scores": [1, 1]}\nDone.'
        assert parse_judge_scores(raw, 2) == [1, 1]

    def test_float_zeros_and_ones_coerced(self):
        assert parse_judge_scores('{"scores": [1.0, 0.0]}', 2) == [1, 0]

    def test_empty_scores_for_empty_key_points(self):
        assert parse_judge_scores('{"scores": []}', 0) == []

    def test_wrong_length_rejected(self):
        with pytest.raises(JudgeError, match="expected 2"):
            parse_judge_scores('{"scores": [1]}', 2)

    def test_non_binary_value_rejected(self):
        with pytest.raises(JudgeError, match="0 or 1"):
            parse_judge_scores('{"scores": [0.5, 1]}', 2)

    def test_boolean_rejected(self):
        with pytest.raises(JudgeError, match="0 or 1"):
            parse_judge_scores('{"scores": [true, false]}', 2)

    def test_missing_scores_field_rejected(self):
        with pytest.raises(JudgeError, match="scores"):
            parse_judge_scores('{"verdict": "good"}', 2)

    def test_garbage_rejected(self):
        with pytest.raises(JudgeError):
            parse_judge_scores("I cannot grade this.", 2)

    def test_scores_not_a_list_rejected(self):
        with pytest.raises(JudgeError, match="list"):
            parse_judge_scores('{"scores": 1}', 1)


class TestScoreOf:
    def test_mean_over_binary_scores(self):
        assert score_of((1, 1, 0)) == pytest.approx(2 / 3)
        assert score_of(()) == 0.0


class TestMechanicalCoverage:
    """The stub judge's scoring rule: a key point is covered iff every
    significant word (lowercased, >= 3 chars) appears in the answer."""

    def test_covered_point(self):
        answer = "The refund window is exactly 14 business days, to the original card."
        assert mechanical_coverage_score("refund window 14 business days", answer) == 1

    def test_missing_word_uncovered(self):
        answer = "The window is exactly 14 days."
        assert mechanical_coverage_score("refund window 14 business days", answer) == 0

    def test_case_insensitive(self):
        assert mechanical_coverage_score("Refund Window", "REFUND window ok") == 1

    def test_punctuation_ignored(self):
        assert mechanical_coverage_score("fee: 1.5%/month!", "fee is 1.5% per month") == 1

    def test_point_without_significant_words_is_covered(self):
        assert mechanical_coverage_score("a / b", "anything") == 1

    def test_significant_words_drop_short_tokens(self):
        # length >= 3 kept: "the" stays (both sides of the match share the
        # rule, so stop-words carry no asymmetry)
        assert significant_words("The cat is on TV now") == ["the", "cat", "now"]

    def test_monotone_in_answer_words(self):
        """Adding words to an answer can only raise coverage (the property
        the stub end-to-end relies on: echo(original) ⊇ echo(compressed))."""
        point = "escalate after two failed suggestions"
        short = "escalate after suggestions"
        longer = short + " two failed"
        assert mechanical_coverage_score(point, short) == 0
        assert mechanical_coverage_score(point, longer) == 1

    def test_stub_response_round_trips_through_the_parser(self):
        raw = mechanical_judge_response(KEY_POINTS, "14 business days answer")
        scores = parse_judge_scores(raw, len(KEY_POINTS))
        expected = [
            mechanical_coverage_score(point, "14 business days answer")
            for point in KEY_POINTS
        ]
        assert scores == expected
