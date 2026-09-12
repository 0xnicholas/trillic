"""Task-quality loop orchestration (issue #7): two answers per golden item
(original + compressed), both judged against the same key_points, per-level
delta + paired bootstrap CI.

Runs against the stub gateway + stub refine client: the whole loop is
deterministic and zero-network, and the stub's echo answers + mechanical
judge give a real signal (compression drops payload words → key-point
coverage drops).
"""

import pytest

from trillic.clients.gateway import GatewayError, StubGatewayClient
from trillic.clients.sidecar import StubRefineClient
from trillic.golden import GoldenItem
from trillic.judge import split_judge_prompt
from trillic.task_quality import TaskQualityLoop

SOURCE = {
    "dataset": "synthetic",
    "subset": "test",
    "license": "original",
    "split": "eval",
}

ITEMS = [
    GoldenItem(
        id="rag-1",
        load_type="rag",
        prompt=(
            "context: the refund window is 14 business days and the late "
            "fee is 1.5 percent per month after 30 days overdue."
        ),
        key_points=(
            "refund window 14 business days",
            "late fee 1.5 percent per month",
        ),
        source=SOURCE,
    ),
    GoldenItem(
        id="sys-1",
        load_type="system_prompt",
        prompt=(
            "system: escalate to a human supervisor after two failed "
            "suggestions and always quote ticket TK-88123."
        ),
        key_points=(
            "escalate after two failed suggestions",
            "quote ticket TK-88123",
        ),
        source=SOURCE,
    ),
    GoldenItem(
        id="chat-1",
        load_type="dialogue",
        prompt=(
            "user: hi. assistant: hello. user: my renewal date is March "
            "15th 2027 and my loyalty discount is 12 percent."
        ),
        key_points=(
            "renewal date March 15th 2027",
            "loyalty discount 12 percent",
        ),
        source=SOURCE,
    ),
]


class RecordingGateway:
    """Stub gateway that records every (model, prompt) call."""

    def __init__(self) -> None:
        self._stub = StubGatewayClient()
        self.calls: list[tuple[str, str]] = []

    def chat(self, model: str, prompt: str):
        self.calls.append((model, prompt))
        return self._stub.chat(model=model, prompt=prompt)


@pytest.fixture
def refined_texts() -> list[str]:
    sidecar = StubRefineClient()
    return [
        sidecar.refine(item.prompt, rewrite=False, compress=True, aggressiveness=0.2).refined_text
        for item in ITEMS
    ]


def make_loop(gateway) -> TaskQualityLoop:
    return TaskQualityLoop(
        gateway,
        answer_model="stub-answerer",
        judge_model="stub-judge",
        seed=0,
        n_resamples=2_000,
    )


class TestPriming:
    def test_originals_answered_and_judged_once(self):
        gateway = RecordingGateway()
        loop = make_loop(gateway)
        loop.prime_originals(ITEMS)
        answer_calls = [(m, p) for m, p in gateway.calls if m == "stub-answerer"]
        judge_calls = [(m, p) for m, p in gateway.calls if m == "stub-judge"]
        assert len(answer_calls) == len(ITEMS)
        assert len(judge_calls) == len(ITEMS)
        # answers carry the load-type instruction + the ORIGINAL payload
        for (model, prompt), item in zip(answer_calls, ITEMS):
            assert prompt.endswith(item.prompt)
        # judges grade against the item's key_points
        for (model, prompt), item in zip(judge_calls, ITEMS):
            key_points, _answer = split_judge_prompt(prompt)
            assert key_points == list(item.key_points)

    def test_original_scores_are_high_with_echo_answers(self):
        loop = make_loop(RecordingGateway())
        loop.prime_originals(ITEMS)
        for item in ITEMS:
            assert loop.original_score(item.id) == pytest.approx(1.0)

    def test_served_models_tracked_per_role(self):
        loop = make_loop(RecordingGateway())
        loop.prime_originals(ITEMS)
        loop.evaluate_level(0.2, ITEMS, [i.prompt for i in ITEMS])
        # the stub gateway echoes the requested id, so served == pinned
        assert loop.served_models() == {
            "answer": ["stub-answerer"],
            "judge": ["stub-judge"],
        }

    def test_priming_twice_is_rejected(self):
        gateway = RecordingGateway()
        loop = make_loop(gateway)
        loop.prime_originals(ITEMS)
        with pytest.raises(RuntimeError, match="once"):
            loop.prime_originals(ITEMS)
        assert len(gateway.calls) == 2 * len(ITEMS)


class TestEvaluateLevel:
    @pytest.fixture(autouse=True)
    def _loop(self, refined_texts):
        self.gateway = RecordingGateway()
        self.loop = make_loop(self.gateway)
        self.loop.prime_originals(ITEMS)
        calls_before = len(self.gateway.calls)
        self.block = self.loop.evaluate_level(0.2, ITEMS, refined_texts)
        self.new_calls = self.gateway.calls[calls_before:]

    def test_block_shape(self):
        assert set(self.block) == {
            "aggressiveness",
            "item_count",
            "mean_original_score",
            "mean_compressed_score",
            "mean_delta",
            "delta_ci95",
            "ci_lower_bound_not_negative",
            "items",
        }
        assert self.block["aggressiveness"] == 0.2
        assert self.block["item_count"] == len(ITEMS)

    def test_two_answers_and_two_judges_per_item(self):
        answers = [c for c in self.new_calls if c[0] == "stub-answerer"]
        judges = [c for c in self.new_calls if c[0] == "stub-judge"]
        assert len(answers) == len(ITEMS)
        assert len(judges) == len(ITEMS)

    def test_judge_sees_the_same_key_points_for_both_conditions(self):
        # the compressed judge calls must grade the SAME golden key_points
        judge_prompts = [p for m, p in self.new_calls if m == "stub-judge"]
        graded_points = [split_judge_prompt(p)[0] for p in judge_prompts]
        assert graded_points == [list(item.key_points) for item in ITEMS]

    def test_compressed_answer_used_the_refined_payload(self, refined_texts):
        answer_prompts = [p for m, p in self.new_calls if m == "stub-answerer"]
        for prompt, refined in zip(answer_prompts, refined_texts):
            assert prompt.endswith(refined)

    def test_per_item_rows_and_scores(self):
        rows = {row["id"]: row for row in self.block["items"]}
        assert set(rows) == {item.id for item in ITEMS}
        for item in ITEMS:
            row = rows[item.id]
            assert row["load_type"] == item.load_type
            assert row["original_score"] == pytest.approx(1.0)
            assert 0.0 <= row["compressed_score"] <= 1.0
            assert row["delta"] == pytest.approx(
                round(row["compressed_score"] - row["original_score"], 4)
            )
            assert len(row["original_judge_scores"]) == len(item.key_points)
            assert len(row["compressed_judge_scores"]) == len(item.key_points)

    def test_stub_compression_never_raises_scores(self):
        for row in self.block["items"]:
            assert row["compressed_score"] <= row["original_score"]

    def test_means_match_rows(self):
        rows = self.block["items"]
        assert self.block["mean_original_score"] == pytest.approx(
            round(sum(r["original_score"] for r in rows) / len(rows), 4)
        )
        assert self.block["mean_delta"] == pytest.approx(
            round(sum(r["delta"] for r in rows) / len(rows), 4), abs=1e-4
        )

    def test_ci_brackets_the_mean_delta(self):
        ci = self.block["delta_ci95"]
        assert ci["low"] <= self.block["mean_delta"] <= ci["high"]
        assert self.block["ci_lower_bound_not_negative"] == (ci["low"] >= 0.0)

    def test_seeded_bootstrap_is_deterministic(self, refined_texts):
        def run_once():
            loop = make_loop(RecordingGateway())
            loop.prime_originals(ITEMS)
            return loop.evaluate_level(0.2, ITEMS, refined_texts)

        assert run_once() == run_once()

    def test_compression_hurts_under_stubs(self, refined_texts):
        """The demo signal: stub compression drops payload words, the
        mechanical judge sees them missing — mean delta is negative."""
        assert self.block["mean_delta"] < 0
        assert self.block["delta_ci95"]["high"] <= 0.0


class TestGuardrails:
    def test_evaluate_before_priming_raises(self, refined_texts):
        loop = make_loop(RecordingGateway())
        with pytest.raises(RuntimeError, match="prime"):
            loop.evaluate_level(0.2, ITEMS, refined_texts)

    def test_length_mismatch_raises(self, refined_texts):
        loop = make_loop(RecordingGateway())
        loop.prime_originals(ITEMS)
        with pytest.raises(RuntimeError, match="same order and length"):
            loop.evaluate_level(0.2, ITEMS, refined_texts[:-1])

    def test_unknown_item_raises(self, refined_texts):
        extra = GoldenItem(
            id="extra-1",
            load_type="rag",
            prompt="a late addition",
            key_points=("late",),
            source=SOURCE,
        )
        loop = make_loop(RecordingGateway())
        loop.prime_originals(ITEMS)
        with pytest.raises(RuntimeError, match="not primed"):
            loop.evaluate_level(
                0.2, ITEMS + [extra], refined_texts + [extra.prompt]
            )


class FlakyJudgeGateway(StubGatewayClient):
    """Real-world judge noise fixture: the first judge reply miscounts
    the key points (5 scores for 6), later replies are well-formed."""

    def __init__(self, bad_replies: int):
        self.bad_replies = bad_replies
        self.judge_calls = 0

    def chat(self, model: str, prompt: str):
        result = super().chat(model, prompt)
        if (
            split_judge_prompt(prompt) is not None
            and self.judge_calls < self.bad_replies
        ):
            self.judge_calls += 1
            bad = '{"scores": [1, 1, 0, 1, 0]}'  # one score short
            return type(result)(content=bad, model=result.model, usage=result.usage)
        return result


class TestJudgeMalformedReplyRetry:
    """A transient malformed judge reply must not kill a 1.8k-call run:
    bounded re-asking, fresh billing per attempt, hard fail only when the
    judge stays malformed."""

    def _loop(self, gateway):
        return TaskQualityLoop(
            gateway,
            answer_model="stub-answerer",
            judge_model="stub-judge",
            seed=0,
            n_resamples=100,
        )

    def test_malformed_then_valid_recovers(self, tmp_path):
        items = [GoldenItem(
            id="rag-1", load_type="rag", prompt="context with 6 facts",
            key_points=tuple(f"fact {i}" for i in range(6)), source=SOURCE,
        )]
        gateway = FlakyJudgeGateway(bad_replies=1)
        loop = self._loop(gateway)
        loop.prime_originals(items, golden_sha256="sha")
        block = loop.evaluate_level(0.2, items, ["compressed" for _ in items])
        row = block["items"][0]
        assert len(row["original_judge_scores"]) == 6
        assert len(row["compressed_judge_scores"]) == 6
        assert loop.call_stats()["judge_retries"] == 1

    def test_persistently_malformed_still_fails_loudly_with_item_id(self, tmp_path):
        items = [GoldenItem(
            id="rag-1", load_type="rag", prompt="context with 6 facts",
            key_points=tuple(f"fact {i}" for i in range(6)), source=SOURCE,
        )]
        gateway = FlakyJudgeGateway(bad_replies=99)
        loop = self._loop(gateway)
        from trillic.judge import JudgeError

        with pytest.raises(JudgeError, match="rag-1"):
            loop.prime_originals(items, golden_sha256="sha")
        # bounded: attempts == 1 + judge retries, not unbounded
        assert gateway.judge_calls == 5


class TestItemConcurrency:
    """Concurrent answer+judge must be behavior-identical to serial:
    same rows (same order), same stats, same journal content (order
    aside) — concurrency is an execution detail, never a results
    detail."""

    def _run(self, gateway, concurrency):
        loop = TaskQualityLoop(
            gateway,
            answer_model="stub-answerer",
            judge_model="stub-judge",
            seed=7,
            n_resamples=200,
            concurrency=concurrency,
        )
        loop.prime_originals(ITEMS, golden_sha256="sha")
        return loop.evaluate_level(0.2, ITEMS, [f"compressed {i.id}" for i in ITEMS])

    def test_concurrent_rows_match_serial_exactly(self):
        serial = self._run(StubGatewayClient(), 1)
        concurrent = self._run(StubGatewayClient(), 4)
        assert concurrent["items"] == serial["items"]
        assert (
            concurrent["delta_ci95"] == serial["delta_ci95"]
        )

    def test_journal_records_all_items_under_concurrency(self, tmp_path):
        from trillic.resume import CallJournal
        import json as _json

        journal = CallJournal(tmp_path / "ledger.jsonl")
        loop = TaskQualityLoop(
            StubGatewayClient(),
            answer_model="stub-answerer",
            judge_model="stub-judge",
            seed=7,
            n_resamples=200,
            concurrency=6,
            journal=journal,
        )
        loop.prime_originals(ITEMS, golden_sha256="sha")
        loop.evaluate_level(0.2, ITEMS, [f"compressed {i.id}" for i in ITEMS])
        journal.close()
        lines = [
            _json.loads(line)
            for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
            if line.strip()
        ]
        originals = sorted(r["item"] for r in lines if r["kind"] == "original")
        compressed = sorted(
            (r["item"], r["level"])
            for r in lines
            if r["kind"] == "compressed"
        )
        assert originals == sorted(i.id for i in ITEMS)
        assert compressed == sorted(
            (i.id, 0.2) for i in ITEMS
        )


class ContentFilterGateway(StubGatewayClient):
    """glm's content moderation (2026-09-12): a specific ANSWER sample
    trips "不安全或敏感内容"; a fresh answer sample passes. Input-side
    is fine — the same item judged cleanly on the other lane."""

    def __init__(self, bad_judge_replies: int):
        self.bad = bad_judge_replies
        self.seen = 0

    def chat(self, model: str, prompt: str):
        if "TRILLIC-JUDGE" in prompt and self.seen < self.bad:
            self.seen += 1
            raise GatewayError(
                'gateway returned 400: {"error":{"message":"系统检测到输入或'
                '生成内容可能包含不安全或敏感内容","code":"invalid_request"}}'
            )
        return super().chat(model, prompt)


class TestJudgeContentFilterRetry:
    def _loop(self, gateway):
        return TaskQualityLoop(
            gateway, answer_model="stub-answerer", judge_model="stub-judge",
            seed=3, n_resamples=100,
        )

    def test_filtered_answer_sample_reanswered(self):
        items = [GoldenItem(
            id="rag-nuke", load_type="rag", prompt="context about WMD reports",
            key_points=("proliferation concern",), source=SOURCE,
        )]
        gateway = ContentFilterGateway(bad_judge_replies=1)
        loop = self._loop(gateway)
        loop.prime_originals(items, golden_sha256="sha")
        assert loop.call_stats()["content_filter_retries"] == 1
        assert loop.call_stats()["answers_fresh"] == 2  # re-answered once

    def test_persistent_filter_raises_with_item_id(self):
        from trillic.task_quality import TaskQualityError

        items = [GoldenItem(
            id="rag-nuke", load_type="rag", prompt="context about WMD reports",
            key_points=("proliferation concern",), source=SOURCE,
        )]
        gateway = ContentFilterGateway(bad_judge_replies=99)
        loop = self._loop(gateway)
        with pytest.raises(TaskQualityError, match="rag-nuke"):
            loop.prime_originals(items, golden_sha256="sha")
        assert gateway.seen == 3  # bounded: 1 + 2 re-asks
