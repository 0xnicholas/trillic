"""Interrupt-resume ledger (issue #8): zero duplicate gateway calls.

Gateway calls are the billing surface (compression through the sidecar is
local compute). Resume = a new run that REPLAYS the prior run's recorded
answer/judge results instead of re-billing them:

- originals: reusable per item id when the golden set (sha256) and the
  quality pins (answer model, judge model, rubric sha) are unchanged —
  the payload is the golden prompt itself;
- compressed payloads: reusable per (level, item id) only when the freshly
  recompressed text is byte-identical to the recorded payload (content
  addressing — a drift in compression invalidates the recorded answer and
  the item is honestly re-billed).

Anything not reusable is computed fresh; both paths are counted in the
level block so the ledger is auditable ("已完成条目零重复网关调用").
"""

import json
from pathlib import Path

import pytest

from trillic.clients.gateway import StubGatewayClient
from trillic.golden import GoldenItem
from trillic.judge import rubric_sha256
from trillic.resume import ResumeError, load_replay
from trillic.task_quality import TaskQualityLoop

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RUN = None  # built per-test via the CLI helper below


def make_item(item_id: str, load_type: str = "rag") -> GoldenItem:
    source = {
        "dataset": "handwritten",
        "subset": "test",
        "license": "original",
        "split": "eval",
    }
    return GoldenItem(
        id=item_id,
        load_type=load_type,
        prompt=f"prompt for {item_id}: invoice 1001 dated 2026-07-19 total 84",
        key_points=(f"fact one for {item_id}", "invoice 1001", "total 84"),
        source=source,
    )


class CountingGateway(StubGatewayClient):
    """Stub gateway that counts every chat call (fresh billing)."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, model: str, prompt: str):
        self.calls += 1
        return super().chat(model, prompt)


def run_loop(gateway, items, levels, replay=None):
    loop = TaskQualityLoop(
        gateway,
        answer_model="stub-answerer",
        judge_model="stub-judge",
        seed=0,
        n_resamples=100,
        replay=replay,
    )
    loop.prime_originals(items, golden_sha256="g" * 64)
    blocks = [
        loop.evaluate_level(level, items, [f"compressed-{level}-{i.id}" for i in items])
        for level in levels
    ]
    return loop, blocks


def prior_metrics(gateway_calls_shape: str = "full") -> dict:
    """A plausible prior-run metrics.json for two items at one level."""
    gateway = CountingGateway()
    items = [make_item("a-1"), make_item("a-2")]
    _, blocks = run_loop(gateway, items, [0.2])
    block = blocks[0]
    return {
        "schema_version": 3,
        "run_id": "20260831T000000000000Z-prior-aaaaaaaa",
        "golden": {"sha256": "g" * 64, "item_count": 2},
        "task_quality": {
            "enabled": True,
            "answer_model": "stub-answerer",
            "judge_model": "stub-judge",
            "judge_rubric_sha256": rubric_sha256(),
        },
        "metrics": {
            "levels": [
                {
                    "aggressiveness": 0.2,
                    "items": [
                        {"id": "a-1", "compressed_text": "compressed-0.2-a-1"},
                        {"id": "a-2", "compressed_text": "compressed-0.2-a-2"},
                    ],
                    "aggregate": {"task_quality": block},
                }
            ]
        },
    }


class TestLoadReplay:
    def test_extracts_originals_and_compressed_payloads(self):
        replay = load_replay(prior_metrics())
        assert set(replay.original_ids) == {"a-1", "a-2"}
        assert set(replay.levels) == {0.2}
        # the recorded payload for the level is recoverable for comparison
        payload = replay.compressed_payload(0.2, "a-1")
        assert payload == "compressed-0.2-a-1"

    def test_rejects_metrics_without_task_quality(self):
        raw = prior_metrics()
        raw["task_quality"] = None
        with pytest.raises(ResumeError, match="task_quality"):
            load_replay(raw)

    def test_rejects_unknown_schema(self):
        raw = prior_metrics()
        raw["schema_version"] = 1
        with pytest.raises(ResumeError, match="schema"):
            load_replay(raw)

    def test_pins_exposed_for_match_check(self):
        replay = load_replay(prior_metrics())
        assert replay.answer_model == "stub-answerer"
        assert replay.judge_model == "stub-judge"
        assert replay.rubric_sha256 == rubric_sha256()
        assert replay.golden_sha256 == "g" * 64


class TestReplayReuse:
    def test_matching_pins_and_golden_replay_without_gateway_calls(self):
        items = [make_item("a-1"), make_item("a-2")]
        replay = load_replay(prior_metrics())
        replay.golden_sha256 = "g" * 64  # same exam
        gateway = CountingGateway()
        loop = TaskQualityLoop(
            gateway,
            answer_model="stub-answerer",
            judge_model="stub-judge",
            seed=0,
            n_resamples=100,
            replay=replay,
        )
        loop.prime_originals(items, golden_sha256="g" * 64)
        assert gateway.calls == 0  # originals fully reused
        block = loop.evaluate_level(
            0.2, items, [f"compressed-0.2-{i.id}" for i in items]
        )
        assert gateway.calls == 0  # identical payloads fully reused
        assert loop.call_stats() == {
            "answers_fresh": 0,
            "answers_reused": 4,  # 2 items x (original + compressed)
            "judges_fresh": 0,
            "judges_reused": 4,
        }
        assert all(row["source_original"] == "reused" for row in block["items"])
        assert all(row["source_compressed"] == "reused" for row in block["items"])

    def test_changed_payload_rebills_that_item_only(self):
        items = [make_item("a-1"), make_item("a-2")]
        replay = load_replay(prior_metrics())
        gateway = CountingGateway()
        loop = TaskQualityLoop(
            gateway,
            answer_model="stub-answerer",
            judge_model="stub-judge",
            seed=0,
            n_resamples=100,
            replay=replay,
        )
        loop.prime_originals(items, golden_sha256="g" * 64)
        payloads = ["compressed-CHANGED-a-1", "compressed-0.2-a-2"]
        block = loop.evaluate_level(0.2, items, payloads)
        # only a-1's compressed side is re-billed: 1 answer + 1 judge
        assert gateway.calls == 2
        assert loop.call_stats()["answers_fresh"] == 1
        assert loop.call_stats()["answers_reused"] == 3
        sources = {row["id"]: row["source_compressed"] for row in block["items"]}
        assert sources == {"a-1": "fresh", "a-2": "reused"}

    def test_new_level_is_all_fresh(self):
        items = [make_item("a-1"), make_item("a-2")]
        replay = load_replay(prior_metrics())
        gateway = CountingGateway()
        loop = TaskQualityLoop(
            gateway,
            answer_model="stub-answerer",
            judge_model="stub-judge",
            seed=0,
            n_resamples=100,
            replay=replay,
        )
        loop.prime_originals(items, golden_sha256="g" * 64)
        block = loop.evaluate_level(0.4, items, [f"c-0.4-{i.id}" for i in items])
        assert gateway.calls == 4  # 2 answers + 2 judges, all fresh
        assert loop.call_stats()["answers_fresh"] == 2

    def test_pin_mismatch_is_rejected(self):
        replay = load_replay(prior_metrics())
        with pytest.raises(ResumeError, match="answer_model"):
            TaskQualityLoop(
                CountingGateway(),
                answer_model="OTHER-answer-model",  # pin drift
                judge_model="stub-judge",
                seed=0,
                n_resamples=100,
                replay=replay,
            )

    def test_golden_mismatch_is_rejected(self):
        items = [make_item("a-1")]
        replay = load_replay(prior_metrics())
        replay.golden_sha256 = "different" * 8
        loop = TaskQualityLoop(
            CountingGateway(),
            answer_model="stub-answerer",
            judge_model="stub-judge",
            seed=0,
            n_resamples=100,
            replay=replay,
        )
        with pytest.raises(ResumeError, match="golden"):
            loop.prime_originals(items, golden_sha256="g" * 64)

    def test_rubric_mismatch_is_rejected(self):
        replay = load_replay(prior_metrics())
        replay.rubric_sha256 = "changed" * 8  # graded under a different rubric
        with pytest.raises(ResumeError, match="rubric"):
            TaskQualityLoop(
                CountingGateway(),
                answer_model="stub-answerer",
                judge_model="stub-judge",
                seed=0,
                n_resamples=100,
                replay=replay,
            )

    def test_unprimed_item_is_answered_fresh(self):
        items = [make_item("a-1"), make_item("NEW-3")]
        replay = load_replay(prior_metrics())
        gateway = CountingGateway()
        loop = TaskQualityLoop(
            gateway,
            answer_model="stub-answerer",
            judge_model="stub-judge",
            seed=0,
            n_resamples=100,
            replay=replay,
        )
        loop.prime_originals(items, golden_sha256="g" * 64)
        # NEW-3 wasn't in the prior run: 1 answer + 1 judge billed for it
        assert gateway.calls == 2


class TestFreshRunAccounting:
    def test_run_without_replay_counts_everything_fresh(self):
        gateway = CountingGateway()
        items = [make_item("a-1"), make_item("a-2")]
        loop, blocks = run_loop(gateway, items, [0.2])
        # originals (2 answers + 2 judges) + compressed (2 + 2) = 8 calls
        assert gateway.calls == 8
        assert loop.call_stats() == {
            "answers_fresh": 4,
            "answers_reused": 0,
            "judges_fresh": 4,
            "judges_reused": 0,
        }
