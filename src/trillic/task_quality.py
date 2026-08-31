"""Task-level quality loop (issue #7): answer → judge → delta → bootstrap.

For every golden item the loop answers the load-type downstream task TWICE
through the gateway — once with the original prompt, once with its
compressed form — and the judge grades BOTH answers against the SAME
golden key_points. Per item:

    score    = covered key points / key points   (binary per point)
    delta    = compressed_score - original_score (compression harm is negative)

Level aggregate: mean scores, mean delta, and a seeded paired bootstrap
over the per-prompt deltas (resample unit = prompt), reported as the 95%
CI whose lower bound the acceptance criterion reads.

Originals are level-independent: `prime_originals` answers and judges them
exactly once per run, so a sweep pays for originals one time only.
"""

from trillic.bootstrap import (
    DEFAULT_CONFIDENCE,
    METHOD,
    paired_bootstrap_ci,
)
from trillic.clients.gateway import GatewayClient
from trillic.golden import GoldenItem
from trillic.judge import judge_prompt, parse_judge_scores, rubric_sha256, score_of
from trillic.quality import rounded_mean
from trillic.resume import Replay
from trillic.tasks import task_prompt, task_templates_sha256

# Per-row rounding matches rounded_mean's 4-decimal metrics caliber.
_ROUND = 4


class TaskQualityError(RuntimeError):
    """The loop was driven out of order or with mismatched inputs."""


class TaskQualityLoop:
    """One gateway + pinned models + bootstrap parameters.

    Usage:
        loop = TaskQualityLoop(gateway, answer_model=..., judge_model=...,
                               seed=..., n_resamples=...)
        loop.prime_originals(items)                    # once per run
        block = loop.evaluate_level(level, items, refined_texts)  # per level
    """

    def __init__(
        self,
        gateway: GatewayClient,
        *,
        answer_model: str,
        judge_model: str,
        seed: int,
        n_resamples: int,
        replay: Replay | None = None,
    ) -> None:
        self._gateway = gateway
        self._answer_model = answer_model
        self._judge_model = judge_model
        self._seed = seed
        self._n_resamples = n_resamples
        # Interrupt-resume ledger (issue #8): when set, recorded gateway
        # results are replayed (never re-billed) under content-addressed
        # conditions; see trillic.resume.
        self._replay = replay
        if replay is not None:
            replay.check_pins(
                answer_model=answer_model,
                judge_model=judge_model,
                rubric_sha256=rubric_sha256(),
                task_templates_sha256=task_templates_sha256(),
            )
        self._originals: dict[str, dict] | None = None
        self._stats = {"answers_fresh": 0, "answers_reused": 0, "judges_fresh": 0, "judges_reused": 0}
        # Models the gateway REPORTS serving (the version of record for the
        # pin discipline — may differ from the requested pins when the
        # gateway aliases).
        self._served_answer_models: set[str] = set()
        self._served_judge_models: set[str] = set()

    def served_models(self) -> dict[str, list[str]]:
        """Sorted unique served model ids per role, for the run report."""
        return {
            "answer": sorted(self._served_answer_models),
            "judge": sorted(self._served_judge_models),
        }

    def call_stats(self) -> dict[str, int]:
        """Run-level gateway-call accounting (fresh = billed, reused = replayed)."""
        return dict(self._stats)

    def prime_originals(self, items: list[GoldenItem], golden_sha256: str | None = None) -> None:
        """Answer + judge the ORIGINAL prompts once (level-independent).

        With a replay ledger whose golden sha matches, originals are
        replayed from the prior run (zero gateway calls)."""
        if self._originals is not None:
            raise TaskQualityError("prime_originals must be called exactly once per run")
        if self._replay is not None:
            self._replay.check_golden(golden_sha256)
        self._originals = {}
        for item in items:
            recorded = self._replay.originals.get(item.id) if self._replay else None
            if recorded is not None:
                self._stats["answers_reused"] += 1
                self._stats["judges_reused"] += 1
                self._originals[item.id] = {
                    "answer": recorded["answer"],
                    "scores": recorded["scores"],
                    "score": score_of(recorded["scores"]),
                    "source": "reused",
                }
            else:
                self._originals[item.id] = {
                    **self._answer_and_judge(item, item.prompt),
                    "source": "fresh",
                }

    def original_score(self, item_id: str) -> float:
        if self._originals is None:
            raise TaskQualityError("prime_originals was never called")
        try:
            return self._originals[item_id]["score"]
        except KeyError:
            raise TaskQualityError(f"item {item_id!r} was never primed") from None

    def evaluate_level(
        self, level: float, items: list[GoldenItem], refined_texts: list[str]
    ) -> dict:
        """One sweep level: answer + judge the compressed payloads, pair
        with the primed originals, bootstrap the deltas."""
        if self._originals is None:
            raise TaskQualityError(
                "prime_originals must run before evaluate_level (originals "
                "are level-independent and paid for once)"
            )
        if len(items) != len(refined_texts):
            raise TaskQualityError(
                f"items and refined_texts must align one-to-one (same order "
                f"and length); got {len(items)} items vs {len(refined_texts)} texts"
            )

        rows = []
        for item, refined in zip(items, refined_texts):
            if item.id not in self._originals:
                raise TaskQualityError(
                    f"item {item.id!r} is not primed — evaluate_level must "
                    "see the same golden items prime_originals saw"
                )
            original = self._originals[item.id]
            recorded = (
                self._replay.compressed.get((level, item.id)) if self._replay else None
            )
            if recorded is not None and recorded["payload"] == refined:
                # content-addressed hit: same compression, same answer
                self._stats["answers_reused"] += 1
                self._stats["judges_reused"] += 1
                compressed = {
                    "answer": recorded["answer"],
                    "scores": recorded["scores"],
                    "score": score_of(recorded["scores"]),
                    "source": "reused",
                }
            else:
                compressed = {
                    **self._answer_and_judge(item, refined),
                    "source": "fresh",
                }
            rows.append(
                {
                    "id": item.id,
                    "load_type": item.load_type,
                    "source_original": original["source"],
                    "source_compressed": compressed["source"],
                    "original_answer": original["answer"],
                    "compressed_answer": compressed["answer"],
                    "original_judge_scores": original["scores"],
                    "compressed_judge_scores": compressed["scores"],
                    "original_score": round(original["score"], _ROUND),
                    "compressed_score": round(compressed["score"], _ROUND),
                    "delta": round(compressed["score"] - original["score"], _ROUND),
                }
            )

        deltas = [row["delta"] for row in rows]
        bootstrap = paired_bootstrap_ci(
            deltas,
            n_resamples=self._n_resamples,
            seed=self._seed,
            confidence=DEFAULT_CONFIDENCE,
        )
        return {
            "aggressiveness": level,
            "item_count": len(rows),
            "mean_original_score": rounded_mean([r["original_score"] for r in rows]),
            "mean_compressed_score": rounded_mean(
                [r["compressed_score"] for r in rows]
            ),
            "mean_delta": rounded_mean(deltas),
            "delta_ci95": {
                "low": round(bootstrap.ci_low, _ROUND),
                "high": round(bootstrap.ci_high, _ROUND),
            },
            "ci_lower_bound_not_negative": bootstrap.ci_lower_bound_not_negative,
            "items": rows,
        }

    def _answer_and_judge(self, item: GoldenItem, payload: str) -> dict:
        """Answer the load-type task for `payload`, judge against the
        golden key_points. One answer call + one judge call (both billed)."""
        key_points = list(item.key_points)
        answered = self._gateway.chat(
            self._answer_model, task_prompt(item.load_type, payload)
        )
        self._served_answer_models.add(answered.model)
        self._stats["answers_fresh"] += 1
        judged = self._gateway.chat(
            self._judge_model, judge_prompt(key_points, answered.content)
        )
        self._served_judge_models.add(judged.model)
        self._stats["judges_fresh"] += 1
        scores = parse_judge_scores(judged.content, len(key_points))
        return {"answer": answered.content, "scores": scores, "score": score_of(scores)}
