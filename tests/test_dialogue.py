"""Multi-turn dialogue scenario families: seeded synthesis, seed-split
discipline, and manifest generation (issue #5).

Ported from the old repo's conversation generator
(0xnicholas/tokencamp compression/eval/conversation_gen.py — same author) and
extended. The split authority is the committed registry TOML: per family,
disjoint eval/train seed ranges. Golden entries are synthesized ONLY from
eval seeds; train seeds are reserved for phase-2 training synthesis that
shares this generator code (硬约束 4: 场景族/种子分流).

Dialogue-specific properties under test:
  - prompt = N-turn user/assistant history + a final user message that asks
    the assistant to recall history facts (issue #1: 合成历史 + 末轮);
  - key_points are HISTORY facts (names / dates / prior decisions), each
    verbatim-grounded in the history so survival is judgeable after
    compression (issue #5: 历史中必须存活的事实).
"""

import json
import re
import string
from pathlib import Path

import pytest

from trillic.golden import load_golden
from trillic.dialogue import (
    FACT_TYPES,
    DialogueError,
    build_dialogue_entries,
    build_dialogue_manifest,
    load_families,
    synthesize_entry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILIES_TOML = REPO_ROOT / "eval" / "manifests" / "dialogue_families.toml"
PILOT_GOLDEN = REPO_ROOT / "eval" / "golden" / "dialogue_pilot.jsonl"
PILOT_MANIFEST = REPO_ROOT / "eval" / "manifests" / "dialogue.json"


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def write_families(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "families.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _split_prompt(prompt: str) -> tuple[list[tuple[str, str]], str]:
    """Split a serialized dialogue prompt into (history turns, final user text).

    Serialization: turns start at a "user: " / "assistant: " line; turn
    content may span further lines (code blocks). The last turn is always
    the final user message.
    """
    turns: list[tuple[str, str]] = []
    current: tuple[str, str] | None = None
    for line in prompt.split("\n"):
        if line.startswith(("user: ", "assistant: ")):
            if current is not None:
                turns.append(current)
            role, _, content = line.partition(": ")
            current = (role, content)
        else:
            assert current is not None, f"content before first turn: {line[:60]!r}"
            current = (current[0], current[1] + "\n" + line)
    assert current is not None
    turns.append(current)
    assert turns[-1][0] == "user"
    return turns[:-1], turns[-1][1]


def _words(text: str) -> list[str]:
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()


def _longest_contiguous_overlap(point: str, history: str) -> int:
    """Longest contiguous word run of `point` appearing verbatim in `history`."""
    p, h = _words(point), _words(history)
    h_runs: set[str] = set()
    for i in range(len(h)):
        for j in range(i + 1, min(i + len(p), len(h)) + 1):
            h_runs.add(" ".join(h[i:j]))
    best = 0
    for i in range(len(p)):
        for j in range(len(p), i, -1):
            if " ".join(p[i:j]) in h_runs:
                best = max(best, j - i)
                break
    return best


def _digit_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_.\-/%]*\d[A-Za-z0-9_.\-/%]*", text)


class TestRegistryLoading:
    def test_committed_registry_loads_with_ten_families(self):
        families = load_families(FAMILIES_TOML)
        assert len(families) == 10
        names = [f["name"] for f in families]
        assert len(set(names)) == 10

    def test_every_family_declares_at_least_two_fact_types(self):
        for family in load_families(FAMILIES_TOML):
            types = family["fact_types"]
            assert set(types) <= set(FACT_TYPES), family["name"]
            assert len(types) >= 2, family["name"]

    def test_eval_and_train_seed_ranges_are_disjoint(self):
        """Split discipline: no seed may ever sit on both sides (硬约束 4)."""
        for family in load_families(FAMILIES_TOML):
            eval_seeds = set(range(*_inclusive(family["eval_seed_range"])))
            train_seeds = set(range(*_inclusive(family["train_seed_range"])))
            assert eval_seeds and train_seeds, family["name"]
            assert not eval_seeds & train_seeds, family["name"]

    def test_overlapping_ranges_are_refused(self, tmp_path):
        path = write_families(
            tmp_path,
            _MINIMAL_TOML.format(eval_range="[101, 110]", train_range="[105, 115]"),
        )
        with pytest.raises(DialogueError, match="overlap"):
            load_families(path)

    def test_unknown_family_name_is_refused(self, tmp_path):
        path = write_families(
            tmp_path,
            _MINIMAL_TOML.format(eval_range="[101,103]", train_range="[901,903]")
            .replace('name = "support_ticket"', 'name = "nonsense_family"'),
        )
        with pytest.raises(DialogueError, match="no scenario template"):
            load_families(path)


class TestSynthesize:
    def test_same_family_and_seed_is_byte_deterministic(self):
        families = load_families(FAMILIES_TOML)
        assert synthesize_entry(families[0], 101) == synthesize_entry(families[0], 101)

    def test_different_seeds_change_prompt_and_id(self):
        family = load_families(FAMILIES_TOML)[0]
        first = synthesize_entry(family, 101)
        second = synthesize_entry(family, 102)
        assert first["id"] != second["id"]
        assert first["prompt"] != second["prompt"]

    def test_entry_shape_and_provenance(self):
        family = load_families(FAMILIES_TOML)[0]
        entry = synthesize_entry(family, 101)
        assert entry["id"] == f"dlg-{family['name']}-s101"
        assert entry["load_type"] == "dialogue"
        assert entry["prompt"].strip()
        assert len(entry["key_points"]) >= 4
        source = entry["source"]
        assert source["split"] == "eval"
        assert source["family"] == family["name"]
        assert source["seed"] == 101
        assert source["content_sha1"] and len(source["content_sha1"]) == 40
        assert source["dataset"] == "synthetic"

    def test_prompt_is_history_plus_final_user_message(self):
        """结构 = N 轮历史 + 末轮用户消息: roles alternate, the history has
        many turns, and the last turn is a user message (not assistant)."""
        for family in load_families(FAMILIES_TOML):
            for seed in (101, 102):
                entry = synthesize_entry(family, seed)
                history, final = _split_prompt(entry["prompt"])
                assert len(history) >= 8, family["name"]
                assert len(final) > 40, family["name"]  # a real recall request
                roles = [role for role, _ in history]
                assert roles[0] == "user"
                for i, role in enumerate(roles):
                    assert role == ("user" if i % 2 == 0 else "assistant"), (
                        f"{family['name']}: history roles must alternate"
                    )

    def test_key_points_grounded_in_history(self):
        """历史事实型: every key point must quote the history verbatim
        (>=4 contiguous shared words), so survival after compression is
        mechanically judgeable."""
        for family in load_families(FAMILIES_TOML):
            for seed in (101, 102):
                entry = synthesize_entry(family, seed)
                history, _ = _split_prompt(entry["prompt"])
                history_text = "\n".join(content for _, content in history)
                for point in entry["key_points"]:
                    overlap = _longest_contiguous_overlap(point, history_text)
                    assert overlap >= 4, (
                        f"{entry['id']}: key point not grounded in history "
                        f"({overlap}-word overlap): {point!r}"
                    )

    def test_key_points_carry_survivable_values(self):
        """名称/日期/先前决定: at least half the key points embed a
        digit-bearing value (dates, IDs, amounts, versions) — exactly the
        kind of fact token-level compression tends to drop."""
        for family in load_families(FAMILIES_TOML):
            entry = synthesize_entry(family, 101)
            with_digits = [p for p in entry["key_points"] if _digit_tokens(p)]
            assert len(with_digits) >= max(2, len(entry["key_points"]) // 2), (
                f"{family['name']}: key points lack digit-bearing facts"
            )

    def test_key_points_never_quote_final_message_only(self):
        """Facts live in the HISTORY; the final user message may reference
        them but a fact grounded only in the final turn would not test
        history compression. Grounding test above already enforces this
        against history; here we assert the final turn stays a question."""
        for family in load_families(FAMILIES_TOML):
            entry = synthesize_entry(family, 101)
            _, final = _split_prompt(entry["prompt"])
            assert "?" in final, family["name"]

    def test_entries_pass_golden_schema(self, tmp_path):
        families = load_families(FAMILIES_TOML)
        entries = [synthesize_entry(f, f["eval_seed_range"][0]) for f in families[:3]]
        items = load_golden(write_jsonl(tmp_path / "draft.jsonl", entries))
        assert [i.id for i in items] == [e["id"] for e in entries]

    def test_train_seed_is_refused_with_split_reason(self):
        family = load_families(FAMILIES_TOML)[0]
        with pytest.raises(DialogueError, match="TRAIN seed"):
            synthesize_entry(family, family["train_seed_range"][0])

    def test_seed_outside_both_pools_is_refused(self):
        family = load_families(FAMILIES_TOML)[0]
        with pytest.raises(DialogueError, match="outside"):
            synthesize_entry(family, 555)

    def test_all_ten_families_render(self):
        """Every template renders without format errors (brace escapes,
        missing slots) at several seeds; no un-substituted {placeholder}
        leaks into the output."""
        families = load_families(FAMILIES_TOML)
        for family in families:
            for seed in (101, 105, 110):
                entry = synthesize_entry(family, seed)
                assert not re.search(r"\{[a-z_]+\}", entry["prompt"]), family["name"]
                for point in entry["key_points"]:
                    assert not re.search(r"\{[a-z_]+\}", point), family["name"]


class TestBuildEntries:
    def test_builds_ordered_unique_entries_from_eval_seeds(self):
        families = load_families(FAMILIES_TOML)
        plan = {f["name"]: [f["eval_seed_range"][0]] for f in families}
        entries = build_dialogue_entries(families, plan)
        assert len(entries) == 10
        assert len({e["id"] for e in entries}) == 10
        for entry, family in zip(entries, families):
            assert entry["source"]["family"] == family["name"]

    def test_plan_with_unknown_family_is_refused(self):
        families = load_families(FAMILIES_TOML)
        with pytest.raises(DialogueError, match="not in the registry"):
            build_dialogue_entries(families, {"nope": [101]})

    def test_plan_with_train_seed_is_refused(self):
        families = load_families(FAMILIES_TOML)
        train_seed = families[0]["train_seed_range"][0]
        with pytest.raises(DialogueError, match="TRAIN seed"):
            build_dialogue_entries(families, {families[0]["name"]: [train_seed]})

    def test_build_refuses_repeated_seed_in_plan(self):
        families = load_families(FAMILIES_TOML)
        name = families[0]["name"]
        seed = families[0]["eval_seed_range"][0]
        with pytest.raises(DialogueError, match="repeats a seed"):
            build_dialogue_entries(families, {name: [seed, seed]})


class TestManifest:
    def test_manifest_without_golden_records_allocation(self, tmp_path):
        manifest = build_dialogue_manifest(load_families(FAMILIES_TOML))
        assert manifest["schema_version"] == 1
        assert "split" in manifest["method"] or "seed" in manifest["method"]
        assert len(manifest["families"]) == 10
        first = manifest["families"][0]
        assert first["eval_seed_range"] == [101, 110]
        assert first["train_seed_range"] == [901, 990]  # widened for issue #17 training synthesis (headroom, not a quota)
        assert "pilot" not in manifest

    def test_manifest_deterministic(self):
        assert build_dialogue_manifest(load_families(FAMILIES_TOML)) == (
            build_dialogue_manifest(load_families(FAMILIES_TOML))
        )

    def test_manifest_with_golden_embeds_pilot_and_review_block(self, tmp_path):
        families = load_families(FAMILIES_TOML)
        plan = {f["name"]: [f["eval_seed_range"][0]] for f in families}
        entries = build_dialogue_entries(families, plan)
        golden = write_jsonl(tmp_path / "pilot.jsonl", entries)
        manifest = build_dialogue_manifest(families, golden_path=golden)
        assert manifest["pilot"]["entries"] == 10
        assert manifest["pilot"]["ids"] == [e["id"] for e in entries]
        assert manifest["pilot"]["seed_plan"] == {
            name: [seeds[0]] for name, seeds in plan.items()
        }
        assert manifest["pilot"]["review"]["status"] in ("pending", "approved")

    def test_manifest_review_override_flows_through(self, tmp_path):
        families = load_families(FAMILIES_TOML)
        plan = {f["name"]: [f["eval_seed_range"][0]] for f in families}
        golden = write_jsonl(
            tmp_path / "pilot.jsonl", build_dialogue_entries(families, plan)
        )
        approved = build_dialogue_manifest(
            families,
            golden_path=golden,
            review={"status": "approved", "reviewer": "nicholas", "notes": "lgtm"},
        )
        assert approved["pilot"]["review"]["status"] == "approved"
        assert approved["pilot"]["review"]["reviewer"] == "nicholas"
        # everything except the owner-owned review block still matches the
        # default regeneration — an approval never masks a drift
        default = build_dialogue_manifest(families, golden_path=golden)
        assert _without_review(approved) == _without_review(default)

    def test_manifest_refuses_drifted_golden(self, tmp_path):
        families = load_families(FAMILIES_TOML)
        plan = {f["name"]: [f["eval_seed_range"][0]] for f in families}
        entries = build_dialogue_entries(families, plan)
        entries[0]["prompt"] += " tampered tail"
        golden = write_jsonl(tmp_path / "drifted.jsonl", entries)
        with pytest.raises(DialogueError, match="content_sha1"):
            build_dialogue_manifest(families, golden_path=golden)

    def test_manifest_refuses_duplicate_pilot_rows(self, tmp_path):
        families = load_families(FAMILIES_TOML)
        plan = {f["name"]: [f["eval_seed_range"][0]] for f in families}
        entries = build_dialogue_entries(families, plan)
        entries.append(dict(entries[0]))  # same (family, seed) twice
        golden = write_jsonl(tmp_path / "dup.jsonl", entries)
        with pytest.raises(DialogueError, match="duplicate row"):
            build_dialogue_manifest(families, golden_path=golden)

    def test_manifest_refuses_golden_from_train_seeds(self, tmp_path):
        families = load_families(FAMILIES_TOML)
        family = families[0]
        # forge an entry whose provenance claims a train seed
        entry = synthesize_entry(family, family["eval_seed_range"][0])
        forged = dict(entry)
        forged["source"] = {**entry["source"], "seed": family["train_seed_range"][0]}
        golden = write_jsonl(tmp_path / "forged.jsonl", [forged])
        with pytest.raises(DialogueError, match="TRAIN seed"):
            build_dialogue_manifest(families, golden_path=golden)


class TestCommittedPilot:
    """Freeze discipline: committed artifacts must equal regeneration."""

    def test_committed_pilot_validates_and_has_ten_entries(self):
        if not PILOT_GOLDEN.exists():
            pytest.skip("dialogue_pilot.jsonl not built yet")
        items = load_golden(PILOT_GOLDEN)
        assert len(items) == 10
        assert all(i.load_type == "dialogue" for i in items)

    def test_committed_pilot_matches_regeneration(self):
        if not PILOT_GOLDEN.exists():
            pytest.skip("dialogue_pilot.jsonl not built yet")
        families = load_families(FAMILIES_TOML)
        plan = {f["name"]: [f["eval_seed_range"][0]] for f in families}
        regenerated = build_dialogue_entries(families, plan)
        committed = [
            json.loads(line)
            for line in PILOT_GOLDEN.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert committed == regenerated

    def test_committed_manifest_matches_regeneration(self):
        """Generator output must match regeneration exactly; the pilot review
        block is owner-owned sign-off metadata and may differ (pending →
        approved) without constituting drift."""
        if not PILOT_MANIFEST.exists():
            pytest.skip("dialogue.json not built yet")
        regenerated = build_dialogue_manifest(
            load_families(FAMILIES_TOML), golden_path=PILOT_GOLDEN
        )
        committed = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
        assert _without_review(committed) == _without_review(regenerated)
        assert committed["pilot"]["review"]["status"] in ("pending", "approved")

    def test_pilot_uses_only_eval_seeds_never_train(self):
        """场景族不跨 split, asserted against the committed artifacts."""
        if not PILOT_GOLDEN.exists():
            pytest.skip("dialogue_pilot.jsonl not built yet")
        by_name = {f["name"]: f for f in load_families(FAMILIES_TOML)}
        for line in PILOT_GOLDEN.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            family = by_name[entry["source"]["family"]]
            seed = entry["source"]["seed"]
            assert seed in range(*_inclusive(family["eval_seed_range"]))
            assert seed not in range(*_inclusive(family["train_seed_range"]))


def _inclusive(bounds: list[int]) -> tuple[int, int]:
    """Registry ranges are inclusive [lo, hi]; range() needs hi + 1."""
    lo, hi = bounds
    return (lo, hi + 1)


def _without_review(manifest: dict) -> dict:
    """Manifest minus the owner-owned review block (drift comparison view)."""
    stripped = json.loads(json.dumps(manifest))  # deep copy
    stripped.get("pilot", {}).pop("review", None)
    return stripped


_MINIMAL_TOML = """
schema_version = 1

[[family]]
name = "support_ticket"
description = "minimal"
fact_types = ["name", "date", "decision"]
license = "original"
train_use = true
eval_seed_range = {eval_range}
train_seed_range = {train_range}
note = "test"
"""
