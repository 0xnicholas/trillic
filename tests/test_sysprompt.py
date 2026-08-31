"""System-prompt scenario families: seeded synthesis, seed-split discipline,
and manifest generation (issue #4).

Production-style system prompts have no public dataset (docs/data-strategy.md:
手写/合成). The split authority is the committed registry TOML: per family,
disjoint eval/train seed ranges. Golden entries are synthesized ONLY from
eval seeds; train seeds are reserved for phase-2 training synthesis that
shares this generator code (硬约束 4: 场景族/种子分流).
"""

import json
from pathlib import Path

import pytest

from trillic.golden import load_golden
from trillic.sysprompt import (
    CONSTRAINT_TYPES,
    SysPromptError,
    build_sysprompt_entries,
    build_sysprompt_manifest,
    load_families,
    synthesize_entry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILIES_TOML = REPO_ROOT / "eval" / "manifests" / "sysprompt_families.toml"
PILOT_GOLDEN = REPO_ROOT / "eval" / "golden" / "sysprompt_pilot.jsonl"
PILOT_MANIFEST = REPO_ROOT / "eval" / "manifests" / "sysprompt.json"


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def write_families(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "families.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestRegistryLoading:
    def test_committed_registry_loads_with_ten_families(self):
        families = load_families(FAMILIES_TOML)
        assert len(families) == 10
        names = [f["name"] for f in families]
        assert len(set(names)) == 10

    def test_every_family_declares_at_least_two_constraint_types(self):
        for family in load_families(FAMILIES_TOML):
            types = family["constraint_types"]
            assert set(types) <= set(CONSTRAINT_TYPES), family["name"]
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
            _MINIMAL_TOML.format(
                eval_range="[101, 110]", train_range="[105, 115]"
            ),
        )
        with pytest.raises(SysPromptError, match="overlap"):
            load_families(path)

    def test_unknown_family_name_is_refused(self, tmp_path):
        path = write_families(
            tmp_path,
            _MINIMAL_TOML.format(eval_range="[101,103]", train_range="[901,903]")
            .replace('name = "support_logistics"', 'name = "nonsense_family"'),
        )
        with pytest.raises(SysPromptError, match="no scenario template"):
            load_families(path)


class TestSynthesize:
    def test_same_family_and_seed_is_byte_deterministic(self):
        families = load_families(FAMILIES_TOML)
        first = synthesize_entry(families[0], 101)
        second = synthesize_entry(families[0], 101)
        assert first == second

    def test_different_seeds_change_prompt_and_id(self):
        family = load_families(FAMILIES_TOML)[0]
        first = synthesize_entry(family, 101)
        second = synthesize_entry(family, 102)
        assert first["id"] != second["id"]
        assert first["prompt"] != second["prompt"]

    def test_entry_shape_and_provenance(self):
        family = load_families(FAMILIES_TOML)[0]
        entry = synthesize_entry(family, 101)
        assert entry["id"] == f"sys-{family['name']}-s101"
        assert entry["load_type"] == "system_prompt"
        assert entry["prompt"].strip()
        assert len(entry["key_points"]) >= 3
        source = entry["source"]
        assert source["split"] == "eval"
        assert source["family"] == family["name"]
        assert source["seed"] == 101
        assert source["content_sha1"] and len(source["content_sha1"]) == 40
        assert source["dataset"] == "synthetic"

    def test_prompt_contains_system_rules_and_activating_message(self):
        """任务 = 激活该系统提示的用户消息: system body and a user turn that
        pushes against the constraints must both be present."""
        entry = synthesize_entry(load_families(FAMILIES_TOML)[0], 101)
        parts = entry["prompt"].split("\n\n")
        assert len(parts) >= 3  # multi-paragraph system rules + user message
        assert "Customer message:" in entry["prompt"]

    def test_key_points_reference_seeded_slot_values(self):
        """Behavioral constraints carry the seeded parameters (thresholds,
        caps, identifiers) so survival is checkable after compression."""
        entry = synthesize_entry(load_families(FAMILIES_TOML)[0], 101)
        prompt_tail = entry["prompt"]
        for point in entry["key_points"]:
            assert point.strip()
        # at least one point embeds a digit-bearing parameter of the family
        assert any(any(ch.isdigit() for ch in p) for p in entry["key_points"])
        # ticket reference from the user turn must survive in some point
        assert any("ticket reference" in p.lower() for p in entry["key_points"])

    def test_entries_pass_golden_schema(self, tmp_path):
        families = load_families(FAMILIES_TOML)
        entries = [
            synthesize_entry(f, f["eval_seed_range"][0]) for f in families[:3]
        ]
        items = load_golden(write_jsonl(tmp_path / "draft.jsonl", entries))
        assert [i.id for i in items] == [e["id"] for e in entries]

    def test_train_seed_is_refused_with_split_reason(self):
        family = load_families(FAMILIES_TOML)[0]
        train_seed = family["train_seed_range"][0]
        with pytest.raises(SysPromptError, match="TRAIN seed"):
            synthesize_entry(family, train_seed)

    def test_seed_outside_both_pools_is_refused(self):
        family = load_families(FAMILIES_TOML)[0]
        with pytest.raises(SysPromptError, match="outside"):
            synthesize_entry(family, 555)


class TestBuildEntries:
    def test_builds_ordered_unique_entries_from_eval_seeds(self):
        families = load_families(FAMILIES_TOML)
        plan = {f["name"]: [f["eval_seed_range"][0]] for f in families}
        entries = build_sysprompt_entries(families, plan)
        assert len(entries) == 10
        assert len({e["id"] for e in entries}) == 10
        for entry, family in zip(entries, families):
            assert entry["source"]["family"] == family["name"]

    def test_plan_with_unknown_family_is_refused(self):
        families = load_families(FAMILIES_TOML)
        with pytest.raises(SysPromptError, match="not in the registry"):
            build_sysprompt_entries(families, {"nope": [101]})

    def test_plan_with_train_seed_is_refused(self):
        families = load_families(FAMILIES_TOML)
        train_seed = families[0]["train_seed_range"][0]
        with pytest.raises(SysPromptError, match="TRAIN seed"):
            build_sysprompt_entries(families, {families[0]["name"]: [train_seed]})


class TestManifest:
    def test_manifest_without_golden_records_allocation(self, tmp_path):
        families = load_families(FAMILIES_TOML)
        manifest = build_sysprompt_manifest(families)
        assert manifest["schema_version"] == 1
        assert "split" in manifest["method"] or "seed" in manifest["method"]
        assert len(manifest["families"]) == 10
        first = manifest["families"][0]
        assert first["eval_seed_range"] == [101, 110]
        assert first["train_seed_range"] == [901, 910]
        assert first["license"].startswith("original")
        assert "pilot" not in manifest

    def test_manifest_deterministic(self):
        assert build_sysprompt_manifest(load_families(FAMILIES_TOML)) == (
            build_sysprompt_manifest(load_families(FAMILIES_TOML))
        )

    def test_manifest_with_golden_embeds_pilot_and_review_block(self, tmp_path):
        families = load_families(FAMILIES_TOML)
        plan = {f["name"]: [f["eval_seed_range"][0]] for f in families}
        entries = build_sysprompt_entries(families, plan)
        golden = write_jsonl(tmp_path / "pilot.jsonl", entries)
        manifest = build_sysprompt_manifest(families, golden_path=golden)
        assert manifest["pilot"]["entries"] == 10
        assert manifest["pilot"]["ids"] == [e["id"] for e in entries]
        assert manifest["pilot"]["seed_plan"] == {
            name: [seeds[0]] for name, seeds in plan.items()
        } or set(manifest["pilot"]["seed_plan"]) == set(plan)
        assert manifest["pilot"]["review"]["status"] in ("pending", "approved")

    def test_manifest_refuses_drifted_golden(self, tmp_path):
        families = load_families(FAMILIES_TOML)
        plan = {f["name"]: [f["eval_seed_range"][0]] for f in families}
        entries = build_sysprompt_entries(families, plan)
        entries[0]["prompt"] += " tampered tail"
        golden = write_jsonl(tmp_path / "drifted.jsonl", entries)
        with pytest.raises(SysPromptError, match="content_sha1"):
            build_sysprompt_manifest(families, golden_path=golden)

    def test_manifest_refuses_golden_from_train_seeds(self, tmp_path):
        families = load_families(FAMILIES_TOML)
        family = families[0]
        # forge an entry whose provenance claims a train seed
        entry = synthesize_entry(family, family["eval_seed_range"][0])
        forged = dict(entry)
        forged["source"] = {**entry["source"], "seed": family["train_seed_range"][0]}
        golden = write_jsonl(tmp_path / "forged.jsonl", [forged])
        with pytest.raises(SysPromptError, match="TRAIN seed"):
            build_sysprompt_manifest(families, golden_path=golden)


class TestCommittedPilot:
    """Freeze discipline: committed artifacts must equal regeneration."""

    def test_committed_pilot_validates_and_has_ten_entries(self):
        if not PILOT_GOLDEN.exists():
            pytest.skip("sysprompt_pilot.jsonl not built yet")
        items = load_golden(PILOT_GOLDEN)
        assert len(items) == 10
        assert all(i.load_type == "system_prompt" for i in items)

    def test_committed_pilot_matches_regeneration(self):
        if not PILOT_GOLDEN.exists():
            pytest.skip("sysprompt_pilot.jsonl not built yet")
        families = load_families(FAMILIES_TOML)
        plan = {f["name"]: [f["eval_seed_range"][0]] for f in families}
        regenerated = build_sysprompt_entries(families, plan)
        committed = [
            json.loads(line)
            for line in PILOT_GOLDEN.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert committed == regenerated

    def test_committed_manifest_matches_regeneration(self):
        if not PILOT_MANIFEST.exists():
            pytest.skip("sysprompt.json not built yet")
        regenerated = build_sysprompt_manifest(
            load_families(FAMILIES_TOML), golden_path=PILOT_GOLDEN
        )
        committed = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
        assert committed == regenerated

    def test_pilot_uses_only_eval_seeds_never_train(self):
        """场景族不跨 split, asserted against the committed artifacts."""
        if not PILOT_GOLDEN.exists():
            pytest.skip("sysprompt_pilot.jsonl not built yet")
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


_MINIMAL_TOML = """
schema_version = 1

[[family]]
name = "support_logistics"
description = "minimal"
constraint_types = ["refusal", "format"]
license = "original"
train_use = true
eval_seed_range = {eval_range}
train_seed_range = {train_range}
note = "test"
"""
