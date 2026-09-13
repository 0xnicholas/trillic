"""Synthetic training layers (issue #17): train-side seeded synthesis with
family/seed diversion from golden, mix-ratio manifest, and the two-sided
zero-overlap validator.

Discipline encoded here (docs/data-strategy.md 硬约束 4 + docs/training-mix.md):
  - generator code is shared with golden, but seeds never cross the split:
    training draws ONLY from each family's train range, golden only from
    the eval range (both sides' guards raise on the other's seeds);
  - zero CONTENT overlap with golden on top of the seed-domain split —
    selection skips any seed whose rendered prompt duplicates a golden
    entry or an already-accepted train entry (small slot pools make
    different seeds collide on identical prompts);
  - counts land the #16 mix doc anchor (rag 30% / system_prompt 35% /
    dialogue 35%) and are recorded in the manifest;
  - zero gateway calls: pure deterministic template rendering.
"""

import json
from pathlib import Path

import pytest

from trillic.corpus import golden_fingerprints
from trillic.dialogue import (
    DialogueError,
    load_families as load_dialogue_families,
    synthesize_entry as synthesize_dialogue_eval,
    synthesize_training_entry as synthesize_dialogue_train,
)
from trillic.sysprompt import (
    SysPromptError,
    load_families as load_sysprompt_families,
    synthesize_entry as synthesize_sysprompt_eval,
    synthesize_training_entry as synthesize_sysprompt_train,
)
from trillic.synthetic import (
    SyntheticError,
    collect_synthetic_errors,
    select_train_seed_plan,
    write_synthetic_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYS_TOML = REPO_ROOT / "eval" / "manifests" / "sysprompt_families.toml"
DLG_TOML = REPO_ROOT / "eval" / "manifests" / "dialogue_families.toml"
GOLDEN_SYNTHETIC = [
    REPO_ROOT / "eval" / "golden" / "sysprompt_pilot.jsonl",
    REPO_ROOT / "eval" / "golden" / "sysprompt_scaled.jsonl",
    REPO_ROOT / "eval" / "golden" / "dialogue_pilot.jsonl",
    REPO_ROOT / "eval" / "golden" / "dialogue_scaled.jsonl",
]
_LOADERS = {
    SYS_TOML: load_sysprompt_families,
    DLG_TOML: load_dialogue_families,
}


def load(path: Path) -> list[dict]:
    return _LOADERS[path](path)


def repo_golden_fingerprints() -> dict[str, str]:
    return golden_fingerprints(GOLDEN_SYNTHETIC)


class TestTrainingSynthesis:
    """synthesize_training_entry: the train-side mirror of synthesize_entry."""

    @pytest.mark.parametrize(
        ("synthesize", "load", "prefix"),
        [
            (synthesize_sysprompt_train, load_sysprompt_families, "sys"),
            (synthesize_dialogue_train, load_dialogue_families, "dlg"),
        ],
    )
    def test_entry_shape_matches_training_schema_v1(
        self, synthesize, load, prefix
    ):
        family = load(SYS_TOML if prefix == "sys" else DLG_TOML)[0]
        entry = synthesize(family, 901)
        assert set(entry) == {"id", "load_type", "prompt", "question", "task", "source"}
        assert entry["id"] == f"{prefix}-{family['name']}-s901"
        assert entry["load_type"] == (
            "system_prompt" if prefix == "sys" else "dialogue"
        )
        assert entry["question"] == "" and entry["task"] == ""  # v1-empty (硬约束 3)
        source = entry["source"]
        assert source["dataset"] == "synthetic"
        assert source["subset"] == family["name"]
        assert source["split"] == "train"
        assert source["family"] == family["name"]
        assert source["seed"] == 901
        assert source["license"] == family["license"]

    def test_content_sha1_is_sha1_of_prompt(self):
        import hashlib

        family = load_sysprompt_families(SYS_TOML)[0]
        entry = synthesize_sysprompt_train(family, 905)
        assert entry["source"]["content_sha1"] == hashlib.sha1(
            entry["prompt"].encode("utf-8")
        ).hexdigest()

    @pytest.mark.parametrize(
        ("synthesize", "path"),
        [
            (synthesize_sysprompt_train, SYS_TOML),
            (synthesize_dialogue_train, DLG_TOML),
        ],
    )
    def test_deterministic(self, synthesize, path):
        family = load(path)[0]
        assert synthesize(family, 903) == synthesize(family, 903)

    @pytest.mark.parametrize(
        ("synthesize", "error", "path"),
        [
            (synthesize_sysprompt_train, SysPromptError, SYS_TOML),
            (synthesize_dialogue_train, DialogueError, DLG_TOML),
        ],
    )
    def test_refuses_eval_seeds(self, synthesize, error, path):
        family = load(path)[0]
        with pytest.raises(error, match="EVAL seed"):
            synthesize(family, 101)

    @pytest.mark.parametrize(
        ("synthesize", "error", "path"),
        [
            (synthesize_sysprompt_train, SysPromptError, SYS_TOML),
            (synthesize_dialogue_train, DialogueError, DLG_TOML),
        ],
    )
    def test_refuses_out_of_pool_seeds(self, synthesize, error, path):
        family = load(path)[0]
        with pytest.raises(error, match="outside both seed pools"):
            synthesize(family, 500)

    @pytest.mark.parametrize(
        ("synthesize", "error", "path"),
        [
            (synthesize_sysprompt_train, SysPromptError, SYS_TOML),
            (synthesize_dialogue_train, DialogueError, DLG_TOML),
        ],
    )
    def test_refuses_non_int_seed(self, synthesize, error, path):
        family = load(path)[0]
        with pytest.raises(error, match="seed must be an int"):
            synthesize(family, "901")

    @pytest.mark.parametrize(
        ("synthesize", "error", "path"),
        [
            (synthesize_sysprompt_eval, SysPromptError, SYS_TOML),
            (synthesize_dialogue_eval, DialogueError, DLG_TOML),
        ],
    )
    def test_golden_side_still_refuses_train_seeds(self, synthesize, error, path):
        """双方校验器互斥断言: the golden-side guard refuses train seeds."""
        family = load(path)[0]
        with pytest.raises(error, match="TRAIN seed"):
            synthesize(family, 901)


class TestSeedPlanSelection:
    def test_hits_targets_within_train_domain(self):
        plan = select_train_seed_plan(
            load_sysprompt_families(SYS_TOML),
            load_dialogue_families(DLG_TOML),
            targets={"system_prompt": 40, "dialogue": 40},
            golden_fingerprints=repo_golden_fingerprints(),
        )
        assert set(plan) == {"system_prompt", "dialogue"}
        for class_key, families in (
            ("system_prompt", load_sysprompt_families(SYS_TOML)),
            ("dialogue", load_dialogue_families(DLG_TOML)),
        ):
            assert sum(len(s) for s in plan[class_key].values()) == 40
            for family in families:
                lo, hi = family["train_seed_range"]
                for seed in plan[class_key][family["name"]]:
                    assert lo <= seed <= hi, (family["name"], seed)

    def test_deterministic(self):
        kwargs = dict(
            targets={"system_prompt": 12, "dialogue": 12},
            golden_fingerprints=repo_golden_fingerprints(),
        )
        first = select_train_seed_plan(
            load_sysprompt_families(SYS_TOML), load_dialogue_families(DLG_TOML), **kwargs
        )
        second = select_train_seed_plan(
            load_sysprompt_families(SYS_TOML), load_dialogue_families(DLG_TOML), **kwargs
        )
        assert first == second

    def test_content_distinct_and_clear_of_golden(self):
        """Accepted seeds render content-distinct prompts, none golden."""
        sys_families = load_sysprompt_families(SYS_TOML)
        dlg_families = load_dialogue_families(DLG_TOML)
        plan = select_train_seed_plan(
            sys_families,
            dlg_families,
            targets={"system_prompt": 60, "dialogue": 60},
            golden_fingerprints=repo_golden_fingerprints(),
        )
        golden = repo_golden_fingerprints()
        seen: set[str] = set()
        for class_key, families, synthesize in (
            ("system_prompt", sys_families, synthesize_sysprompt_train),
            ("dialogue", dlg_families, synthesize_dialogue_train),
        ):
            by_name = {f["name"]: f for f in families}
            for name, seeds in plan[class_key].items():
                for seed in seeds:
                    fingerprint = synthesize(by_name[name], seed)["source"]["content_sha1"]
                    assert fingerprint not in golden
                    assert fingerprint not in seen
                    seen.add(fingerprint)

    def test_skips_seeds_whose_content_is_golden(self):
        """A train seed rendering a golden-identical prompt is skipped."""
        sys_families = load_sysprompt_families(SYS_TOML)
        family = sys_families[0]
        collide = synthesize_sysprompt_train(family, 901)["source"]["content_sha1"]
        plan = select_train_seed_plan(
            sys_families,
            load_dialogue_families(DLG_TOML),
            targets={"system_prompt": 5, "dialogue": 5},
            golden_fingerprints={collide: "sys-golden-collision"},
        )
        assert 901 not in plan["system_prompt"][family["name"]]

    def test_unreachable_target_raises(self):
        with pytest.raises(SyntheticError, match="unreachable"):
            select_train_seed_plan(
                load_sysprompt_families(SYS_TOML),
                load_dialogue_families(DLG_TOML),
                targets={"system_prompt": 100_000, "dialogue": 5},
                golden_fingerprints=None,
            )

    def test_negative_target_raises(self):
        with pytest.raises(SyntheticError, match="non-negative"):
            select_train_seed_plan(
                load_sysprompt_families(SYS_TOML),
                load_dialogue_families(DLG_TOML),
                targets={"system_prompt": -1, "dialogue": 5},
            )

    def test_unknown_class_in_targets_raises(self):
        with pytest.raises(SyntheticError, match="unknown class"):
            select_train_seed_plan(
                load_sysprompt_families(SYS_TOML),
                load_dialogue_families(DLG_TOML),
                targets={"system_prompt": 5, "nonsense": 5},
            )

    def test_train_use_false_family_is_never_selected(self, tmp_path):
        toml = tmp_path / "restricted.toml"
        toml.write_text(
            SYS_TOML.read_text(encoding="utf-8").replace(
                'license = "original (authored in this repo, no third-party content)"\ntrain_use = true\neval_seed_range = [101, 110]\ntrain_seed_range = [901, 990]\nnote = "Archetype: customer tries to lure a refund promise for a lost parcel."',
                'license = "original (authored in this repo, no third-party content)"\ntrain_use = false\neval_seed_range = [101, 110]\ntrain_seed_range = [901, 990]\nnote = "restricted in this fixture"',
                1,
            ),
            encoding="utf-8",
        )
        families = load_sysprompt_families(toml)
        plan = select_train_seed_plan(
            families,
            load_dialogue_families(DLG_TOML),
            targets={"system_prompt": 10, "dialogue": 5},
        )
        assert "support_logistics" not in plan["system_prompt"]
        assert sum(len(s) for s in plan["system_prompt"].values()) == 10


@pytest.fixture
def built(tmp_path) -> tuple[Path, Path, dict]:
    """Small synthetic corpus + manifest built with the real builders."""
    corpus_path = tmp_path / "synthetic-train.jsonl"
    manifest_path = tmp_path / "synthetic-training.json"
    manifest = write_synthetic_artifacts(
        load_sysprompt_families(SYS_TOML),
        load_dialogue_families(DLG_TOML),
        targets={"system_prompt": 25, "dialogue": 25},
        golden_fingerprints=repo_golden_fingerprints(),
        corpus_path=corpus_path,
        manifest_path=manifest_path,
        rag_base=300,
        repo_commit="deadbeef",
        repo_dirty=False,
        record_file="training/corpus/synthetic-train-v1.jsonl",
    )
    return corpus_path, manifest_path, manifest


class TestArtifacts:
    def test_validates_clean_against_full_discipline(self, built):
        corpus_path, _manifest_path, manifest = built
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=manifest,
            golden_fingerprints=repo_golden_fingerprints(),
        )
        assert errors == []

    def test_entry_ordering_is_class_then_family_then_seed(self, built):
        corpus_path, _, _ = built
        rows = [
            json.loads(line)
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [r["load_type"] for r in rows] == ["system_prompt"] * 25 + ["dialogue"] * 25
        for class_key in ("system_prompt", "dialogue"):
            seeds = [
                (r["source"]["family"], r["source"]["seed"])
                for r in rows
                if r["load_type"] == class_key
            ]
            # per-family seeds ascend, and no (family, seed) repeats
            per_family: dict[str, list[int]] = {}
            for family, seed in seeds:
                per_family.setdefault(family, []).append(seed)
            for family, family_seeds in per_family.items():
                assert family_seeds == sorted(family_seeds)
                assert len(set(family_seeds)) == len(family_seeds)

    def test_manifest_pins_bytes_counts_and_mix(self, built, tmp_path):
        import hashlib

        corpus_path, manifest_path, manifest = built
        assert manifest["corpus"]["file"] == "training/corpus/synthetic-train-v1.jsonl"
        assert manifest["corpus"]["sha256"] == hashlib.sha256(
            corpus_path.read_bytes()
        ).hexdigest()
        assert manifest["corpus"]["entries"] == 50
        assert manifest["corpus"]["load_types"] == {"system_prompt": 25, "dialogue": 25}
        assert manifest["targets"] == {"system_prompt": 25, "dialogue": 25}
        # mix record: rag base 300 + achieved shares recomputed from actual counts
        mix = manifest["mix"]
        assert mix["rag_base"] == 300
        assert mix["ratio_target_pct"] == {
            "rag": 30.0,
            "system_prompt": 35.0,
            "dialogue": 35.0,
        }
        total = 300 + 50
        assert mix["achieved_pct"] == {
            "rag": 100 * 300 / total,
            "system_prompt": 100 * 25 / total,
            "dialogue": 100 * 25 / total,
        }
        # per-family seed plans recorded (family/seed diversion is auditable)
        for class_key, families in (
            ("system_prompt", load_sysprompt_families(SYS_TOML)),
            ("dialogue", load_dialogue_families(DLG_TOML)),
        ):
            recorded = {f["name"]: f for f in manifest["families"][class_key]}
            assert set(recorded) == {f["name"] for f in families}
            counted = sum(f["entries"] for f in manifest["families"][class_key])
            assert counted == 25
        assert manifest["query_aware"]["v1_empty"] is True
        assert manifest_path.read_text(encoding="utf-8").endswith("\n")

    def test_manifest_records_split_discipline_and_zero_gateway(self, built):
        _, _, manifest = built
        discipline = manifest["split_discipline"]
        assert "train" in discipline["seed_domain"]
        assert "golden" in discipline["zero_overlap"].lower()
        assert "seed" in manifest["selection_rule"]
        assert manifest["zero_gateway_calls"]

    def test_byte_deterministic(self, tmp_path):
        kwargs = dict(
            targets={"system_prompt": 15, "dialogue": 15},
            golden_fingerprints=repo_golden_fingerprints(),
            rag_base=300,
            repo_commit="cafe",
            repo_dirty=False,
            record_file="corpus.jsonl",
        )
        outputs = []
        for i in (1, 2):
            corpus_path = tmp_path / f"corpus-{i}.jsonl"
            manifest_path = tmp_path / f"manifest-{i}.json"
            write_synthetic_artifacts(
                load_sysprompt_families(SYS_TOML),
                load_dialogue_families(DLG_TOML),
                corpus_path=corpus_path,
                manifest_path=manifest_path,
                **kwargs,
            )
            outputs.append(
                (corpus_path.read_bytes(), manifest_path.read_bytes())
            )
        assert outputs[0] == outputs[1]

    def test_without_rag_base_manifest_records_targets_only(self, tmp_path):
        manifest = write_synthetic_artifacts(
            load_sysprompt_families(SYS_TOML),
            load_dialogue_families(DLG_TOML),
            targets={"system_prompt": 5, "dialogue": 5},
            golden_fingerprints=None,
            corpus_path=tmp_path / "c.jsonl",
            manifest_path=tmp_path / "m.json",
        )
        assert manifest["mix"]["rag_base"] is None
        assert "achieved_pct" not in manifest["mix"]


def rewrite_row(path: Path, row_index: int, mutate) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mutate(rows[row_index])
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


class TestValidator:
    def test_eval_seed_entry_is_rejected_with_precise_error(self, built):
        corpus_path, _, manifest = built
        first_sys_index = 0

        def mutate(row: dict) -> None:
            row["source"]["seed"] = 101
            row["id"] = f"sys-{row['source']['family']}-s101"

        rewrite_row(corpus_path, first_sys_index, mutate)
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=None,
        )
        assert any("EVAL seed" in e for e in errors), errors

    def test_out_of_pool_seed_is_rejected(self, built):
        corpus_path, _, _ = built

        def mutate(row: dict) -> None:
            row["source"]["seed"] = 500
            row["id"] = f"sys-{row['source']['family']}-s500"

        rewrite_row(corpus_path, 0, mutate)
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=None,
        )
        assert any("outside both seed pools" in e for e in errors), errors

    def test_prompt_drift_is_rejected(self, built):
        corpus_path, _, _ = built

        def mutate(row: dict) -> None:
            row["prompt"] += "\nhand-edited"

        rewrite_row(corpus_path, 3, mutate)
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=None,
        )
        assert any("no longer matches regeneration" in e for e in errors), errors

    def test_id_rule_violation_is_rejected(self, built):
        corpus_path, _, manifest = built

        def mutate(row: dict) -> None:
            row["id"] = "hand-written-id"

        rewrite_row(corpus_path, 1, mutate)
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=None,
        )
        assert any("id must be" in e for e in errors), errors

    def test_duplicate_content_is_rejected(self, built):
        corpus_path, _, _ = built
        rows = [
            json.loads(line)
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        clone = dict(rows[2])
        clone["id"] = clone["id"] + "-dup"
        rows.append(clone)
        corpus_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=None,
        )
        assert any("duplicate content fingerprint" in e for e in errors), errors

    def test_nonempty_question_is_rejected(self, built):
        corpus_path, _, manifest = built

        def mutate(row: dict) -> None:
            row["question"] = "premature query-aware fill"

        rewrite_row(corpus_path, 5, mutate)
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=manifest,
        )
        assert any("question/task" in e for e in errors), errors

    def test_meetingbank_is_hard_rejected(self, built):
        corpus_path, _, _ = built

        def mutate(row: dict) -> None:
            row["source"]["dataset"] = "meetingbank"

        rewrite_row(corpus_path, 7, mutate)
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=None,
        )
        assert any("MeetingBank" in e for e in errors), errors

    def test_load_type_class_mismatch_is_rejected(self, built):
        corpus_path, _, _ = built

        def mutate(row: dict) -> None:
            row["load_type"] = "dialogue"  # sys family on the dialogue side

        rewrite_row(corpus_path, 0, mutate)
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=None,
        )
        assert any("load_type" in e for e in errors), errors

    def test_manifest_sha_mismatch_is_rejected(self, built):
        corpus_path, _, manifest = built
        rewrite_row(corpus_path, 9, lambda row: row.update(prompt=row["prompt"] + "x"))
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=manifest,
        )
        assert any("sha256" in e for e in errors), errors

    def test_manifest_count_mismatch_is_rejected(self, built):
        corpus_path, _, manifest = built
        manifest["families"]["system_prompt"][0]["entries"] += 1
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=manifest,
        )
        assert any("entries" in e for e in errors), errors

    def test_golden_content_overlap_is_rejected(self, built):
        corpus_path, _, _ = built
        golden_row = json.loads(
            GOLDEN_SYNTHETIC[0].read_text(encoding="utf-8").splitlines()[0]
        )

        def mutate(row: dict) -> None:
            row["prompt"] = golden_row["prompt"]
            row["source"]["content_sha1"] = golden_row["source"]["content_sha1"]

        rewrite_row(corpus_path, 4, mutate)
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=None,
            golden_fingerprints={golden_row["source"]["content_sha1"]: golden_row["id"]},
        )
        assert any("overlap with golden" in e for e in errors), errors

    def test_train_use_false_family_entry_is_rejected(self, built, tmp_path):
        corpus_path, _, _ = built
        toml = tmp_path / "restricted.toml"
        original = SYS_TOML.read_text(encoding="utf-8")
        target_family = json.loads(
            corpus_path.read_text(encoding="utf-8").splitlines()[0]
        )["source"]["family"]
        replacement = f'name = "{target_family}"\ndescription'
        index = original.index(replacement)
        patched = (
            original[:index]
            + original[index:].replace("train_use = true", "train_use = false", 1)
        )
        toml.write_text(patched, encoding="utf-8")
        errors = collect_synthetic_errors(
            corpus_path,
            sysprompt_families=load_sysprompt_families(toml),
            dialogue_families=load_dialogue_families(DLG_TOML),
            manifest=None,
        )
        assert any("train_use=false" in e for e in errors), errors

    def test_missing_file_reports_error(self, tmp_path):
        errors = collect_synthetic_errors(
            tmp_path / "absent.jsonl",
            sysprompt_families=load_sysprompt_families(SYS_TOML),
            dialogue_families=load_dialogue_families(DLG_TOML),
        )
        assert any("no such file" in e for e in errors), errors
