"""Training corpus v1 (issue #16): train-half extraction, schema with
query-aware reserved fields, manifest, and the validator.

Discipline encoded here (docs/data-strategy.md 硬约束):
  - entries come ONLY from the train half (eval half is golden's);
  - MeetingBank is hard-excluded anywhere in source (硬约束 1);
  - non-train-permitted subsets (qasper) never enter the corpus (硬约束 5);
  - zero overlap with golden is asserted by fingerprint (硬约束 4);
  - question/task are reserved query-aware fields, empty in v1 (硬约束 3).
"""

import json
from pathlib import Path

import pytest

from trillic.corpus import (
    CorpusError,
    build_training_entries,
    build_training_manifest,
    collect_corpus_errors,
)
from trillic.longbench import build_manifest, load_subset_rows
from trillic.splits import row_fingerprint, split_half_rows


def make_longbench_manifest(subsets_toml: Path, data_dir: Path) -> dict:
    return build_manifest(subsets_toml, data_dir)


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


@pytest.fixture
def lb_manifest(subsets_toml, data_dir) -> dict:
    return make_longbench_manifest(subsets_toml, data_dir)


@pytest.fixture
def built(tmp_path, lb_manifest, data_dir) -> tuple[Path, dict, list[dict]]:
    """Built corpus file + training manifest + entries, via the real builders."""
    entries = build_training_entries(lb_manifest, data_dir=data_dir)
    corpus_path = tmp_path / "corpus.jsonl"
    write_jsonl(corpus_path, entries)
    training_manifest = build_training_manifest(
        lb_manifest, corpus_path, repo_commit="deadbeef", repo_dirty=False
    )
    return corpus_path, training_manifest, entries


class TestBuildTrainingEntries:
    def test_emits_only_train_permitted_subsets(self, lb_manifest, data_dir):
        entries = build_training_entries(lb_manifest, data_dir=data_dir)
        subsets = {e["source"]["subset"] for e in entries}
        assert subsets == {"hotpotqa"}  # qasper is train_use=false in the fixture

    def test_train_half_only_and_full(self, lb_manifest, data_dir):
        entries = build_training_entries(lb_manifest, data_dir=data_dir)
        rows = load_subset_rows(data_dir / "hotpotqa_e.jsonl")
        eval_rows, train_rows = split_half_rows(rows)
        emitted_fps = {e["source"]["content_sha1"] for e in entries}
        assert emitted_fps == {row_fingerprint(r) for r in train_rows}
        assert not emitted_fps & {row_fingerprint(r) for r in eval_rows}
        assert len(entries) == 15

    def test_schema_v1_fields(self, lb_manifest, data_dir):
        entries = build_training_entries(lb_manifest, data_dir=data_dir)
        for e in entries:
            assert set(e) == {"id", "load_type", "prompt", "question", "task", "source"}
            assert e["load_type"] == "rag"
            assert e["question"] == "" and e["task"] == ""  # reserved, v1-empty
            assert e["source"]["split"] == "train"
            assert e["source"]["dataset"] == "LongBench"
            assert e["source"]["content_sha1"]
            assert e["id"].startswith("train-hotpotqa-")

    def test_prompt_is_production_shaped(self, lb_manifest, data_dir):
        entries = build_training_entries(lb_manifest, data_dir=data_dir)
        row = next(
            r
            for r in load_subset_rows(data_dir / "hotpotqa_e.jsonl")
            if row_fingerprint(r) == entries[0]["source"]["content_sha1"]
        )
        assert row["context"][:60] in entries[0]["prompt"]
        assert row["input"] in entries[0]["prompt"]

    def test_stable_order(self, lb_manifest, data_dir):
        a = build_training_entries(lb_manifest, data_dir=data_dir)
        b = build_training_entries(lb_manifest, data_dir=data_dir)
        assert [e["id"] for e in a] == [e["id"] for e in b]
        fps = [e["source"]["content_sha1"] for e in a]
        assert fps == sorted(fps)

    def test_no_train_permitted_subsets_is_an_error(self, tmp_path, data_dir):
        toml = tmp_path / "only_qasper.toml"
        toml.write_text(
            """
[[subset]]
name = "qasper"
file = "qasper_e.jsonl"
license = "CC BY-NC 4.0"
train_use = false
note = "eval-only"
"""
        )
        manifest = make_longbench_manifest(toml, data_dir)
        with pytest.raises(CorpusError, match="no train-permitted"):
            build_training_entries(manifest, data_dir=data_dir)

    def test_refuses_drifted_data(self, lb_manifest, data_dir):
        target = data_dir / "hotpotqa_e.jsonl"
        target.write_text(target.read_text() + json.dumps(
            {
                "input": "extra question?",
                "context": "extra context body. It mentions 1970 and 5 units.",
                "answers": ["extra answer 15"],
                "length": 100,
            }
        ) + "\n")
        with pytest.raises(CorpusError, match="regenerate"):
            build_training_entries(lb_manifest, data_dir=data_dir)


class TestBuildTrainingManifest:
    def test_records_corpus_identity_and_counts(self, built):
        corpus_path, training_manifest, entries = built
        import hashlib

        assert training_manifest["corpus"]["sha256"] == hashlib.sha256(
            corpus_path.read_bytes()
        ).hexdigest()
        assert training_manifest["corpus"]["entries"] == len(entries)
        assert training_manifest["corpus"]["load_types"] == {"rag": len(entries)}
        assert training_manifest["corpus"]["file"] == str(corpus_path)

    def test_subsets_carry_license_evidence_and_boundary(self, built):
        _, training_manifest, _ = built
        subsets = training_manifest["subsets"]
        assert [s["name"] for s in subsets] == ["hotpotqa"]
        hotpot = subsets[0]
        assert hotpot["train_use"] is True
        assert hotpot["train_boundary_fingerprint"]
        assert hotpot["entries"] == 15
        assert hotpot["license"]
        assert hotpot["train_use_evidence"]  # evidence links required (issue #16)

    def test_excluded_subsets_recorded_with_license_and_evidence(self, built):
        _, training_manifest, _ = built
        excluded = training_manifest["excluded_subsets"]
        assert [s["name"] for s in excluded] == ["qasper"]
        qasper = excluded[0]
        assert qasper["train_use"] is False
        assert qasper["license"]
        assert qasper["train_use_evidence"]
        assert qasper["reason"]

    def test_hard_exclusions_and_query_aware_policy_recorded(self, built):
        _, training_manifest, _ = built
        assert "meetingbank" in training_manifest["exclusions"]
        assert "customer_prompts" in training_manifest["exclusions"]
        qa = training_manifest["query_aware"]
        assert set(qa["fields"]) == {"question", "task"}
        assert "v1" in qa["v1_policy"]

    def test_generation_provenance_recorded(self, built):
        _, training_manifest, _ = built
        assert training_manifest["generated_by"]["repo_commit"] == "deadbeef"
        assert training_manifest["generated_by"]["repo_dirty"] is False
        assert training_manifest["split_method"]


def corrupt(entries: list[dict], entry_id: str, mutate) -> list[dict]:
    out = []
    for e in entries:
        e = json.loads(json.dumps(e))
        if e["id"] == entry_id:
            mutate(e)
        out.append(e)
    return out


class TestCollectCorpusErrors:
    def test_built_corpus_is_clean(self, built):
        corpus_path, training_manifest, entries = built
        assert (
            collect_corpus_errors(
                corpus_path,
                registry=training_manifest,
                golden_fingerprints={"0" * 40: "rag-x"},
            )
            == []
        )

    @pytest.mark.parametrize(
        "mutate, expect",
        [
            (lambda e: e.pop("question"), "question"),
            (lambda e: e.update(task=None), "task"),
            (lambda e: e.update(prompt="  "), "prompt"),
            (lambda e: e.update(load_type="code"), "load_type"),
            (lambda e: e["source"].update(split="eval"), "split must be 'train'"),
            (lambda e: e["source"].pop("content_sha1"), "content_sha1"),
            (
                lambda e: e["source"].update(dataset="MeetingBank"),
                "MeetingBank is excluded",
            ),
        ],
    )
    def test_bad_rows_reported_with_entry_id(self, built, mutate, expect):
        corpus_path, _, entries = built
        bad_id = entries[0]["id"]
        write_jsonl(corpus_path, corrupt(entries, bad_id, mutate))
        errors = collect_corpus_errors(corpus_path)
        assert any(bad_id in e and expect in e for e in errors), errors

    def test_bad_json_reports_line(self, built):
        corpus_path, _, _ = built
        corpus_path.write_text("{not json}\n", encoding="utf-8")
        errors = collect_corpus_errors(corpus_path)
        assert any("line 1" in e and "parse" in e.lower() for e in errors), errors

    def test_duplicate_id_and_fingerprint(self, built):
        corpus_path, _, entries = built
        dup = json.loads(json.dumps(entries[1]))
        dup["id"] = entries[0]["id"]
        write_jsonl(corpus_path, [entries[0], dup])
        errors = collect_corpus_errors(corpus_path)
        assert any("duplicate id" in e for e in errors), errors
        dup2 = json.loads(json.dumps(entries[0]))
        dup2["id"] = "train-hotpotqa-ffffffff"  # same fingerprint, new id
        write_jsonl(corpus_path, [entries[0], dup2])
        errors = collect_corpus_errors(corpus_path)
        assert any("duplicate content" in e for e in errors), errors

    def test_empty_file_rejected(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("\n\n", encoding="utf-8")
        assert collect_corpus_errors(empty)

    def test_missing_file(self, tmp_path):
        errors = collect_corpus_errors(tmp_path / "nope.jsonl")
        assert any("no such" in e for e in errors), errors


class TestRegistryChecks:
    def test_sha_drift_detected(self, built):
        corpus_path, training_manifest, entries = built
        training_manifest = json.loads(json.dumps(training_manifest))
        training_manifest["corpus"]["sha256"] = "0" * 64
        errors = collect_corpus_errors(corpus_path, registry=training_manifest)
        assert any("sha256" in e and "regenerate" in e for e in errors), errors

    def test_eval_half_leak_detected_by_boundary(self, built):
        corpus_path, training_manifest, entries = built
        training_manifest = json.loads(json.dumps(training_manifest))
        boundary = training_manifest["subsets"][0]["train_boundary_fingerprint"]
        leaked = corrupt(entries, entries[0]["id"], lambda e: e["source"].update(content_sha1="0" * 40))
        assert "0" * 40 < boundary
        write_jsonl(corpus_path, leaked)
        training_manifest["corpus"]["sha256"] = _sha(corpus_path)
        errors = collect_corpus_errors(corpus_path, registry=training_manifest)
        assert any("eval half" in e.lower() and entries[0]["id"] in e for e in errors), errors

    def test_excluded_subset_entry_rejected(self, built):
        corpus_path, training_manifest, entries = built
        training_manifest = json.loads(json.dumps(training_manifest))
        poisoned = corrupt(
            entries, entries[0]["id"], lambda e: e["source"].update(subset="qasper")
        )
        write_jsonl(corpus_path, poisoned)
        training_manifest["corpus"]["sha256"] = _sha(corpus_path)
        errors = collect_corpus_errors(corpus_path, registry=training_manifest)
        assert any("qasper" in e and "excluded" in e for e in errors), errors

    def test_unknown_subset_rejected(self, built):
        corpus_path, training_manifest, entries = built
        training_manifest = json.loads(json.dumps(training_manifest))
        poisoned = corrupt(
            entries, entries[0]["id"], lambda e: e["source"].update(subset="nonsense")
        )
        write_jsonl(corpus_path, poisoned)
        training_manifest["corpus"]["sha256"] = _sha(corpus_path)
        errors = collect_corpus_errors(corpus_path, registry=training_manifest)
        assert any("not in the training registry" in e for e in errors), errors

    def test_license_drift_detected(self, built):
        corpus_path, training_manifest, entries = built
        training_manifest = json.loads(json.dumps(training_manifest))
        poisoned = corrupt(
            entries, entries[0]["id"], lambda e: e["source"].update(license="MIT-ish")
        )
        write_jsonl(corpus_path, poisoned)
        training_manifest["corpus"]["sha256"] = _sha(corpus_path)
        errors = collect_corpus_errors(corpus_path, registry=training_manifest)
        assert any("license" in e and entries[0]["id"] in e for e in errors), errors

    def test_subset_count_mismatch_detected(self, built):
        corpus_path, training_manifest, entries = built
        training_manifest = json.loads(json.dumps(training_manifest))
        training_manifest["subsets"][0]["entries"] = 14
        training_manifest["corpus"]["sha256"] = _sha(corpus_path)
        errors = collect_corpus_errors(corpus_path, registry=training_manifest)
        assert any("14" in e and "15" in e for e in errors), errors


class TestGoldenZeroOverlap:
    def test_v1_empty_query_fields_enforced_when_registry_pins_it(self, built):
        """硬约束 3, v1 恒空: a registry with query_aware.v1_empty rejects
        non-empty question/task (a v2 registry drops the flag and the same
        validator accepts filled fields)."""
        corpus_path, training_manifest, entries = built
        rows = [json.loads(json.dumps(e)) for e in entries]
        rows[0]["question"] = "leaked question"
        write_jsonl(corpus_path, rows)
        errors = collect_corpus_errors(corpus_path, registry=training_manifest)
        assert any(
            "must be empty in a v1 corpus" in e and rows[0]["id"] in e for e in errors
        ), errors

        without_pin = json.loads(json.dumps(training_manifest))
        without_pin["query_aware"].pop("v1_empty")
        errors = collect_corpus_errors(corpus_path, registry=without_pin)
        assert not any("must be empty in a v1 corpus" in e for e in errors)

    def test_disjoint_fingerprints_pass(self, built, data_dir):
        corpus_path, _, _ = built
        rows = load_subset_rows(data_dir / "hotpotqa_e.jsonl")
        eval_rows, _ = split_half_rows(rows)
        golden_fps = {row_fingerprint(r): f"rag-hotpotqa-{row_fingerprint(r)[:8]}" for r in eval_rows}
        assert collect_corpus_errors(corpus_path, golden_fingerprints=golden_fps) == []

    def test_overlap_rejected_naming_both_ids(self, built):
        corpus_path, _, entries = built
        fp = entries[0]["source"]["content_sha1"]
        golden_id = f"rag-hotpotqa-{fp[:8]}"
        errors = collect_corpus_errors(corpus_path, golden_fingerprints={fp: golden_id})
        assert any(
            entries[0]["id"] in e and golden_id in e and "overlap" in e.lower()
            for e in errors
        ), errors


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
