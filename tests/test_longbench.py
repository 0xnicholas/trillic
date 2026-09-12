"""LongBench tooling: subset loading, manifest generation, seeded RAG golden
drafting. All deterministic given (data, manifest, seed) — no wall clock, no
file-order dependence.
"""

import json
from pathlib import Path

import pytest

from trillic.golden import load_golden
from trillic.longbench import (
    build_manifest,
    build_rag_entries,
    derive_key_points,
    load_subset_rows,
    render_prompt,
)
from trillic.splits import row_fingerprint, split_half_rows


class TestLoadSubsetRows:
    def test_loads_rows_and_skips_blanks(self, data_dir):
        target = data_dir / "blanks.jsonl"
        target.write_text(
            (data_dir / "qasper_e.jsonl").read_text().splitlines()[0] + "\n\n"
        )
        rows = load_subset_rows(target)
        assert len(rows) == 1
        assert {"input", "context", "answers"} <= set(rows[0])

    def test_malformed_row_raises_with_line(self, data_dir):
        target = data_dir / "bad.jsonl"
        target.write_text("{not json}\n")
        with pytest.raises(ValueError, match="line 1"):
            load_subset_rows(target)


class TestRenderPrompt:
    def test_each_subset_embeds_context_and_question(self):
        for subset in ("qasper", "hotpotqa"):
            prompt = render_prompt(subset, context="CTX", question="Q")
            assert "CTX" in prompt and "Q" in prompt

    def test_gov_report_is_summarization_no_question(self):
        prompt = render_prompt("gov_report", context="CTX", question="")
        assert "CTX" in prompt and "Summarize" in prompt

    def test_unknown_subset_rejected(self):
        with pytest.raises(ValueError, match="no prompt template"):
            render_prompt("nonsense", context="c", question="q")


class TestDeriveKeyPoints:
    def test_short_answer_becomes_single_point(self):
        assert derive_key_points(["Yanzhou"]) == ["Yanzhou"]

    def test_first_answer_wins_others_ignored(self):
        assert derive_key_points(["", "  ", "the real one"]) == ["the real one"]

    def test_long_answer_keeps_numeric_sentences_plus_first(self):
        answer = (
            "Revenue grew strongly across regions. "
            "The board met twice and agreed on strategy. "
            "Operating margin reached 32% in Q2. "
            "Headcount rose to 1,200 by year end."
        )
        points = derive_key_points([answer])
        assert points[0].startswith("Revenue grew strongly")
        assert any("32%" in p for p in points)
        assert any("1,200" in p for p in points)
        assert not any("board met" in p for p in points)

    def test_points_are_capped_in_count_but_never_truncated(self):
        answer = " ".join(f"Sentence number {i} with digit {i}." for i in range(30))
        points = derive_key_points([answer])
        assert len(points) <= 6
        assert all(p.endswith(".") for p in points)  # whole sentences only


class TestBuildManifest:
    def test_manifest_counts_halves_and_licenses(self, subsets_toml, data_dir):
        manifest = build_manifest(subsets_toml, data_dir)
        assert manifest["split_method"].startswith("sha1")
        by_name = {s["name"]: s for s in manifest["subsets"]}
        assert by_name["qasper"]["entries"] == 30
        assert by_name["qasper"]["eval_half"] == 15
        assert by_name["qasper"]["train_half"] == 15
        assert by_name["qasper"]["train_use"] is False
        assert by_name["qasper"]["license"].startswith("CC BY-NC")
        assert by_name["hotpotqa"]["train_use"] is True
        for s in manifest["subsets"]:
            assert len(s["file_sha256"]) == 64
            assert s["eval_half"] + s["train_half"] == s["entries"]

    def test_manifest_is_deterministic(self, subsets_toml, data_dir):
        assert build_manifest(subsets_toml, data_dir) == build_manifest(subsets_toml, data_dir)


class TestBuildRagEntries:
    COUNTS = {"qasper": 3, "hotpotqa": 2}

    def test_output_passes_golden_validation(self, subsets_toml, data_dir):
        manifest = build_manifest(subsets_toml, data_dir)
        entries = build_rag_entries(
            manifest, data_dir=data_dir, counts=self.COUNTS, seed=7, min_tokens=0, max_tokens=10**6
        )
        out = data_dir.parent / "draft.jsonl"
        out.write_text("".join(json.dumps(e) + "\n" for e in entries))
        items = load_golden(out)
        assert len(items) == 5
        assert {i.source["subset"] for i in items} == {"qasper", "hotpotqa"}
        assert all(i.source["split"] == "eval" for i in items)
        assert all(i.load_type == "rag" for i in items)
        assert all(i.key_points for i in items)

    def test_selection_draws_only_from_eval_half(self, subsets_toml, data_dir):
        """Acceptance criterion: same-source samples never cross splits."""
        manifest = build_manifest(subsets_toml, data_dir)
        entries = build_rag_entries(
            manifest, data_dir=data_dir, counts=self.COUNTS, seed=7, min_tokens=0, max_tokens=10**6
        )
        for subset in self.COUNTS:
            rows = load_subset_rows(data_dir / f"{subset}_e.jsonl")
            eval_rows, train_rows = split_half_rows(rows)
            train_fps = {row_fingerprint(r) for r in train_rows}
            eval_fps = {row_fingerprint(r) for r in eval_rows}
            chosen = [e["source"]["content_sha1"] for e in entries if e["source"]["subset"] == subset]
            assert set(chosen) <= eval_fps
            assert not set(chosen) & train_fps

    def test_deterministic_given_seed(self, subsets_toml, data_dir):
        manifest = build_manifest(subsets_toml, data_dir)
        kwargs = dict(data_dir=data_dir, counts=self.COUNTS, min_tokens=0, max_tokens=10**6)
        assert build_rag_entries(manifest, seed=7, **kwargs) == build_rag_entries(manifest, seed=7, **kwargs)
        assert build_rag_entries(manifest, seed=7, **kwargs) != build_rag_entries(manifest, seed=8, **kwargs)

    def test_refuses_more_than_eval_half(self, subsets_toml, data_dir):
        manifest = build_manifest(subsets_toml, data_dir)
        with pytest.raises(ValueError, match="qasper.*eval-half"):
            build_rag_entries(
                manifest, data_dir=data_dir, counts={"qasper": 16}, seed=7,
                min_tokens=0, max_tokens=10**6,
            )

    def test_consistency_guard_against_manifest_drift(self, subsets_toml, data_dir):
        """If the data file changes after the manifest was generated, the
        builder must refuse rather than silently mix splits."""
        manifest = build_manifest(subsets_toml, data_dir)
        with open(data_dir / "qasper_e.jsonl", "a") as f:
            f.write(json.dumps({"input": "new q", "context": "new ctx", "answers": ["a"]}) + "\n")
        with pytest.raises(ValueError, match="qasper.*manifest"):
            build_rag_entries(
                manifest, data_dir=data_dir, counts={"qasper": 2}, seed=7,
                min_tokens=0, max_tokens=10**6,
            )

    def test_consistency_guard_catches_same_count_content_drift(self, subsets_toml, data_dir):
        """Entry-count checks alone would pass; the sha256 must catch it."""
        manifest = build_manifest(subsets_toml, data_dir)
        # rewrite the LAST row in place — same count, different bytes
        lines = (data_dir / "qasper_e.jsonl").read_text().splitlines()
        row = json.loads(lines[-1])
        row["answers"] = ["mutated answer"]
        lines[-1] = json.dumps(row)
        (data_dir / "qasper_e.jsonl").write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="sha256"):
            build_rag_entries(
                manifest, data_dir=data_dir, counts={"qasper": 2}, seed=7,
                min_tokens=0, max_tokens=10**6,
            )

    def test_manifest_handles_single_row_subset(self, tmp_path, subsets_toml, data_dir):
        (data_dir / "single_e.jsonl").write_text(
            json.dumps({"input": "q", "context": "c", "answers": ["a"]}) + "\n"
        )
        registry = tmp_path / "single.toml"
        registry.write_text(
            '[[subset]]\nname = "single"\nfile = "single_e.jsonl"\n'
            'license = "x"\ntrain_use = false\n'
        )
        manifest = build_manifest(registry, data_dir)
        subset = manifest["subsets"][0]
        assert subset["entries"] == 1
        assert (subset["eval_half"], subset["train_half"]) == (1, 0)
        assert subset["train_boundary_fingerprint"] is None


class TestDeriveKeyPointsSentenceIntegrity:
    """Regression (issue #9 data fix): the old derivation split sentences
    on abbreviation periods ("U.S.", "H.R.", "S.") and hard-truncated
    points at 160 chars — producing fragment "facts" the judge cannot
    count. Points must be whole sentences."""

    def test_full_sentences_survive_no_char_truncation(self):
        long_sentence = (
            "Title 5 of the U.S. Code contains most of the standards "
            "governing federal employment, and OPM is generally responsible "
            "for implementing these requirements across the service."
        )
        points = derive_key_points([long_sentence])
        assert points == [long_sentence]

    def test_abbreviation_periods_do_not_split_sentences(self):
        answer = (
            "In the 116th Congress, the Fair Trade with China Enforcement "
            "Act (H.R. 704 and S. 2) and the Reciprocal Trade Act "
            "(H.R. 764) were introduced. Spending rose 4%."
        )
        points = derive_key_points([answer])
        assert any("(H.R. 704 and S. 2)" in p for p in points)
        assert not any(p.endswith(("H.R", "S.", "U.S.")) for p in points)

    def test_pub_l_no_citation_stays_one_sentence(self):
        answer = (
            "Pub. L. No. 110-229, enacted in 2008, amended the covenant to "
            "apply federal immigration law after a transition period ending in 2019."
        )
        points = derive_key_points([answer])
        assert points == [answer]
