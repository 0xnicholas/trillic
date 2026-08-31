"""Main seam: `trillic eval run` as a black box.

CLI in (config + golden), files out (run directory). The sidecar runs in
stub mode, so the whole flow is deterministic and zero-network. Expected
tiktoken numbers are recomputed here independently (direct tiktoken calls on
the fixture prompt and on the recorded compressed_text) and compared against
what the harness wrote — this is the hand-check wiring for acceptance
criterion "compression ratios are computed correctly for fixture entries".
"""

import hashlib
import json

import pytest
import tiktoken

from trillic import __version__
from trillic.cli import main

ENCODING = tiktoken.get_encoding("cl100k_base")

# Hand-recorded fixture numbers (acceptance criterion "与手工数字对照").
# Produced 2026-08-30 by an independent process: fresh interpreter, direct
# tiktoken cl100k_base calls on the fixture prompts and on the stub's known
# compressed outputs (aggressiveness 0.2). If the fixture file or the stub
# policy changes, these constants must be re-derived — that is the point.
HAND_CHECKED = {
    "fixture-rag-001": (149, 124, 0.8322, 0.1678),
    "fixture-sys-002": (97, 80, 0.8247, 0.1753),
    "fixture-chat-003": (163, 137, 0.8405, 0.1595),
}


def run_eval(tmp_path, config_path, golden_path):
    out_root = tmp_path / "runs"
    exit_code = main(
        [
            "eval",
            "run",
            "--config",
            str(config_path),
            "--golden",
            str(golden_path),
            "--out",
            str(out_root),
        ]
    )
    assert exit_code == 0
    run_dirs = [p for p in out_root.iterdir() if p.is_dir()]
    assert run_dirs  # at least one run directory was written
    return max(run_dirs)  # newest: run ids sort chronologically


def load_metrics(run_dir):
    return json.loads((run_dir / "metrics.json").read_text())


def strip_volatile(metrics):
    """Remove fields that legitimately vary between identical runs."""
    stripped = json.loads(json.dumps(metrics))
    stripped.pop("run_id")
    stripped.pop("created_at")
    stripped["config"].pop("source_path")
    stripped["golden"].pop("source_path")
    for level in stripped["metrics"]["levels"]:
        for item in level["items"]:
            item.pop("latency_seconds")
        level["aggregate"].pop("latency")
    return stripped


class TestEvalRunHappyPath:
    @pytest.fixture(autouse=True)
    def _run(self, tmp_path, fixture_config_path, fixture_golden_path):
        self.run_dir = run_eval(tmp_path, fixture_config_path, fixture_golden_path)
        self.metrics = load_metrics(self.run_dir)
        self.levels = self.metrics["metrics"]["levels"]
        self.level = self.levels[0]  # fixture config sweeps a single level
        self.golden_rows = [
            json.loads(line)
            for line in fixture_golden_path.read_text().splitlines()
            if line.strip()
        ]

    def test_run_dir_contains_metrics_and_report(self):
        assert (self.run_dir / "metrics.json").is_file()
        assert (self.run_dir / "report.md").is_file()

    def test_run_dir_name_embeds_config_name(self):
        assert "fixture" in self.run_dir.name

    def test_metrics_pin_harness_version(self):
        assert self.metrics["harness"]["name"] == "trillic"
        assert self.metrics["harness"]["version"] == __version__

    def test_metrics_embed_config_snapshot(self, fixture_config_path):
        snapshot = self.metrics["config"]
        raw = fixture_config_path.read_text()
        assert snapshot["raw"] == raw
        assert snapshot["sha256"] == hashlib.sha256(raw.encode()).hexdigest()
        assert snapshot["parsed"]["name"] == "fixture"

    def test_metrics_pin_golden_hash_and_count(self, fixture_golden_path):
        raw = fixture_golden_path.read_bytes()
        assert self.metrics["golden"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert self.metrics["golden"]["item_count"] == len(self.golden_rows)

    def test_metrics_schema_and_levels_structure(self):
        assert self.metrics["schema_version"] == 2
        assert [lv["aggressiveness"] for lv in self.levels] == [0.2]
        assert self.metrics["metrics"]["native_caliber"] == "word"

    def test_per_item_ratios_match_hand_checked_constants(self):
        """The letter of the acceptance criterion: literal hand-recorded
        numbers, not just in-test recomputation."""
        items = {i["id"]: i for i in self.level["items"]}
        for entry_id, (orig, comp, kept, removed) in HAND_CHECKED.items():
            item = items[entry_id]
            assert item["original_tokens"] == orig
            assert item["compressed_tokens"] == comp
            assert item["kept_ratio"] == kept
            assert item["compression_ratio"] == removed

    def test_per_item_tiktoken_ratios_match_independent_recomputation(self):
        items = self.level["items"]
        assert [i["id"] for i in items] == [row["id"] for row in self.golden_rows]
        for item, row in zip(items, self.golden_rows):
            expected_original = len(ENCODING.encode(row["prompt"], disallowed_special=()))
            expected_compressed = len(
                ENCODING.encode(item["compressed_text"], disallowed_special=())
            )
            assert item["original_tokens"] == expected_original
            assert item["compressed_tokens"] == expected_compressed
            assert item["kept_ratio"] == round(expected_compressed / expected_original, 4)
            assert item["compression_ratio"] == round(
                1 - expected_compressed / expected_original, 4
            )

    def test_compressed_text_is_extractive_subsequence(self):
        for item, row in zip(self.level["items"], self.golden_rows):
            source_words = iter(row["prompt"].split())
            kept_words = item["compressed_text"].split()
            assert all(word in source_words for word in kept_words)

    def test_aggregate_sums_and_ratios(self):
        items = self.level["items"]
        agg = self.level["aggregate"]
        total_o = sum(i["original_tokens"] for i in items)
        total_c = sum(i["compressed_tokens"] for i in items)
        assert agg["total_original_tokens"] == total_o
        assert agg["total_compressed_tokens"] == total_c
        assert agg["item_count"] == len(items)
        assert agg["corpus_kept_ratio"] == round(total_c / total_o, 4)
        assert agg["mean_kept_ratio"] == round(
            sum(i["kept_ratio"] for i in items) / len(items), 4
        )

    def test_report_md_summarizes_the_run(self):
        report = (self.run_dir / "report.md").read_text()
        assert self.metrics["run_id"] in report
        assert __version__ in report
        for row in self.golden_rows:
            assert row["id"] in report
        # both ratio calibers appear in the human table
        assert "kept" in report.lower()
        assert "compression" in report.lower()

    def test_metrics_record_sidecar_settings(self):
        sidecar = self.metrics["sidecar"]
        assert sidecar["mode"] == "stub"
        assert sidecar["refine_model"] == "stub-refine"
        assert sidecar["rewrite"] is False
        assert sidecar["compress"] is True
        assert sidecar["aggressiveness"] == 0.2
        assert self.metrics["metrics"]["tiktoken_encoding"] == "cl100k_base"

    def test_per_item_quality_metrics_present_and_sound(self):
        """fact recall + token F0.5 on every item, consistent with the
        pure functions (stub compression drops words, so recall < 1 is
        expected on fact-bearing prompts)."""
        from trillic.quality import fact_recall, text_token_f05

        for item, row in zip(self.level["items"], self.golden_rows):
            assert item["fact_recall"] == round(
                fact_recall(row["prompt"], item["compressed_text"]), 4
            )
            assert item["token_f05"] == round(
                text_token_f05(row["prompt"], item["compressed_text"]), 4
            )
            assert 0.0 <= item["fact_recall"] <= 1.0
            assert 0.0 <= item["token_f05"] <= 1.0

    def test_native_caliber_counts_present(self):
        """word flavor = whitespace: native counts must equal the stub
        sidecar's own word counts (cross-check of the second ratio
        caliber)."""
        for item in self.level["items"]:
            assert item["native_original_tokens"] == item["sidecar_original_tokens"]
            assert item["native_compressed_tokens"] == item["sidecar_compressed_tokens"]
            assert item["native_kept_ratio"] == round(
                item["native_compressed_tokens"] / item["native_original_tokens"], 4
            )
            assert item["native_compression_ratio"] == round(
                1 - item["native_compressed_tokens"] / item["native_original_tokens"], 4
            )

    def test_latency_recorded_per_item_and_percentiles(self):
        latencies = [item["latency_seconds"] for item in self.level["items"]]
        assert all(lat >= 0 for lat in latencies)
        latency = self.level["aggregate"]["latency"]
        assert latency["p95_seconds"] >= latency["p50_seconds"] >= 0
        assert latency["mean_seconds"] >= 0

    def test_task_quality_placeholder_present(self):
        """Issue #7 fills this in; the placeholder makes the fourth metric
        slot structural now."""
        assert self.level["aggregate"]["task_quality"] is None

    def test_report_md_covers_all_four_metric_slots(self):
        report = (self.run_dir / "report.md").read_text()
        assert "## Level 0.2" in report
        assert "fact recall" in report.lower()
        assert "f0.5" in report.lower()
        assert "p95" in report.lower()
        assert "word" in report  # native caliber named
        assert "task-level quality" in report  # placeholder slot


class TestSweepRun:
    """0.1-0.5 sweep structure, demonstrated under stubs."""

    @pytest.fixture(autouse=True)
    def _run(self, tmp_path, fixture_golden_path):
        config = tmp_path / "sweep.toml"
        config.write_text(
            'name = "sweep"\n\n[run]\nlevels = [0.1, 0.2, 0.3, 0.4, 0.5]\n',
            encoding="utf-8",
        )
        self.run_dir = run_eval(tmp_path, config, fixture_golden_path)
        self.metrics = load_metrics(self.run_dir)

    def test_five_level_blocks_in_order(self):
        levels = self.metrics["metrics"]["levels"]
        assert [lv["aggressiveness"] for lv in levels] == [0.1, 0.2, 0.3, 0.4, 0.5]
        for level in levels:
            assert len(level["items"]) == 3

    def test_compression_monotonically_deeper_for_stub_policy(self):
        """Stub drops round(10*a) of every 10-word block: deeper levels
        must compress strictly more (tiktoken corpus caliber)."""
        levels = self.metrics["metrics"]["levels"]
        corpus = [
            lv["aggregate"]["total_compressed_tokens"] for lv in levels
        ]
        assert corpus == sorted(corpus, reverse=True)
        assert corpus[0] > corpus[-1]

    def test_sidecar_block_records_levels_not_misleading_primary(self):
        assert self.metrics["sidecar"]["aggressiveness"] is None

    def test_report_lists_every_level(self):
        report = (self.run_dir / "report.md").read_text()
        for level in (0.1, 0.2, 0.3, 0.4, 0.5):
            assert f"## Level {level}" in report


class TestDeterminismAndImmutability:
    def test_same_inputs_produce_identical_metrics(self, tmp_path, fixture_config_path, fixture_golden_path):
        first = strip_volatile(load_metrics(run_eval(tmp_path, fixture_config_path, fixture_golden_path)))
        second_run_dir = run_eval(tmp_path, fixture_config_path, fixture_golden_path)
        second = strip_volatile(load_metrics(second_run_dir))
        assert first == second

    def test_each_run_gets_its_own_directory(self, tmp_path, fixture_config_path, fixture_golden_path):
        run_eval(tmp_path, fixture_config_path, fixture_golden_path)
        run_eval(tmp_path, fixture_config_path, fixture_golden_path)
        out_root = tmp_path / "runs"
        assert len(list(out_root.iterdir())) == 2


class TestCliErrors:
    def test_missing_config_file_fails_cleanly(self, tmp_path, fixture_golden_path, capsys):
        code = main(
            [
                "eval", "run",
                "--config", str(tmp_path / "nope.toml"),
                "--golden", str(fixture_golden_path),
                "--out", str(tmp_path / "runs"),
            ]
        )
        assert code == 1
        assert "nope.toml" in capsys.readouterr().err

    def test_invalid_golden_entry_names_the_entry(self, tmp_path, fixture_config_path, capsys):
        bad_golden = tmp_path / "golden.jsonl"
        source = {
            "dataset": "handwritten", "subset": "test", "license": "original", "split": "eval",
        }
        bad_golden.write_text(
            json.dumps({"id": "good-1", "load_type": "rag", "prompt": "p",
                        "key_points": ["k"], "source": source})
            + "\n"
            + json.dumps({"id": "bad-2", "load_type": "rag", "prompt": "p",
                          "key_points": [], "source": source})
            + "\n"
        )
        code = main(
            [
                "eval", "run",
                "--config", str(fixture_config_path),
                "--golden", str(bad_golden),
                "--out", str(tmp_path / "runs"),
            ]
        )
        assert code == 1
        stderr = capsys.readouterr().err
        assert "bad-2" in stderr
        assert "line 2" in stderr

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        assert __version__ in capsys.readouterr().out
