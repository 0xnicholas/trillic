"""CLI seams for issue #8's zero-cost slice: multi-file golden input,
--resume-from replay, and the sizing tool."""

import json

import pytest

from trillic.cli import main

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
FIXTURE_GOLDEN = REPO_ROOT / "eval" / "golden" / "fixture.jsonl"


def write_extra_golden(tmp_path):
    source = {
        "dataset": "handwritten", "subset": "extra", "license": "original", "split": "eval",
    }
    path = tmp_path / "extra.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "extra-rag-101",
                "load_type": "rag",
                "prompt": "Docs: invoices arrive within 48 hours. Question: when do invoices arrive?",
                "key_points": ["invoices arrive within 48 hours"],
                "source": source,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def run(tmp_path, *extra_args, config=None, golden=None):
    config = config or (tmp_path / "run.toml")
    if not config.exists():
        config.write_text('name = "cli8"\n', encoding="utf-8")
    golden = golden if isinstance(golden, list) else [golden or FIXTURE_GOLDEN]
    out = tmp_path / "runs"
    code = main(
        ["eval", "run", "--config", str(config), "--golden", *[str(g) for g in golden],
         "--out", str(out), *extra_args]
    )
    assert code == 0, extra_args
    run_dir = max(p for p in out.iterdir() if p.is_dir())
    return json.loads((run_dir / "metrics.json").read_text()), run_dir


class TestMultiGoldenInput:
    def test_multiple_golden_files_merge_in_order(self, tmp_path):
        extra = write_extra_golden(tmp_path)
        metrics, _ = run(tmp_path, golden=[FIXTURE_GOLDEN, extra])
        ids = [i["id"] for lv in metrics["metrics"]["levels"] for i in lv["items"]]
        assert ids[:3] == ["fixture-rag-001", "fixture-sys-002", "fixture-chat-003"]
        assert ids[3] == "extra-rag-101"
        assert metrics["golden"]["item_count"] == 4

    def test_combined_sha_is_over_concatenated_bytes_in_order(self, tmp_path):
        import hashlib

        extra = write_extra_golden(tmp_path)
        metrics, _ = run(tmp_path, golden=[FIXTURE_GOLDEN, extra])
        combined = FIXTURE_GOLDEN.read_bytes() + extra.read_bytes()
        assert metrics["golden"]["sha256"] == hashlib.sha256(combined).hexdigest()

    def test_cross_file_duplicate_ids_are_rejected(self, tmp_path, capsys):
        dup = tmp_path / "dup.jsonl"
        dup.write_text(FIXTURE_GOLDEN.read_text(), encoding="utf-8")
        config = tmp_path / "c.toml"
        config.write_text('name = "x"\n', encoding="utf-8")
        code = main(
            ["eval", "run", "--config", str(config),
             "--golden", str(FIXTURE_GOLDEN), str(dup), "--out", str(tmp_path / "runs")]
        )
        assert code == 1
        assert "duplicate id" in capsys.readouterr().err


class TestResumeFrom:
    def test_resume_replays_gateway_results_with_zero_fresh_items(self, tmp_path):
        metrics, run_dir = run(tmp_path)
        prior_id = metrics["run_id"]

        resumed, _ = run(tmp_path, "--resume-from", str(run_dir))
        tq = resumed["task_quality"]
        assert tq["resumed_from"]["run_id"] == prior_id
        assert tq["gateway_calls"]["answers_fresh"] == 0
        assert tq["gateway_calls"]["judges_fresh"] == 0
        assert tq["gateway_calls"]["answers_reused"] == 6  # 3 items x 2 sides
        # per-level rows record the reuse source
        level = resumed["metrics"]["levels"][0]
        rows = level["aggregate"]["task_quality"]["items"]
        assert all(row["source_compressed"] == "reused" for row in rows)
        assert all(row["source_original"] == "reused" for row in rows)

    def test_resume_output_report_mentions_the_prior_run(self, tmp_path):
        config = tmp_path / "run.toml"
        config.write_text('name = "cli8"\n', encoding="utf-8")
        _, prior_dir = run(tmp_path, config=config)
        prior_id = prior_dir.name

        code = main(
            ["eval", "run", "--config", str(config),
             "--golden", str(FIXTURE_GOLDEN), "--out", str(tmp_path / "runs"),
             "--resume-from", str(prior_dir)]
        )
        assert code == 0
        resumed_dir = max(
            p for p in (tmp_path / "runs").iterdir() if p.is_dir() and p != prior_dir
        )
        report = (resumed_dir / "report.md").read_text()
        assert prior_id in report  # prior run id shown
        assert "reused" in report

    def test_resume_with_task_quality_disabled_is_rejected(self, tmp_path, capsys):
        _, run_dir = run(tmp_path)
        config = tmp_path / "noq.toml"
        config.write_text(
            'name = "noq"\n\n[quality]\ntask_quality = false\n', encoding="utf-8"
        )
        code = main(
            ["eval", "run", "--config", str(config), "--golden", str(FIXTURE_GOLDEN),
             "--out", str(tmp_path / "runs2"), "--resume-from", str(run_dir)]
        )
        assert code == 1
        assert "task_quality" in capsys.readouterr().err

    def test_resume_from_missing_dir_is_rejected(self, tmp_path, capsys):
        config = tmp_path / "run.toml"
        config.write_text('name = "x"\n', encoding="utf-8")
        code = main(
            ["eval", "run", "--config", str(config),
             "--golden", str(FIXTURE_GOLDEN), "--out", str(tmp_path / "runs"),
             "--resume-from", str(tmp_path / "nope")]
        )
        assert code == 1
        assert "metrics.json" in capsys.readouterr().err


class TestEvalSizing:
    def test_sizing_reads_a_run_dir_and_reports_required_n(self, tmp_path, capsys):
        metrics, run_dir = run(tmp_path)
        code = main(["eval", "sizing", "--run", str(run_dir)])
        assert code == 0
        out = capsys.readouterr().out
        assert "required n" in out.lower()
        assert "cohen" in out.lower()
        assert "mean delta" in out.lower()

    def test_sizing_writes_json_when_asked(self, tmp_path):
        metrics, run_dir = run(tmp_path)
        out_path = tmp_path / "sizing.json"
        assert main(["eval", "sizing", "--run", str(run_dir), "--out", str(out_path)]) == 0
        report = json.loads(out_path.read_text())
        assert "overall" in report
        assert report["overall"]["n_required"] >= 1

    def test_sizing_level_filter_selects_one_level(self, tmp_path):
        config = tmp_path / "sweep.toml"
        config.write_text(
            'name = "sw"\n\n[run]\nlevels = [0.2, 0.4]\n', encoding="utf-8"
        )
        metrics, run_dir = run(tmp_path, config=config)
        code = main(["eval", "sizing", "--run", str(run_dir), "--level", "0.4"])
        assert code == 0

    def test_sizing_unknown_level_is_rejected(self, tmp_path, capsys):
        config = tmp_path / "sweep.toml"
        config.write_text(
            'name = "sw"\n\n[run]\nlevels = [0.2]\n', encoding="utf-8"
        )
        _, run_dir = run(tmp_path, config=config)
        code = main(["eval", "sizing", "--run", str(run_dir), "--level", "0.9"])
        assert code == 1
        assert "0.9" in capsys.readouterr().err

    def test_sizing_on_qualityless_run_is_rejected(self, tmp_path, capsys):
        config = tmp_path / "noq.toml"
        config.write_text(
            'name = "noq"\n\n[quality]\ntask_quality = false\n', encoding="utf-8"
        )
        _, run_dir = run(tmp_path, config=config)
        code = main(["eval", "sizing", "--run", str(run_dir)])
        assert code == 1
        assert "task_quality" in capsys.readouterr().err


class TestCrashJournal:
    """Issue #8: kill a run mid-flight, rerun with the same config — the
    journal replays completed gateway work, zero re-billing."""

    def test_journal_written_incrementally_and_rerun_replays_it(self, tmp_path):
        config = tmp_path / "run.toml"
        config.write_text('name = "cli8"\n', encoding="utf-8")
        metrics, run_dir = run(tmp_path, config=config)
        # journal exists under the runs root, one line per fresh result
        journals = list((tmp_path / "runs" / ".ledger").glob("*.jsonl"))
        assert len(journals) == 1
        lines = [json.loads(l) for l in journals[0].read_text().splitlines() if l.strip()]
        # one line per _answer_and_judge pair (covers both gateway calls):
        # 3 originals + 3 compressed
        assert len(lines) == 6
        assert len([l for l in lines if l["kind"] == "original"]) == 3
        assert len([l for l in lines if l["kind"] == "compressed"]) == 3

        # simulate the kill: the completed run dir vanishes (metrics never
        # written); rerun the same config + golden against the same root
        import shutil

        shutil.rmtree(run_dir)
        code = main(
            ["eval", "run", "--config", str(config), "--golden", str(FIXTURE_GOLDEN),
             "--out", str(tmp_path / "runs")]
        )
        assert code == 0
        rerun_dir = max(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
        rerun = json.loads((rerun_dir / "metrics.json").read_text())
        calls = rerun["task_quality"]["gateway_calls"]
        assert calls["answers_fresh"] == 0
        assert calls["judges_fresh"] == 0
        assert calls["answers_reused"] == 6

    def test_journal_shares_originals_across_levels_and_reruns(self, tmp_path):
        """Originals are level-independent: a sweep rerun at different
        levels (and, in a dual-checkpoint baseline, the second checkpoint)
        replays the original side from the same journal and only bills the
        new compressed sides."""
        config = tmp_path / "one.toml"
        config.write_text('name = "one"\n', encoding="utf-8")
        metrics, _ = run(tmp_path, config=config)  # level 0.2, 6 fresh pairs
        assert metrics["task_quality"]["gateway_calls"]["answers_fresh"] == 6

        sweep = tmp_path / "sweep.toml"
        sweep.write_text(
            'name = "sweep"\n\n[run]\nlevels = [0.2, 0.4]\n', encoding="utf-8"
        )
        code = main(
            ["eval", "run", "--config", str(sweep), "--golden", str(FIXTURE_GOLDEN),
             "--out", str(tmp_path / "runs")]
        )
        assert code == 0
        rerun_dir = max(
            p for p in (tmp_path / "runs").iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        rerun = json.loads((rerun_dir / "metrics.json").read_text())
        calls = rerun["task_quality"]["gateway_calls"]
        # 3 originals replayed + 3 level-0.2 compressed sides replayed;
        # only the level-0.4 compressed sides are fresh
        assert calls["answers_fresh"] == 3
        assert calls["answers_reused"] == 6
