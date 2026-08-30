"""Report writer: immutable run directory + markdown rendering."""

import json

import pytest

from trillic.report import ReportWriterError, render_report_md, write_run_dir


def sample_metrics() -> dict:
    return {
        "schema_version": 1,
        "run_id": "20260830T120000000000Z-fixture-abcdef12",
        "created_at": "2026-08-30T12:00:00+00:00",
        "harness": {"name": "trillic", "version": "0.1.0"},
        "config": {"raw": "name = \"x\"", "sha256": "0" * 64, "parsed": {"name": "x"}},
        "golden": {"sha256": "1" * 64, "item_count": 2},
        "sidecar": {"mode": "stub", "refine_model": "stub-refine"},
        "metrics": {
            "tiktoken_encoding": "cl100k_base",
            "items": [
                {"id": "a", "original_tokens": 100, "compressed_tokens": 80,
                 "kept_ratio": 0.8, "compression_ratio": 0.2, "compressed_text": "..."},
                {"id": "b", "original_tokens": 50, "compressed_tokens": 25,
                 "kept_ratio": 0.5, "compression_ratio": 0.5, "compressed_text": "..."},
            ],
            "aggregate": {
                "item_count": 2,
                "total_original_tokens": 150,
                "total_compressed_tokens": 105,
                "corpus_kept_ratio": 0.7,
                "corpus_compression_ratio": 0.3,
                "mean_kept_ratio": 0.65,
                "mean_compression_ratio": 0.35,
            },
        },
    }


class TestWriteRunDir:
    def test_writes_metrics_and_report_into_new_dir(self, tmp_path):
        metrics = sample_metrics()
        run_dir = write_run_dir(tmp_path, metrics, report_md="# hello")
        assert run_dir.is_dir()
        assert json.loads((run_dir / "metrics.json").read_text()) == metrics
        assert (run_dir / "report.md").read_text() == "# hello"

    def test_refuses_to_overwrite_existing_run_dir(self, tmp_path):
        write_run_dir(tmp_path, sample_metrics(), report_md="# one")
        with pytest.raises(ReportWriterError, match="immutable"):
            write_run_dir(tmp_path, sample_metrics(), report_md="# two")


class TestRenderReport:
    def test_report_contains_ids_ratios_and_version(self):
        report = render_report_md(sample_metrics())
        assert "20260830T120000000000Z-fixture-abcdef12" in report  # run id
        assert "| a |" in report and "| b |" in report  # per-item rows
        assert "0.1.0" in report  # harness version
        assert "cl100k_base" in report
        assert "150" in report and "105" in report  # totals
        assert "name = \"x\"" in report  # config snapshot echo
