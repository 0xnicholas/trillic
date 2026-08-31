"""Report writer: immutable run directory + markdown rendering."""

import json

import pytest

from trillic.report import ReportWriterError, render_report_md, write_run_dir


def sample_metrics() -> dict:
    return {
        "schema_version": 2,
        "run_id": "20260830T120000000000Z-fixture-abcdef12",
        "created_at": "2026-08-30T12:00:00+00:00",
        "harness": {"name": "trillic", "version": "0.1.0"},
        "config": {"raw": "name = \"x\"", "sha256": "0" * 64, "parsed": {"name": "x"}},
        "golden": {"sha256": "1" * 64, "item_count": 2},
        "sidecar": {"mode": "stub", "refine_model": "stub-refine", "aggressiveness": 0.2},
        "metrics": {
            "tiktoken_encoding": "cl100k_base",
            "native_caliber": "word",
            "levels": [
                {
                    "aggressiveness": 0.2,
                    "items": [
                        {"id": "a", "load_type": "rag",
                         "original_tokens": 100, "compressed_tokens": 80,
                         "kept_ratio": 0.8, "compression_ratio": 0.2,
                         "native_original_tokens": 120, "native_compressed_tokens": 96,
                         "native_kept_ratio": 0.8, "native_compression_ratio": 0.2,
                         "fact_recall": 0.75, "token_f05": 0.8333,
                         "latency_seconds": 0.01,
                         "sidecar_original_tokens": 120, "sidecar_compressed_tokens": 96,
                         "compressed_text": "..."},
                        {"id": "b", "load_type": "dialogue",
                         "original_tokens": 50, "compressed_tokens": 25,
                         "kept_ratio": 0.5, "compression_ratio": 0.5,
                         "native_original_tokens": 60, "native_compressed_tokens": 30,
                         "native_kept_ratio": 0.5, "native_compression_ratio": 0.5,
                         "fact_recall": 1.0, "token_f05": 0.6667,
                         "latency_seconds": 0.02,
                         "sidecar_original_tokens": 60, "sidecar_compressed_tokens": 30,
                         "compressed_text": "..."},
                    ],
                    "aggregate": {
                        "item_count": 2,
                        "total_original_tokens": 150,
                        "total_compressed_tokens": 105,
                        "corpus_kept_ratio": 0.7,
                        "corpus_compression_ratio": 0.3,
                        "mean_kept_ratio": 0.65,
                        "mean_compression_ratio": 0.35,
                        "total_native_original_tokens": 180,
                        "total_native_compressed_tokens": 126,
                        "corpus_native_kept_ratio": 0.7,
                        "corpus_native_compression_ratio": 0.3,
                        "mean_fact_recall": 0.875,
                        "mean_token_f05": 0.75,
                        "latency": {"mean_seconds": 0.015, "p50_seconds": 0.015, "p95_seconds": 0.0195},
                        "task_quality": None,
                    },
                }
            ],
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

    def test_report_covers_all_four_metric_slots(self):
        report = render_report_md(sample_metrics())
        assert "## Level 0.2" in report
        assert "fact recall" in report and "F0.5" in report
        assert "p95" in report
        assert "native" in report and "word" in report
        assert "task-level quality: pending" in report
