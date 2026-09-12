"""`trillic delivery verify` — static drop-in contract checks (issue #11).

Black-box CLI tests: exit codes, stdout summary, machine-readable JSON
report, content-addressed checkpoint identity. Fixtures are hand-built
minimal checkpoint directories (default dependencies only, zero network):
one fully compliant, plus one single-point violation per contract item.
Checkpoint fixture builders live in conftest (shared with the pack tests).
"""

import json
from pathlib import Path

import pytest

from trillic import __version__
from trillic.delivery import load_deps_available
from helpers import (
    VALID_CHECKPOINT_CONFIG as VALID_CONFIG,
    install_fake_load_deps,
    make_checkpoint,
    write_blanking_tokenizer,
)


def make_checkpoint_with_tokenizer(root: Path, name: str, tokenizer_writer) -> Path:
    """Compliant config + a custom tokenizer.json writer."""
    directory = make_checkpoint(root, name)
    tokenizer_writer(directory / "tokenizer.json")
    return directory


def run_verify(directory: Path, *extra: str) -> int:
    from trillic.cli import main

    return main(["delivery", "verify", str(directory), *extra])


@pytest.fixture
def compliant(tmp_path) -> Path:
    return make_checkpoint(tmp_path, "valid-ckpt")


class TestCompliantCheckpoint:
    def test_exit_zero_and_human_summary(self, compliant, capsys):
        exit_code = run_verify(compliant)
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "static layer: PASS" in out
        assert "overall: PASS" in out

    def test_report_is_machine_readable_and_content_addressed(
        self, compliant, tmp_path, capsys
    ):
        report_path = tmp_path / "reports" / "verify.json"
        exit_code = run_verify(compliant, "--report", str(report_path))
        assert exit_code == 0
        report = json.loads(report_path.read_text())
        assert report["overall"] == "pass"
        assert report["harness"]["name"] == "trillic"
        assert report["harness"]["version"] == __version__
        # content-addressed identity, not just a path
        checkpoint = report["checkpoint"]
        assert checkpoint["digest"]
        assert len(checkpoint["digest"]) == 64
        assert {file["name"] for file in checkpoint["files"]} == {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        }
        for file in checkpoint["files"]:
            assert len(file["sha256"]) == 64

    def test_digest_tracks_content_changes(self, compliant, tmp_path, capsys):
        report_a = tmp_path / "a.json"
        run_verify(compliant, "--report", str(report_a))
        (compliant / "model.safetensors").write_bytes(b"\x00" * 17)
        report_b = tmp_path / "b.json"
        run_verify(compliant, "--report", str(report_b))
        digest_a = json.loads(report_a.read_text())["checkpoint"]["digest"]
        digest_b = json.loads(report_b.read_text())["checkpoint"]["digest"]
        assert digest_a != digest_b

    def test_static_checks_all_listed(self, compliant, tmp_path):
        report_path = tmp_path / "verify.json"
        run_verify(compliant, "--report", str(report_path))
        checks = json.loads(report_path.read_text())["layers"]["static"]["checks"]
        assert {c["id"] for c in checks} == {
            "architecture_class",
            "num_labels",
            "id2label_semantics",
            "fast_tokenizer",
            "offsets_available",
        }
        assert all(c["status"] == "pass" for c in checks)


class TestSinglePointViolations:
    @pytest.mark.parametrize(
        "mutate,check_id",
        [
            (
                lambda cfg: cfg | {"architectures": ["RobertaForTokenClassification"]},
                "architecture_class",
            ),
            (
                lambda cfg: cfg
                | {
                    "num_labels": 3,
                    "id2label": {"0": "drop", "1": "keep", "2": "extra"},
                },
                "num_labels",
            ),
            (
                lambda cfg: cfg | {"id2label": {"0": "keep", "1": "drop"}},
                "id2label_semantics",
            ),
            (
                lambda cfg: cfg | {"id2label": {"1": "keep", "2": "drop"}},
                "id2label_semantics",
            ),
        ],
    )
    def test_config_violation_rejected_with_named_check(
        self, tmp_path, capsys, mutate, check_id
    ):
        directory = make_checkpoint(tmp_path, "bad", config=mutate(dict(VALID_CONFIG)))
        report_path = tmp_path / "verify.json"
        exit_code = run_verify(directory, "--report", str(report_path))
        assert exit_code != 0
        err = capsys.readouterr().err
        assert check_id in err
        report = json.loads(report_path.read_text())
        assert report["overall"] == "violation"
        checks = {c["id"]: c for c in report["layers"]["static"]["checks"]}
        assert checks[check_id]["status"] == "violation"
        # actionable: the violation names a remedy, not just a fact
        assert checks[check_id]["remedy"]

    def test_non_fast_tokenizer_rejected(self, tmp_path, capsys):
        directory = make_checkpoint(tmp_path, "slow", fast_tokenizer=False)
        exit_code = run_verify(directory)
        assert exit_code != 0
        report_err = capsys.readouterr().err
        assert "fast_tokenizer" in report_err

    def test_offsets_unavailable_rejected(self, tmp_path, capsys):
        directory = make_checkpoint_with_tokenizer(tmp_path, "no-offsets", write_blanking_tokenizer)
        exit_code = run_verify(directory)
        assert exit_code != 0
        assert "offsets_available" in capsys.readouterr().err

    def test_violation_names_the_contract_item_in_human_output(
        self, tmp_path, capsys
    ):
        directory = make_checkpoint(
            tmp_path,
            "bad-arch",
            config=VALID_CONFIG
            | {"architectures": ["RobertaForTokenClassification"]},
        )
        exit_code = run_verify(directory)
        assert exit_code != 0
        out = capsys.readouterr().out
        assert "architecture_class" in out
        assert "VIOLATION" in out


class TestLoadLayer:
    """Three states, never conflated: pass / violation / unverified.

    The heavy deps are absent in the default environment, so the absent
    path is exercised naturally; the --static-only path is env-
    independent; real instantiation is left to the stage-5 runbook real
    run (it needs a real checkpoint, not the minimal fixture).
    """

    def test_absent_deps_reports_unverified_not_pass(self, tmp_path, capsys):
        if load_deps_available():
            pytest.skip("heavy deps installed — absent path needs them gone")
        directory = make_checkpoint(tmp_path, "valid-ckpt")
        report_path = tmp_path / "verify.json"
        exit_code = run_verify(directory, "--report", str(report_path))
        assert exit_code == 0  # unverified ≠ violation
        report = json.loads(report_path.read_text())
        load = report["layers"]["load"]
        assert load["status"] == "unverified"
        assert "uv sync --group load" in load["reason"]
        assert report["overall"] == "pass"
        out = capsys.readouterr().out
        assert "load layer: UNVERIFIED" in out

    def test_static_only_reports_unverified_with_reason(self, tmp_path, capsys):
        directory = make_checkpoint(tmp_path, "valid-ckpt")
        report_path = tmp_path / "verify.json"
        exit_code = run_verify(
            directory, "--static-only", "--report", str(report_path)
        )
        assert exit_code == 0
        load = json.loads(report_path.read_text())["layers"]["load"]
        assert load["status"] == "unverified"
        assert "--static-only" in load["reason"]
        assert "UNVERIFIED" in capsys.readouterr().out

    def test_fake_deps_pass_state(self, tmp_path, capsys, monkeypatch):
        """Load layer present + compliant model -> PASS with both checks."""
        install_fake_load_deps(monkeypatch)
        directory = make_checkpoint(tmp_path, "valid-ckpt")
        report_path = tmp_path / "verify.json"
        exit_code = run_verify(directory, "--report", str(report_path))
        assert exit_code == 0
        load = json.loads(report_path.read_text())["layers"]["load"]
        assert load["status"] == "pass"
        assert {c["id"] for c in load["checks"]} == {
            "fast_tokenizer_runtime",
            "dot_bert_attribute",
        }
        assert all(c["status"] == "pass" for c in load["checks"])
        assert "load layer: PASS" in capsys.readouterr().out

    def test_fake_deps_violation_without_bert(self, tmp_path, capsys, monkeypatch):
        """XLM-R-style base (no .bert) -> load VIOLATION, non-zero exit."""
        install_fake_load_deps(monkeypatch, model_has_bert=False)
        directory = make_checkpoint(tmp_path, "valid-ckpt")
        report_path = tmp_path / "verify.json"
        exit_code = run_verify(directory, "--report", str(report_path))
        assert exit_code != 0
        report = json.loads(report_path.read_text())
        load = report["layers"]["load"]
        assert load["status"] == "violation"
        checks = {c["id"]: c for c in load["checks"]}
        assert checks["dot_bert_attribute"]["status"] == "violation"
        assert ".bert" in checks["dot_bert_attribute"]["remedy"]
        assert report["overall"] == "violation"

    def test_fake_deps_violation_slow_runtime_tokenizer(
        self, tmp_path, capsys, monkeypatch
    ):
        """Runtime serves a slow tokenizer -> offsets check violates."""
        install_fake_load_deps(monkeypatch, tokenizer_fast=False)
        directory = make_checkpoint(tmp_path, "valid-ckpt")
        exit_code = run_verify(directory)
        assert exit_code != 0
        assert "fast_tokenizer_runtime" in capsys.readouterr().err

    def test_static_violation_still_fails_overall_regardless_of_load(
        self, tmp_path, capsys
    ):
        directory = make_checkpoint(
            tmp_path,
            "bad-num-labels",
            config=VALID_CONFIG | {"num_labels": 5},
        )
        exit_code = run_verify(directory, "--static-only")
        assert exit_code != 0


class TestBadInvocations:
    def test_missing_directory_is_an_error(self, tmp_path, capsys):
        from trillic.cli import main

        exit_code = main(["delivery", "verify", str(tmp_path / "nope")])
        assert exit_code != 0
        assert "error:" in capsys.readouterr().err

    def test_broken_config_json_is_an_error(self, tmp_path, capsys):
        directory = make_checkpoint(tmp_path, "broken")
        (directory / "config.json").write_text("{not json")
        exit_code = run_verify(directory)
        assert exit_code != 0
        assert "config.json" in capsys.readouterr().err
