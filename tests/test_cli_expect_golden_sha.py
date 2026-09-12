"""`trillic eval run --expect-golden-sha` — frozen-reference assertion
(issue #12).

The host calls the frozen evaluation package via a pinned ref; if the
golden set drifts, the run must fail BEFORE any gateway call (money) and
without leaving a run directory behind (a failed run that looks like a
result is worse than no run). The expected digest shares the single
source of truth with the run report's golden.sha256 — no second caliber.
"""

import hashlib
import json

import pytest

from trillic.cli import main


def golden_sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def out_root(tmp_path) -> str:
    return str(tmp_path / "runs")


def run_with_sha(config_path, golden_path, out_root, sha=None):
    argv = [
        "eval",
        "run",
        "--config",
        str(config_path),
        "--golden",
        str(golden_path),
        "--out",
        out_root,
    ]
    if sha is not None:
        argv += ["--expect-golden-sha", sha]
    return main(argv)


class TestMatchingDigest:
    def test_run_proceeds_and_records_same_digest(
        self, tmp_path, fixture_config_path, fixture_golden_path, out_root, capsys
    ):
        exit_code = run_with_sha(
            fixture_config_path, fixture_golden_path, out_root, golden_sha(fixture_golden_path)
        )
        assert exit_code == 0
        run_dir = max(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["golden"]["sha256"] == golden_sha(fixture_golden_path)

    def test_uppercase_and_prefixed_forms_accepted(
        self, fixture_config_path, fixture_golden_path, out_root, capsys
    ):
        sha = golden_sha(fixture_golden_path)
        exit_code = run_with_sha(
            fixture_config_path, fixture_golden_path, out_root, f"sha256:{sha.upper()}"
        )
        assert exit_code == 0

    def test_multi_file_set_uses_combined_digest(
        self, tmp_path, fixture_config_path, fixture_golden_path, out_root
    ):
        split = tmp_path / "split.jsonl"
        rows = [line for line in fixture_golden_path.read_text().splitlines() if line.strip()]
        split.write_text("\n".join(rows[:1]) + "\n")
        rest = tmp_path / "rest.jsonl"
        rest.write_text("\n".join(rows[1:]) + "\n")
        combined = hashlib.sha256(split.read_bytes() + rest.read_bytes()).hexdigest()
        argv = [
            "eval", "run",
            "--config", str(fixture_config_path),
            "--golden", str(split), str(rest),
            "--out", out_root,
            "--expect-golden-sha", combined,
        ]
        assert main(argv) == 0


class TestDriftedExam:
    def test_mismatch_fails_before_any_run_dir_or_ledger(
        self, tmp_path, fixture_config_path, fixture_golden_path, out_root, capsys
    ):
        exit_code = run_with_sha(
            fixture_config_path,
            fixture_golden_path,
            out_root,
            "0" * 64,
        )
        assert exit_code != 0
        err = capsys.readouterr().err
        assert "0" * 64 in err  # expected
        assert golden_sha(fixture_golden_path) in err  # actual
        runs = tmp_path / "runs"
        if runs.exists():
            assert not any(runs.iterdir()), "no run dir, no ledger — failure leaves nothing"

    def test_mismatch_message_names_the_discipline(
        self, fixture_config_path, fixture_golden_path, out_root, capsys
    ):
        exit_code = run_with_sha(
            fixture_config_path, fixture_golden_path, out_root, "1" * 64
        )
        assert exit_code != 0
        err = capsys.readouterr().err
        assert "golden" in err.lower()


class TestBadDigestFormat:
    @pytest.mark.parametrize("bad", ["deadbeef", "z" * 64, ""])
    def test_rejected_before_running(
        self, fixture_config_path, fixture_golden_path, out_root, capsys, bad
    ):
        exit_code = run_with_sha(fixture_config_path, fixture_golden_path, out_root, bad)
        assert exit_code != 0
        assert "error:" in capsys.readouterr().err
