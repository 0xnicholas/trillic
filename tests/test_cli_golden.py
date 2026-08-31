"""`trillic golden *` CLI: validate / manifest / build-rag / sysprompt family.

External behavior only: argv in, files + stdout/stderr + exit codes out.
"""

import json
from pathlib import Path

from trillic.cli import main
from trillic.longbench import build_manifest
from trillic.sysprompt import load_families

REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILIES_TOML = REPO_ROOT / "eval" / "manifests" / "sysprompt_families.toml"

SOURCE = {
    "dataset": "handwritten", "subset": "test", "license": "original", "split": "eval",
}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_entry(entry_id, *, key_points=("k",), split="eval"):
    return {
        "id": entry_id,
        "load_type": "rag",
        "prompt": "p",
        "key_points": list(key_points),
        "source": {**SOURCE, "split": split},
    }


class TestGoldenValidate:
    def test_valid_file_reports_ok(self, tmp_path, capsys):
        golden = tmp_path / "golden.jsonl"
        write_jsonl(golden, [make_entry("a-1"), make_entry("a-2")])
        assert main(["golden", "validate", str(golden)]) == 0
        out = capsys.readouterr().out
        assert "ok" in out and "2" in out

    def test_invalid_file_reports_every_bad_row_with_line_numbers(self, tmp_path, capsys):
        golden = tmp_path / "golden.jsonl"
        write_jsonl(
            golden,
            [
                make_entry("a-1"),
                make_entry("bad-2", key_points=()),
                make_entry("bad-3", split="train"),
            ],
        )
        assert main(["golden", "validate", str(golden)]) == 1
        stderr = capsys.readouterr().err
        assert "bad-2" in stderr and "line 2" in stderr
        assert "bad-3" in stderr and "line 3" in stderr

    def test_multiple_files_all_checked(self, tmp_path, capsys):
        good, bad = tmp_path / "good.jsonl", tmp_path / "bad.jsonl"
        write_jsonl(good, [make_entry("a-1")])
        write_jsonl(bad, [make_entry("b-1", key_points=())])
        assert main(["golden", "validate", str(good), str(bad)]) == 1
        captured = capsys.readouterr()
        assert "ok" in captured.out
        assert "b-1" in captured.err

    def test_rag_pilot_golden_passes(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "eval" / "golden" / "rag_pilot.jsonl"
        if not path.exists():  # built later in this issue's pipeline
            import pytest

            pytest.skip("rag_pilot.jsonl not built yet")
        assert main(["golden", "validate", str(path)]) == 0


class TestGoldenManifest:
    def test_writes_deterministic_manifest_json(self, tmp_path, capsys, subsets_toml, data_dir):
        out = tmp_path / "longbench.json"
        args = ["golden", "manifest",
                "--subsets", str(subsets_toml), "--data-dir", str(data_dir), "--out", str(out)]
        assert main(args) == 0
        first = json.loads(out.read_text())
        assert main(args) == 0
        assert json.loads(out.read_text()) == first
        assert first["subsets"][0]["name"] in ("qasper", "hotpotqa")
        capsys.readouterr()  # drain

    def test_missing_data_file_errors(self, tmp_path, capsys, subsets_toml):
        empty = tmp_path / "data"
        empty.mkdir()
        code = main(
            ["golden", "manifest", "--subsets", str(subsets_toml),
             "--data-dir", str(empty), "--out", str(tmp_path / "m.json")]
        )
        assert code == 1
        assert "error" in capsys.readouterr().err


class TestGoldenBuildRag:
    def test_builds_validated_golden_file_deterministically(self, tmp_path, capsys, subsets_toml, data_dir):
        manifest_path = tmp_path / "longbench.json"
        manifest_path.write_text(json.dumps(build_manifest(subsets_toml, data_dir)))
        out = tmp_path / "draft.jsonl"
        args = [
            "golden", "build-rag",
            "--manifest", str(manifest_path),
            "--data-dir", str(data_dir),
            "--counts", "qasper=3,hotpotqa=2",
            "--seed", "7",
            "--min-tokens", "0",
            "--max-tokens", "1000000",
            "--out", str(out),
        ]
        assert main(args) == 0
        first = out.read_text()
        assert main(args) == 0
        assert out.read_text() == first
        from trillic.golden import load_golden

        assert len(load_golden(out)) == 5
        capsys.readouterr()


class TestGoldenManifestSysprompt:
    def test_writes_manifest_with_pilot_record(self, tmp_path, capsys):
        out = tmp_path / "sysprompt.json"
        golden = tmp_path / "pilot.jsonl"
        _write_pilot(golden)
        args = [
            "golden", "manifest-sysprompt",
            "--families", str(FAMILIES_TOML),
            "--golden", str(golden),
            "--out", str(out),
        ]
        assert main(args) == 0
        manifest = json.loads(out.read_text())
        assert manifest["pilot"]["entries"] == 10
        assert len(manifest["families"]) == 10
        capsys.readouterr()

    def test_manifest_without_golden_records_allocation_only(self, tmp_path):
        out = tmp_path / "sysprompt.json"
        assert main(
            ["golden", "manifest-sysprompt", "--families", str(FAMILIES_TOML),
             "--out", str(out)]
        ) == 0
        assert "pilot" not in json.loads(out.read_text())

    def test_drifted_golden_errors(self, tmp_path, capsys):
        golden = tmp_path / "drifted.jsonl"
        _write_pilot(golden, tamper_first=True)
        code = main(
            ["golden", "manifest-sysprompt", "--families", str(FAMILIES_TOML),
             "--golden", str(golden), "--out", str(tmp_path / "m.json")]
        )
        assert code == 1
        assert "content_sha1" in capsys.readouterr().err

    def test_review_flags_record_owner_signoff(self, tmp_path, capsys):
        golden = tmp_path / "pilot.jsonl"
        _write_pilot(golden)
        out = tmp_path / "sysprompt.json"
        assert main(
            ["golden", "manifest-sysprompt", "--families", str(FAMILIES_TOML),
             "--golden", str(golden), "--review-status", "approved",
             "--reviewer", "nicholas", "--review-notes", "spot-checked all 10",
             "--out", str(out)]
        ) == 0
        review = json.loads(out.read_text())["pilot"]["review"]
        assert review["status"] == "approved"
        assert review["reviewer"] == "nicholas"
        assert "spot-checked" in review["notes"]
        capsys.readouterr()

    def test_review_status_without_golden_errors(self, tmp_path, capsys):
        code = main(
            ["golden", "manifest-sysprompt", "--families", str(FAMILIES_TOML),
             "--review-status", "approved", "--out", str(tmp_path / "m.json")]
        )
        assert code == 1
        assert "--golden" in capsys.readouterr().err


class TestGoldenBuildSysprompt:
    def test_builds_validated_golden_file_deterministically(self, tmp_path, capsys):
        out = tmp_path / "draft.jsonl"
        seeds = "support_logistics=101,finance_analyst=101"
        args = [
            "golden", "build-sysprompt",
            "--families", str(FAMILIES_TOML),
            "--seeds", seeds,
            "--out", str(out),
        ]
        assert main(args) == 0
        first = out.read_text()
        assert main(args) == 0
        assert out.read_text() == first
        from trillic.golden import load_golden

        items = load_golden(out)
        assert [i.id for i in items] == [
            "sys-support_logistics-s101",
            "sys-finance_analyst-s101",
        ]
        capsys.readouterr()

    def test_train_seed_plan_errors(self, tmp_path, capsys):
        code = main(
            ["golden", "build-sysprompt", "--families", str(FAMILIES_TOML),
             "--seeds", "support_logistics=901", "--out", str(tmp_path / "d.jsonl")]
        )
        assert code == 1
        assert "TRAIN seed" in capsys.readouterr().err

    def test_bad_seeds_syntax_errors(self, tmp_path, capsys):
        code = main(
            ["golden", "build-sysprompt", "--families", str(FAMILIES_TOML),
             "--seeds", "support_logistics", "--out", str(tmp_path / "d.jsonl")]
        )
        assert code == 1
        assert "error" in capsys.readouterr().err


def _write_pilot(path: Path, tamper_first: bool = False) -> None:
    families = load_families(FAMILIES_TOML)
    plan = {f["name"]: [f["eval_seed_range"][0]] for f in families}
    from trillic.sysprompt import build_sysprompt_entries

    entries = build_sysprompt_entries(families, plan)
    if tamper_first:
        entries[0]["prompt"] += " tampered tail"
    path.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
        encoding="utf-8",
    )
