"""`trillic golden *` CLI: validate / manifest / build-rag.

External behavior only: argv in, files + stdout/stderr + exit codes out.
"""

import json

from trillic.cli import main
from trillic.longbench import build_manifest

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
