"""CLI surface for the synthetic training layers (issue #17):
`trillic corpus build-synthetic` + `trillic corpus validate-synthetic`."""

import json
from pathlib import Path

import pytest

from trillic.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
SYS_TOML = REPO_ROOT / "eval" / "manifests" / "sysprompt_families.toml"
DLG_TOML = REPO_ROOT / "eval" / "manifests" / "dialogue_families.toml"
GOLDEN_SYNTHETIC = [
    REPO_ROOT / "eval" / "golden" / "sysprompt_pilot.jsonl",
    REPO_ROOT / "eval" / "golden" / "sysprompt_scaled.jsonl",
    REPO_ROOT / "eval" / "golden" / "dialogue_pilot.jsonl",
    REPO_ROOT / "eval" / "golden" / "dialogue_scaled.jsonl",
]


def build_args(tmp_path: Path, *, targets=("8", "8"), golden=True) -> list[str]:
    golden_args = ["--golden", *[str(p) for p in GOLDEN_SYNTHETIC]] if golden else []
    return [
        "corpus", "build-synthetic",
        "--families-sysprompt", str(SYS_TOML),
        "--families-dialogue", str(DLG_TOML),
        *golden_args,
        "--target-sysprompt", targets[0],
        "--target-dialogue", targets[1],
        "--out", str(tmp_path / "synthetic-train.jsonl"),
        "--out-manifest", str(tmp_path / "synthetic-training.json"),
    ]


class TestBuildSynthetic:
    def test_build_writes_corpus_and_manifest(self, tmp_path, capsys):
        assert main(build_args(tmp_path)) == 0
        corpus_path = tmp_path / "synthetic-train.jsonl"
        manifest = json.loads(
            (tmp_path / "synthetic-training.json").read_text(encoding="utf-8")
        )
        rows = [
            json.loads(line)
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 16
        assert manifest["corpus"]["entries"] == 16
        assert "16 entries" in capsys.readouterr().out

    def test_build_records_repo_commit(self, tmp_path):
        args = build_args(tmp_path)
        args += ["--repo-commit", "deadbeef", "--repo-dirty"]
        assert main(args) == 0
        manifest = json.loads(
            (tmp_path / "synthetic-training.json").read_text(encoding="utf-8")
        )
        assert manifest["generated_by"] == {
            "tool": "trillic.synthetic",
            "repo_commit": "deadbeef",
            "repo_dirty": True,
        }

    def test_build_without_golden_warns(self, tmp_path, capsys):
        assert main(build_args(tmp_path, golden=False)) == 0
        assert "--golden" in capsys.readouterr().err

    def test_build_refuses_unreachable_target(self, tmp_path, capsys):
        code = main(build_args(tmp_path, targets=("100000", "8")))
        assert code == 1
        assert "unreachable" in capsys.readouterr().err

    def test_build_requires_target_flags(self, tmp_path):
        args = build_args(tmp_path)
        args.remove("--target-sysprompt")
        args.remove("8")
        with pytest.raises(SystemExit):
            main(args)


class TestValidateSynthetic:
    def test_validate_clean_roundtrip(self, tmp_path, capsys):
        assert main(build_args(tmp_path)) == 0
        code = main(
            [
                "corpus", "validate-synthetic",
                str(tmp_path / "synthetic-train.jsonl"),
                "--families-sysprompt", str(SYS_TOML),
                "--families-dialogue", str(DLG_TOML),
                "--training-manifest", str(tmp_path / "synthetic-training.json"),
                "--golden", *[str(p) for p in GOLDEN_SYNTHETIC],
            ]
        )
        assert code == 0
        assert "ok (16 entries)" in capsys.readouterr().out

    def test_validate_without_golden_warns(self, tmp_path, capsys):
        assert main(build_args(tmp_path)) == 0
        code = main(
            [
                "corpus", "validate-synthetic",
                str(tmp_path / "synthetic-train.jsonl"),
                "--families-sysprompt", str(SYS_TOML),
                "--families-dialogue", str(DLG_TOML),
            ]
        )
        assert code == 0
        assert "--golden" in capsys.readouterr().err

    def test_validate_rejects_tampered_corpus(self, tmp_path, capsys):
        assert main(build_args(tmp_path)) == 0
        corpus_path = tmp_path / "synthetic-train.jsonl"
        rows = [
            json.loads(line)
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows[0]["prompt"] += "\nhand-edited"
        corpus_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
        code = main(
            [
                "corpus", "validate-synthetic",
                str(corpus_path),
                "--families-sysprompt", str(SYS_TOML),
                "--families-dialogue", str(DLG_TOML),
                "--training-manifest", str(tmp_path / "synthetic-training.json"),
            ]
        )
        assert code == 1
        stderr = capsys.readouterr().err
        assert "regeneration" in stderr
        assert "sha256" in stderr
