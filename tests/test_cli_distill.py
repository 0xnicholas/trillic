"""CLI surface for the teacher distillation pipeline (issue #18):
`trillic distill run` — corpus manifest in, labeled dataset + QC report out."""

import json
from pathlib import Path

import pytest

from trillic.cli import main

CORPUS_ROWS = [
    {
        "id": f"rag-{i}",
        "load_type": "rag",
        "prompt": (
            " ".join(
                f"Fixture sentence {j} with number {j * 3} embedded."
                for j in range(6)
            )
        ),
        "question": "",
        "task": "",
        "source": {
            "dataset": "fixture",
            "subset": "test",
            "license": "original",
            "split": "train",
            "content_sha1": f"rag-{i}",
        },
    }
    for i in range(3)
] + [
    {
        "id": f"sys-{i}",
        "load_type": "system_prompt",
        "prompt": f"System prompt {i}. Be terse and precise in replies.",
        "question": "",
        "task": "",
        "source": {
            "dataset": "fixture",
            "subset": "test",
            "license": "original",
            "split": "train",
            "content_sha1": f"sys-{i}",
        },
    }
    for i in range(2)
]


@pytest.fixture
def corpus_file(tmp_path) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in CORPUS_ROWS),
        encoding="utf-8",
    )
    return path


def build_args(corpus_file: Path, tmp_path: Path, *overrides) -> list[str]:
    args = [
        "distill", "run",
        "--corpus", str(corpus_file),
        "--chunk-budget", "rag=3,system_prompt=2",
        "--teacher-model", "stub-teacher",
        "--out", str(tmp_path / "out"),
        "--ledger-root", str(tmp_path / "ledger"),
        "--vr-drop-fraction", "0.0",
        "--ag-drop-fraction", "0.0",
    ]
    args.extend(overrides)
    return args


class TestDistillRun:
    def test_stub_end_to_end(self, corpus_file, tmp_path, capsys):
        assert main(build_args(corpus_file, tmp_path)) == 0
        out = tmp_path / "out"
        for name in ("labeled.jsonl", "report.json", "report.md", "manifest.json"):
            assert (out / name).is_file(), name
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["teacher"]["model"] == "stub-teacher"
        assert manifest["gateway"]["calls_fresh"] == manifest["counts"]["chunks"]
        assert manifest["counts"]["labeled_kept"] == manifest["counts"]["chunks"]
        rows = [
            json.loads(line)
            for line in (out / "labeled.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows and all(r["tokens"] for r in rows)
        assert "labeled" in capsys.readouterr().out

    def test_rerun_replays_journal_zero_calls(self, corpus_file, tmp_path, capsys):
        assert main(build_args(corpus_file, tmp_path)) == 0
        # same identity into a fresh out dir: everything replays
        args = build_args(corpus_file, tmp_path, "--out", str(tmp_path / "out2"))
        assert main(args) == 0
        manifest = json.loads((tmp_path / "out2" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["gateway"]["calls_fresh"] == 0
        assert manifest["gateway"]["calls_reused"] == manifest["counts"]["chunks"]
        assert "reused" in capsys.readouterr().out

    def test_same_identity_rerun_repins_in_place(self, corpus_file, tmp_path, capsys):
        # same content-determining pins: regeneration is journal-backed and
        # byte-deterministic, so overwriting in place is the re-pin flow
        assert main(build_args(corpus_file, tmp_path)) == 0
        first = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
        assert main(build_args(corpus_file, tmp_path)) == 0
        second = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
        assert second["gateway"]["calls_fresh"] == 0
        assert second["gateway"]["calls_reused"] == second["counts"]["chunks"]
        assert first["outputs"]["labeled"]["sha256"] == second["outputs"]["labeled"]["sha256"]

    def test_identity_drift_refuses_overwrite(self, corpus_file, tmp_path, capsys):
        assert main(build_args(corpus_file, tmp_path)) == 0
        code = main(
            build_args(corpus_file, tmp_path, "--teacher-model", "other/teacher")
        )
        assert code == 1
        assert "different identity" in capsys.readouterr().err

    def test_unknown_budget_class_fails(self, corpus_file, tmp_path, capsys):
        code = main(
            build_args(corpus_file, tmp_path, "--chunk-budget", "ghost=1")
        )
        assert code == 1
        assert "not present in the corpus" in capsys.readouterr().err

    def test_bad_budget_spec_fails(self, corpus_file, tmp_path, capsys):
        code = main(build_args(corpus_file, tmp_path, "--chunk-budget", "rag"))
        assert code == 1
        assert "--chunk-budget" in capsys.readouterr().err

    def test_missing_corpus_fails(self, tmp_path, capsys):
        code = main(
            build_args(tmp_path / "nope.jsonl", tmp_path)
        )
        assert code == 1
        assert "no such corpus file" in capsys.readouterr().err

    def test_repo_commit_recorded(self, corpus_file, tmp_path):
        args = build_args(corpus_file, tmp_path, "--repo-commit", "deadbeef", "--repo-dirty")
        assert main(args) == 0
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["generated_by"] == {
            "tool": "trillic.distill",
            "repo_commit": "deadbeef",
            "repo_dirty": True,
        }


class TestFrozenManifestGate:
    def test_drift_refuses_before_any_output(self, corpus_file, tmp_path, capsys):
        manifest = {
            "corpus": {"sha256": "0" * 64, "entries": len(CORPUS_ROWS)}
        }
        manifest_path = tmp_path / "frozen.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        code = main(
            build_args(corpus_file, tmp_path, "--training-manifest", str(manifest_path))
        )
        assert code == 1
        assert "does not match its training manifest" in capsys.readouterr().err
        assert not (tmp_path / "out").exists()

    def test_count_mismatch_rejected(self, corpus_file, tmp_path, capsys):
        code = main(
            build_args(
                corpus_file, tmp_path,
                "--corpus", str(corpus_file), str(corpus_file),
                "--training-manifest", str(corpus_file),
            )
        )
        assert code == 1
        assert "one file per --corpus entry" in capsys.readouterr().err

    def test_matching_sha_passes(self, corpus_file, tmp_path):
        import hashlib

        manifest = {
            "corpus": {
                "sha256": hashlib.sha256(corpus_file.read_bytes()).hexdigest(),
                "entries": len(CORPUS_ROWS),
            }
        }
        manifest_path = tmp_path / "frozen.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert main(
            build_args(corpus_file, tmp_path, "--training-manifest", str(manifest_path))
        ) == 0


class TestRecordPaths:
    def test_repo_relative_identities_recorded(self, corpus_file, tmp_path):
        assert main(
            build_args(
                corpus_file, tmp_path,
                "--record-corpus", "training/corpus/fixture.jsonl",
                "--record-journal", "runs/.ledger/distill-fixture.jsonl",
            )
        ) == 0
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["corpus"]["files"] == [
            {
                "file": "training/corpus/fixture.jsonl",
                "sha256": manifest["corpus"]["files"][0]["sha256"],
                "entries": manifest["corpus"]["files"][0]["entries"],
            }
        ]
        assert manifest["gateway"]["journal"] == "runs/.ledger/distill-fixture.jsonl"

    def test_record_count_mismatch_rejected(self, corpus_file, tmp_path, capsys):
        code = main(
            build_args(corpus_file, tmp_path, "--record-corpus", "a.jsonl", "b.jsonl")
        )
        assert code == 1
        assert "one identity per --corpus entry" in capsys.readouterr().err
