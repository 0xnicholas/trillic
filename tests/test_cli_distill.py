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

    def test_refuses_overwrite(self, corpus_file, tmp_path, capsys):
        assert main(build_args(corpus_file, tmp_path)) == 0
        code = main(build_args(corpus_file, tmp_path))
        assert code == 1
        assert "already exists" in capsys.readouterr().err

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
