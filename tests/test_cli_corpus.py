"""`trillic corpus *` CLI: build (train-half extraction + manifest) and
validate (schema / MeetingBank / train half / registry / golden overlap).

External behavior only: argv in, files + stdout/stderr + exit codes out.
"""

import json
from pathlib import Path

from trillic.cli import main
from trillic.longbench import build_manifest
from trillic.splits import row_fingerprint, split_half_rows
from trillic.longbench import load_subset_rows

GOLDEN_SOURCE = {
    "dataset": "handwritten",
    "subset": "test",
    "license": "original",
    "split": "eval",
}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_golden_entry(entry_id, fingerprint):
    return {
        "id": entry_id,
        "load_type": "rag",
        "prompt": "p",
        "key_points": ["k"],
        "source": {**GOLDEN_SOURCE, "content_sha1": fingerprint},
    }


class TestCorpusBuild:
    def test_build_writes_corpus_and_manifest(self, subsets_toml, data_dir, tmp_path, capsys):
        lb = build_manifest(subsets_toml, data_dir)
        lb_path = tmp_path / "longbench.json"
        lb_path.write_text(json.dumps(lb), encoding="utf-8")
        corpus = tmp_path / "corpus.jsonl"
        manifest_out = tmp_path / "training.json"

        assert (
            main(
                [
                    "corpus", "build",
                    "--manifest", str(lb_path),
                    "--data-dir", str(data_dir),
                    "--out", str(corpus),
                    "--out-manifest", str(manifest_out),
                    "--repo-commit", "abc1234",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "15 entries" in out and "excluded: qasper" in out

        entries = [json.loads(l) for l in corpus.read_text().splitlines() if l.strip()]
        assert len(entries) == 15
        training = json.loads(manifest_out.read_text())
        import hashlib

        assert training["corpus"]["sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
        assert training["generated_by"]["repo_commit"] == "abc1234"

    def test_build_drifted_data_fails(self, subsets_toml, data_dir, tmp_path, capsys):
        lb = build_manifest(subsets_toml, data_dir)
        lb_path = tmp_path / "longbench.json"
        lb_path.write_text(json.dumps(lb), encoding="utf-8")
        target = data_dir / "hotpotqa_e.jsonl"
        target.write_text(
            target.read_text()
            + json.dumps({"input": "q?", "context": "c", "answers": ["a"]}) + "\n"
        )
        code = main(
            [
                "corpus", "build",
                "--manifest", str(lb_path),
                "--data-dir", str(data_dir),
                "--out", str(tmp_path / "c.jsonl"),
                "--out-manifest", str(tmp_path / "m.json"),
            ]
        )
        assert code == 1
        assert "regenerate" in capsys.readouterr().err


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_CORPUS = REPO_ROOT / "training" / "corpus" / "longbench-train-v1.jsonl"
COMMITTED_MANIFEST = REPO_ROOT / "training" / "manifests" / "training-corpus-v1.json"


def frozen_golden_paths() -> list[Path]:
    """The frozen golden file list — single-sourced from the freeze record
    (same authority scripts/build_training_corpus.py reads)."""
    freeze = json.loads(
        (REPO_ROOT / "eval" / "manifests" / "golden-freeze.json").read_text(
            encoding="utf-8"
        )
    )
    return [REPO_ROOT / f for f in freeze["golden"]["files"]]


class TestCommittedArtifacts:
    def test_frozen_corpus_validates_against_manifest_and_golden(self, capsys):
        """Repo invariant (issue #16): the committed base layer always passes
        the full validator — schema, MeetingBank, offline train-boundary
        re-proof, frozen-bytes sha256, and zero overlap with the exam."""
        argv = [
            "corpus", "validate", str(COMMITTED_CORPUS),
            "--training-manifest", str(COMMITTED_MANIFEST),
            "--golden", *[str(p) for p in frozen_golden_paths()],
        ]
        assert main(argv) == 0, capsys.readouterr().err
        out = capsys.readouterr().out
        assert "ok (300 entries)" in out and "zero overlap" in out

    def test_frozen_corpus_question_task_empty_in_v1(self):
        for line in COMMITTED_CORPUS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                assert row["question"] == "" and row["task"] == ""


class TestCorpusValidate:
    def test_roundtrip_validates_with_manifest_and_golden(
        self, subsets_toml, data_dir, tmp_path, capsys
    ):
        lb = build_manifest(subsets_toml, data_dir)
        lb_path = tmp_path / "longbench.json"
        lb_path.write_text(json.dumps(lb), encoding="utf-8")
        corpus = tmp_path / "corpus.jsonl"
        manifest_out = tmp_path / "training.json"
        assert (
            main(
                [
                    "corpus", "build",
                    "--manifest", str(lb_path),
                    "--data-dir", str(data_dir),
                    "--out", str(corpus),
                    "--out-manifest", str(manifest_out),
                ]
            )
            == 0
        )
        # golden drawn from the EVAL half of the same data -> zero overlap
        eval_rows, _ = split_half_rows(load_subset_rows(data_dir / "hotpotqa_e.jsonl"))
        golden = tmp_path / "golden.jsonl"
        write_jsonl(
            golden,
            [
                make_golden_entry(f"rag-hotpotqa-{row_fingerprint(r)[:8]}", row_fingerprint(r))
                for r in eval_rows
            ],
        )
        assert (
            main(
                [
                    "corpus", "validate", str(corpus),
                    "--training-manifest", str(manifest_out),
                    "--golden", str(golden),
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "ok (15 entries)" in out and "zero overlap" in out

    def test_overlap_with_golden_fails(self, subsets_toml, data_dir, tmp_path, capsys):
        lb = build_manifest(subsets_toml, data_dir)
        lb_path = tmp_path / "longbench.json"
        lb_path.write_text(json.dumps(lb), encoding="utf-8")
        corpus = tmp_path / "corpus.jsonl"
        assert (
            main(
                [
                    "corpus", "build",
                    "--manifest", str(lb_path),
                    "--data-dir", str(data_dir),
                    "--out", str(corpus),
                    "--out-manifest", str(tmp_path / "m.json"),
                ]
            )
            == 0
        )
        first = json.loads(corpus.read_text().splitlines()[0])
        poisoned_golden = tmp_path / "golden.jsonl"
        write_jsonl(
            poisoned_golden,
            [make_golden_entry("rag-leaked-00000000", first["source"]["content_sha1"])],
        )
        assert (
            main(["corpus", "validate", str(corpus), "--golden", str(poisoned_golden)]) == 1
        )
        err = capsys.readouterr().err
        assert "overlap" in err and first["id"] in err and "rag-leaked-00000000" in err

    def test_schema_error_names_entry_id(self, tmp_path, capsys):
        corpus = tmp_path / "corpus.jsonl"
        write_jsonl(
            corpus,
            [
                {
                    "id": "train-x-00000000",
                    "load_type": "rag",
                    "prompt": "p",
                    "question": "",
                    # task missing
                    "source": {
                        "dataset": "LongBench",
                        "subset": "x",
                        "license": "l",
                        "split": "train",
                        "content_sha1": "0" * 40,
                    },
                }
            ],
        )
        assert main(["corpus", "validate", str(corpus)]) == 1
        assert "train-x-00000000" in capsys.readouterr().err

    def test_meetingbank_hard_rejected(self, tmp_path, capsys):
        corpus = tmp_path / "corpus.jsonl"
        write_jsonl(
            corpus,
            [
                {
                    "id": "train-mb-00000000",
                    "load_type": "rag",
                    "prompt": "p",
                    "question": "",
                    "task": "",
                    "source": {
                        "dataset": "MeetingBank",
                        "subset": "whatever",
                        "license": "l",
                        "split": "train",
                        "content_sha1": "0" * 40,
                    },
                }
            ],
        )
        assert main(["corpus", "validate", str(corpus)]) == 1
        assert "MeetingBank is excluded" in capsys.readouterr().err

    def test_note_printed_when_golden_absent(self, tmp_path, capsys):
        corpus = tmp_path / "corpus.jsonl"
        write_jsonl(
            corpus,
            [
                {
                    "id": "train-x-00000000",
                    "load_type": "rag",
                    "prompt": "p",
                    "question": "",
                    "task": "",
                    "source": {
                        "dataset": "LongBench",
                        "subset": "x",
                        "license": "l",
                        "split": "train",
                        "content_sha1": "1" * 40,
                    },
                }
            ],
        )
        assert main(["corpus", "validate", str(corpus)]) == 0
        captured = capsys.readouterr()
        assert "zero-overlap assertion is skipped" in captured.err
        assert "ok (1 entries)" in captured.out
