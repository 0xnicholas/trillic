"""CLI surface for the training line (issue #19): `trillic train run` —
labeled dataset in, verified HF checkpoint + pinned run record out.

Runs on default dependencies: the heavy-deps-absent path is part of the
contract (clear refusal with the bootstrap hint), and dataset/argument
validation happens before the train-environment check. The real
fine-tune path is exercised in the train environment
(test_trainloop.py).
"""

from pathlib import Path

import pytest

from helpers import (
    SIMPLE_PROMPT as SIMPLE,
    SIMPLE_TOKENS,
    training_row as row,
    write_training_dataset as write_dataset,
)
from trillic.cli import main
from trillic.train import train_deps_available


def build_args(dataset: Path, out: Path, *overrides) -> list[str]:
    args = [
        "train", "run",
        "--dataset", str(dataset),
        "--out", str(out),
        "--base-revision", "deadbeef" * 5,
    ]
    args.extend(overrides)
    return args


@pytest.fixture
def dataset(tmp_path) -> Path:
    return write_dataset(tmp_path, [row("a#c0", SIMPLE, SIMPLE_TOKENS)])


class TestTrainRun:
    @pytest.mark.skipif(
        train_deps_available(), reason="absent-deps path needs a torchless env"
    )
    def test_absent_train_env_refuses_with_hint(self, dataset, tmp_path, capsys):
        out = tmp_path / "out"
        assert main(build_args(dataset, out)) == 1
        err = capsys.readouterr().err
        assert "train environment" in err
        assert "scripts/train_pilot.py" in err  # the bootstrap hint

    def test_missing_dataset_fails_before_env_check(self, tmp_path, capsys):
        assert main(build_args(tmp_path / "absent.jsonl", tmp_path / "out")) == 1
        assert "no such labeled dataset" in capsys.readouterr().err

    def test_invalid_dataset_fails_with_row_number(self, tmp_path, capsys):
        bad = row("a#c0", SIMPLE, [["the", 0, 99, 1]])
        path = write_dataset(tmp_path, [bad])
        assert main(build_args(path, tmp_path / "out")) == 1
        assert "line 1" in capsys.readouterr().err

    def test_bad_hyperparams_rejected(self, dataset, tmp_path, capsys):
        assert main(build_args(dataset, tmp_path / "out", "--epochs", "0")) == 1
        assert "epochs" in capsys.readouterr().err

    def test_bad_max_seq_len_rejected(self, dataset, tmp_path, capsys):
        assert main(build_args(dataset, tmp_path / "out", "--max-seq-len", "2")) == 1
        assert "max_seq_len" in capsys.readouterr().err
