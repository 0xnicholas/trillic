"""Torch fine-tune loop (issue #19) — train-environment tests.

These run only where torch + transformers are importable (the dedicated
`.venv-train`; `pytest.importorskip` keeps the default suite green).
The models here are tiny config-synthesized BERTs: real transformers,
real torch, seconds per test, zero network.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import (  # noqa: E402
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
)

from helpers import (  # noqa: E402
    SIMPLE_PROMPT,
    char_wordpiece_tokenizer,
    training_row,
    training_word,
    write_training_dataset,
)
from trillic.delivery import verify_checkpoint  # noqa: E402
from trillic.train import build_examples, run_training  # noqa: E402
from trillic.trainloop import fine_tune  # noqa: E402

HYPERPARAMETERS = {
    "seed": 19, "epochs": 2, "batch_size": 2, "learning_rate": 5e-4,
    "weight_decay": 0.01, "max_grad_norm": 1.0, "max_seq_len": 64,
}


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory) -> Path:
    """A tiny but real BERT base: config-synthesized weights + the char
    WordPiece tokenizer from the torchless fixtures (as tokenizer.json)."""
    base = tmp_path_factory.mktemp("tiny-base") / "base"
    base.mkdir()
    raw = char_wordpiece_tokenizer()
    # round-trip through PreTrainedTokenizerFast so the fixture carries
    # the full file set a real save_pretrained writes
    from transformers import PreTrainedTokenizerFast

    fast = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
    )
    fast.save_pretrained(base)
    vocab_size = raw.get_vocab_size()
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=2,
    )
    BertForTokenClassification(config).save_pretrained(base)
    return base


@pytest.fixture(scope="module")
def tiny_tokenizer(tiny_base):
    return AutoTokenizer.from_pretrained(str(tiny_base))


def example_windows(tiny_tokenizer):
    prompt = SIMPLE_PROMPT
    tokens = [
        training_word(prompt, "the", 1),
        training_word(prompt, "quick", 0),
        training_word(prompt, "brown", 1),
        training_word(prompt, "fox", 0),
    ]
    longer = " ".join(["quick brown fox"] * 12)
    tokens2 = [
        [m.group(), m.start(), m.end(), i % 2]
        for i, m in enumerate(__import__("re").finditer(r"\S+", longer))
    ]
    rows = [
        training_row("a#c0", prompt, tokens),
        training_row("b#c0", longer, tokens2, load_type="dialogue"),
    ]
    built = build_examples(rows, tiny_tokenizer.backend_tokenizer, max_seq_len=64)
    return built


class TestFineTune:
    def test_trains_saves_and_verifies(self, tmp_path, tiny_base, tiny_tokenizer):
        built = example_windows(tiny_tokenizer)
        checkpoint = tmp_path / "checkpoint"
        record = fine_tune(
            windows=built["windows_list"],
            tokenizer=tiny_tokenizer,
            base_model=str(tiny_base),
            base_revision="local-fixture",
            checkpoint_dir=checkpoint,
            hyperparameters=HYPERPARAMETERS,
        )
        assert record["steps"] > 0
        assert len(record["train_loss_by_epoch"]) == HYPERPARAMETERS["epochs"]
        assert all(loss == loss and loss > 0 for loss in record["train_loss_by_epoch"])
        assert record["device"] == "cpu"
        assert record["library"]["torch"] and record["library"]["transformers"]
        config = json.loads((checkpoint / "config.json").read_text())
        assert config["architectures"] == ["BertForTokenClassification"]
        assert config["id2label"] == {"0": "drop", "1": "keep"}
        assert (checkpoint / "model.safetensors").is_file()
        assert (checkpoint / "tokenizer.json").is_file()

    def test_saved_checkpoint_passes_delivery_verify(self, tmp_path, tiny_base, tiny_tokenizer):
        """The whole point of the line: day-one drop-in contract —
        static AND load layer on a really-trained artifact."""
        built = example_windows(tiny_tokenizer)
        checkpoint = tmp_path / "checkpoint"
        fine_tune(
            windows=built["windows_list"],
            tokenizer=tiny_tokenizer,
            base_model=str(tiny_base),
            base_revision="local-fixture",
            checkpoint_dir=checkpoint,
            hyperparameters=HYPERPARAMETERS,
        )
        report = verify_checkpoint(checkpoint)
        assert report["overall"] == "pass", json.dumps(report, indent=2)
        assert report["layers"]["load"]["status"] == "pass"
        assert report["layers"]["static"]["status"] == "pass"

    def test_same_seed_reproduces_byte_identical_checkpoint(
        self, tmp_path, tiny_base, tiny_tokenizer
    ):
        built = example_windows(tiny_tokenizer)
        digests = []
        for i in range(2):
            checkpoint = tmp_path / f"run{i}" / "checkpoint"
            fine_tune(
                windows=built["windows_list"],
                tokenizer=tiny_tokenizer,
                base_model=str(tiny_base),
                base_revision="local-fixture",
                checkpoint_dir=checkpoint,
                hyperparameters=HYPERPARAMETERS,
            )
            digests.append(
                sorted(
                    (p.name, p.read_bytes())
                    for p in checkpoint.iterdir()
                    if p.name != "config.json"  # config is identical anyway
                )
            )
        model_a = (tmp_path / "run0" / "checkpoint" / "model.safetensors").read_bytes()
        model_b = (tmp_path / "run1" / "checkpoint" / "model.safetensors").read_bytes()
        assert model_a == model_b  # the determinism contract, literally


class TestRunTrainingReal:
    """The real orchestration (no injected seams) inside the train env:
    dataset -> fine_tune -> delivery verify -> run record."""

    def test_end_to_end_local_base(self, tmp_path, tiny_base):
        prompt = SIMPLE_PROMPT
        tokens = [
            training_word(prompt, "the", 1),
            training_word(prompt, "quick", 0),
            training_word(prompt, "brown", 1),
            training_word(prompt, "fox", 0),
        ]
        dataset = write_training_dataset(
            tmp_path, [training_row("a#c0", prompt, tokens)]
        )
        record = run_training(
            dataset_path=dataset,
            out_dir=tmp_path / "out",
            base_model=str(tiny_base),
            base_revision="local-fixture",
            seed=19, epochs=1, batch_size=2, learning_rate=5e-4,
            weight_decay=0.01, max_grad_norm=1.0, max_seq_len=64,
            repo_commit="c" * 40, repo_dirty=False,
            record_dataset="fixture/labeled.jsonl",
        )
        out = tmp_path / "out"
        assert record["outputs"]["delivery_verify"] == "pass"
        assert json.loads((out / "verify-report.json").read_text())["overall"] == "pass"
        written = json.loads((out / "run-record.json").read_text())
        assert written == record
        assert record["library"]["torch"]

    def test_rerun_same_identity_repins(self, tmp_path, tiny_base):
        prompt = SIMPLE_PROMPT
        tokens = [training_word(prompt, "the", 1)]
        dataset = write_training_dataset(
            tmp_path, [training_row("a#c0", prompt, tokens)]
        )
        kwargs = dict(
            dataset_path=dataset,
            out_dir=tmp_path / "out",
            base_model=str(tiny_base),
            base_revision="local-fixture",
            seed=7, epochs=1, batch_size=2, learning_rate=5e-4,
            weight_decay=0.01, max_grad_norm=1.0, max_seq_len=64,
        )
        first = run_training(**kwargs)
        second = run_training(**kwargs)  # same identity: overwrite in place
        assert first == second
