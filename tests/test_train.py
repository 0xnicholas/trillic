"""Training plumbing core (issue #19), torchless layer.

Under test: labeled-dataset consumption (issue #18's pilot output),
subword label alignment through fast-tokenizer offsets, window packing,
the pinned run record (seed / hyperparameters / data sha256), and the
versioned-output identity guard. Everything here runs on default
dependencies only — the torch fine-tune loop lives in trillic.trainloop
(env-gated tests in test_trainloop.py).
"""

import json
import re
from pathlib import Path

import pytest

from helpers import (
    SIMPLE_PROMPT,
    char_wordpiece_tokenizer,
    training_row as row,
    training_word as word,
    write_training_dataset as write_dataset,
)
from trillic.train import (
    DEFAULT_MAX_SEQ_LEN,
    PLUMBING_ONLY_CLAIM,
    TrainError,
    align_labels,
    build_examples,
    build_run_record,
    check_output_identity,
    load_labeled_dataset,
    pack_windows,
)


@pytest.fixture
def tokenizer():
    return char_wordpiece_tokenizer()


SIMPLE = SIMPLE_PROMPT
SIMPLE_TOKENS = [
    word(SIMPLE, "the", 1),
    word(SIMPLE, "quick", 0),
    word(SIMPLE, "brown", 1),
    word(SIMPLE, "fox", 0),
]


class TestLoadLabeledDataset:
    def test_roundtrip_and_order(self, tmp_path):
        rows = [row("a#c0", SIMPLE, SIMPLE_TOKENS), row("b#c0", "keep all", [])]
        loaded = load_labeled_dataset(write_dataset(tmp_path, rows))
        assert [r["id"] for r in loaded] == ["a#c0", "b#c0"]
        assert loaded[0]["prompt"] == SIMPLE
        assert loaded[0]["tokens"] == SIMPLE_TOKENS

    def test_missing_file(self, tmp_path):
        with pytest.raises(TrainError, match="no such labeled dataset"):
            load_labeled_dataset(tmp_path / "absent.jsonl")

    def test_bad_json_row(self, tmp_path):
        path = tmp_path / "labeled.jsonl"
        good = json.dumps(row("a#c0", SIMPLE, SIMPLE_TOKENS))
        path.write_text(f"{good}\nnot json\n", encoding="utf-8")
        with pytest.raises(TrainError, match="line 2"):
            load_labeled_dataset(path)

    def test_missing_fields(self, tmp_path):
        bad = row("a#c0", SIMPLE, SIMPLE_TOKENS)
        del bad["prompt"]
        with pytest.raises(TrainError, match="prompt"):
            load_labeled_dataset(write_dataset(tmp_path, [bad]))

    def test_span_out_of_bounds(self, tmp_path):
        tokens = [["the", 0, 99, 1]]
        with pytest.raises(TrainError, match="span"):
            load_labeled_dataset(write_dataset(tmp_path, [row("a#c0", "the", tokens)]))

    def test_bad_label(self, tmp_path):
        tokens = [["the", 0, 3, 2]]
        with pytest.raises(TrainError, match="label"):
            load_labeled_dataset(write_dataset(tmp_path, [row("a#c0", "the", tokens)]))

    def test_empty_dataset(self, tmp_path):
        with pytest.raises(TrainError, match="zero rows"):
            load_labeled_dataset(write_dataset(tmp_path, []))


class TestAlignLabels:
    def test_subwords_inherit_word_labels(self, tokenizer):
        aligned = align_labels(SIMPLE, SIMPLE_TOKENS, tokenizer)
        # every subword overlapping a keep-word carries 1, every
        # drop-word subword 0 (word spans are disjoint, so the overlap
        # a subword sees is unique)
        for tok in aligned:
            covered = [
                keep for _, s, e, keep in SIMPLE_TOKENS
                if tok.start < e and s < tok.end
            ]
            expected = covered[0] if covered else 0
            assert tok.label == expected, (tok, covered)

    def test_comma_unlabeled_maps_to_drop(self, tokenizer):
        text = "keep, drop"
        words = [word(text, "keep", 1), word(text, "drop", 0)]
        aligned = align_labels(text, words, tokenizer)
        comma = [t for t in aligned if t.start == 4]
        assert comma and comma[0].label == 0  # unlabeled chars -> drop

    def test_offsets_are_exact(self, tokenizer):
        aligned = align_labels(SIMPLE, SIMPLE_TOKENS, tokenizer)
        assert aligned
        for tok in aligned:
            if tok.text == "[UNK]":
                continue  # unk pieces cover their source span verbatim
            # "##" is a WordPiece marker, not source text
            assert SIMPLE[tok.start : tok.end] == tok.text.removeprefix("##")
            assert tok.input_id >= 0

    def test_all_keep(self, tokenizer):
        text = "the brown"
        words = [word(text, "the", 1), word(text, "brown", 1)]
        assert all(t.label == 1 for t in align_labels(text, words, tokenizer))

    def test_all_drop(self, tokenizer):
        text = "the brown"
        words = [word(text, "the", 0), word(text, "brown", 0)]
        assert all(t.label == 0 for t in align_labels(text, words, tokenizer))

    def test_unknown_chars_get_unk_token(self, tokenizer):
        text = "the zzzq fox"  # z,q not in vocab (z and q are)
        words = [word(text, "the", 1), word(text, "fox", 1)]
        aligned = align_labels(text, words, tokenizer)
        assert any(t.text == "[UNK]" for t in aligned)


class TestPackWindows:
    def test_short_text_is_one_window_lossless(self, tokenizer):
        windows = pack_windows(align_labels(SIMPLE, SIMPLE_TOKENS, tokenizer), max_seq_len=64)
        assert len(windows) == 1
        ids = [i for w in windows for i in w.input_ids]
        flat = [t.input_id for t in align_labels(SIMPLE, SIMPLE_TOKENS, tokenizer)]
        assert list(ids) == flat  # no specials added at this layer
        assert len(windows[0].labels) == len(windows[0].input_ids)

    def test_long_text_splits_within_cap(self, tokenizer):
        parts = [f"word{chr(97 + i % 10)}" for i in range(200)]
        text = " ".join(parts)
        words = [
            [w, m.start(), m.end(), i % 2]
            for i, (w, m) in enumerate(zip(parts, __import__("re").finditer(r"\S+", text)))
        ]
        aligned = align_labels(text, words, tokenizer)
        windows = pack_windows(aligned, max_seq_len=32)
        assert len(windows) > 1
        assert all(len(w.input_ids) <= 32 for w in windows)
        # lossless: concatenating windows reproduces the aligned sequence
        assert [i for w in windows for i in w.input_ids] == [t.input_id for t in aligned]
        assert [l for w in windows for l in w.labels] == [t.label for t in aligned]

    def test_never_splits_a_word_group(self, tokenizer):
        # "brown"/"fox" split into char subwords; caps must never cut a
        # word's pieces across windows
        text = "the quick brown fox"
        words = [
            word(text, "the", 1), word(text, "quick", 0),
            word(text, "brown", 1), word(text, "fox", 0),
        ]
        aligned = align_labels(text, words, tokenizer)
        windows = pack_windows(aligned, max_seq_len=8)
        assert [i for w in windows for i in w.input_ids] == [t.input_id for t in aligned]
        assert len(windows) > 1  # the cap actually forced splits here
        # no window boundary cuts a word: adjacent windows never share a
        # word owner at the seam (a word's own last "##" piece may end
        # a window — that is the whole group, not a cut)
        for a, b in zip(windows, windows[1:]):
            seam = (a.tokens[-1].word_index, b.tokens[0].word_index)
            assert not (seam[0] >= 0 and seam[0] == seam[1])

    def test_pathological_single_group_truncates(self, tokenizer):
        # one word of 95 char subwords (WordPiece caps words at 100
        # chars — beyond that it is a single [UNK], which always fits)
        text = "brown" * 19
        words = [[text, 0, len(text), 1]]
        aligned = align_labels(text, words, tokenizer)
        assert len(aligned) == 95  # all pieces resolve, no [UNK]
        windows = pack_windows(aligned, max_seq_len=8)
        assert windows
        assert all(len(w.input_ids) <= 8 for w in windows)
        # oversized group is truncated (recorded by the caller's stats)
        assert sum(len(w.input_ids) for w in windows) < len(aligned)

    def test_empty_aligns_to_no_windows(self, tokenizer):
        assert pack_windows([], max_seq_len=16) == []

    def test_bad_max_seq_len(self, tokenizer):
        aligned = align_labels(SIMPLE, SIMPLE_TOKENS, tokenizer)
        with pytest.raises(TrainError, match="max_seq_len"):
            pack_windows(aligned, max_seq_len=2)  # CLS+SEP leave no room


class TestBuildExamples:
    def test_stats_and_counts(self, tokenizer, tmp_path):
        rows = [
            row("a#c0", SIMPLE, SIMPLE_TOKENS, load_type="rag"),
            row("b#c0", "the brown", [word("the brown", "the", 0),
                                      word("the brown", "brown", 1)],
                load_type="dialogue"),
        ]
        built = build_examples(rows, tokenizer, max_seq_len=DEFAULT_MAX_SEQ_LEN)
        assert built["examples"] == 2
        assert built["windows"] == 2  # both fit whole
        assert built["by_load_type"] == {"rag": 1, "dialogue": 1}
        assert 0.0 < built["keep_token_ratio"] < 1.0
        assert len(built["windows_list"]) == built["windows"]
        assert built["truncated_tokens"] == 0

    def test_multi_window_example_counts_once(self, tokenizer):
        text = " ".join(["quick"] * 100)
        words = [[w, i * 6, i * 6 + 5, 1] for i, w in enumerate(text.split())]
        built = build_examples([row("a#c0", text, words)], tokenizer, max_seq_len=16)
        assert built["examples"] == 1
        assert built["windows"] > 1


class TestRunRecord:
    def base_kwargs(self, tmp_path, **overrides):
        rows = [row("a#c0", SIMPLE, SIMPLE_TOKENS)]
        dataset = write_dataset(tmp_path, rows)
        kwargs = dict(
            dataset_path=dataset,
            dataset_rows=1,
            base_model="google-bert/bert-base-multilingual-cased",
            base_revision="deadbeef" * 5,
            seed=19,
            epochs=3,
            batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            max_grad_norm=1.0,
            max_seq_len=DEFAULT_MAX_SEQ_LEN,
            examples=1,
            windows=1,
            by_load_type={"rag": 1},
            keep_token_ratio=0.5,
            truncated_tokens=0,
            loop_record={"steps": 1, "train_loss_by_epoch": [1.0]},
            library={"torch": "2.2.2", "transformers": "4.57.6"},
            checkpoint_digest="a" * 64,
            verify_overall="pass",
            repo_commit="c" * 40,
            repo_dirty=False,
            record_dataset="training/distill/pilot-v1/labeled.jsonl",
        )
        kwargs.update(overrides)
        return kwargs

    def test_pins_every_input(self, tmp_path):
        record = build_run_record(**self.base_kwargs(tmp_path))
        assert record["schema_version"] == 1
        assert record["claim"] == PLUMBING_ONLY_CLAIM
        assert "plumbing" in record["claim"] and "quality" in record["claim"]
        assert record["dataset"]["rows"] == 1
        assert len(record["dataset"]["sha256"]) == 64
        assert record["dataset"]["file"] == "training/distill/pilot-v1/labeled.jsonl"
        assert record["base_model"] == {
            "id": "google-bert/bert-base-multilingual-cased",
            "revision": "deadbeef" * 5,
        }
        hp = record["hyperparameters"]
        assert hp["seed"] == 19 and hp["epochs"] == 3 and hp["batch_size"] == 8
        assert hp["learning_rate"] == 2e-5 and hp["max_seq_len"] == DEFAULT_MAX_SEQ_LEN
        assert hp["optimizer"] == "adamw" and hp["lr_schedule"] == "constant"
        assert record["training"]["examples"] == 1
        assert record["outputs"]["checkpoint_digest"] == "a" * 64
        assert record["outputs"]["delivery_verify"] == "pass"
        # deterministic serialization: same inputs -> identical bytes
        again = build_run_record(**self.base_kwargs(tmp_path))
        assert json.dumps(record, sort_keys=True) == json.dumps(again, sort_keys=True)

    def test_no_timestamps(self, tmp_path):
        # run records are deterministic documents: no wall-clock anywhere
        # (wall_seconds lives in the loop record only if passed; not here)
        record = build_run_record(**self.base_kwargs(tmp_path))
        text = json.dumps(record).lower()
        for key in ("timestamp", "date", "created", "wall_"):
            assert key not in text

    def test_loss_values_kept(self, tmp_path):
        record = build_run_record(**self.base_kwargs(tmp_path))
        assert record["training"]["train_loss_by_epoch"] == [1.0]


class TestOutputIdentity:
    def identity(self, dataset_sha: str, **over) -> dict:
        ident = {
            "dataset": {"sha256": dataset_sha},
            "base_model": {
                "id": "google-bert/bert-base-multilingual-cased",
                "revision": "deadbeef" * 5,
            },
            "hyperparameters": {
                "seed": 19, "epochs": 3, "batch_size": 8,
                "learning_rate": 2e-5, "weight_decay": 0.01,
                "max_grad_norm": 1.0, "max_seq_len": 512,
                "optimizer": "adamw", "lr_schedule": "constant",
                "shuffle_rule": "x",
            },
        }
        for key, value in over.items():
            ident[key] = value
        return ident

    def test_fresh_dir_passes(self, tmp_path):
        check_output_identity(tmp_path / "out", self.identity("a" * 64))  # no error

    def test_same_identity_repins_in_place(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "run-record.json").write_text(
            json.dumps(self.identity("a" * 64)), encoding="utf-8"
        )
        check_output_identity(out, self.identity("a" * 64))  # re-pin allowed

    def test_drift_refuses(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "run-record.json").write_text(
            json.dumps(self.identity("b" * 64)), encoding="utf-8"
        )
        with pytest.raises(TrainError, match="different identity"):
            check_output_identity(out, self.identity("a" * 64))

    def test_hyperparam_drift_refuses(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "run-record.json").write_text(
            json.dumps(self.identity("a" * 64)), encoding="utf-8"
        )
        drifted = self.identity("a" * 64)
        drifted["hyperparameters"] = dict(drifted["hyperparameters"], seed=20)
        with pytest.raises(TrainError, match="seed"):
            check_output_identity(out, drifted)

    def test_unreadable_record_refuses(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "run-record.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(TrainError, match="cannot"):
            check_output_identity(out, self.identity("a" * 64))


class TestRunTrainingOrchestration:
    """The full plumbing with the torch loop / verify / hub injected —
    the real path is exercised in the train environment."""

    @pytest.fixture
    def fake_loop(self, tmp_path):
        calls: list[dict] = []

        def loop(*, windows, tokenizer, base_model, base_revision,
                 checkpoint_dir, hyperparameters):
            calls.append(
                {"windows": len(windows), "hp": dict(hyperparameters),
                 "checkpoint_dir": Path(checkpoint_dir)}
            )
            import shutil
            from helpers import make_checkpoint
            ckpt = Path(checkpoint_dir)
            if ckpt.exists():
                shutil.rmtree(ckpt)  # save_pretrained semantics: overwrite
            make_checkpoint(ckpt.parent, ckpt.name)
            return {
                "steps": 3,
                "train_loss_by_epoch": [1.0, 0.5, 0.25],
                "library": {"torch": "9.9", "transformers": "9.9"},
            }

        loop.calls = calls  # type: ignore[attr-defined]
        return loop

    @staticmethod
    def fake_verify(report_overall="pass"):
        def verify(directory):
            return {"overall": report_overall, "checkpoint": {"path": str(directory)}}
        return verify

    def run(self, tmp_path, loop, **over):
        dataset = write_dataset(
            tmp_path, [row("a#c0", SIMPLE, SIMPLE_TOKENS)]
        )
        kwargs = dict(
            dataset_path=dataset,
            out_dir=tmp_path / "out",
            base_model="fixture/mbert",
            base_revision="deadbeef" * 5,
            seed=19, epochs=3, batch_size=8, learning_rate=2e-5,
            weight_decay=0.01, max_grad_norm=1.0, max_seq_len=DEFAULT_MAX_SEQ_LEN,
            repo_commit="c" * 40, repo_dirty=False,
            record_dataset="training/distill/pilot-v1/labeled.jsonl",
            loop=loop,
            verify_fn=self.fake_verify(),
            tokenizer=char_wordpiece_tokenizer(),
        )
        kwargs.update(over)
        from trillic.train import run_training
        return run_training(**kwargs)

    def test_end_to_end_record_and_reports(self, tmp_path, fake_loop):
        record = self.run(tmp_path, fake_loop)
        out = tmp_path / "out"
        assert (out / "run-record.json").is_file()
        written = json.loads((out / "run-record.json").read_text())
        assert written == record  # file is the record, byte-deterministic
        assert (out / "verify-report.json").is_file()
        assert (out / "checkpoint" / "config.json").is_file()
        assert record["claim"] == PLUMBING_ONLY_CLAIM
        assert record["dataset"]["rows"] == 1
        assert record["outputs"]["delivery_verify"] == "pass"
        assert len(record["outputs"]["checkpoint_digest"]) == 64
        # the loop saw the pinned hyperparameters and the windows
        assert fake_loop.calls[0]["hp"]["seed"] == 19
        assert fake_loop.calls[0]["windows"] == record["training"]["windows"]
        assert fake_loop.calls[0]["checkpoint_dir"] == out / "checkpoint"

    def test_verify_violation_refuses_and_keeps_evidence(self, tmp_path, fake_loop):
        from trillic.train import run_training
        dataset = write_dataset(tmp_path, [row("a#c0", SIMPLE, SIMPLE_TOKENS)])
        with pytest.raises(TrainError, match="delivery verify"):
            self.run(tmp_path, fake_loop, verify_fn=self.fake_verify("violation"))
        out = tmp_path / "out"
        assert (out / "verify-report.json").is_file()  # evidence preserved
        assert not (out / "run-record.json").is_file()  # no successful record

    def test_identity_drift_refuses_before_training(self, tmp_path, fake_loop):
        out = tmp_path / "out"
        self.run(tmp_path, fake_loop)  # first run writes run-record.json
        with pytest.raises(TrainError, match="different identity"):
            self.run(tmp_path, fake_loop, seed=20)
        assert len(fake_loop.calls) == 1  # the second run never trained

    def test_same_identity_rerun_repins(self, tmp_path, fake_loop):
        first = self.run(tmp_path, fake_loop)
        second = self.run(tmp_path, fake_loop)
        assert first == second  # deterministic record, overwrite in place
        assert len(fake_loop.calls) == 2
