"""Parameterized model-native token counting (issue #6).

The compression ratio is reported in two calibers (docs/evaluation.md):
tiktoken (billing) and the compression model's native tokenizer. The
native side must be PARAMETERIZED (issue #1 user story 5) so swapping the
base checkpoint later (mBERT WordPiece -> XLM-R SentencePiece) never
rewrites the harness.

Flavors:
  - word: whitespace words (always available; the stub sidecar's native
    caliber — used by stub-mode demos)
  - wordpiece: BERT-style WordPiece via the `tokenizers` library
    (in-memory vocab or vocab.txt)
  - sentencepiece: Unigram sentencepiece-style via `tokenizers`
    (in-memory piece list or tokenizer.json)

All constructors are offline: vocabularies are in-memory or local files;
nothing is ever downloaded.
"""

from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from trillic.native_tokens import (
    SentencePieceCounter,
    WhitespaceCounter,
    WordPieceCounter,
    build_native_counter,
)

WORDPIECE_VOCAB = {
    "[UNK]": 0,
    "un": 1,
    "##happiness": 2,
    "track": 3,
    "2026": 4,
    "10": 5,
    ".": 6,
    "the": 7,
}

UNIGRAM_PIECES = [
    ("<unk>", 0.0),
    ("▁un", -1.0),
    ("▁happ", -2.0),
    ("iness", -3.0),
    ("▁2026", -1.5),
    ("▁.", -2.5),
    ("▁the", -1.1),
    ("▁track", -1.3),
]


class TestWhitespaceCounter:
    def test_counts_words(self):
        assert WhitespaceCounter().count("un happiness track") == 3

    def test_empty_text_is_zero(self):
        assert WhitespaceCounter().count("") == 0

    def test_caliber_name(self):
        assert WhitespaceCounter().caliber_name == "word"


class TestWordPieceCounter:
    def test_subword_split_changes_counts_vs_words(self):
        """Parameterization must be real: a subword tokenizer counts
        differently from whitespace words on the same text."""
        counter = WordPieceCounter(WORDPIECE_VOCAB)
        assert counter.count("unhappiness track") == 3  # un + ##happiness + track
        assert WhitespaceCounter().count("unhappiness track") == 2

    def test_normalization_lowercases(self):
        counter = WordPieceCounter(WORDPIECE_VOCAB)
        assert counter.count("TRACK") == counter.count("track") == 1

    def test_unknown_characters_map_to_unk_not_crash(self):
        counter = WordPieceCounter(WORDPIECE_VOCAB)
        assert counter.count("π") == 1  # [UNK]

    def test_empty_text_is_zero(self):
        assert WordPieceCounter(WORDPIECE_VOCAB).count("") == 0

    def test_from_vocab_file(self, tmp_path: Path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text(
            "\n".join(sorted(WORDPIECE_VOCAB, key=WORDPIECE_VOCAB.get)) + "\n"
        )
        counter = WordPieceCounter.from_vocab_file(vocab_file)
        assert counter.count("unhappiness track") == 3
        assert counter.caliber_name == "wordpiece"

    def test_deterministic(self):
        first = WordPieceCounter(WORDPIECE_VOCAB)
        second = WordPieceCounter(WORDPIECE_VOCAB)
        text = "the un track 2026 10."
        assert first.count(text) == second.count(text)

    def test_missing_vocab_file_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="vocab"):
            WordPieceCounter.from_vocab_file(tmp_path / "nope.txt")


class TestSentencePieceCounter:
    def test_unigram_split_changes_counts_vs_words(self):
        counter = SentencePieceCounter(UNIGRAM_PIECES)
        # ▁un + happ + iness + ▁2026 + ▁.
        assert counter.count("unhappiness 2026.") == 5
        assert WhitespaceCounter().count("unhappiness 2026.") == 2

    def test_empty_text_is_zero(self):
        assert SentencePieceCounter(UNIGRAM_PIECES).count("") == 0

    def test_unknown_text_maps_to_unk(self):
        counter = SentencePieceCounter(UNIGRAM_PIECES)
        assert counter.count("π") >= 1  # unk pieces, never a crash

    def test_from_tokenizer_json(self, tmp_path: Path):
        tokenizer = Tokenizer(models.Unigram(UNIGRAM_PIECES, unk_id=0))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme="always"
        )
        json_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(json_path), pretty=True)
        counter = SentencePieceCounter.from_tokenizer_json(json_path)
        assert counter.count("unhappiness 2026.") == 5
        assert counter.caliber_name == "sentencepiece"

    def test_missing_tokenizer_file_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="tokenizer"):
            SentencePieceCounter.from_tokenizer_json(tmp_path / "nope.json")

    def test_caliber_name(self):
        assert SentencePieceCounter(UNIGRAM_PIECES).caliber_name == "sentencepiece"


class TestBuildNativeCounter:
    def test_word_flavor_needs_no_vocab(self):
        counter = build_native_counter("word")
        assert counter.caliber_name == "word"
        assert counter.count("a b") == 2

    def test_wordpiece_flavor_uses_vocab_file(self, tmp_path: Path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text(
            "\n".join(sorted(WORDPIECE_VOCAB, key=WORDPIECE_VOCAB.get)) + "\n"
        )
        counter = build_native_counter("wordpiece", vocab_path=vocab_file)
        assert counter.caliber_name == "wordpiece"
        assert counter.count("unhappiness") == 2

    def test_sentencepiece_flavor_uses_tokenizer_json(self, tmp_path: Path):
        tokenizer = Tokenizer(models.Unigram(UNIGRAM_PIECES, unk_id=0))
        json_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(json_path), pretty=True)
        counter = build_native_counter("sentencepiece", vocab_path=json_path)
        assert counter.caliber_name == "sentencepiece"

    def test_unknown_flavor_is_rejected(self):
        with pytest.raises(ValueError, match="native tokenizer flavor"):
            build_native_counter("bytepair")

    def test_subword_flavors_require_a_vocab(self):
        with pytest.raises(ValueError, match="requires a vocab"):
            build_native_counter("wordpiece")
        with pytest.raises(ValueError, match="requires a vocab"):
            build_native_counter("sentencepiece")
