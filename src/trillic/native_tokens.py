"""Parameterized model-native token counting (issue #6).

The compression ratio is reported in two calibers (docs/evaluation.md):
tiktoken (billing caliber) and the compression model's native tokenizer.
The native side is PARAMETERIZED by flavor (issue #1 user story 5) so
swapping the base checkpoint later (mBERT WordPiece -> XLM-R
SentencePiece) means pointing the config at another vocabulary, never
rewriting the harness:

  - word: whitespace words. Always available; also the stub sidecar's
    native caliber, so stub-mode sweeps exercise the dual-caliber report
    end to end.
  - wordpiece: BERT-style WordPiece via the `tokenizers` library, from an
    in-memory vocab dict or a vocab.txt file.
  - sentencepiece: Unigram sentencepiece-style via `tokenizers`, from an
    in-memory piece list or a serialized tokenizer.json.

All construction is offline and deterministic: vocabularies are in-memory
or local files; nothing is downloaded. The `tokenizers` import is deferred
to the classes that need it.
"""

from pathlib import Path
from typing import Protocol


class NativeCounter(Protocol):
    """A token-counting caliber."""

    caliber_name: str

    def count(self, text: str) -> int: ...


class WhitespaceCounter:
    """Whitespace-word caliber (the stub sidecar's native count)."""

    caliber_name = "word"

    def count(self, text: str) -> int:
        return len(text.split())


class WordPieceCounter:
    """BERT-style WordPiece caliber via `tokenizers` (mBERT-shaped)."""

    def __init__(self, vocab: dict[str, int]) -> None:
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

        self._tokenizer = Tokenizer(
            models.WordPiece(
                vocab=vocab, unk_token="[UNK]", continuing_subword_prefix="##"
            )
        )
        self._tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self.caliber_name = "wordpiece"

    @classmethod
    def from_vocab_file(cls, path: Path) -> "WordPieceCounter":
        """Build from a vocab.txt (one token per line, line number = id).

        Parsed directly rather than via BertWordPieceTokenizer so a bare
        vocabulary file (no HF special tokens) loads the same way as the
        in-memory constructor.
        """
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"wordpiece vocab file not found: {path}")
        vocab: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            token = line.rstrip("\n")
            if token:
                vocab.setdefault(token, len(vocab))
        if not vocab:
            raise ValueError(f"wordpiece vocab file is empty: {path}")
        return cls(vocab)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


class SentencePieceCounter:
    """Unigram sentencepiece-style caliber via `tokenizers` (XLM-R-shaped)."""

    def __init__(self, pieces: list[tuple[str, float]]) -> None:
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

        unk_ids = [i for i, (piece, _) in enumerate(pieces) if piece == "<unk>"]
        if not unk_ids:
            raise ValueError(
                "sentencepiece piece list must contain an '<unk>' piece"
            )
        self._tokenizer = Tokenizer(models.Unigram(pieces, unk_id=unk_ids[0]))
        self._tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme="always"
        )
        self.caliber_name = "sentencepiece"

    @classmethod
    def from_tokenizer_json(cls, path: Path) -> "SentencePieceCounter":
        """Build from a serialized tokenizer.json (Tokenizer.save output)."""
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"sentencepiece tokenizer file not found: {path}")
        from tokenizers import Tokenizer

        counter = cls([("<unk>", 0.0)])
        counter._tokenizer = Tokenizer.from_file(str(path))
        return counter

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


def build_native_counter(flavor: str, vocab_path: Path | None = None) -> NativeCounter:
    """Factory: flavor name + optional vocab file -> counter.

    `word` needs no vocabulary; `wordpiece` and `sentencepiece` require
    one (vocab.txt / tokenizer.json respectively).
    """
    if flavor == "word":
        return WhitespaceCounter()
    if flavor == "wordpiece":
        if vocab_path is None:
            raise ValueError("wordpiece native flavor requires a vocab file")
        return WordPieceCounter.from_vocab_file(vocab_path)
    if flavor == "sentencepiece":
        if vocab_path is None:
            raise ValueError("sentencepiece native flavor requires a vocab file")
        return SentencePieceCounter.from_tokenizer_json(vocab_path)
    raise ValueError(
        f"unknown native tokenizer flavor {flavor!r} "
        "(expected 'word', 'wordpiece', or 'sentencepiece')"
    )
