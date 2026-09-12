"""Shared non-fixture helpers for the delivery-capability tests.

Candidate-checkpoint builders live here (not conftest) so test files can
import them without the conftest-import anti-pattern. Everything builds
with default dependencies only, zero network.
"""

import importlib.machinery
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from trillic.delivery import PROBE_TEXT

VALID_CHECKPOINT_CONFIG = {
    "architectures": ["BertForTokenClassification"],
    "model_type": "bert",
    "num_labels": 2,
    "id2label": {"0": "drop", "1": "keep"},
    "_name_or_path": "trillic/trillic-v1",
}


def write_fast_tokenizer(path: Path) -> None:
    """A minimal but real fast tokenizer whose vocab covers PROBE_TEXT."""
    words = PROBE_TEXT.split()
    vocab = {"[UNK]": 0, **{word: i + 1 for i, word in enumerate(words)}}
    tokenizer = Tokenizer(
        models.WordLevel(vocab=vocab, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path))


def write_blanking_tokenizer(path: Path) -> None:
    """A fast tokenizer whose normalizer blanks every probe word, so
    encodings carry no offsets — the offsets-unavailable fixture."""
    tokenizer = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Replace(word, "") for word in PROBE_TEXT.split()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path))


def make_checkpoint(
    root: Path,
    name: str,
    *,
    config: dict | None = None,
    fast_tokenizer: bool = True,
) -> Path:
    """Build a minimal candidate checkpoint directory.

    Every knob breaks exactly one contract item; the default builds the
    compliant fixture.
    """
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(config if config is not None else VALID_CHECKPOINT_CONFIG)
    )
    if fast_tokenizer:
        write_fast_tokenizer(directory / "tokenizer.json")
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "BertTokenizerFast"})
        )
    else:
        # slow-tokenizer assets only: no tokenizer.json anywhere
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "BertTokenizer"})
        )
        (directory / "vocab.txt").write_text("[UNK]\norder\n")
    (directory / "model.safetensors").write_bytes(b"\x00" * 16)
    return directory


def install_fake_load_deps(
    monkeypatch: Any,
    *,
    model_has_bert: bool = True,
    tokenizer_fast: bool = True,
) -> None:
    """Drive the load layer's pass/violation branches without torch.

    Registers fake `transformers`/`torch` modules (with specs, so
    importlib.find_spec sees them) whose from_pretrained calls return
    minimal objects. model_has_bert=False simulates an XLM-R-style base
    (no `.bert`); tokenizer_fast=False simulates a slow tokenizer served
    at runtime.
    """

    class FakeTokenizer:
        is_fast = tokenizer_fast

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> "FakeTokenizer":
            return cls()

        def __call__(self, text: str, return_offsets_mapping: bool = False):
            assert return_offsets_mapping
            offsets = [(0, 5)] if tokenizer_fast else []
            return {"offset_mapping": offsets}

    class FakeModel:
        def __init__(self) -> None:
            if model_has_bert:
                self.bert = object()  # what the host guardrail calls

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> "FakeModel":
            return cls()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizer  # type: ignore[attr-defined]
    fake_transformers.AutoModelForTokenClassification = FakeModel  # type: ignore[attr-defined]
    fake_torch = ModuleType("torch")

    for name, module in (("transformers", fake_transformers), ("torch", fake_torch)):
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
