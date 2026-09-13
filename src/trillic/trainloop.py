"""Torch/transformers fine-tune loop (issue #19) — train environment only.

Importable only where torch + transformers are installed (the dedicated
``.venv-train``; bootstrap via ``scripts/train_pilot.py`` — the project
venv and CI stay torchless by design, same posture as the delivery load
layer).

Determinism contract: one ``torch.manual_seed`` before model
construction (the randomly initialized classification head draws from
the global RNG), per-epoch window shuffling from
``random.Random(seed + epoch)``, CPU device, constant learning rate, and
no framework Trainer (whose internals move between versions). Same
pinned inputs on the same machine must reproduce byte-identical
checkpoint files — asserted by the train-env test suite and by the
pilot's double-run re-pin.
"""

import platform
import random
import time
from pathlib import Path
from typing import Any

import torch  # type: ignore[import-not-found]
from transformers import (  # type: ignore[import-not-found]
    AutoModelForTokenClassification,
    AutoTokenizer,
    set_seed,
)

from trillic.train import (
    ID2LABEL,
    LABEL2ID,
    TrainError,
)

IGNORE_LABEL = -100  # CrossEntropyLoss default ignore_index (specials/pads)


def resolve_revision(base_model: str) -> str:
    """The hub commit sha for `base_model` (recorded into the run record
    as the base pin; a rerun that resolves a different sha is a
    different training identity)."""
    try:
        from huggingface_hub import model_info  # type: ignore[import-not-found]

        info = model_info(base_model)
    except Exception as e:  # noqa: BLE001 — hub failure is a TrainError
        raise TrainError(f"cannot resolve revision for {base_model}: {e}") from e
    if not info.sha:
        raise TrainError(f"{base_model}: the hub returned no revision sha")
    return str(info.sha)


def load_tokenizer(base_model: str, base_revision: str):
    """The base model's fast tokenizer (training input caliber = the
    host's inference caliber: same WordPiece, same offsets)."""
    try:
        if Path(base_model).is_dir():
            return AutoTokenizer.from_pretrained(base_model)
        return AutoTokenizer.from_pretrained(base_model, revision=base_revision)
    except Exception as e:  # noqa: BLE001 — load failure is a TrainError
        raise TrainError(f"cannot load tokenizer for {base_model}: {e}") from e


def library_versions() -> dict[str, str]:
    import tokenizers
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "platform": platform.platform(),
    }


def _from_base(base_model: str, base_revision: str):
    """from_pretrained against a hub pin or a local fixture directory."""
    try:
        if Path(base_model).is_dir():
            return AutoModelForTokenClassification.from_pretrained(
                base_model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
            )
        return AutoModelForTokenClassification.from_pretrained(
            base_model,
            revision=base_revision,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    except Exception as e:  # noqa: BLE001 — load failure is a TrainError
        raise TrainError(f"cannot load base model {base_model}: {e}") from e


def _tensored(windows: list, tokenizer: Any, *, batch_size: int):
    """Yield (input_ids, attention_mask, labels) batches, padded.

    [CLS]/[SEP] wrap every window with the ignore label — the host
    force-keeps specials at inference, so their logits are never read.
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    if cls_id is None or sep_id is None or pad_id is None:
        raise TrainError("the base tokenizer lacks CLS/SEP/PAD token ids")
    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        width = max(len(w.input_ids) for w in batch) + 2
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        labels = torch.full((len(batch), width), IGNORE_LABEL, dtype=torch.long)
        for i, window in enumerate(batch):
            seq = [cls_id, *window.input_ids, sep_id]
            lab = [IGNORE_LABEL, *window.labels, IGNORE_LABEL]
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = 1
            labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        yield ids, mask, labels


def fine_tune(
    *,
    windows: list,
    tokenizer: Any,
    base_model: str,
    base_revision: str,
    checkpoint_dir: Path,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Fine-tune the base into a drop-in token classifier and save it.

    Everything that touches the RNG is seeded from
    ``hyperparameters["seed"]``; the loop is plain torch so the op
    sequence (and therefore the RNG consumption) is fully visible.
    Returns the training-stats half of the run record.
    """
    seed = int(hyperparameters["seed"])
    epochs = int(hyperparameters["epochs"])
    batch_size = int(hyperparameters["batch_size"])
    learning_rate = float(hyperparameters["learning_rate"])
    weight_decay = float(hyperparameters["weight_decay"])
    max_grad_norm = float(hyperparameters["max_grad_norm"])

    set_seed(seed)  # python random + numpy-if-present + torch
    torch.manual_seed(seed)
    model = _from_base(base_model, base_revision)
    model.config.architectures = ["BertForTokenClassification"]
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    started = time.monotonic()
    steps = 0
    loss_by_epoch: list[float] = []
    for epoch in range(epochs):
        order = list(range(len(windows)))
        random.Random(seed + epoch).shuffle(order)
        epoch_windows = [windows[i] for i in order]
        total_loss = 0.0
        total_labels = 0
        for ids, mask, labels in _tensored(
            epoch_windows, tokenizer, batch_size=batch_size
        ):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            steps += 1
            # the batch loss is a token mean (ignore_index excluded), so
            # accumulate weighted by real label count — window-count
            # weighting would misweight ragged batches
            n_labels = int((labels != IGNORE_LABEL).sum())
            total_loss += float(loss.item()) * n_labels
            total_labels += n_labels
        loss_by_epoch.append(round(total_loss / total_labels, 6))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    return {
        "steps": steps,
        "train_loss_by_epoch": loss_by_epoch,
        "device": "cpu",
        "wall_seconds": round(time.monotonic() - started, 3),
        "library": library_versions(),
    }
