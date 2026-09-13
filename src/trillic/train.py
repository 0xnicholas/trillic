"""Training plumbing (issue #19): labeled pilot dataset -> HF checkpoint.

Two layers, mirroring the delivery capability's split:

- THIS module (default dependencies, zero torch): consuming issue #18's
  labeled dataset, aligning word-level keep labels onto subword tokens
  through fast-tokenizer offsets (the same `tokenizers` machinery
  transformers wraps), packing windows to the mBERT cap, the pinned run
  record (seed / hyperparameters / data sha256 / base revision — same
  inputs, same record bytes), and the versioned-output identity guard
  (same identity reruns overwrite in place, drift picks a new directory).
- `trillic.trainloop` (optional train environment): the torch/
  transformers fine-tune itself plus checkpoint save. Absent there, the
  CLI refuses with the bootstrap hint before doing any work.

Supervision semantics (pinned, recorded in every run record): a subword
token inherits the keep label of the distill word token it overlaps
(``trainloop`` adds ``[CLS]``/``[SEP]`` with the ignore label −100 at
tensorization); subwords overlapping no word token — commas by design,
regex-external characters by limitation — default to drop. No quality
claims are made anywhere in this line: the run record carries the
plumbing-only banner verbatim (roadmap stage 2; decisions §4).
"""

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trillic import __version__
from trillic.delivery import VERIFY_REPORT_NAME, verify_report_json

TRAIN_RECORD_SCHEMA_VERSION = 1

DEFAULT_BASE_MODEL = "google-bert/bert-base-multilingual-cased"
DEFAULT_MAX_SEQ_LEN = 512  # mBERT cap, [CLS]/[SEP] included by the loop
DEFAULT_SEED = 19
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_GRAD_NORM = 1.0
OPTIMIZER = "adamw"
LR_SCHEDULE = "constant"
SHUFFLE_RULE = "random.Random(seed + epoch).shuffle(window order)"

PLUMBING_ONLY_CLAIM = (
    "plumbing-only pilot training (issue #19): pipeline validation, "
    "explicitly NO quality claims — these numbers are not comparison "
    "evidence and must not enter any acceptance reading (roadmap stage 2)"
)

LABELING_RULE = (
    "subword inherits the keep label of the overlapping distill word "
    "token; subwords overlapping no word token (commas by design, chars "
    "outside the distill word regex by limitation) default to drop; "
    "special tokens are ignored in the loss (-100)"
)

# Positional drop-in semantics (decisions §6): the host reads
# softmax(logits)[..., 1] as the keep probability — index 1 MUST name
# the keep class. Explicit names (passing delivery verify's inversion
# check) instead of neutral LABEL_0/LABEL_1: self-documenting artifact.
ID2LABEL = {0: "drop", 1: "keep"}
LABEL2ID = {"drop": 0, "keep": 1}

TRAIN_ENV_HINT = "scripts/train_pilot.py (dedicated .venv-train; see README)"

RUN_RECORD_NAME = "run-record.json"
CHECKPOINT_DIR_NAME = "checkpoint"


class TrainError(ValueError):
    """Training input, plumbing, or artifact error."""


# ── labeled dataset (issue #18 output) ───────────────────────────────────


def load_labeled_dataset(path: Path) -> list[dict[str, Any]]:
    """Parse + validate labeled.jsonl: every row needs an id, the chunk
    text under ``prompt``, and word tokens ``[text, start, end, keep]``
    whose spans are ordered in-bounds slices of the prompt."""
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise TrainError(f"{path}: no such labeled dataset") from e
    except OSError as e:
        raise TrainError(f"{path}: cannot read ({e})") from e
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise TrainError(f"{path}: line {line_no} is not JSON ({e})") from e
        for field in ("id", "prompt", "tokens", "load_type"):
            if field not in row:
                raise TrainError(f"{path}: line {line_no} lacks {field!r}")
        if not isinstance(row["prompt"], str):
            raise TrainError(f"{path}: line {line_no}: prompt must be a string")
        prompt = row["prompt"]
        last_end = 0
        for tok in row["tokens"]:
            if not (isinstance(tok, list) and len(tok) == 4):
                raise TrainError(
                    f"{path}: line {line_no}: token rows are [text, start, end, keep]"
                )
            _, start, end, keep = tok
            if not (isinstance(start, int) and isinstance(end, int)) or not (
                last_end <= start < end <= len(prompt)
            ):
                raise TrainError(
                    f"{path}: line {line_no}: span [{start}, {end}) out of "
                    f"bounds or unordered (prompt is {len(prompt)} chars)"
                )
            if keep not in (0, 1):
                raise TrainError(f"{path}: line {line_no}: keep label {keep!r}")
            last_end = end
        rows.append(row)
    if not rows:
        raise TrainError(f"{path}: zero rows — nothing to train on")
    return rows


# ── subword alignment ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SubwordToken:
    """One subword with its char span, keep label, vocab id, and the
    index of the distill word token it belongs to (−1 = unlabeled gap,
    e.g. commas)."""

    text: str
    start: int
    end: int
    label: int
    input_id: int
    word_index: int


@dataclass(frozen=True)
class WordSpan:
    """A distill word token as loaded from the dataset."""

    text: str
    start: int
    end: int
    keep: int


def align_labels(
    text: str, tokens: list[list], tokenizer: Any
) -> list[SubwordToken]:
    """Encode `text` with a fast tokenizer and label every subword.

    A subword overlapping a word span inherits that word's keep label
    (word spans are disjoint, so the overlap is unique); subwords over
    unlabeled characters default to drop (label 0). Offsets are the
    ground truth — no string matching between tokenizer and word tokens.
    """
    words = [WordSpan(text=str(w), start=int(s), end=int(e), keep=int(k))
             for w, s, e, k in tokens]
    encoding = tokenizer.encode(text, add_special_tokens=False)
    aligned: list[SubwordToken] = []
    word_idx = 0
    for token_text, (start, end), input_id in zip(
        encoding.tokens, encoding.offsets, encoding.ids
    ):
        label = 0
        owner = -1
        if end > start:
            # advance past words that end before this subword starts
            while word_idx < len(words) and words[word_idx].end <= start:
                word_idx += 1
            if word_idx < len(words):
                w = words[word_idx]
                if w.start < end and start < w.end:
                    label = w.keep
                    owner = word_idx
        aligned.append(
            SubwordToken(
                text=token_text, start=start, end=end, label=label,
                input_id=int(input_id), word_index=owner,
            )
        )
    return aligned


# ── window packing ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Window:
    """A lossless run of subwords, whole word-groups only.

    ``input_ids``/``labels`` are derived views; specials are added by
    the training loop, never here.
    """

    tokens: tuple[SubwordToken, ...]

    @property
    def input_ids(self) -> tuple[int, ...]:
        return tuple(t.input_id for t in self.tokens)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(t.label for t in self.tokens)


def _word_groups(aligned: list[SubwordToken]) -> list[list[SubwordToken]]:
    """Partition subwords so a distill word's pieces never split.

    Group = one word's subwords plus the unlabeled subwords that trail
    it (commas, gaps); a leading unlabeled run forms its own group.
    Ownership comes from alignment (``word_index``), never from label
    equality — adjacent words with the same label stay separate.
    """
    groups: list[list[SubwordToken]] = []
    current: list[SubwordToken] = []
    current_owner = -2  # -2 = nothing open; -1 = unlabeled run
    for tok in aligned:
        owner = tok.word_index if tok.word_index >= 0 else -1
        if current and owner != current_owner:
            groups.append(current)
            current = []
        current.append(tok)
        current_owner = owner
    if current:
        groups.append(current)
    return groups


def pack_windows(
    aligned: list[SubwordToken], *, max_seq_len: int
) -> list[Window]:
    """Greedy-pack word groups into windows of at most
    ``max_seq_len - 2`` subwords (the loop adds [CLS]/[SEP])."""
    room = _validate_max_seq_len(max_seq_len) - 2
    if not aligned:
        return []
    windows: list[Window] = []
    current: list[SubwordToken] = []
    for group in _word_groups(aligned):
        if len(current) + len(group) <= room:
            current.extend(group)
            continue
        if current:
            windows.append(Window(tokens=tuple(current)))
            current = []
        if len(group) <= room:
            current = list(group)
        else:
            # a single group longer than the window: truncate it — the
            # caller records the dropped tokens (pathological by
            # construction: a >510-subword single word)
            windows.append(Window(tokens=tuple(group[:room])))
    if current:
        windows.append(Window(tokens=tuple(current)))
    return windows


def _validate_max_seq_len(max_seq_len: int) -> int:
    if (
        not isinstance(max_seq_len, int)
        or isinstance(max_seq_len, bool)
        or max_seq_len < 3
    ):
        raise TrainError(
            f"max_seq_len must be an integer >= 3 (CLS + SEP + one subword), "
            f"got {max_seq_len!r}"
        )
    return max_seq_len


# ── examples + stats ─────────────────────────────────────────────────────


def build_examples(
    rows: list[dict[str, Any]], tokenizer: Any, *, max_seq_len: int
) -> dict[str, Any]:
    """Every labeled row -> aligned subwords -> packed windows, plus the
    stats the run record pins (counts, keep ratio, truncation)."""
    _validate_max_seq_len(max_seq_len)
    windows_list: list[Window] = []
    by_load_type: dict[str, int] = {}
    kept = labeled = 0
    truncated = 0
    for row in rows:
        by_load_type[row["load_type"]] = by_load_type.get(row["load_type"], 0) + 1
        aligned = align_labels(row["prompt"], row["tokens"], tokenizer)
        flat_ids = [t.input_id for t in aligned]
        kept += sum(t.label for t in aligned)
        labeled += len(aligned)
        windows = pack_windows(aligned, max_seq_len=max_seq_len)
        packed = [i for w in windows for i in w.input_ids]
        truncated += len(flat_ids) - len(packed)
        windows_list.extend(windows)
    return {
        "examples": len(rows),
        "windows": len(windows_list),
        "windows_list": windows_list,
        "by_load_type": dict(sorted(by_load_type.items())),
        "keep_token_ratio": round(kept / labeled, 4) if labeled else 0.0,
        "truncated_tokens": truncated,
    }


# ── run record ───────────────────────────────────────────────────────────


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hyperparameters_pin(
    *, seed: int, epochs: int, batch_size: int, learning_rate: float,
    weight_decay: float, max_grad_norm: float, max_seq_len: int,
) -> dict[str, Any]:
    """The single source of the hyperparameter pin — the run record, the
    output-identity guard, and the loop all read this one dict, so the
    recorded pin can never drift from what actually trained."""
    return {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "max_seq_len": max_seq_len,
        "optimizer": OPTIMIZER,
        "lr_schedule": LR_SCHEDULE,
        "shuffle_rule": SHUFFLE_RULE,
    }


def build_run_record(
    *,
    dataset_path: Path,
    dataset_rows: int,
    base_model: str,
    base_revision: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    max_seq_len: int,
    examples: int,
    windows: int,
    by_load_type: dict[str, int],
    keep_token_ratio: float,
    truncated_tokens: int,
    loop_record: dict[str, Any],
    library: dict[str, str],
    checkpoint_digest: str,
    verify_overall: str,
    repo_commit: str | None,
    repo_dirty: bool | None,
    record_dataset: str | None = None,
) -> dict[str, Any]:
    """The reproducibility pin: everything that determines the trained
    weights, and nothing that varies between identical reruns (no
    timestamps, no machine-local paths)."""
    return {
        "schema_version": TRAIN_RECORD_SCHEMA_VERSION,
        "claim": PLUMBING_ONLY_CLAIM,
        "generated_by": {
            "tool": "trillic.train",
            "version": __version__,
            "repo_commit": repo_commit,
            "repo_dirty": repo_dirty,
        },
        "dataset": {
            "file": record_dataset if record_dataset is not None else str(dataset_path),
            "sha256": dataset_sha256(dataset_path),
            "rows": dataset_rows,
        },
        "base_model": {"id": base_model, "revision": base_revision},
        "hyperparameters": hyperparameters_pin(
            seed=seed, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=weight_decay,
            max_grad_norm=max_grad_norm, max_seq_len=max_seq_len,
        ),
        "labeling": {
            "alignment": LABELING_RULE,
            "keep_token_ratio": keep_token_ratio,
            "truncated_tokens": truncated_tokens,
        },
        "training": {
            "examples": examples,
            "windows": windows,
            "by_load_type": by_load_type,
            **loop_record,
        },
        "library": dict(sorted(library.items())),
        "outputs": {
            "checkpoint_dir": CHECKPOINT_DIR_NAME,
            "checkpoint_digest": checkpoint_digest,
            "delivery_verify": verify_overall,
        },
    }


def run_record_json(record: dict[str, Any]) -> str:
    """Deterministic serialization (sorted keys, no timestamps)."""
    return json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


# ── versioned-output identity guard (mirrors distill #18) ────────────────


def _identity_of(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": {"sha256": record.get("dataset", {}).get("sha256")},
        "base_model": {
            "id": record.get("base_model", {}).get("id"),
            "revision": record.get("base_model", {}).get("revision"),
        },
        "hyperparameters": record.get("hyperparameters"),
    }


def check_output_identity(out_dir: Path, identity: dict[str, Any]) -> None:
    """Same-identity reruns overwrite in place (the deterministic re-pin
    flow); any drift picks a new output directory instead of silently
    rewriting history.

    ``identity`` sections: ``dataset`` (sha256), ``base_model`` (id +
    revision), ``hyperparameters`` (the full pin).
    """
    record_path = Path(out_dir) / RUN_RECORD_NAME
    if not record_path.is_file():
        return
    try:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise TrainError(
            f"{record_path} exists but cannot be read ({e}) — move it aside "
            "and regenerate deliberately"
        ) from e
    mismatches: list[str] = []
    existing_id = _identity_of(existing)
    for section in ("dataset", "base_model", "hyperparameters"):
        if existing_id[section] != identity.get(section):
            if section == "dataset":
                mismatches.append("dataset sha256")
            elif section == "base_model":
                mismatches.append("base model / revision")
            else:
                new_hp = identity.get(section) or {}
                old_hp = existing_id[section] or {}
                drifted = sorted(
                    key for key in set(old_hp) | set(new_hp)
                    if old_hp.get(key) != new_hp.get(key)
                )
                mismatches.extend(f"hyperparameters.{key}" for key in drifted)
    if mismatches:
        raise TrainError(
            f"{record_path} already exists with a different identity "
            f"({', '.join(mismatches)}) — training outputs are versioned "
            "artifacts; pick a new output directory instead of overwriting"
        )


# ── train environment probe ──────────────────────────────────────────────


def train_deps_available() -> bool:
    """True when the optional train stack (torch + transformers) imports."""
    return all(
        importlib.util.find_spec(name) is not None for name in ("torch", "transformers")
    )


def require_train_env() -> None:
    if not train_deps_available():
        raise TrainError(
            "training requires the dedicated train environment "
            f"(torch/transformers not importable here) — bootstrap with "
            f"{TRAIN_ENV_HINT}"
        )


# ── orchestration: dataset in, verified checkpoint out ───────────────────


def _validate_hyperparams(
    *, seed: int, epochs: int, batch_size: int, learning_rate: float,
    weight_decay: float, max_grad_norm: float,
) -> None:
    for name, value in (("seed", seed), ("epochs", epochs), ("batch_size", batch_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise TrainError(f"{name} must be an integer >= 1, got {value!r}")
    float_checks: tuple[tuple[str, float], ...] = (
        ("learning_rate", learning_rate),
        ("weight_decay", weight_decay),
        ("max_grad_norm", max_grad_norm),
    )
    for fname, fvalue in float_checks:
        if not isinstance(fvalue, (int, float)) or isinstance(fvalue, bool) or fvalue <= 0:
            raise TrainError(f"{fname} must be a positive number, got {fvalue!r}")


def run_training(
    *,
    dataset_path: Path,
    out_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    base_revision: str | None = None,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    repo_commit: str | None = None,
    repo_dirty: bool | None = None,
    record_dataset: str | None = None,
    loop: Any = None,
    verify_fn: Any = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Full plumbing: labeled dataset -> fine-tuned HF checkpoint that
    passes delivery verify (static + load) -> pinned run record.

    ``loop``/``verify_fn``/``tokenizer`` are injection seams for the
    torchless test suite; the real path lazily uses ``trillic.trainloop``
    (train environment) and ``trillic.delivery``.
    """
    _validate_hyperparams(
        seed=seed, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
    )
    _validate_max_seq_len(max_seq_len)
    out_dir = Path(out_dir)
    dataset_path = Path(dataset_path)
    rows = load_labeled_dataset(dataset_path)  # cheap validation first
    own_loop = loop is None
    if own_loop:
        require_train_env()
        from trillic.trainloop import fine_tune, load_tokenizer, resolve_revision

        loop = fine_tune
        if base_revision is None:
            base_revision = resolve_revision(base_model)
        if tokenizer is None:
            tokenizer = load_tokenizer(base_model, base_revision)
    elif base_revision is None:
        raise TrainError(
            "base_revision is required when the loop is injected "
            "(revision resolution needs the train environment)"
        )
    check_output_identity(
        out_dir,
        {
            "dataset": {"sha256": dataset_sha256(dataset_path)},
            "base_model": {"id": base_model, "revision": base_revision},
            "hyperparameters": hyperparameters_pin(
                seed=seed, epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, weight_decay=weight_decay,
                max_grad_norm=max_grad_norm, max_seq_len=max_seq_len,
            ),
        },
    )
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    built = build_examples(rows, backend, max_seq_len=max_seq_len)
    if not built["windows_list"]:
        raise TrainError("the dataset produced zero windows — nothing to train on")

    checkpoint_dir = out_dir / CHECKPOINT_DIR_NAME
    hyperparameters = hyperparameters_pin(
        seed=seed, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay,
        max_grad_norm=max_grad_norm, max_seq_len=max_seq_len,
    )
    loop_record = loop(
        windows=built["windows_list"],
        tokenizer=tokenizer,
        base_model=base_model,
        base_revision=base_revision,
        checkpoint_dir=checkpoint_dir,
        hyperparameters=hyperparameters,
    )

    if verify_fn is None:
        from trillic.delivery import verify_checkpoint as default_verify

        verify_fn = default_verify  # load layer included: we are in the train env
    verify_report = verify_fn(checkpoint_dir)
    verify_path = out_dir / VERIFY_REPORT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_path.write_text(verify_report_json(verify_report), encoding="utf-8")
    if verify_report.get("overall") != "pass":
        raise TrainError(
            "trained checkpoint failed delivery verify "
            f"({verify_report.get('overall')!r}) — report at {verify_path}; "
            "the training line refuses to emit a non-drop-in artifact"
        )

    from trillic.delivery import checkpoint_digest, scan_checkpoint_files

    files = scan_checkpoint_files(checkpoint_dir)
    library = dict(loop_record.get("library", {}))
    training_stats = {
        k: v for k, v in loop_record.items()
        if k != "library" and k != "wall_seconds"  # runtime observation,
        # not reproducibility identity — byte-identical reruns (the
        # distill/verify artifacts set the deterministic-record precedent)
    }
    record = build_run_record(
        dataset_path=dataset_path,
        dataset_rows=len(rows),
        base_model=base_model,
        base_revision=base_revision,
        seed=seed, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay,
        max_grad_norm=max_grad_norm, max_seq_len=max_seq_len,
        examples=built["examples"], windows=built["windows"],
        by_load_type=built["by_load_type"],
        keep_token_ratio=built["keep_token_ratio"],
        truncated_tokens=built["truncated_tokens"],
        loop_record=training_stats,
        library=library,
        checkpoint_digest=checkpoint_digest(files),
        verify_overall=str(verify_report["overall"]),
        repo_commit=repo_commit, repo_dirty=repo_dirty,
        record_dataset=record_dataset,
    )
    (out_dir / RUN_RECORD_NAME).write_text(run_record_json(record), encoding="utf-8")
    return record


def print_train_summary(record: dict[str, Any]) -> None:
    """Human-readable summary (machine JSON is run-record.json)."""
    training = record["training"]
    outputs = record["outputs"]
    print(
        f"trained {training['examples']} examples -> {training['windows']} windows "
        f"(keep ratio {record['labeling']['keep_token_ratio']}); "
        f"loss by epoch: {training['train_loss_by_epoch']}"
    )
    print(
        f"checkpoint digest {outputs['checkpoint_digest'][:12]}… — "
        f"delivery verify {outputs['delivery_verify'].upper()}"
    )
    print(f"claim: {record['claim']}")
