"""Run the issue-19 pilot fine-tune (pinned invocation).

  - dataset: the frozen issue-18 pilot labeled set, bytes cross-checked
    against the distillation manifest before any work;
  - base: google-bert/bert-base-multilingual-cased at the pinned hub
    revision (mBERT — same architecture as the production checkpoint,
    drop-in by construction, decisions §6);
  - hyperparameters: seed 19, 3 epochs, batch 8, AdamW lr 2e-5 —
    plumbing defaults, pinned verbatim into the run record;
  - outputs: runs/train/pilot-v1/{checkpoint/, run-record.json,
    verify-report.json}; the run refuses to finish unless the checkpoint
    passes delivery verify (static + load) — day-one drop-in contract;
  - claim: plumbing-only. No quality claims; nothing here may enter a
    comparison report or acceptance reading (roadmap stage 2).

Environment: the project venv and CI are torchless by design; this
script bootstraps the dedicated `.venv-train` (Python 3.12 — the last
torch line with macOS x86_64 wheels — mirroring the host sidecar's
torch 2.2.2 / transformers 4.57.6 pins) and re-execs itself inside it.

Run from the repo root:  .venv/bin/python scripts/train_pilot.py
(the bootstrap upgrades any python into the train env automatically).
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DATASET = REPO / "training" / "distill" / "pilot-v1" / "labeled.jsonl"
DISTILL_MANIFEST = REPO / "training" / "distill" / "pilot-v1" / "manifest.json"
RECORD_DATASET = "training/distill/pilot-v1/labeled.jsonl"
OUT = REPO / "runs" / "train" / "pilot-v1"

BASE_MODEL = "google-bert/bert-base-multilingual-cased"
BASE_REVISION = "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"  # hub pin (frozen)

SEED = 19
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
MAX_SEQ_LEN = 512

TRAIN_VENV = REPO / ".venv-train"
TRAIN_REQUIREMENTS = (
    "torch==2.2.2",          # last macOS x86_64 wheel line; matches the sidecar
    "transformers==4.57.6",  # matches the sidecar (the smoke loads our output)
    "numpy<2",               # torch 2.2.2 is compiled against numpy 1.x
    "pytest",
    "httpx", "tiktoken", "tokenizers>=0.20",  # trillic runtime deps
)


def _bootstrap_train_env() -> None:
    """Create `.venv-train` if missing (Python 3.12 + pinned stack)."""
    print("bootstrapping the train environment (.venv-train)…")
    subprocess.run(
        ["uv", "venv", "--python", "3.12", str(TRAIN_VENV)], check=True, cwd=REPO
    )
    subprocess.run(
        ["uv", "pip", "install", "--python", str(TRAIN_VENV), *TRAIN_REQUIREMENTS],
        check=True,
        cwd=REPO,
    )
    subprocess.run(
        ["uv", "pip", "install", "--python", str(TRAIN_VENV), "-e", ".", "--no-deps"],
        check=True,
        cwd=REPO,
    )


def _ensure_train_env() -> None:
    """Re-exec inside .venv-train unless torch already imports."""
    try:
        import torch  # type: ignore[import-not-found] # noqa: F401
        import transformers  # type: ignore[import-not-found] # noqa: F401

        return
    except ImportError:
        pass
    if not (TRAIN_VENV / "bin" / "python").exists():
        _bootstrap_train_env()
    print(f"re-executing inside {TRAIN_VENV}…", flush=True)
    os_exe = subprocess.run(
        [str(TRAIN_VENV / "bin" / "python"), str(Path(__file__).resolve())],
        check=False,
        cwd=REPO,
    )
    sys.exit(os_exe.returncode)


def _repo_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        return commit, dirty
    except subprocess.SubprocessError:
        return None, None


def _check_dataset_frozen() -> None:
    """Refuse to train on drifted bytes: the distill manifest's
    outputs.labeled.sha256 IS the dataset's frozen identity."""
    manifest = json.loads(DISTILL_MANIFEST.read_text(encoding="utf-8"))
    recorded = manifest["outputs"]["labeled"]["sha256"]
    actual = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    if actual != recorded:
        raise SystemExit(
            f"{DATASET} does not match its distillation manifest "
            f"({DISTILL_MANIFEST}): {actual[:12]}… vs recorded "
            f"{recorded[:12]}… — restore the frozen dataset or regenerate "
            "the manifest deliberately"
        )
    print(
        f"frozen dataset ok: {DATASET.name} "
        f"({manifest['outputs']['labeled']['rows']} rows)"
    )


def main() -> int:
    _ensure_train_env()
    _check_dataset_frozen()
    commit, dirty = _repo_state()
    print(f"repo state: commit {commit} dirty={dirty}")
    from trillic.cli import main as cli_main

    started = time.monotonic()
    args = [
        "train", "run",
        "--dataset", str(DATASET),
        "--out", str(OUT),
        "--base-model", BASE_MODEL,
        "--base-revision", BASE_REVISION,
        "--seed", str(SEED),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LEARNING_RATE),
        "--weight-decay", str(WEIGHT_DECAY),
        "--max-grad-norm", str(MAX_GRAD_NORM),
        "--max-seq-len", str(MAX_SEQ_LEN),
        "--record-dataset", RECORD_DATASET,
    ]
    if commit:
        args += ["--repo-commit", commit]
    if dirty:
        args.append("--repo-dirty")
    code = cli_main(args)
    print(f"total wall: {time.monotonic() - started:.1f}s")
    return code


if __name__ == "__main__":
    sys.exit(main())
