"""Run the issue-18 teacher distillation pilot (pinned invocation).

  - corpus: the frozen v1 training layers (issues #16/#17), bytes
    cross-checked against their manifests before any gateway call;
  - scale: chunk budgets rag=150 / system_prompt=60 / dialogue=90 -> 115
    entries, 299 chunks = 299 teacher calls (~10^2, per the issue); all
    three load types present; the rag budget is filled by hotpotqa (the
    file's head) — gov_report's very long entries join at full scale
    (issue #20), noted in the pilot manifest by construction (selection
    is recorded);
  - teacher: deepseek/deepseek-v4-pro through the tokencamp gateway
    (internal ledger); prompt = llmlingua2-paper-v1 (sha256 pinned);
  - resume: crash journal under runs/.ledger/ keyed by the content
    address (corpus bytes + teacher + prompt + chunking) — kill and
    rerun pays only for never-recorded chunks;
  - outputs: training/distill/pilot-v1/{labeled.jsonl, report.json,
    report.md, manifest.json}.

Run from the repo root with the gateway up (tokencamp-pro
`scripts/dev.sh up`) and REFINE_SERVICE_KEY in the environment.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from trillic.clients.gateway import HttpGatewayClient
from trillic.distill import run_distillation
from trillic.tokens import TokenCounter

REPO = Path(__file__).resolve().parents[1]

CORPUS = [
    REPO / "training" / "corpus" / "longbench-train-v1.jsonl",
    REPO / "training" / "corpus" / "synthetic-train-v1.jsonl",
]
MANIFESTS = [
    REPO / "training" / "manifests" / "training-corpus-v1.json",
    REPO / "training" / "manifests" / "synthetic-training-v1.json",
]
OUT = REPO / "training" / "distill" / "pilot-v1"
LEDGER_ROOT = REPO / "runs"

TEACHER_MODEL = "deepseek/deepseek-v4-pro"
GATEWAY_URL = "http://127.0.0.1:3005"
# Pacing posture proven on the 2026-09-12 full149 runs: >= 3s between
# call starts, 4 workers (headroom vs the ~35/min observed trip point).
MIN_CALL_INTERVAL = 3.0
CONCURRENCY = 4

CHUNK_BUDGETS = {"rag": 150, "system_prompt": 60, "dialogue": 90}
MAX_CHUNK_TOKENS = 512
WINDOW_SIZE = 150
VR_DROP_FRACTION = 0.05
AG_DROP_FRACTION = 0.10


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


def _check_corpus_frozen() -> None:
    """Refuse to distill drifted corpus bytes: the manifest sha256 IS the
    frozen identity of each layer."""
    for corpus_path, manifest_path in zip(CORPUS, MANIFESTS):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = manifest["corpus"]["sha256"]
        actual = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        if actual != recorded:
            raise SystemExit(
                f"{corpus_path} does not match its manifest "
                f"({manifest_path}): {actual[:12]}… vs recorded "
                f"{recorded[:12]}… — restore the frozen corpus or "
                "regenerate the manifest deliberately"
            )
        print(f"frozen corpus ok: {corpus_path.name} ({manifest['corpus']['entries']} entries)")


def main() -> int:
    _check_corpus_frozen()
    if (OUT / "manifest.json").is_file():
        raise SystemExit(f"{OUT / 'manifest.json'} already exists — pilot outputs are versioned; move it first")
    commit, dirty = _repo_state()
    print(f"repo state: commit {commit} dirty={dirty}")
    gateway = HttpGatewayClient(
        base_url=GATEWAY_URL,
        min_call_interval=MIN_CALL_INTERVAL,
    )
    try:
        manifest = run_distillation(
            corpus_paths=CORPUS,
            chunk_budgets=CHUNK_BUDGETS,
            gateway=gateway,
            teacher_model=TEACHER_MODEL,
            out_dir=OUT,
            ledger_root=LEDGER_ROOT,
            counter=TokenCounter("cl100k_base"),
            window_size=WINDOW_SIZE,
            max_chunk_tokens=MAX_CHUNK_TOKENS,
            vr_drop_fraction=VR_DROP_FRACTION,
            ag_drop_fraction=AG_DROP_FRACTION,
            concurrency=CONCURRENCY,
            repo_commit=commit,
            repo_dirty=dirty,
            record_corpus=[
                "training/corpus/longbench-train-v1.jsonl",
                "training/corpus/synthetic-train-v1.jsonl",
            ],
            record_journal="runs/.ledger/ (content-addressed, gitignored runtime ledger)",
        )
    finally:
        gateway.close()
    counts = manifest["counts"]
    gw = manifest["gateway"]
    print(
        f"{OUT}: {counts['labeled_kept']} labeled chunks kept "
        f"({counts['dropped']} dropped of {counts['chunks']} compressed; "
        f"gateway calls fresh {gw['calls_fresh']} / reused {gw['calls_reused']})"
    )
    print(f"crash journal: {gw['journal']} (ledger {gw['ledger_id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
