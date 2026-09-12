"""Delivery capability: drop-in contract verification and artifact
packing (issues #11/#13/#14).

`delivery verify` checks a candidate checkpoint directory against the
drop-in contract the host refine sidecar requires (docs/decisions.md §6):

- architecture class ``BertForTokenClassification`` (2-label head),
- ``num_labels == 2`` and ``id2label`` semantics 0 = drop / 1 = keep —
  the host compressor reads ``softmax(logits)[..., 1]`` as the keep
  probability, so the semantics are positional and an inverted mapping
  silently breaks compression,
- a fast tokenizer with usable offsets (the host's force-keep locates
  numeric-fact spans through ``return_offsets_mapping``),
- (load layer, optional heavy deps) a real instantiation asserting the
  ``.bert`` attribute the host's similarity guardrail calls directly.

`delivery pack` assembles the delivery form behind that gate: a
per-file sha256 manifest plus the finalized integration pack rendered
from the in-repo draft (pure local placeholder replacement, single
source for every value — nothing is hand-filled).

Two layers, three verdict states per layer: pass / violation, plus
"unverified" for a load layer that could not run (heavy deps absent or
``--static-only``). Unverified is never reported as pass.

Everything is local and zero-network. Reports carry no timestamps: the
same checkpoint verifies to byte-identical report bytes, and the same
input packs to byte-identical artifacts (auditable, not random).
"""

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

from trillic import __version__

DELIVERY_REPORT_SCHEMA_VERSION = 1

# Probe text for the offsets checks (static + load): wordy, with a
# numeric fact, like the traffic the host force-keep protects.
PROBE_TEXT = "order 4451 shipped on 2026-09-11"

# Label-name hints for the id2label inversion check. Positional semantics
# (index 1 = keep) are what the host actually executes; explicit opposite
# names are rejected, neutral names ("LABEL_0"/"LABEL_1") pass.
_KEEP_HINTS = ("keep", "kept", "retain", "preserv")
_DROP_HINTS = ("drop", "discard", "delet", "remov")

EXPECTED_ARCHITECTURE = "BertForTokenClassification"

LOAD_DEPS_INSTALL_HINT = "uv sync --group load"

DEFAULT_DRAFT_PATH = Path("docs/delivery/refine-integration.md")
SUMS_NAME = "SHA256SUMS"
INTEGRATION_PACK_NAME = "integration-pack.md"
VERIFY_REPORT_NAME = "verify-report.json"

_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


class DeliveryError(Exception):
    """The checkpoint could not be examined (missing dir, unreadable assets)."""


# ── content-addressed identity ───────────────────────────────────────────


def scan_checkpoint_files(directory: Path) -> list[dict[str, Any]]:
    """Every file under `directory`, sorted by path, each with its sha256.

    The per-file digests are the checkpoint's identity: "which artifact
    was verified" must be answerable by content, not by path.
    """
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            files.append(
                {
                    "name": path.relative_to(directory).as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
    return files


def checkpoint_digest(files: list[dict[str, Any]]) -> str:
    """Directory digest: sha256 over `<sha256>  <name>\\n` lines (sorted)."""
    lines = "".join(f"{file['sha256']}  {file['name']}\n" for file in files)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


# ── static layer ─────────────────────────────────────────────────────────


def _load_config(directory: Path) -> dict[str, Any]:
    config_path = directory / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise DeliveryError(f"{config_path}: no such file (config.json is required)") from e
    except (OSError, json.JSONDecodeError) as e:
        raise DeliveryError(f"{config_path}: cannot parse ({e})") from e
    if not isinstance(config, dict):
        raise DeliveryError(f"{config_path}: expected a JSON object at top level")
    return config


def _check(check_id: str, ok: bool, detail: str, remedy: str) -> dict[str, str]:
    return {
        "id": check_id,
        "status": "pass" if ok else "violation",
        "detail": detail,
        "remedy": "" if ok else remedy,
    }


def _check_architecture(config: dict[str, Any]) -> dict[str, str]:
    architectures = config.get("architectures") or []
    ok = EXPECTED_ARCHITECTURE in architectures
    return _check(
        "architecture_class",
        ok,
        f"architectures = {architectures!r}",
        f"export the checkpoint so config.json architectures includes "
        f"'{EXPECTED_ARCHITECTURE}' (AutoModelForTokenClassification.save_pretrained)",
    )


def _check_num_labels(config: dict[str, Any]) -> dict[str, str]:
    id2label = config.get("id2label")
    id2label_len = len(id2label) if isinstance(id2label, dict) else None
    declared = config.get("num_labels", id2label_len)
    ok = declared == 2
    return _check(
        "num_labels",
        ok,
        f"num_labels = {declared!r}"
        + (f" (id2label has {id2label_len} entries)" if id2label_len else ""),
        "export the classification head with exactly 2 labels "
        "(keep/drop); the host reads a 2-column logits tensor",
    )


def _check_id2label_semantics(config: dict[str, Any]) -> dict[str, str]:
    id2label = config.get("id2label")
    if not isinstance(id2label, dict) or set(id2label) != {"0", "1"}:
        return _check(
            "id2label_semantics",
            False,
            f"id2label = {id2label!r}",
            "id2label must map exactly the indices 0 and 1 "
            "(0 = drop, 1 = keep) — the host treats the head as positional",
        )
    label_drop = str(id2label["0"]).strip().lower()
    label_keep = str(id2label["1"]).strip().lower()
    inverted = any(hint in label_keep for hint in _DROP_HINTS) or any(
        hint in label_drop for hint in _KEEP_HINTS
    )
    return _check(
        "id2label_semantics",
        not inverted,
        f"id2label = {{0: {id2label['0']!r}, 1: {id2label['1']!r}}}"
        + (" — names contradict positional semantics" if inverted else ""),
        "swap id2label so index 0 names the drop class and index 1 the "
        "keep class; the host computes keep probability as softmax(logits)[..., 1]",
    )


def _load_fast_tokenizer(directory: Path):
    """Load tokenizer.json with the `tokenizers` library, or return None."""
    tokenizer_path = directory / "tokenizer.json"
    if not tokenizer_path.is_file():
        return None
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(tokenizer_path))
    except Exception:  # noqa: BLE001 — any load failure is the same verdict
        return None


def _check_fast_tokenizer(directory: Path) -> dict[str, str]:
    tokenizer = _load_fast_tokenizer(directory)
    return _check(
        "fast_tokenizer",
        tokenizer is not None,
        "tokenizer.json loads with the fast-tokenizer library"
        if tokenizer is not None
        else f"no loadable tokenizer.json under {directory}",
        "ship a fast tokenizer (tokenizer.json): the host calls the "
        "tokenizer with return_offsets_mapping=True, which slow "
        "tokenizers cannot serve",
    )


def _check_offsets(directory: Path) -> dict[str, str]:
    tokenizer = _load_fast_tokenizer(directory)
    if tokenizer is None:
        return _check(
            "offsets_available",
            False,
            "cannot check offsets: no loadable fast tokenizer",
            "fix the fast-tokenizer asset first (see fast_tokenizer)",
        )
    try:
        encoding = tokenizer.encode(PROBE_TEXT, add_special_tokens=False)
    except Exception as e:  # noqa: BLE001 — encode failure = offsets unusable
        return _check(
            "offsets_available",
            False,
            f"encoding the probe text failed: {e}",
            "the tokenizer must encode plain text with character offsets "
            "(host force-keep locates fact spans through them)",
        )
    spans = [span for span in encoding.offsets if span[1] > span[0]]
    ok = len(encoding.tokens) > 0 and bool(spans)
    return _check(
        "offsets_available",
        ok,
        f"probe encodes to {len(encoding.tokens)} tokens, "
        f"{len(spans)} non-degenerate offset spans",
        "the fast tokenizer yields no usable offsets for the probe text — "
        "re-export the tokenizer (normalizer/pre-tokenizer drops everything)",
    )


def run_static_checks(directory: Path) -> list[dict[str, str]]:
    """The four drop-in contract items, default dependencies only."""
    config = _load_config(directory)
    return [
        _check_architecture(config),
        _check_num_labels(config),
        _check_id2label_semantics(config),
        _check_fast_tokenizer(directory),
        _check_offsets(directory),
    ]


# ── load layer (issue #14) ───────────────────────────────────────────────


def load_deps_available() -> bool:
    """True when the optional heavy deps (transformers + torch) import."""
    return all(
        importlib.util.find_spec(name) is not None for name in ("transformers", "torch")
    )


def run_load_checks(directory: Path) -> dict[str, Any]:
    """Real instantiation layer: only runs when heavy deps are installed.

    Returns one of three states — pass / violation / unverified. The
    "unverified" state never counts as pass.
    """
    if not load_deps_available():
        return {
            "status": "unverified",
            "reason": (
                "transformers/torch not installed — enable with "
                f"`{LOAD_DEPS_INSTALL_HINT}` to verify by real instantiation"
            ),
            "checks": [],
        }
    checks: list[dict[str, str]] = []
    try:
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForTokenClassification,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        offsets = tokenizer(PROBE_TEXT, return_offsets_mapping=True)["offset_mapping"]
        checks.append(
            _check(
                "fast_tokenizer_runtime",
                getattr(tokenizer, "is_fast", False) and bool(offsets),
                f"is_fast={getattr(tokenizer, 'is_fast', None)}, "
                f"{len(offsets)} offset spans",
                "the runtime tokenizer must be fast with usable offsets "
                "(return_offsets_mapping)",
            )
        )
        model = AutoModelForTokenClassification.from_pretrained(
            str(directory), local_files_only=True
        )
        checks.append(
            _check(
                "dot_bert_attribute",
                hasattr(model, "bert"),
                f"model type {type(model).__name__} "
                f"{'exposes' if hasattr(model, 'bert') else 'lacks'} `.bert`",
                "the host similarity guardrail calls model.bert(**inputs) "
                "directly — only BertForTokenClassification-family bases "
                "qualify (decisions.md §6; XLM-R's .roberta fails here)",
            )
        )
        violations = [c["id"] for c in checks if c["status"] == "violation"]
        return {
            "status": "violation" if violations else "pass",
            "reason": "" if not violations else "; ".join(violations),
            "checks": checks,
        }
    except Exception as e:  # noqa: BLE001 — instantiation failure is a verdict
        return {
            "status": "violation",
            "reason": f"instantiation failed: {e}",
            "checks": checks,
        }


# ── verify report ────────────────────────────────────────────────────────


def verify_checkpoint(directory: Path, *, include_load: bool = True) -> dict[str, Any]:
    """Full verify report for a candidate checkpoint directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise DeliveryError(f"{directory}: no such checkpoint directory")
    files = scan_checkpoint_files(directory)
    static_checks = run_static_checks(directory)
    if include_load:
        load_layer = run_load_checks(directory)
    else:
        load_layer = {
            "status": "unverified",
            "reason": "static-only verification requested (--static-only)",
            "checks": [],
        }
    static_status = (
        "violation"
        if any(c["status"] == "violation" for c in static_checks)
        else "pass"
    )
    overall = (
        "violation"
        if static_status == "violation" or load_layer["status"] == "violation"
        else "pass"
    )
    return {
        "schema_version": DELIVERY_REPORT_SCHEMA_VERSION,
        "harness": {
            "name": "trillic",
            "version": __version__,
            "python": sys.version.split()[0],
        },
        "checkpoint": {
            "path": str(directory),
            "digest": checkpoint_digest(files),
            "file_count": len(files),
            "total_bytes": sum(f["bytes"] for f in files),
            "files": files,
        },
        "layers": {
            "static": {"status": static_status, "checks": static_checks},
            "load": load_layer,
        },
        "overall": overall,
    }


def verify_report_json(report: dict[str, Any]) -> str:
    """Deterministic serialization (sorted keys, no timestamps)."""
    return json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _print_checks(checks: list[dict[str, str]]) -> None:
    for check in checks:
        marker = "PASS" if check["status"] == "pass" else "VIOLATION"
        print(f"  {check['id']}: {marker} — {check['detail']}")
        if check["status"] == "violation":
            print(f"    remedy: {check['remedy']}")


def print_verify_summary(report: dict[str, Any]) -> None:
    """Human-readable verdict on stdout (machine JSON goes to --report)."""
    checkpoint = report["checkpoint"]
    print(
        f"checkpoint: {checkpoint['path']} "
        f"(digest {checkpoint['digest'][:12]}…, "
        f"{checkpoint['file_count']} files)"
    )
    layers = report["layers"]
    static = layers["static"]
    failed = [c for c in static["checks"] if c["status"] == "violation"]
    passed = len(static["checks"]) - len(failed)
    print(f"static layer: {static['status'].upper()} ({passed}/{len(static['checks'])} checks)")
    _print_checks(static["checks"])
    load = layers["load"]
    if load["status"] == "unverified":
        print(f"load layer: UNVERIFIED — {load['reason']}")
    else:
        print(f"load layer: {load['status'].upper()}")
        _print_checks(load["checks"])
    print(f"overall: {report['overall'].upper()}")


def verify_error_message(report: dict[str, Any]) -> str:
    """stderr line naming the violated contract items (for scripts)."""
    static_failed = [
        c["id"] for c in report["layers"]["static"]["checks"] if c["status"] == "violation"
    ]
    parts = list(static_failed)
    if report["layers"]["load"]["status"] == "violation":
        parts.append(f"load:{report['layers']['load'].get('reason') or 'violation'}")
    return (
        "delivery verify: drop-in contract violations: " + ", ".join(parts)
        if parts
        else "delivery verify: violation"
    )


# ── pack (issue #13) ─────────────────────────────────────────────────────


def build_sums_text(files: list[dict[str, Any]]) -> str:
    """`sha256sum -c` compatible manifest, one `<sha256>  <name>` per line."""
    return "".join(f"{file['sha256']}  {file['name']}\n" for file in files)


def model_identity(directory: Path, config: dict[str, Any]) -> tuple[str, str]:
    """(model_id, model_tag) — single source is the checkpoint itself.

    model_id: config.json ``_name_or_path`` when present, else the
    directory name. model_tag: the basename (snapshot dir / metric
    label form, e.g. "trillic-v1").
    """
    name_or_path = config.get("_name_or_path")
    model_id = str(name_or_path) if name_or_path else Path(directory).name
    tag = Path(model_id.rstrip("/")).name or model_id
    return model_id, tag


def render_integration_pack(
    draft_text: str, replacements: dict[str, str], draft_name: str
) -> str:
    """Placeholder substitution with refusal-to-half-render semantics.

    Unknown placeholders abort the pack (the finalized pack must carry
    real values, never silent leftovers), and the result is re-scanned
    so a future draft edit cannot leak placeholders either.
    """
    unknown = sorted(
        {
            match.group(1)
            for match in _PLACEHOLDER_RE.finditer(draft_text)
            if match.group(1) not in replacements
        }
    )
    if unknown:
        raise DeliveryError(
            f"{draft_name}: placeholders this harness cannot render: "
            f"{', '.join('{{' + name + '}}' for name in unknown)} — "
            f"known: {', '.join(sorted(replacements))}"
        )
    rendered = _PLACEHOLDER_RE.sub(
        lambda match: replacements[match.group(1)], draft_text
    )
    residual = _PLACEHOLDER_RE.search(rendered)
    if residual:
        raise DeliveryError(
            f"{draft_name}: residual placeholder {residual.group(0)} after rendering"
        )
    return rendered


def pack_checkpoint(
    directory: Path,
    out_dir: Path,
    draft_path: Path = DEFAULT_DRAFT_PATH,
) -> dict[str, Any]:
    """Assemble the delivery form: manifest + finalized integration pack.

    Gated on verify (a failing checkpoint cannot be packed), refuses to
    overwrite an existing output directory, and is deterministic: no
    timestamps, content-addressed values only.
    """
    directory = Path(directory)
    out_dir = Path(out_dir)
    report = verify_checkpoint(directory)
    if report["overall"] != "pass":
        raise DeliveryError(
            "refusing to pack: drop-in contract violations — "
            f"{verify_error_message(report).split(': ', 1)[-1]} "
            f"(run `trillic delivery verify {directory}` for the full report)"
        )
    if out_dir.exists():
        raise DeliveryError(
            f"{out_dir}: output directory already exists — refusing to "
            f"overwrite (packs are immutable artifacts; use a fresh path)"
        )
    try:
        draft_text = Path(draft_path).read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise DeliveryError(
            f"{draft_path}: integration pack draft not found "
            f"(default: {DEFAULT_DRAFT_PATH} from the repo root)"
        ) from e

    files: list[dict[str, Any]] = report["checkpoint"]["files"]
    sums_text = build_sums_text(files)
    sums_sha256 = hashlib.sha256(sums_text.encode("utf-8")).hexdigest()
    model_id, model_tag = model_identity(directory, _load_config(directory))
    replacements = {
        "MODEL_ID": model_id,
        "MODEL_TAG": model_tag,
        "CHECKPOINT_DIGEST": report["checkpoint"]["digest"],
        "SHA256SUMS_SHA256": sums_sha256,
        "CHECKPOINT_FILE_COUNT": str(len(files)),
        "HARNESS_VERSION": __version__,
    }
    pack_md = render_integration_pack(draft_text, replacements, str(draft_path))

    out_dir.mkdir(parents=True)
    (out_dir / SUMS_NAME).write_text(sums_text, encoding="utf-8")
    (out_dir / INTEGRATION_PACK_NAME).write_text(pack_md, encoding="utf-8")
    # The embedded report is the CONTENT-ADDRESSED form: the filesystem
    # location where verification ran is run provenance, not artifact
    # identity — normalizing it to the model tag lets the same content
    # packed from different staging paths produce byte-identical packs.
    # (The standalone `verify --report` output keeps the full path.)
    packed_report = json.loads(json.dumps(report))
    packed_report["checkpoint"]["path"] = model_tag
    (out_dir / VERIFY_REPORT_NAME).write_text(
        verify_report_json(packed_report), encoding="utf-8"
    )
    return {
        "out_dir": out_dir,
        "model_id": model_id,
        "model_tag": model_tag,
        "checkpoint_digest": report["checkpoint"]["digest"],
        "sums_sha256": sums_sha256,
        "file_count": len(files),
    }


def print_pack_summary(result: dict[str, Any]) -> None:
    out_dir = result["out_dir"]
    print(
        f"packed {result['model_id']} -> {out_dir} "
        f"({result['file_count']} files, digest {result['checkpoint_digest'][:12]}…)"
    )
    print(f"  {out_dir / SUMS_NAME} (sha256 {result['sums_sha256']})")
    print(f"  {out_dir / INTEGRATION_PACK_NAME}")
    print(f"  {out_dir / VERIFY_REPORT_NAME}")
