"""Run configuration: TOML file -> validated RunConfig.

The config is a contract, not a bag: unknown keys are rejected so a typo
never silently changes evaluation behavior. Credentials are deliberately
absent from the schema (the gateway service key is env-only).
"""

import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


class ConfigError(Exception):
    """A run config could not be read or is invalid."""


@dataclass(frozen=True)
class RunConfig:
    name: str = "run"
    seed: int = 0
    tiktoken_encoding: str = "cl100k_base"
    aggressiveness: float = 0.2
    levels: tuple[float, ...] = ()  # empty = sweep [aggressiveness]
    sidecar_mode: str = "stub"
    sidecar_url: str = "http://127.0.0.1:9797"
    sidecar_rewrite: bool = False
    sidecar_compress: bool = True
    gateway_mode: str = "stub"
    gateway_url: str = "http://127.0.0.1:3005"
    native_flavor: str = "word"
    native_vocab: str | None = None
    task_quality: bool = True
    answer_model: str = "stub-answerer"
    judge_model: str = "stub-judge"
    bootstrap_samples: int = 10_000

    def snapshot(self) -> dict:
        return asdict(self)

    def effective_levels(self) -> list[float]:
        """Sweep levels: explicit `run.levels`, else [aggressiveness]."""
        return list(self.levels) if self.levels else [self.aggressiveness]


# Single source of truth for the config schema: section -> {toml key ->
# RunConfig field}. Unknown keys are rejected against this map, and the
# (section, key) -> field lookup can never drift or KeyError.
_SECTION_FIELDS: dict[str, dict[str, str]] = {
    "harness": {"seed": "seed", "tiktoken_encoding": "tiktoken_encoding"},
    "run": {"aggressiveness": "aggressiveness", "levels": "levels"},
    "sidecar": {
        "mode": "sidecar_mode",
        "url": "sidecar_url",
        "rewrite": "sidecar_rewrite",
        "compress": "sidecar_compress",
    },
    "gateway": {"mode": "gateway_mode", "url": "gateway_url"},
    "metrics": {"native_flavor": "native_flavor", "native_vocab": "native_vocab"},
    "quality": {
        "task_quality": "task_quality",
        "answer_model": "answer_model",
        "judge_model": "judge_model",
        "bootstrap_samples": "bootstrap_samples",
    },
}
_NATIVE_FLAVORS = ("word", "wordpiece", "sentencepiece")
_SWEEP_MAX = 0.5  # phase-1 sweep scope (docs/evaluation.md: 0.1-0.5)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def load_config(path: Path) -> RunConfig:
    """Load and validate a TOML run config."""
    path = Path(path)
    try:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"{path}: no such file") from e
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise ConfigError(f"{path}: cannot parse TOML: {e}") from e

    values: dict = {}
    for key in data:
        if key != "name" and key not in _SECTION_FIELDS:
            raise ConfigError(f"{path}: unknown key {key!r}")
    if "name" in data:
        values["name"] = data["name"]

    for section, fields in _SECTION_FIELDS.items():
        section_data = data.get(section, {})
        if not isinstance(section_data, dict):
            raise ConfigError(f"{path}: [{section}] must be a table")
        for key, value in section_data.items():
            if key not in fields:
                raise ConfigError(f"{path}: unknown key {key!r} in [{section}]")
            values[fields[key]] = value
        if (
            section == "run"
            and "levels" in section_data
            and not section_data["levels"]
        ):
            raise ConfigError(
                f"{path}: run.levels must list at least one aggressiveness level"
            )

    config = RunConfig(**values)
    _validate(config, path)
    return config


def _validate(config: RunConfig, path: Path) -> None:
    if not isinstance(config.name, str) or not _NAME_RE.match(config.name):
        raise ConfigError(
            f"{path}: name must match {_NAME_RE.pattern} (filesystem-safe), got {config.name!r}"
        )
    if not 0.0 <= config.aggressiveness <= 1.0:
        raise ConfigError(
            f"{path}: run.aggressiveness must be within [0, 1], got {config.aggressiveness}"
        )
    _validate_levels(config, path)
    _validate_native_tokenizer(config, path)
    _validate_task_quality(config, path)
    if config.sidecar_mode not in ("stub", "http"):
        raise ConfigError(
            f"{path}: sidecar.mode must be 'stub' or 'http', got {config.sidecar_mode!r}"
        )
    if config.gateway_mode not in ("stub", "http"):
        raise ConfigError(
            f"{path}: gateway.mode must be 'stub' or 'http', got {config.gateway_mode!r}"
        )
    if not config.sidecar_rewrite and not config.sidecar_compress:
        raise ConfigError(
            f"{path}: sidecar.rewrite and sidecar.compress cannot both be false "
            "(the refine endpoint rejects that request)"
        )


def _validate_task_quality(config: RunConfig, path: Path) -> None:
    """[quality] knobs (issue #7). The stub-labeled model defaults are
    deliberate: in stub mode they are just deterministic labels, and in
    http mode the real gateway rejects unknown model ids loudly — a
    forgotten pin fails the run, never silently grades with a mystery
    model."""
    if not isinstance(config.task_quality, bool):
        raise ConfigError(
            f"{path}: quality.task_quality must be a boolean, got {config.task_quality!r}"
        )
    for label, model in (
        ("answer_model", config.answer_model),
        ("judge_model", config.judge_model),
    ):
        if not isinstance(model, str) or not model.strip():
            raise ConfigError(
                f"{path}: quality.{label} must be a non-empty string, got {model!r}"
            )
    samples = config.bootstrap_samples
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ConfigError(
            f"{path}: quality.bootstrap_samples must be an integer >= 1, got {samples!r}"
        )


def _validate_levels(config: RunConfig, path: Path) -> None:
    levels = config.levels
    if not levels:
        return
    for level in levels:
        if not isinstance(level, (int, float)) or isinstance(level, bool):
            raise ConfigError(
                f"{path}: run.levels must be numeric, got {level!r}"
            )
        if not 0.0 < level <= _SWEEP_MAX:
            raise ConfigError(
                f"{path}: run.levels entries must be within (0, {_SWEEP_MAX}] "
                f"(phase-1 sweep scope), got {level!r}"
            )
    if len(set(levels)) != len(levels):
        raise ConfigError(
            f"{path}: run.levels entries must be unique, got {list(levels)}"
        )


def _validate_native_tokenizer(config: RunConfig, path: Path) -> None:
    if config.native_flavor not in _NATIVE_FLAVORS:
        raise ConfigError(
            f"{path}: metrics.native_flavor must be one of {list(_NATIVE_FLAVORS)}, "
            f"got {config.native_flavor!r}"
        )
    if config.native_flavor == "word":
        if config.native_vocab is not None:
            raise ConfigError(
                f"{path}: metrics.native_vocab is not used by the 'word' flavor "
                "— drop it"
            )
        return
    if not config.native_vocab or not config.native_vocab.strip():
        raise ConfigError(
            f"{path}: metrics.native_flavor {config.native_flavor!r} "
            "requires a vocab (metrics.native_vocab)"
        )
