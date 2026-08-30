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
    sidecar_mode: str = "stub"
    sidecar_url: str = "http://127.0.0.1:9797"
    sidecar_rewrite: bool = False
    sidecar_compress: bool = True
    gateway_mode: str = "stub"
    gateway_url: str = "http://127.0.0.1:3005"

    def snapshot(self) -> dict:
        return asdict(self)


# Single source of truth for the config schema: section -> {toml key ->
# RunConfig field}. Unknown keys are rejected against this map, and the
# (section, key) -> field lookup can never drift or KeyError.
_SECTION_FIELDS: dict[str, dict[str, str]] = {
    "harness": {"seed": "seed", "tiktoken_encoding": "tiktoken_encoding"},
    "run": {"aggressiveness": "aggressiveness"},
    "sidecar": {
        "mode": "sidecar_mode",
        "url": "sidecar_url",
        "rewrite": "sidecar_rewrite",
        "compress": "sidecar_compress",
    },
    "gateway": {"mode": "gateway_mode", "url": "gateway_url"},
}
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
