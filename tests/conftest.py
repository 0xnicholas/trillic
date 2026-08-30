"""Shared test fixtures.

The suite must be zero-network: the tiktoken encoding cache is pinned to the
committed repo blob so no test ever downloads from the OpenAI blob store.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

os.environ["TIKTOKEN_CACHE_DIR"] = str(REPO_ROOT / "eval" / "assets" / "tiktoken_cache")

import pytest  # noqa: E402


@pytest.fixture
def fixture_golden_path() -> Path:
    return REPO_ROOT / "eval" / "golden" / "fixture.jsonl"


@pytest.fixture
def fixture_config_path() -> Path:
    return REPO_ROOT / "eval" / "configs" / "fixture.toml"
