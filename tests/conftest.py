"""Shared test fixtures.

The suite must be zero-network: the tiktoken encoding cache is pinned to the
committed repo blob so no test ever downloads from the OpenAI blob store.
Importable checkpoint builders for the delivery tests live in tests/helpers.py.
"""

import json
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


SUBSETS_TOML = """
source_url = "https://example.org/data.zip"

[[subset]]
name = "qasper"
file = "qasper_e.jsonl"
license = "CC BY-NC 4.0 (QASPER)"
train_use = false
note = "eval-only"

[[subset]]
name = "hotpotqa"
file = "hotpotqa_e.jsonl"
license = "CC BY-SA 4.0 (HotpotQA)"
train_use = true
note = "share-alike noted"
"""


def write_subset_file(path: Path, n: int, prefix: str) -> None:
    with open(path, "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "input": f"question about {prefix} number {i}?",
                        "context": f"{prefix} document {i} body. It mentions 1970 and {i * 7} units.",
                        "answers": [f"answer {i} with number {i * 3}"],
                        "length": 100,
                    }
                )
                + "\n"
            )


@pytest.fixture
def subsets_toml(tmp_path) -> Path:
    path = tmp_path / "subsets.toml"
    path.write_text(SUBSETS_TOML)
    return path


@pytest.fixture
def data_dir(tmp_path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    write_subset_file(data / "qasper_e.jsonl", 30, "qasper")
    write_subset_file(data / "hotpotqa_e.jsonl", 30, "hotpot")
    return data
