"""`trillic delivery pack` — artifact pack assembly (issue #13).

Black-box CLI tests over file contracts: the pack is gated on verify, the
SHA256SUMS manifest matches `sha256sum -c` format against the checkpoint
bytes, the finalized integration pack carries real values (no residual
placeholders), and packing the same input twice is byte-identical
(a pack is an auditable artifact, not a random product). Rendering is
local text replacement — zero network, zero gateway.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest
from helpers import VALID_CHECKPOINT_CONFIG, make_checkpoint

from trillic import __version__
from trillic.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT = REPO_ROOT / "docs" / "delivery" / "refine-integration.md"

PACK_FILES = {"SHA256SUMS", "integration-pack.md", "verify-report.json"}


def run_pack(checkpoint: Path, out: Path, *extra: str) -> int:
    return main(
        ["delivery", "pack", str(checkpoint), "--out", str(out), *extra]
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestSuccessPath:
    @pytest.fixture(autouse=True)
    def _pack(self, tmp_path, capsys):
        self.checkpoint = make_checkpoint(tmp_path, "valid-ckpt")
        self.out = tmp_path / "pack"
        self.exit_code = run_pack(self.checkpoint, self.out)
        assert self.exit_code == 0
        capsys.readouterr()

    def test_pack_contains_manifest_integration_pack_and_report(self):
        assert {p.name for p in self.out.iterdir()} == PACK_FILES

    def test_sums_manifest_is_sha256sum_c_compatible(self):
        lines = (self.out / "SHA256SUMS").read_text().splitlines()
        assert lines
        names = set()
        for line in lines:
            digest, _, name = line.partition("  ")
            assert re.fullmatch(r"[0-9a-f]{64}", digest)
            names.add(name)
            assert digest == sha256_file(self.checkpoint / name)
        assert names == {
            p.relative_to(self.checkpoint).as_posix()
            for p in self.checkpoint.rglob("*")
            if p.is_file()
        }

    def test_integration_pack_has_no_residual_placeholders(self):
        text = (self.out / "integration-pack.md").read_text()
        assert not re.search(r"\{\{[A-Z][A-Z0-9_]*\}\}", text), [
            m.group(0) for m in re.finditer(r"\{\{[A-Z][A-Z0-9_]*\}\}", text)
        ]

    def test_integration_pack_carries_real_values(self):
        text = (self.out / "integration-pack.md").read_text()
        assert VALID_CHECKPOINT_CONFIG["_name_or_path"] in text  # model identity
        assert "trillic-v1" in text  # model tag (from _name_or_path basename)
        assert __version__ in text  # rendering harness
        sums_sha = sha256_file(self.out / "SHA256SUMS")
        assert sums_sha in text  # manifest hash
        # directory digest: same caliber the verify report records
        report = json.loads((self.out / "verify-report.json").read_text())
        assert report["checkpoint"]["digest"] in text
        assert report["overall"] == "pass"

    def test_stdout_summarizes_the_pack(self, capsys):
        exit_code = run_pack(self.checkpoint, self.out.with_name("pack2"))
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "SHA256SUMS" in out
        assert "integration-pack.md" in out


class TestDeterminism:
    def test_same_input_packs_byte_identical(self, tmp_path):
        checkpoint = make_checkpoint(tmp_path, "valid-ckpt")
        first, second = tmp_path / "pack-a", tmp_path / "pack-b"
        assert run_pack(checkpoint, first) == 0
        assert run_pack(checkpoint, second) == 0
        for name in PACK_FILES:
            assert (first / name).read_bytes() == (second / name).read_bytes()

    def test_same_content_at_a_different_path_packs_identically(self, tmp_path):
        """Packs are content-addressed artifacts: the staging path where
        verification ran is provenance, not artifact identity."""
        checkpoint = make_checkpoint(tmp_path, "staging-a")
        twin = tmp_path / "staging-b"
        shutil.copytree(checkpoint, twin)
        first, second = tmp_path / "pack-a", tmp_path / "pack-b"
        assert run_pack(checkpoint, first) == 0
        assert run_pack(twin, second) == 0
        for name in PACK_FILES:
            assert (first / name).read_bytes() == (second / name).read_bytes()


class TestRefusals:
    def test_contract_violation_refuses_without_creating_out(self, tmp_path, capsys):
        checkpoint = make_checkpoint(
            tmp_path,
            "bad-arch",
            config=VALID_CHECKPOINT_CONFIG
            | {"architectures": ["RobertaForTokenClassification"]},
        )
        out = tmp_path / "pack"
        exit_code = run_pack(checkpoint, out)
        assert exit_code != 0
        assert not out.exists()
        assert "verify" in capsys.readouterr().err

    def test_existing_output_dir_refuses_and_leaves_it_alone(self, tmp_path, capsys):
        checkpoint = make_checkpoint(tmp_path, "valid-ckpt")
        out = tmp_path / "pack"
        out.mkdir()
        (out / "sentinel.txt").write_text("keep me")
        exit_code = run_pack(checkpoint, out)
        assert exit_code != 0
        err = capsys.readouterr().err
        assert "already exists" in err
        assert [p.name for p in out.iterdir()] == ["sentinel.txt"]

    def test_missing_draft_is_an_error(self, tmp_path, capsys):
        checkpoint = make_checkpoint(tmp_path, "valid-ckpt")
        exit_code = run_pack(
            checkpoint, tmp_path / "pack", "--draft", str(tmp_path / "nope.md")
        )
        assert exit_code != 0
        assert "error:" in capsys.readouterr().err
        assert not (tmp_path / "pack").exists()

    def test_missing_checkpoint_dir_is_an_error(self, tmp_path, capsys):
        exit_code = run_pack(tmp_path / "nope", tmp_path / "pack")
        assert exit_code != 0
        assert "error:" in capsys.readouterr().err


class TestDraftPlaceholders:
    def test_unknown_placeholder_refuses_to_half_render(self, tmp_path, capsys):
        checkpoint = make_checkpoint(tmp_path, "valid-ckpt")
        draft = tmp_path / "draft.md"
        draft.write_text("pack for {{MODEL_TAG}} and {{SOMETHING_NEW}}\n")
        exit_code = run_pack(checkpoint, tmp_path / "pack", "--draft", str(draft))
        assert exit_code != 0
        err = capsys.readouterr().err
        assert "SOMETHING_NEW" in err
        assert not (tmp_path / "pack").exists()

    def test_in_repo_draft_carries_expected_placeholders(self):
        text = DRAFT.read_text()
        for name in (
            "MODEL_ID",
            "MODEL_TAG",
            "CHECKPOINT_DIGEST",
            "SHA256SUMS_SHA256",
            "CHECKPOINT_FILE_COUNT",
            "HARNESS_VERSION",
        ):
            assert "{{" + name + "}}" in text
