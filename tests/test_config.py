"""Run configuration loading (TOML) and validation."""

import pytest

from trillic.config import ConfigError, RunConfig, load_config


def write_config(tmp_path, text):
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


class TestDefaults:
    def test_empty_config_yields_defaults(self, tmp_path):
        config = load_config(write_config(tmp_path, ""))
        assert config == RunConfig()

    def test_defaults_are_stub_mode_and_cl100k(self):
        config = RunConfig()
        assert config.sidecar_mode == "stub"
        assert config.gateway_mode == "stub"
        assert config.tiktoken_encoding == "cl100k_base"
        assert config.aggressiveness == 0.2
        assert (config.sidecar_rewrite, config.sidecar_compress) == (False, True)

    def test_missing_file_reports_path(self, tmp_path):
        with pytest.raises(ConfigError, match="no such file"):
            load_config(tmp_path / "missing.toml")


class TestParsing:
    def test_full_config_is_parsed_field_by_field(self, tmp_path):
        path = write_config(
            tmp_path,
            """
name = "smoke"
[harness]
seed = 7
tiktoken_encoding = "o200k_base"
[run]
aggressiveness = 0.4
[sidecar]
mode = "http"
url = "http://sc:1"
rewrite = true
compress = true
[gateway]
mode = "http"
url = "http://gw:2"
""",
        )
        config = load_config(path)
        assert config.name == "smoke"
        assert config.seed == 7
        assert config.tiktoken_encoding == "o200k_base"
        assert config.aggressiveness == 0.4
        assert config.sidecar_mode == "http"
        assert config.sidecar_url == "http://sc:1"
        assert (config.sidecar_rewrite, config.sidecar_compress) == (True, True)
        assert config.gateway_mode == "http"
        assert config.gateway_url == "http://gw:2"

    def test_fixture_config_parses(self, fixture_config_path):
        config = load_config(fixture_config_path)
        assert config.name == "fixture"
        assert config.sidecar_mode == "stub"
        assert config.aggressiveness == 0.2


class TestValidation:
    def test_unknown_key_is_rejected_by_name(self, tmp_path):
        path = write_config(tmp_path, "[sidecar]\nmode = \"stub\"\ntimeout = 30\n")
        with pytest.raises(ConfigError, match="timeout"):
            load_config(path)

    def test_unknown_top_level_section_is_rejected(self, tmp_path):
        path = write_config(tmp_path, "[bogus]\nx = 1\n")
        with pytest.raises(ConfigError, match="bogus"):
            load_config(path)

    @pytest.mark.parametrize(
        "body, bad",
        [
            ("[run]\naggressiveness = 1.5\n", "aggressiveness"),
            ("[run]\naggressiveness = -0.1\n", "aggressiveness"),
            ("[sidecar]\nmode = \"grpc\"\n", "mode"),
            ("[sidecar]\nrewrite = false\ncompress = false\n", "cannot both be false"),
            ("name = \"a b!\"\n", "name"),
        ],
    )
    def test_invalid_values_are_rejected(self, tmp_path, body, bad):
        with pytest.raises(ConfigError, match=bad):
            load_config(write_config(tmp_path, body))

    def test_service_key_never_has_a_config_home(self):
        # The schema must not even know about credentials: keys are env-only.
        # (tiktoken_encoding is a tokenizer name, not a credential field.)
        import dataclasses

        banned = ("key", "secret", "password", "credential")
        assert not any(
            any(b in field.name for b in banned) for field in dataclasses.fields(RunConfig)
        )
