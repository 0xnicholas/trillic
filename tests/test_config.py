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


class TestSweepLevels:
    def test_levels_default_to_single_aggressiveness(self, tmp_path):
        config = load_config(write_config(tmp_path, "[run]\naggressiveness = 0.3\n"))
        assert config.effective_levels() == [0.3]

    def test_unset_levels_effective_default(self):
        assert RunConfig().effective_levels() == [0.2]

    def test_explicit_sweep_parses_in_order(self, tmp_path):
        path = write_config(
            tmp_path,
            "[run]\nlevels = [0.1, 0.2, 0.3, 0.4, 0.5]\n",
        )
        config = load_config(path)
        assert config.effective_levels() == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_empty_levels_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="levels"):
            load_config(write_config(tmp_path, "[run]\nlevels = []\n"))

    def test_out_of_scope_level_is_rejected(self, tmp_path):
        # Sweep scope is pinned to 0.1-0.5 (docs/evaluation.md: 0.5 以上档位
        # out of scope for this phase); the config is the guard.
        with pytest.raises(ConfigError, match="levels"):
            load_config(write_config(tmp_path, "[run]\nlevels = [0.1, 0.9]\n"))

    def test_non_numeric_level_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="levels"):
            load_config(write_config(tmp_path, "[run]\nlevels = [0.1, \"x\"]\n"))

    def test_duplicate_levels_are_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="unique"):
            load_config(write_config(tmp_path, "[run]\nlevels = [0.2, 0.2]\n"))


class TestNativeTokenizerConfig:
    def test_default_flavor_is_word(self, tmp_path):
        config = load_config(write_config(tmp_path, ""))
        assert config.native_flavor == "word"
        assert config.native_vocab is None

    def test_flavor_and_vocab_parse(self, tmp_path):
        path = write_config(
            tmp_path,
            '[metrics]\nnative_flavor = "wordpiece"\nnative_vocab = "assets/vocab.txt"\n',
        )
        config = load_config(path)
        assert config.native_flavor == "wordpiece"
        assert config.native_vocab == "assets/vocab.txt"

    def test_unknown_flavor_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="native_flavor"):
            load_config(write_config(tmp_path, '[metrics]\nnative_flavor = "bpe"\n'))

    @pytest.mark.parametrize("flavor", ["wordpiece", "sentencepiece"])
    def test_subword_flavors_require_vocab(self, tmp_path, flavor):
        with pytest.raises(ConfigError, match="requires a vocab"):
            load_config(write_config(tmp_path, f'[metrics]\nnative_flavor = "{flavor}"\n'))

    def test_word_flavor_rejects_stray_vocab(self, tmp_path):
        with pytest.raises(ConfigError, match="native_vocab"):
            load_config(
                write_config(
                    tmp_path,
                    '[metrics]\nnative_flavor = "word"\nnative_vocab = "v.txt"\n',
                )
            )


class TestTaskQualityConfig:
    """[quality] section (issue #7): task-level quality loop knobs — judge
    and answer model pins, bootstrap size, and the off switch."""

    def test_defaults_enable_the_loop_with_stub_labels(self):
        config = RunConfig()
        assert config.task_quality is True
        assert config.answer_model == "stub-answerer"
        assert config.judge_model == "stub-judge"
        assert config.bootstrap_samples == 10_000

    def test_quality_section_parses(self, tmp_path):
        path = write_config(
            tmp_path,
            '[quality]\n'
            'task_quality = false\n'
            'answer_model = "gpt-x"\n'
            'judge_model = "gpt-judge"\n'
            'bootstrap_samples = 5000\n',
        )
        config = load_config(path)
        assert config.task_quality is False
        assert config.answer_model == "gpt-x"
        assert config.judge_model == "gpt-judge"
        assert config.bootstrap_samples == 5000

    def test_unknown_quality_key_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="judge_temperature"):
            load_config(
                write_config(
                    tmp_path, '[quality]\njudge_temperature = 0.0\n'
                )
            )

    @pytest.mark.parametrize(
        "body, bad",
        [
            ('[quality]\ntask_quality = "yes"\n', "task_quality"),
            ('[quality]\ntask_quality = 1\n', "task_quality"),
            ('[quality]\nanswer_model = ""\n', "answer_model"),
            ('[quality]\njudge_model = ""\n', "judge_model"),
            ('[quality]\nbootstrap_samples = 0\n', "bootstrap_samples"),
            ('[quality]\nbootstrap_samples = true\n', "bootstrap_samples"),
            ('[quality]\nbootstrap_samples = 1.5\n', "bootstrap_samples"),
        ],
    )
    def test_invalid_quality_values_rejected(self, tmp_path, body, bad):
        with pytest.raises(ConfigError, match=bad):
            load_config(write_config(tmp_path, body))


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
