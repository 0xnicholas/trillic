"""Teacher distillation pipeline (issue #18): labeling algorithm and QC.

The seams under test are the LLMLingua-2 method ports (arXiv:2403.12968,
label_word.py): deterministic word-level fuzzy matching, VR/AG metrics,
sentence-aligned chunking, and the sequential quality-control filters.
Orchestration (journal resume, CLI) is covered below and in
test_cli_distill.py.
"""

import json
from pathlib import Path

import pytest

from trillic.distill import (
    PROMPT_TEMPLATE,
    PROMPT_VERSION,
    chunk_text,
    distill_metrics,
    label_chunk,
    match_labels,
    quality_control,
    word_tokens,
)
from trillic.tokens import TokenCounter

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def counter() -> TokenCounter:
    import os

    os.environ["TIKTOKEN_CACHE_DIR"] = str(REPO_ROOT / "eval" / "assets" / "tiktoken_cache")
    return TokenCounter("cl100k_base")


class TestWordTokens:
    def test_words_with_punctuation_and_spans(self):
        toks = word_tokens("The quick, brown fox.")
        assert [t.text for t in toks] == ["The", "quick", "brown", "fox", "."]
        # spans are exact offsets into the source (commas are dropped, so
        # coverage is over tokens, not raw concatenation)
        source = "The quick, brown fox."
        for tok in toks:
            assert source[tok.start : tok.end] == tok.text
        assert toks[0].start == 0 and toks[3].end == 20

    def test_commas_are_ignored(self):
        # the reference algorithm drops "," tokens entirely (they match
        # everywhere and carry no signal)
        assert [t.text for t in word_tokens("a, b ,c")] == ["a", "b", "c"]

    def test_contraction_and_hyphen_stay_whole(self):
        assert [t.text for t in word_tokens("don't re-run")] == ["don't", "re-run"]

    def test_empty(self):
        assert word_tokens("   , ") == []


class TestMatchLabels:
    def test_all_keep(self):
        origin = ["the", "quick", "brown", "fox"]
        labels, num_find = match_labels(origin, list(origin), window_size=150)
        assert labels == [True] * 4
        assert num_find == 4

    def test_all_delete(self):
        labels, num_find = match_labels(["a", "b", "c"], [], window_size=150)
        assert labels == [False] * 3
        assert num_find == 0

    def test_partial_delete(self):
        labels, num_find = match_labels(
            ["keep1", "drop1", "keep2", "drop2"], ["keep1", "keep2"], window_size=150
        )
        assert labels == [True, False, True, False]
        assert num_find == 2

    def test_case_insensitive_match(self):
        labels, _ = match_labels(["the", "fox"], ["THE", "Fox"], window_size=150)
        assert labels == [True, True]

    def test_hallucinated_word_counts_against_find_rate(self):
        # a compressed word absent from the origin is not findable: it
        # inflates Variation Rate but never labels anything
        labels, num_find = match_labels(["a", "b"], ["a", "ghost"], window_size=150)
        assert labels == [True, False]
        assert num_find == 1

    def test_duplicate_compressed_words_label_once(self):
        labels, num_find = match_labels(["a", "b"], ["a", "a"], window_size=150)
        assert labels == [True, False]
        assert num_find == 2

    def test_window_bounds_block_far_matches(self):
        # official semantics: the window scans at most window_size
        # positions around prev_idx, clamped at the text ends — a word
        # 200 positions ahead is NOT reachable with window 150
        origin = [f"w{i}" for i in range(300)]
        labels, num_find = match_labels(origin, ["w299"], window_size=150)
        assert labels == [False] * 300
        assert num_find == 1  # membership still counts (VR stays 0)

    def test_window_advance_is_half(self):
        # after a far forward match the window advances by window//2,
        # not to the matched position ("window do not go too fast")
        origin = [f"w{i}" for i in range(200)]
        labels, num_find = match_labels(origin, ["w0", "w160"], window_size=150)
        # w160 is within the forward scan from prev_idx=0 (0..149 clamped
        # to 199: positions 0..149 only) -> unreachable, stays unlabeled
        assert labels[0] is True
        assert labels[160] is False
        assert num_find == 2

    def test_backward_match(self):
        # comp order may jitter slightly: a token just behind prev_idx
        # is still matched by the backward scan
        labels, _ = match_labels(["a", "b", "c"], ["a", "b"], window_size=150)
        assert labels == [True, True, False]

    def test_empty_origin(self):
        labels, num_find = match_labels([], ["a"], window_size=150)
        assert labels == []
        assert num_find == 0


class TestDistillMetrics:
    def test_perfect_alignment(self):
        m = distill_metrics(n_origin=10, n_comp=10, n_kept=10, num_find=10)
        assert m["variation_rate"] == 0.0
        assert m["matching_rate"] == 1.0
        assert m["hitting_rate"] == 1.0
        assert m["alignment_gap"] == 0.0

    def test_empty_compression(self):
        # everything deleted: VR is defined as 1 (no compressed words to
        # verify), MR=0, HR=0, AG=0 — not a hallucination case
        m = distill_metrics(n_origin=10, n_comp=0, n_kept=0, num_find=0)
        assert m["variation_rate"] == 1.0
        assert m["matching_rate"] == 0.0
        assert m["alignment_gap"] == 0.0

    def test_hallucination_and_alignment(self):
        # 5 origin words, comp has 3 words of which 1 is hallucinated;
        # the 2 real ones label 2 origin words
        m = distill_metrics(n_origin=5, n_comp=3, n_kept=2, num_find=2)
        assert m["variation_rate"] == pytest.approx(1 - 2 / 3, abs=1e-4)
        assert m["hitting_rate"] == pytest.approx(2 / 5, abs=1e-4)
        assert m["matching_rate"] == pytest.approx(2 / 5, abs=1e-4)
        assert m["alignment_gap"] == 0.0

    def test_alignment_gap_positive_when_matching_loses_words(self):
        # findable words that the window could not label: HR > MR
        m = distill_metrics(n_origin=5, n_comp=2, n_kept=1, num_find=2)
        assert m["hitting_rate"] == pytest.approx(2 / 5, abs=1e-4)
        assert m["matching_rate"] == pytest.approx(1 / 5)
        assert m["alignment_gap"] == pytest.approx(1 / 5, abs=1e-4)


class TestLabelChunk:
    def test_end_to_end_labels_carry_spans(self):
        text = "Alpha beta gamma. Delta epsilon."
        result = label_chunk(text, "Alpha gamma Delta", window_size=150)
        assert [t.text for t in result.tokens] == [
            "Alpha", "beta", "gamma", ".", "Delta", "epsilon", ".",
        ]
        assert [t.keep for t in result.tokens] == [True, False, True, False, True, False, False]
        # spans are exact offsets into the chunk text
        for tok in result.tokens:
            assert text[tok.start : tok.end] == tok.text
        assert result.metrics["matching_rate"] == pytest.approx(3 / 7, abs=1e-4)

    def test_metrics_consistency_with_manual_call(self):
        text = "one two three four"
        result = label_chunk(text, "one four", window_size=4)
        labels, num_find = match_labels(["one", "two", "three", "four"], ["one", "four"], window_size=4)
        assert [t.keep for t in result.tokens] == labels
        assert result.metrics == distill_metrics(
            n_origin=4, n_comp=2, n_kept=sum(labels), num_find=num_find
        )


class TestChunkText:
    def test_lossless_coverage(self, counter):
        text = (
            "First sentence here. Second one follows! Third? "
            "A longer fourth sentence with many words inside it. "
            "Fifth, with a comma. Sixth ends.\n\nSeventh after a break."
        )
        chunks = chunk_text(text, max_tokens=20, counter=counter)
        assert "".join(c.text for c in chunks) == text
        assert [(c.start, c.end) for c in chunks][0][0] == 0
        # contiguous offsets
        cursor = 0
        for c in chunks:
            assert c.start == cursor
            cursor = c.end
        assert cursor == len(text)

    def test_respects_max_tokens(self, counter):
        text = ". ".join(f"Sentence number {i} has some words in it" for i in range(40)) + "."
        chunks = chunk_text(text, max_tokens=30, counter=counter)
        assert len(chunks) > 1
        for c in chunks:
            assert counter.count(c.text) <= 30
            assert c.text.strip()

    def test_sentence_aligned_when_possible(self, counter):
        # clean sentences of ~6 tokens each, cap 15 -> whole sentences per
        # chunk, every non-final chunk ends with sentence punctuation
        text = " ".join(f"Sentence {i} is short and clear." for i in range(6))
        chunks = chunk_text(text, max_tokens=15, counter=counter)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.text.rstrip().endswith(".")

    def test_oversized_sentence_is_hard_split(self, counter):
        # a single sentence beyond the cap must still be packed into
        # word-boundary chunks of at most max tokens
        text = " ".join(f"word{i}" for i in range(200)) + "."
        chunks = chunk_text(text, max_tokens=16, counter=counter)
        assert len(chunks) > 1
        for c in chunks:
            assert counter.count(c.text) <= 16
        assert "".join(c.text for c in chunks) == text

    def test_short_text_single_chunk(self, counter):
        text = "Tiny prompt."
        chunks = chunk_text(text, max_tokens=512, counter=counter)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_multiline_boundaries(self, counter):
        # prompt-shaped text (dialogue turns) breaks on newlines even
        # without sentence punctuation
        text = "User: hello there\nAssistant: hi how can I help\nUser: billing question"
        chunks = chunk_text(text, max_tokens=8, counter=counter)
        assert len(chunks) >= 2
        assert "".join(c.text for c in chunks) == text
        for c in chunks:
            assert counter.count(c.text) <= 8


class TestQualityControl:
    @staticmethod
    def _row(row_id: str, vr: float, ag: float) -> dict:
        return {
            "id": row_id,
            "metrics": {"variation_rate": vr, "alignment_gap": ag},
        }

    def test_drop_fractions_and_order(self):
        rows = [self._row(f"r{i:02d}", vr=i / 40, ag=i / 50) for i in range(40)]
        kept, report = quality_control(
            rows, vr_drop_fraction=0.05, ag_drop_fraction=0.10
        )
        # 5% of 40 -> 2 by VR; then 10% of 38 -> 4 by AG (ceil)
        assert report["vr_dropped"] == 2
        assert report["ag_dropped"] == 4
        assert len(kept) == 34
        dropped_ids = {r["id"] for r in report["dropped"]}
        # the two highest-VR rows are gone (r39, r38)
        assert {"r39", "r38"} <= dropped_ids
        # thresholds are recorded (audit surface)
        assert report["vr_threshold"] == 37 / 40
        assert report["ag_threshold"] == 33 / 50

    def test_deterministic_tie_break(self):
        rows = [self._row(f"r{i}", vr=0.5, ag=0.1) for i in range(10)]
        kept_a, report_a = quality_control(rows, vr_drop_fraction=0.1, ag_drop_fraction=0.1)
        shuffled = list(reversed(rows))
        kept_b, report_b = quality_control(shuffled, vr_drop_fraction=0.1, ag_drop_fraction=0.1)
        # ties on VR: the highest ids drop; membership is order-independent
        assert sorted(r["id"] for r in kept_a) == sorted(r["id"] for r in kept_b)
        assert {r["id"] for r in kept_a} == {f"r{i}" for i in range(8)}

    def test_zero_fractions_keep_everything(self):
        rows = [self._row("a", 1.0, 1.0), self._row("b", 0.9, 0.9)]
        kept, report = quality_control(rows, vr_drop_fraction=0.0, ag_drop_fraction=0.0)
        assert len(kept) == 2
        assert report["vr_dropped"] == 0 and report["ag_dropped"] == 0

    def test_empty_population(self):
        kept, report = quality_control([], vr_drop_fraction=0.05, ag_drop_fraction=0.1)
        assert kept == []
        assert report["vr_threshold"] is None

    def test_rejects_bad_fractions(self):
        with pytest.raises(ValueError):
            quality_control([self._row("a", 0.1, 0.1)], vr_drop_fraction=-0.1, ag_drop_fraction=0.1)
        with pytest.raises(ValueError):
            quality_control([self._row("a", 0.1, 0.1)], vr_drop_fraction=0.1, ag_drop_fraction=1.5)


class TestPromptPin:
    def test_prompt_template_is_versioned_and_filled(self):
        assert PROMPT_VERSION == "llmlingua2-paper-v1"
        assert "{text}" in PROMPT_TEMPLATE
        filled = PROMPT_TEMPLATE.format(text="hello")
        assert "following text:\nhello\nThe compressed text is:" in filled


# ---------------------------------------------------------------------------
# Orchestration: teacher loop, journal resume, selection, artifacts
# ---------------------------------------------------------------------------

import os

from trillic.clients.gateway import ChatResult, GatewayError
from trillic.distill import (
    distill_ledger_id,
    load_chunk_results,
    load_corpus_rows,
    run_distillation,
    select_entries,
)


def _entry(entry_id: str, load_type: str, prompt: str) -> dict:
    return {
        "id": entry_id,
        "load_type": load_type,
        "prompt": prompt,
        "question": "",
        "task": "",
        "source": {
            "dataset": "fixture",
            "subset": "test",
            "license": "original",
            "split": "train",
            "content_sha1": entry_id,
        },
    }


def _write_corpus(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


SENTENCES = " ".join(
    f"Fixture sentence number {i} carries a number {i * 7} and some words." for i in range(8)
)


class ScriptedTeacher:
    """Deterministic 'teacher': keeps every other word (compresses ~50%)."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, model: str, prompt: str) -> ChatResult:
        self.calls += 1
        tail = prompt.rsplit("following text:\n", 1)[1].rsplit("\nThe compressed text is:", 1)[0]
        words = tail.split(" ")
        content = " ".join(words[::2])
        return ChatResult(content=content, model=f"served/{model}", usage=None)


class CrashingTeacher(ScriptedTeacher):
    """Dies on the Nth call, simulating a killed run mid-flight."""

    def __init__(self, die_at: int) -> None:
        super().__init__()
        self.die_at = die_at

    def chat(self, model: str, prompt: str) -> ChatResult:
        if self.calls >= self.die_at:
            raise GatewayError("simulated kill mid-run")
        return super().chat(model, prompt)


@pytest.fixture
def corpus_paths(tmp_path) -> list[Path]:
    rag = _write_corpus(
        tmp_path,
        "rag.jsonl",
        [_entry(f"rag-{i}", "rag", SENTENCES) for i in range(2)],
    )
    synth = _write_corpus(
        tmp_path,
        "synth.jsonl",
        [
            _entry(f"sys-{i}", "system_prompt", f"System prompt {i}. Be terse and precise.")
            for i in range(3)
        ]
        + [
            _entry(f"dlg-{i}", "dialogue", f"User: turn {i}\nAssistant: reply {i}")
            for i in range(3)
        ],
    )
    return [rag, synth]


class TestLoadCorpusRows:
    def test_loads_and_validates(self, corpus_paths):
        rows, files = load_corpus_rows(corpus_paths)
        assert len(rows) == 8
        assert [f["entries"] for f in files] == [2, 6]
        assert all(len(f["sha256"]) == 64 for f in files)

    def test_rejects_duplicate_ids(self, tmp_path):
        path = _write_corpus(tmp_path, "dup.jsonl", [_entry("a", "rag", "x."), _entry("a", "rag", "y.")])
        with pytest.raises(Exception, match="duplicate"):
            load_corpus_rows([path])

    def test_rejects_bad_row(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"id": "x", "load_type": "rag"}) + "\n", encoding="utf-8")
        with pytest.raises(Exception, match="prompt"):
            load_corpus_rows([path])

    def test_missing_file(self, tmp_path):
        with pytest.raises(Exception, match="no such"):
            load_corpus_rows([tmp_path / "nope.jsonl"])


class TestSelectEntries:
    def test_budget_respected_with_skip_rule(self):
        # entry chunk counts: r1=2, r2=5, r3=2, r4=1 with budget 5 ->
        # r1 (2 used), r2 skipped (5 > 3 left), r3 (4 used), r4 (5 used)
        rows = [
            _entry(f"r{i}", "rag", "text") for i in (1, 2, 3, 4)
        ]
        chunk_counts = {"r1": 2, "r2": 5, "r3": 2, "r4": 1}
        selected = select_entries(rows, chunk_counts, {"rag": 5})
        assert [r["id"] for r in selected] == ["r1", "r3", "r4"]

    def test_unknown_budget_class_rejected(self):
        with pytest.raises(Exception, match="not present in the corpus"):
            select_entries([_entry("r1", "rag", "t")], {"r1": 1}, {"ghost": 3})

    def test_unbudgeted_classes_excluded(self):
        selected = select_entries(
            [_entry("r1", "rag", "t"), _entry("s1", "system_prompt", "t")],
            {"r1": 1, "s1": 1},
            {"rag": 1},
        )
        assert [r["id"] for r in selected] == ["r1"]


class TestLedgerId:
    def test_identity_parts(self):
        a = distill_ledger_id(
            corpus_shas=["aa", "bb"],
            teacher_model="m1",
            prompt_sha256="p" * 64,
            max_chunk_tokens=512,
            encoding="cl100k_base",
        )
        b = distill_ledger_id(
            corpus_shas=["aa", "bb"],
            teacher_model="m2",
            prompt_sha256="p" * 64,
            max_chunk_tokens=512,
            encoding="cl100k_base",
        )
        c = distill_ledger_id(
            corpus_shas=["aa", "bb"],
            teacher_model="m1",
            prompt_sha256="p" * 64,
            max_chunk_tokens=256,
            encoding="cl100k_base",
        )
        d = distill_ledger_id(
            corpus_shas=["bb", "aa"],  # order matters: file identity order
            teacher_model="m1",
            prompt_sha256="p" * 64,
            max_chunk_tokens=512,
            encoding="cl100k_base",
        )
        assert len(a) == 16 and a != b != c
        assert a != d


class TestRunDistillation:
    @staticmethod
    def _run(corpus_paths, tmp_path, gateway, **overrides):
        params = dict(
            corpus_paths=corpus_paths,
            chunk_budgets={"rag": 4, "system_prompt": 3, "dialogue": 3},
            gateway=gateway,
            teacher_model="fixture/teacher",
            out_dir=tmp_path / "out",
            ledger_root=tmp_path / "ledger",
            counter=TokenCounter("cl100k_base"),
            window_size=40,
            max_chunk_tokens=32,
            vr_drop_fraction=0.0,
            ag_drop_fraction=0.0,
            concurrency=1,
        )
        params.update(overrides)
        return run_distillation(**params)

    def test_end_to_end_artifacts(self, corpus_paths, tmp_path):
        teacher = ScriptedTeacher()
        manifest = self._run(corpus_paths, tmp_path, teacher)
        out = tmp_path / "out"
        assert teacher.calls == manifest["gateway"]["calls_fresh"]
        assert manifest["gateway"]["calls_reused"] == 0
        rows = [json.loads(l) for l in (out / "labeled.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == manifest["outputs"]["labeled"]["rows"]
        for row in rows:
            assert row["prompt"]
            assert row["compressed_text"]
            assert row["tokens"] and all(len(t) == 4 for t in row["tokens"])
            keeps = [t[3] for t in row["tokens"]]
            assert set(keeps) <= {0, 1}
            assert row["question"] == "" and row["task"] == ""
            assert row["source"]["split"] == "train"
            assert row["chunk"]["index"] >= 0
            assert isinstance(row["metrics"]["alignment_gap"], float)
        # every other word kept: labels exist in both states somewhere
        all_keeps = [t[3] for r in rows for t in r["tokens"]]
        assert 0 in all_keeps and 1 in all_keeps
        # manifest pins the method surface
        assert manifest["teacher"]["model"] == "fixture/teacher"
        assert manifest["teacher"]["prompt_version"] == PROMPT_VERSION
        assert len(manifest["teacher"]["prompt_sha256"]) == 64
        assert manifest["teacher"]["served_models"] == ["served/fixture/teacher"]
        assert manifest["quality_control"]["vr_drop_fraction"] == 0.0
        assert (out / "report.md").is_file() and (out / "report.json").is_file()

    def test_budget_bounds_gateway_calls(self, corpus_paths, tmp_path):
        teacher = ScriptedTeacher()
        manifest = self._run(corpus_paths, tmp_path, teacher)
        assert teacher.calls == manifest["counts"]["chunks"]
        assert manifest["counts"]["by_load_type"]["rag"]["chunks"] <= 4

    def test_kill_and_rerun_zero_duplicate_billing(self, corpus_paths, tmp_path):
        # reference: a clean run of everything
        reference_teacher = ScriptedTeacher()
        reference = self._run(corpus_paths, tmp_path, reference_teacher)
        total = reference["counts"]["chunks"]
        assert total >= 6

        # phase 1: die mid-run; journal keeps every completed chunk
        work = tmp_path / "kill"
        work.mkdir()
        corpus2 = [
            _write_corpus(work, p.name, load_corpus_rows([p])[0])
            for p in corpus_paths
        ]
        out1 = work / "out"
        crashing = CrashingTeacher(die_at=3)
        params = dict(
            corpus_paths=corpus2,
            chunk_budgets={"rag": 4, "system_prompt": 3, "dialogue": 3},
            gateway=crashing,
            teacher_model="fixture/teacher",
            out_dir=out1,
            ledger_root=work / "ledger",
            counter=TokenCounter("cl100k_base"),
            window_size=40,
            max_chunk_tokens=32,
            vr_drop_fraction=0.0,
            ag_drop_fraction=0.0,
        )
        with pytest.raises(GatewayError, match="simulated kill"):
            run_distillation(**params)
        journal = next((work / "ledger" / ".ledger").glob("distill-*.jsonl"))
        recorded = load_chunk_results(journal)
        assert len(recorded) == 3  # died on the 4th call, 3 journaled
        assert not out1.exists()  # nothing half-written

        # phase 2: fresh teacher, same identity -> journal replays
        resumed = ScriptedTeacher()
        manifest2 = run_distillation(**{**params, "gateway": resumed})
        assert resumed.calls == total - 3  # ONLY the unrecorded chunks billed
        assert manifest2["gateway"]["calls_fresh"] == total - 3
        assert manifest2["gateway"]["calls_reused"] == 3
        # the dataset is identical to the clean reference
        ref_rows = (tmp_path / "out" / "labeled.jsonl").read_text(encoding="utf-8")
        new_rows = (out1 / "labeled.jsonl").read_text(encoding="utf-8")
        assert new_rows == ref_rows

    def test_teacher_change_invalidates_replay(self, corpus_paths, tmp_path):
        self._run(corpus_paths, tmp_path, ScriptedTeacher())
        second = ScriptedTeacher()
        with pytest.raises(Exception, match="already exists"):
            self._run(corpus_paths, tmp_path, second)
        # different output dir, different teacher: full re-bill (new ledger)
        manifest = self._run(
            corpus_paths, tmp_path, second, teacher_model="other/teacher",
            out_dir=tmp_path / "out2",
        )
        assert manifest["gateway"]["calls_reused"] == 0
        assert manifest["gateway"]["calls_fresh"] == second.calls

    def test_same_ledger_replays_without_out_dir(self, corpus_paths, tmp_path):
        # same identity into a THIRD out dir: everything replays (the
        # journal, not the output dir, is the ledger of record)
        first = ScriptedTeacher()
        self._run(corpus_paths, tmp_path, first)
        replayed = ScriptedTeacher()
        manifest = self._run(corpus_paths, tmp_path, replayed, out_dir=tmp_path / "out3")
        assert replayed.calls == 0
        assert manifest["gateway"]["calls_reused"] == manifest["counts"]["chunks"]

    def test_qc_filtering_drops_and_records(self, tmp_path):
        # one tiny corpus where every chunk shares metrics -> ties drop
        # deterministically by id
        rows = [
            _entry(f"s{i}", "system_prompt", f"Prompt {i} here. Keep it short.")
            for i in range(10)
        ]
        corpus = [_write_corpus(tmp_path, "tie.jsonl", rows)]
        manifest = self._run(
            corpus, tmp_path, ScriptedTeacher(),
            chunk_budgets={"system_prompt": 10},
            vr_drop_fraction=0.2, ag_drop_fraction=0.0,
        )
        assert manifest["counts"]["labeled_kept"] == 8
        report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
        assert report["quality_control"]["vr_dropped"] == 2
        assert len(report["quality_control"]["dropped"]) == 2

    def test_concurrency_matches_sequential_results(self, corpus_paths, tmp_path):
        seq = ScriptedTeacher()
        m1 = self._run(corpus_paths, tmp_path, seq)
        par = ScriptedTeacher()
        m2 = self._run(
            corpus_paths, tmp_path, par,
            out_dir=tmp_path / "out-par", ledger_root=tmp_path / "ledger-par",
            concurrency=3,
        )
        assert par.calls == seq.calls
        a = (tmp_path / "out" / "labeled.jsonl").read_text(encoding="utf-8")
        b = (tmp_path / "out-par" / "labeled.jsonl").read_text(encoding="utf-8")
        assert a == b
