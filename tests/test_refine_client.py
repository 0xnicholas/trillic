"""Refine-sidecar client: HTTP impl (tested via httpx.MockTransport, zero
network) and a deterministic in-process stub.

The evaluation boundary is the sidecar's POST /refine endpoint (issue #1:
"评测只经 sidecar 的 refine HTTP 端点"), matching the contract in
tokencamp-pro/sidecars/refine/server.py's RefineRequest/RefineResponse.
"""

import json

import httpx
import pytest

from trillic.clients.sidecar import (
    HttpRefineClient,
    RefineError,
    RefineResult,
    StubRefineClient,
)

REFINE_RESPONSE = {
    "refined_text": "compressed version",
    "original_tokens": 50,
    "refined_tokens": 20,
    "cached": False,
    "fallback": False,
    "warnings": ["fact-recall guardrail triggered"],
    "refine_model": "test-model",
}


def make_http_client(handler) -> HttpRefineClient:
    return HttpRefineClient(
        base_url="http://sidecar.local:9797",
        transport=httpx.MockTransport(handler),
    )


class TestHttpRefineClient:
    def test_posts_to_refine_endpoint_with_expected_payload(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=REFINE_RESPONSE)

        client = make_http_client(handler)
        result = client.refine(
            "some prompt", rewrite=False, compress=True, aggressiveness=0.3
        )
        assert captured["url"] == "http://sidecar.local:9797/refine"
        assert captured["method"] == "POST"
        assert captured["body"] == {
            "text": "some prompt",
            "rewrite": False,
            "compress": True,
            "aggressiveness": 0.3,
        }
        assert result == RefineResult(
            refined_text="compressed version",
            original_tokens=50,
            refined_tokens=20,
            cached=False,
            fallback=False,
            warnings=("fact-recall guardrail triggered",),
            refine_model="test-model",
        )

    def test_non_2xx_raises_refine_error_with_status_and_body(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"detail": "compressor unavailable"})

        client = make_http_client(handler)
        with pytest.raises(RefineError, match="503.*compressor unavailable"):
            client.refine("x", rewrite=True, compress=False, aggressiveness=0.2)

    def test_connection_failure_raises_refine_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = make_http_client(handler)
        with pytest.raises(RefineError, match="refine sidecar"):
            client.refine("x", rewrite=True, compress=False, aggressiveness=0.2)

    def test_malformed_response_raises_refine_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": "shape"})

        client = make_http_client(handler)
        with pytest.raises(RefineError, match="missing field"):
            client.refine("x", rewrite=True, compress=False, aggressiveness=0.2)


TEN_WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa"


class TestStubRefineClient:
    def test_deterministic_extractive_policy_pinned_by_hand(self):
        # aggressiveness 0.2 -> drop the first 2 words of every 10-word block.
        # Hand-checked: indices 0 and 1 (alpha, beta) are removed.
        result = StubRefineClient().refine(
            TEN_WORDS, rewrite=False, compress=True, aggressiveness=0.2
        )
        assert result.refined_text == "gamma delta epsilon zeta eta theta iota kappa"
        assert result.original_tokens == 10
        assert result.refined_tokens == 8
        assert result.refine_model == "stub-refine"

    def test_output_is_strict_subsequence_of_input(self):
        source = "one two three four five six seven eight nine ten eleven twelve"
        result = StubRefineClient().refine(
            source, rewrite=False, compress=True, aggressiveness=0.3
        )
        kept_words = result.refined_text.split()
        # every kept word must appear in the input, in order (extractive property)
        iterator = iter(source.split())
        assert all(word in iterator for word in kept_words)
        assert len(kept_words) < len(source.split())

    def test_zero_aggressiveness_keeps_everything(self):
        result = StubRefineClient().refine(
            "  alpha   beta  gamma ", rewrite=False, compress=True, aggressiveness=0.0
        )
        assert result.refined_text == "alpha beta gamma"  # whitespace normalized
        assert result.refined_tokens == 3

    def test_full_aggressiveness_keeps_at_least_one_word(self):
        result = StubRefineClient().refine(
            TEN_WORDS, rewrite=False, compress=True, aggressiveness=1.0
        )
        assert result.refined_text == "alpha"
        assert result.refined_tokens == 1

    def test_rewrite_only_returns_text_unchanged(self):
        result = StubRefineClient().refine(
            TEN_WORDS, rewrite=True, compress=False, aggressiveness=0.2
        )
        assert result.refined_text == TEN_WORDS
        assert result.fallback is False
        assert result.cached is False

    def test_both_flags_false_is_rejected(self):
        with pytest.raises(ValueError, match="rewrite and compress"):
            StubRefineClient().refine(
                TEN_WORDS, rewrite=False, compress=False, aggressiveness=0.2
            )

    def test_same_inputs_same_outputs(self):
        a = StubRefineClient().refine(TEN_WORDS, rewrite=False, compress=True, aggressiveness=0.4)
        b = StubRefineClient().refine(TEN_WORDS, rewrite=False, compress=True, aggressiveness=0.4)
        assert a == b
