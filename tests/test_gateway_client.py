"""Gateway LLM client: HTTP impl (tested via httpx.MockTransport, zero
network) and a deterministic stub.

Contract follows the old harness (tokencamp compression/eval/run_eval.py):
POST {base}/v1/chat/completions with a bearer service key that comes ONLY
from the REFINE_SERVICE_KEY environment variable (never from config files),
and the X-TC-Refine: false header so eval/judge calls are never refined.

Since issue #7 the stub speaks the judge protocol: prompts on the
TRILLIC-JUDGE envelope get mechanically graded JSON back (deterministic,
zero cost); everything else is echoed. That is what makes the task-quality
loop end-to-end demonstrable with zero network.
"""

import json

import httpx
import pytest

from trillic.clients.gateway import (
    GatewayError,
    HttpGatewayClient,
    StubGatewayClient,
)
from trillic.judge import (
    JudgeError,
    judge_prompt,
    mechanical_judge_response,
    parse_judge_scores,
)

COMPLETION_RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "model answer"}}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
}


def make_client(handler, **kwargs) -> HttpGatewayClient:
    return HttpGatewayClient(
        base_url="http://gateway.local:3005",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


class TestHttpGatewayClient:
    def test_posts_chat_completion_with_auth_and_no_refine_header(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=COMPLETION_RESPONSE)

        client = make_client(handler, service_key="secret-key")
        result = client.chat(model="task-model", prompt="question?")

        assert captured["headers"]["authorization"] == "Bearer secret-key"
        assert captured["headers"]["x-tc-refine"] == "false"
        assert captured["body"] == {
            "model": "task-model",
            "messages": [{"role": "user", "content": "question?"}],
            "stream": False,
        }
        assert result.content == "model answer"
        assert result.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}

    def test_served_model_is_recorded_over_the_requested_id(self):
        """Issue #7 pin discipline: the gateway's served model id (e.g. an
        alias resolving to a dated snapshot) is the version of record."""

        def handler(request: httpx.Request) -> httpx.Response:
            response = dict(COMPLETION_RESPONSE)
            response["model"] = "task-model-2026-08-01"
            return httpx.Response(200, json=response)

        client = make_client(handler, service_key="k")
        result = client.chat(model="task-model", prompt="q?")
        assert result.model == "task-model-2026-08-01"

    def test_missing_served_model_falls_back_to_requested_id(self):
        client = make_client(
            handler=lambda request: httpx.Response(200, json=COMPLETION_RESPONSE),
            service_key="k",
        )
        result = client.chat(model="task-model", prompt="q?")
        assert result.model == "task-model"

    def test_service_key_defaults_to_env_variable(self, monkeypatch):
        monkeypatch.setenv("REFINE_SERVICE_KEY", "env-key")

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"] == "Bearer env-key"
            return httpx.Response(200, json=COMPLETION_RESPONSE)

        client = make_client(handler)
        client.chat(model="m", prompt="p")

    def test_missing_service_key_is_a_clear_error(self, monkeypatch):
        monkeypatch.delenv("REFINE_SERVICE_KEY", raising=False)
        client = make_client(handler=lambda request: httpx.Response(200))
        with pytest.raises(GatewayError, match="REFINE_SERVICE_KEY"):
            client.chat(model="m", prompt="p")

    def test_non_2xx_raises_gateway_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "bad key"})

        client = make_client(handler, service_key="k")
        with pytest.raises(GatewayError, match="401.*bad key"):
            client.chat(model="m", prompt="p")

    def test_malformed_response_raises_gateway_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"nope": True})

        client = make_client(handler, service_key="k")
        with pytest.raises(GatewayError, match="missing field"):
            client.chat(model="m", prompt="p")


class TestStubGatewayClient:
    def test_deterministic_and_model_aware(self):
        stub = StubGatewayClient()
        a = stub.chat(model="judge-model", prompt="score this")
        b = stub.chat(model="judge-model", prompt="score this")
        assert a == b
        assert "judge-model" in a.content
        assert "score this" in a.content
        assert a.usage is None  # stub performs no billable work

    def test_distinct_models_distinct_outputs(self):
        stub = StubGatewayClient()
        assert stub.chat(model="a", prompt="p") != stub.chat(model="b", prompt="p")

    def test_judge_protocol_prompt_gets_graded_json_back(self):
        """The issue #7 fake-stub seam: a judge-envelope prompt is graded
        mechanically, and the reply parses with the real judge parser."""
        key_points = ["refund window 14 business days", "late fee 1.5% monthly"]
        answer = "The refund window is 14 business days. No fee details."
        stub = StubGatewayClient()
        result = stub.chat(model="stub-judge", prompt=judge_prompt(key_points, answer))
        assert result.model == "stub-judge"
        scores = parse_judge_scores(result.content, len(key_points))
        assert scores == [
            1,  # every significant word of the point is in the answer
            0,  # "fee" / "1.5%" / "monthly" never appear
        ]

    def test_judge_stub_grades_content_not_noise(self):
        stub = StubGatewayClient()
        points = ["escalate after two failed suggestions"]
        strict = stub.chat(model="m", prompt=judge_prompt(points, "I escalated.")).content
        full = stub.chat(
            model="m", prompt=judge_prompt(points, "escalate after two failed suggestions")
        ).content
        assert parse_judge_scores(strict, 1) == [0]
        assert parse_judge_scores(full, 1) == [1]

    def test_judge_stub_matches_mechanical_judge_reference(self):
        stub = StubGatewayClient()
        result = stub.chat(
            model="m", prompt=judge_prompt(["a point"], "an answer")
        )
        assert result.content == mechanical_judge_response(["a point"], "an answer")

    def test_malformed_judge_envelope_propagates_judge_error(self):
        with pytest.raises(JudgeError):
            StubGatewayClient().chat(model="m", prompt="TRILLIC-JUDGE/1 not json\nrest")
