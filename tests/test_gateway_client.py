"""Gateway LLM client: HTTP impl (tested via httpx.MockTransport, zero
network) and a deterministic stub.

Contract follows the old harness (tokencamp compression/eval/run_eval.py):
POST {base}/v1/chat/completions with a bearer service key that comes ONLY
from the REFINE_SERVICE_KEY environment variable (never from config files),
and the X-TC-Refine: false header so eval/judge calls are never refined.

The gateway is not called by `eval run` yet (task-level quality arrives with
issue #7); this module establishes the injectable seam now.
"""

import json

import httpx
import pytest

from trillic.clients.gateway import (
    GatewayError,
    HttpGatewayClient,
    StubGatewayClient,
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
