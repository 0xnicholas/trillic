"""Gateway LLM client.

All LLM traffic (downstream tasks + judges, from issue #7 on) goes through
the tokencamp gateway so costs land in the internal ledger. Contract follows
the old harness (tokencamp compression/eval/run_eval.py):

    POST {base_url}/v1/chat/completions
    Authorization: Bearer <service key>
    X-TC-Refine: false          (eval/judge calls must never be refined)

The service key is read ONLY from the REFINE_SERVICE_KEY environment
variable (issue #1: credentials never land in config files or the repo).
"""

import os
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

SERVICE_KEY_ENV = "REFINE_SERVICE_KEY"


class GatewayError(Exception):
    """The gateway could not be reached, is unauthenticated, or misbehaved."""


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    usage: dict[str, int] | None


class GatewayClient(Protocol):
    def chat(self, model: str, prompt: str) -> ChatResult:
        """Send a single-user-message chat completion request."""
        ...


class HttpGatewayClient:
    """Talks to the real tokencamp gateway over HTTP."""

    def __init__(
        self,
        base_url: str,
        service_key: str | None = None,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )
        self._service_key = service_key if service_key is not None else os.environ.get(SERVICE_KEY_ENV)

    def chat(self, model: str, prompt: str) -> ChatResult:
        if not self._service_key:
            raise GatewayError(
                f"gateway service key missing: set ${SERVICE_KEY_ENV} (keys are "
                "environment-only and never stored in run configs)"
            )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self._service_key}", "X-TC-Refine": "false"}
        try:
            response = self._client.post(
                "/v1/chat/completions", json=payload, headers=headers
            )
        except httpx.HTTPError as e:
            raise GatewayError(f"gateway unreachable: {e}") from e
        if response.status_code < 200 or response.status_code >= 300:
            raise GatewayError(
                f"gateway returned {response.status_code}: {response.text[:500]}"
            )
        return _parse_completion(response.json(), model)

    def close(self) -> None:
        self._client.close()


def _parse_completion(data: object, model: str) -> ChatResult:
    if not isinstance(data, dict) or "choices" not in data or not data["choices"]:
        raise GatewayError("malformed gateway response: missing field 'choices'")
    choice = data["choices"][0]
    try:
        content = choice["message"]["content"]
    except (KeyError, TypeError) as e:
        raise GatewayError(f"malformed gateway response: missing field 'content': {e}") from e
    usage: Any = data.get("usage")
    return ChatResult(content=content, model=model, usage=usage)


class StubGatewayClient:
    """Deterministic in-process stand-in for the gateway (zero cost, zero network)."""

    def chat(self, model: str, prompt: str) -> ChatResult:
        content = f"[stub:{model}] {prompt}"
        return ChatResult(content=content, model=model, usage=None)
