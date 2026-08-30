"""Client layer for the evaluation harness.

Both remote dependencies — the refine sidecar (POST /refine) and the LLM
gateway (POST /v1/chat/completions) — sit behind injectable client protocols.
Tests (and offline runs) swap in deterministic in-process stubs; the HTTP
implementations exist for real-link runs.
"""

from trillic.clients.gateway import GatewayClient, HttpGatewayClient, StubGatewayClient
from trillic.clients.sidecar import HttpRefineClient, RefineClient, RefineResult, StubRefineClient

__all__ = [
    "GatewayClient",
    "HttpGatewayClient",
    "HttpRefineClient",
    "RefineClient",
    "RefineResult",
    "StubGatewayClient",
    "StubRefineClient",
]
