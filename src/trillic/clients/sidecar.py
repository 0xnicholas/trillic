"""Refine-sidecar client.

The evaluation boundary is the sidecar's POST /refine endpoint (issue #1),
whose request/response contract lives in tokencamp-pro/sidecars/refine/
server.py (RefineRequest / RefineResponse). The harness never imports
sidecar code — HTTP only.
"""

from dataclasses import dataclass
from typing import Protocol

import httpx

_REQUIRED_FIELDS = (
    "refined_text",
    "original_tokens",
    "refined_tokens",
    "cached",
    "fallback",
    "warnings",
    "refine_model",
)


class RefineError(Exception):
    """The refine sidecar could not be reached or returned a bad response."""


@dataclass(frozen=True)
class RefineResult:
    refined_text: str
    original_tokens: int
    refined_tokens: int
    cached: bool
    fallback: bool
    warnings: tuple[str, ...]
    refine_model: str


class RefineClient(Protocol):
    def refine(
        self,
        text: str,
        *,
        rewrite: bool,
        compress: bool,
        aggressiveness: float,
    ) -> RefineResult:
        """Refine (rewrite and/or compress) `text` via the sidecar."""
        ...


class HttpRefineClient:
    """Talks to a real refine sidecar over HTTP."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    def refine(
        self,
        text: str,
        *,
        rewrite: bool,
        compress: bool,
        aggressiveness: float,
    ) -> RefineResult:
        payload = {
            "text": text,
            "rewrite": rewrite,
            "compress": compress,
            "aggressiveness": aggressiveness,
        }
        try:
            response = self._client.post("/refine", json=payload)
        except httpx.HTTPError as e:
            raise RefineError(f"refine sidecar unreachable: {e}") from e
        if response.status_code < 200 or response.status_code >= 300:
            raise RefineError(
                f"refine sidecar returned {response.status_code}: {response.text[:500]}"
            )
        return _parse_refine_response(response.json())

    def close(self) -> None:
        self._client.close()


def _parse_refine_response(data: object) -> RefineResult:
    if not isinstance(data, dict):
        raise RefineError(f"malformed refine response: expected object, got {type(data).__name__}")
    missing = [field for field in _REQUIRED_FIELDS if field not in data]
    if missing:
        raise RefineError(f"malformed refine response: missing field(s) {missing}")
    return RefineResult(
        refined_text=data["refined_text"],
        original_tokens=data["original_tokens"],
        refined_tokens=data["refined_tokens"],
        cached=data["cached"],
        fallback=data["fallback"],
        warnings=tuple(data["warnings"]),
        refine_model=data["refine_model"],
    )


class StubRefineClient:
    """Deterministic in-process stand-in for the refine sidecar.

    Policy (documented so tests can pin it by hand): compression splits text
    on whitespace and drops the first K words of every 10-word block, where
    K = round(10 * aggressiveness) (Python banker's rounding). If that would
    remove everything, one word survives. The result is therefore always a
    strict subsequence of the input, matching the extractive guarantee.
    Rewrite alone returns the text unchanged (whitespace-normalized).

    The *_tokens fields report whitespace word counts — the stub's native
    caliber. Real ratio accounting is done harness-side with tiktoken.
    """

    refine_model = "stub-refine"

    def refine(
        self,
        text: str,
        *,
        rewrite: bool,
        compress: bool,
        aggressiveness: float,
    ) -> RefineResult:
        if not rewrite and not compress:
            raise ValueError("rewrite and compress cannot both be false")
        words = text.split()
        if not words:
            return RefineResult("", 0, 0, False, False, (), self.refine_model)
        kept = words
        if compress:
            drop = round(10 * aggressiveness)
            kept = [w for i, w in enumerate(words) if i % 10 >= drop]
            if not kept:
                kept = words[:1]
        refined_text = " ".join(kept)
        return RefineResult(
            refined_text=refined_text,
            original_tokens=len(words),
            refined_tokens=len(kept),
            cached=False,
            fallback=False,
            warnings=(),
            refine_model=self.refine_model,
        )
