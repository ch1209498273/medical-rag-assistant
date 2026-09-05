"""Application error types shared by cloud-provider adapters."""

from __future__ import annotations

import re
from typing import Literal

_SECRET_PATTERNS = (
    (re.compile(r"(?i)(bearer\s+)[^\s,;]+"), r"\1[redacted]"),
    (re.compile(r"(?i)(api[_ -]?key\s*[:=]\s*)[^\s,;]+"), r"\1[redacted]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]+\b"), "[redacted]"),
)


def _safe_message(message: str) -> str:
    """Keep provider errors useful while removing common credential forms."""

    safe = str(message).replace("\r", " ").replace("\n", " ")
    for pattern, replacement in _SECRET_PATTERNS:
        safe = pattern.sub(replacement, safe)
    return safe[:500] or "provider request failed"


ProviderFailureKind = Literal["timeout", "transport", "schema", "truncated", "unknown"]
_FAILURE_KINDS = frozenset({"timeout", "transport", "schema", "truncated", "unknown"})


class ProviderError(RuntimeError):
    """A safe, typed failure at an external model-provider boundary."""

    def __init__(
        self,
        provider: str,
        operation: str,
        status_code: int | None,
        retryable: bool,
        message: str,
        failure_kind: ProviderFailureKind = "unknown",
    ) -> None:
        if failure_kind not in _FAILURE_KINDS:
            raise ValueError("failure_kind is invalid")
        self.provider = provider
        self.operation = operation
        self.status_code = status_code
        self.retryable = retryable
        self.failure_kind = failure_kind
        self.safe_message = _safe_message(message)
        super().__init__(self.safe_message)


def provider_failure_reason(
    error: BaseException, *, fallback: str
) -> str:
    """Map typed provider failures to the finite public reason-code contract."""

    kind = getattr(error, "failure_kind", "unknown")
    if kind == "timeout":
        return "PROVIDER_TIMEOUT"
    if kind == "transport":
        return "PROVIDER_TRANSPORT_ERROR"
    if kind == "schema":
        return "PROVIDER_SCHEMA_INVALID"
    if kind == "truncated":
        return "OUTPUT_TRUNCATED"
    return fallback


__all__ = ["ProviderError", "ProviderFailureKind", "provider_failure_reason"]
