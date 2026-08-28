"""Application error types shared by cloud-provider adapters."""

from __future__ import annotations

import re

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


class ProviderError(RuntimeError):
    """A safe, typed failure at an external model-provider boundary."""

    def __init__(
        self,
        provider: str,
        operation: str,
        status_code: int | None,
        retryable: bool,
        message: str,
    ) -> None:
        self.provider = provider
        self.operation = operation
        self.status_code = status_code
        self.retryable = retryable
        self.safe_message = _safe_message(message)
        super().__init__(self.safe_message)
