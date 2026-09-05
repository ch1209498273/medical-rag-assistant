"""Controlled local bad-case lifecycle used by feedback operations.

Bad cases deliberately contain only operational metadata.  The original
question, answer, evidence and provider payload remain outside the export
surface so that a quality loop does not become a second content store.
"""

from __future__ import annotations

from typing import Final

from app.domain.ports import BadCaseStatus, FeedbackReason

FEEDBACK_REASON_CODES: Final[dict[FeedbackReason, str]] = {
    "not_answered": "USER_FEEDBACK_NOT_ANSWERED",
    "missing_step": "USER_FEEDBACK_MISSING_STEP",
    "version_mismatch": "USER_FEEDBACK_VERSION_MISMATCH",
    "citation_mismatch": "USER_FEEDBACK_CITATION_MISMATCH",
    "too_slow": "USER_FEEDBACK_TOO_SLOW",
}

BAD_CASE_STATUS_TRANSITIONS: Final[dict[BadCaseStatus, BadCaseStatus]] = {
    "new": "triaged",
    "triaged": "fixed",
    "fixed": "regression_checked",
    "regression_checked": "closed",
}


def feedback_reason_to_code(reason: FeedbackReason) -> str:
    """Map a user-facing reason to a stable, non-sensitive case code."""

    try:
        return FEEDBACK_REASON_CODES[reason]
    except KeyError as error:
        raise ValueError("feedback reason is not allowed") from error


def apply_bad_case_transition(
    current_status: BadCaseStatus, target_status: BadCaseStatus
) -> BadCaseStatus:
    """Apply exactly one adjacent lifecycle transition."""

    expected = BAD_CASE_STATUS_TRANSITIONS.get(current_status)
    if expected != target_status:
        raise ValueError("invalid transition")
    return target_status


__all__ = [
    "BAD_CASE_STATUS_TRANSITIONS",
    "FEEDBACK_REASON_CODES",
    "apply_bad_case_transition",
    "feedback_reason_to_code",
]
