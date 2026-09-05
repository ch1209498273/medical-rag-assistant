"""SQLite schema for the derived private feedback projection."""

from __future__ import annotations

import json
import sqlite3

from app.feedback.models import (
    FeedbackCase,
    FeedbackEvent,
    PromotionRecord,
    ReviewDecision,
)


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class FeedbackProjectionSchema:
    """Own the private feedback database without touching production schema."""

    CURRENT_VERSION = 1

    @classmethod
    def create(cls, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY CHECK (version > 0)
            );

            CREATE TABLE IF NOT EXISTS feedback_cases (
                case_id TEXT PRIMARY KEY,
                message_id_hash TEXT NOT NULL,
                session_id_hash TEXT NOT NULL,
                collected_at TEXT NOT NULL,
                question_redacted TEXT NOT NULL,
                answer_redacted TEXT NOT NULL,
                redaction_status TEXT NOT NULL CHECK (
                    redaction_status IN ('passed', 'review_required', 'blocked')
                ),
                redaction_version TEXT NOT NULL,
                redaction_flags_json TEXT NOT NULL DEFAULT '[]',
                answer_status TEXT NOT NULL CHECK (
                    answer_status IN ('answered', 'refused', 'error')
                ),
                reason_code TEXT,
                source_version TEXT,
                retrieval_profile TEXT,
                citation_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
                citation_count INTEGER NOT NULL CHECK (citation_count >= 0),
                model_id TEXT,
                prompt_version TEXT,
                latency_bucket TEXT,
                audience_scope TEXT NOT NULL,
                retention_expires_at TEXT NOT NULL,
                triage_priority TEXT NOT NULL CHECK (triage_priority IN ('P0', 'P1', 'P2', 'P3')),
                triage_score INTEGER NOT NULL CHECK (triage_score >= 0),
                triage_reasons_json TEXT NOT NULL DEFAULT '[]',
                review_status TEXT NOT NULL CHECK (
                    review_status IN ('unreviewed', 'in_review', 'approved', 'rejected', 'adjudication_required')
                ),
                promotion_status TEXT NOT NULL CHECK (
                    promotion_status IN ('not_promoted', 'silver', 'golden_v2_candidate', 'golden_v2')
                ),
                event_count INTEGER NOT NULL DEFAULT 0 CHECK (event_count >= 0),
                UNIQUE(case_id, message_id_hash)
            );

            CREATE TABLE IF NOT EXISTS feedback_events (
                event_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES feedback_cases(case_id),
                helpful INTEGER NOT NULL CHECK (helpful IN (0, 1)),
                feedback_reason TEXT,
                event_at TEXT NOT NULL,
                source TEXT NOT NULL CHECK (source IN ('user_click', 'admin_correction', 'import')),
                event_schema_version INTEGER NOT NULL CHECK (event_schema_version > 0)
            );

            CREATE TABLE IF NOT EXISTS review_decisions (
                review_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES feedback_cases(case_id),
                reviewer_role TEXT NOT NULL CHECK (
                    reviewer_role IN ('product', 'engineering', 'clinical_reviewer')
                ),
                decision TEXT NOT NULL CHECK (
                    decision IN ('approve', 'reject', 'needs_adjudication')
                ),
                evidence_ok INTEGER CHECK (evidence_ok IN (0, 1) OR evidence_ok IS NULL),
                points_ok INTEGER CHECK (points_ok IN (0, 1) OR points_ok IS NULL),
                safety_ok INTEGER CHECK (safety_ok IN (0, 1) OR safety_ok IS NULL),
                note_code TEXT,
                review_version TEXT NOT NULL,
                reviewed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS promotion_records (
                promotion_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES feedback_cases(case_id),
                target_set_id TEXT NOT NULL,
                target_version TEXT NOT NULL,
                target_split TEXT NOT NULL CHECK (target_split IN ('dev', 'holdout')),
                promotion_reason TEXT NOT NULL,
                manifest_id TEXT NOT NULL,
                promoted_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_cases_status
                ON feedback_cases(review_status, promotion_status);
            CREATE INDEX IF NOT EXISTS idx_feedback_cases_triage
                ON feedback_cases(triage_priority, triage_score DESC);
            CREATE INDEX IF NOT EXISTS idx_feedback_cases_retention
                ON feedback_cases(retention_expires_at);
            CREATE INDEX IF NOT EXISTS idx_feedback_events_case
                ON feedback_events(case_id, event_at);
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO schema_version(version) VALUES (?)",
            (cls.CURRENT_VERSION,),
        )
        connection.commit()

    @staticmethod
    def insert_case(connection: sqlite3.Connection, case: FeedbackCase) -> None:
        connection.execute(
            """
            INSERT INTO feedback_cases(
                case_id, message_id_hash, session_id_hash, collected_at,
                question_redacted, answer_redacted, redaction_status,
                redaction_version, redaction_flags_json, answer_status,
                reason_code, source_version, retrieval_profile,
                citation_chunk_ids_json, citation_count, model_id,
                prompt_version, latency_bucket, audience_scope,
                retention_expires_at, triage_priority, triage_score,
                triage_reasons_json, review_status, promotion_status, event_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case.case_id,
                case.message_id_hash,
                case.session_id_hash,
                case.collected_at,
                case.question_redacted,
                case.answer_redacted,
                case.redaction_status,
                case.redaction_version,
                _json_dump(case.redaction_flags),
                case.answer_status,
                case.reason_code,
                case.source_version,
                case.retrieval_profile,
                _json_dump(case.citation_chunk_ids),
                case.citation_count,
                case.model_id,
                case.prompt_version,
                case.latency_bucket,
                case.audience_scope,
                case.retention_expires_at,
                case.triage_priority,
                case.triage_score,
                _json_dump(case.triage_reasons),
                case.review_status,
                case.promotion_status,
                case.event_count,
            ),
        )

    @staticmethod
    def insert_event(connection: sqlite3.Connection, event: FeedbackEvent) -> None:
        connection.execute(
            """
            INSERT INTO feedback_events(
                event_id, case_id, helpful, feedback_reason, event_at,
                source, event_schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.case_id,
                int(event.helpful),
                event.feedback_reason,
                event.event_at,
                event.source,
                event.event_schema_version,
            ),
        )

    @staticmethod
    def insert_review(connection: sqlite3.Connection, review: ReviewDecision) -> None:
        connection.execute(
            """
            INSERT INTO review_decisions(
                review_id, case_id, reviewer_role, decision,
                evidence_ok, points_ok, safety_ok, note_code,
                review_version, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review.review_id,
                review.case_id,
                review.reviewer_role,
                review.decision,
                None if review.evidence_ok is None else int(review.evidence_ok),
                None if review.points_ok is None else int(review.points_ok),
                None if review.safety_ok is None else int(review.safety_ok),
                review.note_code,
                review.review_version,
                review.reviewed_at,
            ),
        )

    @staticmethod
    def insert_promotion(connection: sqlite3.Connection, promotion: PromotionRecord) -> None:
        connection.execute(
            """
            INSERT INTO promotion_records(
                promotion_id, case_id, target_set_id, target_version,
                target_split, promotion_reason, manifest_id, promoted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                promotion.promotion_id,
                promotion.case_id,
                promotion.target_set_id,
                promotion.target_version,
                promotion.target_split,
                promotion.promotion_reason,
                promotion.manifest_id,
                promotion.promoted_at,
            ),
        )
