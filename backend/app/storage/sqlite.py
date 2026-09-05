"""SQLite implementation of the document-version lifecycle."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from uuid import uuid4

from app.chat.models import (
    AssistantMessageStatus,
    ChatMessage,
    ChatSessionSummary,
    Feedback,
)
from app.chat.safety import (
    REFERENCE_REASON_CODES,
    is_safe_answer_text,
    is_safe_id,
    normalize_chat_message,
    normalize_feedback,
    validate_citations,
    validate_feedback_reason,
    validate_reason_code,
)
from app.domain.ports import (
    AudienceScope,
    BadCaseRecord,
    BadCaseStatus,
    DocumentBusinessMetadata,
    DocumentBusinessMetadataInput,
    DocumentSummary,
    DocumentVersion,
    EligibilityResult,
    FeedbackReason,
    IndexAction,
    VersionStatus,
)
from app.evaluation.bad_cases import apply_bad_case_transition, feedback_reason_to_code

LOGGER = logging.getLogger(__name__)
DEFAULT_CHAT_SESSION_TITLE = "新会话"
MAX_CHAT_SESSION_TITLE_CHARS = 40


class SqliteDocumentRepository:
    """Persist document metadata while publishing new versions atomically."""

    CURRENT_SCHEMA_VERSION = 8

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI may run request and lifespan teardown callbacks on different
        # worker threads.  SQLite transactions remain serialized by the
        # repository's BEGIN IMMEDIATE boundaries.
        connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection = connection
        try:
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._initialize_schema()
        except BaseException:
            try:
                connection.close()
            except BaseException:  # noqa: BLE001 - preserve schema error
                LOGGER.warning(
                    "sqlite constructor cleanup failed",
                    extra={"stage": "sqlite_constructor_cleanup"},
                )
            raise

    def _initialize_schema(self) -> None:
        if not self._table_exists("documents"):
            self._begin_schema_transaction()
            try:
                self._create_schema_objects()
                self._ensure_chat_schema()
                self._write_schema_version()
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            finally:
                self._restore_foreign_keys()
            return

        version = self._read_schema_version()
        if version < self.CURRENT_SCHEMA_VERSION:
            if version < 4:
                self._migrate_to_current_schema()
            else:
                self._migrate_incrementally()
        else:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._create_schema_objects()
                self._ensure_chat_schema()
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def _create_schema_objects(self) -> None:
        """Create current tables/indexes without implicit transaction commits."""

        statements = (
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS document_versions (
                version_id TEXT PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(document_id),
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN (
                    'indexing', 'active', 'failed', 'superseded', 'inactive', 'needs_ocr'
                )),
                failure_reason TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS index_jobs (
                job_id TEXT PRIMARY KEY,
                version_id TEXT NOT NULL REFERENCES document_versions(version_id),
                status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS document_business_metadata (
                version_id TEXT PRIMARY KEY REFERENCES document_versions(version_id),
                content_type TEXT NOT NULL CHECK (content_type IN (
                    'policy', 'training', 'procedure', 'other'
                )),
                applicable_scope TEXT NOT NULL CHECK (applicable_scope IN (
                    'unspecified', 'all_staff', 'nurse', 'doctor', 'pharmacist', 'administrator'
                )),
                effective_from TEXT,
                review_due_at TEXT,
                business_status TEXT NOT NULL CHECK (business_status IN (
                    'draft', 'approved', 'superseded', 'retired', 'unknown'
                )),
                owner_role TEXT,
                supersedes_version_id TEXT REFERENCES document_versions(version_id),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '新会话',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(session_id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'submitted',
                reply_to_message_id TEXT REFERENCES messages(message_id),
                rewritten_question TEXT,
                citations_json TEXT NOT NULL DEFAULT '[]',
                reason_code TEXT,
                reference_answer TEXT,
                audience_scope TEXT NOT NULL DEFAULT 'unspecified',
                workflow_summary_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                session_id TEXT REFERENCES chat_sessions(session_id),
                message_id TEXT REFERENCES messages(message_id),
                helpful INTEGER NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bad_cases (
                case_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL REFERENCES messages(message_id),
                code TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN (
                    'new', 'triaged', 'fixed', 'regression_checked', 'closed'
                )),
                document_version_id TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY CHECK (version > 0)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_document_versions_active
                ON document_versions(document_id, status)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_document_business_metadata_status
                ON document_business_metadata(business_status, applicable_scope)
            """,
        )
        for statement in statements:
            self._connection.execute(statement)

    def _ensure_chat_schema(self) -> None:
        """Add Task 8 chat columns and make feedback updates idempotent.

        This method is intentionally usable inside the schema transaction.  It
        handles databases created by Task 3/7 where the chat tables existed
        but only contained the original minimal columns.
        """

        if not self._table_exists("chat_sessions"):
            self._connection.execute(
                """
                CREATE TABLE chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '新会话',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        if not self._table_exists("messages"):
            self._connection.execute(
                """
                CREATE TABLE messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'submitted',
                    reply_to_message_id TEXT REFERENCES messages(message_id),
                    rewritten_question TEXT,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    reason_code TEXT,
                    reference_answer TEXT,
                    audience_scope TEXT NOT NULL DEFAULT 'unspecified',
                    workflow_summary_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        else:
            session_columns = {
                row["name"]
                for row in self._connection.execute("PRAGMA table_info(chat_sessions)")
            }
            if "title" not in session_columns:
                self._connection.execute(
                    "ALTER TABLE chat_sessions ADD COLUMN title TEXT NOT NULL DEFAULT '新会话'"
                )
            columns = {
                row["name"]
                for row in self._connection.execute("PRAGMA table_info(messages)")
            }
            additions = (
                ("status", "TEXT NOT NULL DEFAULT 'submitted'"),
                ("reply_to_message_id", "TEXT"),
                ("rewritten_question", "TEXT"),
                ("citations_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("reason_code", "TEXT"),
                ("reference_answer", "TEXT"),
                ("audience_scope", "TEXT NOT NULL DEFAULT 'unspecified'"),
                ("workflow_summary_json", "TEXT"),
            )
            for name, definition in additions:
                if name not in columns:
                    self._connection.execute(
                        f"ALTER TABLE messages ADD COLUMN {name} {definition}"
                    )

        self._backfill_chat_session_titles()
        if not self._table_exists("feedback"):
            self._connection.execute(
                """
                CREATE TABLE feedback (
                    feedback_id TEXT PRIMARY KEY,
                    session_id TEXT REFERENCES chat_sessions(session_id),
                    message_id TEXT REFERENCES messages(message_id),
                    helpful INTEGER NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        else:
            feedback_columns = {
                row["name"]
                for row in self._connection.execute("PRAGMA table_info(feedback)")
            }
            if "reason" not in feedback_columns:
                self._connection.execute("ALTER TABLE feedback ADD COLUMN reason TEXT")

        if not self._table_exists("bad_cases"):
            self._connection.execute(
                """
                CREATE TABLE bad_cases (
                    case_id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL REFERENCES messages(message_id),
                    code TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN (
                        'new', 'triaged', 'fixed', 'regression_checked', 'closed'
                    )),
                    document_version_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

        # Keep the newest existing row before adding the unique index.  The
        # rowid tie-breaker makes equal CURRENT_TIMESTAMP values deterministic.
        duplicate_ids = self._connection.execute(
            """
            SELECT message_id
            FROM feedback
            WHERE message_id IS NOT NULL
            GROUP BY message_id
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        for duplicate in duplicate_ids:
            rows = self._connection.execute(
                """
                SELECT feedback_id
                FROM feedback
                WHERE message_id = ?
                ORDER BY created_at DESC, rowid DESC
                """,
                (duplicate["message_id"],),
            ).fetchall()
            for row in rows[1:]:
                self._connection.execute(
                    "DELETE FROM feedback WHERE feedback_id = ?",
                    (row["feedback_id"],),
                )
        self._deduplicate_terminal_assistant_rows()
        self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_message_id "
            "ON feedback(message_id)"
        )
        self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_bad_cases_message_code "
            "ON bad_cases(message_id, code)"
        )
        self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_terminal_reply "
            "ON messages(reply_to_message_id) "
            "WHERE role = 'assistant' "
            "AND status IN ('answered', 'refused', 'error') "
            "AND reply_to_message_id IS NOT NULL"
        )

    def _deduplicate_terminal_assistant_rows(self) -> None:
        """Keep the newest terminal reply before creating its unique index."""

        duplicate_ids = self._connection.execute(
            """
            SELECT reply_to_message_id
            FROM messages
            WHERE role = 'assistant'
              AND status IN ('answered', 'refused', 'error')
              AND reply_to_message_id IS NOT NULL
            GROUP BY reply_to_message_id
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        for duplicate in duplicate_ids:
            reply_to = duplicate["reply_to_message_id"]
            rows = self._connection.execute(
                """
                SELECT message_id
                FROM messages
                WHERE role = 'assistant'
                  AND status IN ('answered', 'refused', 'error')
                  AND reply_to_message_id = ?
                ORDER BY created_at DESC, rowid DESC
                """,
                (reply_to,),
            ).fetchall()
            keeper = rows[0]["message_id"]
            for row in rows[1:]:
                duplicate_message_id = row["message_id"]
                self._merge_feedback_for_message(keeper, duplicate_message_id)
                self._connection.execute(
                    "UPDATE messages SET reply_to_message_id = ? "
                    "WHERE reply_to_message_id = ?",
                    (keeper, duplicate_message_id),
                )
                self._connection.execute(
                    "DELETE FROM messages WHERE message_id = ?",
                    (duplicate_message_id,),
                )

    def _merge_feedback_for_message(self, keeper: str, duplicate: str) -> None:
        """Move one duplicate's feedback to the keeper without losing newest state."""

        rows = self._connection.execute(
            """
            SELECT feedback_id, message_id, created_at
            FROM feedback
            WHERE message_id IN (?, ?)
            ORDER BY created_at DESC, rowid DESC
            """,
            (keeper, duplicate),
        ).fetchall()
        if not rows:
            return
        newest = rows[0]
        for row in rows[1:]:
            self._connection.execute(
                "DELETE FROM feedback WHERE feedback_id = ?",
                (row["feedback_id"],),
            )
        if newest["message_id"] != keeper:
            self._connection.execute(
                "UPDATE feedback SET message_id = ? WHERE feedback_id = ?",
                (keeper, newest["feedback_id"]),
            )

    def _migrate_to_current_schema(self) -> None:
        """Rebuild Task 3 tables atomically for the inactive/NOCASE contract."""

        self._begin_schema_transaction()
        try:
            self._connection.execute(
                """
                CREATE TABLE documents_new (
                    document_id INTEGER PRIMARY KEY,
                    file_name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE document_versions_new (
                    version_id TEXT PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(document_id),
                    sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN (
                        'indexing', 'active', 'failed', 'superseded', 'inactive', 'needs_ocr'
                    )),
                    failure_reason TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._connection.execute(
                """
                INSERT INTO documents_new(document_id, file_name, created_at)
                SELECT document_id, file_name, created_at FROM documents
                """
            )
            self._connection.execute(
                """
                INSERT INTO document_versions_new(
                    version_id, document_id, sha256, status, failure_reason, created_at
                )
                SELECT version_id, document_id, sha256, status, failure_reason, created_at
                FROM document_versions
                """
            )
            self._connection.execute("DROP INDEX IF EXISTS idx_document_versions_active")
            self._connection.execute("DROP TABLE document_versions")
            self._connection.execute("DROP TABLE documents")
            self._connection.execute("ALTER TABLE documents_new RENAME TO documents")
            self._connection.execute(
                "ALTER TABLE document_versions_new RENAME TO document_versions"
            )
            self._create_schema_objects()
            self._ensure_chat_schema()
            self._write_schema_version()
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        finally:
            self._restore_foreign_keys()

    def _migrate_incrementally(self) -> None:
        """Apply incremental schema changes while preserving existing rows."""

        self._begin_schema_transaction()
        try:
            self._migrate_audience_scope_constraint()
            self._create_schema_objects()
            self._ensure_chat_schema()
            self._write_schema_version()
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        finally:
            self._restore_foreign_keys()

    def _migrate_audience_scope_constraint(self) -> None:
        """Rebuild the metadata table so existing DBs accept pharmacist scope."""

        if not self._table_exists("document_business_metadata"):
            return
        self._connection.execute("ALTER TABLE document_business_metadata RENAME TO document_business_metadata_old")
        self._connection.execute(
            """
            CREATE TABLE document_business_metadata (
                version_id TEXT PRIMARY KEY REFERENCES document_versions(version_id),
                content_type TEXT NOT NULL CHECK (content_type IN (
                    'policy', 'training', 'procedure', 'other'
                )),
                applicable_scope TEXT NOT NULL CHECK (applicable_scope IN (
                    'unspecified', 'all_staff', 'nurse', 'doctor', 'pharmacist', 'administrator'
                )),
                effective_from TEXT,
                review_due_at TEXT,
                business_status TEXT NOT NULL CHECK (business_status IN (
                    'draft', 'approved', 'superseded', 'retired', 'unknown'
                )),
                owner_role TEXT,
                supersedes_version_id TEXT REFERENCES document_versions(version_id),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.execute(
            """
            INSERT INTO document_business_metadata(
                version_id, content_type, applicable_scope, effective_from,
                review_due_at, business_status, owner_role,
                supersedes_version_id, updated_at
            )
            SELECT version_id, content_type, applicable_scope, effective_from,
                   review_due_at, business_status, owner_role,
                   supersedes_version_id, updated_at
            FROM document_business_metadata_old
            """
        )
        self._connection.execute("DROP TABLE document_business_metadata_old")

    def _begin_schema_transaction(self) -> None:
        # SQLite cannot toggle foreign_keys inside a transaction.  The rebuild
        # is still transactional; disabling checks only permits replacing the
        # referenced tables while the unchanged index_jobs rows are retained.
        self._connection.commit()
        self._connection.execute("PRAGMA foreign_keys = OFF")
        self._connection.execute("BEGIN IMMEDIATE")

    def _restore_foreign_keys(self) -> None:
        self._connection.execute("PRAGMA foreign_keys = ON")

    def _table_exists(self, table_name: str) -> bool:
        return (
            self._connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table_name,),
            ).fetchone()
            is not None
        )

    def _read_schema_version(self) -> int:
        if not self._table_exists("schema_version"):
            return 1
        row = self._connection.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return 1 if row is None else int(row["version"])

    def _write_schema_version(self) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO schema_version(version) VALUES (?)",
            (self.CURRENT_SCHEMA_VERSION,),
        )

    def begin_index(
        self,
        file_name: str,
        sha256: str,
        *,
        needs_ocr: bool = False,
        force: bool = False,
    ) -> DocumentVersion:
        """Create an unpublished version unless identical bytes are already live.

        ``force`` is used by an explicit user reindex request.  It deliberately
        creates a new version even when the current active bytes are unchanged.
        """
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            existing = None
            if not force:
                existing = self._connection.execute(
                    """
                    SELECT version_id, sha256, status, failure_reason
                    FROM document_versions
                    JOIN documents USING (document_id)
                    WHERE file_name = ? COLLATE NOCASE AND sha256 = ?
                        AND status IN ('active', 'indexing', 'needs_ocr')
                    """,
                    (file_name, sha256),
                ).fetchone()
            if existing is not None:
                action: IndexAction = {
                    "active": "skip",
                    "indexing": "in_progress",
                    "needs_ocr": "needs_ocr",
                }[existing["status"]]
                self._connection.execute("COMMIT")
                return self._version_from_row(existing, file_name, action=action)

            self._connection.execute(
                "INSERT INTO documents (file_name) VALUES (?) ON CONFLICT(file_name) DO NOTHING",
                (file_name,),
            )
            document_id = self._connection.execute(
                "SELECT document_id FROM documents WHERE file_name = ? COLLATE NOCASE",
                (file_name,),
            ).fetchone()["document_id"]
            status: VersionStatus = "needs_ocr" if needs_ocr else "indexing"
            action: IndexAction = "needs_ocr" if needs_ocr else "index"
            version_id = str(uuid4())
            self._connection.execute(
                """
                INSERT INTO document_versions (version_id, document_id, sha256, status)
                VALUES (?, ?, ?, ?)
                """,
                (version_id, document_id, sha256, status),
            )
            self._connection.execute(
                "INSERT INTO index_jobs (job_id, version_id, status) VALUES (?, ?, ?)",
                (str(uuid4()), version_id, status),
            )
            self._connection.execute("COMMIT")
            return DocumentVersion(
                version_id=version_id,
                file_name=file_name,
                sha256=sha256,
                status=status,
                action=action,
            )
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def mark_needs_ocr(self, version_id: str) -> DocumentVersion:
        """Move a newly-created indexing version to the OCR waiting state."""

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            version = self._connection.execute(
                """
                SELECT version_id, file_name, sha256, status, failure_reason
                FROM document_versions JOIN documents USING (document_id)
                WHERE version_id = ?
                """,
                (version_id,),
            ).fetchone()
            if version is None:
                raise KeyError(f"Unknown document version: {version_id}")
            if version["status"] != "indexing":
                raise ValueError(
                    "Only indexing versions can be marked needs_ocr; "
                    f"status is {version['status']}"
                )
            self._connection.execute(
                "UPDATE document_versions SET status = 'needs_ocr' WHERE version_id = ?",
                (version_id,),
            )
            self._connection.execute(
                "UPDATE index_jobs SET status = 'needs_ocr' WHERE version_id = ?",
                (version_id,),
            )
            self._connection.execute("COMMIT")
            return DocumentVersion(
                version_id=version_id,
                file_name=version["file_name"],
                sha256=version["sha256"],
                status="needs_ocr",
                action="needs_ocr",
                failure_reason=version["failure_reason"],
            )
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def begin_ocr_indexing(self, version_id: str) -> DocumentVersion:
        """Atomically resume a scanned document after OCR produced usable text."""
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            version = self._connection.execute(
                """
                SELECT version_id, file_name, sha256, status, failure_reason
                FROM document_versions JOIN documents USING (document_id)
                WHERE version_id = ?
                """,
                (version_id,),
            ).fetchone()
            if version is None:
                raise KeyError(f"Unknown document version: {version_id}")
            if version["status"] != "needs_ocr":
                raise ValueError(
                    "Only needs_ocr versions can begin OCR indexing; "
                    f"status is {version['status']}"
                )
            self._connection.execute(
                """
                UPDATE document_versions
                SET status = 'indexing', failure_reason = NULL
                WHERE version_id = ?
                """,
                (version_id,),
            )
            self._connection.execute(
                """
                UPDATE index_jobs SET status = 'indexing', error_message = NULL
                WHERE version_id = ?
                """,
                (version_id,),
            )
            self._connection.execute("COMMIT")
            return DocumentVersion(
                version_id=version_id,
                file_name=version["file_name"],
                sha256=version["sha256"],
                status="indexing",
                action="index",
            )
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def activate(self, version_id: str) -> None:
        """Publish one fully-indexed version and supersede its predecessor together."""
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            target = self._connection.execute(
                "SELECT document_id, status FROM document_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            if target is None:
                raise KeyError(f"Unknown document version: {version_id}")
            if target["status"] != "indexing":
                raise ValueError(
                    f"Only indexing versions can be activated; status is {target['status']}"
                )
            self._connection.execute(
                """
                UPDATE document_versions SET status = 'superseded'
                WHERE document_id = ? AND status = 'active'
                """,
                (target["document_id"],),
            )
            self._connection.execute(
                """
                UPDATE index_jobs SET status = 'superseded'
                WHERE version_id IN (
                    SELECT version_id FROM document_versions
                    WHERE document_id = ? AND status = 'superseded'
                )
                """,
                (target["document_id"],),
            )
            self._connection.execute(
                "UPDATE document_versions SET status = 'active' WHERE version_id = ?",
                (version_id,),
            )
            self._connection.execute(
                "UPDATE index_jobs SET status = 'active', error_message = NULL WHERE version_id = ?",
                (version_id,),
            )
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def fail(self, version_id: str, reason: str) -> None:
        """Record an unpublished indexing error without altering a live version."""
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE document_versions
                SET status = 'failed', failure_reason = ?
                WHERE version_id = ? AND status IN ('indexing', 'needs_ocr')
                """,
                (reason, version_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Only unpublished versions can fail: {version_id}")
            self._connection.execute(
                "UPDATE index_jobs SET status = 'failed', error_message = ? WHERE version_id = ?",
                (reason, version_id),
            )

    def get_active(self, file_name: str) -> DocumentVersion | None:
        row = self._connection.execute(
            """
            SELECT version_id, sha256, status, failure_reason
            FROM document_versions
            JOIN documents USING (document_id)
            WHERE file_name = ? COLLATE NOCASE AND status = 'active'
            """,
            (file_name,),
        ).fetchone()
        return None if row is None else self._version_from_row(row, file_name, action="skip")

    def get_version(self, version_id: str) -> DocumentVersion:
        row = self._connection.execute(
            """
            SELECT version_id, file_name, sha256, status, failure_reason
            FROM document_versions JOIN documents USING (document_id)
            WHERE version_id = ?
            """,
            (version_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown document version: {version_id}")
        return self._version_from_row(row, row["file_name"], action="index")

    def active_version_ids(self) -> set[str]:
        rows = self._connection.execute(
            "SELECT version_id FROM document_versions WHERE status = 'active'"
        )
        return {row["version_id"] for row in rows}

    def deactivate_missing(self, present_file_names: set[str]) -> list[str]:
        """Logically remove absent files from retrieval after a full snapshot.

        The caller is responsible for proving that enumeration completed.  This
        method only changes the current ``active`` version; historical metadata
        and Qdrant points remain available for audit/rebuild purposes.
        """

        names = {name for name in present_file_names}
        with self._connection:
            rows = self._connection.execute(
                """
                SELECT d.file_name, v.version_id
                FROM documents AS d
                JOIN document_versions AS v ON v.document_id = d.document_id
                WHERE v.status = 'active'
                """
            ).fetchall()
            present_casefolded = {name.casefold() for name in names}
            missing = [
                row["file_name"]
                for row in rows
                if row["file_name"].casefold() not in present_casefolded
            ]
            for row in rows:
                if row["file_name"].casefold() in present_casefolded:
                    continue
                self._connection.execute(
                    """
                    UPDATE document_versions
                    SET status = 'inactive'
                    WHERE version_id = ? AND status = 'active'
                    """,
                    (row["version_id"],),
                )
                self._connection.execute(
                    """
                    UPDATE index_jobs SET status = 'inactive'
                    WHERE version_id = ?
                    """,
                    (row["version_id"],),
                )
        return sorted(missing, key=str.casefold)

    def list_documents(self) -> list[DocumentSummary]:
        """Return safe metadata, preferring the searchable active version."""

        rows = self._connection.execute(
            """
            SELECT
                d.document_id,
                d.file_name,
                COALESCE(active.version_id, latest.version_id) AS version_id,
                COALESCE(active.status, latest.status) AS status,
                COALESCE(active.sha256, latest.sha256) AS sha256,
                COALESCE(active.failure_reason, latest.failure_reason)
                    AS failure_reason,
                COALESCE(active.created_at, latest.created_at) AS updated_at,
                metadata.version_id AS business_version_id,
                metadata.content_type AS business_content_type,
                metadata.applicable_scope AS business_applicable_scope,
                metadata.effective_from AS business_effective_from,
                metadata.review_due_at AS business_review_due_at,
                metadata.business_status AS business_status,
                metadata.owner_role AS business_owner_role,
                metadata.supersedes_version_id AS business_supersedes_version_id,
                metadata.updated_at AS business_updated_at
            FROM documents AS d
            LEFT JOIN document_versions AS active
                ON active.document_id = d.document_id AND active.status = 'active'
            LEFT JOIN document_versions AS latest
                ON latest.version_id = (
                    SELECT candidate.version_id
                    FROM document_versions AS candidate
                    WHERE candidate.document_id = d.document_id
                    ORDER BY candidate.created_at DESC, candidate.rowid DESC
                    LIMIT 1
                )
            LEFT JOIN document_business_metadata AS metadata
                ON metadata.version_id = COALESCE(active.version_id, latest.version_id)
            ORDER BY d.file_name COLLATE NOCASE
            """
        ).fetchall()
        return [self._summary_from_row(row) for row in rows]

    def get_document(self, document_id: int) -> DocumentSummary | None:
        """Return one safe metadata record for an opaque API identifier."""

        return next(
            (document for document in self.list_documents() if document.document_id == document_id),
            None,
        )

    def get_document_business_metadata(
        self, document_id: int
    ) -> DocumentBusinessMetadata | None:
        """Return metadata for the version currently shown to the local manager."""

        document = self.get_document(document_id)
        return None if document is None else document.business_metadata

    def upsert_document_business_metadata(
        self,
        document_id: int,
        metadata: DocumentBusinessMetadataInput,
    ) -> DocumentBusinessMetadata:
        """Attach controlled business status to this document's display version."""

        document = self.get_document(document_id)
        if document is None:
            raise KeyError(document_id)
        if document.version_id is None:
            raise ValueError("document has no version")
        values = _validate_business_metadata_input(metadata)
        supersedes_version_id = values["supersedes_version_id"]
        if supersedes_version_id is not None:
            superseded = self._connection.execute(
                """
                SELECT document_id FROM document_versions WHERE version_id = ?
                """,
                (supersedes_version_id,),
            ).fetchone()
            if superseded is None or int(superseded["document_id"]) != document.document_id:
                raise ValueError("supersedes version must belong to the same document")
            if supersedes_version_id == document.version_id:
                raise ValueError("supersedes version must differ from current version")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                """
                INSERT INTO document_business_metadata (
                    version_id, content_type, applicable_scope, effective_from,
                    review_due_at, business_status, owner_role, supersedes_version_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(version_id) DO UPDATE SET
                    content_type = excluded.content_type,
                    applicable_scope = excluded.applicable_scope,
                    effective_from = excluded.effective_from,
                    review_due_at = excluded.review_due_at,
                    business_status = excluded.business_status,
                    owner_role = excluded.owner_role,
                    supersedes_version_id = excluded.supersedes_version_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (document.version_id, *values.values()),
            )
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        result = self.get_document_business_metadata(document_id)
        if result is None:  # pragma: no cover - protects a broken SQLite write.
            raise RuntimeError("business metadata write was not persisted")
        return result

    def eligible_version_ids(
        self, audience_scope: AudienceScope, as_of: date
    ) -> EligibilityResult:
        """Return approved, in-date versions whose scope matches the request."""

        _validate_audience_scope(audience_scope)
        if not isinstance(as_of, date):
            raise TypeError("as_of must be a date")
        rows = self._connection.execute(
            """
            SELECT dv.version_id, bm.business_status, bm.applicable_scope,
                   bm.effective_from, bm.review_due_at
            FROM document_versions AS dv
            LEFT JOIN document_business_metadata AS bm
              ON bm.version_id = dv.version_id
            WHERE dv.status = 'active'
            """
        ).fetchall()
        if not rows:
            return EligibilityResult(frozenset(), "DOCUMENT_BUSINESS_STATUS_UNKNOWN")
        eligible: set[str] = set()
        has_unknown_status = False
        requested_scope = "all_staff" if audience_scope == "unspecified" else audience_scope
        for row in rows:
            if row["business_status"] != "approved":
                has_unknown_status = True
                continue
            scope = row["applicable_scope"]
            if scope not in {"all_staff", requested_scope}:
                continue
            effective_from = row["effective_from"]
            review_due_at = row["review_due_at"]
            if effective_from is not None and effective_from > as_of.isoformat():
                continue
            if review_due_at is not None and review_due_at < as_of.isoformat():
                continue
            eligible.add(str(row["version_id"]))
        if eligible:
            return EligibilityResult(frozenset(eligible), None)
        reason = (
            "DOCUMENT_BUSINESS_STATUS_UNKNOWN"
            if has_unknown_status
            else "EVIDENCE_SCOPE_UNCLEAR"
        )
        return EligibilityResult(frozenset(), reason)

    def create_chat_session(self) -> str:
        """Create an opaque session identifier in a short transaction."""

        session_id = str(uuid4())
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                "INSERT INTO chat_sessions(session_id) VALUES (?)", (session_id,)
            )
            self._connection.execute("COMMIT")
            return session_id
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def chat_session_exists(self, session_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None

    def list_chat_sessions(self, limit: int = 20) -> list[ChatSessionSummary]:
        """Return recent session metadata without message bodies."""

        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
            raise ValueError("session limit is invalid")
        rows = self._connection.execute(
            """
            SELECT s.session_id,
                   s.title,
                   s.created_at,
                   COALESCE(MAX(m.created_at), s.created_at) AS last_activity_at,
                   COUNT(m.message_id) AS message_count
            FROM chat_sessions AS s
            LEFT JOIN messages AS m ON m.session_id = s.session_id
            GROUP BY s.session_id, s.title, s.created_at, s.rowid
            ORDER BY last_activity_at DESC, s.created_at DESC, s.rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [ChatSessionSummary(**dict(row)) for row in rows]

    def _backfill_chat_session_titles(self) -> None:
        """Derive readable names for sessions created before title support."""

        message_columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(messages)")
        }
        order_by = "created_at ASC, rowid ASC" if "created_at" in message_columns else "rowid ASC"
        sessions = self._connection.execute(
            """
            SELECT session_id
            FROM chat_sessions
            WHERE title IS NULL OR TRIM(title) = '' OR title = ?
            """,
            (DEFAULT_CHAT_SESSION_TITLE,),
        ).fetchall()
        for session in sessions:
            first_user = self._connection.execute(
                """
                SELECT content
                FROM messages
                WHERE session_id = ? AND role = 'user'
                ORDER BY """ + order_by + """
                LIMIT 1
                """,
                (session["session_id"],),
            ).fetchone()
            if first_user is None:
                continue
            title = _derive_chat_session_title(first_user["content"])
            self._connection.execute(
                "UPDATE chat_sessions SET title = ? WHERE session_id = ?",
                (title, session["session_id"]),
            )

    def append_user_message(
        self,
        session_id: str,
        content: str,
        audience_scope: AudienceScope = "unspecified",
    ) -> ChatMessage:
        """Persist a submitted user question before any provider call."""

        _validate_message_text(content)
        _validate_audience_scope(audience_scope)
        message_id = str(uuid4())
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            if not self.chat_session_exists(session_id):
                raise KeyError(f"Unknown chat session: {session_id}")
            session = self._connection.execute(
                "SELECT title FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            has_user_message = self._connection.execute(
                "SELECT 1 FROM messages WHERE session_id = ? AND role = 'user' LIMIT 1",
                (session_id,),
            ).fetchone() is not None
            self._connection.execute(
                """
                INSERT INTO messages(
                    message_id, session_id, role, content, status, citations_json,
                    audience_scope
                ) VALUES (?, ?, 'user', ?, 'submitted', '[]', ?)
                """,
                (message_id, session_id, content, audience_scope),
            )
            if (
                not has_user_message
                and session is not None
                and (session["title"] or DEFAULT_CHAT_SESSION_TITLE) == DEFAULT_CHAT_SESSION_TITLE
            ):
                self._connection.execute(
                    "UPDATE chat_sessions SET title = ? WHERE session_id = ?",
                    (_derive_chat_session_title(content), session_id),
                )
            row = self._connection.execute(
                "SELECT * FROM messages WHERE message_id = ?", (message_id,)
            ).fetchone()
            self._connection.execute("COMMIT")
            return self._message_from_row(row)
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def set_rewritten_question(self, message_id: str, rewritten_question: str) -> None:
        """Attach a validated standalone question to a user message."""

        _validate_message_text(rewritten_question)
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                "SELECT role FROM messages WHERE message_id = ?", (message_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown chat message: {message_id}")
            if row["role"] != "user":
                raise ValueError("rewritten_question belongs to a user message")
            self._connection.execute(
                "UPDATE messages SET rewritten_question = ? WHERE message_id = ?",
                (rewritten_question, message_id),
            )
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def append_assistant_message(
        self,
        session_id: str,
        reply_to_message_id: str,
        content: str,
        status: AssistantMessageStatus,
        citations: Sequence[Mapping[str, object]],
        reason_code: str | None,
        reference_answer: str | None = None,
        workflow_summary: Mapping[str, object] | None = None,
    ) -> ChatMessage:
        """Persist one terminal assistant result with a safe citation snapshot."""

        if status not in {"answered", "refused", "error"}:
            raise ValueError("assistant message status is invalid")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            user = self._connection.execute(
                """
                SELECT role, session_id
                FROM messages
                WHERE message_id = ?
                """,
                (reply_to_message_id,),
            ).fetchone()
            if not self.chat_session_exists(session_id):
                raise KeyError(f"Unknown chat session: {session_id}")
            if user is None:
                raise KeyError(f"Unknown user message: {reply_to_message_id}")
            if user["role"] != "user" or user["session_id"] != session_id:
                raise ValueError("assistant reply must belong to a user message")

            existing = self._connection.execute(
                """
                SELECT *
                FROM messages
                WHERE role = 'assistant'
                  AND status IN ('answered', 'refused', 'error')
                  AND reply_to_message_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (reply_to_message_id,),
            ).fetchone()
            if existing is not None:
                if existing["session_id"] != session_id:
                    raise ValueError("assistant reply belongs to another session")
                self._connection.execute("COMMIT")
                return self._message_from_row(existing)

            _validate_message_text(content, allow_empty=True)
            if reference_answer is not None:
                _validate_message_text(reference_answer)
                allowed = (
                    status == "answered" and reason_code is None
                ) or (
                    status == "refused"
                    and reason_code in REFERENCE_REASON_CODES | {"MODEL_REFUSED"}
                )
                if not allowed:
                    raise ValueError("reference answer metadata is invalid")
            try:
                validate_reason_code(reason_code, allow_none=True)
                citation_rows = validate_citations(citations)
                citations_json = json.dumps(
                    citation_rows,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as error:
                raise ValueError("assistant metadata is not safe") from error

            summary_json: str | None = None
            if workflow_summary is not None:
                try:
                    summary = _validate_workflow_summary(workflow_summary)
                    summary_json = json.dumps(
                        summary,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError("workflow summary is not safe") from error

            message_id = str(uuid4())
            self._connection.execute(
                """
                INSERT INTO messages(
                    message_id, session_id, role, content, status,
                    reply_to_message_id, citations_json, reason_code, reference_answer,
                    workflow_summary_json
                ) VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    content,
                    status,
                    reply_to_message_id,
                    citations_json,
                    reason_code,
                    reference_answer,
                    summary_json,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM messages WHERE message_id = ?", (message_id,)
            ).fetchone()
            self._connection.execute("COMMIT")
            return self._message_from_row(row)
        except sqlite3.IntegrityError:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            existing = self._connection.execute(
                """
                SELECT *
                FROM messages
                WHERE role = 'assistant'
                  AND status IN ('answered', 'refused', 'error')
                  AND reply_to_message_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (reply_to_message_id,),
            ).fetchone()
            if existing is not None:
                return self._message_from_row(existing)
            raise
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def list_chat_messages(self, session_id: str) -> list[ChatMessage]:
        """Return only safe chat fields in stable chronological order."""

        rows = self._connection.execute(
            """
            SELECT *
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC, rowid ASC
            """,
            (session_id,),
        ).fetchall()
        return [self._message_from_row(row, fallback_session_id=session_id) for row in rows]

    def upsert_feedback(
        self,
        message_id: str,
        helpful: bool,
        reason: FeedbackReason | None = None,
    ) -> Feedback:
        """Set feedback and optionally open one aggregate-only bad case."""

        if not isinstance(helpful, bool):
            raise TypeError("helpful must be a boolean")
        reason = validate_feedback_reason(reason, allow_none=True)
        if helpful and reason is not None:
            raise ValueError("helpful feedback cannot include a reason")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            message = self._connection.execute(
                """
                SELECT session_id, role, status
                FROM messages
                WHERE message_id = ?
                """,
                (message_id,),
            ).fetchone()
            if message is None:
                raise KeyError(f"Unknown chat message: {message_id}")
            if message["role"] != "assistant" or message["status"] not in {
                "answered",
                "refused",
            }:
                raise ValueError("feedback requires a final assistant message")
            self._connection.execute(
                """
                INSERT INTO feedback(feedback_id, session_id, message_id, helpful, reason)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    helpful = excluded.helpful,
                    reason = excluded.reason,
                    created_at = CURRENT_TIMESTAMP
                """,
                (str(uuid4()), message["session_id"], message_id, int(helpful), reason),
            )
            if not helpful and reason is not None:
                self._insert_bad_case(message_id, feedback_reason_to_code(reason))
            row = self._connection.execute(
                """
                SELECT message_id, helpful, reason, created_at
                FROM feedback
                WHERE message_id = ?
                """,
                (message_id,),
            ).fetchone()
            self._connection.execute("COMMIT")
            return Feedback(
                message_id=str(row["message_id"]),
                helpful=bool(row["helpful"]),
                created_at=str(row["created_at"]),
                reason=validate_feedback_reason(row["reason"], allow_none=True),
            )
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def get_feedback(self, message_id: str) -> Feedback | None:
        row = self._connection.execute(
            """
            SELECT message_id, helpful, reason, created_at
            FROM feedback
            WHERE message_id = ?
            """,
            (message_id,),
        ).fetchone()
        if row is None:
            return None
        safe = normalize_feedback(row)
        if safe is None:
            return None
        return Feedback(
            message_id=str(safe["message_id"]),
            helpful=bool(safe["helpful"]),
            created_at=str(safe["created_at"]),
            reason=safe.get("reason"),
        )

    def create_bad_case(self, message_id: str, code: str) -> BadCaseRecord:
        """Open a new bad case for a final assistant message idempotently."""

        if not isinstance(code, str) or code not in _BAD_CASE_CODES:
            raise ValueError("bad case code is invalid")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            message = self._connection.execute(
                "SELECT role, status FROM messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if message is None:
                raise KeyError(f"Unknown chat message: {message_id}")
            if message["role"] != "assistant" or message["status"] not in {
                "answered",
                "refused",
            }:
                raise ValueError("bad case requires a final assistant message")
            self._insert_bad_case(message_id, code)
            row = self._connection.execute(
                "SELECT * FROM bad_cases WHERE message_id = ? AND code = ?",
                (message_id, code),
            ).fetchone()
            self._connection.execute("COMMIT")
            return self._bad_case_from_row(row)
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def transition_bad_case(
        self, case_id: str, target_status: BadCaseStatus
    ) -> BadCaseRecord:
        """Advance one case through exactly one controlled lifecycle step."""

        if target_status not in _BAD_CASE_STATUSES:
            raise ValueError("bad case status is invalid")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                "SELECT * FROM bad_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown bad case: {case_id}")
            next_status = apply_bad_case_transition(row["status"], target_status)
            self._connection.execute(
                "UPDATE bad_cases SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE case_id = ?",
                (next_status, case_id),
            )
            updated = self._connection.execute(
                "SELECT * FROM bad_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            self._connection.execute("COMMIT")
            return self._bad_case_from_row(updated)
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def list_bad_case_aggregates(self) -> dict[str, dict[str, int]]:
        """Return code/status counts only; no message or source fields."""

        rows = self._connection.execute(
            """
            SELECT code, status, COUNT(*) AS count
            FROM bad_cases
            GROUP BY code, status
            ORDER BY code ASC, status ASC
            """
        ).fetchall()
        result: dict[str, dict[str, int]] = {}
        for row in rows:
            result.setdefault(str(row["code"]), {})[str(row["status"])] = int(
                row["count"]
            )
        return result

    def _insert_bad_case(self, message_id: str, code: str) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO bad_cases(
                case_id, message_id, code, status, document_version_id
            ) VALUES (?, ?, ?, 'new', NULL)
            """,
            (str(uuid4()), message_id, code),
        )

    def schema_version(self) -> int:
        return self._read_schema_version()

    def close(self) -> None:
        self._connection.close()

    @staticmethod
    def _version_from_row(
        row: sqlite3.Row, file_name: str, *, action: IndexAction
    ) -> DocumentVersion:
        return DocumentVersion(
            version_id=row["version_id"],
            file_name=file_name,
            sha256=row["sha256"],
            status=row["status"],
            action=action,
            failure_reason=row["failure_reason"],
        )

    @staticmethod
    def _summary_from_row(row: sqlite3.Row) -> DocumentSummary:
        metadata = None
        if row["business_version_id"] is not None:
            metadata = DocumentBusinessMetadata(
                version_id=str(row["business_version_id"]),
                content_type=row["business_content_type"],
                applicable_scope=row["business_applicable_scope"],
                effective_from=row["business_effective_from"],
                review_due_at=row["business_review_due_at"],
                business_status=row["business_status"],
                owner_role=row["business_owner_role"],
                supersedes_version_id=row["business_supersedes_version_id"],
                updated_at=str(row["business_updated_at"]),
            )
        return DocumentSummary(
            document_id=int(row["document_id"]),
            file_name=str(row["file_name"]),
            version_id=None if row["version_id"] is None else str(row["version_id"]),
            status=row["status"],
            sha256=None if row["sha256"] is None else str(row["sha256"]),
            failure_reason=row["failure_reason"],
            updated_at=None if row["updated_at"] is None else str(row["updated_at"]),
            business_metadata=metadata,
        )

    @staticmethod
    def _message_from_row(
        row: sqlite3.Row, *, fallback_session_id: str = "invalid-session"
    ) -> ChatMessage:
        raw = dict(row)
        try:
            raw["citations"] = json.loads(raw.get("citations_json") or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            raw["citations"] = None
        summary_json = raw.get("workflow_summary_json")
        if summary_json is not None:
            try:
                raw["workflow_summary"] = json.loads(summary_json)
            except (TypeError, ValueError, json.JSONDecodeError):
                # Keep corrupted metadata distinct from legacy NULL so the
                # safety normalizer can degrade the whole row safely.
                raw["workflow_summary"] = {"_invalid": True}
        else:
            raw["workflow_summary"] = None
        return normalize_chat_message(raw, fallback_session_id=fallback_session_id)

    @staticmethod
    def _bad_case_from_row(row: sqlite3.Row) -> BadCaseRecord:
        if row is None:
            raise RuntimeError("bad case row was not found")
        return BadCaseRecord(
            case_id=str(row["case_id"]),
            code=str(row["code"]),
            status=row["status"],
            document_version_id=(
                None
                if row["document_version_id"] is None
                else str(row["document_version_id"])
            ),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )


def _validate_message_text(value: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise TypeError("message content must be text")
    if not allow_empty and not value.strip():
        raise ValueError("message content must not be empty")
    if len(value) > 1_000_000:
        raise ValueError("message content is too long")
    if value and not is_safe_answer_text(value):
        raise ValueError("message content is unsafe")


def _validate_workflow_summary(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize the finite workflow summary contract."""

    from app.agents.contracts import WorkflowSummary

    summary = WorkflowSummary.model_validate(value)
    # The Pydantic contract owns shape/ranges. Storage additionally restricts
    # the optional reason to the public reason-code allowlist.
    validate_reason_code(summary.reason_code, allow_none=True)
    return summary.model_dump()


def _derive_chat_session_title(content: str) -> str:
    if not isinstance(content, str) or not content.strip() or not is_safe_answer_text(content):
        return DEFAULT_CHAT_SESSION_TITLE
    compact = re.sub(r"\s+", " ", content).strip()
    if len(compact) <= MAX_CHAT_SESSION_TITLE_CHARS:
        return compact or DEFAULT_CHAT_SESSION_TITLE
    return compact[: MAX_CHAT_SESSION_TITLE_CHARS - 1] + "…"


_CONTENT_TYPES = frozenset({"policy", "training", "procedure", "other"})
_AUDIENCE_SCOPES = frozenset(
    {"unspecified", "all_staff", "nurse", "doctor", "pharmacist", "administrator"}
)
_BUSINESS_STATUSES = frozenset(
    {"draft", "approved", "superseded", "retired", "unknown"}
)
_BAD_CASE_CODES = frozenset(
    {
        "USER_FEEDBACK_NOT_ANSWERED",
        "USER_FEEDBACK_MISSING_STEP",
        "USER_FEEDBACK_VERSION_MISMATCH",
        "USER_FEEDBACK_CITATION_MISMATCH",
        "USER_FEEDBACK_TOO_SLOW",
    }
)
_BAD_CASE_STATUSES = frozenset(
    {"new", "triaged", "fixed", "regression_checked", "closed"}
)


def _validate_business_metadata_input(
    metadata: DocumentBusinessMetadataInput,
) -> dict[str, object]:
    if metadata.content_type not in _CONTENT_TYPES:
        raise ValueError("content_type is invalid")
    if metadata.applicable_scope not in _AUDIENCE_SCOPES:
        raise ValueError("applicable_scope is invalid")
    if metadata.business_status not in _BUSINESS_STATUSES:
        raise ValueError("business_status is invalid")
    effective_from = _normalize_iso_date(metadata.effective_from, "effective_from")
    review_due_at = _normalize_iso_date(metadata.review_due_at, "review_due_at")
    owner_role = metadata.owner_role
    if owner_role is not None:
        if not isinstance(owner_role, str) or not owner_role.strip() or len(owner_role) > 80:
            raise ValueError("owner_role is invalid")
        owner_role = owner_role.strip()
        if not is_safe_answer_text(owner_role):
            raise ValueError("owner_role is invalid")
    supersedes_version_id = metadata.supersedes_version_id
    if supersedes_version_id is not None and not is_safe_id(supersedes_version_id):
        raise ValueError("supersedes version is invalid")
    return {
        "content_type": metadata.content_type,
        "applicable_scope": metadata.applicable_scope,
        "effective_from": effective_from,
        "review_due_at": review_due_at,
        "business_status": metadata.business_status,
        "owner_role": owner_role,
        "supersedes_version_id": supersedes_version_id,
    }


def _normalize_iso_date(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} date is invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} date is invalid") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} date is invalid")
    return value


def _validate_audience_scope(value: object) -> None:
    if value not in _AUDIENCE_SCOPES:
        raise ValueError("audience_scope is invalid")
