"""Behavioral tests for the durable document-version repository."""

from __future__ import annotations

import json
import sqlite3
from datetime import date

import pytest

from app.domain.ports import DocumentBusinessMetadataInput
from app.storage.sqlite import SqliteDocumentRepository


def _safe_docx_citation(*, reference_id: str = "S1") -> dict[str, object]:
    return {
        "reference_id": reference_id,
        "file_name": "制度.docx",
        "heading_path": ["正文"],
        "page": None,
        "page_end": None,
        "paragraph_start": 1,
        "paragraph_end": 1,
        "excerpt": "公开样例制度摘录",
    }


def _create_task3_schema(database_path, *, duplicate_case_names: bool = False) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            """
            CREATE TABLE documents (
                document_id INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE document_versions (
                version_id TEXT PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(document_id),
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN (
                    'indexing', 'active', 'failed', 'superseded', 'needs_ocr'
                )),
                failure_reason TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE index_jobs (
                job_id TEXT PRIMARY KEY,
                version_id TEXT NOT NULL REFERENCES document_versions(version_id),
                status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE chat_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(session_id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE feedback (
                feedback_id TEXT PRIMARY KEY,
                session_id TEXT REFERENCES chat_sessions(session_id),
                message_id TEXT REFERENCES messages(message_id),
                helpful INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        names = ["制度.DOCX"]
        if duplicate_case_names:
            names.append("制度.docx")
        for index, name in enumerate(names, start=1):
            connection.execute(
                "INSERT INTO documents(document_id, file_name) VALUES (?, ?)",
                (index, name),
            )
            connection.execute(
                """
                INSERT INTO document_versions(version_id, document_id, sha256, status)
                VALUES (?, ?, ?, 'active')
                """,
                (f"legacy-version-{index}", index, f"legacy-hash-{index}"),
            )
        connection.execute(
            "INSERT INTO index_jobs(job_id, version_id, status) VALUES (?, ?, 'active')",
            ("legacy-job-1", "legacy-version-1"),
        )
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def repository(tmp_path):
    instance = SqliteDocumentRepository(tmp_path / "metadata.sqlite3")
    try:
        yield instance
    finally:
        instance.close()


def test_unchanged_hash_is_skipped(repository):
    """A regression would re-embed an unchanged active document."""
    first = repository.begin_index("制度.docx", "abc")
    repository.activate(first.version_id)

    second = repository.begin_index("制度.docx", "abc")

    assert second.action == "skip"
    assert second.version_id == first.version_id
    assert repository.get_active("制度.docx").version_id == first.version_id


def test_failed_new_version_does_not_replace_active(repository):
    """A failed replacement must leave the previously answerable version live."""
    old = repository.begin_index("制度.docx", "old")
    repository.activate(old.version_id)
    new = repository.begin_index("制度.docx", "new")

    repository.fail(new.version_id, "embedding timeout")

    assert repository.get_active("制度.docx").version_id == old.version_id
    assert repository.get_version(new.version_id).status == "failed"
    assert repository.get_version(new.version_id).failure_reason == "embedding timeout"


def test_activate_atomically_supersedes_old_active_version(repository):
    """A broken two-step activation would leave two active versions behind."""
    old = repository.begin_index("制度.docx", "old")
    repository.activate(old.version_id)
    new = repository.begin_index("制度.docx", "new")

    repository.activate(new.version_id)

    assert repository.get_active("制度.docx").version_id == new.version_id
    assert repository.get_version(old.version_id).status == "superseded"
    assert repository.get_version(new.version_id).status == "active"


def test_needs_ocr_version_cannot_be_activated(repository):
    """A scanned document must not become searchable before OCR succeeds."""
    version = repository.begin_index("扫描制度.pdf", "hash", needs_ocr=True)

    with pytest.raises(ValueError, match="needs_ocr"):
        repository.activate(version.version_id)

    assert repository.get_version(version.version_id).status == "needs_ocr"
    assert repository.get_active("扫描制度.pdf") is None


def test_completed_ocr_moves_version_to_indexing_then_can_be_activated(repository, tmp_path):
    """Without this transition, a successfully OCRed policy remains permanently offline."""
    old = repository.begin_index("扫描制度.pdf", "old")
    repository.activate(old.version_id)
    scanned = repository.begin_index("扫描制度.pdf", "new", needs_ocr=True)

    resumed = repository.begin_ocr_indexing(scanned.version_id)
    assert resumed.status == "indexing"
    assert resumed.action == "index"
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        job_status = connection.execute(
            "SELECT status FROM index_jobs WHERE version_id = ?", (scanned.version_id,)
        ).fetchone()[0]
    finally:
        connection.close()
    assert job_status == "indexing"

    repository.activate(resumed.version_id)

    assert repository.get_active("扫描制度.pdf").version_id == scanned.version_id
    assert repository.get_version(old.version_id).status == "superseded"
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        job_status = connection.execute(
            "SELECT status FROM index_jobs WHERE version_id = ?", (scanned.version_id,)
        ).fetchone()[0]
    finally:
        connection.close()
    assert job_status == "active"


def test_failed_ocr_does_not_replace_an_active_version(repository):
    """An OCR failure must be as safe as a normal embedding failure."""
    old = repository.begin_index("扫描制度.pdf", "old")
    repository.activate(old.version_id)
    scanned = repository.begin_index("扫描制度.pdf", "new", needs_ocr=True)

    repository.fail(scanned.version_id, "ocr timeout")

    assert repository.get_active("扫描制度.pdf").version_id == old.version_id
    assert repository.get_version(scanned.version_id).status == "failed"


def test_only_needs_ocr_versions_can_resume_ocr_indexing(repository):
    """A loose OCR transition could republish failed or active data accidentally."""
    indexing = repository.begin_index("制度.docx", "hash")

    with pytest.raises(ValueError, match="indexing"):
        repository.begin_ocr_indexing(indexing.version_id)

    assert repository.get_version(indexing.version_id).status == "indexing"


def test_same_hash_while_indexing_returns_the_existing_version(repository):
    """A second request must not pay to embed bytes already being indexed."""
    first = repository.begin_index("制度.docx", "same")

    second = repository.begin_index("制度.docx", "same")

    assert second.version_id == first.version_id
    assert second.action == "in_progress"
    assert second.status == "indexing"


def test_failed_hash_can_be_started_again(repository):
    """Deduplication must not prevent a user from retrying a failed indexing run."""
    failed = repository.begin_index("制度.docx", "same")
    repository.fail(failed.version_id, "temporary provider error")

    retry = repository.begin_index("制度.docx", "same")

    assert retry.action == "index"
    assert retry.version_id != failed.version_id


def test_same_hash_needing_ocr_returns_the_existing_ocr_version(repository):
    """Repeating scan detection must not create another pending OCR job."""
    first = repository.begin_index("扫描制度.pdf", "same", needs_ocr=True)

    second = repository.begin_index("扫描制度.pdf", "same", needs_ocr=True)

    assert second.version_id == first.version_id
    assert second.action == "needs_ocr"


def test_metadata_persists_across_repository_restart(tmp_path):
    """Replacing a process must not lose the active-version boundary."""
    database_path = tmp_path / "metadata.sqlite3"
    first = SqliteDocumentRepository(database_path)
    version = first.begin_index("制度.docx", "abc")
    first.activate(version.version_id)
    first.close()

    restarted = SqliteDocumentRepository(database_path)
    try:
        active = restarted.get_active("制度.docx")
        assert active is not None
        assert active.version_id == version.version_id
        assert restarted.active_version_ids() == {version.version_id}
    finally:
        restarted.close()


def test_initializes_all_planned_metadata_tables(repository, tmp_path):
    """A schema regression would strand later jobs, chats, or feedback data."""
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()

    assert {
        "documents",
        "document_versions",
        "index_jobs",
        "chat_sessions",
        "messages",
        "feedback",
    } <= tables


def test_migrates_task3_schema_and_preserves_data_for_inactive_snapshots(tmp_path):
    database_path = tmp_path / "legacy.sqlite3"
    _create_task3_schema(database_path)

    repository = SqliteDocumentRepository(database_path)
    try:
        assert repository.deactivate_missing(set()) == ["制度.DOCX"]
        assert repository.get_version("legacy-version-1").status == "inactive"
        assert repository.get_active("制度.docx") is None
    finally:
        repository.close()

    restarted = SqliteDocumentRepository(database_path)
    try:
        assert [item.file_name for item in restarted.list_documents()] == ["制度.DOCX"]
    finally:
        restarted.close()


def test_migration_applies_windows_case_insensitive_unique_semantics(tmp_path):
    database_path = tmp_path / "legacy-case.sqlite3"
    _create_task3_schema(database_path)

    repository = SqliteDocumentRepository(database_path)
    try:
        new_version = repository.begin_index("制度.docx", "new-hash")
        assert new_version.action == "index"
        assert [item.file_name for item in repository.list_documents()] == ["制度.DOCX"]
        assert repository.get_active("制度.docx").version_id == "legacy-version-1"
    finally:
        repository.close()


def test_failed_case_collision_migration_rolls_back_legacy_schema(tmp_path):
    database_path = tmp_path / "legacy-rollback.sqlite3"
    _create_task3_schema(database_path, duplicate_case_names=True)

    with pytest.raises(sqlite3.IntegrityError):
        SqliteDocumentRepository(database_path)

    connection = sqlite3.connect(database_path)
    try:
        names = [row[0] for row in connection.execute("SELECT file_name FROM documents")]
        schema_version = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
        ).fetchone()
    finally:
        connection.close()
    assert names == ["制度.DOCX", "制度.docx"]
    assert schema_version is None


def test_schema_initialization_failure_preserves_root_error_when_close_also_fails(
    tmp_path, monkeypatch
):
    import app.storage.sqlite as sqlite_module

    class DoubleFailureConnection:
        row_factory = None

        def execute(self, *args, **kwargs):
            return self

        def close(self):
            raise RuntimeError("sqlite-close-secondary")

    connection = DoubleFailureConnection()
    monkeypatch.setattr(
        sqlite_module.sqlite3, "connect", lambda *args, **kwargs: connection
    )
    monkeypatch.setattr(
        SqliteDocumentRepository,
        "_initialize_schema",
        lambda self: (_ for _ in ()).throw(ValueError("schema-init-original")),
    )

    with pytest.raises(ValueError, match="schema-init-original"):
        SqliteDocumentRepository(tmp_path / "metadata.sqlite3")


def test_schema_v3_migrates_existing_chat_rows_without_loss(tmp_path):
    database = tmp_path / "app.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version(version) VALUES (2);
        CREATE TABLE documents (
            document_id INTEGER PRIMARY KEY,
            file_name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE document_versions (
            version_id TEXT PRIMARY KEY,
            document_id INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            failure_reason TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE index_jobs (
            job_id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE chat_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE feedback (
            feedback_id TEXT PRIMARY KEY,
            session_id TEXT,
            message_id TEXT,
            helpful INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    connection.execute("INSERT INTO chat_sessions(session_id) VALUES ('s1')")
    connection.execute(
        "INSERT INTO messages(message_id, session_id, role, content) VALUES (?, ?, ?, ?)",
        ("m1", "s1", "user", "旧问题"),
    )
    connection.commit()
    connection.close()

    repository = SqliteDocumentRepository(database)
    try:
        messages = repository.list_chat_messages("s1")

        assert [message.content for message in messages] == ["旧问题"]
        assert repository.list_chat_sessions()[0].title == "旧问题"
        assert repository.schema_version() == 8
    finally:
        repository.close()


def test_feedback_upsert_is_idempotent_and_only_allows_final_assistant(repository, tmp_path):
    session_id = repository.create_chat_session()
    user = repository.append_user_message(session_id, "问题")
    assistant = repository.append_assistant_message(
        session_id,
        user.message_id,
        "答复",
        "answered",
        (),
        None,
    )

    first = repository.upsert_feedback(assistant.message_id, True)
    second = repository.upsert_feedback(assistant.message_id, False)

    assert first.message_id == second.message_id == assistant.message_id
    assert second.helpful is False
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1
    finally:
        connection.close()

    with pytest.raises(ValueError):
        repository.upsert_feedback(user.message_id, True)


def test_unhelpful_feedback_persists_controlled_reason_and_opens_bad_case(repository):
    session_id = repository.create_chat_session()
    user = repository.append_user_message(session_id, "问题")
    assistant = repository.append_assistant_message(
        session_id,
        user.message_id,
        "答复",
        "answered",
        (),
        None,
    )

    feedback = repository.upsert_feedback(
        assistant.message_id, False, "missing_step"
    )

    assert feedback.reason == "missing_step"
    assert repository.list_bad_case_aggregates() == {
        "USER_FEEDBACK_MISSING_STEP": {"new": 1}
    }


def test_helpful_feedback_rejects_a_feedback_reason(repository):
    session_id = repository.create_chat_session()
    user = repository.append_user_message(session_id, "问题")
    assistant = repository.append_assistant_message(
        session_id,
        user.message_id,
        "答复",
        "answered",
        (),
        None,
    )

    with pytest.raises(ValueError, match="reason"):
        repository.upsert_feedback(assistant.message_id, True, "too_slow")


def test_bad_case_status_machine_only_allows_adjacent_transitions(repository):
    session_id = repository.create_chat_session()
    user = repository.append_user_message(session_id, "问题")
    assistant = repository.append_assistant_message(
        session_id,
        user.message_id,
        "答复",
        "answered",
        (),
        None,
    )
    case = repository.create_bad_case(
        assistant.message_id, "USER_FEEDBACK_NOT_ANSWERED"
    )

    with pytest.raises(ValueError, match="invalid transition"):
        repository.transition_bad_case(case.case_id, "closed")

    for target in ("triaged", "fixed", "regression_checked", "closed"):
        case = repository.transition_bad_case(case.case_id, target)
        assert case.status == target


def test_v5_feedback_rows_migrate_with_a_nullable_reason(tmp_path):
    database = tmp_path / "v5-feedback.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version(version) VALUES (5);
        CREATE TABLE documents (document_id INTEGER PRIMARY KEY, file_name TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE document_versions (version_id TEXT PRIMARY KEY, document_id INTEGER NOT NULL, sha256 TEXT NOT NULL, status TEXT NOT NULL, failure_reason TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE index_jobs (job_id TEXT PRIMARY KEY, version_id TEXT NOT NULL, status TEXT NOT NULL, error_message TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE chat_sessions (session_id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '新会话', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE messages (message_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'submitted', reply_to_message_id TEXT, rewritten_question TEXT, citations_json TEXT NOT NULL DEFAULT '[]', reason_code TEXT, reference_answer TEXT, audience_scope TEXT NOT NULL DEFAULT 'unspecified', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE feedback (feedback_id TEXT PRIMARY KEY, session_id TEXT, message_id TEXT, helpful INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        INSERT INTO chat_sessions(session_id) VALUES ('s1');
        INSERT INTO messages(message_id, session_id, role, content) VALUES ('m1', 's1', 'user', '旧问题');
        INSERT INTO feedback(feedback_id, session_id, message_id, helpful) VALUES ('f1', 's1', 'm1', 0);
        """
    )
    connection.commit()
    connection.close()

    repository = SqliteDocumentRepository(database)
    try:
        assert repository.schema_version() == 8
        assert repository.get_feedback("m1").reason is None
    finally:
        repository.close()


def test_schema_v7_migrates_scope_constraint_to_accept_pharmacist(tmp_path):
    database = tmp_path / "v7-pharmacist.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version(version) VALUES (7);
        CREATE TABLE documents (document_id INTEGER PRIMARY KEY, file_name TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE document_versions (version_id TEXT PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(document_id), sha256 TEXT NOT NULL, status TEXT NOT NULL, failure_reason TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE document_business_metadata (
            version_id TEXT PRIMARY KEY REFERENCES document_versions(version_id),
            content_type TEXT NOT NULL,
            applicable_scope TEXT NOT NULL CHECK (applicable_scope IN ('unspecified', 'all_staff', 'nurse', 'doctor', 'administrator')),
            effective_from TEXT,
            review_due_at TEXT,
            business_status TEXT NOT NULL,
            owner_role TEXT,
            supersedes_version_id TEXT REFERENCES document_versions(version_id),
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO documents(document_id, file_name) VALUES (1, '药师制度.docx');
        INSERT INTO document_versions(version_id, document_id, sha256, status) VALUES ('v1', 1, 'hash', 'active');
        INSERT INTO document_business_metadata(version_id, content_type, applicable_scope, business_status) VALUES ('v1', 'training', 'all_staff', 'approved');
        """
    )
    connection.commit()
    connection.close()

    repository = SqliteDocumentRepository(database)
    try:
        assert repository.schema_version() == 8
        metadata = repository.upsert_document_business_metadata(
            1,
            DocumentBusinessMetadataInput(
                content_type="training",
                applicable_scope="pharmacist",
                effective_from=None,
                review_due_at=None,
                business_status="approved",
                owner_role=None,
                supersedes_version_id=None,
            ),
        )
        assert metadata.applicable_scope == "pharmacist"
    finally:
        repository.close()


def test_list_chat_sessions_orders_by_activity_and_limits(repository):
    first = repository.create_chat_session()
    second = repository.create_chat_session()
    repository.append_user_message(first, "第一会话")
    repository.append_user_message(second, "第二会话")

    summaries = repository.list_chat_sessions(limit=1)

    assert len(summaries) == 1
    assert summaries[0].session_id == second
    assert summaries[0].message_count == 1
    assert summaries[0].title == "第二会话"


def test_chat_session_title_uses_first_question_and_stays_stable(repository):
    session_id = repository.create_chat_session()
    repository.append_user_message(session_id, "透析低血压怎么处理？")
    repository.append_user_message(session_id, "后来还需要注意什么？")

    summary = repository.list_chat_sessions()[0]

    assert summary.title == "透析低血压怎么处理？"


def test_chat_session_title_is_bounded_for_long_questions(repository):
    session_id = repository.create_chat_session()
    repository.append_user_message(session_id, "长问题" * 30)

    title = repository.list_chat_sessions()[0].title

    assert len(title) <= 40
    assert title.endswith("…")


def test_reference_answer_round_trips_and_old_rows_default_to_none(repository):
    session_id = repository.create_chat_session()
    user = repository.append_user_message(session_id, "问题")
    repository.append_assistant_message(
        session_id,
        user.message_id,
        "依据不足",
        "refused",
        (),
        "INSUFFICIENT_EVIDENCE",
        reference_answer="通用学习参考",
    )

    restored = repository.list_chat_messages(session_id)[-1]

    assert restored.reference_answer == "通用学习参考"


def test_answered_message_round_trips_a_separate_reference_answer(repository):
    session_id = repository.create_chat_session()
    user = repository.append_user_message(session_id, "请假流程和审批时长？")
    repository.append_assistant_message(
        session_id,
        user.message_id,
        "资料可确认应先申请。",
        "answered",
        (_safe_docx_citation(),),
        None,
        reference_answer="通常时长需向负责人确认。",
    )

    restored = repository.list_chat_messages(session_id)[-1]

    assert restored.status == "answered"
    assert restored.reference_answer == "通常时长需向负责人确认。"


def test_document_summary_reports_display_version_updated_at(repository):
    version = repository.begin_index("制度.docx", "hash")
    repository.activate(version.version_id)

    summary = repository.list_documents()[0]

    assert summary.updated_at


def test_business_metadata_is_version_scoped_and_reports_unknown_by_default(repository):
    version = repository.begin_index("制度.docx", "hash")
    repository.activate(version.version_id)

    summary = repository.list_documents()[0]

    assert summary.business_metadata is None
    metadata = repository.upsert_document_business_metadata(
        summary.document_id,
        DocumentBusinessMetadataInput(
            content_type="policy",
            applicable_scope="all_staff",
            effective_from="2026-08-01",
            review_due_at="2027-08-01",
            business_status="unknown",
            owner_role="护理部资料管理员",
            supersedes_version_id=None,
        ),
    )

    assert metadata.version_id == version.version_id
    assert metadata.business_status == "unknown"
    assert repository.list_documents()[0].business_metadata == metadata


@pytest.mark.parametrize("field,value", [("effective_from", "2026/08/01"), ("review_due_at", "tomorrow")])
def test_business_metadata_rejects_non_iso_dates(repository, field, value):
    version = repository.begin_index("制度.docx", "hash")
    repository.activate(version.version_id)
    values = {
        "content_type": "policy",
        "applicable_scope": "all_staff",
        "effective_from": "2026-08-01",
        "review_due_at": "2027-08-01",
        "business_status": "approved",
        "owner_role": None,
        "supersedes_version_id": None,
    }
    values[field] = value

    with pytest.raises(ValueError, match="date"):
        repository.upsert_document_business_metadata(
            repository.list_documents()[0].document_id,
            DocumentBusinessMetadataInput(**values),
        )


def test_business_metadata_rejects_cross_document_supersedes(repository):
    old = repository.begin_index("旧制度.docx", "old")
    repository.activate(old.version_id)
    current = repository.begin_index("新制度.docx", "new")
    repository.activate(current.version_id)

    with pytest.raises(ValueError, match="supersedes"):
        repository.upsert_document_business_metadata(
            repository.list_documents()[0].document_id,
            DocumentBusinessMetadataInput(
                content_type="policy",
                applicable_scope="all_staff",
                effective_from="2026-08-01",
                review_due_at=None,
                business_status="approved",
                owner_role=None,
                supersedes_version_id=current.version_id,
            ),
        )


def test_schema_v4_migration_preserves_existing_chat_rows(tmp_path):
    database = tmp_path / "v3-chat.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version(version) VALUES (3);
        CREATE TABLE documents (document_id INTEGER PRIMARY KEY, file_name TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE document_versions (version_id TEXT PRIMARY KEY, document_id INTEGER NOT NULL, sha256 TEXT NOT NULL, status TEXT NOT NULL, failure_reason TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE index_jobs (job_id TEXT PRIMARY KEY, version_id TEXT NOT NULL, status TEXT NOT NULL, error_message TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE chat_sessions (session_id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '新会话', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE messages (message_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'submitted', reply_to_message_id TEXT, rewritten_question TEXT, citations_json TEXT NOT NULL DEFAULT '[]', reason_code TEXT, reference_answer TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE feedback (feedback_id TEXT PRIMARY KEY, session_id TEXT, message_id TEXT, helpful INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        INSERT INTO chat_sessions(session_id, title) VALUES ('session-1', '虚构会话');
        INSERT INTO messages(message_id, session_id, role, content) VALUES ('message-1', 'session-1', 'user', '虚构问题');
        """
    )
    connection.commit()
    connection.close()

    repository = SqliteDocumentRepository(database)
    try:
        assert repository.schema_version() == 8
        assert repository.list_chat_messages("session-1")[0].content == "虚构问题"
    finally:
        repository.close()


def test_eligible_versions_require_approved_status_scope_and_effective_dates(repository):
    all_staff = repository.begin_index("全员制度.docx", "all")
    repository.activate(all_staff.version_id)
    nurse = repository.begin_index("护士制度.docx", "nurse")
    repository.activate(nurse.version_id)
    unknown = repository.begin_index("待确认制度.docx", "unknown")
    repository.activate(unknown.version_id)

    for document in repository.list_documents():
        scope = "all_staff" if document.file_name == "全员制度.docx" else (
            "nurse" if document.file_name == "护士制度.docx" else "all_staff"
        )
        status = "approved" if document.file_name != "待确认制度.docx" else "unknown"
        repository.upsert_document_business_metadata(
            document.document_id,
            DocumentBusinessMetadataInput(
                content_type="policy",
                applicable_scope=scope,
                effective_from="2026-01-01",
                review_due_at="2026-12-31",
                business_status=status,
                owner_role=None,
                supersedes_version_id=None,
            ),
        )

    unspecified = repository.eligible_version_ids("unspecified", date(2026, 8, 30))
    nurse_scope = repository.eligible_version_ids("nurse", date(2026, 8, 30))

    assert unspecified.version_ids == frozenset({all_staff.version_id})
    assert nurse_scope.version_ids == frozenset({all_staff.version_id, nurse.version_id})
    assert unknown.version_id not in nurse_scope.version_ids


def test_eligible_versions_exclude_expired_approved_metadata(repository):
    version = repository.begin_index("过期制度.docx", "expired")
    repository.activate(version.version_id)
    document = repository.list_documents()[0]
    repository.upsert_document_business_metadata(
        document.document_id,
        DocumentBusinessMetadataInput(
            content_type="policy",
            applicable_scope="all_staff",
            effective_from="2025-01-01",
            review_due_at="2025-12-31",
            business_status="approved",
            owner_role=None,
            supersedes_version_id=None,
        ),
    )

    result = repository.eligible_version_ids("unspecified", date(2026, 8, 30))

    assert result.version_ids == frozenset()
    assert result.exclusion_reason == "EVIDENCE_SCOPE_UNCLEAR"


def test_schema_v3_migration_keeps_newest_feedback_row_before_unique_index(tmp_path):
    database = tmp_path / "duplicate-feedback.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version(version) VALUES (2);
        CREATE TABLE documents (
            document_id INTEGER PRIMARY KEY,
            file_name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE document_versions (
            version_id TEXT PRIMARY KEY,
            document_id INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            failure_reason TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE index_jobs (
            job_id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE chat_sessions (session_id TEXT PRIMARY KEY);
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        );
        CREATE TABLE feedback (
            feedback_id TEXT PRIMARY KEY,
            session_id TEXT,
            message_id TEXT,
            helpful INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO chat_sessions(session_id) VALUES ('s1');
        INSERT INTO messages(message_id, session_id, role, content)
            VALUES ('a1', 's1', 'assistant', '答');
        INSERT INTO feedback(feedback_id, session_id, message_id, helpful, created_at)
            VALUES ('f-old', 's1', 'a1', 0, '2020-01-01 00:00:00');
        INSERT INTO feedback(feedback_id, session_id, message_id, helpful, created_at)
            VALUES ('f-new', 's1', 'a1', 1, '2025-01-01 00:00:00');
        """
    )
    connection.commit()
    connection.close()

    repository = SqliteDocumentRepository(database)
    try:
        assert repository.get_feedback("a1").helpful is True
        connection = sqlite3.connect(database)
        try:
            assert connection.execute(
                "SELECT COUNT(*) FROM feedback WHERE message_id = 'a1'"
            ).fetchone()[0] == 1
        finally:
            connection.close()
    finally:
        repository.close()


def test_assistant_message_rejects_path_bearing_citations(repository):
    session_id = repository.create_chat_session()
    user = repository.append_user_message(session_id, "问题")

    with pytest.raises(ValueError):
        repository.append_assistant_message(
            session_id,
            user.message_id,
            "答复",
            "answered",
            (
                {
                    "reference_id": "S1",
                    "file_name": "C:" + chr(92) + "secret" + chr(92) + "policy.docx",
                },
            ),
            None,
        )

    assert [item.role for item in repository.list_chat_messages(session_id)] == ["user"]


def test_corrupt_or_unsafe_citation_snapshot_restores_as_error(repository, tmp_path):
    session_id = repository.create_chat_session()
    user = repository.append_user_message(session_id, "问题")
    assistant = repository.append_assistant_message(
        session_id,
        user.message_id,
        "答复",
        "answered",
        (_safe_docx_citation(),),
        None,
    )
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute(
            "UPDATE messages SET citations_json = ? WHERE message_id = ?",
            (
                json.dumps(
                    [
                        {
                            "reference_id": "S1",
                            "file_name": "C:"
                            + chr(92)
                            + "secret"
                            + chr(92)
                            + "policy.docx",
                        }
                    ]
                ),
                assistant.message_id,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    restored = repository.list_chat_messages(session_id)[-1]
    assert restored.status == "error"
    assert restored.citations == ()


def test_terminal_assistant_append_is_idempotent_for_one_user_message(repository, tmp_path):
    session_id = repository.create_chat_session()
    user = repository.append_user_message(session_id, "问题")

    first = repository.append_assistant_message(
        session_id,
        user.message_id,
        "第一份答复",
        "answered",
        (_safe_docx_citation(),),
        None,
    )
    replay = repository.append_assistant_message(
        session_id,
        user.message_id,
        "重放时不应覆盖",
        "answered",
        (_safe_docx_citation(),),
        None,
    )

    assert replay.message_id == first.message_id
    assert replay.content == "第一份答复"
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM messages WHERE reply_to_message_id = ?",
            (user.message_id,),
        ).fetchone()[0] == 1
        index = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_messages_terminal_reply'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert "UNIQUE" in index.upper()


def test_schema_migration_deduplicates_terminal_replies_before_unique_index(tmp_path):
    database = tmp_path / "duplicate-replies.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version(version) VALUES (2);
        CREATE TABLE documents (
            document_id INTEGER PRIMARY KEY,
            file_name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE document_versions (
            version_id TEXT PRIMARY KEY,
            document_id INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            failure_reason TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE index_jobs (
            job_id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE chat_sessions (session_id TEXT PRIMARY KEY);
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT NOT NULL,
            reply_to_message_id TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE feedback (
            feedback_id TEXT PRIMARY KEY,
            session_id TEXT,
            message_id TEXT,
            helpful INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO chat_sessions(session_id) VALUES ('s1');
        INSERT INTO messages(message_id, session_id, role, content, status, reply_to_message_id, created_at)
            VALUES ('u1', 's1', 'user', '问题', 'submitted', NULL, '2026-01-01 00:00:00');
        INSERT INTO messages(message_id, session_id, role, content, status, reply_to_message_id, created_at)
            VALUES ('a-old', 's1', 'assistant', '旧答复', 'answered', 'u1', '2026-01-01 00:00:01');
        INSERT INTO messages(message_id, session_id, role, content, status, reply_to_message_id, created_at)
            VALUES ('a-new', 's1', 'assistant', '新答复', 'answered', 'u1', '2026-01-01 00:00:02');
        """
    )
    connection.commit()
    connection.close()

    repository = SqliteDocumentRepository(database)
    try:
        restored = repository.list_chat_messages("s1")
        assert [item.message_id for item in restored] == ["u1", "a-new"]
        assert repository.schema_version() == 8
    finally:
        repository.close()


def test_tampered_message_row_fails_closed_to_safe_error_dto(repository, tmp_path):
    session_id = repository.create_chat_session()
    user = repository.append_user_message(session_id, "问题")
    assistant = repository.append_assistant_message(
        session_id,
        user.message_id,
        "安全答复",
        "answered",
        (_safe_docx_citation(),),
        None,
    )
    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute(
            "UPDATE messages SET content = ?, status = ?, reason_code = ?, created_at = ? WHERE message_id = ?",
            (
                "Traceback (most recent call last): C:"
                + chr(92)
                + "private"
                + chr(92)
                + "provider.log",
                "provider-stack",
                "raw-provider-detail",
                "C:" + chr(92) + "private" + chr(92) + "timestamp",
                assistant.message_id,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    restored = repository.list_chat_messages(session_id)[-1]
    assert restored.role == "assistant"
    assert restored.status == "error"
    assert restored.content == "当前会话记录无法安全恢复。"
    assert restored.reason_code == "CHAT_UNAVAILABLE"
    assert "provider" not in restored.content.lower()
