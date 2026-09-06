from __future__ import annotations

from pathlib import Path

from app.feedback.evidence import EvidenceBundle, EvidenceReference
from app.feedback.repository import FeedbackProjectionRepository
from app.feedback.review_service import FeedbackReviewService
from app.main import create_app
from app.settings import Settings
from fastapi.testclient import TestClient


class FakeEvidenceResolver:
    def resolve(self, *, source_version, citation_chunk_ids):
        if source_version == "2026-standard-manual-v1" and citation_chunk_ids:
            return EvidenceBundle(
                "available",
                references=(
                    EvidenceReference(
                        chunk_id=citation_chunk_ids[0],
                        source_version=source_version,
                        file_name="制度.docx",
                        heading_path=("透析管理",),
                        page=None,
                        page_end=None,
                        paragraph_start=1,
                        paragraph_end=2,
                        excerpt="按制度执行。",
                    ),
                ),
            )
        return EvidenceBundle("unavailable", reason_code="citation_chunk_missing")


def _path(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "private" / "feedback" / "feedback.db"
    path.parent.mkdir(parents=True)
    return path


def _client(tmp_path: Path, *, enabled: bool = True) -> TestClient:
    repository = FeedbackProjectionRepository(_path(tmp_path))
    from app.feedback.models import FeedbackCase

    case = FeedbackCase(
        case_id="fc_safe",
        message_id_hash="a" * 64,
        session_id_hash="b" * 64,
        collected_at="2026-09-03T10:00:00+08:00",
        question_redacted="透析中低血压怎么处理？",
        answer_redacted="请按本单位流程处理。",
        redaction_status="passed",
        redaction_version="redactor-v1",
        redaction_flags=(),
        answer_status="answered",
        reason_code=None,
        source_version="2026-standard-manual-v1",
        retrieval_profile="baseline_v1+vector+c1",
        citation_chunk_ids=("chunk-1",),
        citation_count=1,
        model_id="deepseek-v4-flash",
        prompt_version="c1",
        latency_bucket="lt_5s",
        audience_scope="nurse",
        retention_expires_at="2026-12-02T10:00:00+08:00",
        triage_priority="P0",
        triage_score=8,
        triage_reasons=("user_unhelpful",),
        review_status="unreviewed",
        promotion_status="not_promoted",
    )
    repository.upsert_case(case)
    service = FeedbackReviewService(repository)
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        private_data_dir=tmp_path / "private",
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
        feedback_review_ui_enabled=enabled,
    )
    app = create_app(
        settings,
        document_service=object(),
        feedback_review_service=service,
        evidence_resolver=FakeEvidenceResolver(),
    )
    app.state._test_feedback_repository = repository
    return TestClient(app)


def _close_client(client: TestClient) -> None:
    repository = getattr(client.app.state, "_test_feedback_repository", None)
    if repository is not None:
        repository.close()


def test_admin_feedback_detail_returns_evidence_and_history_without_raw_fields(tmp_path):
    client = _client(tmp_path)
    try:
        with client:
            response = client.get("/api/admin/feedback/cases/fc_safe")
    finally:
        _close_client(client)

    assert response.status_code == 200
    body = response.json()
    assert body["evidence"]["status"] == "available"
    assert body["case"]["question"] == "透析中低血压怎么处理？"
    assert "message_id" not in body["case"]
    assert "source_path" not in body["case"]


def test_p0_review_without_clinical_reviewer_cannot_promote(tmp_path):
    client = _client(tmp_path)
    try:
        with client:
            response = client.get(
                "/api/admin/feedback/cases/fc_safe/promotion-check"
            )
    finally:
        _close_client(client)

    assert response.status_code == 200
    assert response.json()["status"] == "not_promoted"
    assert "review_missing" in response.json()["reasons"]


def test_promotion_endpoint_keeps_review_gate_closed(tmp_path):
    client = _client(tmp_path)
    try:
        with client:
            response = client.post(
                "/api/admin/feedback/cases/fc_safe/promote",
                json={
                    "target_set_id": "2026-standard-manual-v1-golden-v2",
                    "target_version": "candidate-001",
                    "target_split": "dev",
                    "manifest_id": "manifest-synthetic-001",
                },
            )
    finally:
        _close_client(client)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "FEEDBACK_PROMOTION_BLOCKED"
    assert "review_missing" in response.json()["detail"]["promotion"]["reasons"]


def test_disabled_admin_feedback_returns_fixed_error_code(tmp_path):
    client = _client(tmp_path, enabled=False)
    try:
        with client:
            response = client.get("/api/admin/feedback/summary")
    finally:
        _close_client(client)

    assert response.status_code == 503
    assert response.json()["detail"] == "FEEDBACK_REVIEW_DISABLED"


def test_review_submission_appends_history_and_updates_only_private_status(tmp_path):
    client = _client(tmp_path)
    try:
        with client:
            response = client.post(
                "/api/admin/feedback/cases/fc_safe/reviews",
                json={
                    "reviewer_role": "clinical_reviewer",
                    "decision": "approve",
                    "evidence_ok": True,
                    "points_ok": True,
                    "safety_ok": True,
                    "note_code": "confirmed",
                    "review_version": "task17b-v1",
                },
            )
            assert response.status_code == 200
            assert response.json()["review"]["reviewer_role"] == "clinical_reviewer"

            detail = client.get("/api/admin/feedback/cases/fc_safe")
            assert detail.status_code == 200
            assert len(detail.json()["reviews"]) == 1
            assert detail.json()["case"]["review_status"] == "approved"
    finally:
        _close_client(client)


def test_explicit_promotion_appends_candidate_record_after_review_gates(tmp_path):
    client = _client(tmp_path)
    try:
        with client:
            review = client.post(
                "/api/admin/feedback/cases/fc_safe/reviews",
                json={
                    "reviewer_role": "clinical_reviewer",
                    "decision": "approve",
                    "evidence_ok": True,
                    "points_ok": True,
                    "safety_ok": True,
                    "note_code": "confirmed",
                    "review_version": "task17b-v1",
                },
            )
            assert review.status_code == 200

            response = client.post(
                "/api/admin/feedback/cases/fc_safe/promote",
                json={
                    "target_set_id": "2026-standard-manual-v1-golden-v2",
                    "target_version": "candidate-001",
                    "target_split": "dev",
                    "manifest_id": "manifest-synthetic-001",
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert body["promotion"]["status"] == "golden_v2_candidate"
            assert body["record"]["target_split"] == "dev"
            assert body["record"]["target_set_id"] == "2026-standard-manual-v1-golden-v2"

            detail = client.get("/api/admin/feedback/cases/fc_safe")
            assert detail.status_code == 200
            assert detail.json()["case"]["promotion_status"] == "golden_v2_candidate"
            assert len(detail.json()["promotions"]) == 1
    finally:
        _close_client(client)
