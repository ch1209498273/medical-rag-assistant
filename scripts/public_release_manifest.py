"""Allow-listed files and directories for the public project export."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicReleaseManifest:
    """Relative paths that may be included in a public release."""

    files: tuple[str, ...]
    directories: tuple[str, ...]
    excluded_files: tuple[str, ...] = ()


PUBLIC_RELEASE_MANIFEST = PublicReleaseManifest(
    files=(
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "SECURITY.md",
        "ROADMAP.md",
        ".env.example",
        ".gitignore",
        ".gitattributes",
        "backend/pyproject.toml",
        "backend/app/__init__.py",
        "backend/app/errors.py",
        "backend/app/main.py",
        "backend/app/settings.py",
        "backend/app/evaluation/__init__.py",
        "backend/app/evaluation/bad_cases.py",
        "backend/app/evaluation/industry_metrics.py",
        "backend/app/evaluation/protocol.py",
        "backend/app/evaluation/semantic.py",
        "backend/app/evaluation/service.py",
        "backend/app/evaluation/retrieval_diagnostics.py",
        "backend/app/evaluation/task14d_metrics.py",
        "backend/app/evaluation/task14d_models.py",
        "backend/app/evaluation/task14d_protocol.py",
        "frontend/index.html",
        "frontend/package.json",
        "frontend/package-lock.json",
        "frontend/tsconfig.json",
        "frontend/vite.config.ts",
        "demo/questions.json",
        ".github/workflows/ci.yml",
        ".github/PULL_REQUEST_TEMPLATE.md",
        "scripts/__init__.py",
        "scripts/build_demo_documents.py",
        "scripts/demo_smoke.py",
        "scripts/export_bad_case_summary.py",
        "scripts/start_demo.ps1",
        "scripts/start_cloud.ps1",
        "scripts/task17a_feedback_projection.py",
        "scripts/task17a_feedback_retention.py",
        "scripts/stop_local.ps1",
        "scripts/security_scan.py",
        "scripts/public_release_manifest.py",
        "scripts/prepare_public_release.py",
        "scripts/validate_public_release.py",
        "backend/tests/test_demo_runtime.py",
        "backend/tests/test_free_model_policy.py",
        "backend/tests/test_health.py",
        "backend/tests/test_runtime_async_cleanup.py",
        "backend/tests/test_runtime_mode.py",
        "backend/tests/test_security_scan.py",
        "backend/tests/test_settings_paths.py",
        "backend/tests/evaluation/test_bad_case_summary.py",
        "backend/tests/evaluation/test_industry_metrics.py",
        "backend/tests/evaluation/test_protocol.py",
        "backend/tests/evaluation/test_retrieval_diagnostics.py",
        "backend/tests/evaluation/test_semantic.py",
        "backend/tests/evaluation/test_task14d_metrics.py",
        "backend/tests/evaluation/test_task14d_protocol.py",
        "docs/architecture.md",
        "docs/evaluation.md",
        "docs/engineering-decisions.md",
        "docs/multi-agent-development.md",
        "docs/portfolio-brief.md",
        "docs/project-case-study.md",
        "docs/demo-guide.md",
        "docs/release-evidence.md",
    ),
    directories=(
        "backend/app/agents",
        "backend/app/api",
        "backend/app/chat",
        "backend/app/domain",
        "backend/app/feedback",
        "backend/app/ingestion",
        "backend/app/providers",
        "backend/app/rag",
        "backend/app/storage",
        "backend/tests/agents",
        "backend/tests/api",
        "backend/tests/chat",
        "backend/tests/demo",
        "backend/tests/feedback",
        "backend/tests/ingestion",
        "backend/tests/integration",
        "backend/tests/providers",
        "backend/tests/rag",
        "backend/tests/release",
        "backend/tests/storage",
        "frontend/src",
        "demo/sources",
        "demo/documents",
        "docs/assets",
    ),
    excluded_files=(
        # The heading-audit test accepts a real local source directory and is
        # intentionally kept out of the public portfolio export.
        "backend/tests/ingestion/test_heading_extraction_audit.py",
        # These tests exercise private staging CLIs or contain path/credential
        # canaries that are useful in the development tree but add no public
        # runtime coverage.
        "backend/tests/ingestion/test_docling_adapter.py",
        "backend/tests/ingestion/test_task15_cli.py",
        "backend/tests/feedback/test_demo_seed.py",
        "backend/tests/ingestion/test_task15_archive_preview.py",
        "backend/tests/rag/test_task9_instrumentation.py",
        "backend/tests/rag/test_weighted_rrf.py",
        "frontend/src/FeedbackReviewPage.test.tsx",
        "frontend/src/api/client.test.ts",
    ),
)
