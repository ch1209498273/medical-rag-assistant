"""Allow-listed files and directories for the public project export."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicReleaseManifest:
    """Relative paths that may be included in a public release."""

    files: tuple[str, ...]
    directories: tuple[str, ...]


PUBLIC_RELEASE_MANIFEST = PublicReleaseManifest(
    files=(
        "README.md",
        "LICENSE",
        "SECURITY.md",
        "ROADMAP.md",
        ".env.example",
        ".gitignore",
        ".gitattributes",
        "backend/pyproject.toml",
        "frontend/index.html",
        "frontend/package.json",
        "frontend/package-lock.json",
        "frontend/tsconfig.json",
        "frontend/vite.config.ts",
        "demo/questions.json",
        ".github/workflows/ci.yml",
        "scripts/__init__.py",
        "scripts/build_demo_documents.py",
        "scripts/demo_smoke.py",
        "scripts/start_demo.ps1",
        "scripts/start_cloud.ps1",
        "scripts/stop_local.ps1",
        "scripts/security_scan.py",
        "scripts/public_release_manifest.py",
        "scripts/prepare_public_release.py",
        "scripts/validate_public_release.py",
        "docs/architecture.md",
        "docs/evaluation.md",
        "docs/engineering-decisions.md",
        "docs/multi-agent-development.md",
        "docs/project-case-study.md",
        "docs/demo-guide.md",
    ),
    directories=(
        "backend/app",
        "backend/tests",
        "frontend/src",
        "demo/sources",
        "demo/documents",
        "docs/assets",
    ),
)
