import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.security_scan import contains_absolute_path, contains_secret
from scripts.validate_public_release import (
    Finding,
    HistoryUnavailable,
    validate_public_release,
)


def _windows_path(*parts: str) -> str:
    return "C:" + chr(92) + chr(92).join(parts)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def initialise_git_repository(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-b", "main")
    git(path, "config", "user.email", "tests@example.invalid")
    git(path, "config", "user.name", "Release Tests")
    return path


def commit_file(repository: Path, relative: str, text: str) -> None:
    write(repository / relative, text)
    git(repository, "add", "--", relative)
    git(repository, "commit", "-m", "fixture")


def remove_and_commit(repository: Path, relative: str) -> None:
    git(repository, "rm", "--", relative)
    git(repository, "commit", "-m", "remove fixture")


def png_bytes(width: int = 1, height: int = 1) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        import struct
        import zlib

        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    import struct

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", b"\x78\x9c\x63\x00\x00\x00\x02\x00\x01")
        + chunk(b"IEND", b"")
    )


def gif_bytes(width: int = 1, height: int = 1) -> bytes:
    import struct

    return (
        b"GIF89a"
        + struct.pack("<HH", width, height)
        + b"\x80\x00\x00"
        + b"\x00\x00\x00\xff\xff\xff"
        + b","
        + struct.pack("<HHHH", 0, 0, width, height)
        + b"\x00\x02\x02\x44\x01\x00\x3b"
    )


def docx_bytes(*, include_required_parts: bool = True) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        if include_required_parts:
            archive.writestr(
                "[Content_Types].xml",
                "<?xml version='1.0'?><Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types' />",
            )
            archive.writestr(
                "_rels/.rels",
                "<?xml version='1.0'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships' />",
            )
            archive.writestr(
                "word/document.xml",
                "<?xml version='1.0'?><w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'><w:body /></w:document>",
            )
        else:
            archive.writestr("content.txt", "not a docx")
    return stream.getvalue()


def run_validator_cli(*args: str) -> subprocess.CompletedProcess[str]:
    script = (
        Path(__file__).resolve().parents[3] / "scripts" / "validate_public_release.py"
    )
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_validator_blocks_env_database_and_machine_path(tmp_path: Path):
    write(tmp_path / ".env", "DEEPSEEK_API_KEY=" + "sk-" + "x" * 40)
    write(tmp_path / "data" / "app.sqlite3", "not-public")
    separator = chr(92)
    write(
        tmp_path / "README.md",
        "machine path: "
        + "C:"
        + separator
        + "Users"
        + separator
        + "Example"
        + separator
        + "资料",
    )

    kinds = {
        item.kind for item in validate_public_release(tmp_path, include_history=False)
    }

    assert {"blocked_path", "secret", "absolute_path"} <= kinds


def test_history_scan_finds_a_secret_deleted_from_head(tmp_path: Path):
    repository = initialise_git_repository(tmp_path)
    commit_file(repository, "leaked.txt", "token=" + "sk-" + "x" * 40)
    remove_and_commit(repository, "leaked.txt")

    findings = validate_public_release(repository, include_history=True)

    assert any(item.kind == "history_secret" for item in findings)


def test_clean_public_text_allows_domain_slashes_routes_and_urls(tmp_path: Path):
    write(tmp_path / "README.md", "GB/T 8982 uses 5 mg/kg over TCP/IP.\n")
    write(
        tmp_path / "docs.md",
        "See [the API](" + "/api/chat/stream" + ") and [guide](guide.md).\n",
    )
    write(tmp_path / "guide.md", "https://example.invalid/public\n")

    assert validate_public_release(tmp_path, include_history=False) == ()


def test_findings_are_structured_and_never_include_secret_values(tmp_path: Path):
    secret = "sk-" + "x" * 40
    write(tmp_path / "config.txt", "DEEPSEEK_API_KEY=" + secret)

    findings = validate_public_release(tmp_path, include_history=False)

    assert findings
    assert all(isinstance(item, Finding) for item in findings)
    assert all(item.path == "config.txt" for item in findings)
    assert all(secret not in item.detail for item in findings)
    assert {item.kind for item in findings} == {"secret"}


def test_validator_allows_explicit_canary_but_not_secret_word_in_long_value(
    tmp_path: Path,
):
    write(tmp_path / "fixture.txt", "token=sk-" + "test-canary-123456789\n")
    write(
        tmp_path / "config.txt",
        "DEEPSEEK_API_KEY=" + "real-secret-key-" + "x" * 32 + "\n",
    )

    findings = validate_public_release(tmp_path, include_history=False)

    assert [item.path for item in findings if item.kind == "secret"] == ["config.txt"]


@pytest.mark.parametrize(
    "relative",
    [
        ".env",
        "data/runtime/app.sqlite3",
        "data/private/report.json",
        ".superpowers/sdd/private.md",
        "docs/superpowers/plan.md",
        "notes.zip",
        "notes.sqlite3",
        ".netrc",
        "credentials.json",
        "id_rsa",
        "runtime.log.bak",
    ],
)
def test_validator_blocks_protected_paths_and_artifacts(tmp_path: Path, relative: str):
    write(tmp_path / relative, "not-public")

    findings = validate_public_release(tmp_path, include_history=False)

    assert any(item.kind in {"blocked_path", "blocked_file"} for item in findings)
    assert all(
        item.path == relative.replace("\\", "/")
        or relative.replace("\\", "/").startswith(item.path + "/")
        for item in findings
    )


def test_validator_scans_index_content_when_worktree_differs(tmp_path: Path):
    repository = initialise_git_repository(tmp_path)
    commit_file(repository, "config.txt", "safe")
    write(repository / "config.txt", "DEEPSEEK_API_KEY=" + "x" * 40)
    git(repository, "add", "--", "config.txt")
    write(repository / "config.txt", "safe working tree")

    findings = validate_public_release(repository, include_history=False)

    assert any(item.kind == "secret" and item.path == "config.txt" for item in findings)


def test_validator_scans_index_for_a_nested_git_root(tmp_path: Path):
    repository = initialise_git_repository(tmp_path / "repository")
    project = repository / "project"
    commit_file(repository, "project/config.txt", "safe")
    write(project / "config.txt", "DEEPSEEK_API_KEY=" + "x" * 40)
    git(repository, "add", "--", "project/config.txt")
    write(project / "config.txt", "safe working tree")

    findings = validate_public_release(project, include_history=False)

    assert any(item.kind == "secret" and item.path == "config.txt" for item in findings)


def test_validator_scans_tracked_unapproved_binary_when_worktree_is_missing(
    tmp_path: Path,
):
    repository = initialise_git_repository(tmp_path)
    write(repository / "private.png", "not-an-approved-public-asset")
    git(repository, "add", "--", "private.png")
    git(repository, "commit", "-m", "fixture")
    (repository / "private.png").unlink()

    findings = validate_public_release(repository, include_history=False)

    assert any(
        item.kind == "blocked_file" and item.path == "private.png" for item in findings
    )


def test_history_scan_reports_deleted_protected_content(tmp_path: Path):
    repository = initialise_git_repository(tmp_path)
    commit_file(repository, ".env", "DEEPSEEK_API_KEY=" + "sk-" + "x" * 40)
    remove_and_commit(repository, ".env")

    findings = validate_public_release(repository, include_history=True)
    kinds = {item.kind for item in findings}

    assert "history_secret" in kinds
    assert "history_blocked_path" in kinds


def test_validator_checks_markdown_links(tmp_path: Path):
    write(tmp_path / "README.md", "[missing](missing.md)\n")

    findings = validate_public_release(tmp_path, include_history=False)

    assert any(
        item.kind == "broken_link" and item.path == "README.md" for item in findings
    )


def test_validator_blocks_markdown_links_to_protected_content(tmp_path: Path):
    write(tmp_path / "README.md", "[private](data/private/report.md)\n")
    write(tmp_path / "data" / "private" / "report.md", "private\n")

    findings = validate_public_release(tmp_path, include_history=False)

    assert any(
        item.kind == "broken_link" and item.path == "README.md" for item in findings
    )


def test_validator_allows_protocol_relative_external_markdown_links(tmp_path: Path):
    write(tmp_path / "README.md", "[external](//example.invalid/guide)\n")

    assert validate_public_release(tmp_path, include_history=False) == ()


def test_validator_checks_asset_manifest_hashes_and_membership(tmp_path: Path):
    asset = tmp_path / "docs" / "assets" / "image.png"
    write(asset, "not-an-image")
    write(
        asset.parent / "manifest.json",
        '{"schema_version": 1, "assets": [{"file": "image.png", '
        '"scenario_id": "demo", "sha256": "0" * 64, '
        '"width": 1, "height": 1, "reviewed": true}]}',
    )

    findings = validate_public_release(tmp_path, include_history=False)

    assert any(item.kind == "asset_manifest" for item in findings)


@pytest.mark.parametrize("file_name", ["CON.png", "image.png ", "C:image.png"])
def test_validator_rejects_platform_ambiguous_asset_names(
    tmp_path: Path, file_name: str
):
    asset_root = tmp_path / "docs" / "assets"
    write(asset_root / "image.png", "not-an-image")
    write(
        asset_root / "manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "file": file_name,
                        "scenario_id": "demo",
                        "sha256": "0" * 64,
                        "width": 1,
                        "height": 1,
                        "reviewed": True,
                    }
                ],
            }
        ),
    )

    findings = validate_public_release(tmp_path, include_history=False)

    assert any(
        item.kind == "asset_manifest"
        and item.detail == "asset file names are invalid or duplicated"
        for item in findings
    )


def test_validator_rejects_invalid_asset_scenario_id(tmp_path: Path):
    asset_root = tmp_path / "docs" / "assets"
    write(
        asset_root / "manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "file": "image.png",
                        "scenario_id": {"unsafe": True},
                        "sha256": "0" * 64,
                        "width": 1,
                        "height": 1,
                        "reviewed": True,
                    }
                ],
            }
        ),
    )

    findings = validate_public_release(tmp_path, include_history=False)

    assert any(
        item.kind == "asset_manifest" and item.detail == "asset entry schema is invalid"
        for item in findings
    )


def test_validator_reports_symlink_without_following_it(tmp_path: Path):
    outside = tmp_path / "outside.txt"
    write(outside, "DEEPSEEK_API_KEY=" + "x" * 40)
    linked = tmp_path / "linked.txt"
    try:
        linked.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")

    findings = validate_public_release(tmp_path, include_history=False)

    assert any(
        item.kind == "blocked_path" and item.path == "linked.txt" for item in findings
    )
    assert not any(
        item.path == "outside.txt" and item.kind == "secret" for item in findings
    )


def test_validator_rejects_root_with_a_symlinked_ancestor(tmp_path: Path):
    outside = tmp_path / "outside"
    write(outside / "child" / "README.md", "DEEPSEEK_API_KEY=" + "x" * 40)
    linked = tmp_path / "linked-root"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this host")

    with pytest.raises(ValueError, match="symlink|reparse"):
        validate_public_release(linked / "child", include_history=False)


def test_validator_blocks_a_tracked_symlink_even_when_worktree_is_missing(
    tmp_path: Path,
):
    repository = initialise_git_repository(tmp_path)
    target = repository / "target.txt"
    write(target, "safe")
    linked = repository / "linked.txt"
    try:
        linked.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this host")
    git(repository, "add", "--", "target.txt", "linked.txt")
    git(repository, "commit", "-m", "fixture")
    linked.unlink()

    findings = validate_public_release(repository, include_history=False)

    assert any(
        item.kind == "blocked_path" and item.path == "linked.txt" for item in findings
    )


def test_validator_checks_staged_markdown_links(tmp_path: Path):
    repository = initialise_git_repository(tmp_path)
    commit_file(repository, "README.md", "[guide](guide.md)\n")
    write(repository / "README.md", "[missing](missing.md)\n")
    git(repository, "add", "--", "README.md")
    write(repository / "README.md", "[guide](guide.md)\n")
    write(repository / "guide.md", "safe\n")

    findings = validate_public_release(repository, include_history=False)

    assert any(
        item.kind == "broken_link" and item.path == "README.md" for item in findings
    )


def test_validator_checks_staged_asset_manifest(tmp_path: Path):
    repository = initialise_git_repository(tmp_path)
    manifest = repository / "docs" / "assets" / "manifest.json"
    write(manifest, '{"schema_version": 1, "assets": []}\n')
    git(repository, "add", "--", "docs/assets/manifest.json")
    git(repository, "commit", "-m", "fixture")
    write(manifest, '{"schema_version": "unsafe", "assets": []}\n')
    git(repository, "add", "--", "docs/assets/manifest.json")
    write(manifest, '{"schema_version": 1, "assets": []}\n')

    findings = validate_public_release(repository, include_history=False)

    assert any(
        item.kind == "asset_manifest"
        and item.detail == "asset manifest schema is invalid"
        for item in findings
    )


def test_cli_returns_zero_for_clean_root_and_one_for_findings(tmp_path: Path):
    write(tmp_path / "README.md", "GB/T 8982 and TCP/IP; [guide](guide.md)\n")
    write(tmp_path / "guide.md", "safe\n")
    clean = run_validator_cli("--root", str(tmp_path))
    assert clean.returncode == 0
    assert clean.stdout == ""

    write(tmp_path / ".env", "DEEPSEEK_API_KEY=" + "sk-" + "x" * 40)
    findings = run_validator_cli("--root", str(tmp_path))
    assert findings.returncode == 1
    assert "sk-" not in findings.stdout


def test_cli_returns_two_when_history_is_unavailable(tmp_path: Path):
    result = run_validator_cli("--root", str(tmp_path), "--history")

    assert result.returncode == 2
    assert "history" in result.stderr.casefold()
    assert "Traceback" not in result.stderr


def test_history_rejects_symlink_and_uninspectable_binary_from_an_old_tree(
    tmp_path: Path,
):
    repository = initialise_git_repository(tmp_path)
    write(repository / "target.txt", "safe\n")
    linked = repository / "linked.txt"
    try:
        linked.symlink_to(repository / "target.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    write_bytes(repository / "old.bin", b"\x00\xff\x00")
    git(repository, "add", "--", "target.txt", "linked.txt", "old.bin")
    git(repository, "commit", "-m", "old tree")
    git(repository, "rm", "--", "linked.txt", "old.bin")
    git(repository, "commit", "-m", "remove old tree")

    findings = validate_public_release(repository, include_history=True)

    assert any(item.kind == "history_symlink" and item.path == "linked.txt" for item in findings)
    assert any(item.kind == "history_blocked_file" and item.path == "old.bin" for item in findings)


def test_history_rejects_shallow_repository(tmp_path: Path):
    repository = initialise_git_repository(tmp_path)
    commit_file(repository, "README.md", "safe\n")
    shallow = repository / subprocess.run(
        ["git", "rev-parse", "--git-path", "shallow"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    shallow.write_text(
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout,
        encoding="ascii",
    )

    with pytest.raises(HistoryUnavailable):
        validate_public_release(repository, include_history=True)


@pytest.mark.parametrize(
    "credential",
    [
        "-----BEGIN " + "PRIVATE KEY-----",
        "gh" + "p_" + "A" * 36,
        "github_" + "pat_" + "A" * 22 + "_" + "B" * 20,
        "AK" + "IA" + "A" * 16,
        "aws_" + "secret_access_key=" + "A" * 40,
        "Authorization: "
        + "Bearer "
        + "eyJ"
        + "A" * 20
        + ".eyJ"
        + "B" * 20
        + "."
        + "C" * 20,
        "Aiz" + "a" + "A" * 35,
        "xox" + "b-" + "A" * 30,
    ],
)
def test_security_scan_rejects_provider_secret_canaries(credential: str):
    assert contains_secret(credential)


def test_security_scan_allows_only_the_fixed_provider_canary():
    fixed = "sk-test-canary-123456789"
    near_miss = "sk-test-canary-" + "123456780"

    assert not contains_secret("DEEPSEEK_API_KEY=" + fixed)
    assert contains_secret("DEEPSEEK_API_KEY=" + near_miss)


def test_security_scan_masks_only_actual_regex_literals():
    assert not contains_absolute_path('re.compile(r"/workspace/secret")')
    assert not contains_absolute_path('re.search(pattern=r"/workspace/secret", text)')
    assert contains_absolute_path("documentation: " + "/" + "workspace/secret")
    assert contains_absolute_path('text = r"' + "/" + 'workspace/secret"')


def test_validator_checks_markdown_reference_definitions_and_html_attributes(
    tmp_path: Path,
):
    write(
        tmp_path / "README.md",
        "[private]: data/private/report.md\n"
        "<a href=\"data/private/report.md\">private</a>\n"
        "<img src='docs/security/threat.md'>\n",
    )
    write(tmp_path / "data" / "private" / "report.md", "private\n")
    write(tmp_path / "docs" / "security" / "threat.md", "private\n")

    findings = validate_public_release(tmp_path, include_history=False)

    assert any(item.kind == "broken_link" and item.path == "README.md" for item in findings)


def test_validator_rejects_incomplete_public_binary_containers(tmp_path: Path):
    write_bytes(tmp_path / "demo" / "documents" / "dialysis-basics.pdf", b"%PDF-1.7\nnot complete")
    write_bytes(tmp_path / "demo" / "documents" / "learning-points.docx", docx_bytes(include_required_parts=False))
    write_bytes(tmp_path / "docs" / "assets" / "chat-answer.png", png_bytes()[:-4])
    write_bytes(tmp_path / "docs" / "assets" / "question-flow.gif", gif_bytes() + b"trailing")

    findings = validate_public_release(tmp_path, include_history=False)

    assert sum(item.kind == "blocked_file" for item in findings) >= 4


def test_validator_requires_complete_release_layout_when_markers_are_present(
    tmp_path: Path,
):
    write(tmp_path / "README.md", "public release\n")
    write(tmp_path / "backend" / "app" / "__init__.py", "\n")
    write(tmp_path / "frontend" / "package.json", "{}\n")
    write(tmp_path / "demo" / "questions.json", "{}\n")

    findings = validate_public_release(tmp_path, include_history=False)

    assert any(item.kind == "release_layout" for item in findings)


def test_staged_manifest_uses_staged_asset_bytes(tmp_path: Path):
    repository = initialise_git_repository(tmp_path)
    asset = repository / "docs" / "assets" / "chat-answer.png"
    valid = png_bytes()
    write_bytes(asset, valid)
    write(
        asset.parent / "manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "file": "chat-answer.png",
                        "scenario_id": "demo",
                        "sha256": __import__("hashlib").sha256(valid).hexdigest(),
                        "width": 1,
                        "height": 1,
                        "reviewed": True,
                    }
                ],
            }
        ),
    )
    git(repository, "add", "--", "docs/assets")
    git(repository, "commit", "-m", "valid asset")
    write_bytes(asset, b"invalid staged png")
    git(repository, "add", "--", "docs/assets/chat-answer.png")
    write_bytes(asset, valid)

    findings = validate_public_release(repository, include_history=False)

    assert any(item.kind == "asset_manifest" and item.path == "docs/assets/manifest.json" for item in findings)


def test_non_history_reports_unreadable_git_metadata(tmp_path: Path, monkeypatch):
    repository = initialise_git_repository(tmp_path)
    write(repository / "README.md", "safe\n")

    def unavailable(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr("scripts.validate_public_release.subprocess.run", unavailable)

    findings = validate_public_release(repository, include_history=False)

    assert findings == (
        Finding("unreadable", ".git", "Git repository cannot be inspected"),
    )
