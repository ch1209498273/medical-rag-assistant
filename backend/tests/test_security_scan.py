import subprocess
from pathlib import Path

import pytest

from scripts.security_scan import contains_secret, scan_repository


def test_scan_rejects_candidate_secrets_and_real_documents(tmp_path: Path):
    (tmp_path / ".env").write_text(
        "DEEPSEEK_" + "API_KEY=" + "sk-secret", encoding="utf-8"
    )
    private = tmp_path / "资料"
    private.mkdir()
    (private / "真实制度.pdf").write_bytes(b"%PDF")

    problems = scan_repository(tmp_path, [Path(".env"), Path("资料/真实制度.pdf")])

    assert any(".env" in item for item in problems)
    assert any("资料" in item for item in problems)


def test_scan_allows_invalid_example_placeholders(tmp_path: Path):
    example = tmp_path / ".env.example"
    example.write_text(
        "DEEPSEEK_API_KEY=your-key-here\nSILICONFLOW_API_KEY=your-key-here\n",
        encoding="utf-8",
    )

    assert scan_repository(tmp_path, [Path(".env.example")]) == []


def test_scan_detects_common_secret_assignment_in_allowed_text(tmp_path: Path):
    config = tmp_path / "config.txt"
    config.write_text("DEEPSEEK_API_KEY=" + "abcdefghijkl", encoding="utf-8")

    problems = scan_repository(tmp_path, [Path("config.txt")])

    assert problems == ["possible " + "sec" + "ret: config.txt"]


def test_secret_scan_checks_every_assignment_and_requires_unquoted_code_reference():
    unsafe = "DEEPSEEK_" + "API_KEY=" + "abcdefghijkl"
    mixed = "DEEPSEEK_API_KEY=api_key; " + unsafe
    quoted_reference = "DEEPSEEK_API_KEY=" + '"' + "api_key" + '"'

    assert contains_secret(mixed)
    assert contains_secret(quoted_reference)
    assert not contains_secret("DEEPSEEK_API_KEY=api_key")


def test_scan_allows_explicitly_marked_fictional_canary(tmp_path: Path):
    fixture = tmp_path / "test-fixture.txt"
    fixture.write_text(
        "SILICONFLOW_API_KEY=sf-test-canary-123456789\n",
        encoding="utf-8",
    )

    assert scan_repository(tmp_path, [Path("test-fixture.txt")]) == []


def test_scan_blocks_documents_archives_and_oversized_files_before_text_scan(tmp_path: Path):
    (tmp_path / "notes.pdf").write_bytes(b"not a real document")
    (tmp_path / "bundle.zip").write_bytes(b"PK")
    (tmp_path / "large.txt").write_bytes(b"x" * (1_000_001))

    problems = scan_repository(
        tmp_path,
        [Path("notes.pdf"), Path("bundle.zip"), Path("large.txt")],
    )

    assert any("notes.pdf" in item for item in problems)
    assert any("bundle.zip" in item for item in problems)
    assert any("large.txt" in item for item in problems)


def test_scan_reports_only_safe_relative_paths_for_outside_candidate(tmp_path: Path):
    outside = Path("C:" + "/" + "sensitive/enterprise-policy.txt")

    problems = scan_repository(tmp_path, [outside])

    assert problems
    assert all(str(tmp_path) not in item for item in problems)
    assert all("sensitive" not in item for item in problems)


def test_scan_rejects_relative_path_traversal_without_opening_outside_root(
    tmp_path: Path,
):
    problems = scan_repository(tmp_path, [Path("nested/../../outside.txt")])

    assert problems == ["blocked path: <outside-root>"]


def test_scan_blocks_absolute_machine_paths_in_tracked_text(tmp_path: Path):
    document = tmp_path / "test-strategy.md"
    document.write_text(
        "公开文档不应包含 " + "C:" + "\\" + "Users" + "\\" + "real-user" + "\\" + "资料" + "\\" + "policy.docx\n",
        encoding="utf-8",
    )

    problems = scan_repository(tmp_path, [Path("test-strategy.md")])

    assert problems == ["possible absolute path: test-strategy.md"]


def test_scan_allows_explicit_fictional_absolute_path_canary(tmp_path: Path):
    fixture = tmp_path / "tests" / "fixture.txt"
    fixture.parent.mkdir()
    fixture.write_text(
        "FICTIONAL_PATH_CANARY=" + "C" + ":" + chr(92) + "fictional" + chr(92) + "policy.docx\n",
        encoding="utf-8",
    )

    assert scan_repository(tmp_path, [Path("tests/fixture.txt")]) == []


@pytest.mark.parametrize(
    "value",
    [
        "/" + "tmp/real-user/secret",
        "/" + "var/lib/real-user/secret",
        "/" + "opt/real-user/secret",
        "/" + "mnt/real-user/secret",
        "/" + "etc/real-user/secret",
        chr(92) * 2 + "server" + chr(92) + "share" + chr(92) + "secret.docx",
        "\\" * 2 + "?" + "\\" + "C:" + "\\" + "secret.docx",
        "C:" + "/" + "Users" + "/" + "中文用户" + "/" + "secret.docx",
    ],
)
def test_scan_blocks_unix_windows_unc_and_device_absolute_paths(
    tmp_path: Path, value: str
):
    document = tmp_path / "docs.txt"
    document.write_text(f"path={value}\n", encoding="utf-8")

    problems = scan_repository(tmp_path, [Path("docs.txt")])

    assert problems == ["possible absolute path: docs.txt"]


def test_canary_does_not_exempt_another_real_path_in_same_file(tmp_path: Path):
    fixture = tmp_path / "tests" / "fixture.txt"
    fixture.parent.mkdir()
    fixture.write_text(
        "FICTIONAL_PATH_CANARY=" + "C" + ":" + chr(92) + "fictional" + chr(92) + "policy.docx\n"
        "real=/" + "home/real-user/secret\n",
        encoding="utf-8",
    )

    assert scan_repository(tmp_path, [Path("tests/fixture.txt")]) == [
        "possible absolute path: tests/fixture.txt"
    ]


def test_scan_does_not_treat_urls_as_absolute_paths(tmp_path: Path):
    document = tmp_path / "links.txt"
    document.write_text(
        "https://example.com/path and file://" + "/" + "tmp/not-a-path\n",
        encoding="utf-8",
    )

    assert scan_repository(tmp_path, [Path("links.txt")]) == []


@pytest.mark.parametrize(
    "value",
    [
        'Path("' + "/" + 'workspace/secret")',
        'r"' + "C:" + "\\" + 'Users\\u\\secret.docx"',
        'f"' + "C:" + "\\" + 'Users\\u\\secret.docx"',
        "prefix:" + "/" + "workspace/secret",
        'prefix=("' + "/" + 'workspace/secret")',
    ],
)
def test_scan_blocks_wrapped_absolute_path_literals(tmp_path: Path, value: str):
    document = tmp_path / "wrapped.txt"
    document.write_text(value + "\n", encoding="utf-8")

    assert scan_repository(tmp_path, [Path("wrapped.txt")]) == [
        "possible absolute path: wrapped.txt"
    ]


@pytest.mark.parametrize(
    "value",
    ["/" + "api-private/secret", "/" + "api_secret/secret", "/" + "api2"],
)
def test_scan_blocks_api_lookalike_prefixes(tmp_path: Path, value: str):
    document = tmp_path / "api-lookalike.txt"
    document.write_text(value + "\n", encoding="utf-8")

    assert scan_repository(tmp_path, [Path("api-lookalike.txt")]) == [
        "possible absolute path: api-lookalike.txt"
    ]


def test_scan_does_not_skip_unapproved_route_declarations(tmp_path: Path):
    document = tmp_path / "route-declaration.txt"
    document.write_text(
        'router.get("' + "/" + "api-private/secret" + '")\n',
        encoding="utf-8",
    )

    assert scan_repository(tmp_path, [Path("route-declaration.txt")]) == [
        "possible absolute path: route-declaration.txt"
    ]


@pytest.mark.parametrize(
    "value",
    [
        "/" + "api",
        "/" + "api/health",
        "/" + "api?stream=1",
        "/" + "api#fragment",
        "/" + "api/chat/stream",
    ],
)
def test_scan_allows_exact_api_routes_and_explicit_boundaries(
    tmp_path: Path, value: str
):
    document = tmp_path / "api-routes.txt"
    document.write_text(value + "\n", encoding="utf-8")

    assert scan_repository(tmp_path, [Path("api-routes.txt")]) == []


@pytest.mark.parametrize(
    "value",
    [
        "/" + "workspace/secret",
        "/" + "bin/secret",
        "/" + "proc/1/environ",
        "/" + "dev/null",
        "/" * 2 + "server/share/secret",
        "\\" + "Windows\\secret",
        "/" + "workspace\\secret",
    ],
)
def test_scan_blocks_unlisted_unix_rooted_and_forward_unc_paths(tmp_path: Path, value: str):
    document = tmp_path / "rooted.txt"
    document.write_text(f"path={value}\n", encoding="utf-8")

    assert scan_repository(tmp_path, [Path("rooted.txt")]) == [
        "possible absolute path: rooted.txt"
    ]


def test_scan_reads_staged_blob_when_worktree_content_diverges(tmp_path: Path):
    subprocess.run(
        ["git", "init", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    config = tmp_path / "config.txt"
    config.write_text("DEEPSEEK_API_KEY=" + "abcdefghijkl", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--", "config.txt"],
        check=True,
        capture_output=True,
        text=True,
    )
    config.write_text("DEEPSEEK_API_KEY=your-key-here", encoding="utf-8")

    problems = scan_repository(tmp_path)

    assert problems == ["possible " + "sec" + "ret: config.txt"]


def test_scan_reads_index_when_scan_root_is_a_git_subdirectory(tmp_path: Path):
    repository = tmp_path / "repository"
    project = repository / "project"
    project.mkdir(parents=True)
    subprocess.run(
        ["git", "init", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    config = project / "config.txt"
    config.write_text("DEEPSEEK_API_KEY=" + "abcdefghijkl", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repository), "add", "--", "project/config.txt"],
        check=True,
        capture_output=True,
        text=True,
    )

    problems = scan_repository(project)

    assert problems == ["possible " + "sec" + "ret: config.txt"]
