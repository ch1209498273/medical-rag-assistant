"""Fail-closed, pre-commit-oriented repository safety scanner.

The scanner enumerates Git candidates first, rejects protected paths from
metadata alone, and only opens small text candidates that are explicitly
allowed.  It is intentionally not a recursive reader for the private source
documents directory.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path, PureWindowsPath

MAX_TEXT_BYTES = 1_000_000
BLOCKED_PATH_PARTS = frozenset(
    {
        "资料",
        "data/private",
        "data/processed",
        "data/qdrant",
        "data/evaluations",
        "logs",
    }
)
BLOCKED_EXTENSIONS = frozenset(
    {
        ".7z",
        ".bz2",
        ".db",
        ".doc",
        ".docx",
        ".gz",
        ".log",
        ".pdf",
        ".ppt",
        ".pptx",
        ".rar",
        ".sqlite",
        ".sqlite3",
        ".tar",
        ".tgz",
        ".xls",
        ".xlsx",
        ".zip",
    }
)
PUBLIC_DOCUMENT_ROOTS = ("samples/policies",)
SECRET_ASSIGNMENT = re.compile(
    r"(?ix)(?:api[_-]?key|authorization|access[_-]?token|client[_-]?secret|"
    r"password|passwd|private[_-]?key|secret(?:[_-]?key)?|signing[_-]?key|"
    r"aws[_-]?secret[_-]?access[_-]?key)\b"
    r"\s*[:=]\s*(?P<quote>['\"])?"
    r"(?P<value>[a-z0-9_./+=:-]{4,})"
    r"(?P=quote)?"
    r"(?=[\s,;}\"']|$)"
)
TOKEN_PATTERN = re.compile(r"(?i)\b(?:sk|sf|mm)-[a-z0-9]{16,}\b")
TOKEN_ASSIGNMENT = re.compile(
    r"(?ix)\btoken\b\s*[:=]\s*['\"]?(?P<value>[a-z0-9][a-z0-9_./+=-]{15,})"
    r'(?=(?:["\x27]|$|[\s,;}]))'
)
BEARER_PATTERN = re.compile(
    r"(?ix)\b(?:authorization\s*[:=]\s*)?bearer\s+(?P<value>[a-z0-9_./+=-]{16,})"
    r'(?=(?:["\x27]|$|[\s,;}]))'
)
PEM_PRIVATE_KEY_PATTERN = re.compile(
    r"(?i)-----BEGIN(?: [A-Z0-9][A-Z0-9 -]*)? PRIVATE KEY-----"
)
GITHUB_TOKEN_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,})(?![A-Za-z0-9_])"
)
AWS_ACCESS_KEY_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])")
JWT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
)
PROVIDER_TOKEN_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:AIza[A-Za-z0-9_-]{30,}|"
    r"xox[baprs]-[A-Za-z0-9-]{20,}|(?:hf|r8|npm|pypi|pat)_[A-Za-z0-9_-]{20,}|"
    r"(?:sk|sf|mm)-[A-Za-z0-9][A-Za-z0-9_-]{15,})(?![A-Za-z0-9_])"
)
DRIVE_PATH_PATTERN = re.compile(r"(?i)^[a-z]:[\\/]")
URL_PATTERN = re.compile(r"(?i)^(?:https?|file)://")
URL_LITERAL_PATTERN = re.compile(r"""(?i)(?:https?|file)://[^\s`"'(){}\[\],;]+""")
PROTOCOL_RELATIVE_URL_PATTERN = re.compile(
    r"""(?i)//(?=[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:[/:]|$))[^\s`"'(){}\[\],;]+"""
)
_PATH_TAIL = r"""[^\s`"'(){}\[\],;]+"""
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?<![\w.<])(?P<path>(?:"
    + r"[a-z]:[\\/]"
    + _PATH_TAIL
    + r"|\\\\\?[\\/][a-z]:[\\/]"
    + _PATH_TAIL
    + r"|\\\\[A-Za-z0-9_.~-]+[\\/]"
    + _PATH_TAIL
    + r"|//[A-Za-z0-9_.~-]+[\\/]"
    + _PATH_TAIL
    + r"|\\[A-Za-z0-9_.~-]+(?:[\\/]"
    + _PATH_TAIL
    + r")?"
    + r"|/[A-Za-z0-9_.~-]+(?:[\\/]"
    + _PATH_TAIL
    + r")?"
    + r"))"
)
ALLOWED_API_ROUTES = (
    "/api",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/chat",
    "/embeddings",
    "/rerank",
    "/assets",
)
FICTIONAL_PATH_CANARY = (
    "FICTIONAL_PATH_CANARY=C:" + "\\" + "fictional" + "\\" + "policy.docx"
)
_SAFE_CANARY_VALUES = frozenset(
    {
        "changeme",
        "dummy",
        "example",
        "fake",
        "fictional",
        "fixture",
        "local",
        "placeholder",
        "test",
        "your-key-here",
        "sk-cloud-canary-123456789",
        "sk-test-canary-123456789",
        "sf-test-canary-123456789",
        "mm-test-canary-123456789",
    }
)
_SAFE_CODE_REFERENCE_VALUES = frozenset(
    {
        "api_key",
        "credential",
        "credentials",
        "ds_key",
    }
)
ABSOLUTE_UNIX_ROOT_NAMES = frozenset(
    "/" + name
    for name in (
        "workspace",
        "bin",
        "proc",
        "dev",
        "etc",
        "home",
        "tmp",
        "var",
        "opt",
        "mnt",
        "usr",
        "root",
    )
)


def _git_candidates(root: Path) -> tuple[list[Path], list[Path], bool]:
    """Return index paths and worktree-only paths without reading file bytes."""

    indexed: list[Path] = []
    worktree: list[Path] = []
    seen_indexed: set[Path] = set()
    seen_worktree: set[Path] = set()
    commands = (
        ("index", ["git", "-C", str(root), "ls-files", "--cached", "-z"]),
        ("worktree", ["git", "-C", str(root), "diff", "--name-only", "-z"]),
        (
            "worktree",
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
        ),
    )
    succeeded = True
    for source, command in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            succeeded = False
            continue
        if result.returncode != 0:
            succeeded = False
            continue
        for item in result.stdout.split("\x00"):
            if not item:
                continue
            path = Path(item)
            if source == "index" and path not in seen_indexed:
                seen_indexed.add(path)
                indexed.append(path)
            if source == "worktree" and path not in seen_worktree:
                seen_worktree.add(path)
                worktree.append(path)
    return indexed, worktree, succeeded


def _relative_candidate(root: Path, candidate: Path) -> tuple[Path | None, Path | None]:
    """Resolve a candidate lexically, never following it for display purposes."""

    root_absolute = root.absolute()
    # ``Path.is_absolute`` follows the host platform's rules.  A repository
    # checked out on Linux can still receive a Windows-style candidate from a
    # cross-platform caller, so reject drive-rooted and UNC paths before
    # joining them to the local root.  Otherwise a Windows
    # path would be misclassified as a harmless relative filename.
    candidate_text = str(candidate)
    windows_candidate = PureWindowsPath(candidate_text)
    if windows_candidate.drive or windows_candidate.root:
        return None, None
    candidate_path = candidate if candidate.is_absolute() else root_absolute / candidate
    candidate_path = Path(os.path.normpath(str(candidate_path)))
    try:
        relative = candidate_path.relative_to(root_absolute)
    except ValueError:
        return None, None
    return candidate_path, relative


def _is_reparse_or_link(path: Path) -> bool:
    """Reject symbolic links and Windows reparse points before opening files."""

    if path.is_symlink():
        return True
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError:
        return False
    reparse_flag = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _is_blocked_path(relative: Path) -> bool:
    normalized = relative.as_posix().casefold()
    parts = {part.casefold() for part in relative.parts}
    name = relative.name.casefold()
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return True
    if any(
        normalized == part or normalized.startswith(part + "/")
        for part in BLOCKED_PATH_PARTS
    ):
        return True
    if any(part.casefold() == "data" and "private" in parts for part in relative.parts):
        return True
    return relative.suffix.casefold() in BLOCKED_EXTENSIONS and not _is_public_document(
        relative
    )


def _is_public_document(relative: Path) -> bool:
    normalized = relative.as_posix().casefold()
    return any(
        normalized == root or normalized.startswith(root + "/")
        for root in PUBLIC_DOCUMENT_ROOTS
    )


def _safe_label(relative: Path | None) -> str:
    return relative.as_posix() if relative is not None else "<outside-root>"


def _scan_text(text: str, label: str) -> str | None:
    for line in text.splitlines():
        if line.strip() == FICTIONAL_PATH_CANARY:
            continue
        if contains_secret(line):
            return f"possible secret: {label}"
        if _contains_absolute_path(line):
            return f"possible absolute path: {label}"
    return None


def _contains_absolute_path(line: str) -> bool:
    """Find rooted path substrings while excluding URL literals and routes."""

    masked = URL_LITERAL_PATTERN.sub(lambda match: " " * len(match.group(0)), line)
    masked = PROTOCOL_RELATIVE_URL_PATTERN.sub(
        lambda match: " " * len(match.group(0)), masked
    )
    masked = re.sub(
        r"""(?i)(["']integrity["']\s*:\s*["']sha\d+-)[^"']+""",
        lambda match: (
            match.group(1) + " " * (len(match.group(0)) - len(match.group(1)))
        ),
        masked,
    )
    masked = _mask_regex_literals(masked)
    for match in ABSOLUTE_PATH_PATTERN.finditer(masked):
        if match.start() and line[match.start() - 1] in ".\\":
            continue
        value = match.group("path")
        if value.startswith("\\") and re.search(
            r"\\(?:[.\[\]{}()*+^$|]|[bBdDsSwW]\b)", value
        ):
            continue
        if _looks_like_absolute_path(value):
            return True
    return False


def _is_safe_canary(value: str) -> bool:
    """Allow only the fixed, non-secret values used by local tests/examples."""

    normalized = value.strip("'\"` ,;:.").casefold()
    return normalized in _SAFE_CANARY_VALUES


def _is_code_reference(value: str) -> bool:
    """Recognise a bare variable reference, not an embedded credential literal."""

    return value.strip().casefold() in _SAFE_CODE_REFERENCE_VALUES


def _mask_regex_literals(text: str) -> str:
    """Mask only strings passed to a regex constructor, preserving real paths."""

    result = list(text)
    constructor = re.compile(
        r"(?ix)\bre\.(?:compile|search|match|fullmatch|findall|finditer|"
        r"split|sub|subn)\s*\(\s*(?:pattern\s*=\s*)?"
        r"(?:r|u|b|f|br|rb|fr|rf){0,2}(?P<quote>['\"])(?P<body>(?:\\.|(?!\1).)*)\1"
    )
    for match in constructor.finditer(text):
        body_start, body_end = match.span("body")
        for index in range(body_start, body_end):
            result[index] = " "
    return "".join(result)


def contains_secret(text: str) -> bool:
    """Return whether text contains a credential-like value.

    Explicitly labelled fictional/test values are allowed so public tests can
    exercise redaction without embedding a usable-looking credential.  This is
    value-based, not directory-based: tests receive no blanket exemption.
    """

    for line in text.splitlines():
        if PEM_PRIVATE_KEY_PATTERN.search(line):
            return True
        if GITHUB_TOKEN_PATTERN.search(line):
            return True
        if AWS_ACCESS_KEY_PATTERN.search(line):
            return True
        if JWT_PATTERN.search(line):
            return True
        provider = PROVIDER_TOKEN_PATTERN.search(line)
        if provider and not _is_safe_canary(provider.group(0)):
            return True
        for assignment in SECRET_ASSIGNMENT.finditer(line):
            value = assignment.group("value")
            is_unquoted_code_reference = (
                assignment.group("quote") is None and _is_code_reference(value)
            )
            if not (_is_safe_canary(value) or is_unquoted_code_reference):
                return True
        token = TOKEN_PATTERN.search(line)
        if token and not _is_safe_canary(token.group(0)):
            return True
        token_assignment = TOKEN_ASSIGNMENT.search(line)
        if token_assignment and not _is_safe_canary(token_assignment.group("value")):
            return True
        bearer = BEARER_PATTERN.search(line)
        if bearer and not _is_safe_canary(bearer.group("value")):
            return True
    return False


def contains_absolute_path(line: str) -> bool:
    """Expose the strict path matcher for the public-release validator."""

    return _contains_absolute_path(line)


def _looks_like_absolute_path(value: str) -> bool:
    if not value or URL_PATTERN.match(value):
        return False
    if DRIVE_PATH_PATTERN.match(value):
        return True
    if value.startswith("//"):
        return len(value) > 2 and "/" in value[2:]
    if value.startswith("\\"):
        if (
            len(value) >= 2
            and value[1] in "abdfnrtvwsSDWZBxuUN"
            and (len(value) == 2 or not value[2].isalpha())
        ):
            return False
        if value.startswith("\\\\") and len(value) > 2 and value[2] in "`\"'":
            return False
        if len(value) > 1 and value[1] in "`\"'":
            return False
        return len(value) > 2 and ("\\" in value[1:] or "/" in value[1:])
    if value.startswith("/"):
        if value == "/" or len(value) == 1 or value[1] in "`\"'()[]{}<>.:":
            return False
        if value.endswith("/") and any(marker in value for marker in "*[]()"):
            return False
        if not any(separator in value[1:] for separator in "/\\"):
            normalized = value.casefold()
            route_prefix = any(
                normalized.startswith(route.casefold()) for route in ALLOWED_API_ROUTES
            )
            if normalized not in ABSOLUTE_UNIX_ROOT_NAMES and not route_prefix:
                return False
        return not _is_allowed_api_route(value)
    return False


def _is_allowed_api_route(value: str) -> bool:
    """Allow only exact routes or routes followed by a URL/path boundary."""

    for route in ALLOWED_API_ROUTES:
        if value == route:
            return True
        if value.startswith(route) and value[len(route) : len(route) + 1] in {
            "/",
            "?",
            "#",
        }:
            return True
    return False


def _scan_index_candidate(root: Path, relative: Path, label: str) -> str | None:
    """Scan an indexed blob, failing closed if Git cannot provide its bytes."""

    try:
        prefix_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-prefix"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return f"unreadable candidate: {label}"
    if prefix_result.returncode != 0:
        return f"unreadable candidate: {label}"
    object_name = f":{prefix_result.stdout.strip()}{relative.as_posix()}"
    try:
        size_result = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-s", object_name],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return f"unreadable candidate: {label}"
    if size_result.returncode != 0:
        return f"unreadable candidate: {label}"
    try:
        size = int(size_result.stdout.strip())
    except ValueError:
        return f"unreadable candidate: {label}"
    if size > MAX_TEXT_BYTES:
        return f"oversized candidate: {label}"
    try:
        content_result = subprocess.run(
            ["git", "-C", str(root), "show", object_name],
            check=False,
            capture_output=True,
        )
    except OSError:
        return f"unreadable candidate: {label}"
    if content_result.returncode != 0:
        return f"unreadable candidate: {label}"
    return _scan_text(content_result.stdout.decode("utf-8", errors="ignore"), label)


def _scan_worktree_candidate(path: Path, label: str) -> str | None:
    """Scan one non-indexed file after link and size checks."""

    if _is_reparse_or_link(path):
        return f"blocked path: {label}"
    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
    except OSError:
        return f"unreadable candidate: {label}"
    if size > MAX_TEXT_BYTES:
        return f"oversized candidate: {label}"
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return f"unreadable candidate: {label}"
    return _scan_text(text, label)


def scan_repository(root: Path, candidates: list[Path] | None = None) -> list[str]:
    """Return safe relative-path findings for a repository candidate set."""

    root = Path(root)
    indexed_candidates: list[Path] = []
    worktree_candidates: list[Path]
    if candidates is None:
        indexed_candidates, worktree_candidates, git_ok = _git_candidates(root)
        if not git_ok:
            return ["git candidate enumeration failed"]
    else:
        worktree_candidates = candidates

    problems: list[str] = []
    candidate_sources = [(True, candidate) for candidate in indexed_candidates] + [
        (False, candidate) for candidate in worktree_candidates
    ]
    for from_index, candidate in candidate_sources:
        candidate = Path(candidate)
        path, relative = _relative_candidate(root, candidate)
        label = _safe_label(relative)
        if path is None or relative is None:
            problems.append(f"blocked path: {label}")
            continue
        if _is_blocked_path(relative):
            problems.append(f"blocked path: {label}")
            continue
        finding = (
            _scan_index_candidate(root, relative, label)
            if from_index
            else _scan_worktree_candidate(path, label)
        )
        if finding and finding not in problems:
            problems.append(finding)
    return problems


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    problems = scan_repository(root)
    if problems:
        print("\n".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
