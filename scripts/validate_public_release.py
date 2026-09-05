"""Validate the files and Git history of a release candidate.

The validator is deliberately fail-closed.  It reports only repository-relative
paths and stable descriptions; file contents, credentials, and command output
are never included in findings.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import struct
import subprocess
import sys
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import unquote
from xml.etree import ElementTree

try:
    from scripts.security_scan import contains_absolute_path, contains_secret
except ModuleNotFoundError:  # Running this file directly from ``scripts``.
    from security_scan import contains_absolute_path, contains_secret


MAX_TEXT_BYTES = 1_000_000
MAX_BINARY_BYTES = 8 * 1024 * 1024
_REPARSE_POINT_ATTRIBUTE = 0x400
_PATH_SEPARATOR = re.compile(r"[\\/]+")
_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)>\s]+)>?")
_MARKDOWN_REFERENCE = re.compile(
    r"(?im)^[ \t]{0,3}\[[^\]\r\n]+\]:[ \t]*(?:<([^>\r\n]+)>|([^\s]+))"
)
_HTML_LINK_ATTRIBUTE = re.compile(
    r"(?is)\b(?:href|src)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_URL = re.compile(r"^(?:https?|file|mailto|data):", re.IGNORECASE)
_PROTOCOL_RELATIVE_URL = re.compile(
    r"^//[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:[/:]|$)", re.IGNORECASE
)
_ALLOWED_API_ROUTES = (
    "/api",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/chat",
    "/embeddings",
    "/rerank",
    "/assets",
)

_BLOCKED_COMPONENTS = frozenset(
    {
        ".superpowers",
        ".worktrees",
        "confidential",
        "internal",
        "private",
        "sdd",
        "资料",
    }
)
_BLOCKED_PREFIXES = (
    ("data", "private"),
    ("data", "processed"),
    ("data", "qdrant"),
    ("data", "evaluations"),
    ("data", "runtime"),
    ("docs", "project"),
    ("docs", "security"),
    ("docs", "testing"),
    ("docs", "superpowers"),
)
_BLOCKED_SUFFIXES = frozenset(
    {
        ".7z",
        ".arj",
        ".bz2",
        ".cab",
        ".db",
        ".doc",
        ".gz",
        ".lz",
        ".lz4",
        ".log",
        ".rar",
        ".sqlite",
        ".sqlite3",
        ".tar",
        ".tbz",
        ".tbz2",
        ".tgz",
        ".txz",
        ".xz",
        ".zip",
        ".zst",
    }
)
_CREDENTIAL_NAMES = frozenset(
    {".dockerconfigjson", ".netrc", ".npmrc", ".pypirc", "authorized_keys"}
)
_SSH_KEY_NAMES = frozenset({"id_dsa", "id_ecdsa", "id_ed25519", "id_rsa"})
_CREDENTIAL_MARKER = re.compile(
    r"(?:^|[._-])(?:access[_-]?key|access[_-]?token|api[_-]?key|"
    r"credential|credentials|passphrase|passwd|password|private[_-]?key|"
    r"secret|secrets|token|tokens)(?:$|[._-]|\d)",
    re.IGNORECASE,
)
_BLOCKED_BINARY_SUFFIXES = frozenset(
    {
        ".bin",
        ".dll",
        ".dylib",
        ".exe",
        ".iso",
        ".jar",
        ".msi",
        ".pdb",
        ".so",
    }
)
_SKIP_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "dist",
        "node_modules",
    }
)
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "aux",
        "clock$",
        "com1",
        "com2",
        "com3",
        "com4",
        "com5",
        "com6",
        "com7",
        "com8",
        "com9",
        "con",
        "conin$",
        "conout$",
        "lpt1",
        "lpt2",
        "lpt3",
        "lpt4",
        "lpt5",
        "lpt6",
        "lpt7",
        "lpt8",
        "lpt9",
        "nul",
        "prn",
    }
)
_PUBLIC_DOCUMENT_PREFIX = ("demo", "documents")
_PUBLIC_ASSET_PREFIX = ("docs", "assets")
_PUBLIC_ASSET_SUFFIXES = frozenset({".gif", ".png"})
_PUBLIC_DOCUMENT_SUFFIXES = frozenset({".docx", ".pdf"})
_PUBLIC_DEMO_STEMS = frozenset(
    {
        "dialysis-basics",
        "learning-points",
        "pre-dialysis-check",
        "training-policy",
    }
)
_REQUIRED_PUBLIC_ASSETS = frozenset(
    {
        "chat-answer.png",
        "chat-history.png",
        "chat-refusal.png",
        "documents.png",
        "runtime-mode.png",
        "question-flow.gif",
    }
)
_REQUIRED_PUBLIC_DOCUMENTS = frozenset(
    f"demo/documents/{stem}{suffix}"
    for stem in _PUBLIC_DEMO_STEMS
    for suffix in _PUBLIC_DOCUMENT_SUFFIXES
)
_INDEX_REGULAR_MODES = frozenset({"100644", "100755"})
_HISTORY_REGULAR_MODES = frozenset({"100644", "100755"})
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SAFE_SCENARIO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MACHINE_ABSOLUTE_PATH = re.compile(
    r"(?ix)(?<![A-Za-z0-9])(?:[a-z]:[\\/]|\\\\[A-Za-z0-9_.~-]+[\\/]|"
    r"/(?:users|home|workspace|var|tmp|opt|mnt|etc|root|private)/)"
)


@dataclass(frozen=True, order=True)
class Finding:
    """A deterministic, sanitized release validation finding."""

    kind: str
    path: str
    detail: str


class HistoryUnavailable(RuntimeError):
    """Raised when the requested Git history cannot be inspected safely."""


@dataclass(frozen=True)
class _GitScope:
    top: Path
    prefix: str


@dataclass(frozen=True)
class _IndexSnapshot:
    """The exact staged files visible through Git's index."""

    paths: frozenset[str]
    modes: dict[str, str]
    blobs: dict[str, bytes]


@dataclass(frozen=True)
class _HistoryRecord:
    """One file entry from one reachable commit tree."""

    commit: str
    path: str
    mode: str
    object_type: str
    object_id: str


def _safe_relative_text(value: str | Path) -> str:
    """Return a repository-relative display path without exposing host paths."""

    text = str(value).replace("\\", "/")
    if not text or text in {".", "<outside-root>"}:
        return text or "<root>"
    if _DRIVE_PATH.match(text) or text.startswith("/"):
        return "<outside-root>"
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        return "<outside-root>"
    return "/".join(parts) or "<root>"


def _finding(kind: str, path: str | Path, detail: str) -> Finding:
    return Finding(kind, _safe_relative_text(path), detail)


def _normalised_parts(relative: str | Path) -> tuple[str, ...]:
    text = str(relative).replace("\\", "/")
    parts: list[str] = []
    for raw in _PATH_SEPARATOR.split(text):
        if not raw or raw == ".":
            continue
        parts.append(raw.rstrip(" .").casefold())
    return tuple(parts)


def _is_blocked_path(relative: str | Path) -> bool:
    parts = _normalised_parts(relative)
    if not parts:
        return False
    if any(part in _BLOCKED_COMPONENTS for part in parts):
        return True
    if parts[0] == "release":
        return True
    name = parts[-1]
    if name == ".git":
        return True
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return True
    if "logs" in parts:
        return True
    if any(parts[: len(prefix)] == prefix for prefix in _BLOCKED_PREFIXES):
        return True
    if name.endswith(("-wal", "-shm", "-journal")):
        return True
    if name.endswith(".log") or ".log." in name:
        return True
    if (
        name in _CREDENTIAL_NAMES
        or _CREDENTIAL_MARKER.search(name) is not None
        or name.split(".", maxsplit=1)[0] in _SSH_KEY_NAMES
    ):
        return True
    return name.endswith(tuple(_BLOCKED_SUFFIXES))


def _is_public_document(relative: str | Path) -> bool:
    parts = _normalised_parts(relative)
    return len(parts) == 3 and parts[:2] == _PUBLIC_DOCUMENT_PREFIX


def _is_public_asset(relative: str | Path) -> bool:
    parts = _normalised_parts(relative)
    return len(parts) == 3 and parts[:2] == _PUBLIC_ASSET_PREFIX


def _blocked_file_detail(relative: str | Path) -> str | None:
    """Return a stable reason when a file type/location is not exportable."""

    parts = _normalised_parts(relative)
    name = parts[-1] if parts else ""
    native_name = PurePosixPath(name)
    suffix = native_name.suffix.casefold()
    if suffix in _BLOCKED_BINARY_SUFFIXES or suffix in _BLOCKED_SUFFIXES:
        return "file type is not allowed in a public release"
    if suffix in _PUBLIC_DOCUMENT_SUFFIXES and not _is_public_document(relative):
        return "document is outside the public demo directory"
    if suffix in _PUBLIC_ASSET_SUFFIXES and not _is_public_asset(relative):
        return "binary asset is outside the public asset directory"
    if _is_public_document(relative) and (
        suffix not in _PUBLIC_DOCUMENT_SUFFIXES
        or native_name.stem.casefold() not in _PUBLIC_DEMO_STEMS
    ):
        return "demo document is not an approved public asset"
    if (
        _is_public_asset(relative)
        and name != "manifest.json"
        and suffix not in _PUBLIC_ASSET_SUFFIXES
    ):
        return "asset type is not allowed"
    return None


def _is_link_or_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        try:
            return os.path.lexists(path)
        except OSError:
            return True
    except OSError:
        return True
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE)


def _has_link_or_reparse_ancestor(path: Path) -> bool:
    """Inspect every path component without resolving links."""

    absolute = path.absolute()
    parts = absolute.parts
    if not parts:
        return False
    current = Path(parts[0])
    if _is_link_or_reparse(current):
        return True
    for part in parts[1:]:
        current /= part
        if _is_link_or_reparse(current):
            return True
    return False


def _validate_root(root: Path) -> Path:
    try:
        candidate = Path(root)
    except TypeError as exc:
        raise ValueError("root must be a directory") from exc
    if _has_link_or_reparse_ancestor(candidate):
        raise ValueError("root must not be a symlink or reparse point")
    try:
        if not candidate.exists() or not candidate.is_dir():
            raise ValueError("root must be an existing directory")
        return candidate.resolve(strict=True)
    except ValueError:
        raise
    except (OSError, RuntimeError) as exc:
        raise ValueError("root cannot be resolved") from exc


def _run_git(scope_root: Path, args: list[str], *, history: bool = False) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(scope_root), "-c", "core.quotePath=false", *args],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        if history:
            raise HistoryUnavailable("Git history is unavailable") from exc
        raise
    if result.returncode != 0:
        if history:
            raise HistoryUnavailable("Git history is unavailable")
        raise subprocess.CalledProcessError(result.returncode, result.args)
    return result.stdout


def _run_git_input(
    scope_root: Path,
    args: list[str],
    input_data: bytes,
    *,
    history: bool = False,
) -> bytes:
    """Run Git with explicit input for batch object inspection."""

    try:
        result = subprocess.run(
            ["git", "-C", str(scope_root), "-c", "core.quotePath=false", *args],
            input=input_data,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        if history:
            raise HistoryUnavailable("Git history is unavailable") from exc
        raise
    if result.returncode != 0:
        if history:
            raise HistoryUnavailable("Git history is unavailable")
        raise subprocess.CalledProcessError(result.returncode, result.args)
    return result.stdout


def _git_scope(root: Path) -> _GitScope | None:
    try:
        if _is_link_or_reparse(root / ".git"):
            return None
    except OSError:
        return None
    try:
        raw = _run_git(root, ["rev-parse", "--show-toplevel"])
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        top = Path(raw.decode("utf-8", errors="strict").strip()).resolve(strict=True)
        relative = root.resolve(strict=True).relative_to(top)
    except (OSError, RuntimeError, UnicodeError, ValueError):
        return None
    prefix = "" if relative == Path(".") else relative.as_posix().strip("/") + "/"
    return _GitScope(top, prefix)


def _repo_path_for_scope(repo_path: str, scope: _GitScope) -> str | None:
    value = repo_path.replace("\\", "/")
    value = value.removeprefix("./")
    if scope.prefix:
        if not value.startswith(scope.prefix):
            return None
        value = value[len(scope.prefix) :]
    return value or None


def _git_paths(root: Path, scope: _GitScope, command: list[str]) -> tuple[str, ...]:
    raw = _run_git(root, command)
    result: list[str] = []
    for item in raw.decode("utf-8", errors="strict").split("\x00"):
        if not item:
            continue
        relative = _repo_path_for_scope(item, scope)
        if relative is not None:
            result.append(relative)
    return tuple(sorted(set(result)))


def _scan_text(
    text: str, path: str, findings: set[Finding], *, prefix: str = ""
) -> None:
    for line in text.splitlines():
        if contains_secret(line):
            findings.add(
                _finding(prefix + "secret", path, "possible credential or token")
            )
        if contains_absolute_path(line):
            findings.add(
                _finding(
                    prefix + "absolute_path", path, "machine-specific absolute path"
                )
            )


def _read_text_file(
    path: Path, relative: str, findings: set[Finding], *, prefix: str = ""
) -> str | None:
    try:
        size = path.stat().st_size
    except OSError:
        findings.add(
            _finding(prefix + "unreadable", relative, "file cannot be inspected")
        )
        return None
    if size > MAX_TEXT_BYTES:
        findings.add(
            _finding(
                prefix + "oversized", relative, "text file exceeds release size limit"
            )
        )
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError):
        return None
    _scan_text(text, relative, findings, prefix=prefix)
    return text


def _validate_png(raw: bytes) -> tuple[int, int] | None:
    """Validate a complete PNG stream and return its dimensions."""

    if len(raw) < len(_PNG_SIGNATURE) or raw[:8] != _PNG_SIGNATURE:
        return None
    offset = 8
    dimensions: tuple[int, int] | None = None
    saw_idat = False
    saw_iend = False
    idat_parts: list[bytes] = []
    known_chunks = {
        b"IHDR",
        b"PLTE",
        b"IDAT",
        b"IEND",
        b"tRNS",
        b"cHRM",
        b"gAMA",
        b"iCCP",
        b"sRGB",
        b"sBIT",
        b"pHYs",
        b"tEXt",
        b"zTXt",
        b"iTXt",
        b"bKGD",
        b"hIST",
        b"sPLT",
        b"tIME",
        b"eXIf",
        b"acTL",
        b"fcTL",
        b"fdAT",
    }
    while offset < len(raw):
        if offset + 12 > len(raw):
            return None
        length = int.from_bytes(raw[offset : offset + 4], "big")
        kind = raw[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(raw) or len(kind) != 4 or not all(
            (65 <= value <= 90) or (97 <= value <= 122) for value in kind
        ):
            return None
        data = raw[offset + 8 : offset + 8 + length]
        expected_crc = int.from_bytes(raw[offset + 8 + length : end], "big")
        if zlib.crc32(kind + data) & 0xFFFFFFFF != expected_crc:
            return None
        if kind == b"IHDR":
            if dimensions is not None or length != 13 or offset != 8:
                return None
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", data
            )
            valid_depths = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if (
                width < 1
                or height < 1
                or bit_depth not in valid_depths.get(color_type, set())
                or compression != 0
                or filtering != 0
                or interlace not in {0, 1}
            ):
                return None
            dimensions = (width, height)
        elif kind == b"IDAT":
            saw_idat = True
            idat_parts.append(data)
        elif kind == b"IEND":
            if length != 0 or dimensions is None or not saw_idat:
                return None
            saw_iend = True
            offset = end
            break
        elif kind not in known_chunks and kind[:1].isupper():
            return None
        offset = end
    if not saw_iend or offset != len(raw):
        return None
    try:
        decompressor = zlib.decompressobj()
        decompressor.decompress(b"".join(idat_parts))
        decompressor.flush()
        if (
            not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            return None
    except zlib.error:
        return None
    return dimensions


def _gif_sub_blocks(raw: bytes, offset: int) -> int | None:
    """Consume a GIF data-sub-block sequence and return the next offset."""

    while True:
        if offset >= len(raw):
            return None
        size = raw[offset]
        offset += 1
        if size == 0:
            return offset
        if offset + size > len(raw):
            return None
        offset += size


def _validate_gif(raw: bytes) -> tuple[int, int] | None:
    """Validate GIF structure through the trailer, rejecting trailing bytes."""

    if len(raw) < 13 or raw[:6] not in {b"GIF87a", b"GIF89a"}:
        return None
    width, height = struct.unpack_from("<HH", raw, 6)
    if width < 1 or height < 1:
        return None
    packed = raw[10]
    offset = 13
    has_color_table = bool(packed & 0x80)
    if has_color_table:
        table_size = 3 * (2 ** ((packed & 0x07) + 1))
        if offset + table_size > len(raw):
            return None
        offset += table_size
    image_count = 0
    while offset < len(raw):
        marker = raw[offset]
        offset += 1
        if marker == 0x3B:
            return (width, height) if image_count and offset == len(raw) else None
        if marker == 0x21:
            if offset >= len(raw):
                return None
            offset += 1
            next_offset = _gif_sub_blocks(raw, offset)
            if next_offset is None:
                return None
            offset = next_offset
            continue
        if marker != 0x2C or offset + 9 > len(raw):
            return None
        _left, _top, image_width, image_height, image_packed = struct.unpack_from(
            "<HHHHB", raw, offset
        )
        if (
            image_width < 1
            or image_height < 1
            or _left + image_width > width
            or _top + image_height > height
            or image_packed & 0x18
        ):
            return None
        offset += 9
        if image_packed & 0x80:
            has_color_table = True
            table_size = 3 * (2 ** ((image_packed & 0x07) + 1))
            if offset + table_size > len(raw):
                return None
            offset += table_size
        if not has_color_table:
            return None
        if offset >= len(raw):
            return None
        lzw_minimum_code_size = raw[offset]
        if not 2 <= lzw_minimum_code_size <= 8:
            return None
        offset += 1
        if offset >= len(raw) or raw[offset] == 0:
            return None
        next_offset = _gif_sub_blocks(raw, offset)
        if next_offset is None:
            return None
        offset = next_offset
        image_count += 1
    return None


def _validate_pdf(raw: bytes) -> bool:
    """Validate a PDF using a strict trailer check and the installed parser."""

    if re.match(rb"^%PDF-1\.[0-7](?:\r?\n|\r)", raw) is None:
        return False
    eof = raw.rfind(b"%%EOF")
    if eof < 0 or raw[eof + len(b"%%EOF") :].strip(b"\x00\t\r\n\f "):
        return False
    if b" obj" not in raw or b"trailer" not in raw:
        return False
    try:
        import pymupdf

        document = pymupdf.open(stream=raw, filetype="pdf")
        if getattr(document, "is_repaired", False):
            document.close()
            return False
        document.close()
    except (ImportError, OSError, RuntimeError, ValueError):
        return False
    return True


def _validate_docx(raw: bytes) -> tuple[bool, tuple[str, ...]]:
    """Validate every DOCX ZIP member and return UTF-8 textual parts."""

    required = {"[content_types].xml", "_rels/.rels", "word/document.xml"}
    names: list[str] = []
    text_parts: list[str] = []
    total_size = 0
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except (OSError, ValueError, zipfile.BadZipFile):
        return False, ()
    with archive:
        for member in archive.infolist():
            name = member.filename
            name_key = name.casefold()
            if (
                not name
                or "\\" in name
                or name.startswith("/")
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or any(part.rstrip(" .") != part for part in name.split("/"))
                or any(
                    part.split(".", maxsplit=1)[0].casefold()
                    in _WINDOWS_RESERVED_NAMES
                    for part in name.split("/")
                )
                or ":" in name
                or name_key in {item.casefold() for item in names}
                or (member.flag_bits & 0x1)
                or "vbaproject.bin" in name_key
            ):
                return False, ()
            names.append(name)
            total_size += member.file_size
            if total_size > MAX_BINARY_BYTES or member.file_size > MAX_TEXT_BYTES:
                return False, ()
            if member.is_dir():
                continue
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                return False, ()
            try:
                content = archive.read(member)
            except (OSError, KeyError, RuntimeError, zipfile.BadZipFile):
                return False, ()
            try:
                text_parts.append(content.decode("utf-8", errors="strict"))
            except UnicodeError:
                continue
        if archive.testzip() is not None:
            return False, ()
    if not required <= {name.casefold() for name in names}:
        return False, ()
    try:
        for required_name in required:
            # Match case-insensitively but parse the original member bytes.
            member_name = next(
                name for name in names if name.casefold() == required_name
            )
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                ElementTree.fromstring(archive.read(member_name))
    except (KeyError, OSError, UnicodeError, ElementTree.ParseError, zipfile.BadZipFile):
        return False, ()
    return True, tuple(text_parts)


def _scan_approved_binary(
    raw: bytes,
    suffix: str,
    relative: str,
    findings: set[Finding],
    *,
    prefix: str = "",
) -> None:
    """Validate and scan the complete approved public container."""

    valid = True
    text_parts: tuple[str, ...] = ()
    if suffix == ".png":
        valid = _validate_png(raw) is not None
    elif suffix == ".gif":
        valid = _validate_gif(raw) is not None
    elif suffix == ".pdf":
        valid = _validate_pdf(raw)
        text_parts = (raw.decode("latin-1"),)
    elif suffix == ".docx":
        valid, text_parts = _validate_docx(raw)
    if not valid:
        findings.add(
            _finding(prefix + "blocked_file", relative, "public binary container is invalid")
        )
        return
    for text in text_parts:
        if contains_secret(text):
            findings.add(
                _finding(prefix + "secret", relative, "possible credential or token")
            )
        if any(_MACHINE_ABSOLUTE_PATH.search(line) for line in text.splitlines()):
            findings.add(
                _finding(
                    prefix + "absolute_path", relative, "machine-specific absolute path"
                )
            )


def _classify_file(path: Path, relative: str, findings: set[Finding]) -> str | None:
    suffix = path.suffix.casefold()
    if _is_blocked_path(relative):
        findings.add(
            _finding("blocked_path", relative, "path is not public release content")
        )
    blocked_detail = _blocked_file_detail(relative)
    if blocked_detail is not None:
        findings.add(_finding("blocked_file", relative, blocked_detail))
        return None
    try:
        size = path.stat().st_size
    except OSError:
        findings.add(_finding("unreadable", relative, "file cannot be inspected"))
        return None
    size_limit = (
        MAX_BINARY_BYTES
        if suffix in _PUBLIC_ASSET_SUFFIXES | _PUBLIC_DOCUMENT_SUFFIXES
        else MAX_TEXT_BYTES
    )
    if size > size_limit:
        findings.add(_finding("oversized", relative, "file exceeds release size limit"))
        return None
    if suffix in _PUBLIC_DOCUMENT_SUFFIXES or suffix in _PUBLIC_ASSET_SUFFIXES:
        try:
            raw = path.read_bytes()
        except OSError:
            findings.add(_finding("unreadable", relative, "file cannot be inspected"))
            return None
        _scan_approved_binary(raw, suffix, relative, findings)
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        findings.add(_finding("unreadable", relative, "file cannot be inspected"))
        return None
    if b"\x00" in raw:
        findings.add(
            _finding("blocked_file", relative, "binary content is not allowed")
        )
        return None
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        findings.add(
            _finding("blocked_file", relative, "binary content is not allowed")
        )
        return None
    _scan_text(text, relative, findings)
    return text


def _iter_worktree(root: Path, findings: set[Finding]) -> tuple[tuple[Path, str], ...]:
    stack: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath())]
    markdown_files: list[tuple[Path, str]] = []
    while stack:
        directory, relative_directory = stack.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
        except OSError:
            relative = relative_directory.as_posix()
            findings.add(
                _finding("unreadable", relative, "directory cannot be inspected")
            )
            continue
        for entry in entries:
            relative_path = relative_directory / entry.name
            relative = relative_path.as_posix()
            if (
                relative_directory == PurePosixPath()
                and entry.name.casefold() == ".git"
            ):
                continue
            if _is_link_or_reparse(entry):
                findings.add(
                    _finding(
                        "blocked_path",
                        relative,
                        "symlinks and reparse points are not public content",
                    )
                )
                continue
            try:
                is_directory = entry.is_dir()
                is_file = entry.is_file()
            except OSError:
                findings.add(
                    _finding("unreadable", relative, "path cannot be inspected")
                )
                continue
            if is_directory:
                if entry.name.casefold() in _SKIP_DIRECTORY_NAMES:
                    continue
                if _is_blocked_path(relative):
                    findings.add(
                        _finding(
                            "blocked_path",
                            relative,
                            "directory is not public release content",
                        )
                    )
                    continue
                stack.append((entry, relative_path))
            elif is_file:
                _classify_file(entry, relative, findings)
                if entry.suffix.casefold() == ".md":
                    markdown_files.append((entry, relative))
            else:
                findings.add(
                    _finding("blocked_file", relative, "path is not a regular file")
                )
    return tuple(sorted(markdown_files, key=lambda item: item[1].casefold()))


def _scan_index(root: Path, scope: _GitScope, findings: set[Finding]) -> _IndexSnapshot:
    """Scan the index and return the exact staged file snapshot."""

    try:
        paths = _git_paths(root, scope, ["ls-files", "--cached", "--full-name", "-z"])
    except (OSError, UnicodeError, subprocess.CalledProcessError):
        findings.add(
            _finding("unreadable", ".git/index", "Git index cannot be inspected")
        )
        return _IndexSnapshot(frozenset(), {}, {})
    index_modes: dict[str, str] = {}
    metadata_valid = True
    try:
        staged = _run_git(
            root, ["ls-files", "--cached", "--stage", "--full-name", "-z"]
        )
        for record in staged.split(b"\x00"):
            if not record:
                continue
            header, separator, raw_path = record.partition(b"\t")
            fields = header.split()
            if not separator or not fields:
                metadata_valid = False
                continue
            try:
                mode = fields[0].decode("ascii", errors="strict")
                repo_path = raw_path.decode("utf-8", errors="strict")
            except UnicodeError:
                metadata_valid = False
                continue
            relative_path = _repo_path_for_scope(repo_path, scope)
            if relative_path is not None:
                index_modes[relative_path] = mode
    except (OSError, subprocess.CalledProcessError):
        metadata_valid = False
    if not metadata_valid:
        findings.add(
            _finding("unreadable", ".git/index", "Git index metadata cannot be inspected")
        )

    available_paths = set(paths)
    blocked_index_paths = {
        relative
        for relative, mode in index_modes.items()
        if mode not in _INDEX_REGULAR_MODES
    }
    blobs: dict[str, bytes] = {}
    for relative in paths:
        index_mode = index_modes.get(relative)
        if index_mode is None:
            findings.add(
                _finding("unreadable", relative, "tracked entry metadata is unavailable")
            )
            continue
        if index_mode == "120000":
            findings.add(
                _finding(
                    "blocked_path",
                    relative,
                    "symlinks and reparse points are not public content",
                )
            )
            continue
        if index_mode not in _INDEX_REGULAR_MODES:
            findings.add(
                _finding(
                    "blocked_path",
                    relative,
                    "tracked entry is not a regular file",
                )
            )
            continue
        if _is_blocked_path(relative):
            findings.add(
                _finding("blocked_path", relative, "path is not public release content")
            )
        blocked_detail = _blocked_file_detail(relative)
        if blocked_detail is not None:
            findings.add(_finding("blocked_file", relative, blocked_detail))
        repo_path = f"{scope.prefix}{relative}" if scope.prefix else relative
        try:
            size_raw = _run_git(root, ["cat-file", "-s", f":{repo_path}"])
            size = int(size_raw.decode("ascii").strip())
        except (OSError, ValueError, subprocess.CalledProcessError):
            findings.add(
                _finding("unreadable", relative, "tracked file cannot be inspected")
            )
            continue
        suffix = PurePosixPath(relative.replace("\\", "/")).suffix.casefold()
        if size > (
            MAX_BINARY_BYTES
            if suffix in _PUBLIC_ASSET_SUFFIXES | _PUBLIC_DOCUMENT_SUFFIXES
            else MAX_TEXT_BYTES
        ):
            findings.add(
                _finding(
                    "oversized", relative, "tracked file exceeds release size limit"
                )
            )
            continue
        try:
            raw = _run_git(root, ["show", f":{repo_path}"])
        except (OSError, subprocess.CalledProcessError):
            findings.add(
                _finding("unreadable", relative, "tracked file cannot be inspected")
            )
            continue
        blobs[relative] = raw
        if blocked_detail is not None:
            continue
        if suffix in _PUBLIC_ASSET_SUFFIXES or suffix in _PUBLIC_DOCUMENT_SUFFIXES:
            _scan_approved_binary(raw, suffix, relative, findings)
            continue
        if b"\x00" in raw:
            findings.add(
                _finding("blocked_file", relative, "binary content is not allowed")
            )
            continue
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeError:
            findings.add(
                _finding("blocked_file", relative, "binary content is not allowed")
            )
            continue
        _scan_text(text, relative, findings)
        if (
            suffix == ".md"
            and not _is_blocked_path(relative)
            and blocked_detail is None
        ):
            _scan_markdown_links(
                root,
                relative,
                text,
                findings,
                available_paths=available_paths,
                blocked_paths=blocked_index_paths,
            )
    return _IndexSnapshot(frozenset(paths), index_modes, blobs)


def _scan_markdown_links(
    root: Path,
    relative: str,
    text: str,
    findings: set[Finding],
    *,
    available_paths: set[str] | None = None,
    blocked_paths: set[str] | None = None,
) -> None:
    targets: list[str] = [match.group(1) for match in _MARKDOWN_LINK.finditer(text)]
    targets.extend(
        first or second
        for first, second in _MARKDOWN_REFERENCE.findall(text)
        if first or second
    )
    targets.extend(
        first or second or third
        for first, second, third in _HTML_LINK_ATTRIBUTE.findall(text)
        if first or second or third
    )
    for raw_target in targets:
        target = unquote(raw_target.strip())
        if not target or target.startswith("#") or _URL.match(target):
            continue
        target_path, _, _fragment = target.partition("#")
        target_path = target_path.replace("\\", "/")
        if not target_path:
            continue
        if _PROTOCOL_RELATIVE_URL.match(target_path):
            continue
        if _DRIVE_PATH.match(target_path) or target_path.startswith(("/", "\\")):
            if any(
                target_path == route
                or target_path.startswith((route + "/", route + "?", route + "#"))
                for route in _ALLOWED_API_ROUTES
            ):
                continue
            findings.add(
                _finding(
                    "absolute_path", relative, "link target is not repository-relative"
                )
            )
            continue
        parts: list[str] = list(PurePosixPath(relative).parent.parts)
        escaped = False
        for part in PurePosixPath(target_path).parts:
            if part in {"", "."}:
                continue
            if part == "..":
                if not parts:
                    escaped = True
                    break
                parts.pop()
            else:
                parts.append(part)
        if escaped:
            findings.add(
                _finding(
                    "broken_link", relative, "relative link escapes the release root"
                )
            )
            continue
        candidate_relative = "/".join(parts)
        if _is_blocked_path(candidate_relative):
            findings.add(
                _finding(
                    "broken_link",
                    relative,
                    "relative link targets protected release content",
                )
            )
            continue
        candidate = root.joinpath(*PurePosixPath(candidate_relative).parts)
        if blocked_paths is not None and candidate_relative in blocked_paths:
            findings.add(
                _finding(
                    "broken_link", relative, "relative link targets a blocked path"
                )
            )
            continue
        if _has_link_or_reparse_ancestor(candidate):
            findings.add(
                _finding(
                    "broken_link", relative, "relative link targets a blocked path"
                )
            )
            continue
        if available_paths is not None and candidate_relative in available_paths:
            continue
        try:
            if not candidate.exists():
                findings.add(
                    _finding(
                        "broken_link", relative, "relative link target does not exist"
                    )
                )
        except OSError:
            findings.add(
                _finding(
                    "broken_link", relative, "relative link target cannot be inspected"
                )
            )


def _image_dimensions(raw: bytes, suffix: str) -> tuple[int, int] | None:
    if suffix == ".png":
        return _validate_png(raw)
    if suffix == ".gif":
        return _validate_gif(raw)
    return None


def _scan_asset_manifest(
    root: Path,
    findings: set[Finding],
    *,
    snapshot: _IndexSnapshot | None = None,
) -> None:
    """Validate the manifest against one coherent worktree or index snapshot."""

    manifest_relative = "docs/assets/manifest.json"
    asset_relative_prefix = "docs/assets/"
    asset_root = root / "docs" / "assets"
    asset_files: set[str] = set()
    asset_bytes: dict[str, bytes] = {}
    manifest_raw: bytes | None = None

    if snapshot is not None:
        snapshot_asset_paths = {
            path
            for path in snapshot.paths
            if path.casefold().startswith(asset_relative_prefix.casefold())
            and path.casefold() != manifest_relative.casefold()
        }
        for path in snapshot_asset_paths:
            parts = path.split("/")
            if len(parts) != 3:
                findings.add(
                    _finding(
                        "asset_manifest",
                        path,
                        "asset paths must be direct children of docs/assets",
                    )
                )
                continue
            name = parts[-1]
            asset_files.add(name)
            if path in snapshot.blobs:
                asset_bytes[name] = snapshot.blobs[path]
        manifest_mode = snapshot.modes.get(manifest_relative)
        if manifest_mode is not None and manifest_mode not in _INDEX_REGULAR_MODES:
            findings.add(
                _finding(
                    "blocked_path",
                    manifest_relative,
                    "tracked entry is not a regular file",
                )
            )
            return
        manifest_raw = snapshot.blobs.get(manifest_relative)
        if manifest_raw is None and not asset_files:
            return
    else:
        if _has_link_or_reparse_ancestor(asset_root):
            findings.add(
                _finding(
                    "blocked_path",
                    "docs/assets",
                    "symlinks and reparse points are not public content",
                )
            )
            return
        try:
            if not asset_root.exists():
                return
            entries = sorted(asset_root.iterdir(), key=lambda item: item.name.casefold())
        except OSError:
            findings.add(
                _finding(
                    "asset_manifest", "docs/assets", "asset directory cannot be inspected"
                )
            )
            return
        manifest_path = asset_root / "manifest.json"
        if _is_link_or_reparse(manifest_path):
            findings.add(
                _finding(
                    "blocked_path",
                    manifest_relative,
                    "symlinks and reparse points are not public content",
                )
            )
            return
        for path in entries:
            if _is_link_or_reparse(path):
                findings.add(
                    _finding(
                        "blocked_path",
                        f"docs/assets/{path.name}",
                        "symlinks and reparse points are not public content",
                    )
                )
                continue
            try:
                if path.is_dir():
                    findings.add(
                        _finding(
                            "asset_manifest",
                            f"docs/assets/{path.name}",
                            "asset paths must be direct children of docs/assets",
                        )
                    )
                    continue
                if not path.is_file():
                    findings.add(
                        _finding(
                            "asset_manifest",
                            f"docs/assets/{path.name}",
                            "asset entry cannot be inspected",
                        )
                    )
                    continue
            except OSError:
                findings.add(
                    _finding(
                        "asset_manifest",
                        f"docs/assets/{path.name}",
                        "asset entry cannot be inspected",
                    )
                )
                continue
            if path.name == "manifest.json":
                try:
                    if path.stat().st_size > MAX_TEXT_BYTES:
                        findings.add(
                            _finding(
                                "asset_manifest",
                                manifest_relative,
                                "asset manifest exceeds release size limit",
                            )
                        )
                        return
                    manifest_raw = path.read_bytes()
                except OSError:
                    findings.add(
                        _finding(
                            "unreadable", manifest_relative, "file cannot be inspected"
                        )
                    )
                    return
                continue
            asset_files.add(path.name)
            try:
                if path.stat().st_size > MAX_BINARY_BYTES:
                    findings.add(
                        _finding(
                            "asset_manifest",
                            manifest_relative,
                            "asset entry exceeds release size limit",
                        )
                    )
                    continue
                asset_bytes[path.name] = path.read_bytes()
            except OSError:
                findings.add(
                    _finding(
                        "asset_manifest",
                        manifest_relative,
                        "asset entry cannot be inspected",
                    )
                )
        if manifest_raw is None and not asset_files:
            return
    if manifest_raw is None:
        findings.add(
            _finding("asset_manifest", "docs/assets", "asset manifest is missing")
        )
        return
    try:
        payload = json.loads(manifest_raw.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        findings.add(
            _finding(
                "asset_manifest",
                "docs/assets/manifest.json",
                "asset manifest is not valid JSON",
            )
        )
        return
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("assets"), list)
    ):
        findings.add(
            _finding(
                "asset_manifest",
                "docs/assets/manifest.json",
                "asset manifest schema is invalid",
            )
        )
        return
    listed: set[str] = set()
    listed_keys: set[str] = set()
    for entry in payload["assets"]:
        if not isinstance(entry, dict) or set(entry) != {
            "file",
            "scenario_id",
            "sha256",
            "width",
            "height",
            "reviewed",
        }:
            findings.add(
                _finding(
                    "asset_manifest",
                    "docs/assets/manifest.json",
                    "asset entry schema is invalid",
                )
            )
            continue
        scenario_id = entry["scenario_id"]
        if (
            not isinstance(scenario_id, str)
            or _SAFE_SCENARIO_ID.fullmatch(scenario_id) is None
        ):
            findings.add(
                _finding(
                    "asset_manifest",
                    "docs/assets/manifest.json",
                    "asset entry schema is invalid",
                )
            )
            continue
        file_name = entry["file"]
        if (
            not isinstance(file_name, str)
            or not file_name.strip()
            or "/" in file_name
            or "\\" in file_name
            or PureWindowsPath(file_name).drive
            or PurePosixPath(file_name).name != file_name
            or file_name in {".", ".."}
            or file_name.rstrip(" .") != file_name
            or any(ord(character) < 32 for character in file_name)
            or ":" in file_name
            or file_name.split(".", maxsplit=1)[0].casefold() in _WINDOWS_RESERVED_NAMES
            or file_name in listed
            or file_name.casefold() in listed_keys
        ):
            findings.add(
                _finding(
                    "asset_manifest",
                    "docs/assets/manifest.json",
                    "asset file names are invalid or duplicated",
                )
            )
            continue
        listed.add(file_name)
        listed_keys.add(file_name.casefold())
        if PurePosixPath(file_name).suffix.casefold() not in _PUBLIC_ASSET_SUFFIXES:
            findings.add(
                _finding(
                    "asset_manifest",
                    "docs/assets/manifest.json",
                    "asset entry type is invalid",
                )
            )
            continue
        if file_name not in asset_files or file_name not in asset_bytes:
            findings.add(
                _finding(
                    "asset_manifest",
                    "docs/assets/manifest.json",
                    "manifest references a missing asset",
                )
            )
            continue
        raw = asset_bytes[file_name]
        digest = entry["sha256"]
        if (
            not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or hashlib.sha256(raw).hexdigest() != digest
        ):
            findings.add(
                _finding(
                    "asset_manifest",
                    "docs/assets/manifest.json",
                    "asset hash does not match",
                )
            )
        width = entry["width"]
        height = entry["height"]
        if type(width) is not int or type(height) is not int or width < 1 or height < 1:
            findings.add(
                _finding(
                    "asset_manifest",
                    "docs/assets/manifest.json",
                    "asset dimensions are invalid",
                )
            )
        suffix = PurePosixPath(file_name).suffix.casefold()
        actual = _validate_png(raw) if suffix == ".png" else _validate_gif(raw)
        if actual is None or actual != (width, height):
            findings.add(
                _finding(
                    "asset_manifest",
                    "docs/assets/manifest.json",
                    "asset dimensions do not match",
                )
            )
        if (
            entry["reviewed"] is not True
            or suffix not in _PUBLIC_ASSET_SUFFIXES
        ):
            findings.add(
                _finding(
                    "asset_manifest",
                    "docs/assets/manifest.json",
                    "asset review or type is invalid",
                )
            )
    asset_keys = {name.casefold() for name in asset_files}
    if len(asset_keys) != len(asset_files) or listed_keys != asset_keys:
        findings.add(
            _finding(
                "asset_manifest",
                "docs/assets/manifest.json",
                "manifest membership does not match assets",
            )
        )


def _validate_release_layout(
    root: Path, findings: set[Finding], *, snapshot: _IndexSnapshot | None = None
) -> None:
    """Require the documented public assets once a full release is present."""

    if snapshot is not None:
        available = {path.casefold() for path in snapshot.paths}
        has_markers = (
            "readme.md" in available
            and any(path.startswith("backend/") for path in available)
            and any(path.startswith("frontend/") for path in available)
            and any(path.startswith("demo/") for path in available)
        )
    else:
        marker_paths = (root / "README.md", root / "backend", root / "frontend", root / "demo")
        try:
            has_markers = all(
                path.is_file() if path.name == "README.md" else path.is_dir()
                for path in marker_paths
            ) and not any(_has_link_or_reparse_ancestor(path) for path in marker_paths)
        except OSError:
            has_markers = False
    if not has_markers:
        return

    if snapshot is not None:
        available = {path.casefold() for path in snapshot.paths}
        for required in sorted(_REQUIRED_PUBLIC_DOCUMENTS):
            if required.casefold() not in available:
                findings.add(
                    _finding(
                        "release_layout",
                        required,
                        "required public demo document is missing",
                    )
                )
        for required in sorted(_REQUIRED_PUBLIC_ASSETS):
            path = f"docs/assets/{required}"
            if path.casefold() not in available:
                findings.add(
                    _finding("release_layout", path, "required public visual asset is missing")
                )
        if "docs/assets/manifest.json" not in available:
            findings.add(
                _finding(
                    "release_layout",
                    "docs/assets/manifest.json",
                    "required public asset manifest is missing",
                )
            )
        return

    for required in sorted(_REQUIRED_PUBLIC_DOCUMENTS):
        path = root.joinpath(*PurePosixPath(required).parts)
        if _has_link_or_reparse_ancestor(path) or not path.is_file():
            findings.add(
                _finding(
                    "release_layout", required, "required public demo document is missing"
                )
            )
    for required in sorted(_REQUIRED_PUBLIC_ASSETS):
        relative = f"docs/assets/{required}"
        path = root.joinpath(*PurePosixPath(relative).parts)
        if _has_link_or_reparse_ancestor(path) or not path.is_file():
            findings.add(
                _finding("release_layout", relative, "required public visual asset is missing")
            )
    manifest = root / "docs" / "assets" / "manifest.json"
    if _has_link_or_reparse_ancestor(manifest) or not manifest.is_file():
        findings.add(
            _finding(
                "release_layout",
                "docs/assets/manifest.json",
                "required public asset manifest is missing",
            )
        )


def _history_tree_records(
    root: Path, scope: _GitScope, commits: tuple[str, ...]
) -> tuple[_HistoryRecord, ...]:
    records: list[_HistoryRecord] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for commit in commits:
        raw = _run_git(
            root,
            ["ls-tree", "-r", "-z", "--full-tree", commit],
            history=True,
        )
        for record in raw.split(b"\x00"):
            if not record:
                continue
            header, separator, raw_path = record.partition(b"\t")
            fields = header.split()
            if not separator or len(fields) != 3:
                raise HistoryUnavailable("Git history is unavailable")
            try:
                mode = fields[0].decode("ascii", errors="strict")
                object_type = fields[1].decode("ascii", errors="strict")
                object_id = fields[2].decode("ascii", errors="strict")
                repo_path = raw_path.decode("utf-8", errors="strict")
            except UnicodeError:
                raise HistoryUnavailable("Git history is unavailable")
            if re.fullmatch(r"[0-9a-f]{40,64}", object_id) is None:
                raise HistoryUnavailable("Git history is unavailable")
            relative = _repo_path_for_scope(repo_path, scope)
            if relative is None:
                continue
            key = (commit, relative, mode, object_type, object_id)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                _HistoryRecord(commit, relative, mode, object_type, object_id)
            )
    return tuple(records)


def _scan_history(root: Path, scope: _GitScope, findings: set[Finding]) -> None:
    shallow = _run_git(root, ["rev-parse", "--is-shallow-repository"], history=True)
    if shallow.decode("ascii", errors="strict").strip().casefold() != "false":
        raise HistoryUnavailable("Git history is unavailable")
    raw_commits = _run_git(root, ["rev-list", "--all"], history=True)
    try:
        commits = tuple(
            sorted(
                {
                    line.decode("ascii", errors="strict").strip()
                    for line in raw_commits.splitlines()
                    if line.strip()
                }
            )
        )
    except UnicodeError:
        raise HistoryUnavailable("Git history is unavailable")
    if any(re.fullmatch(r"[0-9a-f]{40,64}", commit) is None for commit in commits):
        raise HistoryUnavailable("Git history is unavailable")
    if not commits:
        return
    try:
        head_paths = set(
            _git_paths(root, scope, ["ls-tree", "-r", "--name-only", "-z", "HEAD"])
        )
    except (OSError, UnicodeError, subprocess.CalledProcessError):
        raise HistoryUnavailable("Git history is unavailable")
    records = _history_tree_records(root, scope, commits)
    object_ids = sorted({record.object_id for record in records})
    if not object_ids:
        return
    metadata_raw = _run_git_input(
        root,
        ["cat-file", "--batch-check"],
        ("\n".join(object_ids) + "\n").encode("ascii"),
        history=True,
    )
    metadata: dict[str, tuple[str, int]] = {}
    for line in metadata_raw.splitlines():
        fields = line.split()
        if len(fields) != 3:
            raise HistoryUnavailable("Git history is unavailable")
        try:
            object_id = fields[0].decode("ascii", errors="strict")
            object_type = fields[1].decode("ascii", errors="strict")
            size = int(fields[2].decode("ascii", errors="strict"))
        except (UnicodeError, ValueError):
            raise HistoryUnavailable("Git history is unavailable")
        if object_id not in object_ids or size < 0:
            raise HistoryUnavailable("Git history is unavailable")
        metadata[object_id] = (object_type, size)
    if set(metadata) != set(object_ids):
        raise HistoryUnavailable("Git history is unavailable")

    blob_ids: list[str] = []
    for record in records:
        object_type, size = metadata[record.object_id]
        blocked = _is_blocked_path(record.path) or _blocked_file_detail(record.path)
        if blocked and record.path not in head_paths:
            findings.add(
                _finding(
                    "history_blocked_path",
                    record.path,
                    "protected content is absent from HEAD but remains in history",
                )
            )
        if record.mode == "120000":
            findings.add(
                _finding(
                    "history_symlink",
                    record.path,
                    "historical tree contains a symlink entry",
                )
            )
        elif record.mode not in _HISTORY_REGULAR_MODES:
            findings.add(
                _finding(
                    "history_nonregular",
                    record.path,
                    "historical tree contains a non-regular entry",
                )
            )
        if object_type != "blob":
            if record.mode in _HISTORY_REGULAR_MODES:
                findings.add(
                    _finding(
                        "history_blocked_file",
                        record.path,
                        "historical regular entry cannot be inspected as a blob",
                    )
                )
            continue
        suffix = PurePosixPath(record.path.replace("\\", "/")).suffix.casefold()
        limit = (
            MAX_BINARY_BYTES
            if suffix in _PUBLIC_ASSET_SUFFIXES | _PUBLIC_DOCUMENT_SUFFIXES
            else MAX_TEXT_BYTES
        )
        if size > limit:
            findings.add(
                _finding(
                    "history_oversized",
                    record.path,
                    "historical blob exceeds release size limit",
                )
            )
            continue
        blob_ids.append(record.object_id)
    if not blob_ids:
        return

    batch_raw = _run_git_input(
        root,
        ["cat-file", "--batch"],
        ("\n".join(sorted(set(blob_ids))) + "\n").encode("ascii"),
        history=True,
    )
    blobs: dict[str, bytes] = {}
    offset = 0
    while offset < len(batch_raw):
        header_end = batch_raw.find(b"\n", offset)
        if header_end < 0:
            raise HistoryUnavailable("Git history is unavailable")
        fields = batch_raw[offset:header_end].split()
        offset = header_end + 1
        if len(fields) != 3:
            raise HistoryUnavailable("Git history is unavailable")
        try:
            object_id = fields[0].decode("ascii", errors="strict")
            object_type = fields[1].decode("ascii", errors="strict")
            size = int(fields[2].decode("ascii", errors="strict"))
        except (UnicodeError, ValueError):
            raise HistoryUnavailable("Git history is unavailable")
        if object_type != "blob" or size < 0 or offset + size > len(batch_raw):
            raise HistoryUnavailable("Git history is unavailable")
        blobs[object_id] = batch_raw[offset : offset + size]
        offset += size
        if batch_raw[offset : offset + 1] != b"\n":
            raise HistoryUnavailable("Git history is unavailable")
        offset += 1
    if set(blobs) != set(blob_ids):
        raise HistoryUnavailable("Git history is unavailable")

    scanned_content: set[tuple[str, str]] = set()
    for record in records:
        blob = blobs.get(record.object_id)
        if blob is None:
            continue
        content_key = (record.object_id, record.path)
        if content_key in scanned_content:
            continue
        scanned_content.add(content_key)
        suffix = PurePosixPath(record.path.replace("\\", "/")).suffix.casefold()
        if suffix in _PUBLIC_ASSET_SUFFIXES | _PUBLIC_DOCUMENT_SUFFIXES:
            if _is_public_asset(record.path) or _is_public_document(record.path):
                _scan_approved_binary(
                    blob,
                    suffix,
                    record.path,
                    findings,
                    prefix="history_",
                )
            continue
        if b"\x00" in blob:
            findings.add(
                _finding(
                    "history_blocked_file",
                    record.path,
                    "historical binary content cannot be inspected as public text",
                )
            )
            continue
        try:
            text = blob.decode("utf-8", errors="strict")
        except UnicodeError:
            findings.add(
                _finding(
                    "history_blocked_file",
                    record.path,
                    "historical binary content cannot be inspected as public text",
                )
            )
            continue
        _scan_text(text, record.path, findings, prefix="history_")


def validate_public_release(
    root: Path, *, include_history: bool
) -> tuple[Finding, ...]:
    """Return deterministic public-release findings for *root*.

    ``include_history`` performs a strict, checkout-free scan of all reachable
    Git objects.  A missing or unusable Git repository is an invocation error in
    that mode and is reported by the CLI with exit status 2.
    """

    validated_root = _validate_root(root)
    findings: set[Finding] = set()
    markdown_files = _iter_worktree(validated_root, findings)
    scope = _git_scope(validated_root)
    index_snapshot: _IndexSnapshot | None = None
    if scope is not None:
        index_snapshot = _scan_index(validated_root, scope, findings)
        _validate_release_layout(validated_root, findings, snapshot=index_snapshot)
    else:
        try:
            git_metadata = validated_root / ".git"
            git_present = os.path.lexists(git_metadata)
        except OSError:
            git_present = True
        if git_present:
            findings.add(
                _finding("unreadable", ".git", "Git repository cannot be inspected")
            )
        if include_history:
            raise HistoryUnavailable("Git history is unavailable")
    if include_history:
        assert scope is not None
        _scan_history(validated_root, scope, findings)
    for path, relative in markdown_files:
        if _is_blocked_path(relative) or _is_link_or_reparse(path):
            continue
        text = _read_text_file(path, relative, findings)
        if text is not None:
            _scan_markdown_links(validated_root, relative, text, findings)
    if index_snapshot is not None:
        _scan_asset_manifest(validated_root, findings, snapshot=index_snapshot)
    _scan_asset_manifest(validated_root, findings)
    _validate_release_layout(validated_root, findings)
    return tuple(sorted(findings))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a public release candidate")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="release root (default: current directory)",
    )
    parser.add_argument(
        "--history", action="store_true", help="scan all reachable Git history"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        findings = validate_public_release(args.root, include_history=args.history)
    except (HistoryUnavailable, ValueError, OSError):
        print(
            "validation unavailable: root or Git history could not be inspected",
            file=sys.stderr,
        )
        return 2
    for finding in findings:
        print(f"{finding.kind}: {finding.path}: {finding.detail}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
