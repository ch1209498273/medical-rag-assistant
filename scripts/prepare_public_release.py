"""Safely export the allow-listed public project surface."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

try:
    from scripts.public_release_manifest import (
        PUBLIC_RELEASE_MANIFEST,
        PublicReleaseManifest,
    )
except ModuleNotFoundError as exc:
    # ``python scripts/prepare_public_release.py`` puts only ``scripts/`` on
    # sys.path. Keep the documented direct-script invocation working without
    # weakening the package import used by tests and ``python -m``.
    if exc.name != "scripts.public_release_manifest":
        raise
    from public_release_manifest import (  # type: ignore[no-redef]
        PUBLIC_RELEASE_MANIFEST,
        PublicReleaseManifest,
    )

_REPARSE_POINT_ATTRIBUTE = 0x400
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
        "com¹",
        "com²",
        "com³",
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
        "lpt¹",
        "lpt²",
        "lpt³",
        "nul",
        "prn",
    }
)
_CREDENTIAL_NAMES = frozenset(
    {
        ".dockerconfigjson",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "authorized_keys",
    }
)
_SSH_KEY_NAMES = frozenset({"id_dsa", "id_ecdsa", "id_ed25519", "id_rsa"})
_CREDENTIAL_MARKER_PATTERN = re.compile(
    r"(?:^|[._-])(?:access[_-]?key|access[_-]?token|api[_-]?key|"
    r"credential|credentials|passphrase|passwd|password|private[_-]?key|"
    r"secret|secrets|token|tokens)(?:$|[._-]|\d)"
)
_DATABASE_SUFFIXES = (".db", ".sqlite", ".sqlite3")
_DATABASE_SIDECARS = ("-wal", "-shm", "-journal")
_GENERATED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "dist",
        "node_modules",
    }
)
_GENERATED_FILE_SUFFIXES = frozenset({".pyc", ".pyo"})
_ARCHIVE_SUFFIXES = frozenset(
    {
        ".7z",
        ".arj",
        ".bz2",
        ".cab",
        ".gz",
        ".lz",
        ".lz4",
        ".rar",
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


class PublicReleaseError(RuntimeError):
    """Raised before copying when the public-export boundary is unsafe."""


def _is_link_or_reparse(path: Path) -> bool:
    """Return whether *path* is a symbolic link or Windows reparse point."""

    try:
        if path.is_symlink():
            return True
        metadata = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE)


def _assert_no_link_ancestors(path: Path, label: str) -> None:
    """Reject links in the path chain before resolving or opening it."""

    absolute = path.absolute()
    parts = absolute.parts
    current = Path(parts[0])
    if _is_link_or_reparse(current):
        raise PublicReleaseError(f"{label} contains a symlink or reparse point")
    for part in parts[1:]:
        current /= part
        if _is_link_or_reparse(current):
            raise PublicReleaseError(f"{label} contains a symlink or reparse point")


def _path_key(path: Path) -> tuple[str, ...]:
    """Return a case-insensitive path key for cross-platform collisions."""

    return tuple(part.casefold() for part in path.parts)


def _is_same_or_descendant(path: Path, parent: Path) -> bool:
    path_key = _path_key(path)
    parent_key = _path_key(parent)
    return path_key[: len(parent_key)] == parent_key


def _safe_relative_if_descendant(path: Path, parent: Path, label: str) -> Path | None:
    """Return a relative path or reject ambiguous case-folded containment."""

    try:
        return path.relative_to(parent)
    except ValueError as exc:
        if _is_same_or_descendant(path, parent):
            raise PublicReleaseError(f"{label} path containment is ambiguous") from exc
        return None


def _is_blocked_path(relative: Path) -> bool:
    """Return whether a source or destination relative path is protected."""

    parts = tuple(part.casefold() for part in relative.parts)
    if not parts:
        return False

    name = parts[-1]
    if ".git" in parts or "资料" in parts:
        return True
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return True
    if "logs" in parts or name.endswith(".log") or ".log." in name:
        return True

    protected_pairs = (
        ("data", "private"),
        ("data", "processed"),
        ("data", "qdrant"),
        ("data", "evaluations"),
        ("data", "runtime"),
    )
    if any(
        any(parts[index : index + 2] == pair for index in range(len(parts) - 1))
        for pair in protected_pairs
    ):
        return True

    if any(name.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES):
        return True
    if name in _CREDENTIAL_NAMES or _CREDENTIAL_MARKER_PATTERN.search(name):
        return True
    if name.split(".", maxsplit=1)[0] in _SSH_KEY_NAMES:
        return True
    return any(
        name.endswith(database_suffix)
        or any(
            name.endswith(database_suffix + sidecar) for sidecar in _DATABASE_SIDECARS
        )
        for database_suffix in _DATABASE_SUFFIXES
    )


def _validate_windows_path_parts(
    parts: tuple[str, ...], category: str, display: str
) -> None:
    for part in parts:
        if (
            part.rstrip(" .") != part
            or any(character in part for character in '<>"|?*\\')
            or any(ord(character) < 32 for character in part)
            or ":" in part
        ):
            raise PublicReleaseError(
                f"{category} path is unsafe Windows path: {display!r}"
            )
        windows_name = part.split(".", maxsplit=1)[0].casefold()
        if windows_name in _WINDOWS_RESERVED_NAMES:
            raise PublicReleaseError(
                f"{category} path is unsafe Windows path: {display!r}"
            )


def _validate_manifest_path(value: object, category: str) -> Path:
    """Validate one manifest value and return its native relative path."""

    if not isinstance(value, str) or not value:
        raise PublicReleaseError(f"{category} path must be a relative POSIX path")
    if "\x00" in value or "\\" in value:
        raise PublicReleaseError(
            f"{category} path must be a relative POSIX path: {value!r}"
        )

    posix = PurePosixPath(value)
    if (
        posix.is_absolute()
        or not posix.parts
        or value != posix.as_posix()
        or any(part in {"", ".", ".."} for part in posix.parts)
        or PureWindowsPath(value).drive
    ):
        raise PublicReleaseError(
            f"{category} path must be a relative POSIX path: {value!r}"
        )
    _validate_windows_path_parts(posix.parts, category, value)
    return Path(*posix.parts)


def _normalise_manifest(
    manifest: PublicReleaseManifest,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    """Validate manifest syntax, duplicates, and file/directory overlaps."""

    try:
        raw_files = tuple(manifest.files)
        raw_directories = tuple(manifest.directories)
        raw_excluded_files = tuple(manifest.excluded_files)
    except (AttributeError, TypeError) as exc:
        raise PublicReleaseError(
            "manifest must define files, directories, and excluded files"
        ) from exc

    files = tuple(_validate_manifest_path(value, "file") for value in raw_files)
    directories = tuple(
        _validate_manifest_path(value, "directory") for value in raw_directories
    )
    excluded_files = tuple(
        _validate_manifest_path(value, "excluded file")
        for value in raw_excluded_files
    )

    seen: dict[tuple[str, ...], str] = {}
    for category, entries in (
        ("file", files),
        ("directory", directories),
        ("excluded file", excluded_files),
    ):
        for entry in entries:
            key = _path_key(entry)
            if key in seen:
                raise PublicReleaseError(
                    f"duplicate manifest entry: {entry.as_posix()}"
                )
            seen[key] = category
            if _is_blocked_path(entry):
                raise PublicReleaseError(f"blocked manifest path: {entry.as_posix()}")

    for file_path in files:
        for directory_path in directories:
            file_key = _path_key(file_path)
            directory_key = _path_key(directory_path)
            if (
                file_key[: len(directory_key)] == directory_key
                or directory_key[: len(file_key)] == file_key
            ):
                raise PublicReleaseError(
                    "manifest file/directory paths overlap: "
                    f"{file_path.as_posix()} and {directory_path.as_posix()}"
                )

    for index, first in enumerate(files):
        first_key = _path_key(first)
        for second in files[index + 1 :]:
            second_key = _path_key(second)
            if (
                first_key[: len(second_key)] == second_key
                or second_key[: len(first_key)] == first_key
            ):
                raise PublicReleaseError(
                    "manifest file paths overlap: "
                    f"{first.as_posix()} and {second.as_posix()}"
                )

    for excluded_file in excluded_files:
        excluded_key = _path_key(excluded_file)
        if not any(
            excluded_key[: len(_path_key(directory))] == _path_key(directory)
            and excluded_key != _path_key(directory)
            for directory in directories
        ):
            raise PublicReleaseError(
                "excluded file must be inside an allowed manifest directory: "
                f"{excluded_file.as_posix()}"
            )
    return files, directories, excluded_files


def _validate_output_root(output_root: Path) -> Path:
    """Validate an output path without creating or changing it."""

    _assert_no_link_ancestors(output_root, "output path")
    try:
        exists = output_root.exists()
    except OSError as exc:
        raise PublicReleaseError("unable to inspect output path") from exc

    if exists:
        try:
            if not output_root.is_dir():
                raise PublicReleaseError("output path must be a directory")
            if any(output_root.iterdir()):
                raise PublicReleaseError("output directory must be empty")
        except OSError as exc:
            raise PublicReleaseError("unable to inspect output directory") from exc

    try:
        return output_root.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise PublicReleaseError("unable to resolve output path") from exc


def _resolve_source_root(source_root: Path) -> Path:
    _assert_no_link_ancestors(source_root, "source root")
    try:
        if not source_root.exists():
            raise PublicReleaseError("source root does not exist")
        if not source_root.is_dir():
            raise PublicReleaseError("source root must be a directory")
        return source_root.resolve(strict=True)
    except PublicReleaseError:
        raise
    except (OSError, RuntimeError) as exc:
        raise PublicReleaseError("unable to resolve source root") from exc


def _resolve_source_entry(
    source_root: Path,
    relative: Path,
    category: str,
    output_root: Path,
) -> Path:
    candidate = source_root.joinpath(*relative.parts)
    if _is_same_or_descendant(candidate, output_root):
        raise PublicReleaseError(
            f"manifest {category} overlaps output directory: {relative.as_posix()}"
        )
    _assert_no_link_ancestors(candidate, f"manifest {category}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PublicReleaseError(
            f"manifest {category} is missing: {relative.as_posix()}"
        ) from exc
    try:
        resolved.relative_to(source_root)
    except ValueError as exc:
        raise PublicReleaseError(
            f"manifest {category} is outside source root: {relative.as_posix()}"
        ) from exc
    if _is_link_or_reparse(candidate):
        raise PublicReleaseError(
            f"manifest {category} contains a symlink or reparse point: "
            f"{relative.as_posix()}"
        )
    try:
        is_directory = candidate.is_dir()
        is_file = candidate.is_file()
    except OSError as exc:
        raise PublicReleaseError(
            f"unable to inspect manifest {category}: {relative.as_posix()}"
        ) from exc
    if category == "directory" and not is_directory:
        raise PublicReleaseError(
            f"manifest directory is not a directory: {relative.as_posix()}"
        )
    if category == "file" and not is_file:
        raise PublicReleaseError(f"manifest file is not a file: {relative.as_posix()}")
    return resolved


def _collect_directory(
    source_root: Path,
    source_directory: Path,
    relative_directory: Path,
    output_root: Path,
    excluded_file_keys: set[tuple[str, ...]],
    directory_items: dict[tuple[str, ...], tuple[Path, Path]],
    file_items: dict[tuple[str, ...], tuple[Path, Path]],
) -> None:
    """Recursively collect files without ever following a link."""

    directory_key = _path_key(relative_directory)
    existing_directory = directory_items.get(directory_key)
    if existing_directory is None:
        directory_items[directory_key] = (source_directory, relative_directory)
    elif existing_directory[0] != source_directory:
        raise PublicReleaseError(
            f"source directory collision: {relative_directory.as_posix()}"
        )

    try:
        children = sorted(
            source_directory.iterdir(), key=lambda path: path.name.casefold()
        )
    except OSError as exc:
        raise PublicReleaseError(
            f"unable to inspect manifest directory: {relative_directory.as_posix()}"
        ) from exc

    for child in children:
        relative_child = relative_directory / child.name
        if _is_same_or_descendant(child, output_root):
            continue
        if _path_key(relative_child) in excluded_file_keys:
            continue
        if _is_link_or_reparse(child):
            raise PublicReleaseError(
                f"source path contains a symlink or reparse point: "
                f"{relative_child.as_posix()}"
            )
        if child.name.casefold() in _GENERATED_DIRECTORY_NAMES or (
            child.suffix.casefold() in _GENERATED_FILE_SUFFIXES
        ):
            continue
        _validate_windows_path_parts(
            tuple(relative_child.parts), "source", relative_child.as_posix()
        )
        if _is_blocked_path(relative_child):
            raise PublicReleaseError(
                f"blocked source path: {relative_child.as_posix()}"
            )
        try:
            is_directory = child.is_dir()
            is_file = child.is_file()
        except OSError as exc:
            raise PublicReleaseError(
                f"unable to inspect source path: {relative_child.as_posix()}"
            ) from exc
        if is_directory:
            _collect_directory(
                source_root,
                child,
                relative_child,
                output_root,
                excluded_file_keys,
                directory_items,
                file_items,
            )
        elif is_file:
            file_key = _path_key(relative_child)
            if file_key in directory_items:
                raise PublicReleaseError(
                    f"source file/directory collision: {relative_child.as_posix()}"
                )
            try:
                resolved_child = child.resolve(strict=True)
                resolved_child.relative_to(source_root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise PublicReleaseError(
                    f"source file is outside source root: {relative_child.as_posix()}"
                ) from exc
            existing_file = file_items.get(file_key)
            if existing_file is not None:
                if existing_file[0] != resolved_child:
                    raise PublicReleaseError(
                        f"source file collision: {relative_child.as_posix()}"
                    )
            else:
                file_items[file_key] = (resolved_child, relative_child)
        else:
            raise PublicReleaseError(
                f"unsupported source entry: {relative_child.as_posix()}"
            )


def _collect_copy_plan(
    source_root: Path,
    output_root: Path,
    files: tuple[Path, ...],
    directories: tuple[Path, ...],
    excluded_files: tuple[Path, ...],
) -> tuple[tuple[tuple[Path, Path], ...], tuple[tuple[Path, Path], ...]]:
    directory_items: dict[tuple[str, ...], tuple[Path, Path]] = {}
    file_items: dict[tuple[str, ...], tuple[Path, Path]] = {}

    for relative_file in files:
        source_file = _resolve_source_entry(
            source_root, relative_file, "file", output_root
        )
        file_key = _path_key(relative_file)
        file_items[file_key] = (source_file, relative_file)

    excluded_file_keys: set[tuple[str, ...]] = set()
    for excluded_file in excluded_files:
        candidate = source_root.joinpath(*excluded_file.parts)
        if _is_link_or_reparse(candidate):
            raise PublicReleaseError(
                "excluded file contains a symlink or reparse point: "
                f"{excluded_file.as_posix()}"
            )
        try:
            exists = candidate.exists()
        except OSError as exc:
            raise PublicReleaseError(
                "unable to inspect excluded file: " f"{excluded_file.as_posix()}"
            ) from exc
        if exists:
            _resolve_source_entry(source_root, excluded_file, "file", output_root)
        excluded_file_keys.add(_path_key(excluded_file))

    for relative_directory in directories:
        source_directory = _resolve_source_entry(
            source_root, relative_directory, "directory", output_root
        )
        _collect_directory(
            source_root,
            source_directory,
            relative_directory,
            output_root,
            excluded_file_keys,
            directory_items,
            file_items,
        )

    for file_key in file_items:
        if file_key in directory_items:
            relative = file_items[file_key][1]
            raise PublicReleaseError(
                f"source file/directory collision: {relative.as_posix()}"
            )

    sorted_directories = tuple(
        sorted(directory_items.values(), key=lambda item: item[1].as_posix())
    )
    sorted_files = tuple(
        sorted(file_items.values(), key=lambda item: item[1].as_posix())
    )
    return sorted_directories, sorted_files


def _ensure_output_directory(output_root: Path) -> None:
    """Create the output root only after the complete copy plan is valid."""

    _assert_no_link_ancestors(output_root, "output path")
    try:
        if output_root.exists():
            if not output_root.is_dir():
                raise PublicReleaseError("output path must be a directory")
            if any(output_root.iterdir()):
                raise PublicReleaseError("output directory must be empty")
            return
        output_root.mkdir(parents=True, exist_ok=False)
    except PublicReleaseError:
        raise
    except OSError as exc:
        raise PublicReleaseError("unable to create output directory") from exc


def _copy_plan(
    output_root: Path,
    directories: tuple[tuple[Path, Path], ...],
    files: tuple[tuple[Path, Path], ...],
) -> tuple[Path, ...]:
    """Apply a prevalidated plan; copy errors leave deterministic partial output."""

    for _, relative_directory in directories:
        destination = output_root.joinpath(*relative_directory.parts)
        _assert_no_link_ancestors(destination, "output path")
        try:
            if destination.exists() and not destination.is_dir():
                raise PublicReleaseError(
                    f"output path changed during export: {relative_directory.as_posix()}"
                )
            destination.mkdir(parents=True, exist_ok=True)
        except PublicReleaseError:
            raise
        except OSError as exc:
            raise PublicReleaseError(
                f"unable to create output directory: {relative_directory.as_posix()}"
            ) from exc

    copied: list[Path] = []
    for source_file, relative_file in files:
        destination = output_root.joinpath(*relative_file.parts)
        _assert_no_link_ancestors(destination.parent, "output path")
        try:
            if destination.exists() or _is_link_or_reparse(destination):
                raise PublicReleaseError(
                    f"output path changed during export: {relative_file.as_posix()}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
        except PublicReleaseError:
            raise
        except (OSError, shutil.Error) as exc:
            raise PublicReleaseError(
                f"unable to copy public file: {relative_file.as_posix()}"
            ) from exc
        copied.append(destination)
    return tuple(sorted(copied, key=lambda path: path.as_posix()))


def prepare_public_release(
    source_root: Path,
    output_root: Path,
    manifest: PublicReleaseManifest = PUBLIC_RELEASE_MANIFEST,
) -> tuple[Path, ...]:
    """Copy only a validated public manifest into a new or empty directory."""

    try:
        source_candidate = Path(source_root)
        output_candidate = Path(output_root)
    except TypeError as exc:
        raise PublicReleaseError("source and output roots must be paths") from exc

    output = _validate_output_root(output_candidate)
    files, directories, excluded_files = _normalise_manifest(manifest)
    source = _resolve_source_root(source_candidate)

    if output == source:
        raise PublicReleaseError("output directory must differ from source root")
    relative_output = _safe_relative_if_descendant(output, source, "output")
    if relative_output is not None and _is_blocked_path(relative_output):
        raise PublicReleaseError(
            f"output directory is blocked: {relative_output.as_posix()}"
        )

    planned_directories, planned_files = _collect_copy_plan(
        source,
        output,
        files,
        directories,
        excluded_files,
    )
    _ensure_output_directory(output)
    return _copy_plan(output, planned_directories, planned_files)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the public release surface")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("."),
        help="source project root (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or empty public release directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        copied = prepare_public_release(args.source, args.output)
    except PublicReleaseError as exc:
        print(f"public release export failed: {exc}", file=sys.stderr)
        return 1

    output = Path(args.output).resolve()
    for path in copied:
        print(path.resolve().relative_to(output).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
