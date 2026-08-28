import os
from pathlib import Path, PurePosixPath

import pytest

from scripts.prepare_public_release import PublicReleaseError, prepare_public_release
from scripts.public_release_manifest import PublicReleaseManifest


def test_export_copies_only_manifest_entries(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    (source / "backend").mkdir(parents=True)
    (source / "backend" / "app.py").write_text("safe", encoding="utf-8")
    (source / ".env").write_text(
        "DEEPSEEK_API_KEY=" + "forbidden", encoding="utf-8"
    )
    manifest = PublicReleaseManifest(files=(), directories=("backend",))

    copied = prepare_public_release(source, output, manifest)

    assert output / "backend" / "app.py" in copied
    assert not (output / ".env").exists()


def test_export_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "public"
    output.mkdir()
    (output / "keep.txt").write_text("owned by user", encoding="utf-8")
    with pytest.raises(PublicReleaseError, match="output directory must be empty"):
        prepare_public_release(tmp_path / "source", output)


def test_export_returns_sorted_files_and_preserves_empty_directories(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    (source / "empty").mkdir(parents=True)
    (source / "z.txt").write_text("z", encoding="utf-8")
    (source / "a.txt").write_text("a", encoding="utf-8")
    manifest = PublicReleaseManifest(files=("z.txt", "a.txt"), directories=("empty",))

    copied = prepare_public_release(source, output, manifest)

    assert [path.relative_to(output).as_posix() for path in copied] == [
        "a.txt",
        "z.txt",
    ]
    assert (output / "empty").is_dir()


def test_export_validates_all_entries_before_creating_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    source.mkdir()
    (source / "safe.txt").write_text("safe", encoding="utf-8")
    manifest = PublicReleaseManifest(files=("safe.txt", "missing.txt"), directories=())

    with pytest.raises(PublicReleaseError, match="manifest file is missing"):
        prepare_public_release(source, output, manifest)

    assert not output.exists()


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "",
        ".",
        "../outside.txt",
        "/absolute.txt",
        "C:" + "/outside.txt",
        "foo\\bar.txt",
        "foo/../bar.txt",
        "foo//bar.txt",
    ],
)
def test_export_rejects_non_relative_posix_manifest_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    source.mkdir()

    with pytest.raises(PublicReleaseError, match="relative POSIX path"):
        prepare_public_release(
            source,
            output,
            PublicReleaseManifest(files=(unsafe_path,), directories=()),
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("files", "directories"),
    [
        (("safe.txt", "safe.txt"), ()),
        ((), ("safe", "safe")),
        (("safe.txt",), ("safe.txt",)),
        (("safe/nested.txt",), ("safe",)),
    ],
)
def test_export_rejects_duplicate_or_overlapping_manifest_entries(
    tmp_path: Path, files: tuple[str, ...], directories: tuple[str, ...]
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    source.mkdir()

    with pytest.raises(PublicReleaseError, match="manifest"):
        prepare_public_release(
            source,
            output,
            PublicReleaseManifest(files=files, directories=directories),
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "blocked_path",
    [
        ".git/config",
        ".env",
        ".env.local",
        "nested/data/private/secret.txt",
        "data/qdrant/index",
        "logs/run.txt",
        "trace.log",
        "database/app.sqlite3-wal",
        "bundle.zip",
    ],
)
def test_export_rejects_blocked_manifest_paths(
    tmp_path: Path, blocked_path: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    blocked = source / Path(blocked_path)
    blocked.parent.mkdir(parents=True)
    blocked.write_text("private", encoding="utf-8")

    with pytest.raises(PublicReleaseError, match="blocked"):
        prepare_public_release(
            source,
            output,
            PublicReleaseManifest(files=(blocked_path,), directories=()),
        )

    assert not output.exists()


def test_export_rejects_symlink_inside_allowed_directory_before_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    allowed = source / "allowed"
    allowed.mkdir(parents=True)
    (allowed / "safe.txt").write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    link = allowed / "linked.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(PublicReleaseError, match="symlink"):
        prepare_public_release(
            source,
            output,
            PublicReleaseManifest(files=(), directories=("allowed",)),
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "unsafe_path",
    [
        ".git./config",
        "logs./run.txt",
        "data/private./secret.txt",
        ".. /outside.txt",
        ". /outside.txt",
        "CON.txt",
        "CONIN$.txt",
        "aux",
        "safe.txt:secret",
        "wild*card.txt",
    ],
)
def test_export_rejects_windows_normalized_manifest_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    source.mkdir()

    with pytest.raises(PublicReleaseError, match="Windows path"):
        prepare_public_release(
            source,
            output,
            PublicReleaseManifest(files=(unsafe_path,), directories=()),
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "blocked_path",
    [
        "资料/真实制度.txt",
        "nested/资料/真实制度.txt",
        "data/runtime/state.json",
        "nested/data/runtime/state.json",
    ],
)
def test_export_rejects_newly_protected_paths(
    tmp_path: Path, blocked_path: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    blocked = source / Path(blocked_path)
    blocked.parent.mkdir(parents=True)
    blocked.write_text("private", encoding="utf-8")

    with pytest.raises(PublicReleaseError, match="blocked"):
        prepare_public_release(
            source,
            output,
            PublicReleaseManifest(files=(blocked_path,), directories=()),
        )

    assert not output.exists()


def test_export_rejects_protected_paths_discovered_in_allowed_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    protected = source / "backend" / "资料" / "真实制度.txt"
    protected.parent.mkdir(parents=True)
    protected.write_text("private", encoding="utf-8")

    with pytest.raises(PublicReleaseError, match="blocked"):
        prepare_public_release(
            source,
            output,
            PublicReleaseManifest(files=(), directories=("backend",)),
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "credential_name",
    [
        "credentials.json",
        "secrets.yaml",
        ".npmrc",
        ".pypirc",
        "id_rsa",
        "request-token.txt",
        "token123.txt",
        "service_access_key.json",
    ],
)
def test_export_rejects_nested_credential_artifacts(
    tmp_path: Path, credential_name: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    credential = source / "allowed" / "fixtures" / credential_name
    credential.parent.mkdir(parents=True)
    credential.write_text("credential", encoding="utf-8")

    with pytest.raises(PublicReleaseError, match="blocked"):
        prepare_public_release(
            source,
            output,
            PublicReleaseManifest(
                files=(credential.relative_to(source).as_posix(),), directories=()
            ),
        )

    assert not output.exists()


def test_export_rejects_credentials_discovered_in_allowed_test_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "public"
    credential = source / "backend" / "tests" / "fixtures" / "credentials.json"
    credential.parent.mkdir(parents=True)
    credential.write_text("credential", encoding="utf-8")

    with pytest.raises(PublicReleaseError, match="blocked"):
        prepare_public_release(
            source,
            output,
            PublicReleaseManifest(files=(), directories=("backend",)),
        )

    assert not output.exists()


def test_export_reports_casefolded_output_boundary_as_public_release_error() -> None:
    with pytest.raises(PublicReleaseError, match="output"):
        from scripts.prepare_public_release import _safe_relative_if_descendant

        _safe_relative_if_descendant(
            PurePosixPath("/" + "tmp/source/public"),
            PurePosixPath("/" + "tmp/Source"),
            "output",
        )


def test_export_wraps_copy_failure_without_overwriting_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    release_module = importlib.import_module("scripts.prepare_public_release")
    source = tmp_path / "source"
    output = tmp_path / "public"
    source.mkdir()
    (source / "a.txt").write_text("a", encoding="utf-8")
    (source / "b.txt").write_text("b", encoding="utf-8")
    original_copy2 = release_module.shutil.copy2

    def fail_for_second_file(source_file: Path, destination: Path) -> None:
        if source_file.name == "b.txt":
            raise PermissionError("synthetic copy failure")
        original_copy2(source_file, destination)

    monkeypatch.setattr(release_module.shutil, "copy2", fail_for_second_file)

    with pytest.raises(PublicReleaseError, match="unable to copy public file: b.txt"):
        prepare_public_release(
            source,
            output,
            PublicReleaseManifest(files=("a.txt", "b.txt"), directories=()),
        )

    assert (output / "a.txt").read_text(encoding="utf-8") == "a"
    assert not (output / "b.txt").exists()
    assert (source / "a.txt").read_text(encoding="utf-8") == "a"
    assert (source / "b.txt").read_text(encoding="utf-8") == "b"
