from pathlib import Path

from app.settings import PROJECT_ROOT, Settings


def test_relative_storage_paths_are_resolved_from_backend_directory() -> None:
    settings = Settings(
        _env_file=None,
        source_documents_dir=Path("../fictional-documents"),
        private_data_dir=Path("../data/private"),
        qdrant_path=Path("../data/qdrant"),
        sqlite_path=Path("../data/app.sqlite3"),
    )

    assert settings.source_documents_dir == PROJECT_ROOT / "fictional-documents"
    assert settings.private_data_dir == PROJECT_ROOT / "data/private"
    assert settings.qdrant_path == PROJECT_ROOT / "data/qdrant"
    assert settings.sqlite_path == PROJECT_ROOT / "data/app.sqlite3"


def test_absolute_storage_paths_are_not_rebased(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path / "documents",
        private_data_dir=tmp_path / "private",
        qdrant_path=tmp_path / "qdrant",
        sqlite_path=tmp_path / "app.sqlite3",
    )

    assert settings.source_documents_dir == tmp_path / "documents"
    assert settings.private_data_dir == tmp_path / "private"
    assert settings.qdrant_path == tmp_path / "qdrant"
    assert settings.sqlite_path == tmp_path / "app.sqlite3"
