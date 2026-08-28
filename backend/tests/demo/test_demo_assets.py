"""Contract tests for the fictional public demo asset bundle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from app.api.chat import _safe_event
from app.chat.orchestrator import ChatOrchestrator
from app.ingestion.extractors import extract_document
from app.main import build_document_runtime, build_rag_runtime, create_app
from app.rag.models import ChatEvent
from app.settings import Settings
from fastapi.testclient import TestClient

import scripts.build_demo_documents as document_builder
from scripts.build_demo_documents import build_demo_documents

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V1_DOCUMENT_NAMES = frozenset(
    f"{stem}.{extension}"
    for stem in (
        "training-policy",
        "pre-dialysis-check",
        "learning-points",
        "dialysis-basics",
    )
    for extension in ("docx", "pdf")
)
V1_DOCUMENTS_DIR = PROJECT_ROOT / "demo" / "documents"
V1_DEMO_DOCUMENTS = tuple(
    V1_DOCUMENTS_DIR / name for name in sorted(V1_DOCUMENT_NAMES)
)
DEMO_QUESTIONS = PROJECT_ROOT / "demo" / "questions.json"
VISUAL_ASSET_DIR = PROJECT_ROOT / "docs" / "assets"
VISUAL_SCENARIOS = {
    "chat-answer.png": "fact-training-hours",
    "chat-refusal.png": "no-answer-reimbursement",
    "chat-history.png": "followup-history-restore",
    "documents.png": "document-list",
    "runtime-mode.png": "runtime-mode-demo",
    "question-flow.gif": "question-flow",
}


def test_every_v1_demo_document_is_explicitly_fictional():
    assert len(V1_DEMO_DOCUMENTS) == 8  # four PDF/DOCX pairs
    assert {path.name for path in V1_DOCUMENTS_DIR.iterdir()} == V1_DOCUMENT_NAMES
    for path in V1_DEMO_DOCUMENTS:
        extracted = extract_document(path)
        text = "\n".join(block.text for block in extracted.blocks)
        assert "虚构演示资料" in text
        assert "不代表真实医疗制度或诊疗建议" in text


def test_demo_questions_cover_six_types_and_one_followup():
    payload = json.loads(DEMO_QUESTIONS.read_text(encoding="utf-8"))
    assert {item["type"] for item in payload["scenarios"]} >= {
        "fact",
        "procedure",
        "condition",
        "prohibition",
        "synthesis",
        "no_answer",
        "followup",
    }


def test_public_visual_manifest_matches_reviewed_metadata_free_assets():
    manifest_path = VISUAL_ASSET_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    entries = manifest["assets"]
    assert {entry["file"] for entry in entries} == set(VISUAL_SCENARIOS)
    assert len(entries) == len(VISUAL_SCENARIOS)

    for entry in entries:
        assert set(entry) == {
            "file",
            "scenario_id",
            "sha256",
            "width",
            "height",
            "reviewed",
        }
        file_name = entry["file"]
        assert entry["scenario_id"] == VISUAL_SCENARIOS[file_name]
        assert entry["reviewed"] is True
        assert isinstance(entry["sha256"], str)
        assert len(entry["sha256"]) == 64
        assert all(character in "0123456789abcdef" for character in entry["sha256"])

        asset = VISUAL_ASSET_DIR / file_name
        assert asset.is_file()
        raw = asset.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        width, height = _read_image_dimensions(asset, raw)
        assert (width, height) == (entry["width"], entry["height"])
        assert (width, height) == (1440, 900)
        assert not _contains_metadata(raw, asset.suffix.lower())
        assert not any(
            marker in raw.lower()
            for marker in (
                b"deepseek",
                b"siliconflow",
                b"authorization",
                b"bearer",
                b"api_key",
                b"c:" + b"\\",
                b"/" + b"users/",
            )
        )
        if asset.suffix.lower() == ".gif":
            assert len(raw) <= 8 * 1024 * 1024


def _read_image_dimensions(path: Path, raw: bytes) -> tuple[int, int]:
    if path.suffix.lower() == ".png":
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")
        assert raw[12:16] == b"IHDR"
        return struct.unpack(">II", raw[16:24])
    assert path.suffix.lower() == ".gif"
    assert raw[:6] in {b"GIF87a", b"GIF89a"}
    return struct.unpack("<HH", raw[6:10])


def _contains_metadata(raw: bytes, suffix: str) -> bool:
    if suffix == ".png":
        metadata_chunks = {
            b"cHRM",
            b"eXIf",
            b"gAMA",
            b"iCCP",
            b"iTXt",
            b"pHYs",
            b"sRGB",
            b"tEXt",
            b"zTXt",
        }
        offset = 8
        while offset + 12 <= len(raw):
            size = struct.unpack(">I", raw[offset : offset + 4])[0]
            kind = raw[offset + 4 : offset + 8]
            if kind in metadata_chunks:
                return True
            offset += 12 + size
        return False
    if len(raw) < 13:
        return True
    offset = 13
    packed = raw[10]
    if packed & 0x80:
        offset += 3 * (2 ** ((packed & 0x07) + 1))
    while offset < len(raw):
        block = raw[offset]
        offset += 1
        if block == 0x3B:
            return False
        if block == 0x2C:
            if offset + 9 > len(raw):
                return True
            local_packed = raw[offset + 8]
            offset += 9
            if local_packed & 0x80:
                offset += 3 * (2 ** ((local_packed & 0x07) + 1))
            if offset >= len(raw):
                return True
            offset += 1  # LZW minimum code size
            offset = _skip_gif_sub_blocks(raw, offset)
            continue
        if block != 0x21 or offset >= len(raw):
            return True
        label = raw[offset]
        offset += 1
        if label in {0x01, 0xFE, 0xFF}:
            return True
        if label == 0xF9:
            if offset >= len(raw):
                return True
            size = raw[offset]
            offset += 1 + size
            if offset >= len(raw) or raw[offset] != 0:
                return True
            offset += 1
            continue
        offset = _skip_gif_sub_blocks(raw, offset)
    return True


def _skip_gif_sub_blocks(raw: bytes, offset: int) -> int:
    while offset < len(raw):
        size = raw[offset]
        offset += 1
        if size == 0:
            return offset
        offset += size
    return offset


def _remove_empty_launcher_runtime(pid_file: Path) -> None:
    """Keep launcher tests from leaving private runtime parents in the checkout."""

    for directory in (
        pid_file.parent,
        pid_file.parent.parent,
        pid_file.parent.parent.parent,
    ):
        try:
            directory.rmdir()
        except OSError:
            break


def test_demo_route_accepts_internal_retrieval_diagnostics_without_adapter(tmp_path):
    """The real Demo RAG/orchestrator chain must reach the public final event."""

    settings = Settings(
        _env_file=None,
        app_runtime_mode="demo",
        source_documents_dir=PROJECT_ROOT / "demo" / "documents",
        demo_questions_path=DEMO_QUESTIONS,
        private_data_dir=tmp_path / "private",
        qdrant_path=tmp_path / "qdrant",
        sqlite_path=tmp_path / "app.sqlite3",
    )
    document_runtime = build_document_runtime(settings)
    rag_runtime = build_rag_runtime(settings, document_runtime)
    app = create_app(
        settings,
        document_service=document_runtime.service,
        rag_service=rag_runtime.service,
        chat_orchestrator=rag_runtime.orchestrator,
    )
    try:
        with TestClient(app, base_url="http://demo.test") as client:
            scan = client.post("/api/documents/scan")
            assert scan.status_code == 200
            assert scan.json()["snapshot_complete"] is True
            response = client.post(
                "/api/chat/stream",
                json={"question": "新员工独立值班前需要完成多少学时岗前培训？"},
            )
            assert response.status_code == 200
            events = []
            for frame in response.text.strip().split("\n\n"):
                lines = frame.splitlines()
                events.append(
                    {
                        "event": lines[0].removeprefix("event: "),
                        "data": json.loads(lines[1].removeprefix("data: ")),
                    }
                )
            assert events[-1]["event"] == "final"
            assert events[-1]["data"]["refused"] is False
            assert not any(
                event["event"] == "error"
                and event["data"].get("reason_code") == "INVALID_EVENT"
                for event in events
            )
    finally:
        asyncio.run(rag_runtime.close())


def test_demo_smoke_import_isolated_from_operator_env():
    """A clean smoke process must not instantiate Settings from operator ``.env``."""

    harness = r'''
import runpy
import sys
from pathlib import Path

backend_root = Path(sys.argv[1])
smoke_path = Path(sys.argv[2])
sys.path.insert(0, str(backend_root))
import app.settings as settings_module

original_settings = settings_module.Settings

class GuardedSettings(original_settings):
    def __init__(self, *args, **kwargs):
        if "_env_file" not in kwargs:
            raise AssertionError("operator .env was consulted")
        super().__init__(*args, **kwargs)

settings_module.Settings = GuardedSettings
module = runpy.run_path(str(smoke_path), run_name="demo_smoke_isolated")
result = module["run_demo_smoke"]("http://demo.test")
assert result.network_calls == 0
print("isolated_demo_smoke=PASS")
'''
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            harness,
            str(PROJECT_ROOT / "backend"),
            str(PROJECT_ROOT / "scripts" / "demo_smoke.py"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "isolated_demo_smoke=PASS" in result.stdout


def test_orchestrator_validates_internal_retrieval_diagnostics_shape():
    event = ChatEvent(
        type="final",
        data={
            "refused": False,
            "citations": [],
            "retrieval_diagnostics": {
                "candidate_count": 4,
                "evidence_count": 2,
                "highest_rerank_score": 0.34,
                "status": "ready",
                "reason_code": None,
            },
        },
    )

    validated = ChatOrchestrator._validate_event_data(event)

    assert validated is not None
    assert validated["retrieval_diagnostics"] == event.data["retrieval_diagnostics"]
    public = _safe_event(event, require_terminal_metadata=False)
    assert public is not None
    assert "retrieval_diagnostics" not in public.data
    error = ChatEvent(
        type="error",
        data={
            "stage": "retrieving",
            "reason_code": "RETRIEVAL_UNAVAILABLE",
            "retrieval_diagnostics": {
                "candidate_count": 0,
                "evidence_count": 0,
                "highest_rerank_score": None,
                "status": "unavailable",
                "reason_code": "RETRIEVAL_UNAVAILABLE",
            },
        },
    )
    error_validated = ChatOrchestrator._validate_event_data(error)
    assert error_validated is not None
    assert error_validated["retrieval_diagnostics"] == error.data["retrieval_diagnostics"]
    public_error = _safe_event(error, require_terminal_metadata=False)
    assert public_error is not None
    assert "retrieval_diagnostics" not in public_error.data
    for invalid in (
        {"candidate_count": -1},
        {
            "candidate_count": 1,
            "evidence_count": 1,
            "highest_rerank_score": float("nan"),
            "status": "ready",
            "reason_code": None,
        },
        {
            "candidate_count": 1,
            "evidence_count": 1,
            "highest_rerank_score": 10**1000,
            "status": "ready",
            "reason_code": None,
        },
        {
            "candidate_count": 1,
            "evidence_count": 1,
            "highest_rerank_score": 0.3,
            "status": "ready",
            "reason_code": "provider-secret",
        },
        {
            "candidate_count": 1,
            "evidence_count": 1,
            "highest_rerank_score": 0.3,
            "status": [],
            "reason_code": None,
        },
        {
            "candidate_count": 1,
            "evidence_count": 1,
            "highest_rerank_score": 0.3,
            "status": "ready",
            "reason_code": {},
        },
    ):
        malformed = ChatEvent(
            type="final",
            data={"refused": False, "citations": [], "retrieval_diagnostics": invalid},
        )
        assert ChatOrchestrator._validate_event_data(malformed) is None


def test_document_generator_rejects_source_directory_outside_public_tree(tmp_path):
    outside = tmp_path / "outside-sources"
    outside.mkdir()
    _write_synthetic_sources(outside)

    with pytest.raises(ValueError, match="approved demo source"):
        build_demo_documents(
            source_dir=outside,
            document_dir=tmp_path / "documents",
        )


def test_document_generator_rejects_symlinked_source_directory(tmp_path):
    outside = tmp_path / "outside-sources"
    outside.mkdir()
    _write_synthetic_sources(outside)
    linked = tmp_path / "linked-sources"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this Windows host")

    with pytest.raises(ValueError, match="symlink|reparse"):
        build_demo_documents(
            source_dir=linked,
            document_dir=tmp_path / "documents",
        )


def test_document_generator_rejects_dangling_source_symlink(tmp_path):
    linked = tmp_path / "dangling-sources"
    try:
        linked.symlink_to(tmp_path / "missing-sources", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this Windows host")

    with pytest.raises(ValueError, match="symlink|reparse"):
        build_demo_documents(
            source_dir=linked,
            document_dir=tmp_path / "documents",
        )


def test_document_generator_rejects_symlinked_output_directory(tmp_path):
    output_target = tmp_path / "output-target"
    output_target.mkdir()
    linked_output = tmp_path / "linked-output"
    try:
        linked_output.symlink_to(output_target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this Windows host")

    with pytest.raises(ValueError, match="symlink|reparse"):
        build_demo_documents(document_dir=linked_output)


def test_document_generator_rejects_dangling_directory_junction(tmp_path):
    target = tmp_path / "junction-target"
    target.mkdir()
    linked_output = tmp_path / "dangling-junction"
    if not _create_directory_junction(linked_output, target):
        pytest.skip("directory junctions are unavailable on this host")
    target.rmdir()
    try:
        with pytest.raises(ValueError, match="symlink|reparse"):
            build_demo_documents(document_dir=linked_output)
    finally:
        _remove_reparse_path(linked_output)


def test_document_generator_rejects_symlinked_source_file_inside_approved_tree(
    tmp_path, monkeypatch
):
    approved = tmp_path / "approved-sources"
    approved.mkdir()
    _write_synthetic_sources(approved)
    outside = tmp_path / "outside.md"
    outside.write_text("# 非批准资料\n\n合成内容。\n", encoding="utf-8")
    linked_source = approved / "training-policy.md"
    linked_source.unlink()
    try:
        linked_source.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this host")

    monkeypatch.setattr(document_builder, "DEFAULT_SOURCE_DIR", approved)
    with pytest.raises(ValueError, match="symlink|reparse"):
        build_demo_documents(
            source_dir=approved,
            document_dir=tmp_path / "documents",
        )


def test_launchers_use_cwd_safe_cloud_check_and_pid_ownership_contract():
    demo = (PROJECT_ROOT / "scripts" / "start_demo.ps1").read_text(encoding="utf-8")
    cloud = (PROJECT_ROOT / "scripts" / "start_cloud.ps1").read_text(encoding="utf-8")
    stop = (PROJECT_ROOT / "scripts" / "stop_local.ps1").read_text(encoding="utf-8")

    assert "Push-Location $backendRoot" in cloud
    for script in (demo, cloud):
        assert "process_path" in script
        assert "command_line_marker" in script
        assert "start_time" in script
        assert "start_time_ticks" in script
    assert "Get-CimInstance" in stop
    assert "command_line_marker" in stop
    assert "start_time" in stop
    assert "start_time_ticks" in stop
    assert "Stop-Process -Id $processId" in stop
    assert "if ($owned)" in stop


def test_stop_launcher_skips_unverified_pid_without_killing_current_process():
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is unavailable")
    pid_file = PROJECT_ROOT / "data" / "runtime" / "demo" / "pids.json"
    if pid_file.exists():
        pytest.skip("launcher PID file is already in use")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(
        json.dumps([{"name": "unverified", "process_id": os.getpid()}]),
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [
                shell,
                "-NoProfile",
                "-File",
                str(PROJECT_ROOT / "scripts" / "stop_local.ps1"),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "Stopped 0 launcher-owned process(es); skipped 1" in result.stdout
        os.kill(os.getpid(), 0)
    finally:
        pid_file.unlink(missing_ok=True)
        _remove_empty_launcher_runtime(pid_file)


def test_cloud_launcher_checks_configuration_from_arbitrary_cwd():
    if os.name != "nt":
        pytest.skip("the PowerShell launcher is a Windows-only local entrypoint")
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is unavailable")
    environment = os.environ.copy()
    environment["DEEPSEEK_API_KEY"] = "placeholder"
    environment["SILICONFLOW_API_KEY"] = "placeholder"
    pid_file = PROJECT_ROOT / "data" / "runtime" / "cloud" / "pids.json"
    try:
        result = subprocess.run(
            [
                shell,
                "-NoProfile",
                "-File",
                str(PROJECT_ROOT / "scripts" / "start_cloud.ps1"),
            ],
            cwd=PROJECT_ROOT.parent,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "Cloud provider status: deepseek=missing, siliconflow=missing" in combined
        assert "No module named 'app'" not in combined
    finally:
        pid_file.unlink(missing_ok=True)
        _remove_empty_launcher_runtime(pid_file)


def _write_synthetic_sources(root: Path) -> None:
    facts = {
        "training-policy": "合成培训资料。",
        "pre-dialysis-check": "合成核对资料。",
        "learning-points": "合成积分资料。",
        "dialysis-basics": "合成基础资料。",
    }
    for stem, fact in facts.items():
        (root / f"{stem}.md").write_text(f"# 合成资料\n\n{fact}\n", encoding="utf-8")


def _create_directory_junction(link: Path, target: Path) -> bool:
    if os.name != "nt":
        return False
    command_shell = os.environ.get("COMSPEC") or shutil.which("cmd.exe")
    if command_shell is None:
        return False
    result = subprocess.run(
        [command_shell, "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and link.exists()


def _remove_reparse_path(path: Path) -> None:
    command_shell = os.environ.get("COMSPEC") or shutil.which("cmd.exe")
    if os.name == "nt" and command_shell is not None:
        subprocess.run(
            [command_shell, "/c", "rmdir", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return
    path.unlink(missing_ok=True)

