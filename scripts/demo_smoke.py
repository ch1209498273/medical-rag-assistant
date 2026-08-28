"""Run the public fictional demo entirely in-process and without network I/O."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

warnings.filterwarnings(
    "ignore",
    message=r"Using `httpx` with `starlette.testclient` is deprecated",
)
warnings.filterwarnings("ignore", message=r"The `fitz` API is deprecated")
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = SCRIPT_ROOT / "backend"
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
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@dataclass(frozen=True)
class DemoSmokeResult:
    """Small, safe summary emitted by the offline smoke test."""

    indexed_documents: int
    answered_questions: int
    refused_questions: int
    network_calls: int


class NoNetworkTransport:
    """Transport seam that records and rejects every outbound request."""

    def __init__(self) -> None:
        self.calls = 0

    def request(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.calls += 1
        raise AssertionError("demo smoke attempted an outbound request")

    def stream(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.calls += 1
        raise AssertionError("demo smoke attempted an outbound stream")


def _import_demo_components() -> tuple[Any, Any, Any, Any, type[Any]]:
    """Import the app assembly without allowing an operator ``.env`` to leak in."""

    # ``app.main`` retains a module-level ``app = create_app()`` for Uvicorn.
    # Import it with an explicit no-env Settings subclass so that this in-process
    # smoke path remains independent of any operator cloud configuration.
    from app import settings as settings_module

    original_settings = settings_module.Settings

    class ImportSafeSettings(original_settings):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("_env_file", None)
            super().__init__(*args, **kwargs)

    settings_module.Settings = ImportSafeSettings
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            from app.main import (
                build_document_runtime,
                build_rag_runtime,
                create_app,
            )
            from app.settings import Settings
            from fastapi.testclient import TestClient
    finally:
        settings_module.Settings = original_settings
    return Settings, build_document_runtime, build_rag_runtime, create_app, TestClient


def run_demo_smoke(base_url: str) -> DemoSmokeResult:
    """Index the fictional bundle and exercise answer/refusal/follow-up paths."""

    (
        Settings,
        build_document_runtime,
        build_rag_runtime,
        create_app,
        TestClient,
    ) = _import_demo_components()

    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty URL")
    document_dir = SCRIPT_ROOT / "demo" / "documents"
    questions_path = SCRIPT_ROOT / "demo" / "questions.json"
    document_names = (
        {path.name for path in document_dir.iterdir()}
        if document_dir.is_dir()
        else set()
    )
    if document_names != V1_DOCUMENT_NAMES:
        raise RuntimeError("fictional demo documents are not built")
    if not questions_path.is_file():
        raise RuntimeError("fictional demo questions are missing")

    # Keep smoke-test state outside the checkout.  The release validator treats
    # ``data/`` as private application state, so even an empty runtime parent
    # created by this command must not appear in the public copy.
    with tempfile.TemporaryDirectory(prefix="medical-rag-demo-smoke-") as temp:
        storage_root = Path(temp)
        settings = Settings(
            _env_file=None,
            app_runtime_mode="demo",
            source_documents_dir=document_dir,
            demo_questions_path=questions_path,
            private_data_dir=storage_root / "private",
            qdrant_path=storage_root / "qdrant",
            sqlite_path=storage_root / "app.sqlite3",
        )
        transport = NoNetworkTransport()
        document_runtime = build_document_runtime(settings, transport=transport)
        rag_runtime = build_rag_runtime(
            settings,
            document_runtime,
            transport=transport,
        )
        # The public stateful orchestrator accepts only the public event
        # contract; RagService's private retrieval diagnostics are validated
        # by the orchestrator and stripped by the public API boundary.
        if rag_runtime.orchestrator is None:
            raise RuntimeError("demo stateful orchestrator is unavailable")
        app = create_app(
            settings,
            document_service=document_runtime.service,
            rag_service=rag_runtime.service,
            chat_orchestrator=rag_runtime.orchestrator,
        )
        try:
            # PyMuPDF's compatibility shim writes one deprecation notice when
            # the first PDF is opened; keep smoke output to its summary.
            with (
                TestClient(app, base_url=base_url) as client,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                scan = _post_json(client, "/api/documents/scan")
                results = scan.get("results")
                if (
                    scan.get("snapshot_complete") is not True
                    or not isinstance(results, list)
                    or len(results) != 8
                    or any(
                        not isinstance(item, dict)
                        or item.get("status") not in {"active", "skipped"}
                        for item in results
                    )
                ):
                    raise RuntimeError("demo document scan contract failed")
                indexed_documents = sum(
                    1
                    for item in results
                    if isinstance(item, dict)
                    and item.get("status") in {"active", "skipped"}
                )

                first_frames = _chat(client, "新员工独立值班前需要完成多少学时岗前培训？")
                first_final = _terminal(first_frames)
                _require_answered(first_frames, first_final)
                session_id = _session_id(first_frames)

                refusal_frames = _chat(client, "血液透析费用报销比例是多少？")
                refusal_final = _terminal(refusal_frames)
                _require_refused(refusal_frames, refusal_final)

                followup_frames = _chat(
                    client,
                    "那没通过怎么办？",
                    session_id=session_id,
                )
                followup_final = _terminal(followup_frames)
                _require_answered(followup_frames, followup_final)

                result = DemoSmokeResult(
                    indexed_documents=indexed_documents,
                    answered_questions=2,
                    refused_questions=1,
                    network_calls=transport.calls,
                )
                if result.network_calls != 0:
                    raise RuntimeError("demo smoke observed outbound network calls")
                return result
        finally:
            asyncio.run(rag_runtime.close())


def _post_json(client: TestClient, path: str) -> dict[str, Any]:
    response = client.post(path)
    if response.status_code != 200:
        raise RuntimeError(f"demo API request failed: {path}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"demo API returned invalid JSON: {path}")
    return payload


def _chat(
    client: TestClient,
    question: str,
    *,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    body: dict[str, str] = {"question": question}
    if session_id is not None:
        body["session_id"] = session_id
    response = client.post("/api/chat/stream", json=body)
    if response.status_code != 200:
        raise RuntimeError("demo chat request failed")
    return _parse_sse(response.text)


def _parse_sse(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for frame in text.strip().split("\n\n"):
        if not frame.strip():
            continue
        event_name: str | None = None
        data_text: str | None = None
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_name = line.partition(":")[2].strip()
            elif line.startswith("data:"):
                data_text = line.partition(":")[2].strip()
        if event_name is None or data_text is None:
            raise RuntimeError("demo chat returned malformed SSE")
        data = json.loads(data_text)
        if not isinstance(data, dict):
            raise TypeError("demo chat returned malformed event data")
        events.append({"event": event_name, "data": data})
    if not events:
        raise RuntimeError("demo chat returned no SSE events")
    return events


def _terminal(events: list[dict[str, Any]]) -> dict[str, Any]:
    terminal = [
        event["data"]
        for event in events
        if event.get("event") == "final"
    ]
    if len(terminal) != 1:
        raise RuntimeError("demo chat did not return one final event")
    return terminal[0]


def _session_id(events: list[dict[str, Any]]) -> str:
    accepted = [
        event["data"]
        for event in events
        if event.get("event") == "status"
        and isinstance(event.get("data"), dict)
        and event["data"].get("stage") == "accepted"
    ]
    if len(accepted) != 1 or not isinstance(accepted[0].get("session_id"), str):
        raise RuntimeError("demo chat did not return an accepted session")
    return accepted[0]["session_id"]


def _require_answered(events: list[dict[str, Any]], final: dict[str, Any]) -> None:
    if final.get("refused") is not False or not final.get("citations"):
        raise RuntimeError("demo supported question was not answered with citations")
    if not any(event.get("event") == "answer_delta" for event in events):
        raise RuntimeError("demo supported question emitted no answer")


def _require_refused(events: list[dict[str, Any]], final: dict[str, Any]) -> None:
    if final.get("refused") is not True or final.get("citations") != []:
        raise RuntimeError("demo unsupported question was not refused")
    if any(event.get("event") == "answer_delta" for event in events):
        raise RuntimeError("demo refusal emitted answer text")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://demo.local",
        help="in-process TestClient base URL (no network connection is made)",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="assert the smoke test uses a transport that rejects every outbound request",
    )
    args = parser.parse_args()
    if not args.no_network:
        parser.error("--no-network is required for the public demo smoke test")
    result = run_demo_smoke(args.base_url)
    print(
        "indexed_documents="
        f"{result.indexed_documents} answered_questions={result.answered_questions} "
        f"refused_questions={result.refused_questions} network_calls={result.network_calls}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
