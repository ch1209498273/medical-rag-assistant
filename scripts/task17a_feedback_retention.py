"""Delete expired Task17A projection rows and record counts only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_ROOT = _PROJECT_ROOT / "backend"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.feedback.repository import (
    FeedbackProjectionRepository,
    validate_private_feedback_path,
)


def _purge(args: argparse.Namespace) -> int:
    database = validate_private_feedback_path(Path(args.database))
    try:
        as_of_date = date.fromisoformat(args.as_of)
    except ValueError as error:
        raise ValueError("as-of must be YYYY-MM-DD") from error
    as_of = datetime.combine(as_of_date, time.min, tzinfo=timezone.utc)
    repository = FeedbackProjectionRepository(database)
    try:
        summary = repository.delete_expired(as_of)
    finally:
        repository.close()
    payload = summary.to_dict()
    payload["operation"] = "purge"
    payload["network_calls"] = 0
    (database.parent / "retention-log.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task17A private feedback retention")
    subparsers = parser.add_subparsers(dest="command", required=True)
    purge = subparsers.add_parser("purge", help="delete expired projection rows")
    purge.add_argument("--database", required=True)
    purge.add_argument("--as-of", required=True)
    purge.set_defaults(handler=_purge)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as error:  # noqa: BLE001 - never print source text
        category = "private_path_or_contract" if isinstance(error, ValueError) else "runtime"
        print(f"error_type={type(error).__name__} category={category}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
