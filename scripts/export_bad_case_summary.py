"""Export non-sensitive bad-case aggregates from the private SQLite store."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.storage.sqlite import SqliteDocumentRepository


def build_bad_case_summary(database: Path) -> dict[str, object]:
    """Build an aggregate-only report; never include message or file content."""

    repository = SqliteDocumentRepository(database)
    try:
        aggregates = repository.list_bad_case_aggregates()
    finally:
        repository.close()

    counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for code, statuses in aggregates.items():
        counts[code] = sum(statuses.values())
        for status, count in statuses.items():
            status_counts[status] = status_counts.get(status, 0) + count
    return {
        "schema_version": "bad-case-summary-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "counts": counts,
        "status_counts": status_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_bad_case_summary(args.database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
