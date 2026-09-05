"""Privacy-preserving team aggregates for Commit and chill."""

from __future__ import annotations

import csv
import io
from collections import Counter
from collections.abc import Iterable
from typing import Any


class PrivacyThresholdError(ValueError):
    """Raised when a team cohort is too small to report safely."""


def aggregate_team_signals(
    members: Iterable[dict[str, Any]], *, minimum_members: int = 5
) -> dict[str, Any]:
    """Return aggregate signals without exposing member identities or raw activity."""
    records = list(members)
    if minimum_members < 2:
        raise ValueError("minimum_members must be at least 2")
    if len(records) < minimum_members:
        raise PrivacyThresholdError(
            f"Team insights require at least {minimum_members} members."
        )

    scores = [float(record.get("burnout_score", 0.0)) for record in records]
    levels = Counter(str(record.get("burnout_level", "Unknown")) for record in records)
    return {
        "member_count": len(records),
        "average_burnout_score": round(sum(scores) / len(scores), 4),
        "burnout_levels": dict(sorted(levels.items())),
        "total_activity": sum(int(record.get("activity", 0)) for record in records),
        "total_breaks": sum(int(record.get("breaks_taken", 0)) for record in records),
        "disclaimer": (
            "Aggregate workload trends only; no individual-level monitoring or diagnosis."
        ),
    }


def team_signals_csv(summary: dict[str, Any]) -> str:
    """Serialize an aggregate summary without member names or identifiers."""
    fields = [
        "member_count",
        "average_burnout_score",
        "total_activity",
        "total_breaks",
        "disclaimer",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerow({field: summary.get(field, "") for field in fields})
    return output.getvalue()