"""Transparent productivity and non-clinical workload signals."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

from src.github_client import Activity


def filter_activity(activities: Iterable[Activity], *, days: int = 30, now: datetime | None = None) -> list[Activity]:
    """Keep activity inside a UTC rolling window."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    return [item for item in activities if cutoff <= item.timestamp <= now]


def aggregate_activity(activities: Iterable[Activity]) -> dict[str, Any]:
    """Aggregate activities for charts and explainable workload metrics."""
    items = list(activities)
    by_day = Counter(item.timestamp.date().isoformat() for item in items)
    by_kind = Counter(item.kind for item in items)
    late_night = [item for item in items if item.timestamp.hour < 6 or item.timestamp.hour >= 22]
    weekend = [item for item in items if item.timestamp.weekday() >= 5]
    return {
        "total": len(items),
        "by_day": dict(sorted(by_day.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "late_night_count": len(late_night),
        "weekend_count": len(weekend),
        "repositories": sorted({item.repository for item in items}),
        "activities": items,
    }


def healthy_streak(activities: Iterable[Activity], *, max_daily_hours: int = 10) -> int:
    """Count consecutive active days ending at the latest activity day."""
    days = sorted({item.timestamp.date() for item in activities})
    if not days:
        return 0
    streak = 1
    for current, previous in zip(reversed(days), reversed(days[:-1])):
        if current - previous != timedelta(days=1):
            break
        streak += 1
    return streak


def calculate_burnout_signal(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded workload signal, never a medical diagnosis."""
    total = max(1, int(metrics.get("total", 0)))
    factors = {
        "late_night_work": min(1.0, metrics.get("late_night_count", 0) / max(3, total * 0.25)),
        "weekend_work": min(1.0, metrics.get("weekend_count", 0) / max(3, total * 0.25)),
        "workload_volume": min(1.0, total / 40),
    }
    score = round(sum(factors.values()) / len(factors), 4)
    level = "Low" if score < 0.34 else "Moderate" if score < 0.67 else "High"
    return {
        "score": score,
        "level": level,
        "factors": {key: round(value, 4) for key, value in factors.items()},
        "disclaimer": "This is a workload signal, not a medical or psychological diagnosis.",
    }


def chart_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert daily activity counts into Streamlit/Plotly-friendly rows."""
    return [{"date": day, "activity": count} for day, count in metrics.get("by_day", {}).items()]
