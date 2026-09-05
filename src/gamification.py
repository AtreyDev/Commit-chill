"""Healthy-work gamification for Commit and chill."""

from __future__ import annotations

from typing import Any


def calculate_points(metrics: dict[str, Any], *, breaks_taken: int = 0) -> int:
    """Award points for balanced activity, not raw hours."""
    points = int(metrics.get("by_kind", {}).get("review", 0)) * 15
    points += int(metrics.get("by_kind", {}).get("pull_request", 0)) * 10
    points += min(30, int(breaks_taken) * 10)
    if metrics.get("late_night_count", 0) == 0:
        points += 20
    if metrics.get("weekend_count", 0) == 0:
        points += 20
    return max(0, points)


def get_badges(metrics: dict[str, Any], *, breaks_taken: int = 0) -> list[dict[str, str]]:
    """Return earned personal badges with healthy-work framing."""
    badges = []
    if metrics.get("total", 0) and metrics.get("late_night_count", 0) == 0:
        badges.append({"name": "Early Bird", "description": "Worked inside healthy hours"})
    if breaks_taken >= 2:
        badges.append({"name": "Chill Master", "description": "Took two intentional breaks"})
    if metrics.get("by_kind", {}).get("review", 0) >= 2:
        badges.append({"name": "Team Player", "description": "Helped teammates through reviews"})
    if metrics.get("total", 0) >= 5:
        badges.append({"name": "Consistent Contributor", "description": "Made steady progress"})
    return badges
