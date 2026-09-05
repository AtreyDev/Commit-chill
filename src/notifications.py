"""In-app Chill suggestions based on workload signals."""

from __future__ import annotations

from typing import Any


def build_chill_notifications(metrics: dict[str, Any], burnout: dict[str, Any]) -> list[dict[str, str]]:
    """Create gentle, actionable notifications without claiming medical certainty."""
    notifications: list[dict[str, str]] = []
    if metrics.get("late_night_count", 0):
        notifications.append({"id": "late-night", "title": "Protect your evening", "message": "You had late-night activity. Plan a clean stop-work time today."})
    if metrics.get("weekend_count", 0):
        notifications.append({"id": "weekend", "title": "Make room for recovery", "message": "Weekend activity showed up in your window. Consider a coding-free block."})
    if burnout.get("level") == "High":
        notifications.append({"id": "high-load", "title": "Time to chill", "message": "Your recent workload signal is high. Take a five-minute reset before your next task."})
    notifications.append({"id": "break", "title": "Two-minute reset", "message": "Unclench your shoulders, look away from the screen, and breathe slowly."})
    return notifications


def dismiss_notification(notifications: list[dict[str, str]], notification_id: str) -> list[dict[str, str]]:
    """Return notifications with one local notification dismissed."""
    return [item for item in notifications if item.get("id") != notification_id]
