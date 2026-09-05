"""Tests for Commit and chill activity, wellness, and gamification services."""

from datetime import UTC, datetime

from src.gamification import calculate_points, get_badges
from src.github_client import Activity, GitHubClient, GitHubClientError
from src.notifications import build_chill_notifications, dismiss_notification
from src.productivity import (
    aggregate_activity,
    calculate_burnout_signal,
    filter_activity,
    healthy_streak,
)
from src.team import PrivacyThresholdError, aggregate_team_signals, team_signals_csv


def activities():
    return [
        Activity("commit", datetime(2026, 9, 5, 23, 0, tzinfo=UTC), "a/repo"),
        Activity("review", datetime(2026, 9, 6, 10, 0, tzinfo=UTC), "a/repo"),
        Activity("pull_request", datetime(2026, 9, 7, 10, 0, tzinfo=UTC), "a/repo"),
    ]


def test_activity_aggregation_and_filtering():
    items = activities()
    assert len(filter_activity(items, days=3, now=datetime(2026, 9, 8, tzinfo=UTC))) == 3
    metrics = aggregate_activity(items)
    assert metrics["late_night_count"] == 1
    assert metrics["weekend_count"] == 2
    assert healthy_streak(items) == 3


def test_burnout_signal_is_bounded_and_explained():
    result = calculate_burnout_signal(aggregate_activity(activities()))
    assert 0 <= result["score"] <= 1
    assert set(result["factors"]) == {"late_night_work", "weekend_work", "workload_volume"}
    assert "not a medical" in result["disclaimer"]


def test_points_badges_and_notifications():
    metrics = aggregate_activity(activities())
    points = calculate_points(metrics, breaks_taken=2)
    assert points > 0
    badges = get_badges(metrics, breaks_taken=2)
    assert any(badge["name"] == "Chill Master" for badge in badges)
    notifications = build_chill_notifications(metrics, calculate_burnout_signal(metrics))
    assert notifications
    assert dismiss_notification(notifications, notifications[0]["id"]) != notifications


def test_github_client_does_not_leak_token(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        class Response:
            status_code = 200
            headers = {}
            def json(self):
                return []
        return Response()

    monkeypatch.setattr("src.github_client.requests.get", fake_get)
    GitHubClient("secret-token").fetch_user_events("atreydev")
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert "secret-token" not in repr(captured).replace("Bearer secret-token", "")


def test_github_client_rejects_api_errors(monkeypatch):
    class Response:
        status_code = 404
        headers = {}
        def json(self):
            return {}
    monkeypatch.setattr("src.github_client.requests.get", lambda *args, **kwargs: Response())
    try:
        GitHubClient().fetch_user_events("missing")
    except GitHubClientError as error:
        assert "not found" in str(error)
    else:
        raise AssertionError("Expected GitHubClientError")


def test_team_aggregation_enforces_privacy_threshold_and_exports_only_aggregates():
    members = [
        {
            "username": f"person-{index}",
            "burnout_score": 0.2 + index / 100,
            "burnout_level": "Low",
            "activity": 4,
            "breaks_taken": 1,
        }
        for index in range(5)
    ]
    summary = aggregate_team_signals(members)
    csv_report = team_signals_csv(summary)

    assert summary["member_count"] == 5
    assert "person-0" not in csv_report
    assert "average_burnout_score" in csv_report

    try:
        aggregate_team_signals(members[:4])
    except PrivacyThresholdError:
        pass
    else:
        raise AssertionError("Expected small cohorts to be blocked")
