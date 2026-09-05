"""Small, token-safe GitHub REST client for Commit and chill."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests


@dataclass(frozen=True)
class Activity:
    """Normalized GitHub activity used by the dashboard and scoring engine."""

    kind: str
    timestamp: datetime
    repository: str
    title: str = ""
    url: str = ""
    additions: int = 0
    deletions: int = 0


class GitHubClientError(RuntimeError):
    """Raised when GitHub cannot return usable activity data."""


class GitHubClient:
    """Fetch public or authenticated GitHub activity without exposing tokens."""

    def __init__(self, token: str | None = None, *, timeout: float = 10.0) -> None:
        self._token = token.strip() if token else None
        self.timeout = timeout

    def _get(self, url: str, **params: Any) -> Any:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise GitHubClientError("GitHub is unavailable right now.") from exc
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            raise GitHubClientError("GitHub rate limit reached. Try again later or configure a token.")
        if response.status_code == 404:
            raise GitHubClientError("GitHub user or repository was not found.")
        if response.status_code >= 400:
            raise GitHubClientError(f"GitHub request failed with HTTP {response.status_code}.")
        return response.json()

    def fetch_user_events(self, username: str, *, pages: int = 2) -> list[Activity]:
        """Normalize recent public events for a user."""
        if not username.strip():
            raise ValueError("GitHub username is required")
        activities: list[Activity] = []
        for page in range(1, pages + 1):
            events = self._get(
                f"https://api.github.com/users/{username.strip()}/events",
                per_page=100,
                page=page,
            )
            if not events:
                break
            for event in events:
                activity = self._normalize_event(event)
                if activity:
                    activities.append(activity)
        return activities

    def fetch_repository_activity(self, owner: str, repository: str, *, pages: int = 1) -> list[Activity]:
        """Fetch commit activity for a public repository."""
        activities: list[Activity] = []
        for page in range(1, pages + 1):
            commits = self._get(
                f"https://api.github.com/repos/{owner.strip()}/{repository.strip()}/commits",
                per_page=100,
                page=page,
            )
            if not commits:
                break
            for item in commits:
                commit = item.get("commit", {})
                timestamp = _parse_timestamp(commit.get("author", {}).get("date"))
                if timestamp:
                    activities.append(Activity("commit", timestamp, f"{owner}/{repository}", commit.get("message", "").splitlines()[0], item.get("html_url", "")))
        return activities

    @staticmethod
    def _normalize_event(event: dict[str, Any]) -> Activity | None:
        timestamp = _parse_timestamp(event.get("created_at"))
        if not timestamp:
            return None
        repo = event.get("repo", {}).get("name", "")
        payload = event.get("payload", {})
        kind_map = {
            "PushEvent": "commit",
            "PullRequestEvent": "pull_request",
            "PullRequestReviewEvent": "review",
            "IssuesEvent": "issue",
        }
        kind = kind_map.get(event.get("type"))
        if not kind:
            return None
        title = payload.get("pull_request", {}).get("title") or payload.get("issue", {}).get("title", "")
        return Activity(kind, timestamp, repo, title, payload.get("pull_request", {}).get("html_url", ""))


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def demo_activities() -> list[Activity]:
    """Return deterministic offline activity for demos and screenshots."""
    from datetime import timedelta

    now = datetime.now(timezone.utc).replace(minute=30, second=0, microsecond=0)
    return [
        Activity("commit", now - timedelta(hours=2), "demo/commit-and-chill", "Improve dashboard"),
        Activity("pull_request", now - timedelta(days=1, hours=3), "demo/commit-and-chill", "Add chill mode"),
        Activity("review", now - timedelta(days=2, hours=1), "demo/commit-and-chill", "Review wellness metrics"),
        Activity("commit", now - timedelta(days=3, hours=5), "demo/commit-and-chill", "Add tests"),
        Activity("commit", now - timedelta(days=5, hours=11), "demo/commit-and-chill", "Refine README"),
    ]
