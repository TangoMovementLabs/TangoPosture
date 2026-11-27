"""Send GitHub Actions notifications for PRs, releases, and pipelines to Discord."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

USER_AGENT = "TangoPosture-CI-Notifier/1.0"
MAX_TITLE_LEN = 256
MAX_DESC_LEN = 4096
MAX_RELEASE_DESC_LEN = 1000
MAX_FIELD_NAME_LEN = 256
MAX_FIELD_VALUE_LEN = 1024
MAX_RETRY_WAIT_SECONDS = 5.0

COLOR_BLUE = 0x3498DB
COLOR_TEAL = 0x1ABC9C
COLOR_PURPLE = 0x9B59B6
COLOR_GRAY = 0x95A5A6
COLOR_GOLD = 0xF1C40F
COLOR_RED = 0xE74C3C
COLOR_ORANGE = 0xE67E22
COLOR_GREEN = 0x2ECC71

JOB_STATUS_ICONS = {
    "success": "\u2705",
    "failure": "\u274c",
    "cancelled": "\u26a0\ufe0f",
    "skipped": "\u23ed\ufe0f",
}
DEFAULT_JOB_ICON = "\u2022"


def truncate(text: str, limit: int) -> str:
    """Truncate text to at most `limit` characters, appending ellipsis when cut."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def make_field(name: str, value: str, *, inline: bool = True) -> dict[str, Any]:
    """Create a Discord embed field clamped to API limits."""
    clean_value = value.strip() or "-"
    return {
        "name": truncate(name.strip() or "Field", MAX_FIELD_NAME_LEN),
        "value": truncate(clean_value, MAX_FIELD_VALUE_LEN),
        "inline": inline,
    }


def load_event_payload() -> dict[str, Any]:
    """Load the GitHub Actions event JSON payload if GITHUB_EVENT_PATH is present."""
    event_path = os.environ.get("GITHUB_EVENT_PATH", "").strip()
    if not event_path:
        return {}
    path = Path(event_path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO-8601 Z format."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_pull_request_embed(event: dict[str, Any], repo: str, actor: str) -> dict[str, Any]:
    """Build a Discord embed for pull_request webhook events."""
    action = str(event.get("action", "")).strip()
    pr = event.get("pull_request")
    if not isinstance(pr, dict):
        pr = {}

    number = pr.get("number", "?")
    pr_title = str(pr.get("title", "")).strip() or "Untitled"
    merged = bool(pr.get("merged", False))
    draft = bool(pr.get("draft", False))

    if action == "opened":
        color = COLOR_BLUE
        title = f"PR #{number} opened: {pr_title}"
    elif action == "reopened":
        color = COLOR_BLUE
        title = f"PR #{number} reopened: {pr_title}"
    elif action == "ready_for_review":
        color = COLOR_TEAL
        title = f"PR #{number} ready for review: {pr_title}"
    elif action == "closed" and merged:
        color = COLOR_PURPLE
        title = f"PR #{number} merged: {pr_title}"
    elif action == "closed":
        color = COLOR_GRAY
        title = f"PR #{number} closed: {pr_title}"
    else:
        color = COLOR_BLUE
        label = action or "updated"
        title = f"PR #{number} {label}: {pr_title}"

    head_ref = str((pr.get("head") or {}).get("ref", "")).strip() or "unknown"
    base_ref = str((pr.get("base") or {}).get("ref", "")).strip() or "unknown"
    author = str((pr.get("user") or {}).get("login", "")).strip() or actor or "unknown"

    fields: list[dict[str, Any]] = [
        make_field("Branches", f"`{head_ref}` -> `{base_ref}`", inline=True),
        make_field("Author", author, inline=True),
    ]
    if merged:
        merged_by = str((pr.get("merged_by") or {}).get("login", "")).strip() or actor
        if merged_by:
            fields.append(make_field("Merged by", merged_by, inline=True))
    elif draft and action in {"opened", "reopened"}:
        fields.append(make_field("Status", "Draft", inline=True))

    embed: dict[str, Any] = {
        "title": truncate(title, MAX_TITLE_LEN),
        "color": color,
        "fields": fields,
        "footer": {"text": repo or "TangoPosture"},
        "timestamp": utc_timestamp(),
    }
    html_url = str(pr.get("html_url", "")).strip()
    if html_url:
        embed["url"] = html_url
    return embed


def build_release_embed(event: dict[str, Any], repo: str, actor: str) -> dict[str, Any]:
    """Build a Discord embed for published GitHub release events."""
    release = event.get("release")
    if not isinstance(release, dict):
        release = {}

    tag_name = str(release.get("tag_name", "")).strip() or "unknown"
    rel_name = str(release.get("name", "")).strip()
    if rel_name and rel_name != tag_name:
        title = f"Release {tag_name}: {rel_name}"
    else:
        title = f"Release {tag_name}"

    body = str(release.get("body") or "").strip()
    target = str(release.get("target_commitish", "")).strip() or "main"
    author = str((release.get("author") or {}).get("login", "")).strip() or actor or "unknown"
    release_type = "Pre-release" if release.get("prerelease") else "Stable"

    fields = [
        make_field("Tag", f"`{tag_name}`", inline=True),
        make_field("Target", f"`{target}`", inline=True),
        make_field("Author", author, inline=True),
        make_field("Type", release_type, inline=True),
    ]

    embed: dict[str, Any] = {
        "title": truncate(title, MAX_TITLE_LEN),
        "color": COLOR_GOLD,
        "fields": fields,
        "footer": {"text": repo or "TangoPosture"},
        "timestamp": utc_timestamp(),
    }
    if body:
        embed["description"] = truncate(truncate(body, MAX_RELEASE_DESC_LEN), MAX_DESC_LEN)
    html_url = str(release.get("html_url", "")).strip()
    if html_url:
        embed["url"] = html_url
    return embed


def format_event_label(event_name: str, event: dict[str, Any]) -> str:
    """Format the triggering GitHub event with PR number or release tag when available."""
    if event_name == "pull_request":
        pr = event.get("pull_request")
        if isinstance(pr, dict) and pr.get("number") is not None:
            return f"pull_request (#{pr['number']})"
    elif event_name == "release":
        release = event.get("release")
        if isinstance(release, dict) and release.get("tag_name"):
            return f"release ({release['tag_name']})"
    return event_name or "unknown"


def build_pipeline_embed(event: dict[str, Any], repo: str, actor: str) -> dict[str, Any]:
    """Build a Discord embed summarizing CI/CD pipeline job outcomes."""
    workflow = os.environ.get("GITHUB_WORKFLOW", "").strip() or "CI/CD Pipeline"
    event_name = os.environ.get("GITHUB_EVENT_NAME", "").strip() or "unknown"
    run_id = os.environ.get("GITHUB_RUN_ID", "").strip()
    sha = os.environ.get("GITHUB_SHA", "").strip()
    head_ref = os.environ.get("GITHUB_HEAD_REF", "").strip()
    ref_name = os.environ.get("GITHUB_REF_NAME", "").strip()

    raw_jobs = [
        ("checks", os.environ.get("JOB_CHECKS_RESULT", "").strip().lower()),
        ("build", os.environ.get("JOB_BUILD_RESULT", "").strip().lower()),
        ("publish", os.environ.get("JOB_PUBLISH_RESULT", "").strip().lower()),
    ]
    active_results = [result for _, result in raw_jobs if result and result != "skipped"]

    if any(result == "failure" for result in active_results):
        color = COLOR_RED
        title = f"Pipeline Failed: {workflow}"
    elif any(result == "cancelled" for result in active_results):
        color = COLOR_ORANGE
        title = f"Pipeline Cancelled: {workflow}"
    else:
        color = COLOR_GREEN
        title = f"Pipeline Succeeded: {workflow}"

    job_lines = [
        f"{JOB_STATUS_ICONS.get(result, DEFAULT_JOB_ICON)} **{name}**: `{result}`"
        for name, result in raw_jobs
        if result
    ]
    jobs_summary = "\n".join(job_lines) if job_lines else "No job results reported"

    branch_or_ref = head_ref or ref_name or "unknown"
    commit_short = sha[:7] if sha else "unknown"

    fields = [
        make_field("Event", format_event_label(event_name, event), inline=True),
        make_field("Branch / Ref", f"`{branch_or_ref}`", inline=True),
        make_field("Commit", f"`{commit_short}`", inline=True),
        make_field("Triggered by", actor or "unknown", inline=True),
        make_field("Jobs", jobs_summary, inline=False),
    ]

    embed: dict[str, Any] = {
        "title": truncate(title, MAX_TITLE_LEN),
        "color": color,
        "fields": fields,
        "footer": {"text": repo or "TangoPosture"},
        "timestamp": utc_timestamp(),
    }
    if repo and run_id:
        embed["url"] = f"https://github.com/{repo}/actions/runs/{run_id}"
    return embed


def build_payload(mode: str, event: dict[str, Any]) -> dict[str, Any]:
    """Build the full Discord webhook JSON payload for the given notification mode."""
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip() or "TangoMovementLabs/TangoPostureOpenCommunity"
    actor = os.environ.get("GITHUB_ACTOR", "").strip()

    if mode == "pull_request":
        embed = build_pull_request_embed(event, repo, actor)
    elif mode == "release":
        embed = build_release_embed(event, repo, actor)
    elif mode == "pipeline":
        embed = build_pipeline_embed(event, repo, actor)
    else:
        raise ValueError(f"Unsupported notification mode: {mode}")

    return {"embeds": [embed]}


def parse_retry_after(headers: Any, default_wait: float = 2.0) -> float:
    """Parse Retry-After header seconds, clamped to MAX_RETRY_WAIT_SECONDS."""
    if headers is None:
        return default_wait
    raw = headers.get("Retry-After")
    if raw is None:
        return default_wait
    try:
        seconds = float(raw)
    except ValueError:
        return default_wait
    return max(0.0, min(seconds, MAX_RETRY_WAIT_SECONDS))


def post_webhook(webhook_url: str, payload: dict[str, Any]) -> None:
    """POST the JSON payload to Discord, retrying once on 429 or 5xx without failing CI."""
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    for attempt in range(2):
        request = urllib.request.Request(webhook_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                if 200 <= response.status < 300:
                    return
                print(
                    f"::warning::Discord webhook returned unexpected status {response.status}.",
                    file=sys.stderr,
                )
                return
        except urllib.error.HTTPError as exc:
            should_retry = attempt == 0 and (exc.code == 429 or 500 <= exc.code < 600)
            if should_retry:
                wait_seconds = parse_retry_after(exc.headers)
                time.sleep(wait_seconds)
                continue
            print(
                f"::warning::Discord notification failed with HTTP {exc.code}: {exc.reason}",
                file=sys.stderr,
            )
            return
        except urllib.error.URLError as exc:
            print(
                f"::warning::Discord notification failed due to network error: {exc.reason}",
                file=sys.stderr,
            )
            return


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Send GitHub Actions notifications to Discord.")
    parser.add_argument(
        "mode",
        choices=("pull_request", "release", "pipeline"),
        help="Notification type to send.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Discord JSON payload to stdout without sending an HTTP request.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI execution."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dry_run = args.dry_run or os.environ.get("DRY_RUN", "").strip() == "1"
    webhook_url = (
        os.environ.get("DISCORD_CICD_WEBHOOK_URL")
        or os.environ.get("DISCORD_WEBHOOK_URL")
        or ""
    ).strip()

    if not dry_run and not webhook_url:
        print(
            "::warning::DISCORD_CICD_WEBHOOK_URL is not configured; "
            "skipping Discord notification.",
            file=sys.stderr,
        )
        return 0

    event = load_event_payload()
    payload = build_payload(args.mode, event)

    if dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    post_webhook(webhook_url, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
