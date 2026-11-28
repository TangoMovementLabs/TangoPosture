"""Dispatch TangoPostureOpenCommunity release events to websites and Documentation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

USER_AGENT = "TangoMovementLabs-Release-Dispatcher/1.0"
WEBSITES_REPO = "TangoMovementLabs/websites"
DOCUMENTATION_REPO = "TangoMovementLabs/Documentation"
DEFAULT_SOURCE_REPO = "TangoMovementLabs/TangoPostureOpenCommunity"
DEFAULT_PRODUCT_ID = "tangoposture-community"


def _parse_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_token(*env_names: str) -> str:
    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _build_targets(
    *,
    target: str,
    release_version: str,
    commit_sha: str,
    release_url: str,
    prerelease: bool,
    source_repo: str,
    product_id: str,
) -> list[tuple[str, str, tuple[str, ...], dict[str, Any]]]:
    specs: list[tuple[str, str, tuple[str, ...], dict[str, Any]]] = []

    if target in ("all", "websites"):
        specs.append(
            (
                "websites",
                WEBSITES_REPO,
                ("WEBSITES_DISPATCH_TOKEN", "RELEASE_DISPATCH_TOKEN", "GH_TOKEN"),
                {
                    "event_type": "tangoposture-release",
                    "client_payload": {
                        "product_id": product_id,
                        "repository": source_repo,
                        "release_version": release_version,
                        "commit_sha": commit_sha,
                        "release_url": release_url,
                        "prerelease": prerelease,
                    },
                },
            )
        )

    if target in ("all", "docs"):
        specs.append(
            (
                "docs",
                DOCUMENTATION_REPO,
                ("DOCS_DISPATCH_TOKEN", "RELEASE_DISPATCH_TOKEN", "GH_TOKEN"),
                {
                    "event_type": "source-release",
                    "client_payload": {
                        "product_id": product_id,
                        "repository": source_repo,
                        "release_version": release_version,
                        "commit_sha": commit_sha,
                        "release_url": release_url,
                    },
                },
            )
        )

    return specs


def _send_dispatch(repo: str, token: str, payload: dict[str, Any]) -> int:
    url = f"https://api.github.com/repos/{repo}/dispatches"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"GitHub repository_dispatch to {repo} failed (HTTP {exc.code}): {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub repository_dispatch to {repo} failed: {exc}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch repository_dispatch release events to websites and Documentation."
    )
    parser.add_argument(
        "--target",
        choices=("all", "websites", "docs"),
        default="all",
        help="Downstream target(s) to notify (default: all).",
    )
    parser.add_argument(
        "--release-version",
        default=os.environ.get("RELEASE_VERSION", "").strip(),
        help="Released tag or version (defaults to RELEASE_VERSION env).",
    )
    parser.add_argument(
        "--commit-sha",
        default=(
            os.environ.get("COMMIT_SHA", "").strip()
            or os.environ.get("GITHUB_SHA", "").strip()
        ),
        help="Commit SHA associated with the release.",
    )
    parser.add_argument(
        "--release-url",
        default=os.environ.get("RELEASE_URL", "").strip(),
        help="GitHub Release HTML URL.",
    )
    parser.add_argument(
        "--prerelease",
        default=os.environ.get("RELEASE_PRERELEASE", "false").strip(),
        help="Whether the release is a pre-release (true/false).",
    )
    parser.add_argument(
        "--source-repo",
        default=(
            os.environ.get("SOURCE_REPO", "").strip()
            or os.environ.get("GITHUB_REPOSITORY", "").strip()
            or DEFAULT_SOURCE_REPO
        ),
        help="Source repository owner/name.",
    )
    parser.add_argument(
        "--product-id",
        default=os.environ.get("PRODUCT_ID", "").strip() or DEFAULT_PRODUCT_ID,
        help="Product ID configured in Documentation/config/repositories.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target URLs and payloads without sending HTTP requests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    release_version = args.release_version.strip()
    if not release_version:
        print(
            "Error: --release-version (or RELEASE_VERSION env) is required.",
            file=sys.stderr,
        )
        return 2

    source_repo = args.source_repo.strip() or DEFAULT_SOURCE_REPO
    release_url = (
        args.release_url.strip()
        or f"https://github.com/{source_repo}/releases/tag/{release_version}"
    )
    prerelease = _parse_bool(args.prerelease)

    specs = _build_targets(
        target=args.target,
        release_version=release_version,
        commit_sha=args.commit_sha.strip(),
        release_url=release_url,
        prerelease=prerelease,
        source_repo=source_repo,
        product_id=args.product_id.strip() or DEFAULT_PRODUCT_ID,
    )

    for label, repo, token_envs, payload in specs:
        url = f"https://api.github.com/repos/{repo}/dispatches"
        if args.dry_run:
            print(f"[dry-run] {label} -> POST {url}")
            print(json.dumps(payload, indent=2))
            continue

        token = _resolve_token(*token_envs)
        if not token:
            env_list = ", ".join(token_envs)
            print(
                f"Error: missing dispatch token for {label} ({repo}). "
                f"Configure one of: {env_list}.",
                file=sys.stderr,
            )
            return 1

        status = _send_dispatch(repo, token, payload)
        print(
            f"Dispatched {payload['event_type']} to {repo} for {release_version} (HTTP {status})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
