"""Cross-platform release helper invoked by the TangoPostureOpenCommunity Makefile."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_pyproject_version() -> str:
    pyproject_path = ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    version = str(data["project"]["version"]).strip()
    if not version:
        raise ValueError("project.version is empty in pyproject.toml")
    return version


def _normalize_version(raw: str | None) -> tuple[str, str]:
    version = (raw or _read_pyproject_version()).strip()
    version = version.removeprefix("v")
    if not version:
        raise ValueError("Version cannot be empty")
    return version, f"v{version}"


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    print(f"+ ({cwd}) {_format_cmd(cmd)}")
    if dry_run:
        return
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(cmd, cwd=str(cwd), env=merged_env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {_format_cmd(cmd)}")


def _capture(cmd: list[str], *, cwd: Path = ROOT) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"Command failed ({proc.returncode}): {_format_cmd(cmd)}\n{detail}")
    return proc.stdout.strip()


def _tag_exists(tag: str) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def run_checks(
    *,
    expected_version: str | None = None,
    allow_dirty: bool = False,
    dry_run: bool = False,
) -> tuple[str, str]:
    pyproject_ver, tag = _normalize_version(expected_version)
    actual_ver = _read_pyproject_version()
    if expected_version and pyproject_ver != actual_ver:
        raise ValueError(
            f"Requested version {pyproject_ver!r} does not match pyproject.toml version {actual_ver!r}."
        )

    status = _capture(["git", "status", "--porcelain"])
    if status and not allow_dirty and not dry_run:
        raise RuntimeError(
            "Working tree has uncommitted changes. Commit or stash before releasing "
            "(or pass --allow-dirty)."
        )

    _run(["uv", "run", "--frozen", "ruff", "check", "src", "tests"], dry_run=dry_run)
    _run(["uv", "run", "--frozen", "mypy", "src"], dry_run=dry_run)
    _run(
        ["uv", "run", "--frozen", "pytest", "-q"],
        env={"QT_QPA_PLATFORM": "offscreen"},
        dry_run=dry_run,
    )
    return actual_ver, tag


def cmd_check(args: argparse.Namespace) -> int:
    ver, tag = run_checks(
        expected_version=args.version,
        allow_dirty=args.allow_dirty,
        dry_run=args.dry_run,
    )
    print(f"Release checks passed for v{ver} ({tag}).")
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    ver, tag = _normalize_version(args.version)
    if not args.skip_checks:
        run_checks(
            expected_version=args.version,
            allow_dirty=args.allow_dirty,
            dry_run=args.dry_run,
        )
    else:
        actual_ver = _read_pyproject_version()
        if args.version and ver != actual_ver:
            raise ValueError(
                f"Requested version {ver!r} does not match pyproject.toml version {actual_ver!r}."
            )
        status = _capture(["git", "status", "--porcelain"])
        if status and not args.allow_dirty and not args.dry_run:
            raise RuntimeError(
                "Working tree has uncommitted changes. Commit or stash before releasing."
            )

    if not args.dry_run and shutil.which("gh") is None:
        raise RuntimeError(
            "GitHub CLI (`gh`) is required for `make release`. Install `gh` and run `gh auth login`."
        )

    target = args.target.strip() or _capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if not _tag_exists(tag):
        _run(
            ["git", "tag", "-a", tag, target, "-m", f"Release {tag}"],
            dry_run=args.dry_run,
        )
    else:
        print(f"Tag {tag} already exists locally; reusing existing tag.")

    _run(["git", "push", "origin", tag], dry_run=args.dry_run)

    gh_cmd = [
        "gh",
        "release",
        "create",
        tag,
        "--target",
        target,
        "--title",
        f"TangoPosture {tag}",
        "--generate-notes",
    ]
    if args.prerelease:
        gh_cmd.append("--prerelease")
    _run(gh_cmd, dry_run=args.dry_run)
    print(
        f"Published release {tag} (target={target}). GitHub Actions desktop-release.yml will "
        "build installers and dispatch PR updates to websites and Documentation."
    )
    return 0


def cmd_sync_local(args: argparse.Namespace) -> int:
    websites_dir = (ROOT.parent / "websites").resolve()
    docs_dir = (ROOT.parent / "Documentation").resolve()

    if websites_dir.is_dir():
        _run(
            ["node", "scripts/update-websites.mjs", "--tango-repo", str(ROOT)],
            cwd=websites_dir,
            dry_run=args.dry_run,
        )
        _run(["node", "scripts/validate.mjs"], cwd=websites_dir, dry_run=args.dry_run)
    else:
        print(f"Skipping websites sync; sibling directory not found: {websites_dir}")

    if docs_dir.is_dir():
        _run(
            [
                "poetry",
                "run",
                "docgen",
                "run",
                "--product",
                "tangoposture-community",
                "--provider",
                "mock",
                "--incremental",
            ],
            cwd=docs_dir,
            dry_run=args.dry_run,
        )
    else:
        print(f"Skipping Documentation sync; sibling directory not found: {docs_dir}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Release helper for TangoPostureOpenCommunity.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_p = subparsers.add_parser("check", help="Run release validation checks.")
    check_p.add_argument("--version", default=None, help="Expected version (e.g. 0.5.3 or v0.5.3).")
    check_p.add_argument("--allow-dirty", action="store_true", help="Allow uncommitted changes.")
    check_p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    check_p.set_defaults(func=cmd_check)

    pub_p = subparsers.add_parser("publish", help="Tag, push, and publish a GitHub Release.")
    pub_p.add_argument("--version", default=None, help="Version to release (defaults to pyproject).")
    pub_p.add_argument("--target", default="main", help="Git branch/commit target for release.")
    pub_p.add_argument(
        "--prerelease",
        action="store_true",
        help="Mark the GitHub release as a pre-release.",
    )
    pub_p.add_argument("--skip-checks", action="store_true", help="Skip ruff/mypy/pytest checks.")
    pub_p.add_argument("--allow-dirty", action="store_true", help="Allow uncommitted changes.")
    pub_p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    pub_p.set_defaults(func=cmd_publish)

    sync_p = subparsers.add_parser(
        "sync-local",
        help="Run local website and Documentation sync against sibling checkouts.",
    )
    sync_p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    sync_p.set_defaults(func=cmd_sync_local)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
