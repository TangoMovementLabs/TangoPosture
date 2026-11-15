"""Coach and movement companion persistence and report lookups."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..movement import validate_movement_analysis
from ..pose.recovery import validate_pose_recovery_payload
from ..review import ReportRecord

logger = logging.getLogger(__name__)


def save_movement_companion(path: Path, data: dict[str, Any]) -> Path:
    """Save a validated movement analysis JSON companion file."""
    validate_movement_analysis(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_movement_companion(path: Path) -> dict[str, Any] | None:
    """Load and validate a movement analysis JSON companion file if it exists."""
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return validate_movement_analysis(raw)
    except (OSError, ValueError, TypeError):
        return None


def save_coach_companion(path: Path, data: dict[str, Any]) -> Path:
    """Save a per-frame Open Practice Coach overlay and insights companion file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def save_couple_companion(path: Path, data: dict[str, Any]) -> Path:
    """Save a validated Couple analysis JSON companion file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_couple_companion(path: Path) -> dict[str, Any] | None:
    """Load a Couple analysis JSON companion file if it exists and is valid."""
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1 or not raw.get("run_id"):
            return None
        return raw
    except (OSError, ValueError, TypeError):
        return None


def load_coach_companion(path: Path) -> dict[str, Any] | None:
    """Load a per-frame Open Practice Coach companion file if it exists and is valid."""
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("frames"), dict):
            return None
        if raw.get("pose_recovery") is not None:
            frames = raw["frames"]
            valid_indices = {int(k) for k in frames if str(k).isdigit()} if frames else None
            dancer_sel = raw.get("dancer_selection") if isinstance(raw.get("dancer_selection"), dict) else None
            try:
                validate_pose_recovery_payload(
                    raw["pose_recovery"],
                    valid_frame_indices=valid_indices,
                    source_sha256=raw.get("source_sha256"),
                    model_sha256=raw.get("model_hash"),
                    source_width=raw.get("width"),
                    source_height=raw.get("height"),
                    dancer_selection=dancer_sel,
                )
            except (ValueError, TypeError) as exc:
                logger.warning("Ignoring invalid or stale pose_recovery in %s: %s", path.name, exc)
                raw = dict(raw)
                raw.pop("pose_recovery", None)
        return raw
    except (OSError, ValueError, TypeError):
        return None


def load_report_coach_data(report: ReportRecord) -> dict[str, Any] | None:
    """Load the per-frame coach overlay & insights JSON associated with a report if valid."""
    if report.relinked_content_changed:
        return None
    candidates: list[Path] = []
    coach_file = getattr(report, "coach_overlay_file", None)
    if coach_file:
        candidates.append(report.path.parent / coach_file)
    candidates.append(report.path.parent / f"{report.path.stem}.coach.v1.json")
    for cand in candidates:
        data = load_coach_companion(cand)
        if data is not None:
            if report.source_sha256 and data.get("source_sha256") != report.source_sha256:
                continue
            if data.get("pose_recovery") is not None:
                timestamps = list(report.timestamps or [])
                frames = data.get("frames") or {}
                valid_indices = {int(k) for k in frames if str(k).isdigit()} if frames else None
                try:
                    validate_pose_recovery_payload(
                        data["pose_recovery"],
                        frame_count=len(timestamps) if timestamps else None,
                        valid_frame_indices=valid_indices,
                        timestamps=[float(t) for t in timestamps] if timestamps else None,
                        source_sha256=report.source_sha256 or data.get("source_sha256"),
                        model_sha256=data.get("model_hash"),
                        source_width=data.get("width"),
                        source_height=data.get("height"),
                    )
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "Ignoring invalid or stale pose_recovery for report %s: %s",
                        report.path.name,
                        exc,
                    )
                    data = dict(data)
                    data.pop("pose_recovery", None)
            return data
    return None


def load_report_movement_data(report: ReportRecord) -> dict[str, Any] | None:
    """Load the movement analysis JSON associated with a report if valid."""
    if report.relinked_content_changed:
        return None
    candidates: list[Path] = []
    if report.movement_analysis_file:
        candidates.append(report.path.parent / report.movement_analysis_file)
    candidates.append(report.path.parent / f"{report.path.stem}.movement.v1.json")
    for cand in candidates:
        data = load_movement_companion(cand)
        if data is not None:
            if report.source_sha256 and data.get("source_sha256") != report.source_sha256:
                continue
            return data
    return None
