"""Typed results of reference clip analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..calibration.readiness import CalibrationReadiness
from ..domain import (
    Calibration,
    FrameObservation,
)
from .coverage import FamilyCoverage
from .links import FacebookReelSourceLink, YouTubeClockAlignment, YouTubeSourceLink
from .soundtrack import SoundtrackAnalysisSummary


@dataclass(frozen=True)
class RunManifest:
    """Immutable provenance and execution metadata for an offline clip analysis run."""

    model_variant: str
    model_hash: str
    metric_version: str
    capture_profile: dict[str, Any]
    source_sha256: str
    source_filename: str
    width: int
    height: int
    fps: float
    selected_in: float
    selected_out: float
    clip_duration: float
    analysis_mode: str
    processed_fps: float | None = None


@dataclass(frozen=True)
class ClipAnalysisResult:
    """In-memory result of analyzing a local video clip without database or report side effects."""

    manifest: RunManifest
    video_path: Path
    title: str
    source_sha256: str
    selected_in: float
    selected_out: float
    clip_duration: float
    analysis_mode: str
    readiness: CalibrationReadiness
    reference: Calibration | None
    calibration_window: tuple[float, float] | None
    coach_data: dict[str, Any]
    movement_data: dict[str, Any]
    soundtrack: SoundtrackAnalysisSummary
    coverage: dict[str, FamilyCoverage]
    frame_log: list[dict[str, Any]]
    scored_frame_log: list[dict[str, Any]]
    frame_observations: list[tuple[int, float, float, FrameObservation]]
    performance: dict[str, Any]
    session_id: str | None = None
    session_summary: dict[str, Any] | None = None
    session_events: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class ImportedClipResult:
    """Complete result of importing and analysing a reference video clip."""

    report_path: Path
    metadata_path: Path
    movement_path: Path
    title: str
    video_path: Path
    source_sha256: str
    selected_in: float
    selected_out: float
    analysis_mode: str
    readiness: CalibrationReadiness
    reference: Calibration | None
    movement_data: dict[str, Any]
    soundtrack: SoundtrackAnalysisSummary
    coverage: dict[str, FamilyCoverage]
    youtube_link: YouTubeSourceLink | None = None
    youtube_alignment: YouTubeClockAlignment | None = None
    facebook_reel: FacebookReelSourceLink | None = None
    coach_path: Path | None = None
    coach_data: dict[str, Any] | None = None
