"""Shared pre-flight context and routing utilities for the Solo and Couple engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..calibration.readiness import CalibrationReadiness
from ..config import Config
from ..couple import AutoDecisionResult, CoupleAnalysisRequest
from ..domain import METRIC_VERSION, Calibration, FrameObservation
from ..references.results import ClipAnalysisResult, RunManifest
from ..references.soundtrack import SoundtrackAnalysisSummary
from ..references.video import VideoProbeInfo


@dataclass(frozen=True)
class CommonEngineContext:
    """Immutable pre-flight outputs shared by both the Solo and Couple engines."""

    probe: VideoProbeInfo
    cfg: Config
    model_path: Path
    model_hash: str
    source_sha: str
    clip_in: float
    clip_out: float
    clip_duration: float
    clean_title: str
    cap_profile: dict[str, Any]
    readiness: CalibrationReadiness
    decoded_frames: list[tuple[int, float, float, FrameObservation]]
    raw_observations: list[tuple[int, FrameObservation]]
    auto_observations: list[tuple[int, FrameObservation]]
    frame_times: list[tuple[int, float, float]]
    auto_decision: AutoDecisionResult
    requested_mode: str
    resolved_mode: str
    request_obj: CoupleAnalysisRequest | None
    performance: dict[str, Any]
    processed_fps: float | None
    analysis_mode: str
    soundtrack_summary: SoundtrackAnalysisSummary | None = None


def extract_soundtrack_beat_timestamps(
    soundtrack_summary: SoundtrackAnalysisSummary,
    frame_times: list[tuple[int, float, float]],
) -> list[float]:
    """Return sorted source timestamps marking soundtrack beat transitions."""
    beat_timestamps: list[float] = []
    if not (soundtrack_summary.enabled and soundtrack_summary.frame_states):
        return beat_timestamps
    prev_beat: int | None = None
    for frame_index, _, _ in frame_times:
        state = soundtrack_summary.frame_states.get(frame_index)
        if not state or not state.get("playing"):
            prev_beat = None
            continue
        beat_num = state.get("beat")
        if isinstance(beat_num, int) and beat_num > 0 and beat_num != prev_beat:
            src_time = state.get("source_time")
            if src_time is not None:
                beat_timestamps.append(float(src_time))
        prev_beat = beat_num if isinstance(beat_num, int) and beat_num > 0 else None
    return beat_timestamps


def build_needs_selection_result(
    ctx: CommonEngineContext,
    *,
    reference: Calibration | None,
    calibration_interval: tuple[float, float] | None,
    effective_soundtrack: bool,
) -> ClipAnalysisResult:
    """Return an early selection-required result when Auto mode is ambiguous."""
    manifest = RunManifest(
        model_variant=ctx.cfg.analysis_model,
        model_hash=ctx.model_hash,
        metric_version=METRIC_VERSION,
        capture_profile=ctx.cap_profile,
        source_sha256=ctx.source_sha,
        source_filename=ctx.probe.path.name,
        width=ctx.probe.width,
        height=ctx.probe.height,
        fps=ctx.probe.fps,
        selected_in=ctx.clip_in,
        selected_out=ctx.clip_out,
        clip_duration=ctx.clip_duration,
        analysis_mode=ctx.analysis_mode,
        processed_fps=ctx.processed_fps,
        requested_dancer_mode=ctx.requested_mode,
        resolved_dancer_mode="needs_selection",
        detector_version="couple_auto_v1",
    )
    return ClipAnalysisResult(
        manifest=manifest,
        video_path=ctx.probe.path,
        title=ctx.clean_title,
        source_sha256=ctx.source_sha,
        selected_in=ctx.clip_in,
        selected_out=ctx.clip_out,
        clip_duration=ctx.clip_duration,
        analysis_mode=ctx.analysis_mode,
        readiness=ctx.readiness,
        reference=reference if ctx.analysis_mode == "posture" else None,
        calibration_window=(
            (float(calibration_interval[0]), float(calibration_interval[1]))
            if ctx.analysis_mode == "posture" and calibration_interval is not None
            else None
        ),
        coach_data={},
        movement_data={},
        soundtrack=SoundtrackAnalysisSummary(
            enabled=effective_soundtrack,
            available=False,
            status="disabled",
            reason="Stopped for manual Single/Couple selection",
        ),
        coverage={},
        frame_log=[],
        scored_frame_log=[],
        frame_observations=[],
        performance=ctx.performance,
        requested_dancer_mode=ctx.requested_mode,
        resolved_dancer_mode="needs_selection",
        auto_decision=ctx.auto_decision.to_dict(),
        couple_result=None,
        selection_required=True,
    )
