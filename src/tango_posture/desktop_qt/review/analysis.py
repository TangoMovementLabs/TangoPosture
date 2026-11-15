"""Background posture analysis for Qt review playback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from ...config import Config
from ...movement import render_floor_guide, render_movement_overlay
from ...references.coach import render_coach_frame_overlay
from ...references.errors import ImportCancelledError
from ...references.pipeline import analyze_clip
from ...references.results import ClipAnalysisResult, RunManifest
from ...replay import VideoReader, save_overlay_video
from ...reports import render_markdown_report


@dataclass(frozen=True)
class ReviewAnalysis:
    coach_data: dict[str, Any]
    movement_data: dict[str, Any]
    manifest: RunManifest | None = None
    result: ClipAnalysisResult | None = None


def analyse_review_clip(
    video_path: Path,
    cfg: Config,
    *,
    selected_out: float,
    progress_callback: Callable[[int, int, str], None],
    cancel_check: Callable[[], bool],
) -> ReviewAnalysis:
    """Analyze a local clip in memory without persisting a temporary report or database."""
    result = analyze_clip(
        video_path=video_path,
        cfg=cfg,
        selected_in=0.0,
        selected_out=selected_out,
        analysis_mode="movement",
        analyse_soundtrack=False,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    if result.coach_data is None:
        raise ValueError("Posture analysis completed without frame-level coaching data")
    return ReviewAnalysis(result.coach_data, result.movement_data, result.manifest, result)


def render_review_report(
    result: ClipAnalysisResult, cfg: Config, index: int, provenance: dict[str, Any],
) -> str:
    movement = result.movement_data
    context = {
        "is_imported_reference": True,
        "title": provenance.get("title") or result.title,
        "source_filename": provenance.get("source_filename") or result.manifest.source_filename,
        "source_sha256": result.source_sha256,
        "clip_in": 0.0,
        "clip_out": result.selected_out,
        "clip_duration": result.clip_duration,
        "analysis_mode": result.analysis_mode,
        "readiness": result.readiness.to_dict(),
        "youtube_link": provenance.get("youtube_link"),
        "facebook_reel": provenance.get("facebook_reel"),
        "coverage": {key: value.to_dict() for key, value in result.coverage.items()},
        "soundtrack": result.soundtrack.to_metadata_dict(),
        "pivot_count": len(movement.get("pivots", [])),
        "step_count": len(movement.get("steps", [])),
    }
    return render_markdown_report(
        cfg=cfg, source=f"import:{context['source_filename']}", status="completed",
        frame_log=result.scored_frame_log, performance=result.performance,
        reference=result.reference, session_id=result.session_id,
        session_summary=result.session_summary, session_events=result.session_events,
        recorded_video=Path(f"video_{index:02d}.mp4"),
        recorded_overlay=Path(f"overlay_{index:02d}.mp4"),
        import_context=context, movement_data=movement,
    )


def render_review_overlays(
    video: Path, folder: Path, analysis: ReviewAnalysis, cfg: Config,
    cancel_check: Callable[[], bool],
) -> tuple[Path, Path]:
    reader = VideoReader(video)
    try:
        count = reader.count
    finally:
        reader.close()
    frames = analysis.coach_data.get("frames")
    if not isinstance(frames, dict) or set(frames) != {str(index) for index in range(count)}:
        raise ValueError("Analysis did not cover every video frame; archive was not changed")
    if analysis.result is None or len(analysis.result.frame_log) != count:
        raise ValueError("Analysis has an incomplete frame timeline; archive was not changed")

    def render(image: Any, index: int, fps: float, *, floor: bool) -> Any:
        if cancel_check():
            raise ImportCancelledError("Posture analysis canceled")
        if floor:
            image = render_floor_guide(image)
        image = render_coach_frame_overlay(image, analysis.coach_data, index, cfg, lines_only=True)
        return render_movement_overlay(
            image, analysis.movement_data, index, fps, show_pivots=True, show_steps=True,
            step_history_seconds=cfg.replay_step_history_seconds,
        )

    paths = (folder / "overlay.mp4", folder / "lines.mp4")
    for path, floor in zip(paths, (False, True), strict=True):
        save_overlay_video(
            video, path, 0, count - 1,
            frame_transform=partial(render, floor=floor),
        )
        rendered = VideoReader(path)
        try:
            if rendered.count != count:
                raise ValueError("Overlay video is incomplete; archive was not changed")
            rendered.frame(count - 1)
        finally:
            rendered.close()
    return paths
