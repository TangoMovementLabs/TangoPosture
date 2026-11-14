"""Background posture analysis for Qt review playback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import Config
from ...references.pipeline import analyze_clip
from ...references.results import RunManifest


@dataclass(frozen=True)
class ReviewAnalysis:
    coach_data: dict[str, Any]
    movement_data: dict[str, Any]
    manifest: RunManifest | None = None


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
    return ReviewAnalysis(result.coach_data, result.movement_data, result.manifest)
