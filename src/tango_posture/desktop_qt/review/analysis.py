"""Background posture analysis for Qt review playback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from ...config import Config
from ...couple.overlay import render_couple_frame_overlay, smooth_couple_frame_overlays
from ...movement import render_floor_guide, render_movement_overlay
from ...references.coach import interpolate_coach_data, render_coach_frame_overlay
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
    requested_dancer_mode: str = "auto"
    resolved_dancer_mode: str = "single"
    auto_decision: dict[str, Any] | None = None
    couple_result: dict[str, Any] | None = None
    selection_required: bool = False
    saved_archive_path: Path | None = None


def analyse_review_clip(
    video_path: Path,
    cfg: Config,
    *,
    selected_out: float,
    progress_callback: Callable[[int, int, str], None],
    cancel_check: Callable[[], bool],
    recover_tracking: bool | None = None,
    dancer_request: dict[str, Any] | None = None,
) -> ReviewAnalysis:
    """Analyze a local clip in memory without persisting a temporary report or database."""
    extra_kwargs: dict[str, Any] = {}
    if dancer_request is not None:
        extra_kwargs["dancer_request"] = dancer_request
    result = analyze_clip(
        video_path=video_path,
        cfg=cfg,
        selected_in=0.0,
        selected_out=selected_out,
        analysis_mode="movement",
        analyse_soundtrack=(
            bool(dancer_request.get("analyze_soundtrack", False))
            if isinstance(dancer_request, dict)
            else False
        ),
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        recover_tracking=recover_tracking,
        **extra_kwargs,
    )
    if getattr(result, "selection_required", False):
        return ReviewAnalysis(
            coach_data={},
            movement_data={},
            manifest=getattr(result, "manifest", None),
            result=result,
            requested_dancer_mode=getattr(result, "requested_dancer_mode", "auto"),
            resolved_dancer_mode="needs_selection",
            auto_decision=getattr(result, "auto_decision", None),
            couple_result=None,
            selection_required=True,
        )
    if result.coach_data is None:
        raise ValueError("Posture analysis completed without frame-level coaching data")
    return ReviewAnalysis(
        result.coach_data,
        result.movement_data,
        getattr(result, "manifest", None),
        result,
        requested_dancer_mode=getattr(result, "requested_dancer_mode", "auto"),
        resolved_dancer_mode=getattr(result, "resolved_dancer_mode", "single"),
        auto_decision=getattr(result, "auto_decision", None),
        couple_result=getattr(result, "couple_result", None),
        selection_required=False,
    )


def render_couple_review_report(
    couple_result: dict[str, Any],
    result: ClipAnalysisResult,
    provenance: dict[str, Any],
) -> str:
    """Format a dedicated Markdown report for a Couple Axis & Coordination run."""
    title = str(provenance.get("title") or result.title or "Couple Analysis")
    ann = couple_result.get("annotations") or {}
    a_name = str(ann.get("dancer_a_name") or "Dancer A")
    b_name = str(ann.get("dancer_b_name") or "Dancer B")
    dec = couple_result.get("auto_decision") or {}
    summaries = couple_result.get("summaries") or {}
    posture = summaries.get("posture") or {}
    coord = summaries.get("coordination") or {}
    trans = summaries.get("transitions") or {}
    turns = summaries.get("turns") or {}

    def _fmt_posture(key: str, label: str) -> str:
        s = posture.get(key) or {}
        if not isinstance(s, dict):
            return f"- **{label}**: unavailable"
        med = s.get("median_deg")
        iqr = s.get("iqr_deg")
        cov = float(s.get("coverage") or 0.0) * 100.0
        n = int(s.get("sample_count") or 0)
        if med is None:
            return f"- **{label}**: unavailable (coverage {cov:.0f}%)"
        return f"- **{label}**: median `{float(med):+.1f}°` (IQR `{float(iqr or 0.0):.1f}°`, {n} samples, {cov:.0f}% coverage)"

    def _fmt_lag(key: str, label: str) -> str:
        s = coord.get(key) or {}
        if not isinstance(s, dict) or s.get("value") is None:
            reason = str(s.get("reason") or "unavailable") if isinstance(s, dict) else "unavailable"
            return f"- **{label}**: unavailable ({reason})"
        val = float(s["value"])
        iqr = s.get("iqr")
        cnt = int(s.get("count") or 0)
        iqr_txt = f", IQR {float(iqr):.0f} ms" if isinstance(iqr, int | float) else ""
        return f"- **{label}**: `{val:+.0f} ms` ({cnt} paired events{iqr_txt}; positive = {b_name} later)"

    lines = [
        f"# Couple Axis & Coordination Report — {title}",
        "",
        "## Decision & Model Provenance",
        f"- **Requested mode**: `{dec.get('requested_mode', result.requested_dancer_mode)}` · **Resolved**: `{dec.get('resolved_mode', 'couple')}`",
        f"- **Model variant**: `{result.manifest.model_variant}` (`{result.manifest.model_hash[:12]}`)",
        f"- **Dancers**: `{a_name}` (A) · `{b_name}` (B)",
        f"- **Decision note**: {dec.get('explanation') or 'Explicit couple analysis.'}",
        "",
        "## Individual & Couple Midpoint Axes",
        _fmt_posture("torso_tilt_a", f"{a_name} torso axis tilt"),
        _fmt_posture("torso_tilt_b", f"{b_name} torso axis tilt"),
        _fmt_posture("couple_midpoint_tilt", "Couple midpoint axis tilt"),
        _fmt_posture("torso_separation", "Torso axis separation"),
        "",
        "## Starting, Travelling & Settling Coordination",
        _fmt_lag("start_lag_ms", "Start lag (B − A)"),
        _fmt_lag("slow_lag_ms", "Slowing lag (B − A)"),
        _fmt_lag("settle_lag_ms", "Settle lag (B − A)"),
        "",
        "## Transitions & Turns",
        (
            f"- **Transitions**: `{trans.get('status', 'unavailable')}`"
            + (f" · secondary rises: {trans.get('secondary_speed_peaks')}" if trans.get("secondary_speed_peaks") is not None else "")
        ),
        (
            f"- **Turns & exits**: {int(turns.get('turn_count') or 0)} marked/detected turns · status `{turns.get('status', 'unavailable')}`"
        ),
        "",
    ]
    return "\n".join(lines)


def render_review_report(
    result: ClipAnalysisResult, cfg: Config, index: int, provenance: dict[str, Any],
) -> str:
    if result.resolved_dancer_mode == "couple" and isinstance(result.couple_result, dict):
        return render_couple_review_report(result.couple_result, result, provenance)
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

    is_couple = analysis.resolved_dancer_mode == "couple" and isinstance(analysis.couple_result, dict)
    single_coach_data = (
        interpolate_coach_data(
            analysis.coach_data,
            cfg,
            pose_recovery=analysis.coach_data.get("pose_recovery"),
            movement_data=analysis.movement_data,
        )
        if not is_couple
        else analysis.coach_data
    )
    couple_by_idx: dict[int, dict[str, Any]] = {}
    couple_events: list[dict[str, Any]] = []
    couple_clip_in = 0.0
    a_label = "A"
    b_label = "B"
    if is_couple and isinstance(analysis.couple_result, dict):
        ann = analysis.couple_result.get("annotations") or {}
        a_label = str(ann.get("dancer_a_name") or "A")
        b_label = str(ann.get("dancer_b_name") or "B")
        couple_clip_in = float(analysis.couple_result.get("clip_in") or 0.0)
        couple_events = [
            ev for ev in (analysis.couple_result.get("events") or [])
            if isinstance(ev, dict) and not ev.get("deleted")
        ]
        by_frame = (analysis.couple_result.get("overlay") or {}).get("by_frame") or {}
        if isinstance(by_frame, dict):
            for k, v in by_frame.items():
                if str(k).lstrip("-").isdigit() and isinstance(v, dict):
                    couple_by_idx[int(k)] = v
        raw_fo_list = analysis.couple_result.get("frame_overlays") or []
        if isinstance(raw_fo_list, list) and raw_fo_list:
            fo_list = smooth_couple_frame_overlays(
                [fo for fo in raw_fo_list if isinstance(fo, dict)],
                tau_s=max(float(cfg.overlay_smoothing), 0.14),
                max_gap_s=float(cfg.dancer_analysis.display_interpolation_max_gap_s),
            )
            for pos, fo in enumerate(fo_list):
                fidx = fo.get("frame_index")
                if isinstance(fidx, int) and fidx not in couple_by_idx:
                    couple_by_idx[fidx] = fo
                if pos not in couple_by_idx and not any(
                    x.get("frame_index") == pos for x in fo_list
                ):
                    couple_by_idx[pos] = fo

    def render(image: Any, index: int, fps: float, *, floor: bool) -> Any:
        if cancel_check():
            raise ImportCancelledError("Posture analysis canceled")
        if floor:
            image = render_floor_guide(image)
        if is_couple:
            overlay_entry = couple_by_idx.get(index)
            t_sec = index / max(1.0, fps)
            active = [
                ev for ev in couple_events
                if abs(float(ev.get("timestamp", -999.0)) - t_sec) <= 0.35
                or (couple_clip_in > 0.0 and abs(float(ev.get("timestamp", -999.0)) - (t_sec + couple_clip_in)) <= 0.35)
            ]
            return render_couple_frame_overlay(
                image,
                overlay_entry,
                show_dancer_a=True,
                show_dancer_b=True,
                show_couple_axis=True,
                show_steps_turns=True,
                dancer_a_name=a_label,
                dancer_b_name=b_label,
                active_events=active,
            )
        image = render_coach_frame_overlay(
            image,
            single_coach_data,
            index,
            cfg,
            lines_only=True,
            show_estimates=cfg.replay_show_recovered_joints,
        )
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
