"""Create portable, indexed Library archives without changing source media."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2

from ...archive import (
    _archive_lock,
    append_segment,
    file_sha256,
    practice_directory,
    practice_path,
    read_analytics,
    read_manifest,
    rename_archive,
)
from ...config import Config
from ...couple.overlay import render_couple_frame_overlay, smooth_couple_frame_overlays
from ...movement import render_floor_guide, render_movement_overlay
from ...pose.recovery import rebase_pose_recovery
from ...references.archives import (
    mark_reference,
    promote_reference,
    reference_directory,
    reference_path,
)
from ...references.coach import render_coach_frame_overlay
from ...references.companions import (
    load_couple_companion,
    load_report_coach_data,
    load_report_movement_data,
)
from ...references.errors import ImportCancelledError
from ...references.results import ClipAnalysisResult
from ...replay import VideoReader, save_overlay_video
from ...review import read_report
from ..review.analysis import ReviewAnalysis, render_review_overlays, render_review_report


def _clean_archive_stem(name: str, fallback: str = "Analysis") -> str:
    cleaned = name.strip().replace("/", "-").replace("\\", "-").replace("\x00", "")
    if cleaned in ("", ".", ".."):
        return fallback
    return cleaned


def _unique_archive_path(folder: Path, name: str, suffix: str) -> Path:
    stem = _clean_archive_stem(name)
    base = (
        reference_path(folder, stem)
        if suffix == ".tpr"
        else practice_path(folder, stem).with_suffix(suffix)
    )
    if not base.exists():
        return base
    counter = 2
    while True:
        candidate = base.with_name(f"{base.stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _extract_couple_overlay_state(
    couple_data: dict[str, Any],
    cfg: Config | None = None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], float, str, str]:
    ann = couple_data.get("annotations") or {}
    a_label = str(ann.get("dancer_a_name") or "A")
    b_label = str(ann.get("dancer_b_name") or "B")
    couple_clip_in = float(couple_data.get("clip_in") or 0.0)
    couple_events = [
        ev
        for ev in (couple_data.get("events") or [])
        if isinstance(ev, dict) and not ev.get("deleted")
    ]
    by_idx: dict[int, dict[str, Any]] = {}
    by_frame = (couple_data.get("overlay") or {}).get("by_frame") or {}
    if isinstance(by_frame, dict):
        for k, v in by_frame.items():
            if str(k).lstrip("-").isdigit() and isinstance(v, dict):
                by_idx[int(k)] = v
    raw_fo_list = couple_data.get("frame_overlays") or []
    fo_list = (
        smooth_couple_frame_overlays(
            [fo for fo in raw_fo_list if isinstance(fo, dict)],
            tau_s=max(float(cfg.overlay_smoothing), 0.14) if cfg is not None else 0.20,
            max_gap_s=(
                float(cfg.dancer_analysis.display_interpolation_max_gap_s)
                if cfg is not None
                else 0.13
            ),
        )
        if isinstance(raw_fo_list, list) and raw_fo_list
        else []
    )
    if fo_list:
        for fo in fo_list:
            if isinstance(fo.get("frame_index"), int):
                by_idx.setdefault(int(fo["frame_index"]), fo)
    return by_idx, couple_events, couple_clip_in, a_label, b_label


def _rebase_coach_data(
    coach: dict[str, Any] | None,
    start: int,
    end: int,
    raw_sha: str,
    *,
    timestamps: list[float] | None = None,
    ensure_complete_count: int | None = None,
) -> dict[str, Any] | None:
    if coach is None:
        return None
    indexed_coach = copy.deepcopy(coach)
    raw_frames = indexed_coach.get("frames", {})
    rebased: dict[str, Any] = {}
    if isinstance(raw_frames, dict):
        for key, entry in raw_frames.items():
            if str(key).isdecimal() and start <= int(key) <= end:
                rebased[str(int(key) - start)] = entry
    if ensure_complete_count is not None:
        for i in range(ensure_complete_count):
            sk = str(i)
            if sk not in rebased:
                ts_val = (
                    timestamps[i]
                    if timestamps is not None and i < len(timestamps)
                    else float(i) / 30.0
                )
                ts0 = timestamps[0] if timestamps else 0.0
                rebased[sk] = {
                    "timestamp": round(float(ts_val), 4),
                    "clip_timestamp": round(max(0.0, float(ts_val) - float(ts0)), 4),
                    "poses": [],
                    "measurement_poses": [],
                }
    indexed_coach["frames"] = rebased
    indexed_coach["source_sha256"] = raw_sha
    if indexed_coach.get("pose_recovery") is not None:
        rebased_rec = rebase_pose_recovery(
            indexed_coach["pose_recovery"],
            start,
            end + 1,
            new_source_sha256=raw_sha,
        )
        if rebased_rec is None:
            indexed_coach.pop("pose_recovery", None)
        else:
            indexed_coach["pose_recovery"] = rebased_rec
    return indexed_coach


def _rebase_movement_data(
    movement: dict[str, Any] | None,
    start: int,
    end: int,
    raw_sha: str,
    fps: float,
) -> dict[str, Any] | None:
    if movement is None:
        return None
    indexed_movement = copy.deepcopy(movement)
    indexed_movement.update(
        source_sha256=raw_sha,
        segment_index=0,
        frame_count=end - start + 1,
    )
    for key in ("steps", "hip_frames"):
        indexed_movement[key] = [
            {
                **entry,
                "frame": entry["frame"] - start,
                **({"time": (entry["frame"] - start) / fps} if key == "steps" else {}),
            }
            for entry in indexed_movement.get(key, [])
            if isinstance(entry, dict) and isinstance(entry.get("frame"), int) and start <= entry["frame"] <= end
        ]
    indexed_movement["pivots"] = [
        {
            **entry,
            "start_frame": max(start, entry["start_frame"]) - start,
            "end_frame": min(end + 1, entry["end_frame"]) - start,
        }
        for entry in indexed_movement.get("pivots", [])
        if isinstance(entry, dict) and entry["end_frame"] > start and entry["start_frame"] <= end
    ]
    indexed_movement["cut_frames"] = [
        frame - start
        for frame in indexed_movement.get("cut_frames", [])
        if isinstance(frame, int) and start <= frame <= end
    ]
    return indexed_movement


def _rebase_couple_data(
    couple_data: dict[str, Any] | None,
    start: int,
    end: int,
    raw_sha: str,
) -> dict[str, Any] | None:
    if not isinstance(couple_data, dict):
        return None
    indexed_couple = copy.deepcopy(couple_data)
    indexed_couple["video_sha256"] = raw_sha
    indexed_couple["segment_index"] = 0
    couple_overlay = indexed_couple.get("overlay")
    if isinstance(couple_overlay, dict) and isinstance(couple_overlay.get("by_frame"), dict):
        couple_overlay["by_frame"] = {
            str(int(k) - start): entry
            for k, entry in couple_overlay["by_frame"].items()
            if str(k).lstrip("-").isdigit() and start <= int(k) <= end
        }
    if isinstance(indexed_couple.get("samples"), list):
        indexed_couple["samples"] = [
            {**s, "frame_index": int(s["frame_index"]) - start}
            for s in indexed_couple["samples"]
            if isinstance(s, dict) and start <= int(s.get("frame_index", -1)) <= end
        ]
    if isinstance(indexed_couple.get("frame_overlays"), list):
        indexed_couple["frame_overlays"] = [
            {**fo, "frame_index": int(fo["frame_index"]) - start}
            for fo in indexed_couple["frame_overlays"]
            if isinstance(fo, dict) and start <= int(fo.get("frame_index", -1)) <= end
        ]
    return indexed_couple


def _build_couple_dancer_analysis(
    indexed_couple: dict[str, Any],
    raw_sha: str,
) -> dict[str, Any]:
    run_id = str(indexed_couple.get("run_id") or f"couple-0-{raw_sha[:8]}")
    decision = indexed_couple.get("auto_decision")
    return {
        "schema_version": 1,
        "active_mode": "couple",
        "active_run_id": run_id,
        "runs": {
            "couple": {
                "run_id": run_id,
                "requested_mode": (
                    str(decision.get("requested_mode", "auto"))
                    if isinstance(decision, dict)
                    else "auto"
                ),
                "resolved_mode": "couple",
                "auto_decision": decision,
                "couple_result": indexed_couple,
            }
        },
    }


def archive_report(
    report_path: Path, cfg: Config, cancel_check: Callable[[], bool], *,
    as_reference: bool = False,
) -> Path:
    record = read_report(report_path)
    source = record.recorded_video
    if source is None or not source.is_file():
        raise ValueError(f"{report_path.name}: linked video is unavailable")
    target_dir = reference_directory(cfg.data_dir) if as_reference else practice_directory(cfg.data_dir)
    suffix = ".tpr" if as_reference else ".tp"
    destination = target_dir / f"{report_path.stem}{suffix}"
    if destination.exists():
        read_manifest(destination)
        stored = read_analytics(destination, 0)
        if stored is not None and stored.get("original_report_sha256") == record.digest:
            return destination
        destination = _unique_archive_path(target_dir, report_path.stem, suffix)
    coach = load_report_coach_data(record)
    movement = load_report_movement_data(record)
    couple_data = load_couple_companion(report_path.parent / f"{report_path.stem}.couple.v1.json")
    if couple_data is None and isinstance(coach, dict) and isinstance(coach.get("couple_analysis"), dict):
        couple_data = copy.deepcopy(coach["couple_analysis"])
    frame_keys = (
        [int(k) for k in coach.get("frames", {}) if str(k).isdecimal()]
        if isinstance(coach, dict) and isinstance(coach.get("frames"), dict)
        else []
    )
    reader = VideoReader(source)
    try:
        fps = reader.fps
        frame_count = reader.count
        duration = reader.duration
    finally:
        reader.close()
    if frame_keys:
        start = max(0, min(frame_keys))
        end = min(frame_count - 1, max(frame_keys))
    else:
        start = max(0, round((record.clip_in or 0) * fps))
        end = min(frame_count - 1, round((record.clip_out or duration) * fps) - 1)
    if end < start:
        raise ValueError("Selected clip contains no video frames")

    couple_overlay_idx: dict[int, dict[str, Any]] = {}
    couple_events: list[dict[str, Any]] = []
    couple_clip_in = 0.0
    a_label = "A"
    b_label = "B"
    is_couple_archive = isinstance(couple_data, dict)
    if isinstance(couple_data, dict):
        (
            couple_overlay_idx,
            couple_events,
            couple_clip_in,
            a_label,
            b_label,
        ) = _extract_couple_overlay_state(couple_data, cfg)

    def render(image: Any, index: int, rate: float, *, floor: bool = False) -> Any:
        if cancel_check():
            raise ImportCancelledError("Archive import canceled")
        height, width = image.shape[:2]
        target_width = max(960, width) if (coach is not None or is_couple_archive) and not floor else width
        target_size = (target_width, round(height * target_width / width))
        image = render_floor_guide(image) if floor else image
        if (image.shape[1], image.shape[0]) != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        if is_couple_archive:
            overlay_entry = couple_overlay_idx.get(index) or couple_overlay_idx.get(index - start)
            t_sec = index / max(1.0, rate)
            rel_idx = index - start
            src_t = (
                float(record.timestamps[rel_idx])
                if 0 <= rel_idx < len(record.timestamps)
                else t_sec
            )
            active = [
                ev
                for ev in couple_events
                if abs(float(ev.get("timestamp", -999.0)) - t_sec) <= 0.35
                or abs(float(ev.get("timestamp", -999.0)) - src_t) <= 0.35
                or (
                    couple_clip_in > 0.0
                    and abs(float(ev.get("timestamp", -999.0)) - (t_sec + couple_clip_in)) <= 0.35
                )
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
        if coach is not None:
            image = render_coach_frame_overlay(
                image,
                coach,
                index,
                cfg,
                lines_only=floor,
                show_estimates=cfg.replay_show_recovered_joints,
            )
        return render_movement_overlay(
            image, movement, index, rate, show_pivots=True, show_steps=True,
            step_history_seconds=cfg.replay_step_history_seconds,
        )

    if as_reference:
        destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="tango-library-archive-", dir=destination.parent if as_reference else None,
    ) as temporary:
        folder = Path(temporary)
        raw = folder / "raw.mp4"
        overlay = folder / "overlay.mp4"
        lines = folder / "lines.mp4"

        def raw_frame(image: Any, _index: int, _fps: float) -> Any:
            if cancel_check():
                raise ImportCancelledError("Archive import canceled")
            return image

        save_overlay_video(source, raw, start, end, frame_transform=raw_frame)
        raw_reader = VideoReader(raw)
        try:
            raw_count = raw_reader.count
            raw_fps = raw_reader.fps if math.isfinite(raw_reader.fps) and raw_reader.fps > 0 else fps
        finally:
            raw_reader.close()
        raw_end = start + raw_count - 1
        save_overlay_video(source, overlay, start, end, frame_transform=render)
        save_overlay_video(
            source, lines, start, end,
            frame_transform=lambda image, index, rate: render(image, index, rate, floor=True),
        )
        raw_sha = file_sha256(raw)
        safe_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else 30.0
        if (
            len(record.timestamps) == raw_count
            and all(math.isfinite(val) and val >= 0.0 for val in record.timestamps)
            and all(earlier < later for earlier, later in pairwise(record.timestamps))
        ):
            timestamps = [float(val) for val in record.timestamps]
        else:
            timestamps = [round((start + idx) / safe_fps, 6) for idx in range(raw_count)]
        indexed_coach = _rebase_coach_data(
            coach,
            start,
            raw_end,
            raw_sha,
            timestamps=timestamps,
            ensure_complete_count=raw_count if is_couple_archive else None,
        )
        indexed_movement = _rebase_movement_data(movement, start, raw_end, raw_sha, safe_fps)
        indexed_couple = _rebase_couple_data(couple_data, start, raw_end, raw_sha)
        if indexed_couple is not None and indexed_coach is not None:
            indexed_coach["couple_analysis"] = indexed_couple
        dancer_analysis = (
            _build_couple_dancer_analysis(indexed_couple, raw_sha)
            if indexed_couple is not None
            else None
        )
        analytics: dict[str, Any] = {
            "model_hash": coach.get("model_hash") if coach else None,
            "capture_profile": coach.get("capture_profile") if coach else None,
            "max_gap": cfg.max_gap,
            "frame_timestamps": timestamps,
            "coach_data": indexed_coach, "movement_data": indexed_movement,
            "original_report_sha256": record.digest,
            "source_provenance": {
                "sha256": record.source_sha256 or file_sha256(source),
                "clip_in": record.clip_in, "clip_out": record.clip_out,
                "youtube_link": record.youtube_link, "facebook_reel": record.facebook_reel,
            },
        }
        if dancer_analysis is not None:
            analytics["dancer_analysis"] = dancer_analysis
        if cancel_check():
            raise ImportCancelledError("Archive import canceled")
        output = folder / "reference.tp" if as_reference else destination
        append_segment(
            output, report_path, raw, overlay, timestamps,
            analytics=analytics, lines=lines,
        )
        rename_archive(output, record.title)
        if as_reference:
            mark_reference(output, record.title, {
                "provenance": analytics["source_provenance"],
                "camera_view": analytics.get("capture_profile"),
                "clip_start": record.clip_in, "clip_end": record.clip_out,
            })
            destination.parent.mkdir(parents=True, exist_ok=True)
            with _archive_lock(destination):
                if destination.exists():
                    destination = _unique_archive_path(destination.parent, record.title, ".tpr")
                output.replace(destination)
    return destination


def promote_segment_to_tpr(
    archive_path: Path,
    segment_index: int,
    cfg: Config,
    *,
    preferred_name: str | None = None,
    cancel_check: Callable[[], bool] = lambda: False,
) -> Path:
    """Create a new unique `.tpr` reference archive from an analyzed segment of a `.tp` file."""
    manifest = read_manifest(archive_path)
    base_name = _clean_archive_stem(
        preferred_name or str(manifest.get("name") or archive_path.stem),
        fallback=archive_path.stem or "Reference",
    )
    ref_dir = reference_directory(cfg.data_dir)
    dest_tpr = _unique_archive_path(ref_dir, base_name, ".tpr")
    return promote_reference(
        archive_path,
        dest_tpr,
        base_name,
        index=segment_index,
        cancel_check=cancel_check,
    )


def save_clip_result_as_tpr(
    result: ClipAnalysisResult,
    cfg: Config,
    cancel_check: Callable[[], bool] = lambda: False,
    *,
    preferred_name: str | None = None,
) -> Path:
    """Package an in-memory ClipAnalysisResult (Single or Couple) into a standalone `.tpr` archive."""
    source = Path(result.video_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Source video unavailable for saving .tpr archive: {source}")
    title = _clean_archive_stem(preferred_name or result.title or source.stem, fallback=source.stem)
    ref_dir = reference_directory(cfg.data_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_archive_path(ref_dir, title, ".tpr")

    reader = VideoReader(source)
    try:
        src_fps = reader.fps if math.isfinite(reader.fps) and reader.fps > 0 else 30.0
        src_count = reader.count
    finally:
        reader.close()

    log_indices = [
        int(row["frame_index"])
        for row in (result.frame_log or [])
        if isinstance(row, dict) and isinstance(row.get("frame_index"), int)
    ]
    if log_indices:
        start = max(0, min(log_indices))
        end = min(src_count - 1, max(log_indices))
    else:
        start = max(0, round(float(result.selected_in) * src_fps))
        end = min(src_count - 1, round(float(result.selected_out) * src_fps) - 1)
    if end < start:
        raise ValueError("Selected clip contains no video frames")

    with TemporaryDirectory(prefix="tango-tpr-export-", dir=destination.parent) as temporary:
        folder = Path(temporary)
        raw = folder / "raw.mp4"

        def raw_frame(image: Any, _index: int, _fps: float) -> Any:
            if cancel_check():
                raise ImportCancelledError("Archive save canceled")
            return image

        save_overlay_video(source, raw, start, end, frame_transform=raw_frame)
        raw_reader = VideoReader(raw)
        try:
            raw_count = raw_reader.count
            raw_fps = raw_reader.fps if math.isfinite(raw_reader.fps) and raw_reader.fps > 0 else src_fps
        finally:
            raw_reader.close()

        raw_sha = file_sha256(raw)
        raw_ts = [
            float(row["timestamp"])
            for row in (result.frame_log or [])
            if isinstance(row, dict) and isinstance(row.get("timestamp"), (int, float))
        ]
        if (
            len(raw_ts) == raw_count
            and all(math.isfinite(val) and val >= 0.0 for val in raw_ts)
            and all(earlier < later for earlier, later in pairwise(raw_ts))
        ):
            timestamps = raw_ts
        else:
            step_s = 1.0 / max(raw_fps, 1.0)
            base_ts = max(0.0, float(result.selected_in))
            timestamps = [round(base_ts + idx * step_s, 6) for idx in range(raw_count)]

        indexed_coach = _rebase_coach_data(
            result.coach_data,
            start,
            start + raw_count - 1,
            raw_sha,
            timestamps=timestamps,
            ensure_complete_count=raw_count,
        )
        assert indexed_coach is not None
        indexed_movement = _rebase_movement_data(
            result.movement_data, start, start + raw_count - 1, raw_sha, raw_fps
        ) or {"steps": [], "pivots": []}
        indexed_couple = _rebase_couple_data(
            result.couple_result, start, start + raw_count - 1, raw_sha
        )
        if indexed_couple is not None:
            indexed_coach["couple_analysis"] = indexed_couple

        rebased_frame_log = (
            [
                {**row, "frame_index": idx}
                for idx, row in enumerate(result.frame_log)
            ]
            if len(result.frame_log) == raw_count
            else [
                {
                    "frame_index": idx,
                    "timestamp": ts,
                    "clip_timestamp": max(0.0, ts - timestamps[0]),
                    "in_calibration_window": False,
                }
                for idx, ts in enumerate(timestamps)
            ]
        )
        overlay_result = replace(result, frame_log=rebased_frame_log)
        review_analysis = ReviewAnalysis(
            coach_data=indexed_coach,
            movement_data=indexed_movement,
            manifest=result.manifest,
            result=overlay_result,
            requested_dancer_mode=result.requested_dancer_mode,
            resolved_dancer_mode=result.resolved_dancer_mode,
            auto_decision=result.auto_decision,
            couple_result=indexed_couple,
        )
        overlay, lines = render_review_overlays(raw, folder, review_analysis, cfg, cancel_check)

        provenance = {
            "sha256": result.source_sha256 or file_sha256(source),
            "clip_in": float(result.selected_in),
            "clip_out": float(result.selected_out),
            "title": title,
            "source_filename": source.name,
        }
        report_text = render_review_report(result, cfg, 0, provenance)
        report_path = folder / "report.md"
        report_path.write_text(report_text, encoding="utf-8")

        resolved_mode = (
            "couple"
            if result.resolved_dancer_mode == "couple" and indexed_couple is not None
            else "single"
        )
        if resolved_mode == "couple" and indexed_couple is not None:
            dancer_analysis = _build_couple_dancer_analysis(indexed_couple, raw_sha)
            stored_frames: list[dict[str, Any]] = []
        else:
            run_id = f"single-0-{raw_sha[:8]}"
            dancer_analysis = {
                "schema_version": 1,
                "active_mode": "single",
                "active_run_id": run_id,
                "runs": {
                    "single": {
                        "run_id": run_id,
                        "requested_mode": result.requested_dancer_mode,
                        "resolved_mode": "single",
                        "auto_decision": result.auto_decision,
                        "coach_data": indexed_coach,
                        "movement_data": indexed_movement,
                        "session": result.session_summary,
                        "frames": result.scored_frame_log or [],
                    }
                },
            }
            stored_frames = list(result.scored_frame_log or [])

        analytics: dict[str, Any] = {
            "model_hash": result.manifest.model_hash,
            "capture_profile": result.manifest.capture_profile,
            "max_gap": cfg.max_gap,
            "frame_timestamps": timestamps,
            "frames": stored_frames,
            "coach_data": indexed_coach,
            "movement_data": indexed_movement,
            "session": result.session_summary,
            "session_events": result.session_events or [],
            "source_provenance": {
                "sha256": result.source_sha256 or file_sha256(source),
                "clip_in": float(result.selected_in),
                "clip_out": float(result.selected_out),
                "youtube_link": None,
                "facebook_reel": None,
            },
            "dancer_analysis": dancer_analysis,
        }
        metadata_extra: dict[str, Any] = {
            "analysis_mode": result.analysis_mode,
            "is_imported_reference": True,
            "source": f"import:{source.name}",
            "source_sha256": result.source_sha256,
            "title": title,
            "clip_in": float(result.selected_in),
            "clip_out": float(result.selected_out),
            "readiness": result.readiness.to_dict(),
            "reference_summary": None,
            "coverage": {key: val.to_dict() for key, val in result.coverage.items()},
            "soundtrack": result.soundtrack.to_metadata_dict(),
            "capture_profile": result.manifest.capture_profile,
            "requested_dancer_mode": result.requested_dancer_mode,
            "resolved_dancer_mode": resolved_mode,
        }
        if cancel_check():
            raise ImportCancelledError("Archive save canceled")
        staged = folder / "reference.tp"
        append_segment(
            staged,
            report_path,
            raw,
            overlay,
            timestamps,
            analytics=analytics,
            lines=lines,
            metadata_extra=metadata_extra,
        )
        rename_archive(staged, title)
        mark_reference(staged, title, {
            "provenance": analytics["source_provenance"],
            "camera_view": analytics.get("capture_profile"),
            "clip_start": float(result.selected_in),
            "clip_end": float(result.selected_out),
        })
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _archive_lock(destination):
            if destination.exists():
                destination = _unique_archive_path(destination.parent, title, ".tpr")
            staged.replace(destination)
    return destination
