"""Coordinate cached pose, movement, soundtrack and report analysis."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from ..archive import file_sha256
from ..calibration.baseline import profile
from ..calibration.readiness import check_calibration_readiness
from ..config import Config
from ..couple import (
    AutoDecisionResult,
    CoupleAnalysisRequest,
    classify_opening_window,
    resolve_close_embrace_frame_poses,
)
from ..couple.tracking import (
    _pose_torso_mid_norm,
    _translate_pose,
    _wide_ankle_span_for_pair,
)
from ..domain import Calibration, FrameObservation, PoseEstimator
from ..engines import (
    CommonEngineContext,
    build_needs_selection_result,
    run_couple_dancer_engine,
    run_solo_dancer_engine,
)
from ..models import analysis_model_path, model_manifest, prepare_analysis_model, verify
from ..pose.mediapipe_backend import MediaPipeEstimator
from ..reports import write_markdown_report
from ..review import write_metadata
from ..storage import Storage
from .coach import _build_coach_frame_entry
from .companions import save_coach_companion, save_couple_companion, save_movement_companion
from .constants import MAX_IMPORT_RANGE_SECONDS
from .downloads import download_facebook_reel, download_youtube_video, imports_directory
from .errors import ImportCancelledError
from .links import YouTubeClockAlignment, parse_facebook_reel_url, parse_youtube_url
from .results import ClipAnalysisResult, ImportedClipResult
from .soundtrack import analyse_video_soundtrack
from .video import VideoProbeInfo, iter_virtual_clip_frames, probe_import_video, validate_clip_range

logger = logging.getLogger(__name__)


def analyze_clip(
    *,
    video_path: Path | str,
    cfg: Config,
    selected_in: float = 0.0,
    selected_out: float | None = None,
    title: str = "",
    analysis_mode: str = "movement",
    reference: Calibration | None = None,
    calibration_interval: tuple[float, float] | None = None,
    analyse_soundtrack: bool = False,
    storage: Storage | None = None,
    estimator_factory: Callable[[], PoseEstimator] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    probe: VideoProbeInfo | None = None,
    source_sha256: str | None = None,
    recover_tracking: bool | None = None,
    dancer_request: CoupleAnalysisRequest | dict[str, Any] | None = None,
) -> ClipAnalysisResult:
    """Analyze a local video clip in memory without writing reports or temporary databases."""
    if not video_path or not str(video_path).strip():
        raise ValueError("Choose a local video file to analyze")

    if probe is None:
        probe = probe_import_video(Path(video_path))
    raw_out = float(selected_out) if selected_out is not None else probe.duration_seconds
    clip_in, clip_out = validate_clip_range(probe, float(selected_in), raw_out)
    clip_duration = clip_out - clip_in

    model_path = analysis_model_path(cfg)
    est: PoseEstimator
    if estimator_factory is None:
        model_hash = (
            verify(model_path, cfg.analysis_model) if model_path.is_file() else
            model_manifest(cfg.analysis_model)["sha256"]
        )
    elif reference is not None:
        model_hash = reference.model_hash
    elif model_path.is_file():
        model_hash = file_sha256(model_path)
    else:
        model_hash = hashlib.sha256(b"tango-posture-pose-landmarker").hexdigest()

    readiness = check_calibration_readiness(
        analysis_mode=analysis_mode,
        cfg=cfg,
        probe=probe,
        selected_in=clip_in,
        selected_out=clip_out,
        reference=reference,
        calibration_interval=calibration_interval,
        model_hash=model_hash,
    )
    if not readiness.ready:
        raise ValueError(readiness.reason or readiness.summary)

    if source_sha256 is None:
        if progress_callback is not None:
            progress_callback(0, 100, "Computing file identity…")
        source_sha = file_sha256(probe.path)
    else:
        source_sha = source_sha256

    if estimator_factory is None:
        download_progress = None
        if progress_callback is not None:
            download_progress = lambda current, total, message: progress_callback(
                round(current / max(1, total) * 20), 100, message,
            )
            progress_callback(0, 100, f"Preparing {cfg.analysis_model.capitalize()} analysis model...")
        model_path = prepare_analysis_model(
            cfg, progress_callback=download_progress, cancel_check=cancel_check,
        )
        est = MediaPipeEstimator(model_path, variant=cfg.analysis_model)
    else:
        est = estimator_factory()
    decoded_frames: list[tuple[int, float, float, FrameObservation]] = []
    auto_observations: list[tuple[int, FrameObservation]] = []
    last_split_pos: int | None = None
    last_primary_mid: tuple[float, float] | None = None
    last_split_pair: list[dict[str, Any]] | None = None
    frame_times: list[tuple[int, float, float]] = []
    expected_frames = max(1, round(clip_duration * max(probe.fps, 1.0)))
    infer_times: list[float] = []

    request_obj = (
        CoupleAnalysisRequest.from_dict(dancer_request)
        if dancer_request is not None
        else None
    )
    requested_mode = (
        request_obj.requested_mode
        if request_obj is not None
        else cfg.dancer_analysis.default_dancer_mode
    )
    forced_override = bool(request_obj.forced_override) if request_obj is not None else False
    analysis_region = request_obj.analysis_region if request_obj is not None else None
    anchors = list(request_obj.annotations.anchors) if request_obj is not None else []
    effective_soundtrack = (
        bool(request_obj.analyze_soundtrack)
        if request_obj is not None and request_obj.analyze_soundtrack is not None
        else bool(analyse_soundtrack)
    )
    window_duration = min(float(cfg.dancer_analysis.auto_detection_window_s), float(clip_duration))
    window_end = clip_in + window_duration
    run_auto_window = requested_mode == "auto" and not forced_override
    auto_decision: AutoDecisionResult | None = None
    selection_required = False
    image_detector: Any = None

    def _get_image_detector() -> Any:
        nonlocal image_detector
        if image_detector is None:
            import mediapipe as mp

            image_detector = mp.tasks.vision.PoseLandmarker.create_from_options(
                mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(model_path),
                        delegate=mp.tasks.BaseOptions.Delegate.CPU,
                    ),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_poses=2,
                    min_pose_detection_confidence=0.35,
                    min_pose_presence_confidence=0.35,
                )
            )
        return image_detector

    try:
        for idx, item in enumerate(
            iter_virtual_clip_frames(
                probe.path,
                clip_in,
                clip_out,
                probe=probe,
                cancel_check=cancel_check,
            )
        ):
            if (
                run_auto_window
                and auto_decision is None
                and item.source_time >= window_end - 1e-9
                and auto_observations
            ):
                auto_decision = classify_opening_window(
                    auto_observations,
                    cfg,
                    clip_in=clip_in,
                    clip_out=clip_out,
                    requested_mode=requested_mode,
                    forced_override=forced_override,
                    analysis_region=analysis_region,
                    anchors=anchors,
                )
                if request_obj is not None and auto_decision.resolved_mode == "needs_selection":
                    selection_required = True
                    break

            t0 = time.perf_counter()
            obs = est.estimate(item.image, item.source_time)
            infer_times.append(time.perf_counter() - t0)
            decoded_frames.append((item.frame_index, item.source_time, item.clip_time, obs))
            auto_obs = obs
            split_couple_in_pass = requested_mode == "couple" or (
                auto_decision is not None and auto_decision.resolved_mode == "couple"
            )
            if estimator_factory is None and split_couple_in_pass and len(obs.poses) == 1:
                primary_pose = obs.poses[0]
                cur_mid = _pose_torso_mid_norm(primary_pose, cfg.confidence)
                propagated = False
                if (
                    idx % 3 != 0
                    and last_split_pos is not None
                    and (idx - last_split_pos) <= 2
                    and last_split_pair is not None
                    and last_primary_mid is not None
                    and cur_mid is not None
                ):
                    dx = cur_mid[0] - last_primary_mid[0]
                    dy = cur_mid[1] - last_primary_mid[1]
                    if math.hypot(dx, dy) <= 0.05:
                        auto_obs = FrameObservation(
                            obs.timestamp,
                            obs.width,
                            obs.height,
                            [_translate_pose(p, dx, dy) for p in last_split_pair],
                        )
                        propagated = True
                if not propagated and idx % 3 == 0:
                    resolved_poses = resolve_close_embrace_frame_poses(
                        item.image, primary_pose, _get_image_detector(), obs.width, obs.height, cfg
                    )
                    if len(resolved_poses) >= 2:
                        auto_obs = FrameObservation(
                            obs.timestamp, obs.width, obs.height, resolved_poses
                        )
                        last_split_pos = idx
                        last_primary_mid = cur_mid
                        last_split_pair = resolved_poses
                    else:
                        last_split_pos = None
                        last_primary_mid = None
                        last_split_pair = None
            elif estimator_factory is None and split_couple_in_pass and len(obs.poses) >= 2:
                last_split_pos = None
                last_primary_mid = None
                last_split_pair = None
            elif (
                estimator_factory is None
                and run_auto_window
                and auto_decision is None
                and idx % 2 == 0
                and len(obs.poses) == 1
                and _wide_ankle_span_for_pair(obs.poses[0], cfg.confidence)
            ):
                resolved_poses = resolve_close_embrace_frame_poses(
                    item.image, obs.poses[0], _get_image_detector(), obs.width, obs.height, cfg
                )
                if len(resolved_poses) >= 2:
                    auto_obs = FrameObservation(obs.timestamp, obs.width, obs.height, resolved_poses)
            auto_observations.append((item.frame_index, auto_obs))
            frame_times.append((item.frame_index, item.source_time, item.clip_time))
            if progress_callback is not None and (idx % 5 == 0 or idx + 1 >= expected_frames):
                pct = min(75, 20 + round(((idx + 1) / max(expected_frames, idx + 1)) * 55))
                if (
                    request_obj is not None
                    and run_auto_window
                    and auto_decision is None
                    and idx + 1 < expected_frames
                ):
                    elapsed_w = min(window_duration, max(0.0, item.source_time - clip_in))
                    progress_callback(
                        pct,
                        100,
                        f"Detecting number of dancers · {elapsed_w:.1f} / {window_duration:.1f} s",
                    )
                else:
                    progress_callback(
                        pct,
                        100,
                        f"Analyzing forward frames with {cfg.analysis_model.capitalize()} ({idx + 1})…",
                    )
    finally:
        with suppress(Exception):
            est.close()
        if image_detector is not None:
            with suppress(Exception):
                image_detector.close()
            image_detector = None

    if not decoded_frames:
        raise ValueError("Selected clip range contains no decodable frames")

    if cancel_check is not None and cancel_check():
        raise ImportCancelledError("Reference video analysis was cancelled")

    if auto_decision is None:
        auto_decision = classify_opening_window(
            auto_observations,
            cfg,
            clip_in=clip_in,
            clip_out=clip_out,
            requested_mode=requested_mode,
            forced_override=forced_override,
            analysis_region=analysis_region,
            anchors=anchors,
        )
        if request_obj is not None and auto_decision.resolved_mode == "needs_selection":
            selection_required = True

    cap_profile = profile(cfg, probe.width, probe.height, "video")
    clean_title = title.strip() or probe.path.stem
    mean_duration = float(np.mean(infer_times)) if infer_times else 0.0
    processed_fps = 1.0 / mean_duration if mean_duration > 0 else None
    performance = {
        "model_hash": model_hash,
        "model_variant": cfg.analysis_model,
        "resolution": [probe.width, probe.height],
        "processed_fps": processed_fps,
        "dropped_frames": 0,
        "timestamp_fallback": False,
    }
    raw_observations = [(f_idx, f_obs) for f_idx, _, _, f_obs in decoded_frames]

    if selection_required:
        ctx = CommonEngineContext(
            probe=probe,
            cfg=cfg,
            model_path=model_path,
            model_hash=model_hash,
            source_sha=source_sha,
            clip_in=clip_in,
            clip_out=clip_out,
            clip_duration=clip_duration,
            clean_title=clean_title,
            cap_profile=cap_profile,
            readiness=readiness,
            decoded_frames=decoded_frames,
            raw_observations=raw_observations,
            auto_observations=auto_observations,
            frame_times=frame_times,
            auto_decision=auto_decision,
            requested_mode=requested_mode,
            resolved_mode="needs_selection",
            request_obj=request_obj,
            performance=performance,
            processed_fps=processed_fps,
            analysis_mode=analysis_mode,
        )
        return build_needs_selection_result(
            ctx,
            reference=reference,
            calibration_interval=calibration_interval,
            effective_soundtrack=effective_soundtrack,
        )

    resolved_mode = (
        "single" if auto_decision.resolved_mode == "needs_selection" else auto_decision.resolved_mode
    )

    if progress_callback is not None:
        progress_callback(80, 100, "Analysing soundtrack & movement…")

    soundtrack_summary = analyse_video_soundtrack(
        probe.path,
        clip_in,
        clip_out,
        frame_times,
        cfg,
        enabled=effective_soundtrack,
        video_first_pts_seconds=probe.first_pts_seconds,
    )

    engine_ctx = CommonEngineContext(
        probe=probe,
        cfg=cfg,
        model_path=model_path,
        model_hash=model_hash,
        source_sha=source_sha,
        clip_in=clip_in,
        clip_out=clip_out,
        clip_duration=clip_duration,
        clean_title=clean_title,
        cap_profile=cap_profile,
        readiness=readiness,
        decoded_frames=decoded_frames,
        raw_observations=raw_observations,
        auto_observations=auto_observations,
        frame_times=frame_times,
        auto_decision=auto_decision,
        requested_mode=requested_mode,
        resolved_mode=resolved_mode,
        request_obj=request_obj,
        performance=performance,
        processed_fps=processed_fps,
        analysis_mode=analysis_mode,
        soundtrack_summary=soundtrack_summary,
    )

    if resolved_mode == "couple":
        return run_couple_dancer_engine(
            engine_ctx,
            estimator_factory=estimator_factory,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    return run_solo_dancer_engine(
        engine_ctx,
        reference=reference,
        calibration_interval=calibration_interval,
        storage=storage,
        estimator_factory=estimator_factory,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        recover_tracking=recover_tracking,
        build_coach_frame_entry=_build_coach_frame_entry,
    )


analyse_clip = analyze_clip


def analyse_imported_clip(
    *,
    video_path: Path | str,
    cfg: Config,
    report_dir: Path | None = None,
    storage: Storage | None = None,
    selected_in: float = 0.0,
    selected_out: float | None = None,
    title: str = "",
    youtube_url: str | None = None,
    youtube_alignment: YouTubeClockAlignment | None = None,
    facebook_url: str | None = None,
    analysis_mode: str = "movement",
    reference: Calibration | None = None,
    calibration_interval: tuple[float, float] | None = None,
    analyse_soundtrack: bool = False,
    estimator_factory: Callable[[], PoseEstimator] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    youtube_downloader: Callable[[str, Path], Path] | None = None,
    facebook_downloader: Callable[[str, Path], Path] | None = None,
    recover_tracking: bool | None = None,
    dancer_request: CoupleAnalysisRequest | dict[str, Any] | None = None,
) -> ImportedClipResult:
    """Import and analyse a selected range `[selected_in, selected_out)` of a local video file.

    When `video_path` is blank and `youtube_url` is valid, the video is first downloaded into
    `imports_directory(cfg.data_dir)` (reusing a cached copy) and analysed; a downloaded video is
    the exact same edit as YouTube, so a confirmed 1:1 clock alignment is stored automatically.

    - Leaves the user's video file untouched on disk.
    - Supports `analysis_mode="movement"` (no calibration required) and
      `analysis_mode="posture"` (using `reference` or `calibration_interval`).
    - Optionally decodes and analyses the video soundtrack (`analyse_soundtrack=True`).
    - Writes a Markdown report, companion `.metadata.json`, and `.movement.v1.json`.
    """
    yt_link = parse_youtube_url(youtube_url) if youtube_url and youtube_url.strip() else None
    facebook_link = (
        parse_facebook_reel_url(facebook_url)
        if facebook_url and facebook_url.strip()
        else None
    )
    if yt_link is not None and facebook_link is not None:
        raise ValueError("Import from only one video URL at a time")
    downloaded_from_youtube = False
    downloaded_from_facebook = False
    if not video_path or not str(video_path).strip():
        if yt_link is None and facebook_link is None:
            raise ValueError("Choose a local video file or enter a YouTube or Facebook Reel URL")
        if yt_link is not None:
            downloaded_path, _ = download_youtube_video(
                yt_link.canonical_url,
                imports_directory(cfg.data_dir),
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                downloader=youtube_downloader,
            )
            downloaded_from_youtube = True
        else:
            assert facebook_link is not None
            downloaded_path, _ = download_facebook_reel(
                facebook_link.canonical_url,
                imports_directory(cfg.data_dir),
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                downloader=facebook_downloader,
            )
            downloaded_from_facebook = True
        video_path = downloaded_path

    probe = probe_import_video(Path(video_path))
    if (
        yt_link is not None
        and not downloaded_from_youtube
        and probe.path.name == f"youtube_{yt_link.video_id}.mp4"
        and probe.path.parent.resolve() == imports_directory(cfg.data_dir).resolve()
    ):
        downloaded_from_youtube = True
    if (
        facebook_link is not None
        and not downloaded_from_facebook
        and probe.path.name == f"facebook_reel_{facebook_link.reel_id}.mp4"
        and probe.path.parent.resolve() == imports_directory(cfg.data_dir).resolve()
    ):
        downloaded_from_facebook = True
    effective_in = float(selected_in)
    if downloaded_from_youtube and yt_link is not None and yt_link.anchor_seconds is not None:
        if effective_in == 0.0 and 0.0 < yt_link.anchor_seconds < probe.duration_seconds:
            effective_in = float(yt_link.anchor_seconds)
        elif (
            effective_in >= probe.duration_seconds
            and abs(effective_in - float(yt_link.anchor_seconds)) < 1e-6
        ):
            effective_in = 0.0
    if selected_out is not None:
        raw_out = float(selected_out)
    elif downloaded_from_youtube or downloaded_from_facebook:
        raw_out = min(probe.duration_seconds, effective_in + MAX_IMPORT_RANGE_SECONDS)
    else:
        raw_out = probe.duration_seconds

    target_dir = (
        report_dir
        if report_dir is not None
        else Path("reports")
        if cfg.data_dir == "data"
        else Path(cfg.data_dir) / "reports"
    )
    pre_store = storage
    if pre_store is None and analysis_mode != "movement":
        target_dir.mkdir(parents=True, exist_ok=True)
        db_path = (
            Path(cfg.data_dir) / "tango_posture.db"
            if cfg.data_dir != "data"
            else target_dir / "import_sessions.sqlite3"
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)
        pre_store = Storage(db_path)

    clip_result = analyze_clip(
        video_path=probe.path,
        cfg=cfg,
        selected_in=effective_in,
        selected_out=raw_out,
        title=title,
        analysis_mode=analysis_mode,
        reference=reference,
        calibration_interval=calibration_interval,
        analyse_soundtrack=analyse_soundtrack,
        storage=pre_store,
        estimator_factory=estimator_factory,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        probe=probe,
        recover_tracking=recover_tracking,
        dancer_request=dancer_request,
    )

    if clip_result.selection_required:
        if progress_callback is not None:
            progress_callback(100, 100, "Choose Single or Couple analysis to continue.")
        return ImportedClipResult(
            report_path=target_dir / "_pending_selection.md",
            metadata_path=target_dir / "_pending_selection.metadata.json",
            movement_path=target_dir / "_pending_selection.movement.v1.json",
            title=clip_result.title,
            video_path=probe.path,
            source_sha256=clip_result.source_sha256,
            selected_in=clip_result.selected_in,
            selected_out=clip_result.selected_out,
            analysis_mode=analysis_mode,
            readiness=clip_result.readiness,
            reference=clip_result.reference,
            movement_data={},
            soundtrack=clip_result.soundtrack,
            coverage={},
            youtube_link=yt_link,
            facebook_reel=facebook_link,
            coach_path=None,
            coach_data=None,
            requested_dancer_mode=clip_result.requested_dancer_mode,
            resolved_dancer_mode=clip_result.resolved_dancer_mode,
            auto_decision=clip_result.auto_decision,
            couple_result=None,
            selection_required=True,
        )

    target_dir.mkdir(parents=True, exist_ok=True)

    clip_in = clip_result.selected_in
    clip_out = clip_result.selected_out
    clip_duration = clip_result.clip_duration
    source_sha = clip_result.source_sha256
    readiness = clip_result.readiness
    active_reference = clip_result.reference
    movement_data = clip_result.movement_data
    coach_data = clip_result.coach_data
    coverage = clip_result.coverage
    soundtrack_summary = clip_result.soundtrack
    frame_log = clip_result.frame_log
    scored_frame_log = clip_result.scored_frame_log
    performance = clip_result.performance
    clean_title = clip_result.title
    cap_profile = clip_result.manifest.capture_profile
    session_summary = clip_result.session_summary
    session_events = clip_result.session_events

    validated_alignment: YouTubeClockAlignment | None = None
    if yt_link is not None and downloaded_from_youtube:
        validated_alignment = YouTubeClockAlignment(
            local_source_seconds=clip_in,
            youtube_seconds=clip_in,
            confirmed_same_edit=True,
            source_sha256=source_sha,
            youtube_video_id=yt_link.video_id,
        )
    elif yt_link is not None and youtube_alignment is not None and youtube_alignment.is_valid_for(
        source_sha256=source_sha,
        youtube_video_id=yt_link.video_id,
        clip_in=clip_in,
        clip_out=clip_out,
    ):
        validated_alignment = youtube_alignment

    if progress_callback is not None:
        progress_callback(92, 100, "Writing report & companion metadata…")

    import_context = {
        "is_imported_reference": True,
        "title": clean_title,
        "source_filename": probe.path.name,
        "source_sha256": source_sha,
        "clip_in": round(clip_in, 3),
        "clip_out": round(clip_out, 3),
        "clip_duration": round(clip_duration, 3),
        "analysis_mode": analysis_mode,
        "readiness": readiness.to_dict(),
        "youtube_link": yt_link.to_dict() if yt_link is not None else None,
        "facebook_reel": facebook_link.to_dict() if facebook_link is not None else None,
        "youtube_alignment": (
            validated_alignment.to_dict() if validated_alignment is not None else None
        ),
        "coverage": {k: v.to_dict() for k, v in coverage.items()},
        "soundtrack": soundtrack_summary.to_metadata_dict(),
        "pivot_count": len(movement_data.get("pivots", [])),
        "step_count": len(movement_data.get("steps", [])),
    }

    report_path = write_markdown_report(
        report_dir=target_dir,
        cfg=cfg,
        source=f"import:{probe.path.name}",
        status="completed",
        frame_log=scored_frame_log,
        performance=performance,
        reference=active_reference,
        session_id=clip_result.session_id,
        session_summary=session_summary,
        session_events=session_events,
        recorded_video=probe.path,
        recorded_overlay=None,
        import_context=import_context,
        movement_data=movement_data,
    )

    companion_movement_path = target_dir / f"{report_path.stem}.movement.v1.json"
    save_movement_companion(companion_movement_path, movement_data)
    companion_coach_path = target_dir / f"{report_path.stem}.coach.v1.json"
    save_coach_companion(companion_coach_path, coach_data)
    companion_couple_path: Path | None = None
    if clip_result.couple_result is not None:
        companion_couple_path = target_dir / f"{report_path.stem}.couple.v1.json"
        save_couple_companion(companion_couple_path, clip_result.couple_result)

    meta_path = write_metadata(
        report_path,
        f"import:{probe.path.name}",
        probe.path,
        None,
        [float(row["timestamp"]) for row in frame_log],
        extra={
            "title": clean_title,
            "is_imported_reference": True,
            "source_sha256": source_sha,
            "clip_in": round(clip_in, 4),
            "clip_out": round(clip_out, 4),
            "analysis_mode": analysis_mode,
            "readiness": readiness.to_dict(),
            "reference_summary": (
                {
                    "id": active_reference.id,
                    "name": active_reference.name,
                    "created_at_utc": active_reference.created_at_utc,
                    "profile": active_reference.profile,
                    "baseline": active_reference.baseline,
                    "quality": active_reference.quality,
                    "model_hash": active_reference.model_hash,
                }
                if active_reference is not None
                else None
            ),
            "youtube_link": yt_link.to_dict() if yt_link is not None else None,
            "facebook_reel": facebook_link.to_dict() if facebook_link is not None else None,
            "youtube_alignment": (
                validated_alignment.to_dict() if validated_alignment is not None else None
            ),
            "movement_analysis_file": companion_movement_path.name,
            "coach_overlay_file": companion_coach_path.name,
            "couple_analysis_file": companion_couple_path.name if companion_couple_path is not None else None,
            "requested_dancer_mode": clip_result.requested_dancer_mode,
            "resolved_dancer_mode": clip_result.resolved_dancer_mode,
            "auto_decision": clip_result.auto_decision,
            "coverage": {k: v.to_dict() for k, v in coverage.items()},
            "soundtrack": soundtrack_summary.to_metadata_dict(),
            "capture_profile": cap_profile,
        },
    )

    if progress_callback is not None:
        progress_callback(100, 100, "Reference video analysis complete.")

    return ImportedClipResult(
        report_path=report_path,
        metadata_path=meta_path,
        movement_path=companion_movement_path,
        title=clean_title,
        video_path=probe.path,
        source_sha256=source_sha,
        selected_in=clip_in,
        selected_out=clip_out,
        analysis_mode=analysis_mode,
        readiness=readiness,
        reference=active_reference,
        movement_data=movement_data,
        soundtrack=soundtrack_summary,
        coverage=coverage,
        youtube_link=yt_link,
        youtube_alignment=validated_alignment,
        facebook_reel=facebook_link,
        coach_path=companion_coach_path,
        coach_data=coach_data,
        pose_recovery=clip_result.pose_recovery,
        requested_dancer_mode=clip_result.requested_dancer_mode,
        resolved_dancer_mode=clip_result.resolved_dancer_mode,
        auto_decision=clip_result.auto_decision,
        couple_result=clip_result.couple_result,
        selection_required=False,
    )
