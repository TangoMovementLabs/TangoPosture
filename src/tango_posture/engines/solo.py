"""Dedicated Solo Dancer Engine: full-body skeleton, calibration, posture & footwork."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np

from ..calibration.baseline import calibrate
from ..calibration.measurements import measure_profile
from ..calibration.models import profile_from_calibration
from ..couple import extract_single_target_observations
from ..domain import (
    METRIC_VERSION,
    Calibration,
    MetricObservation,
    PoseEstimator,
)
from ..geometry import (
    AnkleStepTracker,
    AxisCurveTracker,
    EmbraceTracker,
    PoseSmoother,
    RoleTracker,
    axis_curve_alignment,
    calculate,
    detect_role,
    head_horizontal_alignment,
    knee_bend_alignment,
    knee_geometry_kwargs,
    pixel,
    shoulder_geometry_kwargs,
    shoulder_tension_alignment,
    torso_axis_alignment,
)
from ..metrics import measure
from ..movement import analyse_movement_observations
from ..pose.recovery import run_clip_tracking_recovery
from ..references.coach import (
    _AlertTimers,
    _build_coach_frame_entry,
    _movement_note_lines,
    _serialise_pose,
    interpolate_coach_data,
)
from ..references.coverage import _compute_family_coverage
from ..references.errors import ImportCancelledError
from ..references.results import ClipAnalysisResult, RunManifest
from ..references.soundtrack import SoundtrackAnalysisSummary
from ..session import Session
from ..storage import Storage
from .common import CommonEngineContext

logger = logging.getLogger(__name__)


def run_solo_dancer_engine(
    ctx: CommonEngineContext,
    *,
    reference: Calibration | None = None,
    calibration_interval: tuple[float, float] | None = None,
    storage: Storage | None = None,
    estimator_factory: Callable[[], PoseEstimator] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    recover_tracking: bool | None = None,
    build_coach_frame_entry: Callable[..., tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> ClipAnalysisResult:
    """Execute the Solo Dancer Engine (calibration, posture, pivots, steps, gap completion)."""
    cfg = ctx.cfg
    probe = ctx.probe
    decoded_frames = list(ctx.decoded_frames)
    soundtrack_summary = ctx.soundtrack_summary or SoundtrackAnalysisSummary(
        enabled=False, available=False, status="disabled", reason=""
    )
    coach_entry_builder = build_coach_frame_entry or _build_coach_frame_entry

    analysis_region = ctx.request_obj.analysis_region if ctx.request_obj is not None else None
    anchors = list(ctx.request_obj.annotations.anchors) if ctx.request_obj is not None else []
    single_anchor = anchors[0] if anchors else None
    if single_anchor is not None or analysis_region is not None:
        filtered_single = extract_single_target_observations(
            ctx.raw_observations,
            cfg,
            anchor=single_anchor,
            analysis_region=analysis_region,
        )
        decoded_frames = [
            (f_idx, s_time, c_time, f_obs)
            for (f_idx, s_time, c_time, _), (_, f_obs) in zip(
                decoded_frames, filtered_single, strict=True
            )
        ]

    active_reference: Calibration | None = reference if ctx.analysis_mode == "posture" else None
    cal_window: tuple[float, float] | None = (
        (float(calibration_interval[0]), float(calibration_interval[1]))
        if ctx.analysis_mode == "posture" and calibration_interval is not None
        else None
    )

    if ctx.analysis_mode == "posture" and active_reference is None and cal_window is not None:
        cal_in, cal_out = cal_window
        cal_samples: list[MetricObservation] = []
        cal_head_angles: list[float] = []
        cal_shoulder_clearances: list[float] = []
        cal_axis_offsets: list[float] = []
        for _, source_time, _, frame_obs in decoded_frames:
            if cal_in - 1e-6 <= source_time <= cal_out + 1e-6:
                cal_samples.append(measure(frame_obs, cfg, None))
                if len(frame_obs.poses) == 1:
                    pose = frame_obs.poses[0]
                    raw_head = head_horizontal_alignment(
                        pose, frame_obs.width, frame_obs.height, neutral_offset=0.0
                    )
                    if raw_head is not None:
                        cal_head_angles.append(raw_head[0])
                    sh_info = shoulder_tension_alignment(
                        pose,
                        frame_obs.width,
                        frame_obs.height,
                        neutral_clearance=cfg.shoulder_neutral_clearance,
                        **shoulder_geometry_kwargs(cfg),
                    )
                    if sh_info is not None:
                        cal_shoulder_clearances.append(sh_info[1])
                    ax_info = axis_curve_alignment(
                        pose,
                        frame_obs.width,
                        frame_obs.height,
                        facing=cfg.facing,
                        side=cfg.side,
                        neutral_offset=cfg.axis_neutral_offset,
                    )
                    if ax_info is not None:
                        cal_axis_offsets.append(ax_info[1])
        cal_cfg = cfg.model_copy(
            update={
                "calibration_duration": max(
                    float(cal_out - cal_in), float(cfg.calibration_valid_duration)
                )
            }
        )
        active_reference = calibrate(
            cal_samples,
            cal_in,
            cal_cfg,
            ctx.cap_profile,
            ctx.model_hash,
            name=f"{ctx.clean_title or probe.path.stem} clip reference",
            review={
                "label": "imported-clip-interval",
                "interval": [round(cal_in, 3), round(cal_out, 3)],
            },
        )
        if cal_head_angles:
            active_reference.quality["head_neutral_offset"] = float(np.median(cal_head_angles))
        if cal_shoulder_clearances:
            active_reference.quality["shoulder_neutral_clearance"] = float(
                np.median(cal_shoulder_clearances)
            )
        if cal_axis_offsets:
            active_reference.quality["axis_neutral_offset"] = float(np.median(cal_axis_offsets))

    role_tracker = RoleTracker(assign_ms=cfg.role_assign_ms, hold_ms=cfg.role_hold_ms)
    embrace_tracker = EmbraceTracker(
        closed_threshold=cfg.embrace_closed_threshold,
        switch_sec=cfg.embrace_hysteresis_seconds,
    )
    axis_tracker = AxisCurveTracker(
        entry=cfg.axis_curve_entry,
        clear=cfg.axis_curve_clear,
        entry_duration=cfg.axis_curve_duration,
        clear_duration=cfg.axis_curve_clear_duration,
        smoothing=cfg.smoothing,
        max_gap=cfg.max_gap,
    )
    ankle_tracker = AnkleStepTracker(
        high_lift_ratio=cfg.ankle_high_lift_ratio,
        max_consecutive_steps=cfg.max_consecutive_high_ankle_steps,
        slow_tango_max_bpm=cfg.slow_tango_max_bpm,
        max_gap=cfg.max_gap,
    )

    session: Session | None = None
    if active_reference is not None:
        if storage is not None:
            storage.save_calibration(active_reference)
        scored_start = (
            cal_window[1]
            if cal_window is not None and cal_window[0] <= ctx.clip_in + 1e-3
            else ctx.clip_in
        )
        session = Session(
            active_reference,
            cfg,
            storage,
            scored_start,
            f"import:{probe.path.name}",
            {"model_hash": ctx.model_hash, "metric_version": METRIC_VERSION},
        )

    head_smoothed: float | None = None
    head_last_ts: float | None = None
    shoulder_smoothed: float | None = None
    shoulder_last_ts: float | None = None
    knee_smoothed: float | None = None
    knee_last_ts: float | None = None
    relaxed_clearances: list[float] = []
    frame_log: list[dict[str, Any]] = []

    neutral_offset = (
        float(active_reference.quality["head_neutral_offset"])
        if active_reference and "head_neutral_offset" in active_reference.quality
        else cfg.head_neutral_offset
    )
    sh_neutral_base = (
        float(active_reference.quality["shoulder_neutral_clearance"])
        if active_reference and "shoulder_neutral_clearance" in active_reference.quality
        else cfg.shoulder_neutral_clearance
    )
    axis_neutral = (
        float(active_reference.quality["axis_neutral_offset"])
        if active_reference and "axis_neutral_offset" in active_reference.quality
        else cfg.axis_neutral_offset
    )
    view_range = (
        tuple(active_reference.quality["view_range"])
        if active_reference and "view_range" in active_reference.quality
        else None
    )

    pose_smoother = PoseSmoother(smoothing_sec=cfg.overlay_smoothing, max_gap=cfg.max_gap)
    timers = _AlertTimers()
    coach_pending: dict[int, dict[str, Any]] = {}
    coach_frame_meta: dict[int, dict[str, Any]] = {}
    range_profile = (
        profile_from_calibration(active_reference) if active_reference is not None else None
    )

    for frame_index, source_time, clip_time, frame_obs in decoded_frames:
        if cancel_check is not None and cancel_check():
            raise ImportCancelledError("Reference video analysis was cancelled")
        in_cal = bool(
            cal_window is not None and cal_window[0] - 1e-6 <= source_time < cal_window[1] - 1e-6
        )

        observation = (
            measure_profile(frame_obs, cfg, range_profile)
            if range_profile is not None
            else measure(frame_obs, cfg, view_range if not in_cal else None)
        )
        session_rows: list[dict[str, Any]] = []
        if session is not None and not in_cal and source_time >= (session.start or ctx.clip_in):
            session_rows = session.update(observation)

        head_angle = None
        shoulder_elev_pct = None
        knee_bend_deg = None
        axis_dev_raw = None
        upper_body = None
        raw_role = "NONE"
        sh_neutral = (
            max(
                sh_neutral_base,
                float(np.percentile(relaxed_clearances, cfg.shoulder_adaptive_percentile)),
            )
            if len(relaxed_clearances) >= cfg.shoulder_adaptive_min_samples
            else sh_neutral_base
        )

        if len(frame_obs.poses) == 1:
            pose = frame_obs.poses[0]
            raw_role = detect_role(pose)
            metric_tau = max(
                cfg.min_filter_time_constant, cfg.overlay_smoothing or cfg.smoothing
            )
            knee_info = knee_bend_alignment(
                pose,
                frame_obs.width,
                frame_obs.height,
                side=cfg.side,
                **knee_geometry_kwargs(cfg),
            )
            if knee_info is not None:
                raw_knee_deg = knee_info[0]
                if (
                    knee_smoothed is None
                    or knee_last_ts is None
                    or (source_time - knee_last_ts) > cfg.max_gap
                ):
                    knee_smoothed = raw_knee_deg
                else:
                    dt_kn = max(0.0, source_time - knee_last_ts)
                    alpha_kn = 1.0 - math.exp(-dt_kn / metric_tau)
                    knee_smoothed += float(alpha_kn) * (raw_knee_deg - knee_smoothed)
                knee_last_ts = source_time
                knee_bend_deg = knee_smoothed

            axis_info = axis_curve_alignment(
                pose,
                frame_obs.width,
                frame_obs.height,
                facing=cfg.facing,
                side=cfg.side,
                neutral_offset=axis_neutral,
            )
            if axis_info is not None:
                axis_dev_raw = axis_info[0]

            head_info = head_horizontal_alignment(
                pose, frame_obs.width, frame_obs.height, neutral_offset=neutral_offset
            )
            if head_info is not None:
                raw_head_deg = head_info[0]
                if (
                    cfg.overlay_smoothing <= 0
                    or head_smoothed is None
                    or head_last_ts is None
                    or (source_time - head_last_ts) > cfg.max_gap
                ):
                    head_smoothed = raw_head_deg
                else:
                    dt_hd = max(0.0, source_time - head_last_ts)
                    alpha_hd = 1.0 - math.exp(-dt_hd / metric_tau)
                    head_smoothed += float(alpha_hd) * (raw_head_deg - head_smoothed)
                head_last_ts = source_time
                head_angle = head_smoothed

            sh_info = shoulder_tension_alignment(
                pose,
                frame_obs.width,
                frame_obs.height,
                neutral_clearance=sh_neutral,
                **shoulder_geometry_kwargs(cfg),
            )
            if sh_info is not None:
                raw_elev_pct, raw_clear = sh_info[0], sh_info[1]
                if (
                    not in_cal
                    and len(relaxed_clearances) < cfg.shoulder_adaptive_max_samples
                    and cfg.shoulder_adaptive_min_clearance
                    <= raw_clear
                    <= cfg.shoulder_adaptive_max_clearance
                ):
                    relaxed_clearances.append(raw_clear)
                if (
                    shoulder_smoothed is None
                    or shoulder_last_ts is None
                    or (source_time - shoulder_last_ts) > cfg.max_gap
                ):
                    shoulder_smoothed = raw_elev_pct
                else:
                    dt_sh = max(0.0, source_time - shoulder_last_ts)
                    alpha_sh = 1.0 - math.exp(-dt_sh / metric_tau)
                    shoulder_smoothed += float(alpha_sh) * (raw_elev_pct - shoulder_smoothed)
                shoulder_last_ts = source_time
                shoulder_elev_pct = shoulder_smoothed

            if all(observation.values.get(m) is not None for m in observation.values):
                upper_body = {
                    m: float(v) for m, v in observation.values.items() if v is not None
                }
            else:
                for side_cand in (cfg.side, "left", "right"):
                    ear_lm = pose.get(f"{side_cand}_ear")
                    sh_lm = pose.get(f"{side_cand}_shoulder")
                    hip_lm = pose.get(f"{side_cand}_hip")
                    if all(
                        lm is not None
                        and 0 <= lm.x <= 1
                        and 0 <= lm.y <= 1
                        and (lm.visibility is None or lm.visibility >= 0.5)
                        for lm in (ear_lm, sh_lm, hip_lm)
                    ):
                        assert ear_lm is not None and sh_lm is not None and hip_lm is not None
                        try:
                            upper_body = calculate(
                                pixel(ear_lm, frame_obs.width, frame_obs.height),
                                pixel(sh_lm, frame_obs.width, frame_obs.height),
                                pixel(hip_lm, frame_obs.width, frame_obs.height),
                                cfg.facing,
                            )
                            break
                        except ValueError:
                            upper_body = None

        role = role_tracker.update(raw_role, source_time)
        embrace_type = embrace_tracker.update(
            frame_obs.poses[0] if len(frame_obs.poses) == 1 else None,
            role,
            source_time,
        )
        axis_track = axis_tracker.update(axis_dev_raw, source_time)
        axis_curve_smoothed = (
            float(axis_track["smoothed"]) if axis_track["smoothed"] is not None else None
        )
        axis_curve_state = str(axis_track["state"])
        bad_axis = bool(axis_track["alert"])

        quality_reason = next(
            (q.reason for q in observation.quality.values() if not q.valid), ""
        )
        st_state = soundtrack_summary.frame_states.get(frame_index)
        active_bpm = (
            float(st_state["bpm"])
            if st_state and st_state.get("bpm") is not None
            else cfg.tango_bpm
        )
        ankle_track = ankle_tracker.update(
            frame_obs.poses[0] if len(frame_obs.poses) == 1 else None,
            source_time,
            bpm=active_bpm,
            width=frame_obs.width,
            height=frame_obs.height,
        )
        ankle_lift_ratio = (
            float(ankle_track["smoothed_ratio"])
            if ankle_track["smoothed_ratio"] is not None
            else None
        )
        consecutive_high_ankle_steps = int(ankle_track["consecutive_high_steps"] or 0)
        bad_ankles = bool(ankle_track["alert"])

        display_poses = pose_smoother.update(frame_obs.poses, source_time)
        if ctx.analysis_mode == "movement":
            torso = (
                torso_axis_alignment(
                    frame_obs.poses[0], frame_obs.width, frame_obs.height, cfg.confidence
                )
                if len(frame_obs.poses) == 1
                else None
            )
            coach_entry = {
                "timestamp": round(float(source_time), 4),
                "clip_timestamp": round(float(clip_time), 4),
                "poses": [_serialise_pose(pose) for pose in display_poses if pose],
                "head_angle": head_angle,
                "shoulder_elev_pct": shoulder_elev_pct,
                "knee_bend_deg": knee_bend_deg,
                "axis_curve_smoothed": axis_curve_smoothed,
                "torso_axis_tilt": abs(torso[0]) if torso is not None else None,
            }
            coach_meta: dict[str, Any] = {}
        else:
            coach_entry, coach_meta = coach_entry_builder(
                cfg=cfg,
                frame_obs=frame_obs,
                display_poses=display_poses,
                source_time=source_time,
                clip_time=clip_time,
                head_angle=head_angle,
                shoulder_elev_pct=shoulder_elev_pct,
                shoulder_neutral_clearance=sh_neutral,
                knee_bend_deg=knee_bend_deg,
                ankle_lift_ratio=ankle_lift_ratio,
                consecutive_high_ankle_steps=consecutive_high_ankle_steps,
                bad_ankles=bad_ankles,
                ankle_high_now=bool(ankle_track.get("high_now")),
                axis_curve_smoothed=axis_curve_smoothed,
                axis_curve_state=axis_curve_state,
                axis_curve_elapsed=float(axis_track.get("elapsed") or 0.0),
                bad_axis=bad_axis,
                role=role,
                embrace_type=embrace_type,
                upper_body=upper_body,
                st_state=st_state,
                quality_reason=quality_reason,
                timers=timers,
                session_rows=session_rows,
                single_pose_ok=len(frame_obs.poses) == 1,
            )
        coach_pending[frame_index] = coach_entry
        coach_entry["measurement_poses"] = [_serialise_pose(pose) for pose in frame_obs.poses]
        coach_frame_meta[frame_index] = coach_meta

        frame_log.append(
            {
                "frame_index": frame_index,
                "timestamp": source_time,
                "clip_timestamp": clip_time,
                "in_calibration_window": in_cal,
                "head_angle": head_angle,
                "shoulder_elev_pct": shoulder_elev_pct,
                "knee_bend_deg": knee_bend_deg,
                "ankle_lift_ratio": ankle_lift_ratio,
                "consecutive_high_ankle_steps": consecutive_high_ankle_steps,
                "ankle_high_alert": bad_ankles,
                "axis_curve_ratio": axis_curve_smoothed,
                "axis_curve_state": axis_curve_state,
                "axis_curve_alert": bad_axis,
                "role": role,
                "embrace_type": embrace_type,
                "upper_body": upper_body,
                "quality_reason": quality_reason,
                "beat": st_state.get("beat") if st_state else None,
                "phrase": st_state.get("phrase") if st_state else None,
                "bpm": st_state.get("bpm") if st_state else None,
                "beat_source": st_state.get("source") if st_state else None,
            }
        )

    if cancel_check is not None and cancel_check():
        raise ImportCancelledError("Reference video analysis was cancelled")

    total_source_frames = max(probe.frame_count, decoded_frames[-1][0] + 1)
    movement_observations = [
        (frame_index, obs) for frame_index, _, _, obs in decoded_frames
    ]
    movement_data = analyse_movement_observations(
        movement_observations,
        width=probe.width,
        height=probe.height,
        fps=probe.fps,
        frame_count=total_source_frames,
        sample_stride=1,
        model_hash=ctx.model_hash,
        source_sha256=ctx.source_sha,
        segment_index=-1,
        source_member=probe.path.name,
    )

    if cancel_check is not None and cancel_check():
        raise ImportCancelledError("Reference video analysis was cancelled")

    for frame_index, entry in coach_pending.items():
        meta = coach_frame_meta[frame_index]
        movement_notes = _movement_note_lines(
            movement_data, frame_index, probe.fps, cfg.replay_step_history_seconds
        )
        if ctx.analysis_mode == "movement":
            entry["notes"] = movement_notes
        else:
            entry["lines"] = list(meta["lines"])
            entry["notes"] = [meta["primary"], *meta["lines"], *movement_notes]
    coach_data: dict[str, Any] = {
        "version": 1,
        "model_hash": ctx.model_hash,
        "model_variant": cfg.analysis_model,
        "capture_profile": ctx.cap_profile,
        "source_sha256": ctx.source_sha,
        "width": probe.width,
        "height": probe.height,
        "fps": probe.fps,
        "clip_in": round(ctx.clip_in, 4),
        "clip_out": round(ctx.clip_out, 4),
        "neutral_offset": neutral_offset,
        "axis_neutral_offset": axis_neutral,
        "settings_coaching": ctx.analysis_mode != "movement",
        "frames": {str(k): v for k, v in coach_pending.items()},
    }

    should_recover = cfg.replay_recovery_auto if recover_tracking is None else bool(recover_tracking)
    pose_recovery: dict[str, Any] | None = None
    if should_recover:
        try:
            pose_recovery = run_clip_tracking_recovery(
                video_path=probe.path,
                coach_data=coach_data,
                cfg=cfg,
                movement_data=movement_data,
                model_path=ctx.model_path if ctx.model_path.is_file() else None,
                estimator_factory=estimator_factory,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
            coach_data["pose_recovery"] = pose_recovery
        except ImportCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - fallback preserves forward tracking and base analytics
            logger.warning("Tracking recovery failed; keeping forward tracking: %s", exc)
            pose_recovery = {
                "schema_version": 1,
                "status": "failed",
                "status_message": f"Tracking recovery failed ({exc}); keeping forward tracking.",
                "source_sha256": ctx.source_sha,
                "model_name": cfg.analysis_model,
                "model_sha256": ctx.model_hash,
                "source_width": probe.width,
                "source_height": probe.height,
                "video_fps": round(float(probe.fps), 4),
                "frame_count": len(decoded_frames),
                "summary": {
                    "windows_detected": 0,
                    "windows_attempted": 0,
                    "windows_recovered": 0,
                    "windows_unresolved": 0,
                    "windows_skipped_budget": 0,
                    "recovered_frames": 0,
                    "recovered_joints": 0,
                    "interpolated_joints": 0,
                },
                "windows": [],
                "frames": {},
            }
            coach_data["pose_recovery"] = pose_recovery

    try:
        coach_data = interpolate_coach_data(
            coach_data,
            cfg,
            pose_recovery=pose_recovery,
            movement_data=movement_data,
        )
    except Exception as exc:  # noqa: BLE001 - display interpolation must never abort base analysis
        logger.warning("Display pose interpolation failed; keeping forward poses: %s", exc)

    coverage = _compute_family_coverage(
        clip_duration=ctx.clip_duration,
        total_frames=len(decoded_frames),
        fps=probe.fps,
        frame_log=frame_log,
        movement_data=movement_data,
        soundtrack=soundtrack_summary,
        cfg=cfg,
        evaluate_posture=ctx.analysis_mode != "movement",
    )

    session_summary: dict[str, Any] | None = None
    session_events: list[dict[str, Any]] | None = None
    if session is not None:
        last_ts = decoded_frames[-1][1]
        session_summary = session.finish(last_ts, "completed", ctx.performance)
        session_events = [asdict(e) for v in session.evaluators.values() for e in v.events]

    scored_frame_log = [
        row for row in frame_log if not row.get("in_calibration_window", False)
    ] or frame_log

    manifest = RunManifest(
        model_variant=cfg.analysis_model,
        model_hash=ctx.model_hash,
        metric_version=METRIC_VERSION,
        capture_profile=ctx.cap_profile,
        source_sha256=ctx.source_sha,
        source_filename=probe.path.name,
        width=probe.width,
        height=probe.height,
        fps=probe.fps,
        selected_in=ctx.clip_in,
        selected_out=ctx.clip_out,
        clip_duration=ctx.clip_duration,
        analysis_mode=ctx.analysis_mode,
        processed_fps=ctx.processed_fps,
        requested_dancer_mode=ctx.requested_mode,
        resolved_dancer_mode="single",
        detector_version="couple_auto_v1",
    )

    return ClipAnalysisResult(
        manifest=manifest,
        video_path=probe.path,
        title=ctx.clean_title,
        source_sha256=ctx.source_sha,
        selected_in=ctx.clip_in,
        selected_out=ctx.clip_out,
        clip_duration=ctx.clip_duration,
        analysis_mode=ctx.analysis_mode,
        readiness=ctx.readiness,
        reference=active_reference,
        calibration_window=cal_window,
        coach_data=coach_data,
        movement_data=movement_data,
        soundtrack=soundtrack_summary,
        coverage=coverage,
        frame_log=frame_log,
        scored_frame_log=scored_frame_log,
        frame_observations=decoded_frames,
        performance=ctx.performance,
        session_id=session.id if session is not None else None,
        session_summary=session_summary,
        session_events=session_events,
        pose_recovery=pose_recovery,
        requested_dancer_mode=ctx.requested_mode,
        resolved_dancer_mode="single",
        auto_decision=ctx.auto_decision.to_dict(),
        couple_result=None,
        selection_required=False,
    )
