"""Dedicated Couple Dancer Engine: dual torso axes, midpoint axis, and coordination."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from ..couple import resolve_close_embrace_frame_poses, run_couple_analysis
from ..couple.tracking import _pose_torso_mid_norm, _translate_pose
from ..domain import METRIC_VERSION, FrameObservation, PoseEstimator, infer_pose_origin
from ..references.errors import ImportCancelledError
from ..references.results import ClipAnalysisResult, RunManifest
from ..references.soundtrack import SoundtrackAnalysisSummary
from ..references.video import iter_virtual_clip_frames
from .common import CommonEngineContext, extract_soundtrack_beat_timestamps

_MOVEMENT_SCHEMA_VERSION = 1
_MOVEMENT_DETECTOR_VERSION = "1.0"
_MOVEMENT_DETECTOR_CONFIG: dict[str, Any] = {
    "target_sample_fps": 15.0,
    "min_landmark_confidence": 0.55,
    "step_min_landmark_confidence": 0.45,
    "frame_margin": 0.01,
    "max_tracking_gap_s": 0.35,
    "max_interp_gap_s": 0.22,
    "heading_smooth_tau_s": 0.10,
    "min_3d_hip_span": 0.035,
    "min_2d_hip_span_ratio": 0.08,
    "pivot_window_min_s": 0.35,
    "pivot_window_max_s": 0.80,
    "pivot_min_net_deg": 20.0,
    "pivot_entry_rate_deg_s": 30.0,
    "pivot_exit_rate_deg_s": 14.0,
    "pivot_trim_rate_deg_s": 12.0,
    "pivot_max_opposite_fraction": 0.22,
    "pivot_foot_anchor_radius_scale": 0.18,
    "step_smooth_tau_s": 0.10,
    "step_move_speed_scale_s": 0.30,
    "step_settle_speed_scale_s": 0.35,
    "step_settle_radius_scale": 0.045,
    "step_settle_min_duration_s": 0.10,
    "step_min_center_travel_scale": 0.075,
    "step_min_both_ends_travel_scale": 0.065,
    "step_swivel_override_travel_scale": 0.20,
    "step_min_leg_extension_ratio": 0.78,
    "step_max_airborne_lift_scale": 0.14,
    "step_debounce_s": 0.25,
}
_MOVEMENT_DETECTOR_FINGERPRINT = hashlib.sha256(
    json.dumps(_MOVEMENT_DETECTOR_CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _build_couple_movement_shell(
    ctx: CommonEngineContext,
    frame_count: int,
) -> dict[str, Any]:
    """Return an empty validated movement-analysis shell for Couple `.tpr` archives."""
    probe = ctx.probe
    sampled_frames = sum(1 for f_idx, _, _ in ctx.frame_times if 0 <= f_idx < frame_count)
    return {
        "schema_version": _MOVEMENT_SCHEMA_VERSION,
        "detector_version": _MOVEMENT_DETECTOR_VERSION,
        "detector_fingerprint": _MOVEMENT_DETECTOR_FINGERPRINT,
        "model_hash": ctx.model_hash,
        "segment_index": -1,
        "source_member": probe.path.name,
        "source_sha256": ctx.source_sha,
        "width": int(probe.width),
        "height": int(probe.height),
        "fps": float(probe.fps),
        "frame_count": int(frame_count),
        "duration": round(float(frame_count / probe.fps), 6),
        "timing_basis": "frame_index_over_fps",
        "timing_note": "",
        "sample_stride": 1,
        "coverage": {
            "sampled_frames": int(sampled_frames),
            "valid_pose_frames": 0,
            "pivot_valid_frames": 0,
            "left_foot_valid_frames": 0,
            "right_foot_valid_frames": 0,
        },
        "cut_frames": [],
        "pivots": [],
        "hip_frames": [],
        "steps": [],
    }


def _resolve_close_embrace_fallback(
    ctx: CommonEngineContext,
    couple_observations: list[tuple[int, FrameObservation]],
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[tuple[int, FrameObservation]]:
    """Run fallback close-embrace splitting when pre-flight observations hold no paired frames."""
    import mediapipe as mp

    cfg = ctx.cfg
    probe = ctx.probe
    pair_det = mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(ctx.model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=2,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
        )
    )
    obs_by_idx = dict(couple_observations)
    total_frames_n = len(couple_observations)
    resolved_map: dict[int, FrameObservation] = {}
    last_split_pos: int | None = None
    last_primary_mid: tuple[float, float] | None = None
    last_split_pair: list[dict[str, Any]] | None = None
    try:
        for pos, item in enumerate(
            iter_virtual_clip_frames(
                probe.path,
                ctx.clip_in,
                ctx.clip_out,
                probe=probe,
                cancel_check=cancel_check,
            )
        ):
            if cancel_check is not None and cancel_check():
                raise ImportCancelledError("Couple analysis was cancelled")
            cur_obs = obs_by_idx.get(item.frame_index)
            if cur_obs is None:
                continue
            if len(cur_obs.poses) == 1:
                primary_pose = cur_obs.poses[0]
                cur_mid = _pose_torso_mid_norm(primary_pose, cfg.confidence)
                propagated = False
                if (
                    pos % 3 != 0
                    and last_split_pos is not None
                    and (pos - last_split_pos) <= 2
                    and last_split_pair is not None
                    and last_primary_mid is not None
                    and cur_mid is not None
                ):
                    dx = cur_mid[0] - last_primary_mid[0]
                    dy = cur_mid[1] - last_primary_mid[1]
                    if math.hypot(dx, dy) <= 0.05:
                        split_poses = [
                            _translate_pose(p, dx, dy, origin="propagated") for p in last_split_pair
                        ]
                        cur_obs = FrameObservation(
                            cur_obs.timestamp,
                            cur_obs.width,
                            cur_obs.height,
                            split_poses,
                            pose_origins=["propagated"] * len(split_poses),
                        )
                        propagated = True
                if not propagated:
                    split_poses = resolve_close_embrace_frame_poses(
                        item.image,
                        primary_pose,
                        pair_det,
                        cur_obs.width,
                        cur_obs.height,
                        cfg,
                    )
                    if len(split_poses) >= 2:
                        cur_obs = FrameObservation(
                            cur_obs.timestamp,
                            cur_obs.width,
                            cur_obs.height,
                            split_poses,
                            pose_origins=[infer_pose_origin(p) for p in split_poses],
                        )
                        last_split_pos = pos
                        last_primary_mid = cur_mid
                        last_split_pair = split_poses
                    else:
                        last_split_pos = None
                        last_primary_mid = None
                        last_split_pair = None
            elif len(cur_obs.poses) >= 2:
                last_split_pos = None
                last_primary_mid = None
                last_split_pair = None
            resolved_map[item.frame_index] = cur_obs
            if progress_callback is not None and (pos % 10 == 0 or pos + 1 >= total_frames_n):
                pct = min(92, 80 + round(((pos + 1) / max(1, total_frames_n)) * 12))
                progress_callback(
                    pct,
                    100,
                    f"Resolving close-embrace couple axes ({pos + 1}/{total_frames_n})…",
                )
    finally:
        with suppress(Exception):
            pair_det.close()
    if resolved_map:
        return [(f_idx, resolved_map.get(f_idx, obs)) for f_idx, obs in couple_observations]
    return couple_observations


def run_couple_dancer_engine(
    ctx: CommonEngineContext,
    *,
    estimator_factory: Callable[[], PoseEstimator] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> ClipAnalysisResult:
    """Execute the Couple Axis & Coordination Engine without Solo posture/skeleton logic."""
    cfg = ctx.cfg
    probe = ctx.probe
    couple_observations = (
        list(ctx.auto_observations) if ctx.auto_observations else list(ctx.raw_observations)
    )
    if (
        estimator_factory is None
        and any(len(obs.poses) == 1 for _, obs in couple_observations)
        and not any(len(obs.poses) >= 2 for _, obs in couple_observations)
    ):
        couple_observations = _resolve_close_embrace_fallback(
            ctx,
            couple_observations,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    soundtrack_summary = ctx.soundtrack_summary or SoundtrackAnalysisSummary(
        enabled=False, available=False, status="disabled", reason=""
    )
    beat_timestamps = extract_soundtrack_beat_timestamps(soundtrack_summary, ctx.frame_times)

    couple_res = run_couple_analysis(
        couple_observations,
        cfg,
        clip_in=ctx.clip_in,
        clip_out=ctx.clip_out,
        video_sha256=ctx.source_sha,
        request=ctx.request_obj,
        auto_decision=ctx.auto_decision,
        soundtrack_beats=beat_timestamps,
    )
    couple_result_dict = couple_res.to_dict()
    total_source_frames = max(probe.frame_count, ctx.decoded_frames[-1][0] + 1)
    movement_data = _build_couple_movement_shell(ctx, total_source_frames)
    couple_coach_data: dict[str, Any] = {
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
        "neutral_offset": cfg.head_neutral_offset,
        "axis_neutral_offset": cfg.axis_neutral_offset,
        "settings_coaching": False,
        "couple_analysis": couple_result_dict,
        "frames": {
            str(f_idx): {
                "timestamp": round(float(s_time), 4),
                "clip_timestamp": round(float(c_time), 4),
                "poses": [],
                "measurement_poses": [],
            }
            for f_idx, s_time, c_time in ctx.frame_times
        },
    }
    couple_frame_log: list[dict[str, Any]] = [
        {
            "frame_index": f_idx,
            "timestamp": s_time,
            "clip_timestamp": c_time,
            "in_calibration_window": False,
        }
        for f_idx, s_time, c_time in ctx.frame_times
    ]
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
        resolved_dancer_mode="couple",
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
        reference=None,
        calibration_window=None,
        coach_data=couple_coach_data,
        movement_data=movement_data,
        soundtrack=soundtrack_summary,
        coverage={},
        frame_log=couple_frame_log,
        scored_frame_log=couple_frame_log,
        frame_observations=ctx.decoded_frames,
        performance=ctx.performance,
        session_id=None,
        session_summary=None,
        session_events=None,
        pose_recovery=None,
        requested_dancer_mode=ctx.requested_mode,
        resolved_dancer_mode="couple",
        auto_decision=ctx.auto_decision.to_dict(),
        couple_result=couple_result_dict,
        selection_required=False,
    )
