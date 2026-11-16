"""Dedicated Couple Dancer Engine: dual torso axes, midpoint axis, and coordination."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from ..couple import resolve_close_embrace_frame_poses, run_couple_analysis
from ..couple.tracking import _pose_torso_mid_norm, _translate_pose
from ..domain import METRIC_VERSION, FrameObservation, PoseEstimator
from ..movement import analyse_movement_observations
from ..references.errors import ImportCancelledError
from ..references.results import ClipAnalysisResult, RunManifest
from ..references.soundtrack import SoundtrackAnalysisSummary
from ..references.video import iter_virtual_clip_frames
from .common import CommonEngineContext, extract_soundtrack_beat_timestamps


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
    if estimator_factory is None and any(len(obs.poses) == 1 for _, obs in couple_observations):
        import mediapipe as mp

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
                            split_poses = [_translate_pose(p, dx, dy) for p in last_split_pair]
                            cur_obs = FrameObservation(
                                cur_obs.timestamp, cur_obs.width, cur_obs.height, split_poses
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
                                cur_obs.timestamp, cur_obs.width, cur_obs.height, split_poses
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
            couple_observations = [
                (f_idx, resolved_map.get(f_idx, obs)) for f_idx, obs in couple_observations
            ]

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
    empty_obs = [
        (f_idx, FrameObservation(s_time, probe.width, probe.height, []))
        for f_idx, s_time, _ in ctx.frame_times
    ]
    movement_data = analyse_movement_observations(
        empty_obs,
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
