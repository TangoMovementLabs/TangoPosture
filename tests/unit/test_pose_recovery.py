"""Unit tests for display-only offline backward pose recovery and pivot interpolation."""

from __future__ import annotations

import itertools
import logging
import tempfile
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from tango_posture.archive import append_segment, file_sha256, read_analytics, store_analytics
from tango_posture.config import Config
from tango_posture.domain import FrameObservation, Landmark
from tango_posture.overlay import edge_provenance_style
from tango_posture.pose.recovery import (
    FrameIdentity,
    detect_recovery_windows,
    iter_window_frames_reversed,
    run_clip_tracking_recovery,
    validate_pose_recovery_payload,
)
from tango_posture.references.archives import promote_reference
from tango_posture.references.coach import interpolate_coach_data, render_coach_frame_overlay
from tango_posture.references.companions import (
    load_coach_companion,
    load_report_coach_data,
    save_coach_companion,
)
from tango_posture.references.errors import ImportCancelledError
from tango_posture.references.pipeline import analyze_clip
from tango_posture.review import ReportRecord


def _lm(x: float, y: float, conf: float = 0.95) -> list[float]:
    return [round(float(x), 4), round(float(y), 4), 0.0, conf, conf]


def _full_pose(*, shift: float = 0.0, missing: set[str] | None = None, low_conf: set[str] | None = None) -> dict[str, list[float]]:
    missing = missing or set()
    low_conf = low_conf or set()
    coords: dict[str, tuple[float, float]] = {
        "nose": (0.50 + shift, 0.14),
        "left_ear": (0.47 + shift, 0.15),
        "right_ear": (0.53 + shift, 0.15),
        "left_shoulder": (0.43 + shift, 0.25),
        "right_shoulder": (0.57 + shift, 0.25),
        "left_elbow": (0.40 + shift, 0.36),
        "right_elbow": (0.60 + shift, 0.36),
        "left_wrist": (0.39 + shift, 0.45),
        "right_wrist": (0.61 + shift, 0.45),
        "left_hip": (0.45 + shift, 0.48),
        "right_hip": (0.55 + shift, 0.48),
        "left_knee": (0.45 + shift, 0.66),
        "right_knee": (0.55 + shift, 0.66),
        "left_ankle": (0.45 + shift, 0.84),
        "right_ankle": (0.55 + shift, 0.84),
        "left_heel": (0.44 + shift, 0.88),
        "right_heel": (0.56 + shift, 0.88),
        "left_foot_index": (0.46 + shift, 0.90),
        "right_foot_index": (0.54 + shift, 0.90),
    }
    pose: dict[str, list[float]] = {}
    for joint, (x, y) in coords.items():
        if joint in missing:
            continue
        conf = 0.15 if joint in low_conf else 0.96
        pose[joint] = _lm(x, y, conf=conf)
    return pose


def _make_indexed_video(path: Path, frame_count: int = 16, width: int = 64, height: int = 48, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened()
    try:
        for idx in range(frame_count):
            img = np.full((height, width, 3), (idx * 13 + 7) % 250, dtype=np.uint8)
            writer.write(img)
    finally:
        writer.release()
    return path


def _build_coach_data(
    frame_poses: dict[int, dict[str, list[float]]],
    timestamps: list[float],
    *,
    width: int = 64,
    height: int = 48,
    fps: float = 10.0,
    source_sha256: str = "abc123",
    model_hash: str = "model123",
) -> dict:
    frames = {}
    for idx, ts in enumerate(timestamps):
        pose = frame_poses.get(idx, _full_pose())
        frames[str(idx)] = {
            "timestamp": round(float(ts), 6),
            "clip_timestamp": round(float(ts - timestamps[0]), 6),
            "measurement_poses": [deepcopy(pose)] if pose else [],
            "poses": [deepcopy(pose)] if pose else [],
            "calibration_rows": [{"metric": "hip_offset", "state": "UNKNOWN"}],
            "message": "",
        }
    return {
        "version": 1,
        "width": width,
        "height": height,
        "fps": fps,
        "source_sha256": source_sha256,
        "model_hash": model_hash,
        "model_variant": "heavy",
        "frames": frames,
    }


def _dict_to_landmarks(raw_pose: dict[str, list[float]]) -> dict[str, Landmark]:
    return {
        k: Landmark(x=v[0], y=v[1], z=v[2], visibility=v[3], presence=v[4])
        for k, v in raw_pose.items()
    }


def test_window_detection_shot_cuts_priority_context_and_budget():
    cfg = Config(
        replay_recovery_max_gap_s=0.6,
        replay_recovery_context_s=0.2,
        replay_recovery_budget_s=1.0,
    )
    # 30 frames at 10 fps (0.0..2.9s), cut between 18 and 19.
    timestamps = [round(i * 0.1, 3) for i in range(30)]
    missing_legs = {"left_knee", "left_ankle", "left_heel", "left_foot_index"}
    frame_poses = {i: _full_pose() for i in range(30)}
    # Window A (non-pivot, upper body present): frames 2..4 (0.3s gap)
    for f in (2, 3, 4):
        frame_poses[f] = _full_pose(missing=missing_legs)
    # Window B (inside pivot, upper body present): frames 11..13
    for f in (11, 12, 13):
        frame_poses[f] = _full_pose(missing=missing_legs)
    # Window C (too long > 0.6s for auto, frames 20..27 across only 2 post-gap frames 28,29!)
    # First put an auto-sized gap without >= 3 clean post-gap frames right before the end
    for f in (25, 26):
        frame_poses[f] = _full_pose(missing=missing_legs)

    coach = _build_coach_data(frame_poses, timestamps)
    movement = {
        "pivots": [
            {
                "start_frame": 10,
                "end_frame": 15,
                "start_time": 1.0,
                "end_time": 1.5,
                "direction": "left",
                "net_rotation_deg": 90.0,
            }
        ],
        "cut_frames": [19],
    }

    windows = detect_recovery_windows(coach, cfg, movement_data=movement)
    assert len(windows) >= 2

    # First window by priority MUST be the pivot window (frames 11..13)
    pivot_win = next(w for w in windows if w.gap_start_frame == 11)
    assert pivot_win.intersects_pivot is True
    assert pivot_win.clean_later_context is True
    assert pivot_win.budget_skipped is False
    assert windows[0].window_id == pivot_win.window_id

    # End-of-clip gap (25..26) lacks >=3 clean later frames (only 27,28,29 -> wait: 27,28,29 IS 3 frames! Let's check 26..27)
    frame_poses2 = dict(frame_poses)
    frame_poses2[25] = _full_pose()
    for f in (26, 27):
        frame_poses2[f] = _full_pose(missing=missing_legs)
    coach2 = _build_coach_data(frame_poses2, timestamps)
    windows2 = detect_recovery_windows(coach2, cfg, movement_data=movement)
    end_win = next(w for w in windows2 if w.gap_start_frame == 26)
    assert end_win.clean_later_context is False
    assert end_win.skip_reason == "Missing anchors"

    # Tight budget (0.8s) skips lowest-priority runnable window
    tight_cfg = cfg.model_copy(update={"replay_recovery_budget_s": 0.8})
    tight_windows = detect_recovery_windows(coach, tight_cfg, movement_data=movement)
    assert tight_windows[0].intersects_pivot is True
    assert tight_windows[0].budget_skipped is False
    assert any(w.budget_skipped and w.gap_start_frame == 2 for w in tight_windows)

    # Gap spanning across shot cut is split / cannot cross cut 19
    cut_poses = {i: _full_pose() for i in range(30)}
    for f in (17, 18, 19, 20):
        cut_poses[f] = _full_pose(missing=missing_legs)
    cut_coach = _build_coach_data(cut_poses, timestamps)
    cut_windows = detect_recovery_windows(cut_coach, cfg, movement_data=movement)
    for w in cut_windows:
        assert not (w.window_start_frame <= 18 and w.window_end_frame >= 19)


def test_reverse_decoding_frame_order_and_monotonic_synthetic_clock(tmp_path):
    video = _make_indexed_video(tmp_path / "indexed.avi", frame_count=12, fps=10.0)
    # Non-uniform / tight timestamps to exercise max(prev_ms + 1, round((t_last - t_i) * 1000))
    timestamps = [
        0.000, 0.100, 0.200, 0.300,
        0.400, 0.4004, 0.500, 0.600,
        0.700, 0.800, 0.900, 1.000,
    ]
    missing_left = {"left_knee", "left_ankle", "left_heel", "left_foot_index"}
    frame_poses = {i: _full_pose() for i in range(12)}
    frame_poses[5] = _full_pose(missing=missing_left)
    coach = _build_coach_data(
        frame_poses,
        timestamps,
        source_sha256=file_sha256(video),
    )

    calls: list[tuple[int, float, int]] = []

    class RecordingEstimator:
        def estimate(self, image: np.ndarray, timestamp: float) -> FrameObservation:
            pixel_val = int(image[0, 0, 0])
            frame_idx = min(
                range(12),
                key=lambda k: abs(((k * 13 + 7) % 250) - pixel_val),
            )
            ms = max((calls[-1][2] + 1) if calls else 0, round(timestamp * 1000.0))
            calls.append((frame_idx, timestamp, ms))
            return FrameObservation(timestamp, 64, 48, [_dict_to_landmarks(_full_pose())])

        def close(self) -> None:
            return

    cfg = Config(replay_recovery_max_gap_s=1.0, replay_recovery_context_s=0.3, confidence=0.5)
    payload = run_clip_tracking_recovery(
        video_path=video,
        coach_data=coach,
        cfg=cfg,
        estimator_factory=RecordingEstimator,
    )
    assert payload["status"] in ("completed", "partial")
    # Frame indices processed strictly in reverse!
    seen_frames = [c[0] for c in calls]
    assert seen_frames == sorted(seen_frames, reverse=True)
    # Synthetic detector timestamps (and ms clock) strictly increasing
    seen_ms = [c[2] for c in calls]
    assert all(b > a for a, b in itertools.pairwise(seen_ms))
    # Prediction written back to original forward identity for frame 5
    f5 = payload["frames"]["5"]
    assert f5["frame_index"] == 5
    assert f5["source_time"] == pytest.approx(timestamps[5], abs=1e-4)
    assert f5["landmarks"]["left_ankle"]["origin"] == "backward"

    # Simulated alignment mismatch surfaces explicit diagnostic cleanly
    def bad_provider(_idx: int) -> np.ndarray:
        return np.zeros((16, 16, 3), dtype=np.uint8)

    failed_payload = run_clip_tracking_recovery(
        video_path=video,
        coach_data=coach,
        cfg=cfg,
        estimator_factory=RecordingEstimator,
        frame_provider=bad_provider,
    )
    assert failed_payload["status"] == "failed"
    assert failed_payload["frames"]["5"]["diagnostics"]["_frame"] == "Frame alignment mismatch"


def test_fresh_estimator_per_disjoint_window(tmp_path):
    video = _make_indexed_video(tmp_path / "two_wins.avi", frame_count=20, fps=10.0)
    timestamps = [round(i * 0.1, 3) for i in range(20)]
    missing_left = {"left_knee", "left_ankle", "left_heel", "left_foot_index"}
    frame_poses = {i: _full_pose() for i in range(20)}
    frame_poses[3] = _full_pose(missing=missing_left)
    frame_poses[14] = _full_pose(missing=missing_left)
    coach = _build_coach_data(frame_poses, timestamps, source_sha256=file_sha256(video))

    created: list[int] = []
    closed: list[int] = []

    def factory():
        eid = len(created) + 1
        created.append(eid)

        class FakeEst:
            def estimate(self, image: np.ndarray, timestamp: float) -> FrameObservation:
                return FrameObservation(timestamp, 64, 48, [_dict_to_landmarks(_full_pose())])

            def close(self) -> None:
                closed.append(eid)

        return FakeEst()

    cfg = Config(replay_recovery_max_gap_s=0.5, replay_recovery_context_s=0.2, replay_recovery_budget_s=10.0)
    payload = run_clip_tracking_recovery(
        video_path=video,
        coach_data=coach,
        cfg=cfg,
        estimator_factory=factory,
    )
    assert payload["summary"]["windows_recovered"] == 2
    assert created == [1, 2]
    assert closed == [1, 2]


def test_anatomical_swap_jump_and_segment_length_rejection(tmp_path):
    video = _make_indexed_video(tmp_path / "gates.avi", frame_count=10, fps=10.0)
    timestamps = [round(i * 0.1, 3) for i in range(10)]
    missing_left = {"left_knee", "left_ankle", "left_heel", "left_foot_index"}
    frame_poses = {i: _full_pose() for i in range(10)}
    frame_poses[4] = _full_pose(missing=missing_left)
    coach = _build_coach_data(frame_poses, timestamps, source_sha256=file_sha256(video))
    cfg = Config(
        replay_recovery_max_gap_s=0.6,
        replay_recovery_context_s=0.3,
        replay_joint_max_gap=0.0,
        max_gap=0.05,
    )

    # Case 1: Continuity jump (> 0.30 torso lengths) / side swap -> rejected on left leg
    def swapped_est():
        class Est:
            def estimate(self, image: np.ndarray, timestamp: float) -> FrameObservation:
                bad_pose = _full_pose()
                # Jump left ankle & knee across past right side (dx = +0.20 >> 0.30 * 0.23 torso)
                bad_pose["left_knee"] = _lm(0.68, 0.66)
                bad_pose["left_ankle"] = _lm(0.68, 0.84)
                return FrameObservation(timestamp, 64, 48, [_dict_to_landmarks(bad_pose)])

            def close(self) -> None:
                return

        return Est()

    swap_payload = run_clip_tracking_recovery(
        video_path=video,
        coach_data=coach,
        cfg=cfg,
        estimator_factory=swapped_est,
    )
    assert "left_ankle" not in swap_payload["frames"]["4"]["landmarks"]
    assert "left_knee" not in swap_payload["frames"]["4"]["landmarks"]

    # Case 2: Impossible thigh segment elongation (> +30%) -> knee rejected
    def stretched_est():
        class Est:
            def estimate(self, image: np.ndarray, timestamp: float) -> FrameObservation:
                bad_pose = _full_pose()
                # Place left_knee near hip + extreme downward stretch (>30% longer thigh/shin)
                bad_pose["left_knee"] = _lm(0.45, 0.78)
                return FrameObservation(timestamp, 64, 48, [_dict_to_landmarks(bad_pose)])

            def close(self) -> None:
                return

        return Est()

    stretch_payload = run_clip_tracking_recovery(
        video_path=video,
        coach_data=coach,
        cfg=cfg,
        estimator_factory=stretched_est,
    )
    assert "left_knee" not in stretch_payload["frames"]["4"]["landmarks"]
    assert "left_ankle" in stretch_payload["frames"]["4"]["landmarks"]


def test_partial_skeleton_and_provenance_line_styles():
    origins = {
        "left_shoulder": "forward",
        "left_hip": "forward",
        "left_knee": "backward",
        "left_ankle": "backward",
        "right_hip": "forward",
        "right_knee": "interpolated",
        "right_ankle": "backward",
    }
    assert edge_provenance_style("left_shoulder", "left_hip", origins) == "solid"
    assert edge_provenance_style("left_hip", "left_knee", origins) == "dashed"
    assert edge_provenance_style("left_knee", "left_ankle", origins) == "dashed"
    assert edge_provenance_style("right_hip", "right_knee", origins) == "dotted"
    assert edge_provenance_style("right_knee", "right_ankle", origins) == "dotted"

    # Compose coach frame with forward upper body + backward left leg + interpolated right knee (partial)
    cfg = Config(max_gap=0.5, replay_pivot_interpolation_max_gap_s=0.3)
    timestamps = [0.0, 0.1, 0.2, 0.3]
    frame_poses = {
        0: _full_pose(),
        1: _full_pose(missing={"left_knee", "left_ankle", "right_knee", "right_ankle", "right_heel", "right_foot_index"}),
        2: _full_pose(missing={"left_knee", "left_ankle", "right_knee", "right_ankle", "right_heel", "right_foot_index"}),
        3: _full_pose(),
    }
    coach = _build_coach_data(frame_poses, timestamps, width=100, height=100)
    pose_recovery = {
        "schema_version": 1,
        "status": "completed",
        "status_message": "Recovered 1 of 1 window(s)",
        "source_sha256": coach["source_sha256"],
        "model_sha256": coach["model_hash"],
        "source_width": 100,
        "source_height": 100,
        "frame_count": 4,
        "windows": [],
        "frames": {
            "1": {
                "frame_index": 1,
                "source_time": 0.1,
                "clip_time": 0.1,
                "landmarks": {
                    "left_knee": {"x": 0.45, "y": 0.66, "z": 0.0, "visibility": 0.9, "presence": 0.9, "origin": "backward"},
                    "left_ankle": {"x": 0.45, "y": 0.84, "z": 0.0, "visibility": 0.9, "presence": 0.9, "origin": "backward"},
                },
                "diagnostics": {},
                "status_summary": "Recovered for display (backward pass)",
            }
        },
    }
    display_data = interpolate_coach_data(coach, cfg, pose_recovery=pose_recovery)
    f1 = display_data["frames"]["1"]
    assert f1["display_joint_origins"]["left_shoulder"] == "forward"
    assert f1["display_joint_origins"]["left_knee"] == "backward"
    assert f1["display_joint_origins"]["right_knee"] == "interpolated"
    # Raw forward observations stay free of recovered/interpolated joints
    assert "left_knee" not in f1["measurement_poses"][0]
    assert "left_knee" not in f1["display_forward_poses"][0]

    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    with_est = render_coach_frame_overlay(blank, display_data, 1, cfg, lines_only=True, show_estimates=True, forward_only=False)
    fwd_only = render_coach_frame_overlay(blank, display_data, 1, cfg, lines_only=True, show_estimates=True, forward_only=True)
    hidden_est = render_coach_frame_overlay(blank, display_data, 1, cfg, lines_only=True, show_estimates=False, forward_only=False)
    # Upper body drawn in forward_only (y=25 torso), lower limb (y=75 shin) only drawn when estimates enabled
    assert fwd_only[25, 43:58, 1].max() > 0
    assert fwd_only[70:80, :, :].sum() == 0
    assert hidden_est[70:80, :, :].sum() == 0
    assert with_est[70:80, :, 1].max() > 0


def test_measurement_and_movement_isolation_in_analyze_clip(tmp_path):
    video = _make_indexed_video(tmp_path / "pipeline_iso.avi", frame_count=12, fps=10.0)
    missing_left = {"left_knee", "left_ankle", "left_heel", "left_foot_index"}

    class FakeForwardBackEstimator:
        def __init__(self, *, is_forward: bool = False) -> None:
            self.calls = 0
            self.is_forward = is_forward

        def estimate(self, image: np.ndarray, timestamp: float) -> FrameObservation:
            self.calls += 1
            px = int(image[0, 0, 0])
            f_guess = min(range(12), key=lambda k: abs(((k * 13 + 7) % 250) - px))
            if self.is_forward and f_guess in (4, 5):
                return FrameObservation(timestamp, 64, 48, [_dict_to_landmarks(_full_pose(missing=missing_left))])
            return FrameObservation(timestamp, 64, 48, [_dict_to_landmarks(_full_pose())])

        def close(self) -> None:
            return

    instance_idx = {"n": 0}

    def factory():
        instance_idx["n"] += 1
        return FakeForwardBackEstimator(is_forward=(instance_idx["n"] == 1))

    cfg = Config(
        data_dir=str(tmp_path),
        replay_recovery_max_gap_s=0.6,
        replay_recovery_context_s=0.3,
        confidence=0.5,
    )
    progress_msgs: list[str] = []

    res_no_rec = analyze_clip(
        video_path=video,
        cfg=cfg,
        estimator_factory=lambda: FakeForwardBackEstimator(is_forward=True),
        recover_tracking=False,
    )
    instance_idx["n"] = 0
    res_rec = analyze_clip(
        video_path=video,
        cfg=cfg,
        estimator_factory=factory,
        recover_tracking=True,
        progress_callback=lambda _c, _t, msg: progress_msgs.append(msg),
    )

    assert any("Analyzing forward" in m for m in progress_msgs)
    assert any("Recovering tracking in 1 of 1 windows" in m for m in progress_msgs)
    assert res_rec.pose_recovery is not None
    assert res_rec.pose_recovery["summary"]["windows_recovered"] == 1
    # Strict invariant: measurement poses, frame log, coverage, pivots, steps stay IDENTICAL
    assert res_rec.frame_log == res_no_rec.frame_log
    assert res_rec.coverage == res_no_rec.coverage
    assert res_rec.movement_data["pivots"] == res_no_rec.movement_data["pivots"]
    assert res_rec.movement_data["steps"] == res_no_rec.movement_data["steps"]
    assert res_rec.coach_data["frames"]["4"]["measurement_poses"] == res_no_rec.coach_data["frames"]["4"]["measurement_poses"]
    assert "left_ankle" not in res_rec.coach_data["frames"]["4"]["measurement_poses"][0]
    assert "left_ankle" in res_rec.coach_data["frames"]["4"]["display_poses"][0]


def test_archive_companion_validation_and_rebase(tmp_path, caplog):
    video = _make_indexed_video(tmp_path / "arch.avi", frame_count=10, fps=10.0)
    sha = file_sha256(video)
    timestamps = [round(i * 0.1, 3) for i in range(10)]
    frame_poses = {i: _full_pose() for i in range(10)}
    frame_poses[4] = _full_pose(missing={"left_ankle"})
    coach = _build_coach_data(frame_poses, timestamps, source_sha256=sha, model_hash="mhash")
    valid_rec = {
        "schema_version": 1,
        "status": "completed",
        "status_message": "Recovered 1 window",
        "source_sha256": sha,
        "model_sha256": "mhash",
        "source_width": 64,
        "source_height": 48,
        "frame_count": 10,
        "windows": [
            {
                "window_id": "shot0_f2_7",
                "gap_start_frame": 4,
                "gap_end_frame": 4,
                "window_start_frame": 2,
                "window_end_frame": 7,
                "gap_start_time": 0.4,
                "gap_end_time": 0.4,
            }
        ],
        "frames": {
            "4": {
                "frame_index": 4,
                "source_time": 0.4,
                "clip_time": 0.4,
                "landmarks": {
                    "left_ankle": {
                        "x": 0.45,
                        "y": 0.84,
                        "z": 0.0,
                        "visibility": 0.9,
                        "presence": 0.9,
                        "origin": "backward",
                    }
                },
                "diagnostics": {},
            }
        },
        "summary": {"windows_detected": 1, "recovered_frames": 1, "recovered_joints": 1, "interpolated_joints": 0},
    }

    # Companion validation: corrupt/stale pose_recovery logs warning and strips only pose_recovery
    stale_rec = {**valid_rec, "source_sha256": "wrong-hash"}
    comp_path = tmp_path / "rep.coach.v1.json"
    save_coach_companion(comp_path, {**coach, "pose_recovery": stale_rec})
    with caplog.at_level(logging.WARNING):
        loaded = load_coach_companion(comp_path)
    assert loaded is not None
    assert "pose_recovery" not in loaded
    assert len(loaded["frames"]) == 10
    assert any("pose_recovery" in rec.message for rec in caplog.records)

    # Archive load: invalid schema/hash strips pose_recovery and keeps base analytics & video
    tp_path = tmp_path / "practice.tp"
    rep_md = tmp_path / "rep.md"
    rep_md.write_text("# Report\n", encoding="utf-8")
    append_segment(tp_path, rep_md, video, None, timestamps)
    corrupt_rec = {**valid_rec, "frames": {"4": {"frame_index": 4, "source_time": 0.4, "clip_time": 0.4, "landmarks": {"left_ankle": {"x": float("nan") if False else 1.5, "y": 0.5, "visibility": 0.9, "presence": 0.9, "origin": "backward"}}}}}
    store_analytics(tp_path, 0, {"frame_timestamps": timestamps, "frames": [{"frame_index": i, "timestamp": timestamps[i]} for i in range(10)], "coach_data": {**coach, "pose_recovery": corrupt_rec}})
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        arch_analytics = read_analytics(tp_path, 0)
    assert arch_analytics is not None
    assert "pose_recovery" not in arch_analytics["coach_data"]
    assert len(arch_analytics["frames"]) == 10

    # Valid save & rebase during promote_reference excerpt [2, 8)
    store_analytics(tp_path, 0, {"frame_timestamps": timestamps, "frames": [{"frame_index": i, "timestamp": timestamps[i]} for i in range(10)], "coach_data": {**coach, "pose_recovery": valid_rec}})
    ref_path = tmp_path / "excerpt.tpr"
    promote_reference(tp_path, ref_path, "Excerpt", index=0, start_frame=2, end_frame=8)
    ref_analytics = read_analytics(ref_path, 0)
    assert ref_analytics is not None
    rebased = ref_analytics["coach_data"].get("pose_recovery")
    assert rebased is not None
    assert rebased["frame_count"] == 6
    assert "2" in rebased["frames"]
    assert rebased["windows"][0]["window_start_frame"] == 0
    assert rebased["windows"][0]["window_end_frame"] == 5
    validate_pose_recovery_payload(rebased, frame_count=6, source_sha256=ref_analytics["coach_data"]["source_sha256"])

    # Also test ReportRecord loading with mismatched frame count strips pose_recovery cleanly
    bad_count_rec = {**valid_rec, "frame_count": 99}
    save_coach_companion(comp_path, {**coach, "pose_recovery": bad_count_rec})
    report_rec = ReportRecord(path=tmp_path / "rep.md", markdown="# Rep\n", videos={"Raw video": video}, timestamps=timestamps, source_sha256=sha, coach_overlay_file=comp_path.name)
    loaded_rep = load_report_coach_data(report_rec)
    assert loaded_rep is not None and "pose_recovery" not in loaded_rep


def test_disk_spooling_and_cancellation_cleanup():
    spool_before = set(Path(tempfile.gettempdir()).glob("tp_recovery_*.spool"))
    identities = [
        FrameIdentity(frame_index=i, source_time=i * 0.1, clip_time=i * 0.1)
        for i in range(6)
    ]

    def provider(idx: int) -> np.ndarray:
        return np.full((24, 32, 3), idx * 30, dtype=np.uint8)

    # Normal execution exceeding max_memory_bytes forces disk spooling, then cleans up
    seen: list[int] = []
    with iter_window_frames_reversed(
        None,
        identities,
        expected_width=32,
        expected_height=24,
        max_memory_bytes=200,
        frame_provider=provider,
    ) as rev:
        for ident, arr in rev:
            seen.append(ident.frame_index)
            assert int(arr[0, 0, 0]) == ident.frame_index * 30
    assert seen == [5, 4, 3, 2, 1, 0]
    assert set(Path(tempfile.gettempdir()).glob("tp_recovery_*.spool")) == spool_before

    # Cancellation during reversed iteration triggers ImportCancelledError and removes spool file
    flag = {"cancel": False}
    with (
        pytest.raises(ImportCancelledError),
        iter_window_frames_reversed(
            None,
            identities,
            expected_width=32,
            expected_height=24,
            max_memory_bytes=200,
            cancel_check=lambda: flag["cancel"],
            frame_provider=provider,
        ) as rev,
    ):
        for ident, _arr in rev:
            flag["cancel"] = True
    assert set(Path(tempfile.gettempdir()).glob("tp_recovery_*.spool")) == spool_before
