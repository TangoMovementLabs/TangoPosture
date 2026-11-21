"""Comprehensive unit tests for Auto/Single/Couple detection, tracking, math, archive, and CSV."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from tango_posture.archive import (
    append_segment,
    file_sha256,
    read_analytics,
    set_active_dancer_run,
    store_analytics,
    update_couple_annotations,
)
from tango_posture.config import Config
from tango_posture.couple import (
    CoupleAnalysisRequest,
    CoupleAnnotationItem,
    CoupleAnnotations,
    DancerAnchor,
    PlannedTransitionMark,
    classify_opening_window,
    export_couple_csv,
    extract_single_target_observations,
    rebuild_couple_result_with_annotations,
    render_couple_frame_overlay,
    run_couple_analysis,
    track_couple_frames,
)
from tango_posture.domain import FrameObservation, Landmark
from tango_posture.references.companions import load_couple_companion, save_couple_companion
from tango_posture.references.pipeline import analyse_imported_clip, analyze_clip


def _make_person_pose(
    *,
    hip_x: float,
    hip_y: float = 0.62,
    shoulder_dx: float = 0.0,
    torso_dy: float = 0.22,
    shoulder_half_w: float = 0.04,
    hip_half_w: float = 0.03,
    shoulder_vis: float = 0.95,
    hip_vis: float = 0.95,
    ankle_dx: float = 0.0,
    vis: float = 0.95,
    nose_dy: float = 0.06,
) -> dict[str, Landmark]:
    """Create a complete MediaPipe-style 2D pose centered at (hip_x, hip_y)."""
    sh_x = hip_x + shoulder_dx
    sh_y = hip_y - torso_dy
    knee_y = hip_y + 0.14
    ankle_y = hip_y + 0.28
    pose = {
        "nose": Landmark(sh_x, sh_y - nose_dy, vis, vis, 0.0),
        "left_ear": Landmark(sh_x - 0.015, sh_y - nose_dy, vis, vis, 0.0),
        "right_ear": Landmark(sh_x + 0.015, sh_y - nose_dy, vis, vis, 0.0),
        "left_shoulder": Landmark(sh_x - shoulder_half_w, sh_y, shoulder_vis, shoulder_vis, 0.0),
        "right_shoulder": Landmark(sh_x + shoulder_half_w, sh_y, shoulder_vis, shoulder_vis, 0.0),
        "left_hip": Landmark(hip_x - hip_half_w, hip_y, hip_vis, hip_vis, 0.0),
        "right_hip": Landmark(hip_x + hip_half_w, hip_y, hip_vis, hip_vis, 0.0),
        "left_knee": Landmark(hip_x - hip_half_w, knee_y, vis, vis, 0.0),
        "right_knee": Landmark(hip_x + hip_half_w, knee_y, vis, vis, 0.0),
        "left_ankle": Landmark(hip_x - hip_half_w + ankle_dx, ankle_y, vis, vis, 0.0),
        "right_ankle": Landmark(hip_x + hip_half_w - ankle_dx, ankle_y, vis, vis, 0.0),
    }
    return pose


def _obs(
    timestamp: float,
    poses: list[dict[str, Landmark]],
    width: int = 1000,
    height: int = 1000,
) -> FrameObservation:
    return FrameObservation(timestamp=timestamp, width=width, height=height, poses=poses)


def _write_video(path: Path, *, fps: float = 10.0, duration_s: float = 20.0) -> Path:
    width, height = 160, 120
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (width, height)
    )
    assert writer.isOpened()
    total = round(fps * duration_s)
    frame = np.full((height, width, 3), 32, dtype=np.uint8)
    for _ in range(total):
        writer.write(frame)
    writer.release()
    return path


def test_opening_window_bounds_and_short_clip():
    cfg = Config()
    cfg.dancer_analysis.auto_detection_window_s = 15.0

    # Offset excerpt [35.0, 95.0]: window must be [35.0, 50.0), so t=50.0 is excluded
    obs_list = []
    for i in range(600):
        t = 35.0 + i * 0.1
        # Couple present and smoothly co-moving across [35.0, 49.9]
        dx = 0.20 * math.sin(0.5 * (t - 35.0))
        p_a = _make_person_pose(hip_x=0.35 + dx)
        p_b = _make_person_pose(hip_x=0.50 + dx)
        obs_list.append((i, _obs(t, [p_a, p_b])))

    decision = classify_opening_window(
        obs_list,
        cfg,
        clip_in=35.0,
        clip_out=95.0,
        requested_mode="auto",
    )
    assert decision.effective_window_in == pytest.approx(35.0)
    assert decision.effective_window_out == pytest.approx(50.0)
    assert decision.effective_duration_s == pytest.approx(15.0)
    assert decision.resolved_mode == "couple"

    # Frame exactly at 50.0 s must not push effective_window_out past 50.0
    short_obs = [
        (i, _obs(i * 0.1, [_make_person_pose(hip_x=0.35 + 0.02 * math.sin(i * 0.25))]))
        for i in range(60)
    ]
    cfg.dancer_analysis.auto_detection_window_s = 10.0
    short_dec = classify_opening_window(
        short_obs,
        cfg,
        clip_in=0.0,
        clip_out=6.0,
        requested_mode="auto",
    )
    assert short_dec.effective_duration_s == pytest.approx(6.0)
    assert short_dec.effective_window_out == pytest.approx(6.0)
    assert short_dec.resolved_mode == "single"


def test_auto_classification_single_vs_couple_vs_needs_selection():
    cfg = Config()
    cfg.dancer_analysis.auto_detection_window_s = 10.0

    # 1. Clear Single with brief bystander (< brief_second_max_s = 0.5 s)
    single_obs = []
    for i in range(100):
        t = i * 0.1
        p_main = _make_person_pose(hip_x=0.30 + 0.004 * i)
        poses = [p_main]
        if 30 <= i <= 33:  # 0.4 s transient passerby below default 0.5 s
            poses.append(_make_person_pose(hip_x=0.80))
        single_obs.append((i, _obs(t, poses)))
    dec_single = classify_opening_window(
        single_obs, cfg, clip_in=0.0, clip_out=12.0, requested_mode="auto"
    )
    assert dec_single.resolved_mode == "single"

    # 2. Clear Couple: co-moving in embrace distance across >= 2 s
    couple_obs = []
    for i in range(100):
        t = i * 0.1
        x = 0.25 + 0.005 * i
        couple_obs.append(
            (i, _obs(t, [_make_person_pose(hip_x=x), _make_person_pose(hip_x=x + 0.12)]))
        )
    dec_couple = classify_opening_window(
        couple_obs, cfg, clip_in=0.0, clip_out=12.0, requested_mode="auto"
    )
    assert dec_couple.resolved_mode == "couple"

    # 3. Persistent distant static bystander -> ambiguous -> Needs selection
    ambig_obs = []
    for i in range(100):
        t = i * 0.1
        p_main = _make_person_pose(hip_x=0.20 + 0.004 * i)
        p_bystander = _make_person_pose(hip_x=0.85, torso_dy=0.08)  # far apart, stationary
        ambig_obs.append((i, _obs(t, [p_main, p_bystander])))
    dec_ambig = classify_opening_window(
        ambig_obs, cfg, clip_in=0.0, clip_out=12.0, requested_mode="auto"
    )
    assert dec_ambig.resolved_mode == "needs_selection"


def test_pipeline_single_pass_inference_and_needs_selection_early_abort(tmp_path):
    video = _write_video(tmp_path / "couple_20s.avi", fps=5.0, duration_s=20.0)
    cfg = Config()
    cfg.dancer_analysis.auto_detection_window_s = 4.0

    # Clear moving couple across all 100 frames
    class _CountingCoupleEstimator:
        def __init__(self) -> None:
            self.calls = 0

        def estimate(self, _image: np.ndarray, timestamp: float) -> FrameObservation:
            self.calls += 1
            x = 0.20 + 0.025 * timestamp
            return _obs(
                timestamp,
                [_make_person_pose(hip_x=x), _make_person_pose(hip_x=x + 0.10)],
                width=160,
                height=120,
            )

        def close(self) -> None:
            pass

    counter = _CountingCoupleEstimator()
    progress_msgs: list[str] = []
    req = CoupleAnalysisRequest(requested_mode="auto")
    res = analyze_clip(
        video_path=video,
        cfg=cfg,
        estimator_factory=lambda: counter,
        progress_callback=lambda _c, _t, msg: progress_msgs.append(msg),
        dancer_request=req,
    )
    assert res.resolved_dancer_mode == "couple"
    assert not res.selection_required
    assert res.couple_result is not None
    # Single pass across 100 decoded frames (zero duplicate opening window inference)
    assert counter.calls == len(res.frame_log) == 100
    assert any("Detecting number of dancers" in m for m in progress_msgs)

    # Ambiguous opening window -> Needs selection aborts BEFORE running post-window frames
    class _AmbiguousEstimator:
        def __init__(self) -> None:
            self.calls = 0

        def estimate(self, _image: np.ndarray, timestamp: float) -> FrameObservation:
            self.calls += 1
            p_main = _make_person_pose(hip_x=0.18 + 0.02 * timestamp)
            p_other = _make_person_pose(hip_x=0.88, torso_dy=0.06)
            return _obs(timestamp, [p_main, p_other], width=160, height=120)

        def close(self) -> None:
            pass

    ambig_counter = _AmbiguousEstimator()
    report_dir = tmp_path / "reports"
    imp_res = analyse_imported_clip(
        video_path=video,
        cfg=cfg,
        report_dir=report_dir,
        estimator_factory=lambda: ambig_counter,
        dancer_request=CoupleAnalysisRequest(requested_mode="auto"),
    )
    assert imp_res.selection_required is True
    assert imp_res.resolved_dancer_mode == "needs_selection"
    assert not imp_res.report_path.exists()
    assert not report_dir.exists() or not any(report_dir.iterdir())
    # Stopped right after the 4s opening window (~20 frames out of 100)
    assert ambig_counter.calls < 35

    # Forced override to Single runs full clip & strips background bystander
    single_counter = _AmbiguousEstimator()
    forced_single = analyze_clip(
        video_path=video,
        cfg=cfg,
        estimator_factory=lambda: single_counter,
        dancer_request=CoupleAnalysisRequest(
            requested_mode="single",
            forced_override=True,
            annotations=CoupleAnnotations(
                anchors=[DancerAnchor(dancer_id="A", timestamp=0.0, frame_index=0, hip_xy=(0.18, -0.62))]
            ),
        ),
    )
    assert forced_single.resolved_dancer_mode == "single"
    assert not forced_single.selection_required
    assert single_counter.calls == 100
    # Verify single-target extractor drops bystander
    raw_obs = [(i, single_counter.estimate(np.zeros((120, 160, 3), np.uint8), i * 0.2)) for i in range(10)]
    filtered = extract_single_target_observations(
        raw_obs,
        cfg,
        anchor=DancerAnchor(dancer_id="A", timestamp=0.0, frame_index=0, hip_xy=(0.18, -0.62)),
    )
    assert all(len(obs.poses) == 1 for _, obs in filtered)
    assert all(obs.poses[0]["left_hip"].x < 0.5 for _, obs in filtered)


def test_tracking_crossing_swapped_order_close_embrace_nms_and_shot_cut():
    cfg = Config()
    # Two dancers start separated horizontally, cross each other smoothly, AND detector swaps list order!
    obs_list: list[tuple[int, FrameObservation]] = []
    n = 30
    for i in range(n):
        t = i * 0.1
        xa = 0.20 + 0.02 * i  # moves right: 0.20 -> 0.78
        xb = 0.78 - 0.02 * i  # moves left:  0.78 -> 0.20
        # Different heights slightly to check torso length + close crossing
        pa = _make_person_pose(hip_x=xa, torso_dy=0.24)
        pb = _make_person_pose(hip_x=xb, torso_dy=0.20)
        # Swap detector array order every frame after frame 10!
        poses = [pb, pa] if i >= 10 and i % 2 == 0 else [pa, pb]
        # Add near-duplicate NMS phantom for dancer A on frame 5 (must be suppressed!)
        if i == 5:
            dup_a = _make_person_pose(hip_x=xa + 0.005, hip_y=0.625, torso_dy=0.24)
            poses.append(dup_a)
        obs_list.append((i, _obs(t, poses)))

    tracked = track_couple_frames(obs_list, cfg)
    # Track A follows xa monotonically rightward even across crossing and swapped pose list order
    xs_a = [pf.dancer_a.hip_px[0] for pf in tracked.frames if pf.dancer_a and pf.dancer_a.hip_px]
    xs_b = [pf.dancer_b.hip_px[0] for pf in tracked.frames if pf.dancer_b and pf.dancer_b.hip_px]
    assert len(xs_a) >= 25 and len(xs_b) >= 25
    assert xs_a[-1] > xs_a[0]
    assert xs_b[-1] < xs_b[0]

    # Close embrace: two distinct dancers separated horizontally (~0.08 torso, i.e., close side-by-side)
    # should NOT be collapsed by duplicate NMS
    embrace_obs = [
        (i, _obs(i * 0.1, [_make_person_pose(hip_x=0.45), _make_person_pose(hip_x=0.55)]))
        for i in range(10)
    ]
    embrace_tracked = track_couple_frames(embrace_obs, cfg)
    assert all(pf.pair_hip_eligible for pf in embrace_tracked.frames)

    # Shot cut mid-clip (>= 1.5 torso jump + pose change) triggers break flag
    cut_obs = []
    for i in range(12):
        t = i * 0.1
        if i < 6:
            cut_obs.append((i, _obs(t, [_make_person_pose(hip_x=0.20), _make_person_pose(hip_x=0.35)])))
        else:
            cut_obs.append((i, _obs(t, [_make_person_pose(hip_x=0.80, hip_y=0.40), _make_person_pose(hip_x=0.90, hip_y=0.40)])))
    cut_tracked = track_couple_frames(cut_obs, cfg)
    assert any(pf.shot_cut for pf in cut_tracked.frames)


def test_display_only_interpolation_does_not_leak_into_metrics():
    cfg = Config()
    cfg.dancer_analysis.display_interpolation_max_gap_s = 0.15
    cfg.dancer_analysis.max_observation_gap_s = 0.15

    # Build 20 Hz stream (0.05 s step):
    # - Short gap of 0.10 s (frames 5, 6 missing -> duration between frame 4 at 0.20 and frame 7 at 0.35 is 0.15 s)
    # - Long gap of 0.25 s (frames 12..16 missing -> duration 0.30 s > 0.15 s)
    obs_list = []
    for i in range(25):
        t = round(i * 0.05, 4)
        pa = _make_person_pose(hip_x=0.30 + 0.005 * i)
        if i in (5, 6):
            # Dancer B absent for <= 0.15 s (t=0.20 to t=0.35)
            obs_list.append((i, _obs(t, [pa])))
        elif i in (12, 13, 14, 15, 16):
            # Dancer B absent for > 0.15 s (t=0.55 to t=0.85)
            obs_list.append((i, _obs(t, [pa])))
        else:
            pb = _make_person_pose(hip_x=0.45 + 0.005 * i)
            obs_list.append((i, _obs(t, [pa, pb])))

    res = run_couple_analysis(
        obs_list,
        cfg,
        clip_in=0.0,
        clip_out=1.25,
        request=CoupleAnalysisRequest(requested_mode="couple", forced_override=True),
        video_sha256="sha-interp",
    ).to_dict()

    overlays = res["frame_overlays"]
    # Short gap frames 5, 6 must have display_pose with interpolated_display=True
    assert overlays[5]["dancer_b"] is not None
    assert overlays[5]["dancer_b"]["interpolated_display"] is True
    assert overlays[6]["dancer_b"]["interpolated_display"] is True
    # Yet measured torso_tilt_b and pair_midpoint_tilt MUST stay unavailable (None) at frames 5, 6!
    assert res["series"]["torso_tilt_b"][5]["value"] is None
    assert res["series"]["torso_tilt_b"][5]["status"] == "unavailable"
    assert res["series"]["couple_midpoint_tilt"][5]["value"] is None
    assert res["series"]["couple_midpoint_tilt"][5]["status"] == "unavailable"
    assert res["series"]["velocity_b"][5]["value"] is None
    # Long gap frame 14 (> 0.15 s) must NOT have display_pose at all
    assert overlays[14]["dancer_b"] is None

    # Smoke test overlay rendering uses dashed/hollow styling without crashing
    canvas = np.zeros((240, 320, 3), dtype=np.uint8)
    rendered = render_couple_frame_overlay(canvas, overlays[5])
    assert rendered.shape == canvas.shape
    assert int(rendered.sum()) > 0


def test_exact_worked_math_lag_torso_axes_beat_offset_turns_and_planned_reversal():
    cfg = Config()
    cfg.dancer_analysis.start_speed = 0.25
    cfg.dancer_analysis.stop_speed = 0.10
    cfg.dancer_analysis.persistence_s = 0.15
    cfg.dancer_analysis.auto_turn_suggestions = False

    # Case 1: Travel lag B - A = +100 ms & upright pair axis despite opposite +-6.84 deg individual tilts
    # Scale torso_dy = 0.25 -> scale_px = 250 px on 1000x1000 image. Speed > 0.25 torso/s => dx/dt > 0.0625 norm/s.
    obs_list: list[tuple[int, FrameObservation]] = []
    fps = 50.0  # 20 ms steps for crisp event crossing resolution
    for i in range(180):  # 3.6 s
        t = round(i / fps, 4)
        # Dancer A starts moving at t=1.00 s (speed = 0.60 torso/s -> dx_norm/s = 0.15)
        xa = 0.20 if t < 1.00 else min(0.75, 0.20 + 0.15 * (t - 1.00))
        # Dancer B starts moving at t=1.10 s (+100 ms lag!)
        xb = 0.30 if t < 1.10 else min(0.85, 0.30 + 0.15 * (t - 1.10))
        # Torso tilt: A leans +0.03 right over torso_dy=0.25 => atan2(0.03, 0.25) = +6.84277 deg
        # B leans -0.03 left over torso_dy=0.25 => -6.84277 deg
        pa = _make_person_pose(hip_x=xa, shoulder_dx=0.03, torso_dy=0.25)
        pb = _make_person_pose(hip_x=xb, shoulder_dx=-0.03, torso_dy=0.25)
        obs_list.append((i, _obs(t, [pa, pb])))

    # Add user annotations:
    # - Beat at 1.500 s, user-confirmed contact for A at 1.530 s => +30 ms offset
    # - Turn start=2.00 s, end=2.80 s, exit=2.86 s => exit_minus_turn_end = +60 ms
    anns = CoupleAnnotations(
        items=[
            CoupleAnnotationItem(
                annotation_id="beat_1",
                event_type="musical_target",
                subject="beat",
                timestamp=1.500,
                status="user_confirmed",
            ),
            CoupleAnnotationItem(
                annotation_id="step_1",
                event_type="contact",
                subject="A",
                timestamp=1.530,
                status="user_confirmed",
            ),
            CoupleAnnotationItem(
                annotation_id="ts_1",
                event_type="turn_start",
                subject="pair",
                timestamp=2.000,
                status="user_confirmed",
                pivot_behavior="A_pivots_on_place",
            ),
            CoupleAnnotationItem(
                annotation_id="te_1",
                event_type="turn_end",
                subject="pair",
                timestamp=2.800,
                status="user_confirmed",
            ),
            CoupleAnnotationItem(
                annotation_id="ex_1",
                event_type="exit",
                subject="pair",
                timestamp=2.860,
                status="user_confirmed",
            ),
        ]
    )

    res = run_couple_analysis(
        obs_list,
        cfg,
        clip_in=0.0,
        clip_out=3.6,
        request=CoupleAnalysisRequest(
            requested_mode="couple",
            forced_override=True,
            annotations=anns,
        ),
        soundtrack_beats=[],
        video_sha256="sha-math",
        run_id="run-math-1",
    ).to_dict()

    summaries = res["summaries"]
    # Lag B - A ≈ +100 ms (within discrete regression tolerance)
    start_lag = summaries["coordination"]["start_lag_ms"]
    assert start_lag["status"] == "measured"
    assert start_lag["value"] == pytest.approx(100.0, abs=35.0)

    # Signed tilts: +6.84 deg, -6.84 deg, upright pair midpoint = 0.0 deg, separation = 13.69 deg
    posture = summaries["posture"]
    assert posture["torso_tilt_a"]["median_deg"] == pytest.approx(6.84, abs=0.1)
    assert posture["torso_tilt_b"]["median_deg"] == pytest.approx(-6.84, abs=0.1)
    assert posture["couple_midpoint_tilt"]["median_deg"] == pytest.approx(0.0, abs=0.1)
    assert posture["torso_separation"]["median_deg"] == pytest.approx(13.69, abs=0.1)

    # Same direction (+6.84 and +6.84) -> torso_separation = 0.0 deg, midpoint = +6.84 deg
    parallel_obs = [
        (
            i,
            _obs(
                i * 0.1,
                [
                    _make_person_pose(hip_x=0.30, shoulder_dx=0.03, torso_dy=0.25),
                    _make_person_pose(hip_x=0.50, shoulder_dx=0.03, torso_dy=0.25),
                ],
            ),
        )
        for i in range(10)
    ]
    parallel_res = run_couple_analysis(
        parallel_obs,
        cfg,
        clip_in=0.0,
        clip_out=1.0,
        request=CoupleAnalysisRequest(requested_mode="couple", forced_override=True),
    ).to_dict()
    assert parallel_res["summaries"]["posture"]["torso_separation"]["median_deg"] == pytest.approx(0.0, abs=0.05)
    assert parallel_res["summaries"]["posture"]["couple_midpoint_tilt"]["median_deg"] == pytest.approx(6.84, abs=0.1)

    # Beat offset +30 ms on contact A
    a_contacts = [e for e in res["events"] if e["event_type"] == "contact" and e["subject"] == "A"]
    assert any(e["beat_offset_ms"] == pytest.approx(30.0, abs=0.1) for e in a_contacts)

    # Turn duration 0.80 s and exit_minus_turn_end +60.0 ms
    assert summaries["turns"]["turn_count"] == 1
    assert summaries["turns"]["median_turn_duration_s"] == pytest.approx(0.80, abs=0.01)
    assert summaries["turns"]["median_exit_minus_turn_end_ms"] == pytest.approx(60.0, abs=0.1)


def test_planned_reversal_exclusion_annotations_rebuild_archive_companion_and_csv(tmp_path):
    cfg = Config()
    cfg.dancer_analysis.start_speed = 0.25
    cfg.dancer_analysis.stop_speed = 0.10
    cfg.dancer_analysis.persistence_s = 0.15

    # Travel right, slow, then reverse direction at t = 1.6..2.4 s
    obs_list = []
    fps = 20.0
    for i in range(60):  # 3.0 s
        t = i / fps
        if t < 1.2:
            x = 0.20 + 0.18 * t
        elif t < 1.6:
            x = 0.20 + 0.18 * 1.2 + 0.04 * (t - 1.2)
        else:
            x = 0.20 + 0.18 * 1.2 + 0.04 * 0.4 - 0.16 * (t - 1.6)
        obs_list.append(
            (i, _obs(t, [_make_person_pose(hip_x=x), _make_person_pose(hip_x=x + 0.08)]))
        )

    base = run_couple_analysis(
        obs_list,
        cfg,
        clip_in=0.0,
        clip_out=3.0,
        request=CoupleAnalysisRequest(requested_mode="couple", forced_override=True),
        video_sha256="sha-rev",
        run_id="run-rev-1",
    ).to_dict()
    cand_count_before = len(base["automated_candidates"])
    assert cand_count_before >= 1

    # Initially without planned mark, reversal should register as unintended_reversal
    assert base["summaries"]["transitions"]["unintended_reversals"] >= 1
    assert base["summaries"]["transitions"]["planned_reversals"] == 0

    # Add planned_reversal mark covering reversal window -> excluded from unintended, counted in planned
    anns_planned = CoupleAnnotations(
        planned_transitions=[
            PlannedTransitionMark(
                mark_id="pm_rev_1",
                kind="planned_reversal",
                timestamp=1.55,
                window_end=2.40,
                note="Intended back ochos",
            )
        ]
    )
    rebuilt = rebuild_couple_result_with_annotations(base, anns_planned, cfg)
    assert len(rebuilt["automated_candidates"]) == cand_count_before  # preserved untouched
    assert rebuilt["revision_id"] == 2
    assert rebuilt["summaries"]["transitions"]["unintended_reversals"] == 0
    assert rebuilt["summaries"]["transitions"]["planned_reversals"] == 1
    assert any(e.get("reason") == "planned_reversal" for e in rebuilt["events"])

    # Companion round-trip
    companion_base = tmp_path / "demo"
    cpath = save_couple_companion(companion_base, rebuilt)
    loaded = load_couple_companion(cpath)
    assert loaded["run_id"] == "run-rev-1"
    assert loaded["revision_id"] == 2

    # Archive persistence: store single first, then couple; verify dual-run retention + run switch + sha256/run_id checks
    arch = tmp_path / "session.tp"
    video = tmp_path / "vid.mp4"
    report = tmp_path / "rep.md"
    video.write_bytes(b"video-bytes-couple")
    report.write_text("# Session")
    vsha = file_sha256(video)
    append_segment(arch, report, video, None, [0.0])

    store_analytics(
        arch,
        0,
        {"frames": []},
        expected_video_sha256=vsha,
        dancer_analysis_run={
            "run_id": "single-0",
            "requested_mode": "single",
            "resolved_mode": "single",
            "frames": [{"t": 0.0}],
        },
    )
    rebuilt_video_sha = {**base, "video_sha256": vsha}
    store_analytics(
        arch,
        0,
        {"frames": []},
        expected_video_sha256=vsha,
        dancer_analysis_run={
            "run_id": "run-rev-1",
            "requested_mode": "couple",
            "resolved_mode": "couple",
            "couple_result": rebuilt_video_sha,
        },
    )
    stored = read_analytics(arch, 0)
    assert stored is not None
    da = stored["dancer_analysis"]
    assert da["active_mode"] == "couple"
    assert "single" in da["runs"] and "couple" in da["runs"]

    # Switching active mode back and forth retains both runs
    set_active_dancer_run(arch, 0, "single")
    assert read_analytics(arch, 0)["dancer_analysis"]["active_mode"] == "single"
    set_active_dancer_run(arch, 0, "couple")
    assert read_analytics(arch, 0)["dancer_analysis"]["active_mode"] == "couple"

    # Mismatched run_id or video_sha256 must fail atomically without mutating archive
    before_bytes = arch.read_bytes()
    with pytest.raises(ValueError, match="run_id mismatch"):
        update_couple_annotations(
            arch, 0, run_id="wrong-run", annotations=anns_planned.to_dict(), cfg=cfg, expected_video_sha256=vsha
        )
    with pytest.raises(ValueError, match="changed"):
        update_couple_annotations(
            arch, 0, run_id="run-rev-1", annotations=anns_planned.to_dict(), cfg=cfg, expected_video_sha256="bad-sha"
        )
    assert arch.read_bytes() == before_bytes

    # Matching update persists revision_id == 2 atomically
    saved_rebuilt = update_couple_annotations(
        arch, 0, run_id="run-rev-1", annotations=anns_planned.to_dict(), cfg=cfg, expected_video_sha256=vsha
    )
    assert saved_rebuilt["revision_id"] == 2
    assert read_analytics(arch, 0)["dancer_analysis"]["runs"]["couple"]["couple_result"]["revision_id"] == 2

    # Export RFC-4180 CSV and parse back
    csv_text = export_couple_csv(saved_rebuilt, "MilongaTest", 0)
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert any(r["row_type"] == "summary" and r["measure"] == "start_lag_ms" and r["run_id"] == "run-rev-1" for r in rows)
    assert any(r["row_type"] == "summary" and r["measure"] == "planned_reversals" and r["value"] == "1" for r in rows)
    assert any(r["row_type"] == "summary" and r["measure"] == "torso_tilt_pearson_r" for r in rows)
    assert any(r["row_type"] == "event" for r in rows)


def test_torso_tilt_pearson_correlation_and_couple_threshold_params():
    from tango_posture.couple.measurements import compute_torso_tilt_pearson_correlation

    # 1. Exact positive, negative, zero-variance, and insufficient-sample cases
    pos_a = [{"timestamp": i * 0.1, "value": float(i - 5), "status": "measured"} for i in range(10)]
    pos_b = [{"timestamp": i * 0.1, "value": float(2 * i - 3), "status": "measured"} for i in range(10)]
    neg_b = [{"timestamp": i * 0.1, "value": float(-2 * i + 4), "status": "measured"} for i in range(10)]
    flat_b = [{"timestamp": i * 0.1, "value": 5.0, "status": "measured"} for i in range(10)]

    r_pos, n_pos, reason_pos = compute_torso_tilt_pearson_correlation(pos_a, pos_b)
    assert r_pos == pytest.approx(1.0, abs=1e-4)
    assert n_pos == 10
    assert reason_pos == ""

    r_neg, n_neg, _ = compute_torso_tilt_pearson_correlation(pos_a, neg_b)
    assert r_neg == pytest.approx(-1.0, abs=1e-4)
    assert n_neg == 10

    r_flat, _, reason_flat = compute_torso_tilt_pearson_correlation(pos_a, flat_b)
    assert r_flat is None
    assert reason_flat == "zero_variance"

    r_few, n_few, reason_few = compute_torso_tilt_pearson_correlation(pos_a[:2], pos_b[:2])
    assert r_few is None
    assert n_few == 2
    assert reason_few == "insufficient_paired_samples"

    # 2. Full couple run populates torso_tilt_pearson_r & stores overlay visibility parameter
    cfg = Config()
    cfg.dancer_analysis.overlay_min_landmark_visibility = 0.65
    cfg.dancer_analysis.overlay_smoothing_s = 0.30
    cfg.dancer_analysis.axis_bridge_max_gap_s = 0.30
    cfg.dancer_analysis.turn_exit_match_window_s = 5.0
    cfg.dancer_analysis.shot_cut_max_frame_gap_s = 0.50

    obs = []
    for i in range(12):
        t = i * 0.1
        tilt_step = 0.01 * math.sin(i * 0.7)
        pa = _make_person_pose(hip_x=0.35, shoulder_dx=tilt_step, torso_dy=0.25)
        pb = _make_person_pose(hip_x=0.55, shoulder_dx=-tilt_step, torso_dy=0.25)
        obs.append((i, _obs(t, [pa, pb])))

    anns = CoupleAnnotations(
        items=[
            CoupleAnnotationItem(annotation_id="ts", event_type="turn_start", subject="pair", timestamp=0.1, status="user_confirmed"),
            CoupleAnnotationItem(annotation_id="te", event_type="turn_end", subject="pair", timestamp=0.4, status="user_confirmed"),
            CoupleAnnotationItem(annotation_id="ex", event_type="exit", subject="pair", timestamp=4.6, status="user_confirmed"),
        ]
    )
    res = run_couple_analysis(
        obs,
        cfg,
        clip_in=0.0,
        clip_out=1.2,
        request=CoupleAnalysisRequest(requested_mode="couple", forced_override=True, annotations=anns),
    ).to_dict()

    pearson_summary = res["summaries"]["posture"]["torso_tilt_pearson_r"]
    assert pearson_summary["status"] == "measured"
    assert pearson_summary["value"] == pytest.approx(-1.0, abs=0.02)
    assert res["frame_overlays"][0]["min_landmark_visibility"] == pytest.approx(0.65)
    # Turn exit matched 4.2 s after turn_end because turn_exit_match_window_s = 5.0 s (> default 3.0 s)
    assert res["summaries"]["turns"]["median_exit_minus_turn_end_ms"] == pytest.approx(4200.0, abs=1.0)

    # Custom shot_cut_max_frame_gap_s triggers break cleanly
    gap_obs = [
        (0, _obs(0.0, [_make_person_pose(hip_x=0.35), _make_person_pose(hip_x=0.55)])),
        (1, _obs(0.8, [_make_person_pose(hip_x=0.36), _make_person_pose(hip_x=0.56)])),
    ]
    gap_tracked = track_couple_frames(gap_obs, cfg)
    assert gap_tracked.frames[1].shot_cut is True


def test_p1_pose_origins_archive_round_trip_and_propagated_policy(tmp_path):
    from tango_posture.archive import append_segment, file_sha256, read_analytics, store_analytics
    from tango_posture.calibration.evidence import cached_evidence
    from tango_posture.calibration.normalization import extract_coach_observations
    from tango_posture.couple.tracking import _translate_pose
    from tango_posture.domain import tag_pose_origin
    from tango_posture.pose.interpolation import interpolate_tracked_dancer_poses
    from tango_posture.references.coach import _coach_observation, _serialise_pose

    cfg = Config(data_dir=str(tmp_path / "data"))
    pa0 = tag_pose_origin(_make_person_pose(hip_x=0.32, shoulder_dx=0.02, torso_dy=0.24), "crop_detected")
    pb0 = tag_pose_origin(_make_person_pose(hip_x=0.52, shoulder_dx=-0.02, torso_dy=0.24), "crop_detected")
    pa1 = _translate_pose(pa0, 0.01, 0.0, origin="propagated")
    pb1 = _translate_pose(pb0, 0.01, 0.0, origin="propagated")
    pa2 = _translate_pose(pa0, 0.02, 0.0, origin="propagated")
    pb2 = _translate_pose(pb0, 0.02, 0.0, origin="propagated")

    obs_list = [
        (0, FrameObservation(0.0, 640, 480, [pa0, pb0], pose_origins=["crop_detected", "crop_detected"])),
        (1, FrameObservation(0.05, 640, 480, [pa1, pb1], pose_origins=["propagated", "propagated"])),
        (2, FrameObservation(0.10, 640, 480, [pa2, pb2], pose_origins=["propagated", "propagated"])),
    ]

    # 1. Default measured_only excludes propagated samples from coverage & measurement series while keeping display overlays
    res_measured = run_couple_analysis(
        obs_list,
        cfg,
        clip_in=0.0,
        clip_out=0.15,
        request=CoupleAnalysisRequest(requested_mode="couple", forced_override=True),
        video_sha256="sha-p1-origin",
        run_id="run-origin-1",
    ).to_dict()

    assert res_measured["coverage"]["pair_hip_coverage"] == pytest.approx(1.0 / 3.0, abs=1e-3)
    assert res_measured["coverage"]["pair_torso_coverage"] == pytest.approx(1.0 / 3.0, abs=1e-3)
    assert res_measured["coverage"]["measurement_origin_policy"] == "measured_only"
    assert res_measured["coverage"]["origin_counts"]["pair"]["crop_detected"] == 1
    assert res_measured["coverage"]["origin_counts"]["pair"]["propagated"] == 2
    assert res_measured["series"]["torso_tilt_a"][0]["status"] == "measured"
    assert res_measured["series"]["torso_tilt_a"][0]["pose_origin"] == "crop_detected"
    assert res_measured["series"]["torso_tilt_a"][1]["value"] is None
    assert res_measured["series"]["torso_tilt_a"][1]["status"] == "unavailable"
    assert res_measured["series"]["torso_tilt_a"][1]["reason"] == "propagated_pose_excluded"
    assert res_measured["series"]["torso_tilt_a"][1]["pose_origin"] == "propagated"
    assert res_measured["frame_overlays"][1]["dancer_a"] is not None
    assert res_measured["frame_overlays"][1]["dancer_a"]["pose_origin"] == "propagated"

    # 2. Explicit allow_propagated_estimates includes propagated samples cleanly as estimated -> mixed summary
    res_est = run_couple_analysis(
        obs_list,
        cfg,
        clip_in=0.0,
        clip_out=0.15,
        request=CoupleAnalysisRequest(
            requested_mode="couple",
            forced_override=True,
            pose_origin_policy="allow_propagated_estimates",
        ),
        video_sha256="sha-p1-origin",
        run_id="run-origin-est",
    ).to_dict()
    assert res_est["coverage"]["pair_torso_coverage"] == pytest.approx(1.0, abs=1e-3)
    assert res_est["coverage"]["measured_only_coverage"]["pair_torso_coverage"] == pytest.approx(1.0 / 3.0, abs=1e-3)
    assert res_est["series"]["torso_tilt_a"][1]["status"] == "estimated"
    assert res_est["series"]["torso_tilt_a"][1]["pose_origin"] == "propagated"
    assert res_est["summaries"]["posture"]["torso_tilt_a"]["status"] == "mixed"
    assert res_est["summaries"]["posture"]["torso_tilt_a"]["status_counts"]["measured"] == 1
    assert res_est["summaries"]["posture"]["torso_tilt_a"]["status_counts"]["estimated"] == 2

    # 3. Archive round-trip preserves both couple_result origins and coach measurement_pose_origins
    arch = Path(cfg.data_dir) / "practice" / "OriginRoundTrip.tp"
    vid = tmp_path / "dummy.mp4"
    vid.write_bytes(b"synthetic-video-bytes")
    rep = tmp_path / "rep.md"
    rep.write_text("# Origin test\n", encoding="utf-8")
    timestamps = [0.0, 0.05, 0.10]
    append_segment(arch, rep, vid, None, timestamps)
    vsha = file_sha256(vid)

    coach_frames = {
        str(idx): {
            "timestamp": ts,
            "poses": [_serialise_pose(obs.poses[0])],
            "measurement_poses": [_serialise_pose(p) for p in obs.poses],
            "measurement_pose_origins": [obs.pose_origin_at(i) for i in range(len(obs.poses))],
            "measurement_landmark_origins": [
                {k: lm.origin for k, lm in p.items()} for p in obs.poses
            ],
        }
        for idx, (ts, obs) in enumerate([(t, o) for t, (_, o) in zip(timestamps, obs_list)])
    }
    coach_payload = {
        "version": 1,
        "model_hash": "hash-origin",
        "capture_profile": {"side": "left", "facing": "right", "width": 640, "height": 480},
        "width": 640,
        "height": 480,
        "frames": coach_frames,
    }
    store_analytics(
        arch,
        0,
        {
            "model_hash": "hash-origin",
            "capture_profile": {"side": "left", "facing": "right", "width": 640, "height": 480},
            "max_gap": 0.25,
            "frame_timestamps": timestamps,
            "frames": [],
            "coach_data": coach_payload,
            "movement_data": {"steps": [], "pivots": []},
        },
        expected_video_sha256=vsha,
        dancer_analysis_run={
            "run_id": "run-origin-1",
            "requested_mode": "couple",
            "resolved_mode": "couple",
            "couple_result": res_measured,
        },
    )
    loaded_analytics = read_analytics(arch, 0)
    assert loaded_analytics is not None
    loaded_couple = loaded_analytics["dancer_analysis"]["runs"]["couple"]["couple_result"]
    assert [f["dancer_a"] for f in loaded_couple["series"]["frame_pose_origins"]] == [
        "crop_detected",
        "propagated",
        "propagated",
    ]
    assert loaded_couple["frame_overlays"][0]["dancer_a"]["pose_origin"] == "crop_detected"
    assert loaded_couple["frame_overlays"][1]["dancer_a"]["pose_origin"] == "propagated"

    ev = cached_evidence(arch, 0, couple=False)
    assert [f.pose_origin_at(0) for f in ev.frames] == ["crop_detected", "propagated", "propagated"]
    assert ev.frames[0].poses[0]["left_hip"].origin == "crop_detected"
    assert ev.frames[1].poses[0]["left_hip"].origin == "propagated"
    parsed_obs = extract_coach_observations(loaded_analytics["coach_data"])
    assert [o.pose_origin_at(0) for o in parsed_obs] == ["crop_detected", "propagated", "propagated"]
    single_obs = _coach_observation(loaded_analytics["coach_data"]["frames"]["1"], 640, 480)
    assert single_obs.pose_origin_at(0) == "propagated"

    # 4. Display interpolation stamps display_interpolated without mutating raw_pose
    tracked = track_couple_frames(obs_list[:1] + [(2, FrameObservation(0.10, 640, 480, [pa2, pb2], pose_origins=["crop_detected", "crop_detected"]))], cfg)
    sa0, sa2 = tracked.frames[0].dancer_a, tracked.frames[1].dancer_a
    interp_samples = interpolate_tracked_dancer_poses(
        [sa0, None, sa2],
        max_gap_s=0.20,
        confidence=cfg.confidence,
        timestamps=[0.0, 0.05, 0.10],
        frame_indices=[0, 1, 2],
        sample_factory=lambda t, fidx: track_couple_frames([(fidx, FrameObservation(t, 640, 480, []))], cfg).frames[0].dancer_a
        or tracked.frames[0].dancer_a.__class__(
            timestamp=t,
            frame_index=fidx,
            pose_index=None,
            raw_pose=None,
            display_pose={},
            world_pose=None,
            hip_px=None,
            shoulder_px=None,
            torso_length_px=None,
            torso_eligible=False,
            hip_eligible=False,
            head_eligible=False,
            view_usable=False,
            interpolated_display=True,
            quality_status="unavailable",
            pose_origin="display_interpolated",
        ),
    )
    assert interp_samples[1].pose_origin == "display_interpolated"
    assert interp_samples[1].raw_pose is None
    assert interp_samples[1].display_pose["left_hip"].origin == "display_interpolated"
    assert interp_samples[0].raw_pose["left_hip"].origin == "crop_detected"


def test_p1_transition_elapsed_support_breaks_gaps_dedup_and_per_phase_reference():
    from tango_posture.couple.measurements import _evaluate_transitions

    cfg = Config()
    cfg.dancer_analysis.persistence_s = 0.20
    cfg.dancer_analysis.secondary_speed_rise = 0.15
    cfg.dancer_analysis.stop_speed = 0.10
    cfg.dancer_analysis.max_observation_gap_s = 0.10
    cfg.dancer_analysis.transition_min_coverage = 0.50

    # 1. Final-frame spike produces 0 cues (never reaches tk + persistence_s)
    ts_final = [round(i * 0.05, 4) for i in range(12)]  # 0.00..0.55
    vp_final = [0.60, 0.60, 0.60, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.90]
    cues_final, sum_final = _evaluate_transitions(
        ts_final,
        vp_final,
        [True] * len(ts_final),
        [False] * len(ts_final),
        [{"timestamp": 0.15}],
        [],
        CoupleAnnotations(),
        cfg,
    )
    assert cues_final == []
    assert sum_final["evaluated_window_count"] >= 1
    assert sum_final["secondary_speed_peaks"] == 0

    # 2. Spike cut short by a shot cut OR excessive gap before reaching persistence_s produces 0 cues
    ts_cut = [round(i * 0.05, 4) for i in range(14)]
    vp_cut = [0.60, 0.60, 0.60, 0.20, 0.20, 0.55, 0.55, 0.55, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]
    brk_cut = [False] * len(ts_cut)
    brk_cut[7] = True  # cuts off after only 0.05s from 0.25..0.30 (< 0.20s persistence)
    cues_cut, sum_cut = _evaluate_transitions(
        ts_cut,
        vp_cut,
        [True] * len(ts_cut),
        brk_cut,
        [{"timestamp": 0.15}],
        [],
        CoupleAnnotations(),
        cfg,
    )
    assert cues_cut == []
    assert sum_cut["secondary_speed_peaks"] == 0

    ts_gap = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.55, 0.60]  # 0.20s gap between 0.30 and 0.50 (> 0.10s)
    vp_gap = [0.60, 0.60, 0.60, 0.20, 0.20, 0.65, 0.65, 0.65, 0.65, 0.20]
    cues_gap, _ = _evaluate_transitions(
        ts_gap,
        vp_gap,
        [True] * len(ts_gap),
        [False] * len(ts_gap),
        [{"timestamp": 0.15}],
        [],
        CoupleAnnotations(),
        cfg,
    )
    assert cues_gap == []

    # 3. Genuine sustained rise + overlapping windows deduplicates into 1 canonical episode
    ts_ok = [round(i * 0.05, 4) for i in range(20)]  # 0.00..0.95
    vp_ok = [
        0.60, 0.60, 0.60,  # steady pre-slow
        0.20, 0.20, 0.20,  # slowing
        0.55, 0.55, 0.55, 0.55, 0.55, 0.55,  # sustained secondary speed peak >= 0.25s
        0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05,
    ]
    cues_ok, sum_ok = _evaluate_transitions(
        ts_ok,
        vp_ok,
        [True] * len(ts_ok),
        [False] * len(ts_ok),
        [{"timestamp": 0.15}, {"timestamp": 0.20}],  # overlapping slow windows
        [{"timestamp": 0.80}],
        CoupleAnnotations(),
        cfg,
    )
    speed_peaks = [c for c in cues_ok if c["metadata"]["cue_kind"] == "secondary_speed_peak"]
    assert len(speed_peaks) == 1
    assert sum_ok["secondary_speed_peaks"] == 1
    assert speed_peaks[0]["metadata"]["episode_id"].startswith("ep_speed_peak_")
    assert speed_peaks[0]["metadata"]["support_duration_s"] >= 0.20

    # 4. Per-phase reversal reference: Phase 1 (+0.7 -> +0.2) is shorter than Phase 2 (-0.8 -> -0.2 -> +0.5 reversal)
    ts_phase = [round(i * 0.05, 4) for i in range(44)]  # 0.00..2.15
    vp_phase = (
        [0.70] * 6       # 0.00..0.25 Phase 1 +
        + [0.20] * 6     # 0.30..0.55 Phase 1 slow (+ direction, NOT a reversal!)
        + [0.00] * 4     # 0.60..0.75 settled pause
        + [-0.80] * 14   # 0.80..1.45 longer Phase 2 - (drives old clip-wide sign to -1!)
        + [-0.20] * 5    # 1.50..1.70 Phase 2 slow
        + [0.50] * 7     # 1.75..2.05 genuine reversal in Phase 2
        + [0.00] * 2
    )
    cues_phase, sum_phase = _evaluate_transitions(
        ts_phase,
        vp_phase,
        [True] * len(ts_phase),
        [False] * len(ts_phase),
        [{"timestamp": 0.30}, {"timestamp": 1.50}],
        [{"timestamp": 0.65}, {"timestamp": 2.10}],
        CoupleAnnotations(),
        cfg,
    )
    reversals = [c for c in cues_phase if c["metadata"]["cue_kind"] == "unintended_reversal"]
    assert len(reversals) == 1
    assert reversals[0]["timestamp"] == pytest.approx(1.75, abs=0.06)
    assert reversals[0]["metadata"]["phase_reference_sign"] == -1.0
    assert sum_phase["unintended_reversals"] == 1


def test_p1_turn_matching_by_subject_turn_id_and_unmatched_preservation():
    from tango_posture.couple.measurements import _turn_events_from_annotations_and_candidates

    anns = CoupleAnnotations(
        items=[
            # Overlapping A and B turns with explicit turn_id
            CoupleAnnotationItem("a_s", "turn_start", 1.0, subject="A", turn_id="turn_A_1"),
            CoupleAnnotationItem("b_s", "turn_start", 1.2, subject="B", turn_id="turn_B_1"),
            CoupleAnnotationItem("b_e", "turn_end", 1.8, subject="B", turn_id="turn_B_1"),
            CoupleAnnotationItem("a_e", "turn_end", 2.0, subject="A", turn_id="turn_A_1"),
            CoupleAnnotationItem("b_x", "exit", 1.90, subject="B", turn_id="turn_B_1"),
            CoupleAnnotationItem("a_x", "exit", 2.15, subject="A", turn_id="turn_A_1"),
            # Legacy overlapping A/B turns without turn_id -> still matched strictly per subject
            CoupleAnnotationItem("la_s", "turn_start", 4.0, subject="A"),
            CoupleAnnotationItem("lb_s", "turn_start", 4.1, subject="B"),
            CoupleAnnotationItem("lb_e", "turn_end", 4.5, subject="B"),
            CoupleAnnotationItem("la_e", "turn_end", 4.9, subject="A"),
            CoupleAnnotationItem("la_x", "exit", 5.05, subject="A"),
            CoupleAnnotationItem("lb_x", "exit", 4.60, subject="B"),
            # Explicit shared pair turn
            CoupleAnnotationItem("p_s", "turn_start", 7.0, subject="pair", turn_id="turn_pair_1"),
            CoupleAnnotationItem("p_e", "turn_end", 7.6, subject="pair", turn_id="turn_pair_1"),
            # Unmatched turn_start (missing end)
            CoupleAnnotationItem("orphan_s", "turn_start", 9.0, subject="A", turn_id="turn_open"),
            # Reversed order with turn_id -> rejected & preserved unmatched
            CoupleAnnotationItem("bad_s", "turn_start", 11.0, subject="B", turn_id="turn_bad"),
            CoupleAnnotationItem("bad_e", "turn_end", 10.5, subject="B", turn_id="turn_bad"),
            # Legacy orphan exit for B -> preserved unmatched
            CoupleAnnotationItem("orphan_x", "exit", 15.0, subject="B"),
        ]
    )

    events, turn_sum = _turn_events_from_annotations_and_candidates(
        anns, auto_turn_suggestions=False, pivot_candidates=[], turn_exit_match_window_s=3.0
    )
    matched = [e for e in events if e["event_type"] == "turn"]
    unmatched = [e for e in events if e["event_type"] != "turn"]

    assert turn_sum["turn_count"] == 5
    by_tid = {e["turn_id"]: e for e in matched}
    assert by_tid["turn_A_1"]["subject"] == "A"
    assert by_tid["turn_A_1"]["value"] == pytest.approx(1.0, abs=1e-4)
    assert by_tid["turn_A_1"]["metadata"]["exit_minus_turn_end_ms"] == pytest.approx(150.0, abs=0.1)

    assert by_tid["turn_B_1"]["subject"] == "B"
    assert by_tid["turn_B_1"]["value"] == pytest.approx(0.6, abs=1e-4)
    assert by_tid["turn_B_1"]["metadata"]["exit_minus_turn_end_ms"] == pytest.approx(100.0, abs=0.1)

    assert by_tid["turn_pair_1"]["metadata"]["shared_pair_turn"] is True
    assert any(e["subject"] == "A" and e["value"] == pytest.approx(0.9, abs=1e-4) for e in matched)
    assert any(e["subject"] == "B" and e["value"] == pytest.approx(0.4, abs=1e-4) for e in matched)

    # Unmatched markers are preserved for correction (missing end, invalid order, orphan exit)
    unmatched_ids = {m["annotation_id"]: m for m in turn_sum["unmatched_markers"]}
    assert turn_sum["unmatched_count"] == 4
    assert unmatched_ids["orphan_s"]["reason"] == "missing_turn_end"
    assert unmatched_ids["bad_s"]["reason"] == "invalid_turn_marker_order"
    assert unmatched_ids["bad_e"]["reason"] == "invalid_turn_marker_order"
    assert unmatched_ids["orphan_x"]["reason"] == "unmatched_exit_no_matching_turn"
    assert len(unmatched) == 4

    # Mismatched subject sharing a turn_id is rejected and preserved unmatched
    mismatch_anns = CoupleAnnotations(
        items=[
            CoupleAnnotationItem("ms", "turn_start", 1.0, subject="A", turn_id="tid_mix"),
            CoupleAnnotationItem("me", "turn_end", 1.5, subject="B", turn_id="tid_mix"),
        ]
    )
    _, mismatch_sum = _turn_events_from_annotations_and_candidates(
        mismatch_anns, auto_turn_suggestions=False, pivot_candidates=[]
    )
    assert mismatch_sum["turn_count"] == 0
    assert mismatch_sum["unmatched_count"] == 2
    assert all(m["reason"] == "turn_id_subject_mismatch" for m in mismatch_sum["unmatched_markers"])
    assert any("mismatched marker subjects" in w for w in mismatch_sum["warnings"])

    # Nested same-subject legacy turns without turn_id emit an ambiguity warning
    nested_anns = CoupleAnnotations(
        items=[
            CoupleAnnotationItem("ns1", "turn_start", 1.0, subject="A"),
            CoupleAnnotationItem("ns2", "turn_start", 1.2, subject="A"),
            CoupleAnnotationItem("ne1", "turn_end", 1.5, subject="A"),
            CoupleAnnotationItem("ne2", "turn_end", 1.9, subject="A"),
        ]
    )
    _, nested_sum = _turn_events_from_annotations_and_candidates(
        nested_anns, auto_turn_suggestions=False, pivot_candidates=[]
    )
    assert any("Overlapping legacy turn_start" in w for w in nested_sum["warnings"])


def test_p1_summary_evidence_status_mixed_unavailable_and_coordination():
    from tango_posture.couple.measurements import (
        _apply_annotations_to_candidates,
        _turn_events_from_annotations_and_candidates,
    )

    cfg = Config()

    # 1. One confirmed turn + four estimated auto turns -> "mixed" (never "user_confirmed")
    anns_one_turn = CoupleAnnotations(
        items=[
            CoupleAnnotationItem("ts", "turn_start", 1.0, subject="pair", turn_id="t_conf", status="user_confirmed"),
            CoupleAnnotationItem("te", "turn_end", 1.5, subject="pair", turn_id="t_conf", status="user_confirmed"),
        ]
    )
    pivots = [
        {"start_time": 2.0 + i * 1.0, "end_time": 2.4 + i * 1.0, "subject": "pair"}
        for i in range(4)
    ]
    _, mixed_turn_sum = _turn_events_from_annotations_and_candidates(
        anns_one_turn, auto_turn_suggestions=True, pivot_candidates=pivots
    )
    assert mixed_turn_sum["turn_count"] == 5
    assert mixed_turn_sum["status"] == "mixed"
    assert mixed_turn_sum["status_counts"]["user_confirmed"] == 1
    assert mixed_turn_sum["status_counts"]["estimated"] == 4
    assert mixed_turn_sum["eligible_count"] == 5

    # 2. No evaluated transition window -> "unavailable" with None counts (never measured 0)
    series_no_slow = {
        "velocity_pair": [
            {"timestamp": i * 0.05, "value": 0.5, "pair_hip_eligible": True, "break_flag": False}
            for i in range(10)
        ],
        "torso_tilt_a": [{"timestamp": i * 0.05, "value": 1.0, "status": "measured"} for i in range(10)],
        "torso_tilt_b": [{"timestamp": i * 0.05, "value": 1.0, "status": "measured"} for i in range(10)],
        "couple_midpoint_tilt": [{"timestamp": i * 0.05, "value": 1.0, "status": "measured"} for i in range(10)],
        "torso_separation": [{"timestamp": i * 0.05, "value": 0.0, "status": "measured"} for i in range(10)],
        "head_pitch_a": [],
        "head_pitch_b": [],
        "pair_breaks": [],
    }
    coverage_full = {
        "frame_count": 10,
        "a_torso_coverage": 1.0,
        "b_torso_coverage": 1.0,
        "pair_torso_coverage": 1.0,
        "pair_hip_coverage": 1.0,
    }
    _, sum_no_win = _apply_annotations_to_candidates(
        automated_candidates=[],
        annotations=CoupleAnnotations(),
        pivot_candidates=[],
        series=series_no_slow,
        coverage=coverage_full,
        cfg=cfg,
    )
    tr_sum = sum_no_win["transitions"]
    assert tr_sum["status"] == "unavailable"
    assert tr_sum["evaluated_window_count"] == 0
    assert tr_sum["secondary_speed_peaks"] is None
    assert tr_sum["unintended_reversals"] is None

    # 3. Coordination & step_timing reflect edited/mixed status rather than hardcoding "measured"
    candidates = [
        {
            "event_id": "cand_A_start_0",
            "event_type": "start",
            "subject": "A",
            "timestamp": 1.00,
            "status": "measured",
            "metadata": {},
        },
        {
            "event_id": "cand_B_start_0",
            "event_type": "start",
            "subject": "B",
            "timestamp": 1.12,
            "status": "measured",
            "metadata": {},
        },
        {
            "event_id": "cand_A_contact_0",
            "event_type": "contact",
            "subject": "A",
            "timestamp": 1.05,
            "status": "estimated",
            "metadata": {},
        },
    ]
    anns_edited_a = CoupleAnnotations(
        items=[
            CoupleAnnotationItem(
                annotation_id="edit_a",
                event_type="start",
                subject="A",
                timestamp=1.02,
                status="user_confirmed",
                source_candidate_id="cand_A_start_0",
            ),
            CoupleAnnotationItem(
                annotation_id="conf_contact_a",
                event_type="contact",
                subject="A",
                timestamp=1.25,
                status="user_confirmed",
            ),
        ]
    )
    _, sum_mixed_coord = _apply_annotations_to_candidates(
        automated_candidates=candidates,
        annotations=anns_edited_a,
        pivot_candidates=[],
        series=series_no_slow,
        coverage=coverage_full,
        cfg=cfg,
    )
    coord_start = sum_mixed_coord["coordination"]["start_lag_ms"]
    assert coord_start["status"] == "mixed"
    assert coord_start["status_counts"]["user_confirmed"] == 1
    assert coord_start["status_counts"]["measured"] == 1
    assert sum_mixed_coord["step_timing"]["A"]["status"] == "mixed"

    # CSV export leaves unavailable transition counts blank (not "0") and carries status_counts & evaluated_window_count
    mock_res = {
        "run_id": "r-p1-4",
        "coverage": coverage_full,
        "summaries": sum_no_win,
        "events": [],
    }
    csv_rows = list(csv.DictReader(io.StringIO(export_couple_csv(mock_res, "Clip", 0))))
    tr_row = next(r for r in csv_rows if r["measure"] == "secondary_speed_peaks")
    assert tr_row["status"] == "unavailable"
    assert tr_row["value"] == ""
    assert tr_row["evaluated_window_count"] == "0"
