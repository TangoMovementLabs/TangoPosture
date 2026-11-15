"""Comprehensive unit tests for Auto/Single/Couple detection, tracking, math, archive, and CSV."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from zipfile import ZipFile

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
    assert any(r["row_type"] == "event" for r in rows)
