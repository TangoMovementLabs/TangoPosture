"""Unit tests for the Qt-independent solo camera calibration state machine and gates."""

from __future__ import annotations

from tango_posture.calibration.camera_session import (
    SoloCameraSession,
    accumulate_usable_duration,
    build_check_phases,
    build_quick_setup_phases,
    build_walking_phases,
    check_full_body_framing,
    check_neutral_pair_comparable,
    classify_guide_view,
    evaluate_metric_repeat_check,
)
from tango_posture.calibration.models import Passage
from tango_posture.config import Config
from tango_posture.domain import FrameObservation, Landmark


def _lm(x: float, y: float, vis: float = 0.95) -> Landmark:
    return Landmark(x=x, y=y, visibility=vis, presence=vis, z=0.0)


def _front_pose(dx: float = 0.0) -> dict[str, Landmark]:
    return {
        "nose": _lm(0.50 + dx, 0.20),
        "left_ear": _lm(0.54 + dx, 0.20),
        "right_ear": _lm(0.46 + dx, 0.20),
        "left_shoulder": _lm(0.62 + dx, 0.32),
        "right_shoulder": _lm(0.38 + dx, 0.32),
        "left_hip": _lm(0.58 + dx, 0.58),
        "right_hip": _lm(0.42 + dx, 0.58),
        "left_knee": _lm(0.58 + dx, 0.74),
        "right_knee": _lm(0.42 + dx, 0.74),
        "left_ankle": _lm(0.58 + dx, 0.90),
        "right_ankle": _lm(0.42 + dx, 0.90),
    }


def _side_pose(facing: str = "right", *, dx: float = 0.0, ankle_vis: float = 0.95) -> dict[str, Landmark]:
    nose_x = 0.53 + dx if facing == "right" else 0.47 + dx
    return {
        "nose": _lm(nose_x, 0.20),
        "left_ear": _lm(0.50 + dx, 0.20),
        "right_ear": _lm(0.495 + dx, 0.20, 0.5),
        "left_shoulder": _lm(0.51 + dx, 0.32),
        "right_shoulder": _lm(0.49 + dx, 0.32),
        "left_hip": _lm(0.51 + dx, 0.58),
        "right_hip": _lm(0.49 + dx, 0.58),
        "left_knee": _lm(0.51 + dx, 0.74, ankle_vis),
        "right_knee": _lm(0.49 + dx, 0.74, ankle_vis),
        "left_ankle": _lm(0.51 + dx, 0.90, ankle_vis),
        "right_ankle": _lm(0.49 + dx, 0.90, ankle_vis),
    }


def test_classify_guide_view_front_side_and_uncertain():
    cfg = Config(side="left", facing="right", max_view_width=0.45)
    front_frame = FrameObservation(0.0, 640, 480, [_front_pose()])
    assert classify_guide_view(front_frame, cfg, requested_view="front")[0] == "front"

    side_frame = FrameObservation(0.0, 640, 480, [_side_pose("right")])
    assert classify_guide_view(side_frame, cfg, requested_view="side")[0] == "side"

    wrong_facing = FrameObservation(0.0, 640, 480, [_side_pose("left")])
    view, hint = classify_guide_view(wrong_facing, cfg, requested_view="side")
    assert view == "uncertain"
    assert "Face right" in hint

    two_people = FrameObservation(0.0, 640, 480, [_front_pose(), _front_pose(0.2)])
    assert classify_guide_view(two_people, cfg)[0] == "uncertain"


def test_full_body_framing_check_requires_ankles_and_single_dancer():
    cfg = Config()
    ok, _ = check_full_body_framing(FrameObservation(0.0, 640, 480, [_front_pose()]), cfg)
    assert ok

    no_ankle = _front_pose()
    del no_ankle["left_ankle"]
    del no_ankle["right_ankle"]
    ok_missing, msg = check_full_body_framing(FrameObservation(0.0, 640, 480, [no_ankle]), cfg)
    assert not ok_missing
    assert "shoes/ankles" in msg


def test_prompt_speech_scoping_and_settle_stillness_gate():
    cfg = Config()
    session = SoloCameraSession(cfg, now_monotonic=0.0)
    phases = build_quick_setup_phases()[:1]  # front_a only
    pid = session.start_plan(phases, now_monotonic=0.0, speak_enabled=True)
    assert session.state == "Prompt"

    # Stale prompt ID is ignored
    session.on_speech_finished(pid - 1, now_monotonic=0.2)
    assert session.state == "Prompt"

    # Matching prompt ID transitions to Wait for view and pose
    session.on_speech_finished(pid, now_monotonic=0.5, failed=True)
    assert session.state == "Wait for view and pose"
    assert "text" in session.hint_message.lower()

    # Wrong view keeps waiting
    session.process_frame(FrameObservation(1.0, 640, 480, [_side_pose("right")]), now_monotonic=1.0)
    assert session.state == "Wait for view and pose"

    # Matching front view enters Settle
    session.process_frame(FrameObservation(1.1, 640, 480, [_front_pose(0.0)]), now_monotonic=1.1)
    assert session.state == "Settle"

    # Large torso drift re-arms settling
    session.process_frame(FrameObservation(1.3, 640, 480, [_front_pose(0.08)]), now_monotonic=1.3)
    assert session.state == "Settle"

    # Holding still for stable_time_s (0.75s) transitions to Capture
    session.process_frame(FrameObservation(2.1, 640, 480, [_front_pose(0.08)]), now_monotonic=2.1)
    assert session.state == "Capture"


def test_accumulate_usable_duration_rejects_gaps_and_duplicates():
    # Duplicate timestamp + 0.35s gap (> 0.20s max_gap) + invalid None break
    samples = [
        (0.0, 1.0),
        (0.0, 1.0),  # duplicate
        (0.1, 1.1),
        (0.2, 1.2),
        (0.55, 1.3),  # 0.35s gap > 0.20s -> 0 duration added
        (0.65, None),  # invalid
        (0.75, 1.4),
    ]
    assert abs(accumulate_usable_duration(samples, 0.20) - 0.2) < 1e-6


def test_bystander_rearms_active_hold_and_preserves_accepted():
    cfg = Config()
    session = SoloCameraSession(cfg, now_monotonic=0.0)
    phases = build_quick_setup_phases()[:2]  # front_a, front_b
    session.start_plan(phases, now_monotonic=0.0, speak_enabled=False)

    # Complete front_a cleanly (2.6s capture at 10Hz)
    t = 0.0
    session.process_frame(FrameObservation(t, 640, 480, [_front_pose()]), now_monotonic=t)
    t += 0.8
    session.process_frame(FrameObservation(t, 640, 480, [_front_pose()]), now_monotonic=t)
    assert session.state == "Capture"
    for i in range(1, 27):
        ts = t + i * 0.1
        session.process_frame(FrameObservation(ts, 640, 480, [_front_pose()]), now_monotonic=ts, speak_enabled=False)
    assert "front_a" in session.accepted_passages
    assert session.current_phase is not None and session.current_phase.phase_id == "front_b"

    # Start front_b settle -> capture, then inject bystander
    t = 4.0
    session.process_frame(FrameObservation(t, 640, 480, [_front_pose()]), now_monotonic=t)
    t += 0.8
    session.process_frame(FrameObservation(t, 640, 480, [_front_pose()]), now_monotonic=t)
    assert session.state == "Capture"
    session.process_frame(
        FrameObservation(t + 0.1, 640, 480, [_front_pose(), _front_pose(0.2)]),
        now_monotonic=t + 0.1,
    )
    assert session.state == "Wait for view and pose"
    assert "front_a" in session.accepted_passages


def test_camera_stall_watchdog_and_skip_view():
    cfg = Config()
    session = SoloCameraSession(cfg, now_monotonic=0.0)
    session.start_plan(build_quick_setup_phases(), now_monotonic=0.0, speak_enabled=False)
    assert session.current_phase is not None and session.current_phase.view == "front"

    session.skip_view(now_monotonic=0.5, speak_enabled=False)
    assert session.current_phase is not None and session.current_phase.view == "side"

    assert not session.check_watchdog(2.0)
    assert session.check_watchdog(4.0)
    assert session.state == "Camera error"
    assert "stopped responding" in session.error_message.lower()


def test_repeat_check_and_neutral_comparability():
    # Neutral A: ~1.0, Changed A: ~0.6, Neutral B: ~1.0, Changed B: ~0.6 (expected_sign=-1)
    samples = []
    for i in range(10):
        samples.append((i * 0.1, 1.0 + (i % 2) * 0.02))
    for i in range(10, 20):
        samples.append((i * 0.1, 0.6 + (i % 2) * 0.02))
    for i in range(20, 30):
        samples.append((i * 0.1, 1.0 + (i % 2) * 0.02))
    for i in range(30, 40):
        samples.append((i * 0.1, 0.6 + (i % 2) * 0.02))

    holds = [Passage(start=0.0, end=0.95), Passage(start=1.0, end=1.95), Passage(start=2.0, end=2.95), Passage(start=3.0, end=3.95)]
    ok_cmp, _ = check_neutral_pair_comparable(samples, holds[0], holds[2], max_gap=0.2)
    assert ok_cmp
    ok_rep, _ = evaluate_metric_repeat_check(samples, holds, max_gap=0.2, expected_sign=-1)
    assert ok_rep
    ok_wrong_sign, _ = evaluate_metric_repeat_check(samples, holds, max_gap=0.2, expected_sign=1)
    assert not ok_wrong_sign

    # Barely distinguishable change fails
    weak_samples = [(i * 0.1, 1.0 + (i % 2) * 0.05) for i in range(40)]
    ok_weak, _ = evaluate_metric_repeat_check(weak_samples, holds, max_gap=0.2, expected_sign=-1)
    assert not ok_weak


def test_walking_plan_uses_take_states_without_stillness():
    cfg = Config()
    session = SoloCameraSession(cfg, now_monotonic=0.0)
    session.start_plan(build_walking_phases(), now_monotonic=0.0, speak_enabled=False)
    assert session.state == "Wait for take"
    session.start_walking_take(now_monotonic=1.0)
    assert session.state == "Capture take"
    for i in range(25):
        ts = 1.0 + i * 0.1
        session.process_frame(FrameObservation(ts, 640, 480, [_side_pose("right", dx=i * 0.005)]), now_monotonic=ts)
    rec = session.end_walking_take()
    assert rec is not None and rec.usable and rec.role == "walking"
    assert len(build_check_phases(["head_pitch", "shoulder_clearance"])) == 8
