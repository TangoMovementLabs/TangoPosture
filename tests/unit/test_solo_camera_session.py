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


def test_quick_setup_phases_explicit_turn_directions():
    from tango_posture.calibration.camera_session import CHECK_SPECS

    phases = {p.phase_id: p for p in build_quick_setup_phases()}
    assert "turn to the camera" in phases["front_a"].instruction.lower()
    assert "turn to the camera" in phases["front_b"].instruction.lower()
    assert "turn left" in phases["side_a"].instruction.lower()
    assert "turn right" in phases["side_b"].instruction.lower()
    assert phases["front_a"].turn_direction == "camera"
    assert phases["side_a"].turn_direction == "left"
    assert phases["side_b"].turn_direction == "right"

    walking = build_walking_phases()
    assert "turn left" in walking[0].instruction.lower()
    assert "turn right" in walking[1].instruction.lower()

    all_instructions = [p.instruction for p in phases.values()] + [p.instruction for p in walking]
    for spec in CHECK_SPECS.values():
        all_instructions.append(spec.neutral_instruction)
    assert all("direction shown" not in txt.lower() and "as shown" not in txt.lower() for txt in all_instructions)


def test_quick_setup_completes_both_side_turns_with_occluded_far_joints():
    def _profile_with_occluded_far_side(
        *, near_side: str, facing: str, ankle_vis: float = 0.95
    ) -> dict[str, Landmark]:
        nose_x = 0.53 if facing == "right" else 0.47
        far_side = "right" if near_side == "left" else "left"
        pose = {
            "nose": _lm(nose_x, 0.20, 0.95),
            f"{near_side}_ear": _lm(0.50, 0.20, 0.95),
            f"{far_side}_ear": _lm(0.50, 0.20, 0.35),
            f"{near_side}_shoulder": _lm(0.50, 0.32, 0.95),
            f"{far_side}_shoulder": _lm(0.50, 0.32, 0.40),
            f"{near_side}_hip": _lm(0.50, 0.58, 0.95),
            f"{far_side}_hip": _lm(0.50, 0.58, 0.40),
            f"{near_side}_knee": _lm(0.51, 0.74, ankle_vis),
            f"{far_side}_knee": _lm(0.49, 0.74, 0.35),
            f"{near_side}_ankle": _lm(0.50, 0.90, ankle_vis),
            f"{far_side}_ankle": _lm(0.50, 0.90, 0.35),
        }
        return pose

    cfg = Config()
    session = SoloCameraSession(cfg, now_monotonic=0.0, auto_detect_side_facing=True)
    session.start_plan(build_quick_setup_phases(), now_monotonic=0.0, speak_enabled=False)

    t = 0.0
    # Complete front_a and front_b
    for phase_id in ("front_a", "front_b"):
        assert session.current_phase is not None and session.current_phase.phase_id == phase_id
        session.process_frame(FrameObservation(t, 640, 480, [_front_pose()]), now_monotonic=t)
        t += 0.8
        session.process_frame(FrameObservation(t, 640, 480, [_front_pose()]), now_monotonic=t)
        assert session.state == "Capture"
        for i in range(1, 27):
            ts = t + i * 0.1
            session.process_frame(
                FrameObservation(ts, 640, 480, [_front_pose()]),
                now_monotonic=ts,
                speak_enabled=False,
            )
        t += 2.7

    # While still facing camera on side_a, hint must explicitly say Turn left
    assert session.current_phase is not None and session.current_phase.phase_id == "side_a"
    session.process_frame(FrameObservation(t, 640, 480, [_front_pose()]), now_monotonic=t)
    assert "turn left" in session.hint_message.lower()

    # Turn left (left side near camera, facing right) with occluded right shoulder/hip (vis 0.40)
    # and complete within the faster 1.5s side hold window (1.6s total)
    t += 0.1
    pose_left = _profile_with_occluded_far_side(near_side="left", facing="right")
    session.process_frame(FrameObservation(t, 640, 480, [pose_left]), now_monotonic=t)
    assert session.state == "Settle"
    t += 0.8
    session.process_frame(FrameObservation(t, 640, 480, [pose_left]), now_monotonic=t)
    assert session.state == "Capture"
    for i in range(1, 17):
        ts = t + i * 0.1
        # Inject a single-frame view flicker (<= 0.35s) mid-capture and confirm capture continues
        if i == 8:
            session.process_frame(
                FrameObservation(ts, 640, 480, [_front_pose()]),
                now_monotonic=ts,
                speak_enabled=False,
            )
            assert session.state == "Capture"
            continue
        session.process_frame(
            FrameObservation(ts, 640, 480, [pose_left]),
            now_monotonic=ts,
            speak_enabled=False,
        )
    t += 1.7

    # Turn right must advance after the 1.5s side_a capture and complete in 1.5s too
    assert session.current_phase is not None and session.current_phase.phase_id == "side_b"
    session.process_frame(FrameObservation(t, 640, 480, [_front_pose()]), now_monotonic=t)
    assert "turn right" in session.hint_message.lower()

    t += 0.1
    pose_right = _profile_with_occluded_far_side(near_side="right", facing="left")
    session.process_frame(FrameObservation(t, 640, 480, [pose_right]), now_monotonic=t)
    assert session.state == "Settle"
    t += 0.8
    session.process_frame(FrameObservation(t, 640, 480, [pose_right]), now_monotonic=t)
    assert session.state == "Capture"
    for i in range(1, 17):
        ts = t + i * 0.1
        session.process_frame(
            FrameObservation(ts, 640, 480, [pose_right]),
            now_monotonic=ts,
            speak_enabled=False,
        )

    assert session.state == "Review"
    passages = session.accepted_passages
    assert set(passages) == {"front_a", "front_b", "side_a", "side_b"}
    assert passages["side_a"].detected_side == "left"
    assert passages["side_a"].detected_facing == "right"
    assert passages["side_b"].detected_side == "right"
    assert passages["side_b"].detected_facing == "left"
    for pid in ("side_a", "side_b"):
        assert (passages[pid].passage.end - passages[pid].passage.start) <= 1.7
        assert "trunk_inclination" in passages[pid].eligible_metrics
        assert "head_forward" in passages[pid].eligible_metrics


def test_turn_direction_enforcement_and_buffered_side_recording():
    """Turn left requires looking left in real life (facing right in unmirrored image)
    and buffered side poses (view_width up to max_view_width + margin) must be recorded."""
    cfg = Config(side="left", facing="right", max_view_width=0.45)

    def _buffered_side(facing: str, near_side: str) -> dict[str, Landmark]:
        far_side = "right" if near_side == "left" else "left"
        nose_x = 0.56 if facing == "right" else 0.44
        # In 640x480: dx=0.09 -> 57.6px, dy=0.24 -> 115.2px => view_width = 0.50 (buffer zone 0.45..0.60)
        return {
            "nose": _lm(nose_x, 0.18),
            "left_ear": _lm(0.51, 0.18),
            "right_ear": _lm(0.49, 0.18, 0.4),
            f"{near_side}_shoulder": _lm(0.545, 0.32, 0.95),
            f"{far_side}_shoulder": _lm(0.455, 0.32, 0.5),
            f"{near_side}_hip": _lm(0.545, 0.56, 0.95),
            f"{far_side}_hip": _lm(0.455, 0.56, 0.5),
            f"{near_side}_knee": _lm(0.51, 0.72, 0.95),
            f"{far_side}_knee": _lm(0.49, 0.72, 0.5),
            f"{near_side}_ankle": _lm(0.51, 0.90, 0.95),
            f"{far_side}_ankle": _lm(0.49, 0.90, 0.5),
        }

    # Turn left (dancer looks left -> faces right of unmirrored image)
    obs_look_left = FrameObservation(0.0, 640, 480, [_buffered_side("right", "left")])
    obs_look_right = FrameObservation(0.0, 640, 480, [_buffered_side("left", "right")])

    g_ok, _hint_ok = classify_guide_view(
        obs_look_left, cfg, requested_view="side", turn_direction="left"
    )
    assert g_ok == "side"

    g_wrong, hint_wrong = classify_guide_view(
        obs_look_right, cfg, requested_view="side", turn_direction="left"
    )
    assert g_wrong == "uncertain"
    assert "turn left" in hint_wrong.lower()

    # Turn right (dancer looks right -> faces left of unmirrored image)
    g_ok_r, _hint_ok_r = classify_guide_view(
        obs_look_right, cfg, requested_view="side", turn_direction="right"
    )
    assert g_ok_r == "side"
    g_wrong_r, hint_wrong_r = classify_guide_view(
        obs_look_left, cfg, requested_view="side", turn_direction="right"
    )
    assert g_wrong_r == "uncertain"
    assert "turn right" in hint_wrong_r.lower()

    # SoloCameraSession records buffered side observations instead of dropping them at max_view_width=0.45
    phases = [p for p in build_quick_setup_phases() if p.phase_id == "side_a"]
    session = SoloCameraSession(cfg, now_monotonic=0.0, auto_detect_side_facing=True)
    session.start_plan(phases, now_monotonic=0.0, speak_enabled=False)

    # Wrong turn direction must keep waiting and give directional feedback
    session.process_frame(FrameObservation(0.0, 640, 480, [_buffered_side("left", "right")]), now_monotonic=0.0)
    assert session.state == "Wait for view and pose"
    assert "turn left" in session.hint_message.lower()

    # Correct turn direction ("left") enters Settle -> Capture and records buffered frames
    session.process_frame(FrameObservation(0.1, 640, 480, [_buffered_side("right", "left")]), now_monotonic=0.1)
    assert session.state == "Settle"
    session.process_frame(FrameObservation(0.9, 640, 480, [_buffered_side("right", "left")]), now_monotonic=0.9)
    assert session.state == "Capture"
    for i in range(1, 17):
        ts = 0.9 + i * 0.1
        session.process_frame(
            FrameObservation(ts, 640, 480, [_buffered_side("right", "left")]),
            now_monotonic=ts,
            speak_enabled=False,
        )
    assert session.state == "Review"
    assert session.accepted_passages["side_a"].usable
    assert session.accepted_passages["side_a"].eligible_metrics["trunk_inclination"] >= 1.0
    assert session.accepted_passages["side_a"].detected_facing == "right"


def test_resolve_side_frame_config_and_ledger_metric_samples():
    """resolve_side_frame_config leaves front holds intact and repairs opposite-side holds per ledger."""
    from tango_posture.calibration.camera_session import (
        ledger_metric_samples,
        resolve_side_frame_config,
    )

    cfg = Config(side="left", facing="right", max_view_width=0.45)

    def _side_hold_pose(near_side: str, facing: str) -> dict[str, Landmark]:
        far_side = "right" if near_side == "left" else "left"
        nose_x = 0.56 if facing == "right" else 0.44
        return {
            "nose": _lm(nose_x, 0.20, 0.95),
            f"{near_side}_ear": _lm(0.50, 0.20, 0.95),
            f"{far_side}_ear": _lm(0.50, 0.20, 0.10),
            f"{near_side}_shoulder": _lm(0.50, 0.32, 0.95),
            f"{far_side}_shoulder": _lm(0.49, 0.32, 0.35),
            f"{near_side}_hip": _lm(0.50, 0.58, 0.95),
            f"{far_side}_hip": _lm(0.49, 0.58, 0.35),
            f"{near_side}_knee": _lm(0.51, 0.74, 0.95),
            f"{far_side}_knee": _lm(0.49, 0.74, 0.35),
            f"{near_side}_ankle": _lm(0.50, 0.90, 0.95),
            f"{far_side}_ankle": _lm(0.50, 0.90, 0.35),
        }

    ledger = [
        {"phase_id": "front_a", "view": "front", "start": 0.0, "end": 0.5, "detected_side": "left", "detected_facing": "right"},
        {"phase_id": "side_a", "view": "side", "start": 1.0, "end": 1.5, "detected_side": "right", "detected_facing": "right"},
        {"phase_id": "side_b", "view": "side", "start": 2.0, "end": 2.5, "detected_side": "left", "detected_facing": "left"},
    ]

    front_fr = FrameObservation(0.2, 640, 480, [_front_pose()])
    out_front, out_front_cfg = resolve_side_frame_config(front_fr, cfg, ledger)
    assert out_front is front_fr
    assert out_front_cfg.side == cfg.side and out_front_cfg.facing == cfg.facing

    side_a_fr = FrameObservation(1.2, 640, 480, [_side_hold_pose("right", "right")])
    out_a, cfg_a = resolve_side_frame_config(side_a_fr, cfg, ledger)
    assert (cfg_a.side, cfg_a.facing) == ("right", "right")
    assert (out_a.poses[0]["left_shoulder"].visibility or 0.0) >= cfg.confidence

    side_b_fr = FrameObservation(2.2, 640, 480, [_side_hold_pose("left", "left")])
    out_b, cfg_b = resolve_side_frame_config(side_b_fr, cfg, ledger)
    assert (cfg_b.side, cfg_b.facing) == ("left", "left")
    assert (out_b.poses[0]["right_shoulder"].visibility or 0.0) >= cfg.confidence

    samples = ledger_metric_samples([front_fr, side_a_fr, side_b_fr], cfg, ledger, view="side")
    for mk in ("head_pitch", "head_forward", "head_inclination", "trunk_inclination", "hip_offset", "knee_flexion"):
        vals = [val for _, val in samples[mk]]
        assert vals[0] is None
        assert vals[1] is not None and vals[2] is not None


def test_dual_ear_equal_visibility_uses_z_depth_ignoring_stale_cfg_side():
    """Real MediaPipe frames report ~1.0 visibility on both ears; z-depth must determine side and facing."""
    from tango_posture.calibration.camera_session import _detect_side_orientation

    def _mediapipe_side_pose(*, near_side: str, facing: str) -> dict[str, Landmark]:
        # In 60-85 degree profiles, the far ear can sit slightly ahead of the nose in x,
        # while both ears have visibility=0.99 and z differs meaningfully.
        far_side = "right" if near_side == "left" else "left"
        nose_x = 0.520 if facing == "right" else 0.480
        near_ear_x = 0.485 if facing == "right" else 0.515
        far_ear_x = 0.525 if facing == "right" else 0.475  # slightly ahead of nose!
        near_z, far_z = -0.15, 0.08
        return {
            "nose": Landmark(nose_x, 0.20, 0.99, 0.99, -0.08),
            f"{near_side}_ear": Landmark(near_ear_x, 0.20, 0.99, 0.99, near_z),
            f"{far_side}_ear": Landmark(far_ear_x, 0.20, 0.99, 0.99, far_z),
            f"{near_side}_shoulder": Landmark(0.50, 0.32, 0.99, 0.99, near_z),
            f"{far_side}_shoulder": Landmark(0.51, 0.32, 0.99, 0.99, far_z),
            f"{near_side}_hip": Landmark(0.50, 0.58, 0.99, 0.99, near_z),
            f"{far_side}_hip": Landmark(0.51, 0.58, 0.99, 0.99, far_z),
            f"{near_side}_knee": Landmark(0.50, 0.74, 0.99, 0.99, near_z),
            f"{far_side}_knee": Landmark(0.51, 0.74, 0.99, 0.99, far_z),
            f"{near_side}_ankle": Landmark(0.50, 0.90, 0.99, 0.99, near_z),
            f"{far_side}_ankle": Landmark(0.51, 0.90, 0.99, 0.99, far_z),
        }

    # Turn left (left side near camera, facing right) even when cfg.side is stale "right"
    cfg_stale_right = Config(side="right", facing="left")
    pose_left = _mediapipe_side_pose(near_side="left", facing="right")
    orient_l = _detect_side_orientation(pose_left, cfg_stale_right)
    assert orient_l is not None
    assert orient_l[0] == "left" and orient_l[1] == "right"
    view_l, hint_l = classify_guide_view(
        FrameObservation(0.0, 640, 480, [pose_left]),
        cfg_stale_right,
        requested_view="side",
        turn_direction="left",
    )
    assert view_l == "side", hint_l

    # Turn right (right side near camera, facing left) even when cfg.side is stale "left"
    cfg_stale_left = Config(side="left", facing="right")
    pose_right = _mediapipe_side_pose(near_side="right", facing="left")
    orient_r = _detect_side_orientation(pose_right, cfg_stale_left)
    assert orient_r is not None
    assert orient_r[0] == "right" and orient_r[1] == "left"
    view_r, hint_r = classify_guide_view(
        FrameObservation(0.0, 640, 480, [pose_right]),
        cfg_stale_left,
        requested_view="side",
        turn_direction="right",
    )
    assert view_r == "side", hint_r

    # Opposite turn direction reports clear diagnostic hint
    view_opp, hint_opp = classify_guide_view(
        FrameObservation(0.0, 640, 480, [pose_right]),
        cfg_stale_left,
        requested_view="side",
        turn_direction="left",
    )
    assert view_opp == "uncertain"
    assert "opposite side" in hint_opp.lower()
    assert "turn left" in hint_opp.lower()


def test_torso_scaled_profile_threshold_accepts_widescreen_and_rejects_front_pose():
    """1280x720 wide-shot side profile with dx=0.007 (<0.010) passes via torso scaling,
    while a narrow-shouldered front-facing pose is still rejected."""
    cfg = Config(side="left", facing="right", max_view_width=0.45)

    # Full-body widescreen profile: torso_px = 0.18 * 720 = 129.6px; dx = 0.007 * 1280 = 8.96px (0.069 * torso_px)
    wide_side = {
        "nose": Landmark(0.507, 0.20, 0.99, 0.99, -0.05),
        "left_ear": Landmark(0.500, 0.20, 0.99, 0.99, -0.10),
        "right_ear": Landmark(0.501, 0.20, 0.99, 0.99, 0.05),
        "left_shoulder": Landmark(0.500, 0.30, 0.99, 0.99, -0.10),
        "right_shoulder": Landmark(0.495, 0.30, 0.99, 0.99, 0.05),
        "left_hip": Landmark(0.500, 0.48, 0.99, 0.99, -0.10),
        "right_hip": Landmark(0.495, 0.48, 0.99, 0.99, 0.05),
        "left_ankle": Landmark(0.500, 0.80, 0.99, 0.99, -0.10),
        "right_ankle": Landmark(0.495, 0.80, 0.99, 0.99, 0.05),
    }
    view_wide, hint_wide = classify_guide_view(
        FrameObservation(0.0, 1280, 720, [wide_side]),
        cfg,
        requested_view="side",
        turn_direction="left",
    )
    assert view_wide == "side", hint_wide

    # Narrow-shouldered front-facing pose: nose centered between left_ear and right_ear
    narrow_front = {
        "nose": Landmark(0.500, 0.20, 0.99, 0.99, -0.05),
        "left_ear": Landmark(0.520, 0.20, 0.99, 0.99, 0.0),
        "right_ear": Landmark(0.480, 0.20, 0.99, 0.99, 0.0),
        "left_shoulder": Landmark(0.525, 0.30, 0.99, 0.99, 0.0),
        "right_shoulder": Landmark(0.475, 0.30, 0.99, 0.99, 0.0),
        "left_hip": Landmark(0.520, 0.56, 0.99, 0.99, 0.0),
        "right_hip": Landmark(0.480, 0.56, 0.99, 0.99, 0.0),
        "left_ankle": Landmark(0.520, 0.90, 0.99, 0.99, 0.0),
        "right_ankle": Landmark(0.480, 0.90, 0.99, 0.99, 0.0),
    }
    view_nf, hint_nf = classify_guide_view(
        FrameObservation(0.0, 1280, 720, [narrow_front]),
        cfg,
        requested_view="side",
        turn_direction="left",
    )
    assert view_nf == "uncertain"
    assert "turn left" in hint_nf.lower()


def test_settle_brief_view_flicker_grace_and_prolonged_loss():
    """Brief <=0.25s view flicker in Settle keeps settling; >0.25s loss returns to Wait for view and pose."""
    cfg = Config()
    session = SoloCameraSession(cfg, now_monotonic=0.0)
    phases = build_quick_setup_phases()[:1]  # front_a
    session.start_plan(phases, now_monotonic=0.0, speak_enabled=False)

    # Enter Settle at t=1.0
    session.process_frame(FrameObservation(1.0, 640, 480, [_front_pose()]), now_monotonic=1.0)
    assert session.state == "Settle"

    # Brief 0.15s view loss (<= 0.25s) preserves Settle
    session.process_frame(FrameObservation(1.15, 640, 480, [_side_pose("right")]), now_monotonic=1.15)
    assert session.state == "Settle"

    # Recover view at t=1.25 -> still in Settle, then completes settle at t=1.80 -> Capture
    session.process_frame(FrameObservation(1.25, 640, 480, [_front_pose()]), now_monotonic=1.25)
    assert session.state == "Settle"
    session.process_frame(FrameObservation(1.80, 640, 480, [_front_pose()]), now_monotonic=1.80)
    assert session.state == "Capture"

    # Retry phase and test prolonged >0.25s view loss in Settle resets to Wait for view and pose
    session.retry_phase(now_monotonic=2.0, speak_enabled=False)
    session.process_frame(FrameObservation(2.1, 640, 480, [_front_pose()]), now_monotonic=2.1)
    assert session.state == "Settle"
    session.process_frame(FrameObservation(2.2, 640, 480, [_side_pose("right")]), now_monotonic=2.2)
    assert session.state == "Settle"
    session.process_frame(FrameObservation(2.5, 640, 480, [_side_pose("right")]), now_monotonic=2.5)
    assert session.state == "Wait for view and pose"
