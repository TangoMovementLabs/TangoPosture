from copy import deepcopy

import numpy as np
import pytest

from tango_posture.config import Config
from tango_posture.domain import FrameObservation, Landmark
from tango_posture.pose.interpolation import interpolate_poses
from tango_posture.references.coach import interpolate_coach_data, render_coach_frame_overlay


def observation(timestamp, x=None, confidence=1):
    pose = {"left_shoulder": Landmark(x=0.4, y=0.25, visibility=1, presence=1)}
    if x is not None:
        pose["left_hip"] = Landmark(x=x, y=0.5, visibility=confidence, presence=confidence)
    return FrameObservation(timestamp, 100, 100, [pose])


@pytest.mark.parametrize(("method", "expected"), [("linear", 0.6), ("quadratic", 0.5)])
def test_missing_joint_follows_selected_curve_without_changing_valid_points(method, expected):
    frames = [
        observation(1000, 0.1), observation(1001, 0.2),
        observation(1002, 0.9, confidence=0.2), observation(1003, 1),
    ]
    before = deepcopy(frames)
    result = interpolate_poses(frames, method=method, max_gap=3, confidence=0.7)
    assert result[2].poses[0]["left_hip"].x == pytest.approx(expected)
    assert result[2].timestamp == 1002
    assert result[0] is frames[0] and result[1] is frames[1] and result[3] is frames[3]
    assert result[2].poses[0]["left_shoulder"] == frames[2].poses[0]["left_shoulder"]
    assert frames == before


@pytest.mark.parametrize(("max_gap", "filled"), [(3, True), (2.99, False), (0, False)])
def test_maximum_gap_and_disabled_interpolation(max_gap, filled):
    frames = [observation(0, 0.2), observation(1), observation(2), observation(3, 0.8)]
    result = interpolate_poses(frames, method="linear", max_gap=max_gap, confidence=0.7)
    assert ("left_hip" in result[1].poses[0]) is filled
    assert ("left_hip" in result[2].poses[0]) is filled
    if filled:
        assert result[1].poses[0]["left_hip"].x == pytest.approx(0.4)
        assert result[2].poses[0]["left_hip"].x == pytest.approx(0.6)


def test_does_not_extrapolate_or_bridge_ambiguous_dancers():
    frames = [
        observation(0), observation(1, 0.1), observation(2), observation(3, 0.3),
        observation(4), observation(5, 0.5), observation(6),
    ]
    frames[3].poses.append(dict(frames[3].poses[0]))
    result = interpolate_poses(frames, method="linear", max_gap=10, confidence=0.7)
    assert all("left_hip" not in result[index].poses[0] for index in (0, 2, 4, 6))


@pytest.mark.parametrize("extra_anchor", [None, 0.9])
def test_quadratic_uses_linear_when_unsupported_or_outside_frame(extra_anchor):
    frames = [observation(0, extra_anchor), observation(1, 0), observation(2), observation(3, 0)]
    result = interpolate_poses(frames, method="quadratic", max_gap=3, confidence=0.7)
    assert result[2].poses[0]["left_hip"].x == pytest.approx(0)


def test_rendered_connections_bridge_missing_joints_without_replacing_raw_evidence():
    cfg = Config()
    data = {"width": 100, "height": 100, "frames": {
        str(index): {
            "timestamp": index,
            "measurement_poses": [{
                "left_shoulder": [0.4, 0.25, 0, 1, 1],
                **({"left_hip": [x, 0.5, 0, 1, 1]} if x is not None else {}),
            }],
            "message": "", "calibration_rows": [{"metric": "hip_offset", "state": "UNKNOWN"}],
        }
        for index, x in enumerate((0.2, None, 0.6))
    }}
    original = deepcopy(data)
    interpolated = interpolate_coach_data(data, cfg)
    image = np.zeros((100, 100, 3), np.uint8)
    raw_image = render_coach_frame_overlay(image, data, 1, cfg, lines_only=True)
    drawn = render_coach_frame_overlay(image, interpolated, 1, cfg, lines_only=True)
    assert raw_image[40, 40].sum() == 0
    assert drawn[40, 40, 1] > drawn[40, 40, 2]
    assert data == original
    assert interpolated["frames"]["1"]["calibration_rows"] == data["frames"]["1"]["calibration_rows"]
    assert "left_hip" not in interpolated["frames"]["1"]["measurement_poses"][0]


def test_interpolated_skeleton_does_not_invent_posture_measurement_annotations(monkeypatch):
    labels = []
    monkeypatch.setattr(
        "tango_posture.overlay.cv2.putText",
        lambda image, text, *args, **kwargs: labels.append(text) or image,
    )
    data = {"width": 100, "height": 100, "frames": {"0": {
        "measurement_poses": [],
        "display_poses": [{
            "nose": [0.7, 0.3, 0, 1, 1],
            "left_ear": [0.4, 0.2, 0, 1, 1],
            "right_ear": [0.45, 0.2, 0, 1, 1],
            "left_shoulder": [0.4, 0.5, 0, 1, 1],
            "left_hip": [0.4, 0.8, 0, 1, 1],
        }],
    }}}
    image = render_coach_frame_overlay(
        np.zeros((100, 100, 3), np.uint8), data, 0, Config(),
    )
    assert image[:, :, 1].max() > 0
    assert not any(label.startswith(("Head horiz:", "Shoulders:")) for label in labels)


def test_interpolate_coach_data_recovers_limbs_across_pivots_and_low_confidence_boundaries():
    cfg = Config(replay_pivot_interpolation_max_gap_s=0.20, replay_joint_max_gap=3.0)
    timestamps = [round(i * 0.1, 2) for i in range(10)]  # 0.0 .. 0.9 s

    def _lm(x: float, y: float, conf: float = 0.95) -> list[float]:
        return [x, y, 0.0, conf, conf]

    frames: dict[str, dict] = {}
    for idx, ts in enumerate(timestamps):
        is_boundary = idx in (0, 9)
        is_pivot_gap = 3 <= idx <= 6  # 0.3..0.6 -> 0.5s gap (0.2..0.7) > 0.20s pivot clamp
        limb_conf = 0.25 if (is_boundary or is_pivot_gap) else 0.95
        pose = {
            "nose": _lm(0.50, 0.15),
            "left_ear": _lm(0.47, 0.16),
            "right_ear": _lm(0.53, 0.16),
            "left_shoulder": _lm(0.42, 0.25),
            "right_shoulder": _lm(0.58, 0.25),
            "left_elbow": _lm(0.40, 0.36),
            "right_elbow": _lm(0.60, 0.36, limb_conf),
            "left_wrist": _lm(0.39, 0.45),
            "right_wrist": _lm(0.61, 0.45),
            "left_hip": _lm(0.45, 0.48),
            "right_hip": _lm(0.55, 0.48),
            "left_knee": _lm(0.45, 0.65, 0.25 if is_boundary else 0.95),
            "left_ankle": _lm(0.45, 0.83, 0.25 if is_boundary else 0.95),
            "right_knee": _lm(0.55, 0.65),
            "right_ankle": _lm(0.55, 0.83),
        }
        if is_pivot_gap:
            pose.pop("left_knee")
            pose.pop("left_ankle")
        frames[str(idx)] = {
            "timestamp": ts,
            "measurement_poses": [pose],
            "poses": [deepcopy(pose)],
        }

    data = {"width": 100, "height": 100, "frames": frames}
    original = deepcopy(data)
    movement = {
        "pivots": [{"start_time": 0.2, "end_time": 0.7, "start_frame": 2, "end_frame": 8}],
        "cut_frames": [],
    }
    interpolated = interpolate_coach_data(data, cfg, movement_data=movement)

    # Raw measurement_poses must stay untouched
    assert data == original
    for idx in range(10):
        assert interpolated["frames"][str(idx)]["measurement_poses"] == original["frames"][str(idx)]["measurement_poses"]

    # Pivot lower-limb gap > 0.20s is recovered in display_poses
    for idx in (3, 4, 5, 6):
        f = interpolated["frames"][str(idx)]
        assert "left_knee" in f["display_poses"][0]
        assert "left_ankle" in f["display_poses"][0]
        assert f["display_joint_origins"]["left_knee"] == "interpolated"
        assert f["display_joint_origins"]["left_ankle"] == "interpolated"
        assert f["display_poses"][0]["left_knee"][3] >= cfg.confidence

    # Unbracketed start/end frames with low-conf (0.25) limbs are recovered in display_poses
    for idx in (0, 9):
        f = interpolated["frames"][str(idx)]
        for j in ("left_knee", "left_ankle", "right_elbow"):
            assert f["display_poses"][0][j][3] >= cfg.confidence
            assert f["display_joint_origins"][j] == "interpolated"
        assert original["frames"][str(idx)]["measurement_poses"][0]["right_elbow"][3] == pytest.approx(0.25)


def test_interpolate_coach_data_recovers_whole_frame_boundary_dropout_and_near_edge_limbs():
    def _pt(x: float, y: float, conf: float = 0.95) -> list[float]:
        return [x, y, 0.0, conf, conf]

    cfg = Config(confidence=0.65, replay_joint_max_gap=3.0)
    pose_1 = {
        "nose": _pt(0.50, 0.15),
        "left_shoulder": _pt(0.42, 0.26),
        "right_shoulder": _pt(0.58, 0.26),
        "left_elbow": _pt(0.39, 0.40),
        "right_elbow": _pt(0.61, 0.40, 0.10),
        "left_wrist": _pt(0.38, 0.52),
        "right_wrist": _pt(0.62, 0.52),
        "left_hip": _pt(0.45, 0.54),
        "right_hip": _pt(0.55, 0.54),
        "left_knee": _pt(0.45, 0.85),
        "left_ankle": _pt(0.45, 1.04, 0.40),
        "right_knee": _pt(0.55, 0.74),
        "right_ankle": _pt(0.55, 0.91),
    }
    pose_2 = {
        "nose": _pt(0.50, 0.15),
        "left_shoulder": _pt(0.42, 0.26),
        "right_shoulder": _pt(0.58, 0.26),
        "left_elbow": _pt(0.39, 0.40),
        "right_elbow": _pt(0.61, 0.40, 0.10),
        "left_wrist": _pt(0.38, 0.52),
        "right_wrist": _pt(0.62, 0.52),
        "left_hip": _pt(0.45, 0.54),
        "right_hip": _pt(0.55, 0.54),
        "left_knee": _pt(0.45, 0.75),
        "left_ankle": _pt(0.45, 0.92),
        "right_knee": _pt(0.55, 0.74),
        "right_ankle": _pt(0.55, 0.91),
    }
    data = {
        "width": 100,
        "height": 100,
        "frames": {
            "0": {"timestamp": 0.0, "measurement_poses": [], "poses": []},
            "1": {"timestamp": 0.1, "measurement_poses": [pose_1], "poses": [deepcopy(pose_1)]},
            "2": {"timestamp": 0.2, "measurement_poses": [pose_2], "poses": [deepcopy(pose_2)]},
        },
    }
    original = deepcopy(data)

    interpolated = interpolate_coach_data(data, cfg)

    assert data == original
    for idx in ("0", "1", "2"):
        assert interpolated["frames"][idx]["measurement_poses"] == original["frames"][idx]["measurement_poses"]

    # Whole-frame boundary dropout at frame 0 is recovered with bounded coordinates
    f0 = interpolated["frames"]["0"]
    assert len(f0["display_poses"]) == 1
    for joint in ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_elbow"):
        lm = f0["display_poses"][0][joint]
        assert 0.0 <= lm[0] <= 1.0
        assert 0.0 <= lm[1] <= 1.0
        assert lm[3] >= cfg.confidence
        assert f0["display_joint_origins"][joint] == "interpolated"

    # Near-edge left_ankle (raw y=1.04, vis=0.40) and always-low-conf right_elbow (0.10) are recovered & clamped
    f1 = interpolated["frames"]["1"]
    ankle1 = f1["display_poses"][0]["left_ankle"]
    assert 0.0 <= ankle1[0] <= 1.0
    assert 0.0 <= ankle1[1] <= 1.0
    assert ankle1[3] >= cfg.confidence
    assert f1["display_joint_origins"]["left_ankle"] == "interpolated"
    assert original["frames"]["1"]["measurement_poses"][0]["left_ankle"][1] == pytest.approx(1.04)
    assert original["frames"]["1"]["measurement_poses"][0]["left_ankle"][3] == pytest.approx(0.40)

    for idx in ("1", "2"):
        re = interpolated["frames"][idx]["display_poses"][0]["right_elbow"]
        assert 0.0 <= re[0] <= 1.0
        assert 0.0 <= re[1] <= 1.0
        assert re[3] >= cfg.confidence
        assert interpolated["frames"][idx]["display_joint_origins"]["right_elbow"] == "interpolated"
        assert original["frames"][idx]["measurement_poses"][0]["right_elbow"][3] == pytest.approx(0.10)
