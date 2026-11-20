import math
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest
from conftest import frame

from tango_posture.archive import read_calibration_archive, read_manifest, rename_archive
from tango_posture.calibration.archives import read_range_profile, save_range_profile
from tango_posture.calibration.baseline import compatible, profile
from tango_posture.calibration.evidence import (
    Evidence,
    dancer_frames,
    is_static_pose_run,
    single_dancer_anchors,
)
from tango_posture.calibration.measurements import (
    measure_profile,
    measure_ranges,
    measure_targets,
    metric_samples,
    view_eligible,
)
from tango_posture.calibration.models import METRIC_DETAILS, ApprovedRange, Passage, RangeProfile
from tango_posture.calibration.statistics import repeat_check, summarize_examples
from tango_posture.config import Config
from tango_posture.domain import Landmark, MetricObservation, QualityResult
from tango_posture.geometry import torso_axis_alignment
from tango_posture.models import MANIFEST
from tango_posture.references.coach import (
    _serialise_pose,
    compare_coach_data,
    compare_default_coach_data,
    interpolate_coach_data,
)
from tango_posture.session import Session
from tango_posture.storage import Storage
from tango_posture.temporal import Evaluator


def range_profile(*, checked=False, name="Walking"):
    target = ApprovedRange(
        lower=-5, upper=5, unit="degrees", view="side",
        passages=[Passage(start=0, end=2)], median=0, p10=-3, p90=3,
        valid_duration=1.8, selected_duration=2, samples=20, inside_fraction=1,
        camera_check_passed=checked,
    )
    return RangeProfile(
        name=name, context="Solo walking", model_hash=MANIFEST["sha256"],
        capture_profile=profile(Config(), 640, 480, "camera:0"),
        source={"archive": "unavailable.tpr"}, metrics={"head_pitch": target},
    )


@pytest.fixture
def coaching_cache():
    observations = [frame(timestamp) for timestamp in (0, 0.1, 0.2)]
    for observation in observations:
        for side, x in (("left", 320), ("right", 330)):
            observation.poses[0][f"{side}_knee"] = Landmark(
                x=x / 640, y=385 / 480, visibility=0.99, presence=0.99,
            )
    return {
        "width": 640, "height": 480, "model_hash": MANIFEST["sha256"],
        "capture_profile": profile(Config(), 640, 480, "video"),
        "frames": {
            str(index): {
                "timestamp": observation.timestamp,
                "poses": [_serialise_pose(observation.poses[0])],
                "measurement_poses": [_serialise_pose(observation.poses[0])],
                "message": "LOWER YOUR SHOULDERS / BEND YOUR KNEES / BAD HEAD ALIGNMENT",
            }
            for index, observation in enumerate(observations)
        },
    }, observations


@pytest.mark.parametrize("key", [
    "head_pitch", "shoulder_clearance", "hip_offset", "knee_flexion", "torso_axis_tilt",
])
@pytest.mark.parametrize("direction", [-1, 1])
def test_cached_coaching_uses_both_approved_limits_not_settings(coaching_cache, key, direction):
    data, observations = coaching_cache
    cfg = Config(entry_duration=0.1, smoothing=0.0001)
    measured = measure_ranges(observations[0], cfg).values[key]
    assert measured is not None
    target = range_profile()
    limits = target.metrics["head_pitch"].model_copy(update={
        "lower": measured + direction * 10 - 1, "upper": measured + direction * 10 + 1,
        "unit": METRIC_DETAILS[key][2], "view": "all",
    })
    target.metrics = {key: limits}
    compared = compare_coach_data(data, cfg, target.calibration())
    entry = compared["frames"]["2"]
    assert "outside approved range" in entry["message"]
    assert METRIC_DETAILS[key][1] in entry["message"]
    assert entry["calibration_rows"][0]["state"] == "DEVIATION"
    assert len(entry["calibration_rows"]) == 1
    assert all(
        text not in entry["message"]
        for text in ("LOWER YOUR SHOULDERS", "BEND YOUR KNEES", "BAD HEAD ALIGNMENT")
    )
    assert data["frames"]["2"]["message"].startswith("LOWER YOUR SHOULDERS")
    limits.lower, limits.upper = measured - 1, measured + 1
    inside = compare_coach_data(data, cfg, target.calibration())
    assert inside["frames"]["2"]["message"] == ""
    assert inside["frames"]["2"]["calibration_rows"][0]["state"] == "WITHIN_REFERENCE"


def test_cached_comparison_rejects_unverified_evidence_and_abstains_on_tracking_loss(coaching_cache):
    data, _ = coaching_cache
    target = range_profile()
    target.metrics["head_pitch"].view = "all"
    data["frames"]["1"]["measurement_poses"] = []
    compared = compare_coach_data(data, Config(entry_duration=0.1), target.calibration())
    assert compared["frames"]["1"]["message"] == ""
    assert compared["frames"]["1"]["calibration_rows"][0]["state"] == "UNKNOWN"
    data["model_hash"] = "0" * 64
    with pytest.raises(ValueError, match="different measurement models"):
        compare_coach_data(data, Config(), target.calibration())
    data["model_hash"] = target.model_hash
    del data["frames"]["0"]["measurement_poses"]
    with pytest.raises(ValueError, match="display landmarks only"):
        compare_coach_data(data, Config(), target.calibration())


@pytest.mark.parametrize("direction", [-1, 1])
def test_pink_axis_calibration_matches_overlay_magnitude_not_legacy_trunk(direction):
    observation = frame(0)
    pose = observation.poses[0]
    for side, x in (("left", 0.4), ("right", 0.5)):
        pose[f"{side}_hip"] = replace(pose[f"{side}_hip"], x=x, y=0.7)
        pose[f"{side}_shoulder"] = replace(pose[f"{side}_shoulder"], x=x, y=0.3)
    pose["right_shoulder"] = replace(pose["right_shoulder"], x=0.5 + direction * 0.1)
    cfg = Config(max_view_width=2)
    alignment = torso_axis_alignment(pose, 640, 480, cfg.confidence)
    assert alignment is not None
    expected = abs(alignment[0])
    measured = measure_ranges(observation, cfg)
    assert measured.values["torso_axis_tilt"] == pytest.approx(expected)
    assert expected > 5
    assert measured.quality["torso_axis_tilt"].valid
    assert measured.values["trunk_inclination"] == pytest.approx(0)
    assert measure_ranges(
        observation, cfg.model_copy(update={"facing": "left"}),
    ).values["torso_axis_tilt"] == pytest.approx(expected)


@pytest.mark.parametrize("missing", ["left_shoulder", "right_shoulder", "left_hip", "right_hip"])
def test_pink_axis_calibration_abstains_when_any_required_joint_is_unreliable(missing):
    observation = frame(0)
    pose = observation.poses[0]
    pose[missing] = replace(pose[missing], visibility=0.1)
    measured = measure_ranges(observation, Config(allow_all_angles=True))
    assert measured.values["torso_axis_tilt"] is None
    assert not measured.quality["torso_axis_tilt"].valid


def test_settings_default_ranges_apply_without_calibration_and_ignore_interpolated_evidence(coaching_cache):
    data, observations = coaching_cache
    value = measure_ranges(observations[0], Config()).values["head_pitch"]
    cfg = Config(entry_duration=0.1, smoothing=0.0001, calibration_defaults={
        "head_pitch": {"lower": value + 10, "upper": value + 20, "unit": "degrees", "view": "all"},
    })
    data["frames"]["1"]["measurement_poses"] = []
    displayed = interpolate_coach_data(data, cfg)
    assert displayed["frames"]["1"]["display_poses"]
    compared = compare_default_coach_data(displayed, cfg)
    assert compared["frames"]["1"]["calibration_rows"][0]["state"] == "UNKNOWN"
    assert compared["frames"]["1"]["calibration_rows"][0]["smoothed"] is None
    assert {row["metric"] for row in compared["frames"]["0"]["calibration_rows"]} == {"head_pitch"}
    assert not compared["settings_coaching"]
    assert compared["calibration_name"] == "Settings default ranges"
    cfg.calibration_defaults["head_pitch"].lower = value - 1
    cfg.calibration_defaults["head_pitch"].upper = value + 1
    compared = compare_default_coach_data(data, cfg)
    assert compared["frames"]["2"]["message"] == ""
    assert data["frames"]["2"]["message"].startswith("LOWER YOUR SHOULDERS")


def test_profile_is_video_independent_and_rename_persists(tmp_path):
    path = tmp_path / "Walking.tpc"
    original = range_profile()
    save_range_profile(path, original)
    assert read_manifest(path)["segments"] == []
    assert read_range_profile(path) == original
    with ZipFile(path) as archive:
        assert set(archive.namelist()) == {"manifest.json", "profile.json"}
    rename_archive(path, "My walking profile")
    assert read_range_profile(path).name == "My walking profile"
    assert read_calibration_archive(path).name == "My walking profile"
    with pytest.raises(FileExistsError):
        save_range_profile(path, original)


def test_examples_and_target_limits_are_independent_and_gaps_are_unknown():
    samples = [(0, 0), (0.1, 1), (0.2, None), (0.3, 30), (0.4, 40), (2, 50)]
    stats = summarize_examples(samples, [Passage(start=0, end=2.1)], 0.5, 0, 2)
    assert stats["valid_duration"] == pytest.approx(0.2)
    assert stats["inside_fraction"] == pytest.approx(2 / 5)
    assert stats["p90"] > 2
    assert stats["maximum_time"] == 2
    assert stats["lower"] == 0 and stats["upper"] == 2


def test_dancer_identity_ambiguity_and_missing_start_never_silently_reseed():
    first, ambiguous, recovered = frame(0), frame(0.1), frame(0.2)
    ambiguous.poses.append(dict(ambiguous.poses[0]))
    evidence = Evidence([first, ambiguous, recovered], MANIFEST["sha256"], {}, {}, 0.5)
    selected = dancer_frames(evidence, 0, 0.7)
    assert selected[0].poses
    assert not selected[1].poses and not selected[2].poses
    missing = frame(0)
    missing.poses.clear()
    evidence = Evidence([missing, recovered], MANIFEST["sha256"], {}, {}, 0.5)
    assert not any(item.poses for item in dancer_frames(evidence, 0, 0.7))


def test_angular_ranges_do_not_split_at_minus_plus_180():
    from dataclasses import replace

    frames = [frame(0), frame(0.1)]
    for observation in frames:
        observation.poses[0] = {
            name: replace(point, y=1 - point.y)
            for name, point in observation.poses[0].items()
        }
    for side in ("left", "right"):
        point = frames[1].poses[0][f"{side}_shoulder"]
        frames[1].poses[0][f"{side}_shoulder"] = replace(point, x=point.x - 0.002)
    samples = metric_samples(frames, Config())["trunk_inclination"]
    assert 180 <= samples[1][1] < 182
    target = range_profile()
    limits = target.metrics.pop("head_pitch")
    target.metrics["trunk_inclination"] = limits.model_copy(update={"lower": 175, "upper": 185})
    measured = measure_profile(frames[1], Config(), target)
    assert 175 <= measured.values["trunk_inclination"] <= 185


def test_invalid_or_nonfinite_evidence_does_not_inflate_coverage():
    samples = [(0, 0), (0.1, math.nan), (0.2, 1), (0.3, 2)]
    result = summarize_examples(samples, [Passage(start=0, end=0.4)], 0.5)
    assert result["samples"] == 3
    assert result["valid_duration"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="overlap"):
        summarize_examples(samples, [Passage(start=0, end=1), Passage(start=0.5, end=2)], 0.5)
    with pytest.raises(ValueError, match="timestamps"):
        summarize_examples([(1, 1), (0, 2)], [Passage(start=0, end=2)], 0.5)


@pytest.mark.parametrize("levels, expected", [
    ([0, 3, 0.01, 3.01], True), ([0, 0, 0, 0], False),
    ([0, 3, 2, 5], False), ([0, 3, 0, -3], False),
])
def test_repeat_check_requires_distinguishable_change_and_return(levels, expected):
    holds = [Passage(start=i * 2, end=i * 2 + 1) for i in range(4)]
    samples = [(i * 2 + j / 10, value + (j % 2) * 0.01)
               for i, value in enumerate(levels) for j in range(10)]
    assert repeat_check(samples, holds, 0.5)[0] is expected
    assert not repeat_check(samples[:2], holds, 0.5)[0]


def test_range_evaluator_detects_both_limits_and_inclusive_clear():
    cfg = Config(smoothing=0.0001, entry_duration=0.1, clear_duration=0.1)
    evaluator = Evaluator("shoulder_clearance", 0, cfg, bounds=(1, 2))
    assert evaluator.update(0, 0.5)["state"] == "PENDING"
    assert evaluator.update(0.1, 0.5)["state"] == "DEVIATION"
    assert evaluator.update(0.2, None)["state"] == "UNKNOWN"
    assert evaluator.update(0.3, 3)["delta"] == 1
    assert evaluator.update(0.4, 3)["state"] == "DEVIATION"
    evaluator.update(0.5, 2)
    assert evaluator.update(0.6, 2)["state"] == "WITHIN_REFERENCE"


def test_only_camera_checked_ranges_score_and_export_without_source(tmp_path):
    cfg = Config(smoothing=0.0001, entry_duration=0.1)
    unchecked = range_profile()
    with pytest.raises(ValueError, match="Comparison-only"):
        compatible(unchecked.calibration(), unchecked.capture_profile, unchecked.model_hash)
    with closing(Storage(tmp_path)) as storage:
        with pytest.raises(ValueError, match="Comparison-only"):
            Session(unchecked.calibration(), cfg, storage, 0, "webcam", {})
        checked = range_profile(checked=True)
        session = Session(checked.calibration(), cfg, storage, 0, "webcam", {})
        for timestamp in (0, 0.1, 0.2, 0.3):
            session.update(MetricObservation(
                timestamp, {"head_pitch": 12}, {"head_pitch": QualityResult(True)}, 0.1,
            ))
        result = session.finish(0.3)
        assert set(result["metrics"]) == {"head_pitch"}
        assert result["metrics"]["head_pitch"]["event_count"] == 1
        assert result["units"] == {"head_pitch": "degrees"}
        assert (storage.export(session.id, tmp_path / "export") / "summary.json").is_file()
    assert not Path("unavailable.tpr").exists()


def test_front_view_calibration_and_evaluation_share_view_eligibility_gate():
    sample = frame(0.0)
    cfg = Config()
    # Synthetic side-ish frame has view_width 0.0625 <= cfg.max_view_width (0.45)
    assert not view_eligible("front", 0.0625, cfg.max_view_width)
    assert view_eligible("side", 0.0625, cfg.max_view_width)
    assert view_eligible("all", 0.0625, cfg.max_view_width)

    front_samples = metric_samples([sample], cfg, view="front")
    assert front_samples["head_pitch"] == [(0.0, None)]

    target = ApprovedRange(
        lower=-90, upper=90, unit="degrees", view="front",
        passages=[Passage(start=0, end=1)], median=0, p10=-10, p90=10,
        valid_duration=1, selected_duration=1, samples=10, inside_fraction=1,
    )
    evaluated = measure_targets(sample, cfg, {"head_pitch": target})
    assert evaluated.values["head_pitch"] is None
    assert not evaluated.quality["head_pitch"].valid
    assert "approved camera view" in evaluated.quality["head_pitch"].reason

    all_samples = metric_samples([sample], cfg, view="all")
    assert all_samples["head_pitch"][0][1] is not None


def test_single_candidate_step_tolerance_and_auto_anchors():
    f0 = frame(0.0)
    f1 = frame(0.1)
    f1.poses[0] = {
        name: replace(point, x=point.x + 0.18) for name, point in f1.poses[0].items()
    }
    single_ev = Evidence([f0, f1], MANIFEST["sha256"], {}, {}, 0.5)
    selected_single = dancer_frames(single_ev, 0, 0.7)
    assert [bool(item.poses) for item in selected_single] == [True, True]

    # When multiple poses are present, keep the conservative < 0.15 step limit.
    f1_multi = frame(0.1)
    f1_multi.poses[0] = dict(f1.poses[0])
    f1_multi.poses.append({
        name: replace(point, x=point.x + 0.30) for name, point in frame(0.1).poses[0].items()
    })
    multi_ev = Evidence([f0, f1_multi], MANIFEST["sha256"], {}, {}, 0.5)
    selected_multi = dancer_frames(multi_ev, 0, 0.7)
    assert [bool(item.poses) for item in selected_multi] == [True, False]

    # Auto-link computes anchor frames across brief solo pose dropouts.
    gap = frame(0.2)
    gap.poses.clear()
    resumed = frame(0.3)
    resumed.poses[0] = dict(f1.poses[0])
    dropout_ev = Evidence([f0, f1, gap, resumed], MANIFEST["sha256"], {}, {}, 0.5)
    assert single_dancer_anchors(dropout_ev, 0.7) == {0: 0, 3: 0}

    with pytest.raises(ValueError, match="single-dancer"):
        single_dancer_anchors(multi_ev, 0.7)


def test_single_dancer_anchors_skips_trailing_static_poster_frames():
    frames = []
    # Moving dancing frames 0..14 (step > 1e-3 so not a static slide).
    for idx in range(15):
        item = frame(idx * 0.1)
        shift = 0.005 * idx
        item.poses[0] = {
            name: replace(point, x=point.x + shift) for name, point in item.poses[0].items()
        }
        frames.append(item)
    # Brief empty dropout gap 15..17.
    for idx in range(15, 18):
        gap = frame(idx * 0.1)
        gap.poses.clear()
        frames.append(gap)
    # Frozen promotional poster / title slide 18..29 (>= 10 identical static frames).
    static_base = frame(1.8).poses[0]
    static_pose = {name: replace(point, x=point.x + 0.25) for name, point in static_base.items()}
    for idx in range(18, 30):
        poster = frame(idx * 0.1)
        poster.poses[0] = dict(static_pose)
        frames.append(poster)

    ev = Evidence(frames, MANIFEST["sha256"], {}, {}, 0.5)
    assert is_static_pose_run(frames, 0) is False
    assert is_static_pose_run(frames, 18) is True
    assert single_dancer_anchors(ev, 0.7) == {0: 0}


def test_portrait_video_measures_head_and_trunk_across_full_clip():
    from tango_posture.domain import FrameObservation

    width, height = 608, 1080

    def portrait_frame(ts: float) -> FrameObservation:
        # 115 px torso and 30 px ear-shoulder segment: above min(608, 1080) floors
        # (73.0 px and 24.3 px) while below naive 1080 px floors (129.6 px and 43.2 px).
        coords = {
            "nose": (304, 300),
            "left_ear": (304, 330),
            "right_ear": (314, 330),
            "left_shoulder": (304, 360),
            "right_shoulder": (314, 360),
            "left_hip": (304, 475),
            "right_hip": (314, 475),
            "left_ankle": (304, 700),
            "right_ankle": (314, 700),
        }
        return FrameObservation(
            ts,
            width,
            height,
            [{k: Landmark(x / width, y / height, 0.99, 0.99) for k, (x, y) in coords.items()}],
        )

    cfg = Config()
    obs0 = portrait_frame(0.0)
    obs1 = portrait_frame(0.1)
    measured = measure_ranges(obs0, cfg)
    for key in ("head_forward", "head_inclination", "trunk_inclination"):
        assert measured.quality[key].valid
        assert measured.values[key] is not None

    side_samples = metric_samples([obs0, obs1], cfg, view="side")
    all_samples = metric_samples([obs0, obs1], cfg, view="all")
    for key in ("head_forward", "head_inclination", "trunk_inclination"):
        assert all(val is not None for _, val in side_samples[key])
        assert all(val is not None for _, val in all_samples[key])


def test_save_range_profile_embeds_and_extracts_evidence_tpr(tmp_path):
    import cv2
    import numpy as np

    from tango_posture.archive import append_segment
    from tango_posture.calibration.archives import (
        EMBEDDED_EVIDENCE_MEMBER,
        extract_embedded_evidence,
        has_embedded_evidence,
    )
    from tango_posture.calibration.evidence import cached_evidence
    from tango_posture.references.archives import promote_reference

    video = tmp_path / "src.mp4"
    report = tmp_path / "src.md"
    practice = tmp_path / "src.tp"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    entries = []
    for idx in range(3):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        pose = frame(idx / 10.0).poses[0]
        entries.append({"poses": [_serialise_pose(pose)]})
    writer.release()
    report.write_text("# Evidence\n", encoding="utf-8")
    append_segment(
        practice,
        report,
        video,
        None,
        [idx / 10.0 for idx in range(3)],
        analytics={
            "model_hash": MANIFEST["sha256"],
            "capture_profile": profile(Config(), 160, 120, "video"),
            "frames": entries,
            "max_gap": 0.5,
        },
    )
    tpr_path = tmp_path / "src.tpr"
    promote_reference(practice, tpr_path, "Source Evidence")

    tpc_path = tmp_path / "Embedded.tpc"
    original = range_profile()
    save_range_profile(tpc_path, original, embedded_evidence=tpr_path)

    assert read_manifest(tpc_path)["segments"] == []
    assert read_range_profile(tpc_path) == original
    assert has_embedded_evidence(tpc_path)
    with ZipFile(tpc_path) as archive:
        assert EMBEDDED_EVIDENCE_MEMBER in archive.namelist()

    extracted = extract_embedded_evidence(tpc_path, tmp_path / "extracted")
    assert extracted is not None and extracted.is_file()
    ev = cached_evidence(extracted)
    assert len(ev.frames) == 3


