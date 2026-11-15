"""Unit tests for Review per-metric measurement normalization."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pytest

from tango_posture.archive import rename_archive
from tango_posture.calibration.archives import read_range_profile, save_range_profile
from tango_posture.calibration.models import ApprovedRange, Passage, RangeProfile
from tango_posture.calibration.normalization import (
    HoldSourceIdentity,
    MeasurementNormalizationConfig,
    MetricNormalizationRule,
    NormalizationPairRule,
    apply_metric_offset,
    build_source_identity,
    evaluate_matched_holds,
    evaluate_pair_applicability,
    extract_coach_observations,
    frames_fingerprint,
    in_physical_domain,
    make_pair_rule_id,
    metric_offset_between,
    profile_content_digest,
    resolve_raw_media_sha256,
)
from tango_posture.config import Config, export_toml, load
from tango_posture.domain import FrameObservation, Landmark
from tango_posture.references.coach import compare_coach_data


def _lm(x: float, y: float, vis: float = 0.95, pres: float = 0.95) -> Landmark:
    return Landmark(x=x, y=y, visibility=vis, presence=pres, z=0.0)


def _pose_frame(
    timestamp: float,
    *,
    head_pitch_deg: float = 3.0,
    head_inclination_deg: float = 0.0,
    knee_bend_deg: float = 15.0,
    extra_clearance: float = 0.0,
    valid: bool = True,
    width: int = 1000,
    height: int = 1000,
) -> FrameObservation:
    if not valid:
        return FrameObservation(timestamp, width, height, [])

    vis = 0.95
    # Torso aligned vertically: shoulder (0.5, 0.5), hip (0.5, 0.75) -> length = 250px
    ls = _lm(0.50, 0.50, vis, vis)
    rs = _lm(0.50, 0.50, vis, vis)
    lh = _lm(0.50, 0.75, vis, vis)
    rh = _lm(0.50, 0.75, vis, vis)

    # Place ear relative to shoulder at distance d=0.08 to set head_inclination cleanly.
    inc_rad = math.radians(head_inclination_deg)
    d_ear = 0.08
    ear_x = 0.50 + d_ear * math.sin(inc_rad)
    ear_y = 0.50 - d_ear * math.cos(inc_rad) - extra_clearance
    le = _lm(ear_x, ear_y, vis, vis)

    # Set nose relative to ear for exact head_pitch_deg: atan2((ear.y - nose.y)*H, |nose.x - ear.x|*W).
    pitch_rad = math.radians(head_pitch_deg)
    dx_norm = 0.10
    dy_norm = dx_norm * math.tan(pitch_rad)
    nose = _lm(ear_x + dx_norm, ear_y - dy_norm, vis, vis)

    # Straight thigh hip->knee vertical, shin deflected by knee_bend_deg so interior = 180 - bend.
    bend_rad = math.radians(max(0.0, knee_bend_deg))
    lk = _lm(0.50, 0.85, vis, vis)
    rk = _lm(0.50, 0.85, vis, vis)
    shin_len = 0.10
    la_x = 0.50 - shin_len * math.sin(bend_rad)
    la_y = 0.85 + shin_len * math.cos(bend_rad)
    la = _lm(la_x, la_y, vis, vis)
    ra = _lm(la_x, la_y, vis, vis)

    pose = {
        "nose": nose,
        "left_ear": le,
        "left_shoulder": ls,
        "right_shoulder": rs,
        "left_hip": lh,
        "right_hip": rh,
        "left_knee": lk,
        "right_knee": rk,
        "left_ankle": la,
        "right_ankle": ra,
    }
    return FrameObservation(timestamp, width, height, [pose])


def _serialise_pose(pose: dict[str, Landmark]) -> dict[str, list[float | None]]:
    return {
        name: [lm.x, lm.y, lm.z, lm.visibility, lm.presence]
        for name, lm in pose.items()
    }


def _coach_data_from_frames(
    frames: list[FrameObservation],
    *,
    model_hash: str = "model-1",
    raw_media_sha256: str | None = "video-sha-1",
    interpolated_indices: set[int] | None = None,
) -> dict:
    interpolated_indices = interpolated_indices or set()
    entries = {}
    for idx, frame in enumerate(frames):
        entry: dict = {
            "timestamp": frame.timestamp,
            "measurement_poses": [_serialise_pose(p) for p in frame.poses],
        }
        if idx in interpolated_indices:
            entry["interpolated"] = True
        entries[str(idx)] = entry
    payload = {
        "model_hash": model_hash,
        "capture_profile": {"side": "left", "facing": "right", "width": 1000, "height": 1000},
        "width": 1000,
        "height": 1000,
        "frames": entries,
    }
    if raw_media_sha256 is not None:
        payload["source_provenance"] = {"raw_media_sha256": raw_media_sha256}
    return payload


def _range(lower: float, upper: float, unit: str = "degrees") -> ApprovedRange:
    mid = 0.5 * (lower + upper)
    return ApprovedRange(
        lower=lower,
        upper=upper,
        unit=unit,
        view="side",
        passages=[Passage(start=0.0, end=2.0)],
        median=mid,
        p10=lower,
        p90=upper,
        valid_duration=2.0,
        selected_duration=2.0,
        samples=10,
        inside_fraction=1.0,
        approved=True,
        camera_check_passed=True,
    )


def _profile(metrics: dict[str, ApprovedRange], cal_id: str = "cal-pitch") -> RangeProfile:
    return RangeProfile(
        id=cal_id,
        name="Teacher Neutral",
        created_at_utc="2026-11-15T12:00:00Z",
        context="open_embrace",
        model_hash="model-1",
        capture_profile={"side": "left", "facing": "right"},
        source={"archive": "ref.tpr", "archive_sha256": "abc"},
        metrics=metrics,
        camera_checks={},
    )


def test_worked_example_head_pitch_matched_holds_and_review() -> None:
    """Worked example: ref fit 3°, prac fit 7° -> -4°; check 3.2°/7.4° -> error 0.2°; 8°->4°, 11°->7°."""
    cfg = Config(smoothing=0.01, entry_duration=0.0, clear_duration=0.0)
    profile = _profile({"head_pitch": _range(1.0, 5.0)})

    # 10 frames per 1s hold (0..1s Fit, 2..3s Check).
    ref_frames = [
        *[_pose_frame(0.1 * i, head_pitch_deg=3.0) for i in range(10)],
        *[_pose_frame(2.0 + 0.1 * i, head_pitch_deg=3.2) for i in range(10)],
    ]
    prac_hold_frames = [
        *[_pose_frame(0.1 * i, head_pitch_deg=7.0) for i in range(10)],
        *[_pose_frame(2.0 + 0.1 * i, head_pitch_deg=7.4) for i in range(10)],
        _pose_frame(4.0, head_pitch_deg=8.0),
        _pose_frame(5.0, head_pitch_deg=11.0),
    ]

    draft_rule = MetricNormalizationRule(
        enabled=True,
        method="matched_holds",
        fit_reference_window=Passage(start=0.0, end=1.0),
        fit_practice_window=Passage(start=0.0, end=1.0),
        check_reference_window=Passage(start=2.0, end=3.0),
        check_practice_window=Passage(start=2.0, end=3.0),
    )
    matched = evaluate_matched_holds(
        metric="head_pitch",
        target=profile.metrics["head_pitch"],
        rule=draft_rule,
        reference_frames=ref_frames,
        practice_frames=prac_hold_frames,
        cfg=cfg,
    )
    assert matched.eligible is True
    assert matched.rule.verified is True
    assert matched.rule.offset == pytest.approx(-4.0, abs=1e-4)
    assert matched.rule.verification_error == pytest.approx(0.2, abs=1e-4)

    prac_data = _coach_data_from_frames(prac_hold_frames, raw_media_sha256="prac-sha-1")
    prac_id = HoldSourceIdentity(
        raw_media_sha256="prac-sha-1",
        capture_identity={"side": "left", "facing": "right", "width": 1000, "height": 1000},
        settings_fingerprint=build_source_identity(
            archive_path=None,
            segment_index=0,
            cfg=cfg,
            observations=prac_hold_frames,
            capture_profile=prac_data["capture_profile"],
            raw_media_sha256="prac-sha-1",
        ).settings_fingerprint,
        frames_fingerprint=frames_fingerprint(extract_coach_observations(prac_data)),
    )
    pair = NormalizationPairRule(
        id=make_pair_rule_id(profile.id, "prac-sha-1"),
        name="Head Pitch Pair",
        enabled=True,
        calibration_id=profile.id,
        calibration_name=profile.name,
        profile_content_digest=profile_content_digest(profile),
        practice_source=prac_id,
        metrics={"head_pitch": matched.rule},
    )
    norm_cfg = cfg.model_copy(update={
        "measurement_normalization": MeasurementNormalizationConfig(
            enabled=True,
            pairs=[pair],
        )
    })

    compared = compare_coach_data(prac_data, norm_cfg, profile.calibration())
    assert compared["normalization"]["status"] == "applied"

    row_8 = compared["frames"]["20"]["calibration_rows"][0]
    assert row_8["observed_raw"] == pytest.approx(8.0, abs=1e-3)
    assert row_8["normalization_offset"] == pytest.approx(-4.0, abs=1e-4)
    assert row_8["reference_scale_raw"] == pytest.approx(4.0, abs=1e-3)
    assert row_8["raw"] == pytest.approx(4.0, abs=1e-3)
    assert row_8["state"] == "WITHIN_REFERENCE"
    assert row_8["normalization_status"] == "applied"

    row_11 = compared["frames"]["21"]["calibration_rows"][0]
    assert row_11["observed_raw"] == pytest.approx(11.0, abs=1e-3)
    assert row_11["reference_scale_raw"] == pytest.approx(7.0, abs=1e-3)
    assert row_11["state"] == "DEVIATION"

    # Bounds on .tpc remain untouched on original reference scale; disabling gives DEVIATION on 8°.
    baseline = compare_coach_data(prac_data, cfg, profile.calibration())
    assert baseline["frames"]["20"]["calibration_rows"][0]["state"] == "DEVIATION"
    assert profile.metrics["head_pitch"].lower == 1.0
    assert profile.metrics["head_pitch"].upper == 5.0


def test_matched_holds_failure_gates_block_normalization() -> None:
    """Missing check, overlapping fit/check, low samples, low coverage, or large check error fail."""
    cfg = Config(smoothing=0.01, entry_duration=0.0, clear_duration=0.0)
    profile = _profile({"head_pitch": _range(1.0, 5.0)})
    ref_frames = [
        *[_pose_frame(0.1 * i, head_pitch_deg=3.0) for i in range(10)],
        *[_pose_frame(2.0 + 0.1 * i, head_pitch_deg=3.2) for i in range(10)],
    ]
    prac_bad_check = [
        *[_pose_frame(0.1 * i, head_pitch_deg=7.0) for i in range(10)],
        *[_pose_frame(2.0 + 0.1 * i, head_pitch_deg=10.0) for i in range(10)],
    ]

    # 1. Missing check window
    missing = evaluate_matched_holds(
        metric="head_pitch",
        target=profile.metrics["head_pitch"],
        rule=MetricNormalizationRule(
            enabled=True,
            method="matched_holds",
            fit_reference_window=Passage(0.0, 1.0),
            fit_practice_window=Passage(0.0, 1.0),
        ),
        reference_frames=ref_frames,
        practice_frames=prac_bad_check,
        cfg=cfg,
    )
    assert not missing.eligible and not missing.rule.verified

    # 2. Overlapping fit and check windows
    overlap = evaluate_matched_holds(
        metric="head_pitch",
        target=profile.metrics["head_pitch"],
        rule=MetricNormalizationRule(
            enabled=True,
            method="matched_holds",
            fit_reference_window=Passage(0.0, 1.0),
            fit_practice_window=Passage(0.0, 1.5),
            check_reference_window=Passage(2.0, 3.0),
            check_practice_window=Passage(1.0, 2.5),
        ),
        reference_frames=ref_frames,
        practice_frames=prac_bad_check,
        cfg=cfg,
    )
    assert not overlap.eligible and not overlap.rule.verified

    # 3. Sample count < min_hold_samples
    short_prac = [_pose_frame(0.1 * i, head_pitch_deg=7.0) for i in range(5)] + [
        _pose_frame(2.0 + 0.1 * i, head_pitch_deg=7.4) for i in range(10)
    ]
    too_few = evaluate_matched_holds(
        metric="head_pitch",
        target=profile.metrics["head_pitch"],
        rule=MetricNormalizationRule(
            enabled=True,
            method="matched_holds",
            fit_reference_window=Passage(0.0, 1.0),
            fit_practice_window=Passage(0.0, 1.0),
            check_reference_window=Passage(2.0, 3.0),
            check_practice_window=Passage(2.0, 3.0),
        ),
        reference_frames=ref_frames,
        practice_frames=short_prac,
        cfg=cfg,
    )
    assert not too_few.eligible and not too_few.rule.verified

    # 4. Low coverage (< 0.6): 10 timestamps in fit hold, 7 invalid
    gapped_prac = [
        _pose_frame(0.1 * i, head_pitch_deg=7.0, valid=(i < 3)) for i in range(10)
    ] + [_pose_frame(2.0 + 0.1 * i, head_pitch_deg=7.4) for i in range(10)]
    low_cov_cfg = cfg.model_copy(update={
        "measurement_normalization": MeasurementNormalizationConfig(enabled=True, min_hold_samples=2, min_hold_coverage=0.6)
    })
    low_cov = evaluate_matched_holds(
        metric="head_pitch",
        target=profile.metrics["head_pitch"],
        rule=MetricNormalizationRule(
            enabled=True,
            method="matched_holds",
            fit_reference_window=Passage(0.0, 1.0),
            fit_practice_window=Passage(0.0, 1.0),
            check_reference_window=Passage(2.0, 3.0),
            check_practice_window=Passage(2.0, 3.0),
        ),
        reference_frames=ref_frames,
        practice_frames=gapped_prac,
        cfg=low_cov_cfg,
    )
    assert not low_cov.eligible and not low_cov.rule.verified

    # 5. Check error > max_verification_error (6.0 vs 3.2 -> error 2.8 > 1.0)
    over_tol = evaluate_matched_holds(
        metric="head_pitch",
        target=profile.metrics["head_pitch"],
        rule=MetricNormalizationRule(
            enabled=True,
            method="matched_holds",
            fit_reference_window=Passage(0.0, 1.0),
            fit_practice_window=Passage(0.0, 1.0),
            check_reference_window=Passage(2.0, 3.0),
            check_practice_window=Passage(2.0, 3.0),
        ),
        reference_frames=ref_frames,
        practice_frames=prac_bad_check,
        cfg=cfg,
    )
    assert not over_tol.eligible and not over_tol.rule.verified
    assert over_tol.rule.verification_error == pytest.approx(2.8, abs=1e-4)

    # Attempting to use failed matched rule in compare_coach_data triggers needs_recheck, not applied
    data = _coach_data_from_frames(prac_bad_check, raw_media_sha256="bad-check")
    pair = NormalizationPairRule(
        id="pair-bad-check",
        calibration_id=profile.id,
        profile_content_digest=profile_content_digest(profile),
        practice_source=HoldSourceIdentity(
            raw_media_sha256="bad-check",
            settings_fingerprint=build_source_identity(
                archive_path=None,
                segment_index=0,
                cfg=cfg,
                observations=prac_bad_check,
                capture_profile=data["capture_profile"],
                raw_media_sha256="bad-check",
            ).settings_fingerprint,
            frames_fingerprint=frames_fingerprint(extract_coach_observations(data)),
        ),
        metrics={"head_pitch": over_tol.rule},
    )
    bad_cfg = cfg.model_copy(update={
        "measurement_normalization": MeasurementNormalizationConfig(enabled=True, pairs=[pair])
    })
    compared = compare_coach_data(data, bad_cfg, profile.calibration())
    assert compared["normalization"]["status"] == "needs_recheck"
    assert compared["frames"]["0"]["calibration_rows"][0]["raw"] == pytest.approx(7.0, abs=1e-3)


def test_invalid_or_interpolated_frames_remain_unknown() -> None:
    """Missing landmarks or interpolated-only frames stay UNKNOWN rather than using offset."""
    cfg = Config(smoothing=0.01, entry_duration=0.0, clear_duration=0.0)
    profile = _profile({"head_pitch": _range(1.0, 5.0)})
    frames = [
        _pose_frame(0.0, head_pitch_deg=8.0, valid=True),
        _pose_frame(0.5, valid=False),
        _pose_frame(1.0, head_pitch_deg=8.0, valid=True),
    ]
    data = _coach_data_from_frames(frames, raw_media_sha256="interp-sha", interpolated_indices={2})
    pair = NormalizationPairRule(
        id="pair-interp",
        calibration_id=profile.id,
        profile_content_digest=profile_content_digest(profile),
        practice_source=HoldSourceIdentity(
            raw_media_sha256="interp-sha",
            settings_fingerprint=build_source_identity(
                archive_path=None,
                segment_index=0,
                cfg=cfg,
                observations=extract_coach_observations(data),
                capture_profile=data["capture_profile"],
                raw_media_sha256="interp-sha",
            ).settings_fingerprint,
            frames_fingerprint=frames_fingerprint(extract_coach_observations(data)),
        ),
        metrics={"head_pitch": MetricNormalizationRule(enabled=True, method="manual", offset=-4.0)},
    )
    norm_cfg = cfg.model_copy(update={
        "measurement_normalization": MeasurementNormalizationConfig(enabled=True, pairs=[pair])
    })
    compared = compare_coach_data(data, norm_cfg, profile.calibration())
    assert compared["frames"]["0"]["calibration_rows"][0]["state"] == "WITHIN_REFERENCE"
    for bad_idx in ("1", "2"):
        row = compared["frames"][bad_idx]["calibration_rows"][0]
        assert row["state"] == "UNKNOWN"
        assert row["observed_raw"] is None
        assert row["reference_scale_raw"] is None


def test_independent_head_and_shoulder_rules() -> None:
    """One metric can be normalized while another metric stays on Direct comparison."""
    cfg = Config(smoothing=0.01, entry_duration=0.0, clear_duration=0.0)
    profile = _profile({
        "head_pitch": _range(1.0, 5.0),
        "shoulder_clearance": _range(1.0, 2.0, unit="normalized_clearance"),
    })
    frames = [_pose_frame(0.0, head_pitch_deg=8.0)]
    data = _coach_data_from_frames(frames, raw_media_sha256="indep-sha")
    pair = NormalizationPairRule(
        id="pair-indep",
        calibration_id=profile.id,
        profile_content_digest=profile_content_digest(profile),
        practice_source=HoldSourceIdentity(
            raw_media_sha256="indep-sha",
            settings_fingerprint=build_source_identity(
                archive_path=None,
                segment_index=0,
                cfg=cfg,
                observations=frames,
                capture_profile=data["capture_profile"],
                raw_media_sha256="indep-sha",
            ).settings_fingerprint,
            frames_fingerprint=frames_fingerprint(extract_coach_observations(data)),
        ),
        metrics={
            "head_pitch": MetricNormalizationRule(enabled=True, method="manual", offset=-4.0),
            "shoulder_clearance": MetricNormalizationRule(enabled=False, method="manual", offset=9.0),
        },
    )
    norm_cfg = cfg.model_copy(update={
        "measurement_normalization": MeasurementNormalizationConfig(enabled=True, pairs=[pair])
    })
    compared = compare_coach_data(data, norm_cfg, profile.calibration())
    rows = {r["metric"]: r for r in compared["frames"]["0"]["calibration_rows"]}
    assert rows["head_pitch"]["normalization_status"] == "applied"
    assert rows["head_pitch"]["raw"] == pytest.approx(4.0, abs=1e-3)
    assert rows["shoulder_clearance"]["normalization_status"] == "direct_comparison"
    assert rows["shoulder_clearance"]["raw"] == pytest.approx(rows["shoulder_clearance"]["observed_raw"], abs=1e-6)


def test_cyclic_wrapping_and_physical_domain_rejection() -> None:
    """Cyclic inclination wraps across +-180°, while negative knee/shoulder offsets are rejected."""
    offset = metric_offset_between("head_inclination", 179.0, -178.0)
    assert offset == pytest.approx(-3.0, abs=1e-6)
    assert apply_metric_offset("head_inclination", -179.0, offset) == pytest.approx(178.0, abs=1e-6)
    assert apply_metric_offset("trunk_inclination", 178.0, 5.0) == pytest.approx(-177.0, abs=1e-6)

    assert not in_physical_domain("knee_flexion", -0.5)
    assert not in_physical_domain("knee_flexion", 181.0)
    assert in_physical_domain("knee_flexion", 12.0)
    assert not in_physical_domain("shoulder_clearance", -0.01)

    # Frame comparison rejects out-of-domain negative knee_flexion and leaves Evaluator unadjusted
    cfg = Config(smoothing=0.01, entry_duration=0.0, clear_duration=0.0)
    profile = _profile({
        "knee_flexion": _range(10.0, 25.0),
        "shoulder_clearance": _range(1.0, 2.0, unit="normalized_clearance"),
    })
    frames = [_pose_frame(0.0, knee_bend_deg=5.0)]
    data = _coach_data_from_frames(frames, raw_media_sha256="domain-sha")
    pair = NormalizationPairRule(
        id="pair-domain",
        calibration_id=profile.id,
        profile_content_digest=profile_content_digest(profile),
        practice_source=HoldSourceIdentity(
            raw_media_sha256="domain-sha",
            settings_fingerprint=build_source_identity(
                archive_path=None,
                segment_index=0,
                cfg=cfg,
                observations=frames,
                capture_profile=data["capture_profile"],
                raw_media_sha256="domain-sha",
            ).settings_fingerprint,
            frames_fingerprint=frames_fingerprint(extract_coach_observations(data)),
        ),
        metrics={
            "knee_flexion": MetricNormalizationRule(enabled=True, method="manual", offset=-10.0),
            "shoulder_clearance": MetricNormalizationRule(enabled=True, method="manual", offset=-5.0),
        },
    )
    norm_cfg = cfg.model_copy(update={
        "measurement_normalization": MeasurementNormalizationConfig(enabled=True, pairs=[pair])
    })
    compared = compare_coach_data(data, norm_cfg, profile.calibration())
    rows = {r["metric"]: r for r in compared["frames"]["0"]["calibration_rows"]}
    assert rows["knee_flexion"]["normalization_status"] == "unavailable"
    assert rows["knee_flexion"]["reference_scale_raw"] is None
    assert rows["knee_flexion"]["raw"] == pytest.approx(5.0, abs=0.1)
    assert rows["shoulder_clearance"]["normalization_status"] == "unavailable"


def test_identity_rename_vs_reanalysis_settings_and_toml_roundtrip(tmp_path: Path) -> None:
    """Archive rename preserves normalization; reanalysis, settings edits, and bound changes invalidate it."""
    cfg = Config()
    profile = _profile({"head_pitch": _range(1.0, 5.0)})
    tpc_path = tmp_path / "teacher.tpc"
    save_range_profile(tpc_path, profile)
    loaded_profile = read_range_profile(tpc_path)

    frames = [_pose_frame(0.0, head_pitch_deg=8.0)]
    data = _coach_data_from_frames(frames, raw_media_sha256=None)

    # Create a minimal valid .tp archive containing a video member + analytics_00.json
    tp_path = tmp_path / "practice_one.tp"
    video_bytes = b"fake-mp4-stream-123456"
    with ZipFile(tp_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({
            "version": 1,
            "kind": "practice",
            "name": "Practice One",
            "created_at_utc": "2026-11-15T12:00:00Z",
            "segments": [{
                "index": 0,
                "video": "video_00.mp4",
                "overlay": None,
                "report": "report_00.md",
                "metadata": "report_00.metadata.json",
                "lines": None,
                "analytics": "analytics_00.json",
            }],
        }))
        zf.writestr("video_00.mp4", video_bytes)
        zf.writestr("report_00.md", "# Report\n")
        zf.writestr("report_00.metadata.json", json.dumps({"schema_version": 1}))
        zf.writestr("analytics_00.json", json.dumps({"coach_data": data}))

    media_sha = resolve_raw_media_sha256(tp_path, 0)
    assert media_sha is not None
    prac_id = build_source_identity(
        archive_path=tp_path,
        segment_index=0,
        cfg=cfg,
        observations=frames,
        capture_profile=data["capture_profile"],
        raw_media_sha256=media_sha,
    )
    pair = NormalizationPairRule(
        id=make_pair_rule_id(loaded_profile.id, media_sha),
        name="Session 1",
        enabled=True,
        calibration_id=loaded_profile.id,
        calibration_name=loaded_profile.name,
        profile_content_digest=profile_content_digest(loaded_profile),
        practice_source=prac_id,
        metrics={"head_pitch": MetricNormalizationRule(enabled=True, method="manual", offset=-4.0)},
    )
    norm_cfg = cfg.model_copy(update={
        "measurement_normalization": MeasurementNormalizationConfig(enabled=True, pairs=[pair])
    })

    # Rename .tp archive -> media sha and frame fingerprints stay identical -> still applies
    renamed_tp = rename_archive(tp_path, "Renamed Session")
    renamed_sha = resolve_raw_media_sha256(renamed_tp, 0)
    renamed_id = build_source_identity(
        archive_path=renamed_tp,
        segment_index=0,
        cfg=norm_cfg,
        observations=frames,
        capture_profile=data["capture_profile"],
        raw_media_sha256=renamed_sha,
    )
    app_ok = evaluate_pair_applicability(norm_cfg, loaded_profile, renamed_id)
    assert app_ok.eligible is True
    assert app_ok.status == "applied"

    # Changed landmarks (reanalysis) -> needs_recheck
    reanalyzed_frames = [_pose_frame(0.0, head_pitch_deg=9.0)]
    reanalyzed_id = build_source_identity(
        archive_path=renamed_tp,
        segment_index=0,
        cfg=norm_cfg,
        observations=reanalyzed_frames,
        capture_profile=data["capture_profile"],
        raw_media_sha256=renamed_sha,
    )
    app_reanalyzed = evaluate_pair_applicability(norm_cfg, loaded_profile, reanalyzed_id)
    assert app_reanalyzed.status == "needs_recheck"

    # Changed measurement setting -> needs_recheck
    tweaked_cfg = norm_cfg.model_copy(update={"confidence": 0.85})
    app_settings = evaluate_pair_applicability(tweaked_cfg, loaded_profile, renamed_id)
    assert app_settings.status == "needs_recheck"

    # Changed calibration bounds -> needs_recheck
    edited_profile = loaded_profile.model_copy(update={"metrics": {"head_pitch": _range(0.0, 4.0)}})
    app_profile = evaluate_pair_applicability(norm_cfg, edited_profile, renamed_id)
    assert app_profile.status == "needs_recheck"

    # Unrelated recording -> no_pair
    other_frames = [_pose_frame(0.0, head_pitch_deg=6.0)]
    other_id = build_source_identity(
        archive_path=None,
        segment_index=0,
        cfg=norm_cfg,
        observations=other_frames,
        capture_profile=data["capture_profile"],
        raw_media_sha256="completely-different-media",
    )
    assert evaluate_pair_applicability(norm_cfg, loaded_profile, other_id).status == "no_pair"

    # TOML export / reload round-trip without None / null tokens
    toml_text = export_toml(norm_cfg)
    assert "null" not in toml_text
    parsed = tomllib.loads(toml_text)
    reloaded = Config.model_validate(parsed)
    assert reloaded.measurement_normalization == norm_cfg.measurement_normalization

    toml_file = tmp_path / "settings.toml"
    toml_file.write_text(toml_text, encoding="utf-8")
    loaded_from_disk = load(toml_file)
    assert loaded_from_disk.measurement_normalization == norm_cfg.measurement_normalization
