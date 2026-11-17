import tomllib
from pathlib import Path

import pytest

from tango_posture.config import (
    SETTING_GROUPS,
    SETTING_MAP,
    AnalysisRegion,
    Config,
    DancerAnalysisConfig,
    export_toml,
    load,
)


def test_every_setting_has_a_toml_name_and_editor_group():
    expected = (set(Config.model_fields) - {
        "head_forward",
        "head_inclination",
        "trunk_inclination",
        "axis_curve",
        "calibration_defaults",
        "measurement_normalization",
        "dancer_analysis",
        "replay_audio",
        "replay_auto_analyse_movement",
        "view_change_tolerance",
        "calibration_duration",
        "calibration_valid_duration",
        "calibration_samples",
        "offset_mad",
        "angle_mad",
        "camera",
        "width",
        "height",
        "mirror",
        "beat_tracking",
        "beat_analysis_seconds",
        "auto_record_start_delay",
        "auto_record_stop_delay",
        "max_recordings",
    }) | {f"dancer_analysis.{name}" for name in DancerAnalysisConfig.model_fields}
    fields = [field for group in SETTING_GROUPS.values() for field in group]
    assert len(fields) == len(set(fields))
    assert set(fields) == expected == set(SETTING_MAP)
    assert SETTING_MAP["head_down_angle"][0] == "head_down_angle"


def test_default_toml_has_every_editable_setting():
    path = Path(__file__).resolve().parents[2] / "configs" / "default.toml"
    data = tomllib.loads(path.read_text())
    fields = {
        f"{key}.{name}" if isinstance(value, dict) else key
        for key, value in data.items()
        for name in (value if isinstance(value, dict) else [None])
    }
    assert fields == set(SETTING_MAP)
    Config.model_validate(data)


def test_full_toml_round_trip(tmp_path):
    cfg = Config(
        model_path='models/with spaces "quoted".task',
        data_dir="my sessions",
        axis_curve_entry=0.1,
        analysis_model="full",
        replay_joint_interpolation="quadratic",
        replay_joint_max_gap=2.75,
        calibration_defaults={
            "head_pitch": {"lower": -23.125, "upper": 7.5, "unit": "degrees", "view": "all"},
            "head_forward": {
                "lower": -0.03, "upper": 0.2, "unit": "torso_lengths", "view": "side",
            },
            "hip_offset": {"lower": -0.02, "upper": 0.11, "unit": "axis_lengths", "view": "side"},
        },
    )
    assert Config.model_validate(tomllib.loads(export_toml(cfg))) == cfg


def test_old_inactive_replay_flags_load_but_are_not_exported(tmp_path):
    settings = tmp_path / "old-settings.toml"
    settings.write_text(
        'replay_audio=true\nreplay_auto_analyse_movement=false\nanalysis_model="full"\n'
    )
    cfg = load(settings, {})
    assert cfg.replay_audio
    assert not cfg.replay_auto_analyse_movement
    exported = tomllib.loads(export_toml(cfg))
    assert "replay_audio" not in exported
    assert "replay_auto_analyse_movement" not in exported
    assert Config.model_validate(exported).analysis_model == "full"


def test_old_baseline_settings_load_without_returning_to_settings_exports(tmp_path):
    legacy = {
        "view_change_tolerance": 0.22,
        "calibration_duration": 15.0,
        "calibration_valid_duration": 12.0,
        "calibration_samples": 80,
        "offset_mad": 0.04,
        "angle_mad": 5.0,
    }
    settings = tmp_path / "legacy-settings.toml"
    settings.write_text(
        "".join(f"{key}={value}\n" for key, value in legacy.items()) + "confidence=0.8\n"
    )
    cfg = load(settings, {})
    assert {key: getattr(cfg, key) for key in legacy} == legacy
    exported = tomllib.loads(export_toml(cfg))
    assert legacy.keys().isdisjoint(exported)
    assert Config.model_validate(exported).confidence == 0.8


def test_removed_live_controls_remain_importable_without_being_exported(tmp_path):
    legacy = {
        "camera": 3, "width": 1280, "height": 720, "mirror": True,
        "beat_tracking": False, "beat_analysis_seconds": 60.0,
        "auto_record_start_delay": 2.0, "auto_record_stop_delay": 5.0, "max_recordings": 4,
    }
    settings = tmp_path / "live-settings.toml"
    settings.write_text(
        "".join(f"{key}={str(value).lower()}\n" for key, value in legacy.items())
        + 'analysis_model="full"\n'
    )
    cfg = load(settings, {})
    assert cfg.model_dump(include=set(legacy)) == legacy
    exported = tomllib.loads(export_toml(cfg))
    assert legacy.keys().isdisjoint(exported)
    assert Config.model_validate(exported).analysis_model == "full"


@pytest.mark.parametrize("metric,entry,clear", [
    ("head_forward", 0.2, 0.1),
    ("head_inclination", 14.0, 7.0),
    ("trunk_inclination", 12.0, 6.0),
])
def test_old_metric_thresholds_load_without_replacing_current_range_settings(
    tmp_path, metric, entry, clear,
):
    settings = tmp_path / "legacy-thresholds.toml"
    settings.write_text(
        f"entry_duration=2.0\n[{metric}]\nentry={entry}\nclear={clear}\n"
        "[calibration_defaults.head_forward]\n"
        'lower=-0.03\nupper=0.2\nunit="torso_lengths"\nview="side"\n'
    )
    cfg = load(settings, {})
    assert getattr(cfg, metric).model_dump() == {"entry": entry, "clear": clear}
    exported = tomllib.loads(export_toml(cfg))
    assert metric not in exported
    restored = Config.model_validate(exported)
    assert restored.entry_duration == 2.0
    assert restored.calibration_defaults == cfg.calibration_defaults


@pytest.mark.parametrize("change", [
    {"analysis_model": "large"},
    {"replay_joint_interpolation": "cubic"}, {"replay_joint_max_gap": -1},
    {"calibration_defaults": {
        "head_pitch": {"lower": 2, "upper": 1, "unit": "degrees", "view": "all"},
    }},
    {"calibration_defaults": {
        "shoulder_clearance": {"lower": 0, "upper": 1, "unit": "degrees", "view": "all"},
    }},
])
def test_invalid_interpolation_and_default_ranges_are_rejected(change):
    with pytest.raises(ValueError):
        Config.model_validate(change)


def test_environment_is_ignored_and_toml_cli_precedence_is_preserved(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.chdir(tmp_path)
    env.write_text("HEAD_DOWN_ANGLE_DEGREES=-19\ntp_HEAD_DOWN_ANGLE_DEGREES=-21\n")
    monkeypatch.setenv("HEAD_DOWN_ANGLE_DEGREES", "-99")
    monkeypatch.setenv("tp_HEAD_DOWN_ANGLE_DEGREES", "-21")
    assert load(None, {}).head_down_angle == -15
    settings = tmp_path / "settings.toml"
    settings.write_text("head_down_angle=-23\n")
    assert load(settings, {}).head_down_angle == -23
    assert load(settings, {"head_down_angle": -25}).head_down_angle == -25


@pytest.mark.parametrize("line", ['mirror="maybe"', 'camera="foo"', 'tango_bpm=nan', 'axis_curve_clear=0.8'])
def test_invalid_settings_are_rejected(tmp_path, line):
    settings = tmp_path / "settings.toml"
    settings.write_text(line)
    with pytest.raises(ValueError):
        load(settings, {})


@pytest.mark.parametrize("bad", [
    {"calibration_suggest_lower_pct": 90.0, "calibration_suggest_upper_pct": 80.0},
    {"calibration_min_usable_samples": 1},
    {"shoulder_adaptive_min_samples": 100, "shoulder_adaptive_max_samples": 50},
    {"shoulder_adaptive_min_clearance": 1.5, "shoulder_adaptive_max_clearance": 1.2},
    {"knee_visibility_floor": 1.2},
    {"shoulder_visibility_floor": -0.1},
    {"min_filter_time_constant": 0.0},
])
def test_invalid_algorithmic_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        Config.model_validate(bad)


def test_custom_algorithmic_settings_round_trip_through_toml():
    cfg = Config(
        knee_visibility_floor=0.45,
        knee_min_segment_px=12.0,
        knee_tiebreak_other_max_vis=0.55,
        knee_tiebreak_min_advantage=0.20,
        shoulder_visibility_floor=0.60,
        shoulder_ear_hip_scale_factor=0.30,
        shoulder_min_ear_hip_px=25.0,
        shoulder_nose_ear_scale_factor=0.72,
        shoulder_ear_ear_scale_factor=0.62,
        shoulder_width_scale_factor=0.31,
        shoulder_min_ref_scale_px=10.0,
        shoulder_baseline_clamp_floor=0.25,
        shoulder_adaptive_max_samples=400,
        shoulder_adaptive_min_samples=20,
        shoulder_adaptive_min_clearance=0.70,
        shoulder_adaptive_max_clearance=1.40,
        shoulder_adaptive_percentile=80.0,
        shoulder_recovery_min_pct=5.0,
        shoulder_recovery_ratio=0.70,
        knee_recovery_margin=3.0,
        min_filter_time_constant=0.005,
        calibration_suggest_lower_pct=15.0,
        calibration_suggest_upper_pct=85.0,
        calibration_min_usable_samples=5,
    )
    assert Config.model_validate(tomllib.loads(export_toml(cfg))) == cfg


def test_dancer_analysis_settings_round_trip_and_validation():
    cfg = Config(
        dancer_analysis=DancerAnalysisConfig(
            default_dancer_mode="couple",
            auto_detection_window_s=15.0,
            primary_coverage_min=0.70,
            axis_smoothing_s=0.40,
            overlay_smoothing_s=0.35,
            axis_bridge_max_gap_s=0.30,
            overlay_min_landmark_visibility=0.60,
            duplicate_pose_max_distance_torso_lengths=0.22,
            fixed_scale_min_ratio=0.50,
            fixed_scale_max_ratio=2.00,
            shot_cut_max_frame_gap_s=2.00,
            shot_cut_jump_torso_lengths=1.80,
            turn_exit_match_window_s=4.50,
            close_embrace_max_tilt_deg=30.0,
            analyze_soundtrack=True,
            auto_turn_suggestions=True,
        )
    )
    reloaded = Config.model_validate(tomllib.loads(export_toml(cfg)))
    assert reloaded.dancer_analysis == cfg.dancer_analysis

    with pytest.raises(ValueError):
        DancerAnalysisConfig(brief_second_max_s=1.5, second_sustained_min_s=1.5)
    with pytest.raises(ValueError):
        DancerAnalysisConfig(start_speed=0.10, stop_speed=0.15)
    with pytest.raises(ValueError):
        DancerAnalysisConfig(fixed_scale_min_ratio=0.9, fixed_scale_max_ratio=1.0)
    with pytest.raises(ValueError):
        AnalysisRegion(x_min=0.8, y_min=0.1, x_max=0.4, y_max=0.9)
