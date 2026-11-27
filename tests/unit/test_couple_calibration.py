from tango_posture.calibration.couple import compare_couple_series
from tango_posture.calibration.models import ApprovedRange, Passage
from tango_posture.config import SETTING_ENGINE_SCOPES, SETTING_GROUPS, Config, export_toml


def test_couple_temporal_recovery_gaps_and_independent_settings():
    import tomllib

    cfg = Config(entry_duration=99, clear_duration=99, smoothing=99)
    cfg.dancer_analysis.calibration_smoothing_s = 0.001
    cfg.dancer_analysis.calibration_entry_duration_s = 0.2
    cfg.dancer_analysis.calibration_clear_duration_s = 0.2
    target = ApprovedRange(
        lower=-1, upper=1, unit="degrees", view="all",
        passages=[Passage(start=0, end=2)], median=0, p10=-1, p90=1,
        valid_duration=2, selected_duration=2, samples=20, inside_fraction=1,
    )
    points = [(i / 10, 10 if i < 3 else 0) for i in range(6)] + [(0.6, None), (1.2, 10)]
    smoothed, states = compare_couple_series("torso_tilt_a", points, target, cfg)
    assert states[0] == "PENDING"
    assert states[0.2] == "DEVIATION"
    assert states[0.3] == "DEVIATION"
    assert states[0.5] == "WITHIN_REFERENCE"
    assert states[0.6] == "UNKNOWN"
    assert states[1.2] == "PENDING"
    assert smoothed[6][1] is None
    assert "Couple · Temporal / Smoothing" in SETTING_ENGINE_SCOPES["Couple Dancer Engine"]
    assert len(SETTING_GROUPS["Couple · Temporal / Smoothing"]) == 3
    restored = Config.model_validate(tomllib.loads(export_toml(cfg)))
    assert restored.dancer_analysis == cfg.dancer_analysis
