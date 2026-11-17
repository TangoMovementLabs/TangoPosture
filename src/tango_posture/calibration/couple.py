"""Temporal comparison of identity-tracked couple measurements and approved ranges."""

from ..config import Config
from ..temporal import Evaluator
from .models import ApprovedRange


def compare_couple_series(
    key: str, points: list[tuple[float, float | None]], target: ApprovedRange, cfg: Config,
) -> tuple[list[tuple[float, float | None]], dict[float, str]]:
    settings = cfg.dancer_analysis
    temporal_cfg = cfg.model_copy(update={
        "smoothing": settings.calibration_smoothing_s,
        "entry_duration": settings.calibration_entry_duration_s,
        "clear_duration": settings.calibration_clear_duration_s,
        "max_gap": settings.max_observation_gap_s,
    })
    evaluator = Evaluator(key, target.median, temporal_cfg, bounds=(target.lower, target.upper))
    smoothed = []
    states = {}
    for timestamp, value in points:
        result = evaluator.update(timestamp, value)
        smoothed.append((timestamp, result["smoothed"]))
        states[timestamp] = result["state"]
    return smoothed, states
