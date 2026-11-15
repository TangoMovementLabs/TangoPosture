"""Confidence-qualified range measurements shared by editing and scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from ..config import Config, RangeDefault
from ..domain import (
    FrameObservation,
    MetricObservation,
    QualityResult,
)
from ..geometry import (
    axis_curve_alignment,
    head_horizontal_alignment,
    knee_bend_alignment,
    knee_geometry_kwargs,
    shoulder_geometry_kwargs,
    shoulder_tension_alignment,
    torso_axis_alignment,
    wrapped,
)
from ..metrics import measure
from ..quality import reliable
from .models import CYCLIC_METRICS, METRIC_DETAILS, ApprovedRange, RangeProfile


def measure_ranges(frame: FrameObservation, cfg: Config) -> MetricObservation:
    """Use only confidence-qualified landmarks and existing measurement definitions."""
    core = measure(frame, cfg)
    values = {key: core.values.get(key) for key in METRIC_DETAILS}
    quality = {
        key: core.quality.get(key, QualityResult(False, "Required landmarks unavailable"))
        for key in METRIC_DETAILS
    }
    if len(frame.poses) != 1:
        return MetricObservation(frame.timestamp, dict.fromkeys(METRIC_DETAILS), {
            key: QualityResult(False, "Dancer tracking unavailable or ambiguous")
            for key in METRIC_DETAILS
        })
    pose = {name: point for name, point in frame.poses[0].items() if reliable(point, cfg.confidence)}
    for key, result, component in (
        ("head_pitch", head_horizontal_alignment(pose, frame.width, frame.height), 0),
        (
            "shoulder_clearance",
            shoulder_tension_alignment(
                pose, frame.width, frame.height, **shoulder_geometry_kwargs(cfg)
            ),
            1,
        ),
        ("hip_offset", axis_curve_alignment(
            pose, frame.width, frame.height, cfg.facing, cfg.side,
        ), 1),
        (
            "knee_flexion",
            knee_bend_alignment(
                pose, frame.width, frame.height, cfg.side, **knee_geometry_kwargs(cfg)
            ),
            0,
        ),
        ("torso_axis_tilt", torso_axis_alignment(
            pose, frame.width, frame.height, cfg.confidence,
        ), 0),
    ):
        value = result[component] if result is not None else None
        if isinstance(value, (int, float)) and math.isfinite(value):
            values[key] = abs(float(value)) if key == "torso_axis_tilt" else float(value)
            quality[key] = QualityResult(True)
        else:
            values[key] = None
    return MetricObservation(frame.timestamp, values, quality, core.view_width)


def view_eligible(
    view: Literal["side", "front", "all"],
    view_width: float | None,
    max_view_width: float,
) -> bool:
    """Return whether a frame's measured view_width satisfies the target view policy."""
    if view == "all":
        return True
    if view_width is None or not math.isfinite(view_width):
        return False
    if view == "side":
        return view_width <= max_view_width
    if view == "front":
        return view_width > max_view_width
    return False


def measure_profile(frame: FrameObservation, cfg: Config, profile: RangeProfile) -> MetricObservation:
    """Use the same per-view measurement path in approval and subsequent scoring."""
    return measure_targets(frame, cfg, profile.metrics)


def measure_targets(
    frame: FrameObservation, cfg: Config, targets: Mapping[str, RangeDefault | ApprovedRange],
) -> MetricObservation:
    observations = {
        view: measure_ranges(frame, cfg.model_copy(update={"allow_all_angles": view != "side"}))
        for view in {target.view for target in targets.values()}
    }
    result = measure_ranges(frame, cfg.model_copy(update={"allow_all_angles": True}))
    for key, target in targets.items():
        observation = observations[target.view]
        result.values[key] = observation.values[key]
        result.quality[key] = observation.quality[key]
        if key in CYCLIC_METRICS and (value := result.values[key]) is not None:
            center = (target.lower + target.upper) / 2
            result.values[key] = center + wrapped(value - center)
        width = observation.view_width
        if not view_eligible(target.view, width, cfg.max_view_width):
            result.values[key] = None
            result.quality[key] = QualityResult(False, "Return to this metric's approved camera view")
    return result


def metric_samples(
    frames: Sequence[FrameObservation],
    cfg: Config,
    *,
    view: Literal["side", "front", "all"] | None = None,
) -> dict[str, list[tuple[float, float | None]]]:
    effective_view: Literal["side", "front", "all"] = (
        view if view is not None else ("all" if cfg.allow_all_angles else "side")
    )
    eval_cfg = cfg.model_copy(update={"allow_all_angles": effective_view != "side"})
    result: dict[str, list[tuple[float, float | None]]] = {key: [] for key in METRIC_DETAILS}
    previous: dict[str, float] = {}
    for frame in frames:
        observation = measure_ranges(frame, eval_cfg)
        eligible = view_eligible(effective_view, observation.view_width, eval_cfg.max_view_width)
        for key, series in result.items():
            value = (
                observation.values[key]
                if eligible and observation.quality[key].valid
                else None
            )
            if value is not None and key in CYCLIC_METRICS:
                if key in previous:
                    value = previous[key] + wrapped(value - previous[key])
                previous[key] = value
            series.append((frame.timestamp, value))
    return result
