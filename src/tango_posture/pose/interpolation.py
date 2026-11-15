"""Offline 2D display interpolation; inferred joints are never measurement evidence."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from itertools import pairwise
from typing import Literal

from ..domain import FrameObservation, Landmark
from ..quality import reliable

LOWER_LIMB_INTERP_JOINTS = frozenset(
    {
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    }
)


def _coordinate(
    frames: Sequence[FrameObservation], name: str, timestamp: float,
    anchors: list[int], axis: str,
) -> float:
    value = 0.0
    for anchor in anchors:
        component = getattr(frames[anchor].poses[0][name], axis)
        weight = 1.0
        for other in anchors:
            if other != anchor:
                weight *= (timestamp - frames[other].timestamp) / (
                    frames[anchor].timestamp - frames[other].timestamp
                )
        value += component * weight
    return value


def _overlaps_pivot(start: float, end: float, pivot_intervals: Sequence[tuple[float, float]]) -> bool:
    return any(p_start <= end + 1e-9 and p_end >= start - 1e-9 for p_start, p_end in pivot_intervals)


def interpolate_poses(
    frames: Sequence[FrameObservation], *,
    method: Literal["linear", "quadratic"],
    max_gap: float,
    confidence: float,
    pivot_intervals: Sequence[tuple[float, float]] = (),
    pivot_max_gap: float | None = None,
    cut_positions: set[int] | None = None,
    track_keys: Sequence[str] | None = None,
    origins_out: list[dict[str, str]] | None = None,
    validate_point: Callable[[int, str, float, float], bool] | None = None,
) -> list[FrameObservation]:
    result = list(frames)
    if origins_out is not None:
        while len(origins_out) < len(frames):
            origins_out.append({})
        for idx, frame in enumerate(frames):
            if len(frame.poses) == 1:
                for jname, lm in frame.poses[0].items():
                    if jname not in origins_out[idx] and reliable(lm, confidence):
                        origins_out[idx][jname] = "forward"
    if max_gap <= 0:
        return result

    segments: list[list[int]] = []
    for index, frame in enumerate(frames):
        if len(frame.poses) > 1 or not math.isfinite(frame.timestamp):
            segments.append([])
            continue
        split_here = (
            not segments
            or (segments[-1] and frame.timestamp <= frames[segments[-1][-1]].timestamp)
            or (cut_positions is not None and index in cut_positions)
            or (
                track_keys is not None
                and segments[-1]
                and track_keys[index] != track_keys[segments[-1][-1]]
            )
        )
        if split_here:
            segments.append([])
        segments[-1].append(index)

    for segment in segments:
        names = {name for index in segment for pose in frames[index].poses for name in pose}
        for name in names:
            valid = [
                index for index in segment
                if frames[index].poses and reliable(frames[index].poses[0].get(name), confidence)
            ]
            for position, (left, right) in enumerate(pairwise(valid)):
                start, end = frames[left].timestamp, frames[right].timestamp
                if right == left + 1:
                    continue
                effective_max_gap = max_gap
                if (
                    pivot_max_gap is not None
                    and name in LOWER_LIMB_INTERP_JOINTS
                    and _overlaps_pivot(start, end, pivot_intervals)
                ):
                    effective_max_gap = min(max_gap, pivot_max_gap)
                if effective_max_gap <= 0 or end - start > effective_max_gap + 1e-9:
                    continue
                anchors = [left, right]
                if method == "quadratic":
                    candidates = [
                        valid[p] for p in (position - 1, position + 2)
                        if 0 <= p < len(valid)
                        and min(abs(frames[valid[p]].timestamp - start),
                                abs(frames[valid[p]].timestamp - end)) <= effective_max_gap
                    ]
                    if candidates:
                        anchors.append(min(
                            candidates, key=lambda i: abs(frames[i].timestamp - (start + end) / 2),
                        ))
                for index in range(left + 1, right):
                    # Respect higher-priority backward or already reliable landmarks.
                    existing_pose = result[index].poses[0] if result[index].poses else {}
                    if reliable(existing_pose.get(name), confidence):
                        continue
                    timestamp = frames[index].timestamp

                    x, y = (
                        _coordinate(frames, name, timestamp, anchors, axis) for axis in ("x", "y")
                    )
                    curve_valid = (
                        math.isfinite(x)
                        and math.isfinite(y)
                        and 0 <= x <= 1
                        and 0 <= y <= 1
                        and (validate_point is None or validate_point(index, name, x, y))
                    )
                    if not curve_valid and len(anchors) > 2:
                        x, y = (
                            _coordinate(frames, name, timestamp, [left, right], axis)
                            for axis in ("x", "y")
                        )
                        curve_valid = (
                            math.isfinite(x)
                            and math.isfinite(y)
                            and 0 <= x <= 1
                            and 0 <= y <= 1
                            and (validate_point is None or validate_point(index, name, x, y))
                        )
                    if not curve_valid:
                        continue
                    pose = dict(existing_pose)
                    pose[name] = Landmark(
                        x=x, y=y,
                        visibility=1.0, presence=1.0,
                    )
                    result[index] = replace(result[index], poses=[pose])
                    if origins_out is not None:
                        origins_out[index][name] = "interpolated"
    return result
