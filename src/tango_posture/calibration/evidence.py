"""Cached landmark evidence and conservative explicit dancer association."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain import (
    FrameObservation,
    Landmark,
)
from ..quality import reliable


@dataclass(frozen=True)
class Evidence:
    frames: list[FrameObservation]
    model_hash: str
    capture_profile: dict[str, Any]
    source: dict[str, Any]
    max_gap: float


def cached_evidence(path: Path, index: int = 0) -> Evidence:
    from ..archive import _archive_lock

    with _archive_lock(path):
        return _cached_evidence(path, index)


def _cached_evidence(path: Path, index: int) -> Evidence:
    from ..archive import file_sha256, read_analytics, read_manifest

    manifest = read_manifest(path)
    if not 0 <= index < len(manifest["segments"]):
        raise ValueError("Reference recording not found")
    data = read_analytics(path, index)
    if data is None:
        raise ValueError("No landmark cache. Analyze posture & movement in Review first")
    coach = data.get("coach_data") or {}
    capture = data.get("capture_profile") or coach.get("capture_profile") or {}
    if not isinstance(capture, dict):
        raise TypeError("Invalid cached capture settings")
    entries = coach.get("frames")
    from_coach = bool(entries)
    if not entries:
        entries = {str(i): entry for i, entry in enumerate(data.get("frames") or [])}
    timestamps = data.get("frame_timestamps") or []
    width = coach.get("width") or capture.get("width")
    height = coach.get("height") or capture.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("The cache has no verified frame dimensions; reanalyze this recording")
    model_hash = data.get("model_hash") or coach.get("model_hash")
    if not isinstance(model_hash, str):
        raise TypeError("The cache has no measured model identity; reanalyze this recording")
    frames = []
    for i, timestamp in enumerate(timestamps):
        entry = entries.get(str(i), {})
        if not isinstance(entry, dict):
            raise TypeError("Invalid cached frame")
        if from_coach and entry and "measurement_poses" not in entry:
            raise ValueError("Cache contains display landmarks only; reanalyze to cache raw measurements")
        raw_poses = entry.get("measurement_poses", []) if from_coach else entry.get("poses", [])
        if not isinstance(raw_poses, list):
            raise TypeError("Invalid cached frame / pose list")
        poses = []
        for raw in raw_poses:
            if not isinstance(raw, dict):
                raise TypeError("Invalid cached pose")
            pose = {}
            for name, coordinates in raw.items():
                if not isinstance(coordinates, list) or len(coordinates) != 5:
                    raise ValueError("Invalid cached landmark coordinates")
                if any(type(value) not in (int, float) and value is not None for value in coordinates):
                    raise TypeError("Cached landmark coordinates must be numbers or unknown")
                if coordinates[0] is None or coordinates[1] is None:
                    raise ValueError("Cached landmark image coordinates are missing")
                pose[name] = Landmark(
                    coordinates[0], coordinates[1], coordinates[3],
                    coordinates[4], coordinates[2],
                )
            poses.append(pose)
        frames.append(FrameObservation(timestamp, width, height, poses))
    if not frames or not any(frame.poses for frame in frames):
        raise ValueError("No usable landmark cache; reanalyze this recording")
    if "max_gap" not in data:
        raise ValueError("Cached tracking gap limit is missing; reanalyze this recording")
    max_gap = float(data["max_gap"])
    if not math.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("Invalid cached tracking gap")
    return Evidence(frames, model_hash, capture, {
        "archive": path.name, "archive_sha256": file_sha256(path), "recording": index,
        "provenance": data.get("source_provenance") or {},
    }, max_gap)


def _hip_center(pose: dict[str, Landmark], confidence: float) -> tuple[float, float] | None:
    hips = [pose.get(f"{side}_hip") for side in ("left", "right")]
    good = [point for point in hips if point is not None and reliable(point, confidence)]
    if not good:
        return None
    return (
        sum(point.x for point in good) / len(good),
        sum(point.y for point in good) / len(good),
    )


def has_linkable_pose(frame: FrameObservation, confidence: float) -> bool:
    """Return True when the frame has a pose with a reliable hip center for tracking."""
    return bool(frame.poses) and any(
        _hip_center(pose, confidence) is not None for pose in frame.poses
    )


def contiguous_pose_run_end(frames: list[FrameObservation], start_index: int) -> int:
    """Return the exclusive end index of the contiguous pose run starting at `start_index`."""
    end = start_index
    while end < len(frames) and frames[end].poses:
        end += 1
    return end


def is_static_pose_run(
    frames: list[FrameObservation],
    start_index: int,
    *,
    min_frames: int = 10,
    max_median_step: float = 1e-3,
) -> bool:
    """Detect frozen title/poster slides where consecutive poses barely move."""
    if start_index < 0 or start_index >= len(frames) or not frames[start_index].poses:
        return False
    end = contiguous_pose_run_end(frames, start_index)
    run = frames[start_index:end]
    if len(run) < min_frames:
        return False
    steps: list[float] = []
    for f0, f1 in itertools.pairwise(run):
        p0, p1 = f0.poses[0], f1.poses[0]
        common = [math.hypot(p1[k].x - p0[k].x, p1[k].y - p0[k].y) for k in p0 if k in p1]
        if common:
            steps.append(max(common))
    if len(steps) < min_frames - 1:
        return False
    steps.sort()
    return steps[len(steps) // 2] < max_median_step


def single_dancer_anchors(evidence: Evidence, confidence: float) -> dict[int, int]:
    """Compute anchor frames to link a solo dancer across brief dropouts or cuts."""
    if max((len(frame.poses) for frame in evidence.frames), default=0) > 1:
        raise ValueError("Auto-link is only available for single-dancer clips")
    anchors: dict[int, int] = {}
    previous: tuple[float, float] | None = None
    last_timestamp: float | None = None
    need_anchor = True
    skip_until = -1
    for index, frame in enumerate(evidence.frames):
        if index < skip_until:
            continue
        if not frame.poses:
            need_anchor = True
            continue
        center = _hip_center(frame.poses[0], confidence)
        if center is None:
            need_anchor = True
            continue
        if (
            need_anchor
            or previous is None
            or last_timestamp is None
            or frame.timestamp - last_timestamp > evidence.max_gap
            or math.dist(previous, center) >= 0.25
        ):
            if index > 0 and is_static_pose_run(evidence.frames, index):
                skip_until = contiguous_pose_run_end(evidence.frames, index)
                need_anchor = True
                continue
            anchors[index] = 0
            need_anchor = False
        previous = center
        last_timestamp = frame.timestamp
    return anchors or {0: 0}


def dancer_frames(evidence: Evidence, dancer: int, confidence: float) -> list[FrameObservation]:
    """Conservative spatial association; ambiguity/gaps end a track, never silently reseed it."""
    selected = []
    previous: tuple[float, float] | None = None
    last_timestamp: float | None = None
    lost = False
    for frame in evidence.frames:
        candidates = [(_hip_center(pose, confidence), pose) for pose in frame.poses]
        chosen = None
        if not lost:
            if previous is None:
                if 0 <= dancer < len(candidates) and candidates[dancer][0] is not None:
                    chosen = candidates[dancer]
                else:
                    lost = True
            elif last_timestamp is not None and frame.timestamp - last_timestamp <= evidence.max_gap:
                ranked = sorted(
                    ((math.dist(previous, center), center, pose)
                     for center, pose in candidates if center is not None), key=lambda item: item[0],
                )
                max_step = 0.25 if len(candidates) == 1 and len(ranked) == 1 else 0.15
                if (
                    ranked and ranked[0][0] < max_step
                    and (len(ranked) == 1 or ranked[1][0] - ranked[0][0] > 0.03)
                ):
                    chosen = (ranked[0][1], ranked[0][2])
            if previous is not None and chosen is None:
                lost = True
        if chosen is not None:
            previous = chosen[0]
            last_timestamp = frame.timestamp
        selected.append(FrameObservation(
            frame.timestamp, frame.width, frame.height, [chosen[1]] if chosen is not None else [],
        ))
    return selected
