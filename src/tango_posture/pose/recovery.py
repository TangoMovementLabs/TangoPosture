"""Display-only offline backward pose recovery and pivot-aware gap reconciliation."""

from __future__ import annotations

import logging
import math
import os
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..calibration.models import profile_from_calibration
from ..config import Config
from ..domain import FrameObservation, Landmark, PoseEstimator
from ..models import analysis_model_path, model_manifest, prepare_analysis_model, verify
from ..pose.mediapipe_backend import MediaPipeEstimator
from ..quality import reliable
from ..references.errors import ImportCancelledError
from ..references.video import VideoProbeInfo, probe_import_video

logger = logging.getLogger(__name__)

POSE_RECOVERY_SCHEMA_VERSION = 1
MAX_MEMORY_WINDOW_BYTES = 64 * 1024 * 1024
MAX_RECOVERY_FRAMES = 100_000
MAX_RECOVERY_WINDOWS = 5_000

LOWER_LIMB_JOINTS: tuple[str, ...] = (
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
)

LEG_CHAINS: dict[str, tuple[str, ...]] = {
    "left": ("left_hip", "left_knee", "left_ankle", "left_heel", "left_foot_index"),
    "right": ("right_hip", "right_knee", "right_ankle", "right_heel", "right_foot_index"),
}

CORE_UPPER_JOINTS: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)

DIAGNOSTIC_REASONS: tuple[str, ...] = (
    "No pose returned",
    "Landmark below confidence",
    "Identity ambiguous",
    "Missing anchors",
    "Frame alignment mismatch",
    "Recovered for display",
)

VALID_ORIGINS: tuple[str, ...] = ("forward", "backward", "interpolated")
VALID_STATUSES: tuple[str, ...] = ("completed", "partial", "skipped", "failed", "not_run")

CONTINUITY_TOLERANCE_TORSO = 0.30
SEGMENT_LENGTH_REL_TOL = 0.30


@dataclass(frozen=True)
class FrameIdentity:
    """Canonical frame identity tying frame index, source media time, and clip time."""

    frame_index: int
    source_time: float
    clip_time: float


@dataclass(frozen=True)
class CandidateGapWindow:
    """One prioritized, merged backward-recovery window inside a continuous shot."""

    window_id: str
    track_key: str
    shot_index: int
    gap_start_frame: int
    gap_end_frame: int
    window_start_frame: int
    window_end_frame: int
    gap_start_time: float
    gap_end_time: float
    window_start_time: float
    window_end_time: float
    affected_joints: tuple[str, ...]
    intersects_pivot: bool
    upper_body_present: bool
    clean_later_context: bool
    budget_skipped: bool = False
    skip_reason: str = ""
    manual: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "track_key": self.track_key,
            "shot_index": self.shot_index,
            "gap_start_frame": self.gap_start_frame,
            "gap_end_frame": self.gap_end_frame,
            "window_start_frame": self.window_start_frame,
            "window_end_frame": self.window_end_frame,
            "gap_start_time": round(self.gap_start_time, 4),
            "gap_end_time": round(self.gap_end_time, 4),
            "window_start_time": round(self.window_start_time, 4),
            "window_end_time": round(self.window_end_time, 4),
            "affected_joints": list(self.affected_joints),
            "intersects_pivot": self.intersects_pivot,
            "upper_body_present": self.upper_body_present,
            "clean_later_context": self.clean_later_context,
            "budget_skipped": self.budget_skipped,
            "skip_reason": self.skip_reason,
            "manual": self.manual,
        }


@dataclass
class RecoveredFrameEstimate:
    """Display-only recovered/interpolated joints and diagnostics for one frame."""

    frame_index: int
    source_time: float
    clip_time: float
    track_key: str
    landmarks: dict[str, dict[str, Any]] = field(default_factory=dict)
    origins: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, str] = field(default_factory=dict)
    status_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "source_time": round(self.source_time, 6),
            "clip_time": round(self.clip_time, 6),
            "track_key": self.track_key,
            "landmarks": self.landmarks,
            "diagnostics": self.diagnostics,
            "status_summary": self.status_summary,
        }


@dataclass
class BackwardPrediction:
    """Raw backward inference prediction mapped back to canonical FrameIdentity."""

    identity: FrameIdentity
    synthetic_ms: int
    observation: FrameObservation


def _extract_raw_pose(entry: Mapping[str, Any]) -> dict[str, Landmark]:
    raw_poses = entry.get("measurement_poses")
    if not isinstance(raw_poses, list):
        raw_poses = entry.get("poses")
    if not isinstance(raw_poses, list) or not raw_poses or not isinstance(raw_poses[0], Mapping):
        return {}
    pose: dict[str, Landmark] = {}
    for name, coords in raw_poses[0].items():
        if isinstance(coords, list | tuple) and len(coords) >= 3:
            try:
                x = float(coords[0])
                y = float(coords[1])
                z = float(coords[2]) if coords[2] is not None else None
                vis = float(coords[3]) if len(coords) >= 4 and coords[3] is not None else None
                pres = float(coords[4]) if len(coords) >= 5 and coords[4] is not None else None
            except (TypeError, ValueError):
                continue
            pose[str(name)] = Landmark(x=x, y=y, visibility=vis, presence=pres, z=z)
    return pose


def _serialise_landmark_dict(
    lm: Landmark, origin: str, *, include_origin: bool = True
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "x": round(float(lm.x), 5),
        "y": round(float(lm.y), 5),
        "z": round(float(lm.z), 5) if lm.z is not None and math.isfinite(lm.z) else 0.0,
        "visibility": round(float(lm.visibility), 4) if lm.visibility is not None else 1.0,
        "presence": round(float(lm.presence), 4) if lm.presence is not None else 1.0,
    }
    if include_origin:
        out["origin"] = origin
    return out


def _deserialise_landmark(raw: Mapping[str, Any]) -> Landmark | None:
    try:
        x = float(raw["x"])
        y = float(raw["y"])
        z = float(raw.get("z", 0.0))
        vis = float(raw.get("visibility", 1.0))
        pres = float(raw.get("presence", 1.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, z, vis, pres)):
        return None
    return Landmark(x=x, y=y, visibility=vis, presence=pres, z=z)


def _extract_frame_identities(coach_data: Mapping[str, Any]) -> list[FrameIdentity]:
    frames = coach_data.get("frames")
    if not isinstance(frames, Mapping):
        return []
    clip_in = float(coach_data.get("clip_in") or 0.0)
    identities: list[FrameIdentity] = []
    for raw_idx in sorted(frames, key=int):
        entry = frames[raw_idx]
        if not isinstance(entry, Mapping):
            continue
        frame_idx = int(raw_idx)
        source_time = float(entry.get("timestamp", frame_idx * 0.1))
        clip_time = float(entry.get("clip_timestamp", max(0.0, source_time - clip_in)))
        identities.append(FrameIdentity(frame_idx, source_time, clip_time))
    return identities


def _track_key_for_entry(entry: Mapping[str, Any]) -> str:
    if isinstance(entry.get("track_key"), str) and str(entry["track_key"]).strip():
        return str(entry["track_key"]).strip()
    dancer_idx = entry.get("dancer_index")
    if isinstance(dancer_idx, int) and not isinstance(dancer_idx, bool):
        return f"dancer_{dancer_idx}"
    return "single"


def split_shot_segments(
    coach_data: Mapping[str, Any],
    cfg: Config,
    *,
    movement_data: Mapping[str, Any] | None = None,
) -> list[list[FrameIdentity]]:
    """Split clip frames into continuous shot/track segments at cuts and timeline breaks."""
    identities = _extract_frame_identities(coach_data)
    if not identities:
        return []
    frames = coach_data.get("frames")
    assert isinstance(frames, Mapping)
    cut_frames: set[int] = set()
    if isinstance(movement_data, Mapping):
        for raw_cut in movement_data.get("cut_frames") or ():
            if isinstance(raw_cut, int) and not isinstance(raw_cut, bool):
                cut_frames.add(raw_cut)

    break_gap = max(cfg.max_gap * 3.0, 2.0)
    segments: list[list[FrameIdentity]] = [[identities[0]]]
    prev = identities[0]
    prev_track = _track_key_for_entry(frames.get(str(prev.frame_index), {}))
    for ident in identities[1:]:
        cur_track = _track_key_for_entry(frames.get(str(ident.frame_index), {}))
        dt = ident.source_time - prev.source_time
        crossed_cut = any(prev.frame_index < cf <= ident.frame_index for cf in cut_frames)
        if (
            not math.isfinite(ident.source_time)
            or dt <= 0.0
            or dt > break_gap
            or crossed_cut
            or cur_track != prev_track
        ):
            segments.append([])
        segments[-1].append(ident)
        prev = ident
        prev_track = cur_track
    return [seg for seg in segments if seg]


def _upper_body_present(pose: Mapping[str, Landmark], confidence: float) -> bool:
    sh_ok = any(reliable(pose.get(k), confidence) for k in ("left_shoulder", "right_shoulder"))
    hip_ok = any(reliable(pose.get(k), confidence) for k in ("left_hip", "right_hip"))
    return sh_ok and hip_ok


def _torso_geometry_ok(pose: Mapping[str, Landmark], confidence: float) -> bool:
    both_sides = all(reliable(pose.get(k), confidence) for k in CORE_UPPER_JOINTS)
    if both_sides:
        l_sh, r_sh = pose["left_shoulder"], pose["right_shoulder"]
        l_hp, r_hp = pose["left_hip"], pose["right_hip"]
        torso_len = math.hypot((l_sh.x + r_sh.x - l_hp.x - r_hp.x) * 0.5, (l_sh.y + r_sh.y - l_hp.y - r_hp.y) * 0.5)
        return torso_len >= 0.04
    for side in ("left", "right"):
        sh = pose.get(f"{side}_shoulder")
        hp = pose.get(f"{side}_hip")
        if sh is not None and hp is not None and reliable(sh, confidence) and reliable(hp, confidence):
            if math.hypot(sh.x - hp.x, sh.y - hp.y) >= 0.04:
                return True
    return False


def _has_clean_later_context(
    segment: list[FrameIdentity],
    gap_end_pos: int,
    poses_by_frame: Mapping[int, dict[str, Landmark]],
    affected_joints: Iterable[str],
    confidence: float,
    min_frames: int = 3,
) -> bool:
    """Require >= 3 consecutive qualified post-gap frames with ankle, knee, and torso geometry."""
    sides_needed: set[str] = set()
    for name in affected_joints:
        if name.startswith("left_"):
            sides_needed.add("left")
        elif name.startswith("right_"):
            sides_needed.add("right")
    if not sides_needed:
        sides_needed = {"left", "right"}

    consecutive = 0
    for ident in segment[gap_end_pos + 1 :]:
        pose = poses_by_frame.get(ident.frame_index, {})
        if not pose or not _torso_geometry_ok(pose, confidence):
            consecutive = 0
            continue
        sides_ok = True
        for side in sides_needed:
            knee = pose.get(f"{side}_knee")
            ankle = pose.get(f"{side}_ankle")
            if not (reliable(knee, confidence) and reliable(ankle, confidence)):
                sides_ok = False
                break
        if sides_ok:
            consecutive += 1
            if consecutive >= min_frames:
                return True
        else:
            consecutive = 0
    return False


def _pivot_intervals(movement_data: Mapping[str, Any] | None) -> list[tuple[int, int, float, float]]:
    if not isinstance(movement_data, Mapping):
        return []
    out: list[tuple[int, int, float, float]] = []
    for pivot in movement_data.get("pivots") or ():
        if not isinstance(pivot, Mapping):
            continue
        try:
            sf = int(pivot["start_frame"])
            ef = int(pivot["end_frame"])
            st = float(pivot.get("start_time", 0.0))
            et = float(pivot.get("end_time", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        out.append((sf, ef, st, et))
    return out


def _window_intersects_pivot(
    start_frame: int,
    end_frame: int,
    start_time: float,
    end_time: float,
    pivots: list[tuple[int, int, float, float]],
    segment: list[FrameIdentity],
    poses_by_frame: Mapping[int, dict[str, Landmark]],
    confidence: float,
) -> bool:
    for p_sf, p_ef, p_st, p_et in pivots:
        if p_sf <= end_frame and p_ef > start_frame:
            return True
        if p_et > p_st and p_st <= end_time and p_et >= start_time:
            return True
    # Heuristic fallback when movement_data has no pivots: hip-width contraction / crossover
    hip_widths: list[float] = []
    min_w = float("inf")
    max_w = 0.0
    signs: list[int] = []
    for ident in segment:
        if ident.frame_index < start_frame or ident.frame_index > end_frame:
            continue
        pose = poses_by_frame.get(ident.frame_index, {})
        lh = pose.get("left_hip")
        rh = pose.get("right_hip")
        if reliable(lh, confidence) and reliable(rh, confidence):
            assert lh is not None and rh is not None
            dx = lh.x - rh.x
            w = abs(dx)
            hip_widths.append(w)
            min_w = min(min_w, w)
            max_w = max(max_w, w)
            if abs(dx) > 0.01:
                signs.append(1 if dx > 0 else -1)
    if len(hip_widths) >= 3 and max_w > 0.02 and (min_w / max_w < 0.45 or len(set(signs)) > 1):
        return True
    return False


def detect_recovery_windows(
    coach_data: Mapping[str, Any],
    cfg: Config,
    *,
    movement_data: Mapping[str, Any] | None = None,
    target_range: tuple[float, float] | None = None,
    enforce_budget: bool = True,
) -> list[CandidateGapWindow]:
    """Identify, expand, merge, and prioritize lower-limb recovery windows.

    Priority order:
      1. Windows intersecting a detected pivot
      2. Lower-limb gaps where upper-body tracking remains usable
      3. Source-time ascending order
    """
    frames = coach_data.get("frames")
    if not isinstance(frames, Mapping) or not frames:
        return []

    segments = split_shot_segments(coach_data, cfg, movement_data=movement_data)
    if not segments:
        return []

    pivots = _pivot_intervals(movement_data)
    confidence = float(cfg.confidence)
    context_s = float(cfg.replay_recovery_context_s)
    max_gap_s = float(cfg.replay_recovery_max_gap_s)

    poses_by_frame: dict[int, dict[str, Landmark]] = {}
    for k, entry in frames.items():
        if isinstance(entry, Mapping):
            poses_by_frame[int(k)] = _extract_raw_pose(entry)

    raw_windows: list[CandidateGapWindow] = []
    for shot_idx, segment in enumerate(segments):
        if len(segment) < 4:
            continue
        track_key = _track_key_for_entry(frames.get(str(segment[0].frame_index), {}))
        seen_lower: set[str] = set()
        for ident in segment:
            pose = poses_by_frame.get(ident.frame_index, {})
            for joint in LOWER_LIMB_JOINTS:
                if reliable(pose.get(joint), confidence):
                    seen_lower.add(joint)
        if not seen_lower:
            seen_lower = {"left_knee", "right_knee", "left_ankle", "right_ankle"}

        # Detect frames inside this shot that are missing one or more observed lower-limb joints.
        gap_pos_start: int | None = None
        gap_affected: set[str] = set()

        def _flush_gap(start_pos: int, end_pos: int, joints: set[str]) -> None:
            gap_t0 = segment[start_pos].source_time
            gap_t1 = segment[end_pos].source_time
            prev_anchor_t = segment[start_pos - 1].source_time if start_pos > 0 else gap_t0
            next_anchor_t = (
                segment[end_pos + 1].source_time if end_pos + 1 < len(segment) else gap_t1
            )
            elapsed_gap = max(gap_t1 - gap_t0, next_anchor_t - prev_anchor_t)

            if target_range is not None:
                rng_lo, rng_hi = min(target_range), max(target_range)
                clip_t0 = segment[start_pos].clip_time
                clip_t1 = segment[end_pos].clip_time
                overlaps = (gap_t1 >= rng_lo - 1e-6 and gap_t0 <= rng_hi + 1e-6) or (
                    clip_t1 >= rng_lo - 1e-6 and clip_t0 <= rng_hi + 1e-6
                )
                if not overlaps:
                    return
            elif elapsed_gap > max_gap_s + 1e-6:
                return

            has_pre_anchor = any(
                any(reliable(poses_by_frame.get(segment[p].frame_index, {}).get(j), confidence) for j in joints)
                for p in range(max(0, start_pos - 5), start_pos)
            )
            clean_later = _has_clean_later_context(
                segment, end_pos, poses_by_frame, joints, confidence, min_frames=3
            )
            if not has_pre_anchor and start_pos == 0:
                clean_later = False

            win_t0 = gap_t0 - context_s
            win_t1 = gap_t1 + context_s
            win_start_pos = start_pos
            while win_start_pos > 0 and segment[win_start_pos - 1].source_time >= win_t0 - 1e-6:
                win_start_pos -= 1
            win_end_pos = end_pos
            while (
                win_end_pos + 1 < len(segment)
                and segment[win_end_pos + 1].source_time <= win_t1 + 1e-6
            ):
                win_end_pos += 1

            upper_ok = any(
                _upper_body_present(poses_by_frame.get(segment[p].frame_index, {}), confidence)
                for p in range(start_pos, end_pos + 1)
            )
            in_pivot = _window_intersects_pivot(
                segment[win_start_pos].frame_index,
                segment[win_end_pos].frame_index,
                segment[win_start_pos].source_time,
                segment[win_end_pos].source_time,
                pivots,
                segment,
                poses_by_frame,
                confidence,
            )
            skip_reason = "" if clean_later else "Missing anchors"
            raw_windows.append(
                CandidateGapWindow(
                    window_id=f"shot{shot_idx}_f{segment[win_start_pos].frame_index}_{segment[win_end_pos].frame_index}",
                    track_key=track_key,
                    shot_index=shot_idx,
                    gap_start_frame=segment[start_pos].frame_index,
                    gap_end_frame=segment[end_pos].frame_index,
                    window_start_frame=segment[win_start_pos].frame_index,
                    window_end_frame=segment[win_end_pos].frame_index,
                    gap_start_time=gap_t0,
                    gap_end_time=gap_t1,
                    window_start_time=segment[win_start_pos].source_time,
                    window_end_time=segment[win_end_pos].source_time,
                    affected_joints=tuple(j for j in LOWER_LIMB_JOINTS if j in joints),
                    intersects_pivot=in_pivot,
                    upper_body_present=upper_ok,
                    clean_later_context=clean_later,
                    skip_reason=skip_reason,
                    manual=target_range is not None,
                )
            )

        for pos, ident in enumerate(segment):
            pose = poses_by_frame.get(ident.frame_index, {})
            missing = {j for j in seen_lower if not reliable(pose.get(j), confidence)}
            if missing:
                if gap_pos_start is None:
                    gap_pos_start = pos
                    gap_affected = set(missing)
                else:
                    gap_affected |= missing
            else:
                if gap_pos_start is not None:
                    _flush_gap(gap_pos_start, pos - 1, gap_affected)
                    gap_pos_start = None
                    gap_affected = set()
        if gap_pos_start is not None:
            _flush_gap(gap_pos_start, len(segment) - 1, gap_affected)

        # For manual targeted recovery, if no smaller automatic gap matched inside target_range,
        # build an explicit window over the requested range so the user can recover longer dropouts.
        if target_range is not None and not any(w.shot_index == shot_idx for w in raw_windows):
            rng_lo, rng_hi = min(target_range), max(target_range)
            in_range_pos = [
                pos
                for pos, ident in enumerate(segment)
                if (ident.source_time >= rng_lo - 1e-6 and ident.source_time <= rng_hi + 1e-6)
                or (ident.clip_time >= rng_lo - 1e-6 and ident.clip_time <= rng_hi + 1e-6)
            ]
            if in_range_pos:
                s_pos, e_pos = in_range_pos[0], in_range_pos[-1]
                joints_in_gap: set[str] = set()
                for pos in range(s_pos, e_pos + 1):
                    pose = poses_by_frame.get(segment[pos].frame_index, {})
                    joints_in_gap |= {j for j in seen_lower if not reliable(pose.get(j), confidence)}
                if not joints_in_gap:
                    joints_in_gap = set(seen_lower)
                clean_later = _has_clean_later_context(
                    segment, e_pos, poses_by_frame, joints_in_gap, confidence, min_frames=3
                )
                win_t0 = segment[s_pos].source_time - context_s
                win_t1 = segment[e_pos].source_time + context_s
                win_s = s_pos
                while win_s > 0 and segment[win_s - 1].source_time >= win_t0 - 1e-6:
                    win_s -= 1
                win_e = e_pos
                while win_e + 1 < len(segment) and segment[win_e + 1].source_time <= win_t1 + 1e-6:
                    win_e += 1
                upper_ok = any(
                    _upper_body_present(poses_by_frame.get(segment[p].frame_index, {}), confidence)
                    for p in range(s_pos, e_pos + 1)
                )
                in_pivot = _window_intersects_pivot(
                    segment[win_s].frame_index,
                    segment[win_e].frame_index,
                    segment[win_s].source_time,
                    segment[win_e].source_time,
                    pivots,
                    segment,
                    poses_by_frame,
                    confidence,
                )
                raw_windows.append(
                    CandidateGapWindow(
                        window_id=f"shot{shot_idx}_f{segment[win_s].frame_index}_{segment[win_e].frame_index}",
                        track_key=track_key,
                        shot_index=shot_idx,
                        gap_start_frame=segment[s_pos].frame_index,
                        gap_end_frame=segment[e_pos].frame_index,
                        window_start_frame=segment[win_s].frame_index,
                        window_end_frame=segment[win_e].frame_index,
                        gap_start_time=segment[s_pos].source_time,
                        gap_end_time=segment[e_pos].source_time,
                        window_start_time=segment[win_s].source_time,
                        window_end_time=segment[win_e].source_time,
                        affected_joints=tuple(j for j in LOWER_LIMB_JOINTS if j in joints_in_gap),
                        intersects_pivot=in_pivot,
                        upper_body_present=upper_ok,
                        clean_later_context=clean_later,
                        skip_reason="" if clean_later else "Missing anchors",
                        manual=True,
                    )
                )

    # Merge overlapping windows within the same shot & track.
    merged: list[CandidateGapWindow] = []
    for win in sorted(raw_windows, key=lambda w: (w.shot_index, w.track_key, w.window_start_frame)):
        if (
            merged
            and merged[-1].shot_index == win.shot_index
            and merged[-1].track_key == win.track_key
            and win.window_start_frame <= merged[-1].window_end_frame + 1
        ):
            prev = merged[-1]
            combined_joints = tuple(
                j for j in LOWER_LIMB_JOINTS if j in set(prev.affected_joints) | set(win.affected_joints)
            )
            clean_later = win.clean_later_context if win.gap_end_frame >= prev.gap_end_frame else prev.clean_later_context
            skip_reason = "" if clean_later else (win.skip_reason or prev.skip_reason or "Missing anchors")
            new_start_f = min(prev.window_start_frame, win.window_start_frame)
            new_end_f = max(prev.window_end_frame, win.window_end_frame)
            merged[-1] = CandidateGapWindow(
                window_id=f"shot{prev.shot_index}_f{new_start_f}_{new_end_f}",
                track_key=prev.track_key,
                shot_index=prev.shot_index,
                gap_start_frame=min(prev.gap_start_frame, win.gap_start_frame),
                gap_end_frame=max(prev.gap_end_frame, win.gap_end_frame),
                window_start_frame=new_start_f,
                window_end_frame=new_end_f,
                gap_start_time=min(prev.gap_start_time, win.gap_start_time),
                gap_end_time=max(prev.gap_end_time, win.gap_end_time),
                window_start_time=min(prev.window_start_time, win.window_start_time),
                window_end_time=max(prev.window_end_time, win.window_end_time),
                affected_joints=combined_joints,
                intersects_pivot=prev.intersects_pivot or win.intersects_pivot,
                upper_body_present=prev.upper_body_present or win.upper_body_present,
                clean_later_context=clean_later,
                skip_reason=skip_reason,
                manual=prev.manual or win.manual,
            )
        else:
            merged.append(win)

    # Sort by priority: pivot first, then lower-limb with upper-body present, then source time.
    merged.sort(
        key=lambda w: (
            not w.clean_later_context,
            not w.intersects_pivot,
            not w.upper_body_present,
            w.window_start_frame,
        )
    )

    if enforce_budget and target_range is None:
        budget_s = float(cfg.replay_recovery_budget_s)
        spent_s = 0.0
        final_windows: list[CandidateGapWindow] = []
        for win in merged:
            if not win.clean_later_context:
                final_windows.append(win)
                continue
            win_dur = max(0.0, win.window_end_time - win.window_start_time)
            if spent_s + win_dur > budget_s + 1e-6:
                final_windows.append(
                    CandidateGapWindow(
                        window_id=win.window_id,
                        track_key=win.track_key,
                        shot_index=win.shot_index,
                        gap_start_frame=win.gap_start_frame,
                        gap_end_frame=win.gap_end_frame,
                        window_start_frame=win.window_start_frame,
                        window_end_frame=win.window_end_frame,
                        gap_start_time=win.gap_start_time,
                        gap_end_time=win.gap_end_time,
                        window_start_time=win.window_start_time,
                        window_end_time=win.window_end_time,
                        affected_joints=win.affected_joints,
                        intersects_pivot=win.intersects_pivot,
                        upper_body_present=win.upper_body_present,
                        clean_later_context=win.clean_later_context,
                        budget_skipped=True,
                        skip_reason="Automatic recovery budget exceeded",
                        manual=win.manual,
                    )
                )
            else:
                spent_s += win_dur
                final_windows.append(win)
        return final_windows

    return merged


@contextmanager
def iter_window_frames_reversed(
    video_path: Path | None,
    window_identities: list[FrameIdentity],
    *,
    expected_width: int,
    expected_height: int,
    max_memory_bytes: int = MAX_MEMORY_WINDOW_BYTES,
    cancel_check: Callable[[], bool] | None = None,
    frame_provider: Callable[[int], np.ndarray] | None = None,
):
    """Decode only [window_start_frame..window_end_frame], stay within memory cap, yield reversed."""
    if not window_identities:
        yield iter(())
        return

    spool_path: Path | None = None
    in_memory: list[tuple[FrameIdentity, np.ndarray]] = []
    spool_records: list[tuple[FrameIdentity, int, tuple[int, int, int], np.dtype]] = []
    bytes_used = 0
    cap: cv2.VideoCapture | None = None
    spool_fh = None

    try:
        target_indices = [ident.frame_index for ident in window_identities]
        if frame_provider is None:
            if video_path is None or not Path(video_path).is_file():
                raise ValueError("Raw source video is unavailable for tracking recovery")
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot decode video file for recovery: '{Path(video_path).name}'")
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if not math.isfinite(fps) or fps <= 0.0:
                fps = 30.0
            first_frame_idx = target_indices[0]
            if first_frame_idx > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(first_frame_idx))
                actual_pos = int(round(float(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)))
                if actual_pos > first_frame_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                    actual_pos = 0
                while actual_pos < first_frame_idx:
                    if cancel_check is not None and cancel_check():
                        raise ImportCancelledError("Tracking recovery was cancelled")
                    ok_skip, _ = cap.read()
                    if not ok_skip:
                        raise ValueError("Frame alignment mismatch")
                    actual_pos += 1
        else:
            fps = 30.0

        time_tol = max(0.08, 2.0 / max(fps, 1.0))
        first_pts: float | None = None

        for ident in window_identities:
            if cancel_check is not None and cancel_check():
                raise ImportCancelledError("Tracking recovery was cancelled")
            if frame_provider is not None:
                image = frame_provider(ident.frame_index)
            else:
                assert cap is not None
                ok, image = cap.read()
                if not ok or image is None:
                    raise ValueError("Frame alignment mismatch")
                raw_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if math.isfinite(raw_msec) and raw_msec >= 0.0:
                    raw_sec = raw_msec / 1000.0
                    if first_pts is None:
                        first_pts = raw_sec - ident.source_time if ident.frame_index > 0 else raw_sec
                    # Check PTS consistency when container provides non-zero moving timestamps
                    if raw_msec > 0.0 and ident.source_time > 0.0:
                        rel_t = raw_sec - (first_pts if first_pts is not None else 0.0)
                        if abs(rel_t - ident.source_time) > max(time_tol, 0.25):
                            # Check direct raw_sec too in case first_pts was zero-anchored
                            if abs(raw_sec - ident.source_time) > max(time_tol, 0.25):
                                raise ValueError("Frame alignment mismatch")

            if image is None or image.shape[1] != expected_width or image.shape[0] != expected_height:
                raise ValueError("Frame alignment mismatch")

            frame_arr = np.ascontiguousarray(image)
            frame_bytes = int(frame_arr.nbytes)
            if spool_fh is None and bytes_used + frame_bytes <= max_memory_bytes:
                in_memory.append((ident, frame_arr))
                bytes_used += frame_bytes
            else:
                if spool_fh is None:
                    fd, tmp_name = tempfile.mkstemp(prefix="tp_recovery_", suffix=".spool")
                    os.close(fd)
                    spool_path = Path(tmp_name)
                    spool_fh = spool_path.open("w+b")
                    for mem_ident, mem_arr in in_memory:
                        offset = spool_fh.tell()
                        spool_fh.write(mem_arr.tobytes())
                        spool_records.append((mem_ident, offset, mem_arr.shape, mem_arr.dtype))
                    in_memory.clear()
                    bytes_used = 0
                offset = spool_fh.tell()
                spool_fh.write(frame_arr.tobytes())
                spool_records.append((ident, offset, frame_arr.shape, frame_arr.dtype))

        if cap is not None:
            cap.release()
            cap = None

        def _reverse_iter():
            if spool_fh is not None:
                spool_fh.flush()
                for ident, offset, shape, dtype in reversed(spool_records):
                    if cancel_check is not None and cancel_check():
                        raise ImportCancelledError("Tracking recovery was cancelled")
                    count = int(np.prod(shape))
                    byte_len = count * int(np.dtype(dtype).itemsize)
                    spool_fh.seek(offset)
                    buf = spool_fh.read(byte_len)
                    if len(buf) != byte_len:
                        raise ValueError("Corrupt temporary frame spool during tracking recovery")
                    arr = np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
                    yield ident, arr
            else:
                for ident, arr in reversed(in_memory):
                    if cancel_check is not None and cancel_check():
                        raise ImportCancelledError("Tracking recovery was cancelled")
                    yield ident, arr

        yield _reverse_iter()
    finally:
        if cap is not None:
            with suppress(Exception):
                cap.release()
        if spool_fh is not None:
            with suppress(Exception):
                spool_fh.close()
        if spool_path is not None:
            with suppress(OSError):
                spool_path.unlink(missing_ok=True)


def run_backward_window_inference(
    *,
    video_path: Path | None,
    window_identities: list[FrameIdentity],
    expected_width: int,
    expected_height: int,
    estimator_factory: Callable[[], PoseEstimator],
    max_memory_bytes: int = MAX_MEMORY_WINDOW_BYTES,
    cancel_check: Callable[[], bool] | None = None,
    frame_provider: Callable[[int], np.ndarray] | None = None,
) -> list[BackwardPrediction]:
    """Run isolated reverse-order inference on one window with a fresh PoseEstimator."""
    if not window_identities:
        return []
    t_last = window_identities[-1].source_time
    estimator = estimator_factory()
    predictions: list[BackwardPrediction] = []
    prev_ms = -1
    try:
        with iter_window_frames_reversed(
            video_path,
            window_identities,
            expected_width=expected_width,
            expected_height=expected_height,
            max_memory_bytes=max_memory_bytes,
            cancel_check=cancel_check,
            frame_provider=frame_provider,
        ) as rev_frames:
            for ident, image in rev_frames:
                if cancel_check is not None and cancel_check():
                    raise ImportCancelledError("Tracking recovery was cancelled")
                raw_ms = int(round((t_last - ident.source_time) * 1000.0))
                synthetic_ms = max(prev_ms + 1, raw_ms)
                prev_ms = synthetic_ms
                obs = estimator.estimate(image, synthetic_ms / 1000.0)
                predictions.append(
                    BackwardPrediction(
                        identity=ident,
                        synthetic_ms=synthetic_ms,
                        observation=obs,
                    )
                )
    finally:
        with suppress(Exception):
            estimator.close()
    predictions.sort(key=lambda pred: pred.identity.frame_index)
    return predictions


def _landmark_px(lm: Landmark, width: int, height: int) -> tuple[float, float]:
    return (lm.x * width, lm.y * height)


def _pose_torso_center(pose: Mapping[str, Landmark], confidence: float) -> tuple[float, float] | None:
    pts = [pose[k] for k in CORE_UPPER_JOINTS if k in pose and reliable(pose.get(k), confidence)]
    if not pts:
        pts = [
            pose[k]
            for k in ("left_hip", "right_hip")
            if k in pose and reliable(pose.get(k), confidence)
        ]
    if not pts:
        return None
    return (sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts))


def _select_matching_pose(
    poses: list[dict[str, Landmark]],
    anchor_center: tuple[float, float] | None,
    confidence: float,
) -> dict[str, Landmark]:
    if not poses:
        return {}
    if len(poses) == 1 or anchor_center is None:
        return poses[0]
    best_pose = poses[0]
    best_dist = float("inf")
    for pose in poses:
        center = _pose_torso_center(pose, confidence)
        if center is None:
            continue
        dist = math.hypot(center[0] - anchor_center[0], center[1] - anchor_center[1])
        if dist < best_dist:
            best_dist = dist
            best_pose = pose
    return best_pose


def _window_reference_scales(
    window_identities: list[FrameIdentity],
    forward_poses: Mapping[int, dict[str, Landmark]],
    width: int,
    height: int,
    confidence: float,
) -> tuple[float, dict[str, float], dict[str, float]]:
    torso_lengths: list[float] = []
    thighs: dict[str, list[float]] = {"left": [], "right": []}
    shins: dict[str, list[float]] = {"left": [], "right": []}
    for ident in window_identities:
        pose = forward_poses.get(ident.frame_index, {})
        l_sh, r_sh = pose.get("left_shoulder"), pose.get("right_shoulder")
        l_hp, r_hp = pose.get("left_hip"), pose.get("right_hip")
        if all(reliable(p, confidence) for p in (l_sh, r_sh, l_hp, r_hp)):
            assert l_sh and r_sh and l_hp and r_hp
            sh_mid = ((l_sh.x + r_sh.x) * 0.5 * width, (l_sh.y + r_sh.y) * 0.5 * height)
            hp_mid = ((l_hp.x + r_hp.x) * 0.5 * width, (l_hp.y + r_hp.y) * 0.5 * height)
            torso_lengths.append(math.dist(sh_mid, hp_mid))
        else:
            for side in ("left", "right"):
                sh, hp = pose.get(f"{side}_shoulder"), pose.get(f"{side}_hip")
                if reliable(sh, confidence) and reliable(hp, confidence):
                    assert sh and hp
                    torso_lengths.append(
                        math.dist(_landmark_px(sh, width, height), _landmark_px(hp, width, height))
                    )
                    break
        for side in ("left", "right"):
            hp = pose.get(f"{side}_hip")
            kn = pose.get(f"{side}_knee")
            ak = pose.get(f"{side}_ankle")
            if reliable(hp, confidence) and reliable(kn, confidence):
                assert hp and kn
                thighs[side].append(
                    math.dist(_landmark_px(hp, width, height), _landmark_px(kn, width, height))
                )
            if reliable(kn, confidence) and reliable(ak, confidence):
                assert kn and ak
                shins[side].append(
                    math.dist(_landmark_px(kn, width, height), _landmark_px(ak, width, height))
                )

    torso_px = float(np.median(torso_lengths)) if torso_lengths else max(24.0, 0.18 * height)
    med_thigh = {s: float(np.median(vals)) for s, vals in thighs.items() if vals}
    med_shin = {s: float(np.median(vals)) for s, vals in shins.items() if vals}
    return torso_px, med_thigh, med_shin


def _segment_gate_ok(
    joint_name: str,
    candidate_lm: Landmark,
    frame_pose: Mapping[str, Landmark],
    width: int,
    height: int,
    confidence: float,
    med_thigh: Mapping[str, float],
    med_shin: Mapping[str, float],
) -> bool:
    side = "left" if joint_name.startswith("left_") else "right" if joint_name.startswith("right_") else ""
    if not side:
        return True
    test_pose = dict(frame_pose)
    test_pose[joint_name] = candidate_lm
    hp = test_pose.get(f"{side}_hip")
    kn = test_pose.get(f"{side}_knee")
    ak = test_pose.get(f"{side}_ankle")

    def _within_tol(d: float, ref: float | None) -> bool:
        if ref is None or ref <= 1e-3:
            return True
        return (1.0 - SEGMENT_LENGTH_REL_TOL) * ref <= d <= (1.0 + SEGMENT_LENGTH_REL_TOL) * ref

    if joint_name in (f"{side}_hip", f"{side}_knee") and reliable(hp, confidence) and reliable(kn, confidence):
        assert hp and kn
        d_thigh = math.dist(_landmark_px(hp, width, height), _landmark_px(kn, width, height))
        if not _within_tol(d_thigh, med_thigh.get(side)):
            return False
    if joint_name in (f"{side}_knee", f"{side}_ankle") and reliable(kn, confidence) and reliable(ak, confidence):
        assert kn and ak
        d_shin = math.dist(_landmark_px(kn, width, height), _landmark_px(ak, width, height))
        if not _within_tol(d_shin, med_shin.get(side)):
            return False
    return True


def _continuity_proposal(
    identities: list[FrameIdentity],
    pos: int,
    joint: str,
    poses_by_frame: Mapping[int, Mapping[str, Landmark]],
    confidence: float,
) -> tuple[float, float] | None:
    """Compute time-weighted continuity proposal in normalized coordinates from reliable anchors."""
    prev_pos = None
    for p in range(pos - 1, -1, -1):
        lm = poses_by_frame.get(identities[p].frame_index, {}).get(joint)
        if reliable(lm, confidence):
            prev_pos = p
            break
    next_pos = None
    for p in range(pos + 1, len(identities)):
        lm = poses_by_frame.get(identities[p].frame_index, {}).get(joint)
        if reliable(lm, confidence):
            next_pos = p
            break

    if prev_pos is not None and next_pos is not None:
        t0 = identities[prev_pos].source_time
        t1 = identities[next_pos].source_time
        p0 = poses_by_frame[identities[prev_pos].frame_index][joint]
        p1 = poses_by_frame[identities[next_pos].frame_index][joint]
        alpha = (identities[pos].source_time - t0) / max(1e-6, t1 - t0)
        return (p0.x + alpha * (p1.x - p0.x), p0.y + alpha * (p1.y - p0.y))
    if next_pos is not None:
        p1 = poses_by_frame[identities[next_pos].frame_index][joint]
        return (p1.x, p1.y)
    if prev_pos is not None:
        p0 = poses_by_frame[identities[prev_pos].frame_index][joint]
        return (p0.x, p0.y)
    return None


def _side_conflict_detected(
    joint: str,
    cand_lm: Landmark,
    fwd_pose: Mapping[str, Landmark],
    identities: list[FrameIdentity],
    pos: int,
    resolved_poses: Mapping[int, Mapping[str, Landmark]],
    width: int,
    height: int,
    torso_px: float,
    confidence: float,
) -> bool:
    """Reject candidates that plausibly swap left/right anatomy or collide onto the opposite side."""
    if joint.startswith("left_"):
        opp_joint = "right_" + joint[len("left_") :]
    elif joint.startswith("right_"):
        opp_joint = "left_" + joint[len("right_") :]
    else:
        return False

    cand_px = _landmark_px(cand_lm, width, height)
    # 1. Collides closely onto an already reliable opposite-side forward landmark on the same frame.
    opp_fwd = fwd_pose.get(opp_joint)
    if reliable(opp_fwd, confidence):
        assert opp_fwd is not None
        if math.dist(cand_px, _landmark_px(opp_fwd, width, height)) < 0.15 * torso_px:
            # Allow true anatomical overlap only if expected track proposal is equally close to opp_fwd
            own_prop = _continuity_proposal(identities, pos, joint, resolved_poses, confidence)
            if own_prop is None or math.dist(
                (own_prop[0] * width, own_prop[1] * height), _landmark_px(opp_fwd, width, height)
            ) > 0.25 * torso_px:
                return True

    # 2. Candidate sits much closer to opposite-side anchor track than its own side anchor track.
    own_prop = _continuity_proposal(identities, pos, joint, resolved_poses, confidence)
    opp_prop = _continuity_proposal(identities, pos, opp_joint, resolved_poses, confidence)
    if own_prop is not None and opp_prop is not None:
        own_px = (own_prop[0] * width, own_prop[1] * height)
        opp_px = (opp_prop[0] * width, opp_prop[1] * height)
        if math.dist(own_px, opp_px) > 0.20 * torso_px:
            d_own = math.dist(cand_px, own_px)
            d_opp = math.dist(cand_px, opp_px)
            if d_opp < 0.7 * d_own and d_opp <= CONTINUITY_TOLERANCE_TORSO * torso_px:
                return True

    # 3. Forward had a weak/medium plausible alternate assignment contradicting backward side.
    fwd_same = fwd_pose.get(joint)
    if fwd_same is not None and fwd_same.visibility is not None and 0.25 <= fwd_same.visibility < confidence:
        fwd_px = _landmark_px(fwd_same, width, height)
        if opp_prop is not None:
            opp_px = (opp_prop[0] * width, opp_prop[1] * height)
            if math.dist(cand_px, opp_px) < 0.25 * torso_px and math.dist(fwd_px, cand_px) > 0.30 * torso_px:
                return True

    return False


def _human_joint_label(joint: str) -> str:
    return joint.replace("_", " ").capitalize()


def _format_frame_status_summary(
    origins: Mapping[str, str],
    diagnostics: Mapping[str, str],
) -> str:
    backward_joints = [j for j, org in origins.items() if org == "backward"]
    interp_joints = [j for j, org in origins.items() if org == "interpolated"]
    unresolved = [
        (j, reason)
        for j, reason in diagnostics.items()
        if j not in origins and reason != "Recovered for display"
    ]
    parts: list[str] = []
    if backward_joints:
        focus = next(
            (j for j in backward_joints if j.endswith("_ankle") or j.endswith("_knee")),
            backward_joints[0],
        )
        parts.append(f"{_human_joint_label(focus)} estimated from backward pass")
    elif interp_joints:
        focus = next(
            (j for j in interp_joints if j.endswith("_ankle") or j.endswith("_knee")),
            interp_joints[0],
        )
        parts.append(f"{_human_joint_label(focus)} interpolated")
    if unresolved:
        reason = unresolved[0][1]
        if reason == "Identity ambiguous":
            parts.append("Leg identity unresolved")
        else:
            parts.append(f"Leg unresolved ({reason})")
    return "; ".join(parts)


def reconcile_window_poses(
    *,
    window: CandidateGapWindow,
    window_identities: list[FrameIdentity],
    forward_poses: Mapping[int, dict[str, Landmark]],
    backward_predictions: list[BackwardPrediction],
    cfg: Config,
    width: int,
    height: int,
    in_pivot: bool,
) -> dict[int, RecoveredFrameEstimate]:
    """Reconcile forward observations, backward candidates, and gated display interpolation."""
    confidence = float(cfg.confidence)
    torso_px, med_thigh, med_shin = _window_reference_scales(
        window_identities, forward_poses, width, height, confidence
    )
    preds_by_frame = {pred.identity.frame_index: pred for pred in backward_predictions}

    # Build working resolved pose table starting with reliable forward landmarks.
    resolved_poses: dict[int, dict[str, Landmark]] = {}
    origins_by_frame: dict[int, dict[str, str]] = {}
    diagnostics_by_frame: dict[int, dict[str, str]] = {}

    for ident in window_identities:
        fwd = forward_poses.get(ident.frame_index, {})
        resolved: dict[str, Landmark] = {}
        origins: dict[str, str] = {}
        for name, lm in fwd.items():
            if reliable(lm, confidence):
                resolved[name] = lm
                origins[name] = "forward"
        resolved_poses[ident.frame_index] = resolved
        origins_by_frame[ident.frame_index] = origins
        diagnostics_by_frame[ident.frame_index] = {}

    anchor_center: tuple[float, float] | None = None
    for ident in reversed(window_identities):
        c = _pose_torso_center(forward_poses.get(ident.frame_index, {}), confidence)
        if c is not None:
            anchor_center = c
            break

    # Evaluate backward candidates in reverse frame order so accepted post-gap landmarks anchor earlier frames.
    for pos in range(len(window_identities) - 1, -1, -1):
        ident = window_identities[pos]
        fidx = ident.frame_index
        if fidx < window.gap_start_frame or fidx > window.gap_end_frame:
            continue
        fwd_pose = forward_poses.get(fidx, {})
        cur_resolved = resolved_poses[fidx]
        cur_origins = origins_by_frame[fidx]
        cur_diag = diagnostics_by_frame[fidx]

        pred = preds_by_frame.get(fidx)
        if pred is None or not pred.observation.poses:
            for joint in window.affected_joints:
                if joint not in cur_resolved:
                    cur_diag[joint] = "No pose returned"
            continue

        bw_pose = _select_matching_pose(pred.observation.poses, anchor_center, confidence)
        bw_center = _pose_torso_center(bw_pose, confidence)
        if bw_center is not None:
            anchor_center = bw_center

        # Pass through joints in deterministic chain order (hips -> knees -> ankles -> heels -> feet)
        for joint in LOWER_LIMB_JOINTS:
            if joint in cur_resolved:
                continue
            if joint not in window.affected_joints and reliable(fwd_pose.get(joint), confidence):
                continue
            cand = bw_pose.get(joint)
            if cand is None or not reliable(cand, confidence):
                if joint in window.affected_joints:
                    cur_diag[joint] = "Landmark below confidence"
                continue
            if _side_conflict_detected(
                joint, cand, fwd_pose, window_identities, pos, resolved_poses, width, height, torso_px, confidence
            ):
                cur_diag[joint] = "Identity ambiguous"
                continue

            proposal = _continuity_proposal(window_identities, pos, joint, resolved_poses, confidence)
            if proposal is None:
                cur_diag[joint] = "Missing anchors"
                continue

            cand_px = _landmark_px(cand, width, height)
            prop_px = (proposal[0] * width, proposal[1] * height)
            if math.dist(cand_px, prop_px) > CONTINUITY_TOLERANCE_TORSO * torso_px:
                cur_diag[joint] = "Identity ambiguous"
                continue

            if not _segment_gate_ok(
                joint, cand, cur_resolved, width, height, confidence, med_thigh, med_shin
            ):
                cur_diag[joint] = "Identity ambiguous"
                continue

            cur_resolved[joint] = Landmark(
                x=cand.x,
                y=cand.y,
                visibility=cand.visibility,
                presence=cand.presence,
                z=cand.z,
            )
            cur_origins[joint] = "backward"
            cur_diag[joint] = "Recovered for display"

    # Rule 3: gated interpolation fallback when both forward and backward left a joint unresolved.
    max_interp_gap = float(cfg.replay_joint_max_gap)
    pivot_interp_cap = float(cfg.replay_pivot_interpolation_max_gap_s)
    if max_interp_gap > 0.0:
        for joint in LOWER_LIMB_JOINTS:
            effective_max_gap = min(max_interp_gap, pivot_interp_cap) if (in_pivot or window.intersects_pivot) else max_interp_gap
            if effective_max_gap <= 0.0:
                continue
            valid_positions = [
                p
                for p, ident in enumerate(window_identities)
                if reliable(resolved_poses[ident.frame_index].get(joint), confidence)
            ]
            for idx_in_valid in range(len(valid_positions) - 1):
                left_p = valid_positions[idx_in_valid]
                right_p = valid_positions[idx_in_valid + 1]
                if right_p == left_p + 1:
                    continue
                t0 = window_identities[left_p].source_time
                t1 = window_identities[right_p].source_time
                if (t1 - t0) > effective_max_gap + 1e-9:
                    continue
                anchors = [left_p, right_p]
                if cfg.replay_joint_interpolation == "quadratic":
                    candidates = [
                        valid_positions[k]
                        for k in (idx_in_valid - 1, idx_in_valid + 2)
                        if 0 <= k < len(valid_positions)
                        and min(
                            abs(window_identities[valid_positions[k]].source_time - t0),
                            abs(window_identities[valid_positions[k]].source_time - t1),
                        ) <= effective_max_gap
                    ]
                    if candidates:
                        anchors.append(
                            min(
                                candidates,
                                key=lambda p: abs(window_identities[p].source_time - (t0 + t1) * 0.5),
                            )
                        )
                for p in range(left_p + 1, right_p):
                    fidx = window_identities[p].frame_index
                    if fidx < window.gap_start_frame or fidx > window.gap_end_frame:
                        continue
                    if joint in resolved_poses[fidx]:
                        continue
                    ts = window_identities[p].source_time
                    linear_x, linear_y = _lagrange_xy(window_identities, resolved_poses, joint, ts, [left_p, right_p])
                    cand_x, cand_y = (
                        _lagrange_xy(window_identities, resolved_poses, joint, ts, anchors)
                        if len(anchors) > 2
                        else (linear_x, linear_y)
                    )
                    accepted = _try_gate_interpolated_point(
                        joint=joint,
                        cand_xy=(cand_x, cand_y),
                        linear_xy=(linear_x, linear_y),
                        frame_pose=resolved_poses[fidx],
                        width=width,
                        height=height,
                        torso_px=torso_px,
                        confidence=confidence,
                        med_thigh=med_thigh,
                        med_shin=med_shin,
                    )
                    if accepted is None and len(anchors) > 2:
                        accepted = _try_gate_interpolated_point(
                            joint=joint,
                            cand_xy=(linear_x, linear_y),
                            linear_xy=(linear_x, linear_y),
                            frame_pose=resolved_poses[fidx],
                            width=width,
                            height=height,
                            torso_px=torso_px,
                            confidence=confidence,
                            med_thigh=med_thigh,
                            med_shin=med_shin,
                        )
                    if accepted is not None:
                        resolved_poses[fidx][joint] = accepted
                        origins_by_frame[fidx][joint] = "interpolated"
                        diagnostics_by_frame[fidx][joint] = "Recovered for display"

    estimates: dict[int, RecoveredFrameEstimate] = {}
    for ident in window_identities:
        fidx = ident.frame_index
        if fidx < window.gap_start_frame or fidx > window.gap_end_frame:
            continue
        origins = {
            k: v for k, v in origins_by_frame[fidx].items() if v in ("backward", "interpolated")
        }
        diags = diagnostics_by_frame[fidx]
        if not origins and not diags:
            continue
        landmarks = {
            name: _serialise_landmark_dict(resolved_poses[fidx][name], origin)
            for name, origin in origins.items()
        }
        estimates[fidx] = RecoveredFrameEstimate(
            frame_index=fidx,
            source_time=ident.source_time,
            clip_time=ident.clip_time,
            track_key=window.track_key,
            landmarks=landmarks,
            origins=origins,
            diagnostics=dict(diags),
            status_summary=_format_frame_status_summary(origins, diags),
        )
    return estimates


def _lagrange_xy(
    identities: list[FrameIdentity],
    poses: Mapping[int, Mapping[str, Landmark]],
    joint: str,
    timestamp: float,
    anchors: list[int],
) -> tuple[float, float]:
    vx = 0.0
    vy = 0.0
    for a in anchors:
        t_a = identities[a].source_time
        lm_a = poses[identities[a].frame_index][joint]
        w = 1.0
        for b in anchors:
            if b != a:
                t_b = identities[b].source_time
                w *= (timestamp - t_b) / (t_a - t_b)
        vx += lm_a.x * w
        vy += lm_a.y * w
    return vx, vy


def _try_gate_interpolated_point(
    *,
    joint: str,
    cand_xy: tuple[float, float],
    linear_xy: tuple[float, float],
    frame_pose: Mapping[str, Landmark],
    width: int,
    height: int,
    torso_px: float,
    confidence: float,
    med_thigh: Mapping[str, float],
    med_shin: Mapping[str, float],
) -> Landmark | None:
    cx, cy = cand_xy
    if not (math.isfinite(cx) and math.isfinite(cy) and 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None
    cand_px = (cx * width, cy * height)
    lin_px = (linear_xy[0] * width, linear_xy[1] * height)
    if math.dist(cand_px, lin_px) > CONTINUITY_TOLERANCE_TORSO * torso_px:
        return None
    cand_lm = Landmark(x=cx, y=cy, visibility=1.0, presence=1.0)
    if not _segment_gate_ok(
        joint, cand_lm, frame_pose, width, height, confidence, med_thigh, med_shin
    ):
        return None
    return cand_lm


def _default_estimator_factory(cfg: Config, model_path: Path | None = None) -> Callable[[], PoseEstimator]:
    target = model_path or analysis_model_path(cfg)
    prepared = prepare_analysis_model(cfg) if not target.is_file() else target
    if prepared.is_file():
        verify(prepared, cfg.analysis_model)

    def _make() -> PoseEstimator:
        return MediaPipeEstimator(prepared, variant=cfg.analysis_model)

    return _make


def run_clip_tracking_recovery(
    *,
    video_path: Path | None,
    coach_data: Mapping[str, Any],
    cfg: Config,
    movement_data: Mapping[str, Any] | None = None,
    model_path: Path | None = None,
    estimator_factory: Callable[[], PoseEstimator] | None = None,
    target_range: tuple[float, float] | None = None,
    enforce_budget: bool = True,
    max_memory_bytes: int = MAX_MEMORY_WINDOW_BYTES,
    cancel_check: Callable[[], bool] | None = None,
    status_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    frame_provider: Callable[[int], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run prioritized, bounded backward tracking recovery on eligible windows."""
    identities = _extract_frame_identities(coach_data)
    width = int(coach_data.get("width") or 0)
    height = int(coach_data.get("height") or 0)
    fps = float(coach_data.get("fps") or coach_data.get("video_fps") or 30.0)
    source_sha256 = coach_data.get("source_sha256")
    model_hash = coach_data.get("model_hash")
    model_variant = str(coach_data.get("model_variant") or cfg.analysis_model)
    dancer_selection = coach_data.get("dancer_selection") if isinstance(coach_data.get("dancer_selection"), dict) else None

    frames_map = coach_data.get("frames") if isinstance(coach_data.get("frames"), Mapping) else {}
    forward_poses: dict[int, dict[str, Landmark]] = {
        int(k): _extract_raw_pose(entry)
        for k, entry in frames_map.items()
        if isinstance(entry, Mapping)
    }

    windows = detect_recovery_windows(
        coach_data,
        cfg,
        movement_data=movement_data,
        target_range=target_range,
        enforce_budget=enforce_budget,
    )
    runnable = [w for w in windows if w.clean_later_context and not w.budget_skipped]
    skipped_budget = sum(1 for w in windows if w.budget_skipped)

    if not runnable:
        status = "skipped" if windows else "completed"
        msg = (
            "No eligible lower-limb recovery windows found."
            if not windows
            else f"Skipped {len(windows)} recovery window(s): {windows[0].skip_reason or 'missing anchors'}."
        )
        return _assemble_payload(
            status=status,
            status_message=msg,
            source_sha256=source_sha256,
            model_name=model_variant,
            model_sha256=model_hash,
            width=width,
            height=height,
            fps=fps,
            frame_count=len(identities),
            dancer_selection=dancer_selection,
            cfg=cfg,
            windows=windows,
            estimates={},
            skipped_budget=skipped_budget,
            attempted=0,
            recovered_windows=0,
            unresolved_windows=len(windows) - skipped_budget,
        )

    factory = estimator_factory or _default_estimator_factory(cfg, model_path)
    id_by_frame = {ident.frame_index: ident for ident in identities}
    all_estimates: dict[int, RecoveredFrameEstimate] = {}
    attempted = 0
    recovered_windows = 0
    unresolved_windows = 0
    failure_messages: list[str] = []

    total_win = len(runnable)
    for win_idx, win in enumerate(runnable, start=1):
        if cancel_check is not None and cancel_check():
            raise ImportCancelledError("Tracking recovery was cancelled")
        step_msg = f"Recovering tracking in {win_idx} of {total_win} windows"
        if status_callback is not None:
            status_callback(step_msg)
        if progress_callback is not None:
            progress_callback(82 + round(15 * (win_idx - 1) / max(1, total_win)), 100, step_msg)

        win_ids = [
            id_by_frame[f]
            for f in range(win.window_start_frame, win.window_end_frame + 1)
            if f in id_by_frame
        ]
        if not win_ids:
            unresolved_windows += 1
            continue
        attempted += 1
        try:
            preds = run_backward_window_inference(
                video_path=video_path,
                window_identities=win_ids,
                expected_width=width,
                expected_height=height,
                estimator_factory=factory,
                max_memory_bytes=max_memory_bytes,
                cancel_check=cancel_check,
                frame_provider=frame_provider,
            )
            win_estimates = reconcile_window_poses(
                window=win,
                window_identities=win_ids,
                forward_poses=forward_poses,
                backward_predictions=preds,
                cfg=cfg,
                width=width,
                height=height,
                in_pivot=win.intersects_pivot,
            )
            if any(est.landmarks for est in win_estimates.values()):
                recovered_windows += 1
            else:
                unresolved_windows += 1
            all_estimates.update(win_estimates)
        except ImportCancelledError:
            raise
        except Exception as exc:
            unresolved_windows += 1
            failure_messages.append(str(exc))
            logger.warning("Tracking recovery failed on window %s: %s", win.window_id, exc)
            # Surface explicit Frame alignment mismatch or fallback diagnostic
            diag_reason = (
                "Frame alignment mismatch"
                if "alignment mismatch" in str(exc).lower()
                else "Missing anchors"
            )
            for ident in win_ids:
                if win.gap_start_frame <= ident.frame_index <= win.gap_end_frame:
                    all_estimates[ident.frame_index] = RecoveredFrameEstimate(
                        frame_index=ident.frame_index,
                        source_time=ident.source_time,
                        clip_time=ident.clip_time,
                        track_key=win.track_key,
                        landmarks={},
                        origins={},
                        diagnostics={"_frame": diag_reason},
                        status_summary=f"Recovery unavailable ({diag_reason})",
                    )

    if failure_messages and recovered_windows == 0:
        status = "failed"
        msg = f"Tracking recovery failed ({failure_messages[0]}); keeping forward tracking."
    elif unresolved_windows > 0 or skipped_budget > 0:
        status = "partial"
        msg = (
            f"Recovered {recovered_windows} of {attempted} window(s)"
            + (f"; {unresolved_windows} left unresolved" if unresolved_windows else "")
            + (f"; {skipped_budget} skipped by budget" if skipped_budget else "")
            + "."
        )
    else:
        status = "completed"
        msg = f"Recovered {recovered_windows} of {attempted} window(s) for display."

    return _assemble_payload(
        status=status,
        status_message=msg,
        source_sha256=source_sha256,
        model_name=model_variant,
        model_sha256=model_hash,
        width=width,
        height=height,
        fps=fps,
        frame_count=len(identities),
        dancer_selection=dancer_selection,
        cfg=cfg,
        windows=windows,
        estimates=all_estimates,
        skipped_budget=skipped_budget,
        attempted=attempted,
        recovered_windows=recovered_windows,
        unresolved_windows=unresolved_windows,
    )


def recover_clip_tracking_range(
    *,
    video_path: Path | None,
    coach_data: dict[str, Any],
    cfg: Config,
    start_time: float,
    end_time: float,
    movement_data: Mapping[str, Any] | None = None,
    model_path: Path | None = None,
    estimator_factory: Callable[[], PoseEstimator] | None = None,
    max_memory_bytes: int = MAX_MEMORY_WINDOW_BYTES,
    cancel_check: Callable[[], bool] | None = None,
    status_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    frame_provider: Callable[[int], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run manual targeted backward recovery on [start_time, end_time] and merge with existing payload."""
    new_payload = run_clip_tracking_recovery(
        video_path=video_path,
        coach_data=coach_data,
        cfg=cfg,
        movement_data=movement_data,
        model_path=model_path,
        estimator_factory=estimator_factory,
        target_range=(float(start_time), float(end_time)),
        enforce_budget=False,
        max_memory_bytes=max_memory_bytes,
        cancel_check=cancel_check,
        status_callback=status_callback,
        progress_callback=progress_callback,
        frame_provider=frame_provider,
    )
    existing = coach_data.get("pose_recovery")
    if isinstance(existing, Mapping):
        merged_frames = dict(existing.get("frames") or {})
        merged_frames.update(new_payload.get("frames") or {})
        by_id: dict[str, dict[str, Any]] = {
            str(w["window_id"]): dict(w)
            for w in existing.get("windows") or ()
            if isinstance(w, Mapping) and "window_id" in w
        }
        for w in new_payload.get("windows") or ():
            if isinstance(w, Mapping) and "window_id" in w:
                by_id[str(w["window_id"])] = dict(w)
        merged_windows = sorted(by_id.values(), key=lambda w: int(w.get("window_start_frame", 0)))
        recovered_frames = sum(
            1 for item in merged_frames.values() if isinstance(item, Mapping) and item.get("landmarks")
        )
        recovered_joints = sum(
            len(item.get("landmarks") or {})
            for item in merged_frames.values()
            if isinstance(item, Mapping)
        )
        interp_joints = sum(
            sum(
                1
                for lm in (item.get("landmarks") or {}).values()
                if isinstance(lm, Mapping) and lm.get("origin") == "interpolated"
            )
            for item in merged_frames.values()
            if isinstance(item, Mapping)
        )
        return {
            **new_payload,
            "windows": merged_windows,
            "frames": merged_frames,
            "summary": {
                **new_payload["summary"],
                "windows_detected": len(merged_windows),
                "recovered_frames": recovered_frames,
                "recovered_joints": recovered_joints,
                "interpolated_joints": interp_joints,
            },
        }
    return new_payload


def _assemble_payload(
    *,
    status: str,
    status_message: str,
    source_sha256: str | None,
    model_name: str | None,
    model_sha256: str | None,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    dancer_selection: dict[str, Any] | None,
    cfg: Config,
    windows: list[CandidateGapWindow],
    estimates: dict[int, RecoveredFrameEstimate],
    skipped_budget: int,
    attempted: int,
    recovered_windows: int,
    unresolved_windows: int,
) -> dict[str, Any]:
    frames_out = {
        str(fidx): est.to_dict()
        for fidx, est in sorted(estimates.items())
    }
    recovered_joints = sum(len(est.landmarks) for est in estimates.values())
    interp_joints = sum(
        sum(1 for org in est.origins.values() if org == "interpolated")
        for est in estimates.values()
    )
    recovered_frames = sum(1 for est in estimates.values() if est.landmarks)
    return {
        "schema_version": POSE_RECOVERY_SCHEMA_VERSION,
        "status": status,
        "status_message": status_message,
        "source_sha256": source_sha256,
        "model_name": model_name,
        "model_sha256": model_sha256,
        "source_width": int(width),
        "source_height": int(height),
        "video_fps": round(float(fps), 4),
        "frame_count": int(frame_count),
        "dancer_selection": dancer_selection,
        "settings": {
            "replay_recovery_max_gap_s": float(cfg.replay_recovery_max_gap_s),
            "replay_recovery_context_s": float(cfg.replay_recovery_context_s),
            "replay_pivot_interpolation_max_gap_s": float(cfg.replay_pivot_interpolation_max_gap_s),
        },
        "summary": {
            "windows_detected": len(windows),
            "windows_attempted": attempted,
            "windows_recovered": recovered_windows,
            "windows_unresolved": unresolved_windows,
            "windows_skipped_budget": skipped_budget,
            "recovered_frames": recovered_frames,
            "recovered_joints": recovered_joints,
            "interpolated_joints": interp_joints,
        },
        "windows": [w.to_dict() for w in windows],
        "frames": frames_out,
    }


def validate_pose_recovery_payload(
    payload: Any,
    *,
    frame_count: int | None = None,
    valid_frame_indices: set[int] | None = None,
    timestamps: list[float] | None = None,
    source_sha256: str | None = None,
    model_sha256: str | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
    dancer_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate optional `pose_recovery` schema, bounds, provenance, and freshness.

    Raises `ValueError` if corrupt or stale so callers can ignore it with a warning
    while keeping the base forward analytics usable.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("pose_recovery payload must be a mapping")
    if payload.get("schema_version") != POSE_RECOVERY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported pose_recovery schema_version: {payload.get('schema_version')!r}")
    status = payload.get("status")
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid pose_recovery status: {status!r}")

    # Freshness / provenance checks against current analytics metadata
    if source_sha256 and payload.get("source_sha256") and payload["source_sha256"] != source_sha256:
        raise ValueError("pose_recovery source_sha256 does not match recording")
    if model_sha256 and payload.get("model_sha256") and payload["model_sha256"] != model_sha256:
        raise ValueError("pose_recovery model_sha256 does not match pose model")
    if source_width and payload.get("source_width") and int(payload["source_width"]) != int(source_width):
        raise ValueError("pose_recovery source_width does not match video width")
    if source_height and payload.get("source_height") and int(payload["source_height"]) != int(source_height):
        raise ValueError("pose_recovery source_height does not match video height")
    if dancer_selection is not None and payload.get("dancer_selection") is not None:
        if payload.get("dancer_selection") != dict(dancer_selection):
            raise ValueError("pose_recovery dancer_selection does not match current subject assignment")

    payload_count = payload.get("frame_count")
    if frame_count is not None and isinstance(payload_count, int) and payload_count != frame_count:
        raise ValueError("pose_recovery frame_count does not match recording")

    max_valid_idx = (
        max(valid_frame_indices)
        if valid_frame_indices
        else (frame_count - 1 if frame_count is not None else None)
    )

    windows = payload.get("windows")
    if not isinstance(windows, list) or len(windows) > MAX_RECOVERY_WINDOWS:
        raise ValueError("Invalid pose_recovery windows list")
    for w in windows:
        if not isinstance(w, Mapping):
            raise ValueError("Invalid pose_recovery window entry")
        wsf = w.get("window_start_frame")
        wef = w.get("window_end_frame")
        if type(wsf) is not int or type(wef) is not int or wsf < 0 or wef < wsf:
            raise ValueError("Invalid pose_recovery window frame bounds")
        if max_valid_idx is not None and wef > max_valid_idx:
            raise ValueError("pose_recovery window frame exceeds clip length")

    frames = payload.get("frames")
    if not isinstance(frames, Mapping) or len(frames) > MAX_RECOVERY_FRAMES:
        raise ValueError("Invalid pose_recovery frames map")
    for raw_key in frames:
        if not str(raw_key).isdigit():
            raise ValueError(f"Invalid pose_recovery frame key: {raw_key!r}")

    ordered_valid = sorted(valid_frame_indices) if valid_frame_indices else None
    ts_by_idx: dict[int, float] | None = None
    if timestamps is not None:
        if ordered_valid is not None and len(ordered_valid) == len(timestamps):
            ts_by_idx = {fidx: float(ts) for fidx, ts in zip(ordered_valid, timestamps)}
        else:
            ts_by_idx = {idx: float(ts) for idx, ts in enumerate(timestamps)}

    prev_t = -float("inf")
    for raw_key in sorted(frames, key=int):
        idx = int(raw_key)
        entry = frames[raw_key]
        if not isinstance(entry, Mapping):
            raise ValueError("Invalid pose_recovery frame entry")
        if idx < 0 or (
            valid_frame_indices is not None and idx not in valid_frame_indices
        ) or (valid_frame_indices is None and frame_count is not None and idx >= frame_count):
            raise ValueError("pose_recovery frame index out of bounds")
        st = entry.get("source_time")
        ct = entry.get("clip_time")
        if not isinstance(st, int | float) or not math.isfinite(float(st)):
            raise ValueError("Invalid pose_recovery source_time")
        if not isinstance(ct, int | float) or not math.isfinite(float(ct)) or float(ct) < -1e-6:
            raise ValueError("Invalid pose_recovery clip_time")
        if float(st) <= prev_t:
            raise ValueError("Non-monotonic timestamps in pose_recovery payload")
        prev_t = float(st)
        if ts_by_idx is not None and idx in ts_by_idx:
            expected_t = ts_by_idx[idx]
            if abs(float(st) - expected_t) > 0.08 and abs(float(ct) - expected_t) > 0.08:
                raise ValueError("pose_recovery frame timestamps disagree with recording timeline")
        landmarks = entry.get("landmarks")
        if not isinstance(landmarks, Mapping):
            raise ValueError("Invalid pose_recovery landmarks")
        for jname, lm in landmarks.items():
            if not isinstance(lm, Mapping):
                raise ValueError("Invalid pose_recovery landmark record")
            origin = lm.get("origin")
            if origin not in VALID_ORIGINS:
                raise ValueError(f"Invalid landmark origin in pose_recovery: {origin!r}")
            for coord_k in ("x", "y", "visibility", "presence"):
                val = lm.get(coord_k)
                if not isinstance(val, int | float) or not math.isfinite(float(val)):
                    raise ValueError(f"Non-finite coordinate {coord_k} on {jname}")
            if not (0.0 <= float(lm["x"]) <= 1.0 and 0.0 <= float(lm["y"]) <= 1.0):
                raise ValueError(f"Out-of-bounds coordinate on {jname}")

    return dict(payload)


def rebase_pose_recovery(
    payload: Mapping[str, Any] | None,
    start_frame: int,
    end_frame: int | None = None,
    *,
    new_source_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Rebase a `pose_recovery` payload onto an excerpt `[start_frame, end_frame)`."""
    if payload is None or not isinstance(payload, Mapping):
        return None
    start = max(0, int(start_frame))
    raw_frames = payload.get("frames") if isinstance(payload.get("frames"), Mapping) else {}
    rebased_frames: dict[str, Any] = {}
    for raw_key, entry in raw_frames.items():
        if not str(raw_key).isdigit() or not isinstance(entry, Mapping):
            continue
        old_idx = int(raw_key)
        if old_idx < start or (end_frame is not None and old_idx >= end_frame):
            continue
        new_idx = old_idx - start
        rebased_entry = dict(entry)
        rebased_entry["frame_index"] = new_idx
        rebased_frames[str(new_idx)] = rebased_entry

    raw_windows = payload.get("windows") if isinstance(payload.get("windows"), list) else []
    rebased_windows: list[dict[str, Any]] = []
    max_new_frame = (end_frame - start - 1) if end_frame is not None else None
    for win in raw_windows:
        if not isinstance(win, Mapping):
            continue
        wsf = win.get("window_start_frame")
        wef = win.get("window_end_frame")
        gsf = win.get("gap_start_frame", wsf)
        gef = win.get("gap_end_frame", wef)
        if not isinstance(wsf, int) or not isinstance(wef, int):
            continue
        if wef < start or (end_frame is not None and wsf >= end_frame):
            continue
        new_wsf = max(0, wsf - start)
        new_wef = wef - start
        new_gsf = max(0, int(gsf) - start)
        new_gef = int(gef) - start
        if max_new_frame is not None:
            new_wef = min(max_new_frame, new_wef)
            new_gsf = min(max_new_frame, new_gsf)
            new_gef = min(max_new_frame, new_gef)
        if new_wef < new_wsf:
            continue
        rebased_win = dict(win)
        rebased_win["window_start_frame"] = new_wsf
        rebased_win["window_end_frame"] = new_wef
        rebased_win["gap_start_frame"] = min(max(new_gsf, new_wsf), new_wef)
        rebased_win["gap_end_frame"] = max(min(new_gef, new_wef), rebased_win["gap_start_frame"])
        rebased_windows.append(rebased_win)

    orig_count = payload.get("frame_count")
    if end_frame is not None:
        new_count = max(0, int(end_frame) - start)
    elif isinstance(orig_count, int):
        new_count = max(0, orig_count - start)
    else:
        new_count = None

    recovered_frames = sum(
        1 for entry in rebased_frames.values() if isinstance(entry, Mapping) and entry.get("landmarks")
    )
    recovered_joints = sum(
        len(entry.get("landmarks") or {})
        for entry in rebased_frames.values()
        if isinstance(entry, Mapping)
    )
    interpolated_joints = sum(
        sum(
            1
            for lm in (entry.get("landmarks") or {}).values()
            if isinstance(lm, Mapping) and lm.get("origin") == "interpolated"
        )
        for entry in rebased_frames.values()
        if isinstance(entry, Mapping)
    )

    out = dict(payload)
    if new_source_sha256:
        out["source_sha256"] = new_source_sha256
    if new_count is not None:
        out["frame_count"] = new_count
    out["windows"] = rebased_windows
    out["frames"] = rebased_frames
    summary = dict(payload.get("summary") or {}) if isinstance(payload.get("summary"), Mapping) else {}
    summary.update(
        {
            "windows_detected": len(rebased_windows),
            "recovered_frames": recovered_frames,
            "recovered_joints": recovered_joints,
            "interpolated_joints": interpolated_joints,
        }
    )
    out["summary"] = summary
    return out

