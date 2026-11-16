"""Per-frame coach evidence, posture annotations and cached rendering."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from ..calibration.measurements import measure_targets
from ..calibration.models import (
    METRIC_DETAILS,
    ApprovedRange,
    RangeProfile,
    profile_from_calibration,
)
from ..calibration.normalization import (
    HoldSourceIdentity,
    MeasurementNormalizationConfig,
    PairApplicability,
    _matched_rule_passes_current_gates,
    capture_identity_dict,
    evaluate_pair_applicability,
    extract_coach_observations,
    frames_fingerprint,
    normalize_metric_observation,
    settings_fingerprint,
)
from ..config import Config, RangeDefault
from ..domain import (
    Calibration,
    FrameObservation,
    Landmark,
)
from ..geometry import (
    knee_geometry_kwargs,
    shoulder_geometry_kwargs,
    torso_axis_alignment,
)
from ..overlay import (
    MESSAGES,
    draw,
    draw_head_alignment,
    draw_head_posture,
    draw_styled_bone,
    draw_torso_axis,
    edge_provenance_style,
    screen_point,
)
from ..pose.constants import CONNECTIONS
from ..pose.interpolation import interpolate_poses
from ..pose.recovery import LOWER_LIMB_JOINTS, validate_pose_recovery_payload
from ..quality import reliable
from ..temporal import Evaluator
from .constants import SOUNDTRACK_SOURCE_LABEL

logger = logging.getLogger(__name__)


def range_feedback(
    rows: list[dict[str, Any]], profile: RangeProfile,
) -> tuple[str, list[str]]:
    return target_feedback(rows, profile.metrics)


def target_feedback(
    rows: list[dict[str, Any]], targets: Mapping[str, RangeDefault | ApprovedRange],
) -> tuple[str, list[str]]:
    messages = []
    lines = []
    for row in rows:
        key = row["metric"]
        target = targets[key]
        label = METRIC_DETAILS[key][1]
        value = row.get("smoothed")
        measured = "Unknown" if value is None else f"{value:.3g}"
        limits = f"{target.lower:.3g} .. {target.upper:.3g} {target.unit}"
        reason = f" | {row['reason']}" if row.get("reason") else ""
        lines.append(f"{label}: {measured} | approved {limits} | {row['state']}{reason}")
        if row["state"] == "DEVIATION":
            messages.append(f"{label}: {measured} outside approved range {limits}")
    return "; ".join(messages), lines


def _coach_observation(
    entry: dict[str, Any], width: int, height: int, *, require_measurements: bool = False,
    use_display_poses: bool = False,
) -> FrameObservation:
    if require_measurements and "measurement_poses" not in entry:
        raise ValueError("This cache has display landmarks only. Analyze posture again before comparing.")
    raw_poses = entry.get("measurement_poses", entry.get("poses"))
    if use_display_poses and not require_measurements and "display_poses" in entry:
        raw_poses = entry["display_poses"]
    if require_measurements and not isinstance(raw_poses, list):
        raise TypeError("Invalid cached measurement poses. Analyze posture again.")
    if not isinstance(raw_poses, list):
        raw_poses = [entry.get("pose")]
    poses = []
    for raw_pose in raw_poses:
        if require_measurements and not isinstance(raw_pose, dict):
            raise TypeError("Invalid cached measurement pose. Analyze posture again.")
        if not isinstance(raw_pose, dict) or not raw_pose:
            continue
        pose = {}
        for name, coords in raw_pose.items():
            if require_measurements and (
                not isinstance(coords, list | tuple) or len(coords) < 3
            ):
                raise ValueError("Invalid cached measurement landmark. Analyze posture again.")
            if isinstance(coords, list | tuple) and len(coords) >= 3:
                pose[str(name)] = Landmark(
                    x=float(coords[0]), y=float(coords[1]),
                    z=float(coords[2]) if coords[2] is not None else None,
                    visibility=float(coords[3]) if len(coords) >= 4 and coords[3] is not None else None,
                    presence=float(coords[4]) if len(coords) >= 5 and coords[4] is not None else None,
                )
        if pose:
            poses.append(pose)
    return FrameObservation(float(entry.get("timestamp", 0.0)), width, height, poses)


def _data_media_sha256(data: Mapping[str, Any]) -> str | None:
    provenance = data.get("source_provenance")
    for container in (data, provenance):
        if isinstance(container, Mapping):
            for key in ("raw_media_sha256", "media_sha256", "video_sha256"):
                val = container.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return None


def _build_coach_practice_identity(
    data: Mapping[str, Any],
    comparison_cfg: Config,
    practice_identity: HoldSourceIdentity | None,
) -> HoldSourceIdentity:
    observations = extract_coach_observations(data)
    computed_frames_fp = frames_fingerprint(observations) if observations else None
    computed_settings_fp = settings_fingerprint(comparison_cfg)
    computed_capture = capture_identity_dict(
        data.get("capture_profile") if isinstance(data.get("capture_profile"), Mapping) else {}
    )
    computed_media = _data_media_sha256(data)
    if practice_identity is None:
        raw_seg = data.get("segment_index")
        return HoldSourceIdentity(
            archive_path=str(data["archive_path"]) if data.get("archive_path") else None,
            archive_name=str(data["archive_name"]) if data.get("archive_name") else None,
            segment_index=int(raw_seg) if isinstance(raw_seg, int) and raw_seg >= 0 else 0,
            raw_media_sha256=computed_media,
            media_sha256=computed_media,
            capture_identity=computed_capture,
            settings_fingerprint=computed_settings_fp,
            frames_fingerprint=computed_frames_fp,
        )
    return practice_identity.model_copy(update={
        "raw_media_sha256": practice_identity.raw_media_sha256 or computed_media,
        "media_sha256": practice_identity.media_sha256 or practice_identity.raw_media_sha256 or computed_media,
        "capture_identity": practice_identity.capture_identity or computed_capture,
        "settings_fingerprint": practice_identity.settings_fingerprint or computed_settings_fp,
        "frames_fingerprint": practice_identity.frames_fingerprint or computed_frames_fp,
    })


def compare_coach_data(
    data: dict[str, Any],
    cfg: Config,
    reference: Calibration,
    *,
    practice_identity: HoldSourceIdentity | None = None,
    bypass_normalization: bool = False,
) -> dict[str, Any]:
    """Compare raw cached landmarks with approved ranges, without rerunning inference."""
    profile = profile_from_calibration(reference)
    if profile is None:
        raise ValueError("This legacy calibration has no approved ranges. Select a range profile.")
    if data.get("model_hash") != profile.model_hash:
        raise ValueError("Calibration and recording use different measurement models. Reanalyze the recording.")
    capture = data.get("capture_profile")
    if not isinstance(capture, dict) or any(
        capture.get(key) != profile.capture_profile[key] for key in ("side", "facing")
    ):
        raise ValueError("Recording side/facing does not match the selected calibration.")
    comparison_cfg = cfg.model_copy(update={
        "side": profile.capture_profile["side"], "facing": profile.capture_profile["facing"],
    })
    identity = _build_coach_practice_identity(data, comparison_cfg, practice_identity)
    applicability = evaluate_pair_applicability(comparison_cfg, profile, identity)
    if bypass_normalization:
        applicability = PairApplicability(
            False,
            "disabled",
            "Using original calibration scale for this comparison.",
            applicability.pair,
        )
    return _compare_targets(
        data,
        comparison_cfg,
        profile.metrics,
        profile.name,
        {"calibration_id": profile.id, "calibration_name": profile.name},
        applicability=applicability,
        norm_cfg=comparison_cfg.measurement_normalization,
        bypass_normalization=bypass_normalization,
    )


def compare_default_coach_data(data: dict[str, Any], cfg: Config) -> dict[str, Any]:
    capture = data.get("capture_profile")
    if not isinstance(capture, dict) or any(
        capture.get(key) not in ("left", "right") for key in ("side", "facing")
    ):
        raise ValueError("Cached recording side/facing is missing. Analyze posture again.")
    cfg = cfg.model_copy(update={key: capture[key] for key in ("side", "facing")})
    return _compare_targets(data, cfg, cfg.calibration_defaults, "Settings default ranges", {
        "calibration_name": "Settings default ranges",
    })


def compare_approved_coach_data(
    data: dict[str, Any], cfg: Config, targets: Mapping[str, ApprovedRange],
) -> dict[str, Any]:
    """Evaluate in-progress approved ranges against cached raw landmarks."""
    return _compare_targets(data, cfg, targets, "Approved calibration", {})


def _summarize_comparison_normalization(
    targets: Mapping[str, RangeDefault | ApprovedRange],
    applicability: PairApplicability,
    norm_cfg: MeasurementNormalizationConfig,
    *,
    bypass_normalization: bool,
) -> dict[str, Any]:
    metrics_meta: dict[str, dict[str, Any]] = {}
    pair = applicability.pair
    for key in targets:
        rule = pair.metrics.get(key) if pair is not None else None
        if applicability.status in ("no_pair", "disabled"):
            metric_status = "direct_comparison" if (pair is not None and rule is not None and not rule.enabled) else "unadjusted"
            detail = applicability.reason
        elif applicability.status == "needs_recheck":
            metric_status = "needs_recheck"
            detail = applicability.reason
        elif rule is None or not rule.enabled:
            metric_status = "direct_comparison"
            detail = ""
        elif rule.method == "matched_holds":
            ok, detail = _matched_rule_passes_current_gates(rule, norm_cfg)
            metric_status = "applied" if ok else "needs_recheck"
        else:
            metric_status = "applied"
            detail = rule.status_message
        metrics_meta[key] = {
            "status": metric_status,
            "detail": detail,
            "method": rule.method if rule is not None else None,
            "offset": rule.offset if (rule is not None and metric_status == "applied") else None,
            "verified": rule.verified if rule is not None else False,
            "verification_error": rule.verification_error if rule is not None else None,
            "show_calculation_in_review": rule.show_calculation_in_review if rule is not None else False,
        }
    if applicability.status == "needs_recheck" or any(
        m["status"] == "needs_recheck" for m in metrics_meta.values()
    ):
        overall_status = "needs_recheck"
        first_detail = next((m["detail"] for m in metrics_meta.values() if m["status"] == "needs_recheck" and m["detail"]), "")
        reason = applicability.reason or first_detail or "Normalization needs recheck before applying."
    elif applicability.status == "applied" and any(m["status"] == "applied" for m in metrics_meta.values()):
        overall_status = "applied"
        reason = ""
    else:
        overall_status = applicability.status
        reason = applicability.reason
    return {
        "enabled": bool(norm_cfg.enabled),
        "status": overall_status,
        "reason": reason,
        "rule_id": pair.id if pair is not None else None,
        "pair_name": pair.name if pair is not None else "",
        "bypass_normalization": bypass_normalization,
        "metrics": metrics_meta,
    }


def _compare_targets(
    data: dict[str, Any], cfg: Config, targets: Mapping[str, RangeDefault | ApprovedRange],
    name: str, metadata: dict[str, Any],
    *,
    applicability: PairApplicability | None = None,
    norm_cfg: MeasurementNormalizationConfig | None = None,
    bypass_normalization: bool = False,
) -> dict[str, Any]:
    frames = data.get("frames")
    if not isinstance(frames, dict):
        raise TypeError("No cached posture frames. Analyze posture before comparing.")
    if not frames:
        raise ValueError("No cached posture frames. Analyze posture before comparing.")
    width, height = data.get("width"), data.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("Cached frame dimensions are missing or invalid. Analyze posture again.")
    evaluators = {
        key: Evaluator(key, 0, cfg, bounds=(target.lower, target.upper))
        for key, target in targets.items()
    }
    compared = {}
    rule_map = applicability.pair.metrics if (applicability is not None and applicability.pair is not None) else {}
    rule_id = applicability.pair.id if (applicability is not None and applicability.pair is not None) else None
    for index in sorted(frames, key=int):
        entry = frames[index]
        if not isinstance(entry, dict):
            raise TypeError("Invalid cached posture frame")
        observation = _coach_observation(entry, width, height, require_measurements=True)
        if entry.get("interpolated") is True:
            observation = FrameObservation(observation.timestamp, observation.width, observation.height, [])
        measured = measure_targets(observation, cfg, targets)
        rows = []
        for key, evaluator in evaluators.items():
            raw_val = measured.values[key]
            q_valid = measured.quality[key].valid
            if applicability is not None and norm_cfg is not None:
                metric_rule = rule_map.get(key)
                norm = normalize_metric_observation(
                    metric=key,
                    observed_raw=raw_val,
                    quality_valid=q_valid,
                    rule=metric_rule,
                    applicability=applicability,
                    norm_cfg=norm_cfg,
                )
                row = {
                    "metric": key,
                    "unit": targets[key].unit,
                    "reason": measured.quality[key].reason,
                    **evaluator.update(observation.timestamp, norm.evaluator_input),
                    "observed_raw": norm.observed_raw,
                    "normalization_offset": norm.normalization_offset,
                    "reference_scale_raw": norm.reference_scale_raw,
                    "normalization_status": norm.normalization_status,
                    "normalization_detail": norm.status_detail,
                    "rule_id": rule_id,
                    "normalization_method": metric_rule.method if metric_rule is not None else None,
                    "verification_error": metric_rule.verification_error if metric_rule is not None else None,
                    "show_calculation_in_review": (
                        metric_rule.show_calculation_in_review if metric_rule is not None else False
                    ),
                }
            else:
                row = {
                    "metric": key,
                    "unit": targets[key].unit,
                    "reason": measured.quality[key].reason,
                    **evaluator.update(observation.timestamp, raw_val if q_valid else None),
                }
            rows.append(row)
        message, lines = target_feedback(rows, targets)
        compared[index] = {
            **entry, "message": message, "lines": lines,
            "notes": [name, *lines],
            "calibration_rows": rows,
        }
    extra_meta = dict(metadata)
    if applicability is not None and norm_cfg is not None:
        extra_meta["normalization"] = _summarize_comparison_normalization(
            targets, applicability, norm_cfg, bypass_normalization=bypass_normalization,
        )
    return {
        **data, "frames": compared, "settings_coaching": False,
        **extra_meta,
    }


def _deserialise_pose_list(raw_poses: Any) -> list[dict[str, Landmark]]:
    if not isinstance(raw_poses, list):
        return []
    poses: list[dict[str, Landmark]] = []
    for raw_pose in raw_poses:
        if not isinstance(raw_pose, dict) or not raw_pose:
            continue
        pose: dict[str, Landmark] = {}
        for name, coords in raw_pose.items():
            if isinstance(coords, list | tuple) and len(coords) >= 3:
                pose[str(name)] = Landmark(
                    x=float(coords[0]),
                    y=float(coords[1]),
                    z=float(coords[2]) if coords[2] is not None else None,
                    visibility=float(coords[3]) if len(coords) >= 4 and coords[3] is not None else None,
                    presence=float(coords[4]) if len(coords) >= 5 and coords[4] is not None else None,
                )
        if pose:
            poses.append(pose)
    return poses


def _pivot_intervals_from_metadata(
    indices: list[str],
    frames: Mapping[str, Any],
    movement_data: Mapping[str, Any] | None,
    recovery: Mapping[str, Any] | None,
) -> list[tuple[float, float]]:
    time_by_frame = {
        int(idx): float(frames[idx].get("timestamp", int(idx) * 0.1))
        for idx in indices
        if str(idx).isdigit()
    }
    intervals: list[tuple[float, float]] = []
    if isinstance(movement_data, Mapping):
        for pivot in movement_data.get("pivots") or ():
            if not isinstance(pivot, Mapping):
                continue
            st = pivot.get("start_time")
            et = pivot.get("end_time")
            if isinstance(st, int | float) and isinstance(et, int | float) and float(et) > float(st):
                intervals.append((float(st), float(et)))
                continue
            sf = pivot.get("start_frame")
            ef = pivot.get("end_frame")
            if isinstance(sf, int) and isinstance(ef, int):
                t0 = time_by_frame.get(sf)
                t1 = time_by_frame.get(max(sf, ef - 1))
                if t0 is not None and t1 is not None:
                    intervals.append((t0, max(t0, t1)))
    if isinstance(recovery, Mapping):
        for win in recovery.get("windows") or ():
            if not isinstance(win, Mapping) or not win.get("intersects_pivot"):
                continue
            st = win.get("window_start_time", win.get("gap_start_time"))
            et = win.get("window_end_time", win.get("gap_end_time"))
            if isinstance(st, int | float) and isinstance(et, int | float):
                intervals.append((float(st), float(et)))
    return intervals


_TORSO_JOINTS: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)
_KINEMATIC_PARENT: dict[str, str] = {
    "nose": "left_shoulder",
    "left_ear": "left_shoulder",
    "right_ear": "right_shoulder",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "left_heel": "left_ankle",
    "left_foot_index": "left_ankle",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
    "right_heel": "right_ankle",
    "right_foot_index": "right_ankle",
}
_LIMB_JOINTS: frozenset[str] = frozenset(
    {
        "left_elbow",
        "left_wrist",
        "right_elbow",
        "right_wrist",
        "left_knee",
        "left_ankle",
        "left_heel",
        "left_foot_index",
        "right_knee",
        "right_ankle",
        "right_heel",
        "right_foot_index",
    }
)
_CHILD_JOINT: dict[str, str] = {
    "left_elbow": "left_wrist",
    "right_elbow": "right_wrist",
    "left_knee": "left_ankle",
    "right_knee": "right_ankle",
}
_OPPOSITE_JOINT: dict[str, str] = {
    "left_shoulder": "right_shoulder",
    "right_shoulder": "left_shoulder",
    "left_hip": "right_hip",
    "right_hip": "left_hip",
    "left_ear": "right_ear",
    "right_ear": "left_ear",
    "left_elbow": "right_elbow",
    "right_elbow": "left_elbow",
    "left_wrist": "right_wrist",
    "right_wrist": "left_wrist",
    "left_knee": "right_knee",
    "right_knee": "left_knee",
    "left_ankle": "right_ankle",
    "right_ankle": "left_ankle",
    "left_heel": "right_heel",
    "right_heel": "left_heel",
    "left_foot_index": "right_foot_index",
    "right_foot_index": "left_foot_index",
}
_CANONICAL_TORSO_RATIOS: dict[str, float] = {
    "nose": 0.35,
    "left_ear": 0.32,
    "right_ear": 0.32,
    "left_elbow": 0.55,
    "right_elbow": 0.55,
    "left_wrist": 0.50,
    "right_wrist": 0.50,
    "left_knee": 0.85,
    "right_knee": 0.85,
    "left_ankle": 0.82,
    "right_ankle": 0.82,
    "left_heel": 0.12,
    "right_heel": 0.12,
    "left_foot_index": 0.18,
    "right_foot_index": 0.18,
}
_TOPO_JOINTS: tuple[str, ...] = (
    *_TORSO_JOINTS,
    "nose",
    "left_ear",
    "right_ear",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


def _landmark_in_frame(lm: Landmark | None) -> bool:
    return (
        lm is not None
        and math.isfinite(lm.x)
        and math.isfinite(lm.y)
        and 0.0 <= lm.x <= 1.0
        and 0.0 <= lm.y <= 1.0
    )


def _landmark_conf(lm: Landmark) -> float:
    vis = lm.visibility if lm.visibility is not None else 1.0
    pres = lm.presence if lm.presence is not None else 1.0
    return float(min(vis, pres))


def _pose_torso_center(
    pose: Mapping[str, Landmark], confidence: float,
) -> tuple[float, float] | None:
    hips = [
        lm for j in ("left_hip", "right_hip")
        if (lm := pose.get(j)) is not None and reliable(lm, confidence) and _landmark_in_frame(lm)
    ]
    pts = hips or [
        lm for j in _TORSO_JOINTS
        if (lm := pose.get(j)) is not None and reliable(lm, confidence) and _landmark_in_frame(lm)
    ]
    if not pts:
        return None
    return (
        sum(p.x for p in pts) / len(pts),
        sum(p.y for p in pts) / len(pts),
    )


def _filter_true_cut_positions(
    cut_set: set[int],
    indices: list[str],
    working: Sequence[FrameObservation],
    cfg: Config,
) -> set[int]:
    """Keep only shot cuts with a large spatial jump or gap exceeding `replay_joint_max_gap`."""
    if not cut_set:
        return set()
    centers = [
        _pose_torso_center(obs.poses[0], cfg.confidence) if len(obs.poses) == 1 else None
        for obs in working
    ]
    n = len(working)
    true_cuts: set[int] = set()
    for pos, idx_str in enumerate(indices):
        if pos <= 0 or int(idx_str) not in cut_set:
            continue
        prev_p = next((i for i in range(pos - 1, -1, -1) if centers[i] is not None), None)
        next_p = next((i for i in range(pos, n) if centers[i] is not None), None)
        if prev_p is None or next_p is None:
            continue
        c_prev = centers[prev_p]
        c_next = centers[next_p]
        assert c_prev is not None and c_next is not None
        jump = math.hypot(c_next[0] - c_prev[0], c_next[1] - c_prev[1])
        dt = abs(working[next_p].timestamp - working[prev_p].timestamp)
        if jump > 0.35 or dt > cfg.replay_joint_max_gap:
            true_cuts.add(pos)
    return true_cuts


def _display_segment_ids(
    frames: Sequence[FrameObservation],
    cut_positions: set[int],
    track_keys: Sequence[str],
) -> list[int]:
    """Assign contiguous shot-segment IDs matching `interpolate_poses` boundaries."""
    seg_ids: list[int] = []
    cur = 0
    prev_ts: float | None = None
    prev_track: str | None = None
    for idx, frame in enumerate(frames):
        if len(frame.poses) > 1 or not math.isfinite(frame.timestamp):
            cur += 1
            seg_ids.append(-1)
            prev_ts = None
            prev_track = None
            continue
        tk = track_keys[idx] if idx < len(track_keys) else "single"
        if idx > 0 and (
            idx in cut_positions
            or (prev_ts is not None and frame.timestamp <= prev_ts)
            or (prev_track is not None and tk != prev_track)
        ):
            cur += 1
        seg_ids.append(cur)
        prev_ts = frame.timestamp
        prev_track = tk
    return seg_ids


def _compute_reference_bone_lengths(
    frames: Sequence[FrameObservation],
    observed_joints: set[str],
    confidence: float,
) -> dict[str, float]:
    """Estimate median normalized bone lengths across frames for display completion."""
    torso_lengths: list[float] = []
    for frame in frames:
        if len(frame.poses) != 1:
            continue
        pose = frame.poses[0]
        for side in ("left", "right"):
            sh = pose.get(f"{side}_shoulder")
            hp = pose.get(f"{side}_hip")
            if reliable(sh, confidence) and reliable(hp, confidence):
                assert sh is not None and hp is not None
                torso_lengths.append(math.hypot(sh.x - hp.x, sh.y - hp.y))
    med_torso = float(np.median(torso_lengths)) if torso_lengths else 0.22

    ref_len: dict[str, float] = {}
    for joint, parent in _KINEMATIC_PARENT.items():
        if joint not in observed_joints:
            continue
        lengths: list[float] = []
        for frame in frames:
            if len(frame.poses) != 1:
                continue
            pose = frame.poses[0]
            jl = pose.get(joint)
            pl = pose.get(parent)
            if reliable(jl, confidence) and reliable(pl, confidence):
                assert jl is not None and pl is not None
                lengths.append(math.hypot(jl.x - pl.x, jl.y - pl.y))
        if lengths:
            ref_len[joint] = max(0.02, float(np.median(lengths)))

    for joint in _KINEMATIC_PARENT:
        if joint not in observed_joints or joint in ref_len:
            continue
        opp = _OPPOSITE_JOINT.get(joint)
        if opp is not None and opp in ref_len:
            ref_len[joint] = ref_len[opp]
        else:
            ratio = _CANONICAL_TORSO_RATIOS.get(joint)
            ref_len[joint] = max(0.02, med_torso * ratio) if ratio is not None else 0.15
    return ref_len


def _resolve_raw_child_point(
    raw_pose: Mapping[str, Landmark],
    pose: Mapping[str, Landmark],
    joint: str,
    parent_lm: Landmark,
    rlen: float,
    ref_len: Mapping[str, float],
    confidence: float,
) -> tuple[float, float] | None:
    """Validate or anatomically clamp a low-confidence raw landmark relative to its parent."""
    raw_lm = raw_pose.get(joint)
    if not _landmark_in_frame(raw_lm):
        return None
    assert raw_lm is not None
    vx = raw_lm.x - parent_lm.x
    vy = raw_lm.y - parent_lm.y
    d = math.hypot(vx, vy)
    if 0.35 * rlen <= d <= 1.65 * rlen:
        tx, ty = float(raw_lm.x), float(raw_lm.y)
    elif d > 1e-4:
        clamped_d = min(max(d, 0.50 * rlen), 1.35 * rlen)
        tx = parent_lm.x + vx * (clamped_d / d)
        ty = parent_lm.y + vy * (clamped_d / d)
    else:
        return None

    child_name = _CHILD_JOINT.get(joint)
    if child_name is not None:
        disp_child = pose.get(child_name)
        raw_child = raw_pose.get(child_name)
        child_lm = (
            disp_child
            if reliable(disp_child, confidence)
            else raw_child
            if _landmark_in_frame(raw_child)
            else None
        )
        if child_lm is not None:
            child_rlen = ref_len.get(child_name, rlen)
            span = math.hypot(child_lm.x - parent_lm.x, child_lm.y - parent_lm.y)
            if span <= 2.6 * (rlen + child_rlen):
                mid_x = 0.5 * (parent_lm.x + child_lm.x)
                mid_y = 0.5 * (parent_lm.y + child_lm.y)
                tx = 0.65 * tx + 0.35 * mid_x
                ty = 0.65 * ty + 0.35 * mid_y
    return (tx, ty)


def _nearest_anchor_pos(
    pos: int,
    seg_ids: Sequence[int],
    frames: Sequence[FrameObservation],
    max_gap: float,
    required_joints: tuple[str, ...],
    confidence: float,
) -> int | None:
    seg = seg_ids[pos]
    if seg < 0:
        return None
    pos_t = frames[pos].timestamp
    best_pos: int | None = None
    best_dt = float("inf")
    for offset in range(1, len(frames)):
        matched_side = False
        for cand in (pos - offset, pos + offset):
            if not (0 <= cand < len(frames)) or seg_ids[cand] != seg:
                continue
            dt = abs(frames[cand].timestamp - pos_t)
            if dt > max_gap + 1e-9:
                continue
            matched_side = True
            if (
                len(frames[cand].poses) == 1
                and dt < best_dt
                and all(reliable(frames[cand].poses[0].get(j), confidence) for j in required_joints)
            ):
                best_dt = dt
                best_pos = cand
        if best_pos is not None:
            return best_pos
        if not matched_side and best_pos is None:
            left_oob = pos - offset < 0 or seg_ids[pos - offset] != seg or abs(frames[pos - offset].timestamp - pos_t) > max_gap + 1e-9
            right_oob = pos + offset >= len(frames) or seg_ids[pos + offset] != seg or abs(frames[pos + offset].timestamp - pos_t) > max_gap + 1e-9
            if left_oob and right_oob:
                break
    return best_pos


def _complete_occluded_limbs_for_display(
    interpolated: list[FrameObservation],
    observations: Sequence[FrameObservation],
    origins_out: list[dict[str, str]],
    cut_positions: set[int],
    track_keys: Sequence[str],
    cfg: Config,
) -> None:
    """Complete unresolved limbs for display using low-conf guidance and parent kinematics."""
    if cfg.replay_joint_max_gap <= 0 or not interpolated:
        return

    observed_joints = {
        name for obs in observations for pose in obs.poses for name in pose
    }
    if not observed_joints:
        return

    n = len(interpolated)
    confidence = float(cfg.confidence)
    max_gap = float(cfg.replay_joint_max_gap)
    seg_ids = _display_segment_ids(interpolated, cut_positions, track_keys)
    ref_len = _compute_reference_bone_lengths(interpolated, observed_joints, confidence)
    modified_for_smooth: set[tuple[int, str]] = set()
    pose_buffers: list[dict[str, Landmark] | None] = [
        dict(frame.poses[0]) if len(frame.poses) == 1 else None
        for frame in interpolated
    ]
    changed_frames: set[int] = set()

    # Pass A: blend bracketed interpolated limb joints with anatomically plausible raw landmarks.
    for pos in range(n):
        disp_pose = pose_buffers[pos]
        if disp_pose is None or len(observations[pos].poses) != 1:
            continue
        raw_pose = observations[pos].poses[0]
        for joint in _TOPO_JOINTS:
            if joint not in observed_joints or joint not in _LIMB_JOINTS:
                continue
            if origins_out[pos].get(joint) != "interpolated":
                continue
            parent = _KINEMATIC_PARENT.get(joint)
            if parent is None:
                continue
            parent_lm = disp_pose.get(parent)
            cur_lm = disp_pose.get(joint)
            raw_lm = raw_pose.get(joint)
            if (
                not reliable(parent_lm, confidence)
                or cur_lm is None
                or not _landmark_in_frame(raw_lm)
            ):
                continue
            assert parent_lm is not None and raw_lm is not None
            raw_conf = _landmark_conf(raw_lm)
            if raw_conf <= 0.0:
                continue
            rlen = ref_len[joint]
            d = math.hypot(raw_lm.x - parent_lm.x, raw_lm.y - parent_lm.y)
            d_interp = math.hypot(raw_lm.x - cur_lm.x, raw_lm.y - cur_lm.y)
            if 0.45 * rlen <= d <= 1.50 * rlen and d_interp <= 0.22:
                w = min(0.75, max(0.25, raw_conf))
                disp_pose[joint] = Landmark(
                    x=float(cur_lm.x + w * (raw_lm.x - cur_lm.x)),
                    y=float(cur_lm.y + w * (raw_lm.y - cur_lm.y)),
                    visibility=1.0,
                    presence=1.0,
                )
                modified_for_smooth.add((pos, joint))
                changed_frames.add(pos)

    # Flush Pass A into interpolated so Pass B can anchor on Pass A refined positions if needed.
    for pos in changed_frames:
        buf = pose_buffers[pos]
        if buf is not None:
            interpolated[pos] = replace(interpolated[pos], poses=[buf])

    # Pass B: complete still-missing/unreliable joints in topological order.
    for joint in _TOPO_JOINTS:
        if joint not in observed_joints:
            continue
        is_torso = joint in _TORSO_JOINTS
        parent = _KINEMATIC_PARENT.get(joint)
        for pos in range(n):
            disp_pose = pose_buffers[pos]
            if disp_pose is None or reliable(disp_pose.get(joint), confidence):
                continue
            raw_pose = observations[pos].poses[0] if len(observations[pos].poses) == 1 else {}
            pt: tuple[float, float] | None = None
            if is_torso:
                raw_lm = raw_pose.get(joint)
                if _landmark_in_frame(raw_lm):
                    assert raw_lm is not None
                    pt = (float(raw_lm.x), float(raw_lm.y))
                else:
                    anchor = _nearest_anchor_pos(
                        pos,
                        seg_ids,
                        interpolated,
                        max_gap,
                        (joint,),
                        confidence,
                    )
                    if anchor is not None:
                        anchor_pose = interpolated[anchor].poses[0]
                        shared = [
                            t for t in _TORSO_JOINTS
                            if t != joint and reliable(disp_pose.get(t), confidence) and reliable(anchor_pose.get(t), confidence)
                        ]
                        if shared:
                            dx = sum(disp_pose[t].x - anchor_pose[t].x for t in shared) / len(shared)
                            dy = sum(disp_pose[t].y - anchor_pose[t].y for t in shared) / len(shared)
                            pt = (anchor_pose[joint].x + dx, anchor_pose[joint].y + dy)
                        elif abs(interpolated[anchor].timestamp - interpolated[pos].timestamp) <= min(0.5, max_gap):
                            pt = (anchor_pose[joint].x, anchor_pose[joint].y)
            elif parent is not None and reliable(disp_pose.get(parent), confidence):
                parent_lm = disp_pose[parent]
                pt = _resolve_raw_child_point(
                    raw_pose,
                    disp_pose,
                    joint,
                    parent_lm,
                    ref_len[joint],
                    ref_len,
                    confidence,
                )
                if pt is None:
                    anchor = _nearest_anchor_pos(
                        pos,
                        seg_ids,
                        interpolated,
                        max_gap,
                        (joint, parent),
                        confidence,
                    )
                    if anchor is not None:
                        anchor_pose = interpolated[anchor].poses[0]
                        pt = (
                            parent_lm.x + (anchor_pose[joint].x - anchor_pose[parent].x),
                            parent_lm.y + (anchor_pose[joint].y - anchor_pose[parent].y),
                        )

            if pt is not None and math.isfinite(pt[0]) and math.isfinite(pt[1]):
                disp_pose[joint] = Landmark(
                    x=min(1.0, max(0.0, float(pt[0]))),
                    y=min(1.0, max(0.0, float(pt[1]))),
                    visibility=1.0,
                    presence=1.0,
                )
                origins_out[pos][joint] = "interpolated"
                if joint in _LIMB_JOINTS:
                    modified_for_smooth.add((pos, joint))
                changed_frames.add(pos)

        # Keep `interpolated` in sync between topological layers so children anchor on completed parents.
        for pos in changed_frames:
            buf = pose_buffers[pos]
            if buf is not None and (
                len(interpolated[pos].poses) != 1 or interpolated[pos].poses[0] is not buf
            ):
                interpolated[pos] = replace(interpolated[pos], poses=[buf])

    # Pass C: light 3-frame smoothing on raw-guided/completed limb estimates.
    if n >= 5 and modified_for_smooth:
        snapshot: dict[tuple[int, str], tuple[float, float]] = {}
        for pos, joint in modified_for_smooth:
            if not (0 < pos < n - 1) or seg_ids[pos] < 0:
                continue
            if seg_ids[pos - 1] != seg_ids[pos] or seg_ids[pos + 1] != seg_ids[pos]:
                continue
            p_prev, p_cur, p_next = pose_buffers[pos - 1], pose_buffers[pos], pose_buffers[pos + 1]
            if p_prev is None or p_cur is None or p_next is None:
                continue
            l0, l1, l2 = p_prev.get(joint), p_cur.get(joint), p_next.get(joint)
            if reliable(l0, confidence) and reliable(l1, confidence) and reliable(l2, confidence):
                assert l0 is not None and l1 is not None and l2 is not None
                snapshot[(pos, joint)] = (
                    min(1.0, max(0.0, 0.25 * l0.x + 0.50 * l1.x + 0.25 * l2.x)),
                    min(1.0, max(0.0, 0.25 * l0.y + 0.50 * l1.y + 0.25 * l2.y)),
                )
        for (pos, joint), (sx, sy) in snapshot.items():
            buf = pose_buffers[pos]
            assert buf is not None
            buf[joint] = Landmark(x=sx, y=sy, visibility=1.0, presence=1.0)
            changed_frames.add(pos)

    for pos in changed_frames:
        buf = pose_buffers[pos]
        if buf is not None:
            interpolated[pos] = replace(interpolated[pos], poses=[buf])


def interpolate_coach_data(
    data: dict[str, Any],
    cfg: Config,
    *,
    pose_recovery: Mapping[str, Any] | None = None,
    movement_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare a separate display layer, leaving raw evidence and coaching untouched."""
    frames = data.get("frames")
    if not isinstance(frames, dict) or not frames:
        return data

    indices = sorted(frames, key=int)
    raw_recovery = pose_recovery if pose_recovery is not None else data.get("pose_recovery")
    valid_recovery: Mapping[str, Any] | None = None
    if isinstance(raw_recovery, Mapping):
        try:
            dancer_sel = (
                data.get("dancer_selection")
                if isinstance(data.get("dancer_selection"), Mapping)
                else None
            )
            valid_recovery = validate_pose_recovery_payload(
                raw_recovery,
                frame_count=len(indices),
                valid_frame_indices={int(k) for k in indices if str(k).isdigit()},
                timestamps=[float(frames[k].get("timestamp", 0.0)) for k in indices],
                source_sha256=data.get("source_sha256"),
                model_sha256=data.get("model_hash"),
                source_width=int(data.get("width") or 0) or None,
                source_height=int(data.get("height") or 0) or None,
                dancer_selection=dancer_sel,
            )
        except ValueError as exc:
            logger.warning("Ignoring invalid or stale pose_recovery payload: %s", exc)
            valid_recovery = None

    if cfg.replay_joint_max_gap <= 0 and valid_recovery is None:
        return data

    width, height = int(data.get("width", 0)), int(data.get("height", 0))
    observations = [_coach_observation(frames[index], width, height) for index in indices]
    working: list[FrameObservation] = []
    origins_out: list[dict[str, str]] = []
    diagnostics_per_pos: list[dict[str, str]] = []
    status_per_pos: list[str] = []

    rec_frames = valid_recovery.get("frames", {}) if valid_recovery is not None else {}
    for pos, (index, obs) in enumerate(zip(indices, observations, strict=True)):
        origins: dict[str, str] = {}
        poses_copy = [dict(p) for p in obs.poses]
        if len(poses_copy) == 1:
            for jname, lm in poses_copy[0].items():
                if reliable(lm, cfg.confidence):
                    origins[jname] = "forward"

        rec_entry = rec_frames.get(str(index)) if isinstance(rec_frames, Mapping) else None
        diag: dict[str, str] = {}
        status_txt = ""
        if isinstance(rec_entry, Mapping):
            raw_diag = rec_entry.get("diagnostics")
            if isinstance(raw_diag, Mapping):
                diag = {str(k): str(v) for k, v in raw_diag.items()}
            status_txt = str(rec_entry.get("status_summary") or "")
            rec_lms = rec_entry.get("landmarks")
            if isinstance(rec_lms, Mapping) and rec_lms and len(poses_copy) <= 1:
                pose_dict = dict(poses_copy[0]) if poses_copy else {}
                for jname, lm_map in rec_lms.items():
                    if not isinstance(lm_map, Mapping):
                        continue
                    existing_lm = pose_dict.get(str(jname))
                    if reliable(existing_lm, cfg.confidence):
                        continue
                    origin = str(lm_map.get("origin") or "backward")
                    if origin == "backward":
                        pose_dict[str(jname)] = Landmark(
                            x=float(lm_map["x"]),
                            y=float(lm_map["y"]),
                            visibility=float(lm_map.get("visibility", 1.0)),
                            presence=float(lm_map.get("presence", 1.0)),
                            z=float(lm_map.get("z", 0.0)),
                        )
                        origins[str(jname)] = "backward"
                if pose_dict:
                    poses_copy = [pose_dict]
        working.append(FrameObservation(obs.timestamp, obs.width, obs.height, poses_copy))
        origins_out.append(origins)
        diagnostics_per_pos.append(diag)
        status_per_pos.append(status_txt)

    pivot_intervals = _pivot_intervals_from_metadata(indices, frames, movement_data, valid_recovery)
    cut_positions: set[int] = set()
    if isinstance(movement_data, Mapping):
        cut_set = {int(c) for c in movement_data.get("cut_frames") or () if isinstance(c, int)}
        if cut_set:
            cut_positions = _filter_true_cut_positions(cut_set, indices, working, cfg)

    track_keys = [
        str(
            frames[index].get("track_key")
            or (
                f"dancer_{frames[index]['dancer_index']}"
                if isinstance(frames[index].get("dancer_index"), int)
                else "single"
            )
        )
        for index in indices
    ]

    interpolated = interpolate_poses(
        working,
        method=cfg.replay_joint_interpolation,
        max_gap=cfg.replay_joint_max_gap,
        confidence=cfg.confidence,
        pivot_intervals=pivot_intervals,
        pivot_max_gap=None,
        cut_positions=cut_positions,
        track_keys=track_keys,
        origins_out=origins_out,
    )
    _complete_occluded_limbs_for_display(
        interpolated,
        observations,
        origins_out,
        cut_positions,
        track_keys,
        cfg,
    )

    any_changed = valid_recovery is not None or any(
        origin in ("backward", "interpolated")
        for origins in origins_out
        for origin in origins.values()
    )
    if not any_changed:
        return data

    new_frames: dict[str, Any] = {}
    for pos, (index, obs) in enumerate(zip(indices, interpolated, strict=True)):
        orig_entry = frames[index]
        origins = origins_out[pos]
        diag = dict(diagnostics_per_pos[pos])
        status_txt = status_per_pos[pos]
        if not status_txt:
            interp_lower = [
                j
                for j, o in origins.items()
                if o == "interpolated" and j in LOWER_LIMB_JOINTS
            ]
            if interp_lower:
                focus = next(
                    (j for j in interp_lower if j.endswith(("_ankle", "_knee"))),
                    interp_lower[0],
                )
                status_txt = f"{focus.replace('_', ' ').capitalize()} interpolated"
                for j in interp_lower:
                    diag.setdefault(j, "Recovered for display")
        forward_raw = orig_entry.get("poses") or orig_entry.get("measurement_poses") or []
        new_frames[index] = {
            **orig_entry,
            "display_poses": [_serialise_pose(pose) for pose in obs.poses],
            "display_forward_poses": forward_raw,
            "display_joint_origins": origins,
            "display_diagnostics": diag,
            "display_status_summary": status_txt,
        }
    return {
        **data,
        "frames": new_frames,
    }


def _render_coach_measurement_overlay(
    image: np.ndarray,
    coach_data: dict[str, Any],
    frame_index: int,
    draw_pose: Callable[[np.ndarray, dict[str, Landmark]], None],
) -> np.ndarray:
    """Draw a per-pose measurement overlay from raw cached landmarks."""
    frames = coach_data.get("frames")
    entry = frames.get(str(frame_index)) if isinstance(frames, dict) else None
    if not isinstance(entry, dict):
        return image
    width = int(coach_data.get("width") or image.shape[1])
    height = int(coach_data.get("height") or image.shape[0])
    observation = _coach_observation(entry, width, height, require_measurements=True)
    canvas = image.copy()
    for pose in observation.poses:
        draw_pose(canvas, pose)
    return canvas


def render_coach_torso_axis(
    image: np.ndarray, coach_data: dict[str, Any], frame_index: int, cfg: Config, *,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw the axis from raw cached landmarks, independent of posture coaching."""
    return _render_coach_measurement_overlay(
        image,
        coach_data,
        frame_index,
        lambda canvas, pose: draw_torso_axis(
            canvas, pose, False, cfg.confidence, show_labels=show_labels
        ),
    )


def render_coach_head_alignment(
    image: np.ndarray, coach_data: dict[str, Any], frame_index: int, cfg: Config, *,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw head alignment lines from raw cached landmarks, independent of posture coaching."""
    return _render_coach_measurement_overlay(
        image,
        coach_data,
        frame_index,
        lambda canvas, pose: draw_head_alignment(
            canvas, pose, False, cfg.confidence, show_labels=show_labels
        ),
    )


def render_coach_head_posture(
    image: np.ndarray, coach_data: dict[str, Any], frame_index: int, cfg: Config, *,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw head posture lines from raw cached landmarks, independent of posture coaching."""
    capture = coach_data.get("capture_profile")
    capture_map = capture if isinstance(capture, dict) else {}
    side = str(capture_map.get("side") or cfg.side)
    facing = str(capture_map.get("facing") or cfg.facing)
    return _render_coach_measurement_overlay(
        image,
        coach_data,
        frame_index,
        lambda canvas, pose: draw_head_posture(
            canvas,
            pose,
            False,
            cfg.confidence,
            side=side,
            facing=facing,
            show_labels=show_labels,
        ),
    )


def render_coach_frame_overlay(
    image: np.ndarray,
    coach_data: dict[str, Any],
    frame_index: int,
    cfg: Config,
    *,
    verbose_skeleton: bool = True,
    show_vertical_axis: bool = True,
    show_footer: bool = False,
    lines_only: bool = False,
    show_estimates: bool = True,
    forward_only: bool = False,
) -> np.ndarray:
    """Render the Open Practice Coach HUD (axis lines, skeleton, alerts, role, beats) on a frame.

    Uses the per-frame state saved by `analyse_imported_clip` and the live-coach `overlay.draw()`.
    Returns `image` unchanged when the frame has no saved coach state or sizes disagree.
    `lines_only` draws reliable skeleton connections without labels, dots or alerts.
    """
    frames_map = coach_data.get("frames")
    if not isinstance(frames_map, dict):
        return image
    entry = frames_map.get(str(frame_index))
    if not isinstance(entry, dict):
        return image

    width = int(coach_data.get("width") or image.shape[1])
    height = int(coach_data.get("height") or image.shape[0])
    if image.shape[1] != width or image.shape[0] != height:
        return image

    observation = _coach_observation(entry, width, height)
    use_estimates = bool(show_estimates and not forward_only)
    raw_origins = entry.get("display_joint_origins")
    joint_origins: Mapping[str, str] | None = (
        raw_origins if use_estimates and isinstance(raw_origins, Mapping) else None
    )
    if use_estimates:
        poses = _coach_observation(entry, width, height, use_display_poses=True).poses
    else:
        raw_fwd = entry.get("display_forward_poses")
        poses = _deserialise_pose_list(raw_fwd) if isinstance(raw_fwd, list) else observation.poses
        if isinstance(raw_origins, Mapping):
            poses = [
                {
                    k: v
                    for k, v in pose.items()
                    if raw_origins.get(k, "forward") == "forward"
                }
                for pose in poses
            ]

    if lines_only:
        canvas = image.copy()
        for pose in poses:
            for a, b in CONNECTIONS:
                if reliable(pose.get(a), cfg.confidence) and reliable(
                    pose.get(b), cfg.confidence
                ):
                    pa, pb = pose[a], pose[b]
                    draw_styled_bone(
                        canvas,
                        screen_point(pa.x, pa.y, width, height, False),
                        screen_point(pb.x, pb.y, width, height, False),
                        (80, 220, 80),
                        2,
                        style=edge_provenance_style(a, b, joint_origins),
                    )
        return canvas

    def _opt(key: str) -> float | None:
        value = entry.get(key)
        return float(value) if isinstance(value, int | float) else None

    beat_state = entry.get("beat_state")
    return cast(
        np.ndarray,
        draw(
            image,
            observation,
            False,
            list(entry.get("lines") or []),
            str(entry.get("message") or ""),
            cfg.confidence,
            head_down_angle=cfg.head_down_angle,
            head_down_duration=cfg.head_down_duration,
            head_down_elapsed=float(entry.get("head_down_elapsed", 0.0)),
            head_angle_smoothed=_opt("head_angle"),
            role=str(entry.get("role") or "NONE"),
            beat_state=beat_state if isinstance(beat_state, dict) else None,
            head_neutral_offset=float(coach_data.get("neutral_offset", cfg.head_neutral_offset)),
            shoulder_raise_threshold=cfg.shoulder_raise_threshold,
            shoulder_tension_duration=cfg.shoulder_tension_duration,
            shoulder_tension_elapsed=float(entry.get("shoulder_tension_elapsed", 0.0)),
            shoulder_neutral_clearance=float(
                entry.get("shoulder_neutral_clearance", cfg.shoulder_neutral_clearance)
            ),
            shoulder_elev_smoothed=_opt("shoulder_elev_pct"),
            embrace_type=str(entry.get("embrace_type") or "OPEN"),
            embrace_closed_threshold=cfg.embrace_closed_threshold,
            closed_embrace_min_knee_bend=cfg.closed_embrace_min_knee_bend,
            knee_straight_duration=cfg.knee_straight_duration,
            knee_straight_elapsed=float(entry.get("knee_straight_elapsed", 0.0)),
            knee_bend_smoothed=_opt("knee_bend_deg"),
            facing=cfg.facing,
            side=cfg.side,
            axis_neutral_offset=float(
                coach_data.get("axis_neutral_offset", cfg.axis_neutral_offset)
            ),
            axis_curve_entry=cfg.axis_curve_entry,
            axis_curve_clear=cfg.axis_curve_clear,
            axis_curve_duration=cfg.axis_curve_duration,
            axis_curve_elapsed=float(entry.get("axis_curve_elapsed", 0.0)),
            axis_curve_smoothed=_opt("axis_curve_smoothed"),
            axis_curve_state=str(entry.get("axis_curve_state") or "WITHIN_LIMIT"),
            ankle_high_lift_ratio=cfg.ankle_high_lift_ratio,
            ankle_lift_smoothed=_opt("ankle_lift_ratio"),
            consecutive_high_ankle_steps=int(entry.get("consecutive_high_ankle_steps", 0)),
            max_consecutive_high_ankle_steps=cfg.max_consecutive_high_ankle_steps,
            slow_tango_max_bpm=cfg.slow_tango_max_bpm,
            ankle_high_alert=bool(entry.get("ankle_alert", False)),
            verbose_skeleton=verbose_skeleton,
            show_vertical_axis=show_vertical_axis,
            show_footer=show_footer,
            settings_coaching=bool(coach_data.get("settings_coaching", True)),
            skeleton_poses=poses,
            skeleton_joint_origins=joint_origins,
            shoulder_geometry=shoulder_geometry_kwargs(cfg),
            knee_geometry=knee_geometry_kwargs(cfg),
        ),
    )


def coach_notes_for_frame(coach_data: dict[str, Any], frame_index: int) -> list[str] | None:
    """Return the saved Live-coaching-notes lines for a source frame, if present."""
    frames_map = coach_data.get("frames")
    if not isinstance(frames_map, dict):
        return None
    entry = frames_map.get(str(frame_index))
    if not isinstance(entry, dict):
        return None
    notes = entry.get("notes")
    if isinstance(notes, list) and notes:
        return [str(n) for n in notes]
    return None


@dataclass
class _AlertTimers:
    """Sustained-alert timers mirroring the live Open Practice Coach (`Application.run`)."""

    head_down_since: float | None = None
    shoulder_tension_since: float | None = None
    shoulder_last_valid: float | None = None
    knee_straight_since: float | None = None
    knee_last_valid: float | None = None


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def _serialise_pose(pose: dict[str, Landmark]) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {}
    for name, lm in pose.items():
        if not (math.isfinite(lm.x) and math.isfinite(lm.y)):
            continue
        out[name] = [
            round(float(lm.x), 4),
            round(float(lm.y), 4),
            round(float(lm.z), 4) if lm.z is not None and math.isfinite(lm.z) else 0.0,
            _round_or_none(lm.visibility, 3),
            _round_or_none(lm.presence, 3),
        ]
    return out


def _build_coach_frame_entry(
    *,
    cfg: Config,
    frame_obs: FrameObservation,
    display_poses: list[dict[str, Landmark]],
    source_time: float,
    clip_time: float,
    head_angle: float | None,
    shoulder_elev_pct: float | None,
    shoulder_neutral_clearance: float,
    knee_bend_deg: float | None,
    ankle_lift_ratio: float | None,
    consecutive_high_ankle_steps: int,
    bad_ankles: bool,
    ankle_high_now: bool,
    axis_curve_smoothed: float | None,
    axis_curve_state: str,
    axis_curve_elapsed: float,
    bad_axis: bool,
    role: str,
    embrace_type: str,
    upper_body: dict[str, float] | None,
    st_state: dict[str, Any] | None,
    quality_reason: str,
    timers: _AlertTimers,
    session_rows: list[dict[str, Any]],
    single_pose_ok: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute alert timers and per-frame coaching state/notes for one analysed frame."""
    # Head pitch: continuous downward tilt.
    if head_angle is not None and head_angle < cfg.head_down_angle:
        if timers.head_down_since is None:
            timers.head_down_since = source_time
        head_down_elapsed = source_time - timers.head_down_since
    else:
        timers.head_down_since = None
        head_down_elapsed = 0.0
    bad_head = head_down_elapsed >= cfg.head_down_duration

    # Shoulder tension with hysteresis.
    sh_clear = max(
        cfg.shoulder_recovery_min_pct,
        cfg.shoulder_raise_threshold * cfg.shoulder_recovery_ratio,
    )
    if shoulder_elev_pct is not None:
        timers.shoulder_last_valid = source_time
        if shoulder_elev_pct >= cfg.shoulder_raise_threshold:
            if timers.shoulder_tension_since is None:
                timers.shoulder_tension_since = source_time
            shoulder_tension_elapsed = source_time - timers.shoulder_tension_since
        elif timers.shoulder_tension_since is not None and shoulder_elev_pct >= sh_clear:
            shoulder_tension_elapsed = source_time - timers.shoulder_tension_since
        else:
            timers.shoulder_tension_since = None
            shoulder_tension_elapsed = 0.0
    elif (
        timers.shoulder_last_valid is None
        or (source_time - timers.shoulder_last_valid) > cfg.max_gap
    ):
        timers.shoulder_tension_since = None
        shoulder_tension_elapsed = 0.0
    elif timers.shoulder_tension_since is not None:
        shoulder_tension_elapsed = source_time - timers.shoulder_tension_since
    else:
        shoulder_tension_elapsed = 0.0
    bad_shoulders = shoulder_tension_elapsed >= cfg.shoulder_tension_duration

    # Knee bend with hysteresis.
    knee_clear = cfg.closed_embrace_min_knee_bend + cfg.knee_recovery_margin
    if knee_bend_deg is not None:
        timers.knee_last_valid = source_time
        if knee_bend_deg < cfg.closed_embrace_min_knee_bend:
            if timers.knee_straight_since is None:
                timers.knee_straight_since = source_time
            knee_straight_elapsed = source_time - timers.knee_straight_since
        elif timers.knee_straight_since is not None and knee_bend_deg < knee_clear:
            knee_straight_elapsed = source_time - timers.knee_straight_since
        else:
            timers.knee_straight_since = None
            knee_straight_elapsed = 0.0
    elif timers.knee_last_valid is None or (source_time - timers.knee_last_valid) > cfg.max_gap:
        timers.knee_straight_since = None
        knee_straight_elapsed = 0.0
    elif timers.knee_straight_since is not None:
        knee_straight_elapsed = source_time - timers.knee_straight_since
    else:
        knee_straight_elapsed = 0.0
    bad_knees = knee_straight_elapsed >= cfg.knee_straight_duration

    sh_disp = shoulder_elev_pct if shoulder_elev_pct is not None else 0.0
    head_disp = head_angle if head_angle is not None else 0.0
    knee_disp = knee_bend_deg if knee_bend_deg is not None else 0.0
    alert = ""
    if bad_ankles:
        lift_pct = (ankle_lift_ratio or cfg.ankle_high_lift_ratio) * 100.0
        alert = (
            f"KEEP FEET CLOSE TO FLOOR (SLOW TANGO <={cfg.slow_tango_max_bpm:.0f} BPM): "
            f"ankle raised too high ({lift_pct:.1f}%) for {consecutive_high_ankle_steps} "
            f"consecutive steps (>={cfg.max_consecutive_high_ankle_steps})"
        )
    elif bad_axis and axis_curve_smoothed is not None:
        alert = (
            f"HIP OFFSET: hip/pelvis +{axis_curve_smoothed * 100.0:.1f}% "
            f"past ankle-to-head line ({axis_curve_elapsed:.1f}s)"
        )
    elif bad_knees:
        alert = (
            f"BEND YOUR KNEES: knee bend {knee_disp:.1f} deg "
            f"< {cfg.closed_embrace_min_knee_bend:.1f} deg ({knee_straight_elapsed:.1f}s)"
        )
    elif bad_head and bad_shoulders:
        alert = (
            f"BAD HEAD ALIGNMENT ({head_disp:+.1f} deg) & "
            f"LOWER YOUR SHOULDERS ({sh_disp:+.0f}% for {shoulder_tension_elapsed:.1f}s)"
        )
    elif bad_shoulders:
        alert = (
            f"LOWER YOUR SHOULDERS: shoulder tension detected "
            f"({sh_disp:+.0f}% raised for {shoulder_tension_elapsed:.1f}s)"
        )
    elif bad_head:
        alert = (
            f"BAD HEAD ALIGNMENT: head pointing down ({head_disp:+.1f} deg for "
            f"{head_down_elapsed:.1f}s)"
        )

    if alert:
        primary = alert
    elif not frame_obs.poses:
        primary = quality_reason or "Coaching status: No dancer detected in this frame"
    elif quality_reason:
        primary = quality_reason
    elif any(r.get("state") == "DEVIATION" for r in session_rows):
        primary = next(
            MESSAGES.get(r["metric"], f"{r['metric']}: outside approved range")
            for r in session_rows if r.get("state") == "DEVIATION"
        )
    else:
        primary = "Coaching status: Posture & axis within target thresholds"

    # Beat/phrase state for the HUD and notes.
    beat_state: dict[str, Any] | None = None
    if st_state is not None:
        beat_state = {
            "beat": int(st_state.get("beat") or 0),
            "phrase": int(st_state.get("phrase") or 0),
            "bpm": float(st_state.get("bpm") or cfg.tango_bpm),
            "playing": bool(st_state.get("playing", False)),
            "analyzing": bool(st_state.get("analyzing", False)),
            "analyzed_seconds": 0.0,
            "analysis_target_seconds": 30.0,
            "phrase_locked": st_state.get("phrase") is not None,
            "source": str(st_state.get("source") or SOUNDTRACK_SOURCE_LABEL),
        }
    if st_state is None:
        beat_txt = "disabled"
    elif not st_state.get("playing") or not st_state.get("beat"):
        beat_txt = "No rhythmic beat detected at this moment"
    elif st_state.get("phrase") is None:
        beat_txt = f"Beat pulse {st_state.get('bpm') or 0:.0f} BPM ({SOUNDTRACK_SOURCE_LABEL})"
    else:
        beat_txt = (
            f"Phrase {st_state['phrase']} Beat {st_state['beat']}/8 "
            f"({float(st_state.get('bpm') or 0.0):.0f} BPM, {SOUNDTRACK_SOURCE_LABEL})"
        )
    role_txt = (
        f"{role} ({embrace_type})"
        if role in ("LEADER", "FOLLOWER") and embrace_type in ("CLOSED", "OPEN")
        else role
    )

    torso_txt = "N/A"
    torso_tilt = None
    if single_pose_ok:
        torso = torso_axis_alignment(
            frame_obs.poses[0], frame_obs.width, frame_obs.height, cfg.confidence
        )
        if torso is not None:
            torso_tilt = abs(torso[0])
            torso_txt = f"{torso_tilt:.1f} deg from vertical"
    if axis_curve_smoothed is None:
        hip_txt = "N/A"
    elif bad_axis:
        hip_txt = f"HIGH {axis_curve_smoothed * 100.0:+.1f}% [DEVIATION]"
    elif axis_curve_state == "PENDING":
        hip_txt = f"PENDING {axis_curve_smoothed * 100.0:+.1f}%"
    else:
        hip_txt = f"OK ({axis_curve_smoothed * 100.0:+.1f}%)"
    head_txt = "N/A" if head_angle is None else f"{head_angle:+.1f} deg"
    sh_txt = (
        "N/A"
        if shoulder_elev_pct is None
        else (
            f"HIGH {shoulder_elev_pct:+.0f}%"
            if shoulder_elev_pct >= cfg.shoulder_raise_threshold
            else f"OK ({shoulder_elev_pct:+.0f}%)"
        )
    )
    knee_txt = (
        "N/A"
        if knee_bend_deg is None
        else (
            f"STRAIGHT {knee_bend_deg:.1f} deg"
            if bad_knees or knee_bend_deg < cfg.closed_embrace_min_knee_bend
            else f"OK ({knee_bend_deg:.1f} deg)"
        )
    )
    feet_txt = (
        "N/A"
        if ankle_lift_ratio is None
        else (
            f"HIGH {ankle_lift_ratio * 100.0:.1f}% "
            f"({consecutive_high_ankle_steps}/{cfg.max_consecutive_high_ankle_steps} steps)"
            if bad_ankles or ankle_high_now
            else f"OK ({ankle_lift_ratio * 100.0:.1f}%, "
            f"{consecutive_high_ankle_steps}/{cfg.max_consecutive_high_ankle_steps} steps)"
        )
    )
    lines = [
        f"Role: {role_txt} | Beat: {beat_txt}",
        f"Torso axis: {torso_txt} | Hip offset: {hip_txt}",
        f"Head: {head_txt} | Shoulders: {sh_txt} | Knees: {knee_txt} | Feet: {feet_txt}",
    ]
    if upper_body:
        lines.append(
            "Upper body: "
            + " | ".join(f"{name} {float(val):+.3f}" for name, val in upper_body.items())
        )
    for row in session_rows:
        val = "N/A" if row.get("smoothed") is None else f"{float(row['smoothed']):.3f}"
        delta = "N/A" if row.get("delta") is None else f"{float(row['delta']):+.3f}"
        lines.append(f"{row['metric']}: {val} | delta {delta} | {row['state']}")

    entry: dict[str, Any] = {
        "timestamp": round(float(source_time), 4),
        "clip_timestamp": round(float(clip_time), 4),
        "poses": [_serialise_pose(p) for p in display_poses if p],
        "message": alert,
        "lines": [],
        "head_angle": _round_or_none(head_angle, 3),
        "head_down_elapsed": round(head_down_elapsed, 3),
        "shoulder_elev_pct": _round_or_none(shoulder_elev_pct, 2),
        "shoulder_tension_elapsed": round(shoulder_tension_elapsed, 3),
        "shoulder_neutral_clearance": round(float(shoulder_neutral_clearance), 4),
        "knee_bend_deg": _round_or_none(knee_bend_deg, 2),
        "knee_straight_elapsed": round(knee_straight_elapsed, 3),
        "embrace_type": embrace_type,
        "ankle_alert": bool(bad_ankles),
        "ankle_lift_ratio": _round_or_none(ankle_lift_ratio, 4),
        "consecutive_high_ankle_steps": int(consecutive_high_ankle_steps),
        "axis_curve_smoothed": _round_or_none(axis_curve_smoothed, 4),
        "torso_axis_tilt": _round_or_none(torso_tilt, 3),
        "axis_curve_state": axis_curve_state if axis_curve_state != "UNKNOWN" else "WITHIN_LIMIT",
        "axis_curve_elapsed": round(axis_curve_elapsed, 3),
        "role": role,
        "beat_state": beat_state,
        "alerts": {
            "head": bad_head,
            "shoulders": bad_shoulders,
            "knees": bad_knees,
            "axis": bad_axis,
            "ankles": bool(bad_ankles),
        },
    }
    return entry, {"primary": primary, "lines": lines}


def _movement_note_lines(
    movement_data: dict[str, Any],
    frame_index: int,
    fps: float,
    step_history_seconds: float,
) -> list[str]:
    """Pivot/step insights for the current frame plus clip totals."""
    out: list[str] = []
    pivots = list(movement_data.get("pivots") or [])
    steps = list(movement_data.get("steps") or [])
    for p in pivots:
        if int(p["start_frame"]) <= frame_index < int(p["end_frame"]):
            rate = p.get("peak_rate_deg_s", p.get("mean_rate_deg_s"))
            rate_txt = f", peak {abs(float(rate)):.0f} deg/s" if rate is not None else ""
            out.append(
                f"Active Pivot: {str(p['direction']).upper()} "
                f"({float(p['net_rotation_deg']):+.1f} deg net turn{rate_txt})"
            )
            break
    window = max(1, round(step_history_seconds * max(fps, 1.0)))
    for st in steps:
        sf = int(st["frame"])
        if sf <= frame_index < sf + window:
            t = float(st.get("time", sf / max(fps, 1.0)))
            mins = int(t // 60)
            conf = float(st.get("confidence", 0.0)) * 100.0
            out.append(
                f"Recent Step: {str(st['side']).upper()} foot landed at "
                f"{mins:02d}:{t - mins * 60:04.1f} "
                f"(travel {float(st.get('travel_px', 0.0)):.0f} px, conf {conf:.0f}%)"
            )
    out.append(f"Clip Movement Totals: {len(pivots)} pivots, {len(steps)} steps detected")
    return out
