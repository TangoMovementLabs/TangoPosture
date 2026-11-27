"""Display-only single-dancer pose and occluded-limb completion helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ..domain import FrameObservation, Landmark
from ..quality import reliable

if TYPE_CHECKING:
    from ..config import Config

_TORSO_JOINTS: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)
_HEAD_JOINTS: frozenset[str] = frozenset({"nose", "left_ear", "right_ear"})
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
_RAW_BORDER_MIN = -0.35
_RAW_BORDER_MAX = 1.35


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


def _landmark_usable_raw(lm: Landmark | None) -> bool:
    return (
        lm is not None
        and math.isfinite(lm.x)
        and math.isfinite(lm.y)
        and _RAW_BORDER_MIN <= lm.x <= _RAW_BORDER_MAX
        and _RAW_BORDER_MIN <= lm.y <= _RAW_BORDER_MAX
        and _landmark_conf(lm) > 0.0
    )


def _pose_torso_center(
    pose: Mapping[str, Landmark],
    confidence: float,
) -> tuple[float, float] | None:
    hips = [
        lm
        for j in ("left_hip", "right_hip")
        if (lm := pose.get(j)) is not None and reliable(lm, confidence) and _landmark_in_frame(lm)
    ]
    pts = hips or [
        lm
        for j in _TORSO_JOINTS
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


def _effective_parent(
    joint: str,
    pose: Mapping[str, Landmark],
    confidence: float,
) -> str | None:
    """Resolve the active kinematic parent, allowing opposite-shoulder fallback for head joints."""
    parent = _KINEMATIC_PARENT.get(joint)
    if parent is not None and reliable(pose.get(parent), confidence):
        return parent
    if joint in _HEAD_JOINTS:
        alt = "right_shoulder" if parent == "left_shoulder" else "left_shoulder"
        if reliable(pose.get(alt), confidence):
            return alt
    return None


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
    if not _landmark_usable_raw(raw_lm):
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
            if _landmark_usable_raw(raw_child)
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
            left_oob = (
                pos - offset < 0
                or seg_ids[pos - offset] != seg
                or abs(frames[pos - offset].timestamp - pos_t) > max_gap + 1e-9
            )
            right_oob = (
                pos + offset >= len(frames)
                or seg_ids[pos + offset] != seg
                or abs(frames[pos + offset].timestamp - pos_t) > max_gap + 1e-9
            )
            if left_oob and right_oob:
                break
    return best_pos


def _torso_down_and_outward(
    joint: str,
    pose: Mapping[str, Landmark],
    parent_lm: Landmark,
    eff_parent: str,
    confidence: float,
) -> tuple[float, float, float, float]:
    """Return unit `(down_x, down_y, out_x, out_y)` vectors anchored on visible torso landmarks."""
    sh_pts = [
        lm
        for j in ("left_shoulder", "right_shoulder")
        if (lm := pose.get(j)) is not None and reliable(lm, confidence)
    ]
    hp_pts = [
        lm
        for j in ("left_hip", "right_hip")
        if (lm := pose.get(j)) is not None and reliable(lm, confidence)
    ]
    down_x, down_y = 0.0, 1.0
    if sh_pts and hp_pts:
        dx = (sum(p.x for p in hp_pts) / len(hp_pts)) - (sum(p.x for p in sh_pts) / len(sh_pts))
        dy = (sum(p.y for p in hp_pts) / len(hp_pts)) - (sum(p.y for p in sh_pts) / len(sh_pts))
        norm = math.hypot(dx, dy)
        if norm > 1e-4:
            down_x, down_y = dx / norm, dy / norm

    side_pairs = (
        ("left_shoulder", "right_shoulder")
        if "shoulder" in eff_parent or joint in _HEAD_JOINTS or "elbow" in joint or "wrist" in joint
        else ("left_hip", "right_hip")
    )
    opp_parent = _OPPOSITE_JOINT.get(eff_parent)
    opp_lm = pose.get(opp_parent) if opp_parent is not None else None
    out_x, out_y = 0.0, 0.0
    if opp_lm is not None and reliable(opp_lm, confidence):
        ox = parent_lm.x - opp_lm.x
        oy = parent_lm.y - opp_lm.y
        norm = math.hypot(ox, oy)
        if norm > 1e-4:
            out_x, out_y = ox / norm, oy / norm
    if out_x == 0.0 and out_y == 0.0:
        l_lm = pose.get(side_pairs[0])
        r_lm = pose.get(side_pairs[1])
        if l_lm is not None and r_lm is not None and reliable(l_lm, confidence) and reliable(r_lm, confidence):
            sign = 1.0 if joint.startswith("left_") else -1.0
            ox = sign * (l_lm.x - r_lm.x)
            oy = sign * (l_lm.y - r_lm.y)
            norm = math.hypot(ox, oy)
            if norm > 1e-4:
                out_x, out_y = ox / norm, oy / norm
    if out_x == 0.0 and out_y == 0.0:
        sign = 1.0 if joint.startswith("left_") else -1.0
        out_x, out_y = sign * down_y, -sign * down_x
    return (down_x, down_y, out_x, out_y)


def _canonical_child_offset(
    joint: str,
    pose: Mapping[str, Landmark],
    eff_parent: str,
    parent_lm: Landmark,
    rlen: float,
    confidence: float,
) -> tuple[float, float]:
    """Compute a canonical anatomical child position when raw and anchor cues are absent."""
    down_x, down_y, out_x, out_y = _torso_down_and_outward(
        joint, pose, parent_lm, eff_parent, confidence
    )
    if joint in _HEAD_JOINTS:
        inward = -0.35 if joint == "nose" else 0.15
        ux = inward * out_x - 0.94 * down_x
        uy = inward * out_y - 0.94 * down_y
    elif joint.endswith("_elbow"):
        ux = 0.35 * out_x + 0.94 * down_x
        uy = 0.35 * out_y + 0.94 * down_y
    elif joint.endswith("_wrist"):
        grandparent = _KINEMATIC_PARENT.get(eff_parent)
        gp_lm = pose.get(grandparent) if grandparent is not None else None
        if gp_lm is not None and reliable(gp_lm, confidence):
            ax = parent_lm.x - gp_lm.x
            ay = parent_lm.y - gp_lm.y
            norm_a = math.hypot(ax, ay)
            if norm_a > 1e-4:
                ux = 0.70 * (ax / norm_a) + 0.30 * down_x
                uy = 0.70 * (ay / norm_a) + 0.30 * down_y
            else:
                ux = 0.25 * out_x + 0.97 * down_x
                uy = 0.25 * out_y + 0.97 * down_y
        else:
            ux = 0.25 * out_x + 0.97 * down_x
            uy = 0.25 * out_y + 0.97 * down_y
    elif joint.endswith("_knee"):
        ux = 0.12 * out_x + 0.99 * down_x
        uy = 0.12 * out_y + 0.99 * down_y
    elif joint.endswith("_ankle"):
        grandparent = _KINEMATIC_PARENT.get(eff_parent)
        gp_lm = pose.get(grandparent) if grandparent is not None else None
        if gp_lm is not None and reliable(gp_lm, confidence):
            ax = parent_lm.x - gp_lm.x
            ay = parent_lm.y - gp_lm.y
            norm_a = math.hypot(ax, ay)
            if norm_a > 1e-4:
                ux = 0.75 * (ax / norm_a) + 0.25 * down_x
                uy = 0.75 * (ay / norm_a) + 0.25 * down_y
            else:
                ux = 0.08 * out_x + 0.99 * down_x
                uy = 0.08 * out_y + 0.99 * down_y
        else:
            ux = 0.08 * out_x + 0.99 * down_x
            uy = 0.08 * out_y + 0.99 * down_y
    elif joint.endswith("_heel"):
        ux = -0.25 * out_x + 0.97 * down_x
        uy = -0.25 * out_y + 0.97 * down_y
    else:
        ux = 0.80 * out_x + 0.60 * down_x
        uy = 0.80 * out_y + 0.60 * down_y

    norm = math.hypot(ux, uy)
    if norm <= 1e-6:
        ux, uy = 0.0, 1.0
    else:
        ux, uy = ux / norm, uy / norm
    return (parent_lm.x + ux * rlen, parent_lm.y + uy * rlen)


def _complete_torso_joint_point(
    joint: str,
    pos: int,
    raw_pose: Mapping[str, Landmark],
    disp_pose: Mapping[str, Landmark],
    seg_ids: Sequence[int],
    interpolated: Sequence[FrameObservation],
    max_gap: float,
    confidence: float,
) -> tuple[float, float] | None:
    raw_lm = raw_pose.get(joint)
    if _landmark_usable_raw(raw_lm):
        assert raw_lm is not None
        return (float(raw_lm.x), float(raw_lm.y))

    anchor = _nearest_anchor_pos(pos, seg_ids, interpolated, max_gap, (joint,), confidence)
    if anchor is None:
        anchor = _nearest_anchor_pos(pos, seg_ids, interpolated, float("inf"), (joint,), confidence)
    if anchor is not None:
        anchor_pose = interpolated[anchor].poses[0]
        shared = [
            t
            for t in _TORSO_JOINTS
            if t != joint
            and reliable(disp_pose.get(t), confidence)
            and reliable(anchor_pose.get(t), confidence)
        ]
        if shared:
            dx = sum(disp_pose[t].x - anchor_pose[t].x for t in shared) / len(shared)
            dy = sum(disp_pose[t].y - anchor_pose[t].y for t in shared) / len(shared)
            return (anchor_pose[joint].x + dx, anchor_pose[joint].y + dy)
        return (anchor_pose[joint].x, anchor_pose[joint].y)

    # Last-resort torso parallelogram when the other three torso corners are available.
    parallelogram = {
        "left_shoulder": ("right_shoulder", "left_hip", "right_hip"),
        "right_shoulder": ("left_shoulder", "right_hip", "left_hip"),
        "left_hip": ("left_shoulder", "right_hip", "right_shoulder"),
        "right_hip": ("right_shoulder", "left_hip", "left_shoulder"),
    }
    a_k, b_k, opp_k = parallelogram[joint]
    a_lm, b_lm, opp_lm = disp_pose.get(a_k), disp_pose.get(b_k), disp_pose.get(opp_k)
    if reliable(a_lm, confidence) and reliable(b_lm, confidence) and reliable(opp_lm, confidence):
        assert a_lm is not None and b_lm is not None and opp_lm is not None
        return (a_lm.x + b_lm.x - opp_lm.x, a_lm.y + b_lm.y - opp_lm.y)
    return None


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

    observed_joints = {name for obs in observations for pose in obs.poses for name in pose}
    if not observed_joints:
        return

    n = len(interpolated)
    confidence = float(cfg.confidence)
    max_gap = float(cfg.replay_joint_max_gap)
    seg_ids = _display_segment_ids(interpolated, cut_positions, track_keys)
    ref_len = _compute_reference_bone_lengths(interpolated, observed_joints, confidence)
    modified_for_smooth: set[tuple[int, str]] = set()
    has_full_torso = set(_TORSO_JOINTS).issubset(observed_joints)
    segs_with_pose = {
        seg_ids[idx]
        for idx, frame in enumerate(interpolated)
        if seg_ids[idx] >= 0 and len(frame.poses) == 1 and bool(frame.poses[0])
    }

    pose_buffers: list[dict[str, Landmark] | None] = []
    for idx, frame in enumerate(interpolated):
        if len(frame.poses) == 1:
            pose_buffers.append(dict(frame.poses[0]))
        elif (
            len(frame.poses) == 0
            and has_full_torso
            and seg_ids[idx] >= 0
            and seg_ids[idx] in segs_with_pose
        ):
            pose_buffers.append({})
        else:
            pose_buffers.append(None)
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
                or not _landmark_usable_raw(raw_lm)
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
                    x=min(1.0, max(0.0, float(cur_lm.x + w * (raw_lm.x - cur_lm.x)))),
                    y=min(1.0, max(0.0, float(cur_lm.y + w * (raw_lm.y - cur_lm.y)))),
                    visibility=1.0,
                    presence=1.0,
                )
                modified_for_smooth.add((pos, joint))
                changed_frames.add(pos)

    for pos in changed_frames:
        buf = pose_buffers[pos]
        if buf is not None:
            interpolated[pos] = replace(interpolated[pos], poses=[buf])

    # Pass B: complete still-missing/unreliable joints in topological order.
    for joint in _TOPO_JOINTS:
        if joint not in observed_joints:
            continue
        is_torso = joint in _TORSO_JOINTS
        for pos in range(n):
            disp_pose = pose_buffers[pos]
            if disp_pose is None or reliable(disp_pose.get(joint), confidence):
                continue
            raw_pose = observations[pos].poses[0] if len(observations[pos].poses) == 1 else {}
            pt: tuple[float, float] | None = None
            if is_torso:
                pt = _complete_torso_joint_point(
                    joint,
                    pos,
                    raw_pose,
                    disp_pose,
                    seg_ids,
                    interpolated,
                    max_gap,
                    confidence,
                )
            else:
                eff_parent = _effective_parent(joint, disp_pose, confidence)
                if eff_parent is not None:
                    parent_lm = disp_pose[eff_parent]
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
                            pos, seg_ids, interpolated, max_gap, (joint, eff_parent), confidence
                        )
                        if anchor is None:
                            anchor = _nearest_anchor_pos(
                                pos,
                                seg_ids,
                                interpolated,
                                float("inf"),
                                (joint, eff_parent),
                                confidence,
                            )
                        if anchor is not None:
                            anchor_pose = interpolated[anchor].poses[0]
                            pt = (
                                parent_lm.x + (anchor_pose[joint].x - anchor_pose[eff_parent].x),
                                parent_lm.y + (anchor_pose[joint].y - anchor_pose[eff_parent].y),
                            )
                    if pt is None:
                        pt = _canonical_child_offset(
                            joint,
                            disp_pose,
                            eff_parent,
                            parent_lm,
                            ref_len[joint],
                            confidence,
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
