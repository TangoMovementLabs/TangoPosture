"""Two-dancer 1-to-1 gated identity tracker, region filter, duplicate suppression, and fixed scales."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..config import Config
from ..domain import FrameObservation, Landmark
from ..pose.constants import NAMES
from ..quality import reliable
from .models import (
    CandidatePreview,
    DancerAnchor,
    PairedFrameSample,
    SampleQualityStatus,
    TrackedDancerSample,
)

CORE_LANDMARKS = (
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def canonical_xy(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Convert normalized image coordinates `(x, y)` to canonical `(X right, Y up)` pixels."""
    return (float(x) * max(1, width - 1), -float(y) * max(1, height - 1))


@dataclass
class PoseCandidate:
    """Extracted anatomical features for a single pose detection within one frame."""

    pose_index: int
    pose: dict[str, Landmark]
    world_pose: dict[str, Landmark] | None
    hip_norm: tuple[float, float] | None
    hip_px: tuple[float, float] | None
    shoulder_px: tuple[float, float] | None
    torso_length_px: float | None
    hip_eligible: bool
    torso_eligible: bool
    head_eligible: bool
    bbox: tuple[float, float, float, float]
    confidence: float


def _core_confidence(pose: dict[str, Landmark]) -> float:
    vis: list[float] = []
    for name in CORE_LANDMARKS:
        if name in pose:
            v = pose[name].visibility
            if v is not None and math.isfinite(v):
                vis.append(float(v))
    if not vis:
        for lm in pose.values():
            v = lm.visibility
            if v is not None and math.isfinite(v):
                vis.append(float(v))
    return float(sum(vis) / len(vis)) if vis else 0.0


def extract_pose_candidate(
    pose_index: int,
    pose: dict[str, Landmark],
    world_pose: dict[str, Landmark] | None,
    width: int,
    height: int,
    cfg: Config,
) -> PoseCandidate:
    """Extract canonical pixel features and eligibility flags for one pose."""
    conf = float(cfg.confidence)
    l_hip = pose.get("left_hip")
    r_hip = pose.get("right_hip")
    hips_ok = reliable(l_hip, conf) and reliable(r_hip, conf) and l_hip is not None and r_hip is not None
    hip_norm: tuple[float, float] | None = None
    hip_px: tuple[float, float] | None = None
    if hips_ok and l_hip is not None and r_hip is not None:
        hx_n = 0.5 * (l_hip.x + r_hip.x)
        hy_n = 0.5 * (l_hip.y + r_hip.y)
        hip_norm = (hx_n, hy_n)
        hip_px = canonical_xy(hx_n, hy_n, width, height)

    l_sh = pose.get("left_shoulder")
    r_sh = pose.get("right_shoulder")
    sh_ok = reliable(l_sh, conf) and reliable(r_sh, conf) and l_sh is not None and r_sh is not None
    shoulder_px: tuple[float, float] | None = None
    if sh_ok and l_sh is not None and r_sh is not None:
        sx_n = 0.5 * (l_sh.x + r_sh.x)
        sy_n = 0.5 * (l_sh.y + r_sh.y)
        shoulder_px = canonical_xy(sx_n, sy_n, width, height)

    torso_len_px: float | None = None
    min_torso_px = max(8.0, 0.5 * float(cfg.min_torso) * max(1, height))
    if hip_px is not None and shoulder_px is not None:
        torso_len_px = math.hypot(shoulder_px[0] - hip_px[0], shoulder_px[1] - hip_px[1])
        if not math.isfinite(torso_len_px) or torso_len_px < min_torso_px:
            torso_len_px = None

    torso_eligible = bool(hips_ok and sh_ok and torso_len_px is not None)

    nose = pose.get("nose")
    l_ear = pose.get("left_ear")
    r_ear = pose.get("right_ear")
    head_eligible = bool(
        reliable(nose, conf) and (reliable(l_ear, conf) or reliable(r_ear, conf))
    )

    reliable_pts = [lm for lm in pose.values() if reliable(lm, conf)]
    if reliable_pts:
        xs = [lm.x for lm in reliable_pts]
        ys = [lm.y for lm in reliable_pts]
        bbox = (min(xs), min(ys), max(xs), max(ys))
    elif hip_norm is not None:
        bbox = (hip_norm[0], hip_norm[1], hip_norm[0], hip_norm[1])
    else:
        bbox = (0.0, 0.0, 0.0, 0.0)

    return PoseCandidate(
        pose_index=pose_index,
        pose=pose,
        world_pose=world_pose,
        hip_norm=hip_norm,
        hip_px=hip_px,
        shoulder_px=shoulder_px,
        torso_length_px=torso_len_px,
        hip_eligible=hips_ok,
        torso_eligible=torso_eligible,
        head_eligible=head_eligible,
        bbox=bbox,
        confidence=_core_confidence(pose),
    )


def _point_in_region(pt: tuple[float, float], region: dict[str, float]) -> bool:
    return (
        float(region.get("x_min", 0.0)) <= pt[0] <= float(region.get("x_max", 1.0))
        and float(region.get("y_min", 0.0)) <= pt[1] <= float(region.get("y_max", 1.0))
    )


def _are_near_duplicate_poses(
    a: PoseCandidate,
    b: PoseCandidate,
    width: int,
    height: int,
    confidence: float,
) -> bool:
    """Return True only when two detections share nearly coincident joint coordinates."""
    dists: list[float] = []
    for name in CORE_LANDMARKS:
        la = a.pose.get(name)
        lb = b.pose.get(name)
        if la is not None and lb is not None and reliable(la, confidence) and reliable(lb, confidence):
            ax, ay = canonical_xy(la.x, la.y, width, height)
            bx, by = canonical_xy(lb.x, lb.y, width, height)
            dists.append(math.hypot(ax - bx, ay - by))
    if len(dists) < 3:
        return False
    scale = a.torso_length_px or b.torso_length_px or max(20.0, 0.15 * max(1, height))
    median_dist = float(np.median(dists))
    return median_dist < 0.18 * scale


def filter_and_deduplicate_poses(
    obs: FrameObservation,
    cfg: Config,
    analysis_region: dict[str, float] | None = None,
) -> list[PoseCandidate]:
    """Filter poses by analysis region and suppress near-exact duplicate detections."""
    candidates: list[PoseCandidate] = []
    for idx, pose in enumerate(obs.poses):
        world = obs.world_poses[idx] if idx < len(obs.world_poses) else None
        cand = extract_pose_candidate(idx, pose, world, obs.width, obs.height, cfg)
        if not cand.hip_eligible and not cand.torso_eligible:
            continue
        if analysis_region is not None:
            ref_pt = cand.hip_norm
            if ref_pt is None:
                cx = 0.5 * (cand.bbox[0] + cand.bbox[2])
                cy = 0.5 * (cand.bbox[1] + cand.bbox[3])
                ref_pt = (cx, cy)
            if not _point_in_region(ref_pt, analysis_region):
                continue
        candidates.append(cand)

    if len(candidates) <= 1:
        return candidates

    kept: list[PoseCandidate] = []
    for cand in sorted(candidates, key=lambda c: c.confidence, reverse=True):
        if any(
            _are_near_duplicate_poses(cand, prev, obs.width, obs.height, float(cfg.confidence))
            for prev in kept
        ):
            continue
        kept.append(cand)
    kept.sort(key=lambda c: c.pose_index)
    return kept


@dataclass
class _TrackMemory:
    dancer_id: str
    last_time: float | None = None
    last_hip_px: tuple[float, float] | None = None
    vel_px: tuple[float, float] = (0.0, 0.0)
    scale_history: list[float] = field(default_factory=list)
    active: bool = False
    gap_broken: bool = False

    def predicted_hip(self, timestamp: float) -> tuple[float, float] | None:
        if self.last_hip_px is None or self.last_time is None:
            return None
        dt = max(0.0, min(0.55, timestamp - self.last_time))
        return (
            self.last_hip_px[0] + self.vel_px[0] * dt,
            self.last_hip_px[1] + self.vel_px[1] * dt,
        )

    def current_scale(self, default_px: float) -> float:
        if self.scale_history:
            return float(np.median(self.scale_history[-15:]))
        return default_px

    def update(self, timestamp: float, cand: PoseCandidate) -> None:
        if cand.hip_px is None:
            return
        if self.last_hip_px is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if 1e-4 < dt <= 0.60:
                vx = (cand.hip_px[0] - self.last_hip_px[0]) / dt
                vy = (cand.hip_px[1] - self.last_hip_px[1]) / dt
                self.vel_px = (
                    0.5 * self.vel_px[0] + 0.5 * vx,
                    0.5 * self.vel_px[1] + 0.5 * vy,
                )
            else:
                self.vel_px = (0.0, 0.0)
        else:
            self.vel_px = (0.0, 0.0)
        self.last_hip_px = cand.hip_px
        self.last_time = timestamp
        if cand.torso_length_px is not None:
            self.scale_history.append(cand.torso_length_px)
        self.active = True
        self.gap_broken = False


def _assignment_cost(
    track: _TrackMemory,
    cand: PoseCandidate,
    timestamp: float,
    default_scale_px: float,
) -> float:
    pred = track.predicted_hip(timestamp)
    if pred is None or cand.hip_px is None:
        return math.inf
    scale = max(8.0, track.current_scale(default_scale_px))
    dist_norm = math.hypot(cand.hip_px[0] - pred[0], cand.hip_px[1] - pred[1]) / scale
    size_pen = 0.0
    if cand.torso_length_px is not None and track.scale_history:
        size_pen = 0.35 * abs(cand.torso_length_px - scale) / scale
    return dist_norm + size_pen


def _match_anchor_to_candidate(
    anchor: DancerAnchor,
    candidates: list[PoseCandidate],
    width: int,
    height: int,
) -> PoseCandidate | None:
    if not candidates:
        return None
    if anchor.pose_index is not None:
        for cand in candidates:
            if cand.pose_index == anchor.pose_index:
                return cand
    if anchor.hip_xy is not None:
        ax_px, ay_px = canonical_xy(anchor.hip_xy[0], anchor.hip_xy[1], width, height)
        valid = [c for c in candidates if c.hip_px is not None]
        if valid:
            return min(
                valid,
                key=lambda c: math.hypot(c.hip_px[0] - ax_px, c.hip_px[1] - ay_px),  # type: ignore[index]
            )
    return None


def _make_sample(
    timestamp: float,
    frame_index: int,
    cand: PoseCandidate,
    *,
    innovation: float | None = None,
    margin: float | None = None,
) -> TrackedDancerSample:
    status: SampleQualityStatus = (
        "measured" if cand.torso_eligible else ("estimated" if cand.hip_eligible else "unavailable")
    )
    return TrackedDancerSample(
        timestamp=timestamp,
        frame_index=frame_index,
        pose_index=cand.pose_index,
        raw_pose=dict(cand.pose),
        display_pose=dict(cand.pose),
        world_pose=dict(cand.world_pose) if cand.world_pose is not None else None,
        hip_px=cand.hip_px,
        shoulder_px=cand.shoulder_px,
        torso_length_px=cand.torso_length_px,
        torso_eligible=cand.torso_eligible,
        hip_eligible=cand.hip_eligible,
        head_eligible=cand.head_eligible,
        view_usable=cand.torso_eligible,
        interpolated_display=False,
        innovation_torso_lengths=innovation,
        assignment_margin_torso_lengths=margin,
        quality_status=status,
    )


def _fixed_scale_for_track(
    samples: list[TrackedDancerSample | None],
    baseline_passages: list[tuple[float, float]] | None,
) -> float | None:
    torso_samples = [
        s for s in samples if s is not None and s.torso_eligible and s.torso_length_px is not None
    ]
    if not torso_samples:
        return None
    if baseline_passages:
        base_vals = [
            float(s.torso_length_px)
            for s in torso_samples
            if s.torso_length_px is not None
            and any(t0 <= s.timestamp <= t1 for t0, t1 in baseline_passages)
        ]
        if len(base_vals) >= 3:
            return float(np.median(base_vals))
    return float(np.median([float(s.torso_length_px) for s in torso_samples if s.torso_length_px is not None]))


def _enforce_fixed_scale_eligibility(
    sample: TrackedDancerSample | None,
    fixed_scale_px: float | None,
) -> TrackedDancerSample | None:
    if sample is None or not sample.torso_eligible or sample.torso_length_px is None or fixed_scale_px is None:
        return sample
    ratio = sample.torso_length_px / max(1e-6, fixed_scale_px)
    if 0.45 <= ratio <= 2.2:
        return sample
    return replace(
        sample,
        torso_eligible=False,
        view_usable=False,
        quality_status="estimated" if sample.hip_eligible else "unavailable",
    )


@dataclass
class CoupleTrackingOutput:
    """Tracked timeline for A and B plus fixed scales and identity break diagnostics."""

    frames: list[PairedFrameSample]
    fixed_scale_a_px: float | None
    fixed_scale_b_px: float | None
    fixed_pair_scale_px: float | None
    diagnostics: list[dict[str, Any]]
    candidate_previews: list[CandidatePreview]


def track_couple_frames(
    observations: list[tuple[int, FrameObservation]],
    cfg: Config,
    *,
    analysis_region: dict[str, float] | None = None,
    anchors: list[DancerAnchor] | None = None,
    baseline_passages: list[tuple[float, float]] | None = None,
) -> CoupleTrackingOutput:
    """Track Dancer A and Dancer B with 1-to-1 gated association across frames."""
    sorted_obs = sorted(observations, key=lambda item: (item[1].timestamp, item[0]))
    if not sorted_obs:
        return CoupleTrackingOutput([], None, None, None, [], [])

    da_cfg = cfg.dancer_analysis
    max_innov = float(da_cfg.identity_max_innovation_torso_lengths)
    margin_min = float(da_cfg.identity_margin_torso_lengths)
    max_gap = float(da_cfg.max_observation_gap_s)
    shot_cut_min_disp = max(1.5, 2.0 * max_innov)
    shot_cut_frac = 0.6

    pending_anchors = sorted(anchors or [], key=lambda a: a.timestamp)
    consumed_anchors: set[int] = set()

    track_a = _TrackMemory("A")
    track_b = _TrackMemory("B")

    paired_frames: list[PairedFrameSample] = []
    diagnostics: list[dict[str, Any]] = []
    prev_ts: float | None = None

    for frame_index, obs in sorted_obs:
        t = float(obs.timestamp)
        default_scale = max(20.0, 0.22 * max(1, obs.height))
        candidates = filter_and_deduplicate_poses(obs, cfg, analysis_region)

        shot_cut = prev_ts is not None and (t <= prev_ts or (t - prev_ts) > 1.5)
        long_gap = prev_ts is not None and (t - prev_ts) > max_gap
        prev_ts = t

        if not shot_cut and candidates:
            active_mems = [m for m in (track_a, track_b) if m.active and m.last_hip_px is not None]
            if active_mems:
                cand_hips = [c.hip_px for c in candidates if c.hip_px is not None]
                if cand_hips:
                    jumped = 0
                    for m in active_mems:
                        assert m.last_hip_px is not None
                        sc = max(8.0, m.current_scale(default_scale))
                        nearest = min(
                            math.hypot(ch[0] - m.last_hip_px[0], ch[1] - m.last_hip_px[1]) / sc
                            for ch in cand_hips
                        )
                        if nearest >= shot_cut_min_disp:
                            jumped += 1
                    if (jumped / len(active_mems)) >= shot_cut_frac:
                        shot_cut = True
                        diagnostics.append(
                            {
                                "type": "shot_cut",
                                "timestamp": t,
                                "frame_index": frame_index,
                                "message": f"Shot or camera cut detected at {t:.2f}s.",
                            }
                        )

        if shot_cut:
            track_a.active = False
            track_b.active = False
            track_a.gap_broken = True
            track_b.gap_broken = True
            track_a.vel_px = (0.0, 0.0)
            track_b.vel_px = (0.0, 0.0)
            track_a.scale_history.clear()
            track_b.scale_history.clear()

        for t_mem in (track_a, track_b):
            if t_mem.last_time is not None and (t - t_mem.last_time) > max_gap:
                t_mem.gap_broken = True
                if (t - t_mem.last_time) > max(0.60, 3.0 * max_gap):
                    t_mem.vel_px = (0.0, 0.0)

        # Apply explicit user anchors matching this frame or within 0.15s of timestamp
        anchored_a: PoseCandidate | None = None
        anchored_b: PoseCandidate | None = None
        for idx_a, anchor in enumerate(pending_anchors):
            if idx_a in consumed_anchors:
                continue
            matches_frame = anchor.frame_index == frame_index or abs(anchor.timestamp - t) <= 0.15
            if not matches_frame:
                continue
            matched = _match_anchor_to_candidate(anchor, candidates, obs.width, obs.height)
            if matched is not None:
                if anchor.dancer_id == "A" and anchored_a is None:
                    anchored_a = matched
                    consumed_anchors.add(idx_a)
                elif anchor.dancer_id == "B" and anchored_b is None:
                    anchored_b = matched
                    consumed_anchors.add(idx_a)

        if anchored_a is not None:
            track_a.update(t, anchored_a)
        if anchored_b is not None:
            track_b.update(t, anchored_b)

        # Initialize uninitialized tracks cleanly when at least two distinct poses appear
        if not track_a.active and not track_b.active and len(candidates) >= 2:
            # Deterministic left-to-right initialization by canonical X so detector list order never dictates A/B
            ordered = sorted(
                candidates,
                key=lambda c: (c.hip_px[0] if c.hip_px is not None else c.bbox[0], c.pose_index),
            )
            init_a = anchored_a or ordered[0]
            remaining = [c for c in ordered if c.pose_index != init_a.pose_index]
            init_b = anchored_b or (remaining[0] if remaining else None)
            track_a.update(t, init_a)
            sa = _make_sample(t, frame_index, init_a)
            sb = None
            if init_b is not None:
                track_b.update(t, init_b)
                sb = _make_sample(t, frame_index, init_b)
            paired_frames.append(
                PairedFrameSample(
                    timestamp=t,
                    frame_index=frame_index,
                    width=obs.width,
                    height=obs.height,
                    dancer_a=sa,
                    dancer_b=sb,
                    pair_torso_eligible=bool(sa.torso_eligible and sb is not None and sb.torso_eligible),
                    pair_hip_eligible=bool(sa.hip_eligible and sb is not None and sb.hip_eligible),
                    shot_cut=shot_cut,
                    identity_break=False,
                )
            )
            continue

        # Standard 1-to-1 gated association step
        used_indices: set[int] = set()
        sample_a: TrackedDancerSample | None = None
        sample_b: TrackedDancerSample | None = None
        identity_break = False

        if anchored_a is not None:
            used_indices.add(anchored_a.pose_index)
            sample_a = _make_sample(t, frame_index, anchored_a, innovation=0.0, margin=1.0)
        if anchored_b is not None and anchored_b.pose_index not in used_indices:
            used_indices.add(anchored_b.pose_index)
            sample_b = _make_sample(t, frame_index, anchored_b, innovation=0.0, margin=1.0)

        rem_cands = [c for c in candidates if c.pose_index not in used_indices]
        need_a = track_a.active and sample_a is None
        need_b = track_b.active and sample_b is None

        if need_a and need_b and len(rem_cands) >= 2:
            # Evaluate all 1-to-1 assignments among active candidates
            assignments: list[tuple[float, PoseCandidate, PoseCandidate]] = []
            for ca in rem_cands:
                cost_a = _assignment_cost(track_a, ca, t, default_scale)
                if cost_a > max_innov * (1.6 if track_a.gap_broken else 1.0):
                    continue
                for cb in rem_cands:
                    if cb.pose_index == ca.pose_index:
                        continue
                    cost_b = _assignment_cost(track_b, cb, t, default_scale)
                    if cost_b > max_innov * (1.6 if track_b.gap_broken else 1.0):
                        continue
                    assignments.append((cost_a + cost_b, ca, cb))

            assignments.sort(key=lambda item: item[0])
            if assignments:
                best_cost, best_a, best_b = assignments[0]
                alt_cost = assignments[1][0] if len(assignments) > 1 else math.inf
                margin = alt_cost - best_cost
                # If best and swapped assignment are too close (< margin_min), treat as ambiguous -> unknown
                if len(assignments) > 1 and margin < margin_min:
                    identity_break = True
                    diagnostics.append(
                        {
                            "type": "ambiguous_identity",
                            "timestamp": t,
                            "frame_index": frame_index,
                            "margin": float(margin),
                            "message": f"Identity crossing ambiguous at {t:.2f}s (margin {margin:.2f} < {margin_min:.2f}); marked unknown.",
                        }
                    )
                else:
                    innov_a = _assignment_cost(track_a, best_a, t, default_scale)
                    innov_b = _assignment_cost(track_b, best_b, t, default_scale)
                    track_a.update(t, best_a)
                    track_b.update(t, best_b)
                    sample_a = _make_sample(t, frame_index, best_a, innovation=innov_a, margin=margin if math.isfinite(margin) else None)
                    sample_b = _make_sample(t, frame_index, best_b, innovation=innov_b, margin=margin if math.isfinite(margin) else None)
            elif long_gap or shot_cut:
                identity_break = True
        elif rem_cands and (need_a or need_b):
            # Single-candidate or single-track case: gate strictly and check ambiguity between A and B
            best_cand = rem_cands[0]
            if need_a and not need_b:
                cands_sorted = sorted(
                    ((_assignment_cost(track_a, c, t, default_scale), c) for c in rem_cands),
                    key=lambda item: item[0],
                )
                cost, chosen = cands_sorted[0]
                if cost <= max_innov * (1.6 if track_a.gap_broken else 1.0):
                    track_a.update(t, chosen)
                    sample_a = _make_sample(t, frame_index, chosen, innovation=cost)
            elif need_b and not need_a:
                cands_sorted = sorted(
                    ((_assignment_cost(track_b, c, t, default_scale), c) for c in rem_cands),
                    key=lambda item: item[0],
                )
                cost, chosen = cands_sorted[0]
                if cost <= max_innov * (1.6 if track_b.gap_broken else 1.0):
                    track_b.update(t, chosen)
                    sample_b = _make_sample(t, frame_index, chosen, innovation=cost)
            elif need_a and need_b and len(rem_cands) == 1:
                cost_a = _assignment_cost(track_a, best_cand, t, default_scale)
                cost_b = _assignment_cost(track_b, best_cand, t, default_scale)
                diff = abs(cost_a - cost_b)
                if min(cost_a, cost_b) <= max_innov and diff >= margin_min:
                    if cost_a < cost_b:
                        track_a.update(t, best_cand)
                        sample_a = _make_sample(t, frame_index, best_cand, innovation=cost_a, margin=diff)
                    else:
                        track_b.update(t, best_cand)
                        sample_b = _make_sample(t, frame_index, best_cand, innovation=cost_b, margin=diff)
                else:
                    identity_break = True

        # Handle uninitialized single track when only 1 dancer appears initially
        unused_cands = [c for c in candidates if c.pose_index not in used_indices]
        if not track_a.active and unused_cands:
            ordered = sorted(
                unused_cands,
                key=lambda c: (c.hip_px[0] if c.hip_px is not None else c.bbox[0], c.pose_index),
            )
            chosen_a = anchored_a or ordered[0]
            track_a.update(t, chosen_a)
            used_indices.add(chosen_a.pose_index)
            sample_a = _make_sample(t, frame_index, chosen_a)
        elif track_a.active and not track_b.active:
            leftover = [
                c
                for c in candidates
                if c.pose_index not in used_indices
                and (sample_a is None or c.pose_index != sample_a.pose_index)
            ]
            if leftover:
                chosen_b = anchored_b or leftover[0]
                track_b.update(t, chosen_b)
                used_indices.add(chosen_b.pose_index)
                sample_b = _make_sample(t, frame_index, chosen_b)

        paired_frames.append(
            PairedFrameSample(
                timestamp=t,
                frame_index=frame_index,
                width=obs.width,
                height=obs.height,
                dancer_a=sample_a,
                dancer_b=sample_b,
                pair_torso_eligible=bool(
                    sample_a is not None
                    and sample_a.torso_eligible
                    and sample_b is not None
                    and sample_b.torso_eligible
                ),
                pair_hip_eligible=bool(
                    sample_a is not None
                    and sample_a.hip_eligible
                    and sample_b is not None
                    and sample_b.hip_eligible
                ),
                shot_cut=shot_cut,
                identity_break=identity_break,
            )
        )

    fixed_a = _fixed_scale_for_track([f.dancer_a for f in paired_frames], baseline_passages)
    fixed_b = _fixed_scale_for_track([f.dancer_b for f in paired_frames], baseline_passages)
    fixed_pair = (
        0.5 * (fixed_a + fixed_b)
        if fixed_a is not None and fixed_b is not None
        else (fixed_a or fixed_b)
    )

    adjusted_frames: list[PairedFrameSample] = []
    for pf in paired_frames:
        adj_a = _enforce_fixed_scale_eligibility(pf.dancer_a, fixed_a)
        adj_b = _enforce_fixed_scale_eligibility(pf.dancer_b, fixed_b)
        adjusted_frames.append(
            PairedFrameSample(
                timestamp=pf.timestamp,
                frame_index=pf.frame_index,
                width=pf.width,
                height=pf.height,
                dancer_a=adj_a,
                dancer_b=adj_b,
                pair_torso_eligible=bool(adj_a and adj_a.torso_eligible and adj_b and adj_b.torso_eligible),
                pair_hip_eligible=bool(adj_a and adj_a.hip_eligible and adj_b and adj_b.hip_eligible),
                shot_cut=pf.shot_cut,
                identity_break=pf.identity_break,
            )
        )

    previews: list[CandidatePreview] = []
    for did, getter in (("A", lambda p: p.dancer_a), ("B", lambda p: p.dancer_b)):
        samples_d = [getter(p) for p in adjusted_frames if getter(p) is not None and getter(p).hip_px is not None]
        if samples_d:
            rep = next((s for s in samples_d if s.torso_eligible), samples_d[0])
            w = max(1, adjusted_frames[0].width - 1)
            h = max(1, adjusted_frames[0].height - 1)
            assert rep.hip_px is not None
            hip_norm = (rep.hip_px[0] / w, -rep.hip_px[1] / h)
            sh_norm = (
                (rep.shoulder_px[0] / w, -rep.shoulder_px[1] / h)
                if rep.shoulder_px is not None
                else None
            )
            previews.append(
                CandidatePreview(
                    candidate_id=did,
                    pose_index=rep.pose_index or 0,
                    timestamp=rep.timestamp,
                    frame_index=rep.frame_index,
                    hip_xy=hip_norm,
                    shoulder_xy=sh_norm,
                    torso_length_px=rep.torso_length_px,
                    bbox=(
                        max(0.0, hip_norm[0] - 0.12),
                        max(0.0, hip_norm[1] - 0.28),
                        min(1.0, hip_norm[0] + 0.12),
                        min(1.0, hip_norm[1] + 0.28),
                    ),
                    eligible_duration_s=0.0,
                    coverage=len(samples_d) / max(1, len(adjusted_frames)),
                )
            )

    return CoupleTrackingOutput(
        frames=adjusted_frames,
        fixed_scale_a_px=fixed_a,
        fixed_scale_b_px=fixed_b,
        fixed_pair_scale_px=fixed_pair,
        diagnostics=diagnostics,
        candidate_previews=previews,
    )


def extract_single_target_observations(
    observations: list[tuple[int, FrameObservation]],
    cfg: Config,
    *,
    anchor: DancerAnchor | None = None,
    analysis_region: dict[str, float] | None = None,
) -> list[tuple[int, FrameObservation]]:
    """Filter multi-person observations down to one tracked primary dancer for Single mode."""
    anchors = [DancerAnchor(
        dancer_id="A",
        timestamp=anchor.timestamp,
        frame_index=anchor.frame_index,
        hip_xy=anchor.hip_xy,
        pose_index=anchor.pose_index,
    )] if anchor is not None else None

    if (
        anchors is None
        and analysis_region is None
        and all(len(obs.poses) <= 1 for _, obs in observations)
    ):
        # Preserve strictly single-pose clips untouched when no manual anchors or region were given.
        return list(observations)

    tracked = track_couple_frames(observations, cfg, analysis_region=analysis_region, anchors=anchors)
    use_b = False
    if anchor is not None and anchor.dancer_id == "B":
        use_b = True
    elif anchor is None:
        count_a = sum(1 for pf in tracked.frames if pf.dancer_a is not None and pf.dancer_a.torso_eligible)
        count_b = sum(1 for pf in tracked.frames if pf.dancer_b is not None and pf.dancer_b.torso_eligible)
        use_b = count_b > count_a

    obs_by_key = {(idx, obs.timestamp): obs for idx, obs in observations}
    out: list[tuple[int, FrameObservation]] = []
    for pf in tracked.frames:
        chosen = pf.dancer_b if use_b else pf.dancer_a
        orig = obs_by_key.get((pf.frame_index, pf.timestamp))
        width = orig.width if orig is not None else pf.width
        height = orig.height if orig is not None else pf.height
        if chosen is not None and chosen.raw_pose is not None:
            worlds = [chosen.world_pose] if chosen.world_pose is not None else []
            out.append(
                (pf.frame_index, FrameObservation(pf.timestamp, width, height, [chosen.raw_pose], worlds))
            )
        else:
            out.append((pf.frame_index, FrameObservation(pf.timestamp, width, height, [], [])))
    return out


def _pose_torso_mid_norm(pose: dict[str, Landmark], conf: float) -> tuple[float, float] | None:
    keys = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    pts = [pose.get(k) for k in keys]
    if not all(reliable(p, conf) and p is not None for p in pts):
        return None
    valid_pts = [p for p in pts if p is not None]
    return (
        sum(float(p.x) for p in valid_pts) / 4.0,
        sum(float(p.y) for p in valid_pts) / 4.0,
    )


def _translate_pose(pose: dict[str, Landmark], dx: float, dy: float) -> dict[str, Landmark]:
    return {
        name: replace(lm, x=float(lm.x) + dx, y=float(lm.y) + dy)
        for name, lm in pose.items()
    }


def _wide_ankle_span_for_pair(pose: dict[str, Landmark], confidence: float) -> bool:
    lh, rh = pose.get("left_hip"), pose.get("right_hip")
    la, ra = pose.get("left_ankle"), pose.get("right_ankle")
    if not (reliable(lh, confidence) and reliable(rh, confidence) and lh is not None and rh is not None):
        return False
    if not (reliable(la, confidence) and reliable(ra, confidence) and la is not None and ra is not None):
        return False
    hip_span = max(0.02, abs(float(la.x) * 0.0 + float(lh.x) - float(rh.x)))
    ankle_span = abs(float(la.x) - float(ra.x))
    return ankle_span > 1.45 * hip_span


def _mp_landmarks_to_pose(
    landmarks: Any,
    *,
    crop_x0: float = 0.0,
    crop_y0: float = 0.0,
    crop_w: float = 1.0,
    crop_h: float = 1.0,
) -> dict[str, Landmark]:
    return {
        name: Landmark(
            crop_x0 + float(landmarks[i].x) * crop_w,
            crop_y0 + float(landmarks[i].y) * crop_h,
            float(landmarks[i].visibility),
            float(landmarks[i].presence),
            float(getattr(landmarks[i], "z", 0.0) or 0.0),
        )
        for name, i in NAMES.items()
    }


def _pose_head_x(pose: dict[str, Landmark], conf: float) -> float | None:
    nose = pose.get("nose")
    if reliable(nose, conf) and nose is not None:
        return float(nose.x)
    ears = [pose.get(k) for k in ("left_ear", "right_ear")]
    rel = [float(e.x) for e in ears if reliable(e, conf) and e is not None]
    if rel:
        return sum(rel) / len(rel)
    return None


def resolve_close_embrace_frame_poses(
    image: np.ndarray,
    primary_pose: dict[str, Landmark],
    detector: Any,
    width: int,
    height: int,
    cfg: Config,
) -> list[dict[str, Landmark]]:
    """Split a merged close-embrace single detection into left/right dancer poses when valid.

    In close embrace, MediaPipe's single-stage detector can merge both bodies into one pose.
    This helper crops tightly around the primary detection (eliminating mirror reflections or
    distant bystanders) and masks the right/left half of the torso midline before running
    ``detector`` in IMAGE mode, accepting two poses only when strict anatomical gates pass.
    """
    import cv2
    import mediapipe as mp

    conf = float(cfg.confidence)
    l_sh, r_sh = primary_pose.get("left_shoulder"), primary_pose.get("right_shoulder")
    l_hip, r_hip = primary_pose.get("left_hip"), primary_pose.get("right_hip")
    if not (
        reliable(l_sh, conf) and reliable(r_sh, conf) and reliable(l_hip, conf) and reliable(r_hip, conf)
        and l_sh is not None and r_sh is not None and l_hip is not None and r_hip is not None
    ):
        return [primary_pose]

    sh_x = 0.5 * (float(l_sh.x) + float(r_sh.x))
    sh_y = 0.5 * (float(l_sh.y) + float(r_sh.y))
    hip_x = 0.5 * (float(l_hip.x) + float(r_hip.x))
    hip_y = 0.5 * (float(l_hip.y) + float(r_hip.y))
    torso_px = math.hypot((sh_x - hip_x) * width, (sh_y - hip_y) * height)
    min_torso_px = max(12.0, 0.5 * float(cfg.min_torso) * max(1, height))
    if not math.isfinite(torso_px) or torso_px < min_torso_px:
        return [primary_pose]

    cx = 0.5 * (sh_x + hip_x)
    ys = [
        float(lm.y)
        for lm in primary_pose.values()
        if math.isfinite(lm.y) and (lm.visibility is None or lm.visibility >= 0.4)
    ]
    top_y = max(0.0, (min(ys) if ys else min(sh_y, hip_y)) - 0.08)
    bot_y = min(1.0, (max(ys) if ys else max(sh_y, hip_y)) + 0.05)

    # Crop tightly around the embrace cluster so mirrors / distant passers-by cannot leak into masked runs.
    roi_half_x = max(0.14, 1.8 * torso_px / max(1, width))
    cx0 = max(0.0, cx - roi_half_x)
    cx1 = min(1.0, cx + roi_half_x)
    px0, px1 = int(cx0 * width), math.ceil(cx1 * width)
    py0, py1 = int(top_y * height), math.ceil(bot_y * height)
    if px1 - px0 < 32 or py1 - py0 < 48:
        return [primary_pose]

    crop_rgb = cv2.cvtColor(image[py0:py1, px0:px1], cv2.COLOR_BGR2RGB)
    cw = max(1, px1 - px0)
    ch = max(1, py1 - py0)
    crop_x0 = px0 / max(1, width)
    crop_y0 = py0 / max(1, height)
    crop_w = cw / max(1, width)
    crop_h = ch / max(1, height)

    mid_crop_x = max(2, min(cw - 2, round(((cx - crop_x0) / max(1e-6, crop_w)) * cw)))
    span_px = max(12, round((1.3 * torso_px / max(1, width)) * width))

    m_left = crop_rgb.copy()
    m_left[:, min(cw, mid_crop_x + 2):min(cw, mid_crop_x + span_px)] = 128
    rl = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=m_left))

    m_right = crop_rgb.copy()
    m_right[:, max(0, mid_crop_x - span_px):max(0, mid_crop_x - 2)] = 128
    rr = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=m_right))

    if not getattr(rl, "pose_landmarks", None) or not getattr(rr, "pose_landmarks", None):
        return [primary_pose]

    pose_l = _mp_landmarks_to_pose(rl.pose_landmarks[0], crop_x0=crop_x0, crop_y0=crop_y0, crop_w=crop_w, crop_h=crop_h)
    pose_r = _mp_landmarks_to_pose(rr.pose_landmarks[0], crop_x0=crop_x0, crop_y0=crop_y0, crop_w=crop_w, crop_h=crop_h)

    cand_l = extract_pose_candidate(0, pose_l, None, width, height, cfg)
    cand_r = extract_pose_candidate(1, pose_r, None, width, height, cfg)
    if (
        not cand_l.torso_eligible
        or not cand_r.torso_eligible
        or cand_l.hip_norm is None
        or cand_r.hip_norm is None
        or cand_l.hip_px is None
        or cand_r.hip_px is None
        or cand_l.shoulder_px is None
        or cand_r.shoulder_px is None
        or cand_l.torso_length_px is None
        or cand_r.torso_length_px is None
    ):
        return [primary_pose]

    hlx = cand_l.hip_norm[0]
    hrx = cand_r.hip_norm[0]
    if not (hlx < cx < hrx):
        return [primary_pose]

    mean_tl = 0.5 * (cand_l.torso_length_px + cand_r.torso_length_px)
    if mean_tl < 1.0:
        return [primary_pose]
    ratio = cand_l.torso_length_px / cand_r.torso_length_px
    if not (0.72 <= ratio <= 1.38):
        return [primary_pose]

    hip_sep_tl = (cand_r.hip_px[0] - cand_l.hip_px[0]) / mean_tl
    sh_sep_tl = (cand_r.shoulder_px[0] - cand_l.shoulder_px[0]) / mean_tl
    if not (0.38 <= hip_sep_tl <= 1.45 and 0.32 <= sh_sep_tl <= 1.45):
        return [primary_pose]

    tilt_l = abs(math.degrees(math.atan2(cand_l.shoulder_px[0] - cand_l.hip_px[0], cand_l.shoulder_px[1] - cand_l.hip_px[1])))
    tilt_r = abs(math.degrees(math.atan2(cand_r.shoulder_px[0] - cand_r.hip_px[0], cand_r.shoulder_px[1] - cand_r.hip_px[1])))
    if tilt_l > 25.0 or tilt_r > 25.0:
        return [primary_pose]

    head_lx = _pose_head_x(pose_l, conf)
    head_rx = _pose_head_x(pose_r, conf)
    if (
        head_lx is not None
        and head_rx is not None
        and abs(head_rx - head_lx) * width < 0.015 * mean_tl
        and hip_sep_tl < 0.44
    ):
        return [primary_pose]

    return [pose_l, pose_r]
