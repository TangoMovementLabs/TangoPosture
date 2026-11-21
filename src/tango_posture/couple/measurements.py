"""Couple measurements across coordination, step timing, transitions, posture axes, and turns."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import numpy as np

from ..config import Config
from ..domain import VALID_POSE_ORIGINS, FrameObservation, PoseOrigin, normalize_pose_origin
from ..geometry import head_horizontal_alignment
from ..movement import analyse_movement_observations, wrapped_angle_diff
from ..pose.interpolation import interpolate_tracked_dancer_poses
from .classification import classify_opening_window, detect_post_window_scene_changes
from .models import (
    AutoDecisionResult,
    CoupleAnalysisRequest,
    CoupleAnalysisResult,
    CoupleAnnotationItem,
    CoupleAnnotations,
    CoupleEvent,
    CoupleSummaryValue,
    PairedFrameSample,
    PoseOriginPolicy,
    SummaryEvidenceStatus,
    TrackedDancerSample,
)
from .overlay import smooth_couple_frame_overlays
from .tracking import CoupleTrackingOutput, canonical_xy, track_couple_frames


def compute_centred_regression_velocity(
    timestamps: list[float],
    values: list[float | None],
    eligible_mask: list[bool],
    break_mask: list[bool],
    *,
    half_window_s: float = 0.15,
    max_gap_s: float = 0.20,
) -> list[float | None]:
    """Compute centred linear regression slope ``ds/dt`` requiring two-sided support and >=3 samples."""
    n = len(timestamps)
    velocities: list[float | None] = [None] * n
    if n < 3:
        return velocities

    # Partition timeline into contiguous segments bounded by breaks or > max_gap_s gaps
    segments: list[list[int]] = []
    current: list[int] = []
    last_eligible_t: float | None = None
    for idx in range(n):
        if break_mask[idx]:
            if current:
                segments.append(current)
                current = []
            last_eligible_t = None
        if eligible_mask[idx] and values[idx] is not None and math.isfinite(float(values[idx])):  # type: ignore[arg-type]
            t = float(timestamps[idx])
            if last_eligible_t is not None and (t - last_eligible_t) > max_gap_s + 1e-9:
                if current:
                    segments.append(current)
                current = []
            current.append(idx)
            last_eligible_t = t
    if current:
        segments.append(current)

    for seg in segments:
        if len(seg) < 3:
            continue
        seg_t = [float(timestamps[k]) for k in seg]
        seg_v = [float(values[k]) for k in seg]  # type: ignore[arg-type]
        for pos, idx in enumerate(seg):
            t_center = seg_t[pos]
            in_win = [
                (seg_t[j], seg_v[j])
                for j in range(len(seg))
                if abs(seg_t[j] - t_center) <= half_window_s + 1e-9
            ]
            if len(in_win) < 3:
                continue
            has_left = any(tw < t_center - 1e-9 for tw, _ in in_win)
            has_right = any(tw > t_center + 1e-9 for tw, _ in in_win)
            if not (has_left and has_right):
                continue
            ts_arr = np.array([tw - t_center for tw, _ in in_win], dtype=float)
            vs_arr = np.array([vw for _, vw in in_win], dtype=float)
            denom = float(np.dot(ts_arr, ts_arr))
            if denom <= 1e-12:
                continue
            slope = float(np.dot(ts_arr, vs_arr - float(np.mean(vs_arr))) / denom)
            velocities[idx] = slope
    return velocities


def _infer_travel_axis(
    frames: list[PairedFrameSample],
    annotations: CoupleAnnotations,
) -> tuple[tuple[float, float], str]:
    """Return unit travel vector in canonical pixel space and its source."""
    if len(annotations.travel_direction_points) >= 2:
        w = frames[0].width if frames else 640
        h = frames[0].height if frames else 480
        p0 = canonical_xy(
            annotations.travel_direction_points[0][0],
            annotations.travel_direction_points[0][1],
            w,
            h,
        )
        p1 = canonical_xy(
            annotations.travel_direction_points[1][0],
            annotations.travel_direction_points[1][1],
            w,
            h,
        )
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        norm = math.hypot(dx, dy)
        if norm > 1e-6:
            return ((dx / norm, dy / norm), "user_defined")

    pts: list[tuple[float, float]] = []
    for pf in frames:
        if pf.pair_hip_eligible and pf.dancer_a and pf.dancer_b:
            ha, hb = pf.dancer_a.hip_px, pf.dancer_b.hip_px
            if ha is not None and hb is not None:
                pts.append((0.5 * (ha[0] + hb[0]), 0.5 * (ha[1] + hb[1])))
        elif pf.dancer_a and pf.dancer_a.hip_eligible and pf.dancer_a.hip_px is not None:
            pts.append(pf.dancer_a.hip_px)
    if len(pts) >= 2:
        arr = np.array(pts, dtype=float)
        net = arr[-1] - arr[0]
        centered = arr - np.mean(arr, axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, int(np.argmax(eigvals))]
        if float(np.dot(principal, net)) < 0.0:
            principal = -principal
        norm = float(np.hypot(principal[0], principal[1]))
        if norm > 1e-6:
            return ((float(principal[0] / norm), float(principal[1] / norm)), "inferred_dominant")
        net_norm = float(np.hypot(net[0], net[1]))
        if net_norm > 1e-6:
            return ((float(net[0] / net_norm), float(net[1] / net_norm)), "inferred_dominant")
    return ((1.0, 0.0), "inferred_dominant")


def _interpolate_crossing(t0: float, v0: float, t1: float, v1: float, thresh: float) -> float:
    dv = v1 - v0
    if abs(dv) <= 1e-9 or t1 <= t0:
        return 0.5 * (t0 + t1)
    frac = max(0.0, min(1.0, (thresh - v0) / dv))
    return t0 + frac * (t1 - t0)


def _stays_above(
    valid: list[tuple[float, float]],
    start_pos: int,
    threshold: float,
    persistence_s: float,
) -> bool:
    t0 = valid[start_pos][0]
    for pos in range(start_pos, len(valid)):
        t, v = valid[pos]
        if v <= threshold:
            return False
        if (t - t0) >= persistence_s - 1e-6:
            return True
    return (valid[-1][0] - t0) >= persistence_s - 1e-6


def _stays_below(
    valid: list[tuple[float, float]],
    start_pos: int,
    threshold: float,
    persistence_s: float,
) -> bool:
    t0 = valid[start_pos][0]
    for pos in range(start_pos, len(valid)):
        t, v = valid[pos]
        if v >= threshold:
            return False
        if (t - t0) >= persistence_s - 1e-6:
            return True
    return (valid[-1][0] - t0) >= persistence_s - 1e-6


def _detect_motion_events_for_track(
    subject: str,
    timestamps: list[float],
    velocities: list[float | None],
    break_mask: list[bool],
    *,
    start_speed: float,
    stop_speed: float,
    slowing_fraction: float,
    persistence_s: float,
    max_gap_s: float,
    sample_origins: Sequence[str | None] | None = None,
) -> list[CoupleEvent]:
    """Detect start/restart, slowing, and settled events along one subject's travel speed curve."""
    events: list[CoupleEvent] = []
    # Partition into contiguous finite speed segments
    segments: list[list[tuple[float, float, str | None]]] = []
    cur: list[tuple[float, float, str | None]] = []
    last_t: float | None = None
    for idx, (t, v) in enumerate(zip(timestamps, velocities)):
        if break_mask[idx]:
            if cur:
                segments.append(cur)
                cur = []
            last_t = None
        if v is None or not math.isfinite(v):
            continue
        if last_t is not None and (t - last_t) > max_gap_s + 1e-9:
            if cur:
                segments.append(cur)
            cur = []
        org = sample_origins[idx] if sample_origins is not None and idx < len(sample_origins) else None
        cur.append((float(t), abs(float(v)), org))
        last_t = float(t)
    if cur:
        segments.append(cur)

    for seg in segments:
        if len(seg) < 3:
            continue
        seg_tv = [(t, sp) for t, sp, _ in seg]
        state = "IDLE"
        active_speeds: list[float] = []
        slowing_emitted = False
        if (
            len(seg) >= 3
            and seg[0][1] > start_speed
            and _stays_above(seg_tv, 0, start_speed, persistence_s)
        ):
            state = "TRAVELLING"

        def _bracket_status(
            p0: int,
            p1: int,
            segment: Sequence[tuple[float, float, str | None]] = seg,
        ) -> tuple[str, str | None]:
            orgs = [
                segment[k][2]
                for k in range(max(0, p0), min(len(segment), p1 + 1))
                if segment[k][2]
            ]
            if "propagated" in orgs:
                return ("estimated", "propagated")
            if "crop_detected" in orgs:
                return ("measured", "crop_detected")
            return ("measured", orgs[0] if orgs else "detected")

        for pos, (t, sp, _) in enumerate(seg):
            if state == "IDLE":
                if (
                    sp > start_speed
                    and pos > 0
                    and seg[pos - 1][1] <= start_speed
                    and _stays_above(seg_tv, pos, start_speed, persistence_s)
                ):
                    t_prev, sp_prev, _ = seg[pos - 1]
                    t_cross = _interpolate_crossing(t_prev, sp_prev, t, sp, start_speed)
                    ev_status, ev_origin = _bracket_status(pos - 1, pos)
                    events.append(
                        CoupleEvent(
                            event_id=f"cand_{subject}_start_{len(events)}",
                            event_type="start",
                            subject=subject,
                            timestamp=round(t_cross, 6),
                            bracket_start=float(t_prev),
                            bracket_end=float(t),
                            value=round(start_speed, 4),
                            unit="torso_lengths/s",
                            status=ev_status,  # type: ignore[arg-type]
                            metadata={"pose_origin": ev_origin} if ev_origin else {},
                        )
                    )
                    state = "TRAVELLING"
                    active_speeds = [sp]
                    slowing_emitted = False
            elif state == "TRAVELLING":
                if sp >= start_speed:
                    active_speeds.append(sp)
                # Detect slowing before complete stop
                if not slowing_emitted and len(active_speeds) >= 3:
                    v_med = float(np.median(active_speeds))
                    slow_thresh = (1.0 - slowing_fraction) * v_med
                    if (
                        pos > 0
                        and v_med > start_speed
                        and sp < slow_thresh
                        and seg[pos - 1][1] >= slow_thresh
                        and _stays_below(seg_tv, pos, slow_thresh, persistence_s)
                    ):
                        t_prev, sp_prev, _ = seg[pos - 1]
                        t_cross = _interpolate_crossing(t_prev, sp_prev, t, sp, slow_thresh)
                        ev_status, ev_origin = _bracket_status(pos - 1, pos)
                        events.append(
                            CoupleEvent(
                                event_id=f"cand_{subject}_slow_{len(events)}",
                                event_type="slow",
                                subject=subject,
                                timestamp=round(t_cross, 6),
                                bracket_start=float(t_prev),
                                bracket_end=float(t),
                                value=round(slow_thresh, 4),
                                unit="torso_lengths/s",
                                status=ev_status,  # type: ignore[arg-type]
                                metadata={
                                    "steady_median_speed": round(v_med, 4),
                                    **({"pose_origin": ev_origin} if ev_origin else {}),
                                },
                            )
                        )
                        slowing_emitted = True

                # Detect settle
                if (
                    pos > 0
                    and sp < stop_speed
                    and seg[pos - 1][1] >= stop_speed
                    and _stays_below(seg_tv, pos, stop_speed, persistence_s)
                ):
                    t_prev, sp_prev, _ = seg[pos - 1]
                    t_cross = _interpolate_crossing(t_prev, sp_prev, t, sp, stop_speed)
                    ev_status, ev_origin = _bracket_status(pos - 1, pos)
                    events.append(
                        CoupleEvent(
                            event_id=f"cand_{subject}_settle_{len(events)}",
                            event_type="settle",
                            subject=subject,
                            timestamp=round(t_cross, 6),
                            bracket_start=float(t_prev),
                            bracket_end=float(t),
                            value=round(stop_speed, 4),
                            unit="torso_lengths/s",
                            status=ev_status,  # type: ignore[arg-type]
                            metadata={"pose_origin": ev_origin} if ev_origin else {},
                        )
                    )
                    state = "IDLE"
                    active_speeds = []
                    slowing_emitted = False
    return events


def _pair_coordination_events(
    events: list[dict[str, Any]],
    *,
    pairing_window_s: float,
    break_timestamps: list[float],
) -> list[dict[str, Any]]:
    """Pair A and B start/slow/settle events in the same phase with signed lag ``t_B - t_A``."""
    out = [dict(e) for e in events]
    for ev in out:
        if ev.get("event_type") in ("start", "slow", "settle"):
            ev["paired_event_id"] = None
            ev["lag_ms"] = None

    for etype in ("start", "slow", "settle"):
        ev_a = sorted(
            [e for e in out if e.get("event_type") == etype and e.get("subject") == "A"],
            key=lambda e: float(e["timestamp"]),
        )
        ev_b = sorted(
            [e for e in out if e.get("event_type") == etype and e.get("subject") == "B"],
            key=lambda e: float(e["timestamp"]),
        )
        pairs: list[tuple[float, int, int]] = []
        for ia, ea in enumerate(ev_a):
            ta = float(ea["timestamp"])
            for ib, eb in enumerate(ev_b):
                tb = float(eb["timestamp"])
                dt = abs(tb - ta)
                if dt > pairing_window_s + 1e-9:
                    continue
                t_lo, t_hi = min(ta, tb), max(ta, tb)
                if any(t_lo < bt < t_hi for bt in break_timestamps):
                    continue
                pairs.append((dt, ia, ib))
        pairs.sort(key=lambda item: item[0])
        used_a: set[int] = set()
        used_b: set[int] = set()
        for _, ia, ib in pairs:
            if ia in used_a or ib in used_b:
                continue
            used_a.add(ia)
            used_b.add(ib)
            ea = ev_a[ia]
            eb = ev_b[ib]
            lag_ms = (float(eb["timestamp"]) - float(ea["timestamp"])) * 1000.0
            ea["paired_event_id"] = eb["event_id"]
            eb["paired_event_id"] = ea["event_id"]
            ea["lag_ms"] = round(lag_ms, 2)
            eb["lag_ms"] = round(lag_ms, 2)
            ea["reason"] = ""
            eb["reason"] = ""

        for ia, ea in enumerate(ev_a):
            if ia not in used_a and not ea.get("reason"):
                ea["reason"] = "no_partner_event_within_pairing_window"
        for ib, eb in enumerate(ev_b):
            if ib not in used_b and not eb.get("reason"):
                eb["reason"] = "no_partner_event_within_pairing_window"
    return out


def _median_iqr(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return (None, None)
    arr = np.array(values, dtype=float)
    med = float(np.median(arr))
    if len(arr) == 1:
        return (med, 0.0)
    q75, q25 = np.percentile(arr, [75.0, 25.0])
    return (med, float(q75 - q25))


def _expand_beat_targets(
    auto_beats: list[float],
    annotations: CoupleAnnotations,
    confirmed_item_beats: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build beat target timestamps from user-confirmed or estimated beats with subdivisions."""
    subdiv = max(1, int(annotations.beat_subdivision))
    offset = float(annotations.beat_offset_s)
    combined_manual = list(annotations.manual_beats) + list(confirmed_item_beats or [])
    if combined_manual:
        base = sorted(float(t) for t in combined_manual)
        status = "user_confirmed"
    else:
        base = sorted(float(t) for t in auto_beats)
        status = "estimated"
    if not base:
        return []

    diffs = [base[i + 1] - base[i] for i in range(len(base) - 1) if base[i + 1] > base[i]]
    default_period = float(np.median(diffs)) if diffs else 0.5

    targets: list[dict[str, Any]] = []
    for idx, t0 in enumerate(base):
        period = (base[idx + 1] - t0) if idx + 1 < len(base) and base[idx + 1] > t0 else default_period
        for k in range(subdiv):
            tk = t0 + (k / subdiv) * period + offset
            targets.append(
                {
                    "timestamp": tk,
                    "period": period / subdiv,
                    "status": status,
                    "subdivision_index": k,
                }
            )
    return targets


def _attach_beat_offsets(events: list[dict[str, Any]], beat_targets: list[dict[str, Any]]) -> None:
    if not beat_targets:
        return
    beat_times = [float(b["timestamp"]) for b in beat_targets]
    for ev in events:
        if ev.get("event_type") not in ("contact", "start"):
            continue
        t = float(ev["timestamp"])
        best_idx = min(range(len(beat_times)), key=lambda i: abs(beat_times[i] - t))
        b = beat_targets[best_idx]
        dt = t - float(b["timestamp"])
        period = float(b["period"])
        ev["beat_offset_ms"] = round(dt * 1000.0, 2)
        ev["beat_fraction"] = round(dt / period, 4) if period > 1e-6 else None
        ev.setdefault("metadata", {})["nearest_beat_time"] = round(float(b["timestamp"]), 4)
        ev["metadata"]["beat_status"] = b["status"]


_CANONICAL_STATUSES = ("measured", "estimated", "user_confirmed", "unavailable")


def _count_statuses(statuses: Sequence[str | None]) -> dict[str, int]:
    counts = {k: 0 for k in _CANONICAL_STATUSES}
    for st in statuses:
        s = str(st or "unavailable")
        if s in counts:
            counts[s] += 1
        elif s:
            counts[s] = counts.get(s, 0) + 1
    return counts


def _count_origins(origins: Sequence[str | None]) -> dict[str, int]:
    counts = {org: 0 for org in VALID_POSE_ORIGINS}
    for org in origins:
        if isinstance(org, str) and org in counts:
            counts[org] += 1
    return counts


def _summarize_evidence_status(
    statuses: Sequence[str | None],
    *,
    eligible_count: int | None = None,
    min_required: int = 1,
    unavailable_status: SummaryEvidenceStatus = "unavailable",
) -> tuple[SummaryEvidenceStatus, dict[str, int]]:
    counts = _count_statuses(statuses)
    active_kinds = [
        kind for kind in ("measured", "estimated", "user_confirmed") if counts.get(kind, 0) > 0
    ]
    eff_eligible = (
        sum(counts.get(kind, 0) for kind in ("measured", "estimated", "user_confirmed"))
        if eligible_count is None
        else int(eligible_count)
    )
    if eff_eligible < min_required or not active_kinds:
        return (unavailable_status, counts)
    if len(active_kinds) == 1:
        return (active_kinds[0], counts)  # type: ignore[return-value]
    return ("mixed", counts)


def _partition_contiguous_eligible_intervals(
    timestamps: Sequence[float],
    v_pair: Sequence[float | None],
    pair_hip_eligible: Sequence[bool],
    break_mask: Sequence[bool],
    max_gap_s: float,
) -> list[list[tuple[int, float, float]]]:
    """Split velocity samples into contiguous eligible intervals broken by cuts, gaps, or drop-outs."""
    intervals: list[list[tuple[int, float, float]]] = []
    cur: list[tuple[int, float, float]] = []
    last_t: float | None = None
    for idx, (t, v, elig, brk) in enumerate(zip(timestamps, v_pair, pair_hip_eligible, break_mask)):
        if brk:
            if cur:
                intervals.append(cur)
                cur = []
            last_t = None
        if not elig or v is None or not math.isfinite(float(v)):
            if cur:
                intervals.append(cur)
                cur = []
            last_t = None
            continue
        tf = float(t)
        if last_t is not None and (tf - last_t) > max_gap_s + 1e-9:
            if cur:
                intervals.append(cur)
            cur = []
        cur.append((idx, tf, float(v)))
        last_t = tf
    if cur:
        intervals.append(cur)
    return intervals


def _has_elapsed_support(
    pts: Sequence[tuple[int, float, float]],
    start_k: int,
    persistence_s: float,
    max_gap_s: float,
    predicate: Any,
) -> tuple[bool, float, float]:
    """Require actual observed elapsed support across ``[tk, tk + persistence_s]``."""
    if start_k < 0 or start_k >= len(pts):
        return (False, 0.0, 0.0)
    _, t0, v0 = pts[start_k]
    if not predicate(v0):
        return (False, t0, 0.0)
    prev_t = t0
    for m in range(start_k + 1, len(pts)):
        _, tm, vm = pts[m]
        if (tm - prev_t) > max_gap_s + 1e-9:
            return (False, t0, 0.0)
        if not predicate(vm):
            return (False, t0, 0.0)
        prev_t = tm
        if (tm - t0) >= persistence_s - 1e-6:
            return (True, tm, tm - t0)
    return (False, t0, 0.0)


def _evaluate_transitions(
    timestamps: list[float],
    v_pair: list[float | None],
    pair_hip_eligible: list[bool],
    break_mask: list[bool],
    slow_events: list[dict[str, Any]],
    settle_events: list[dict[str, Any]],
    annotations: CoupleAnnotations,
    cfg: Config,
    *,
    max_gap_s: float | None = None,
    v_origins: Sequence[str | None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate secondary speed peaks and direction reversals inside transition windows."""
    da_cfg = cfg.dancer_analysis
    persistence_s = float(da_cfg.persistence_s)
    rise_thresh = float(da_cfg.secondary_speed_rise)
    stop_speed = float(da_cfg.stop_speed)
    min_cov = float(da_cfg.transition_min_coverage)
    eff_max_gap = float(max_gap_s if max_gap_s is not None else da_cfg.max_observation_gap_s)

    # Identify candidate transition brackets (t0, t1, source_slow_t)
    windows: list[tuple[float, float, float | None]] = []
    for ev in sorted(slow_events, key=lambda e: float(e["timestamp"])):
        t0 = float(ev["timestamp"])
        after = [
            float(s["timestamp"])
            for s in settle_events
            if float(s["timestamp"]) > t0 + 0.05
        ]
        t1 = min(after) if after else (min(timestamps[-1], t0 + 2.0) if timestamps else t0 + 2.0)
        if t1 > t0:
            windows.append((t0, t1, t0))

    if not windows and len(timestamps) >= 3:
        for mark in annotations.planned_transitions:
            m0 = float(mark.timestamp)
            m1 = float(mark.window_end if mark.window_end is not None else mark.timestamp)
            t0 = m0 - 0.5
            t1 = m1 + 0.5
            if t1 > t0:
                windows.append((t0, t1, m0))

    windows.sort(key=lambda item: (item[0], item[1]))
    intervals = _partition_contiguous_eligible_intervals(
        timestamps, v_pair, pair_hip_eligible, break_mask, eff_max_gap
    )

    cue_events: list[dict[str, Any]] = []
    planned_marks = list(annotations.planned_transitions)

    def _overlaps_planned(t_a: float, t_b: float, kind: str) -> bool:
        for m in planned_marks:
            if m.kind != kind:
                continue
            m0 = float(m.timestamp)
            m1 = float(m.window_end) if m.window_end is not None else m0
            if m0 - 0.20 <= t_b and m1 + 0.20 >= t_a:
                return True
        return False

    seen_episodes: dict[str, list[tuple[float, float, str]]] = {
        "speed_peak": [],
        "reversal": [],
    }

    def _episode_is_duplicate(cue_group: str, onset_t: float, support_end_t: float) -> bool:
        dedup_tol = max(persistence_s, 0.25)
        for prev_start, prev_end, _ in seen_episodes[cue_group]:
            if prev_start - persistence_s <= support_end_t and prev_end + persistence_s >= onset_t:
                return True
            if abs(prev_start - onset_t) <= dedup_tol:
                return True
        return False

    def _phase_reference_sign(
        interval: Sequence[tuple[int, float, float]],
        sub: Sequence[tuple[int, float, float]],
        anchor_t: float | None,
    ) -> float:
        ref_t = anchor_t if anchor_t is not None else sub[0][1]
        ref_vals = [
            v
            for _, t, v in interval
            if ref_t - 1.5 <= t <= ref_t + 1e-6 and abs(v) > stop_speed
        ]
        if not ref_vals:
            earlier = [v for _, t, v in interval if t <= ref_t + 1e-6 and abs(v) > stop_speed]
            ref_vals = earlier[-10:]
        if not ref_vals:
            # Anchor at very start of contiguous interval: use initial moving run inside sub before any sign flip
            first_sign: float | None = None
            for _, _, v in sub:
                if abs(v) <= stop_speed:
                    if ref_vals:
                        break
                    continue
                cur_s = 1.0 if v >= 0.0 else -1.0
                if first_sign is None:
                    first_sign = cur_s
                elif cur_s != first_sign:
                    break
                ref_vals.append(v)
        if not ref_vals:
            return 0.0
        med = float(np.median(ref_vals))
        return 1.0 if med >= 0.0 else -1.0

    def _sub_origin(sub: Sequence[tuple[int, float, float]]) -> PoseOrigin | None:
        if v_origins is None:
            return None
        orgs = [
            v_origins[idx]
            for idx, _, _ in sub
            if 0 <= idx < len(v_origins) and v_origins[idx]
        ]
        if "propagated" in orgs:
            return "propagated"
        if "crop_detected" in orgs:
            return "crop_detected"
        if "detected" in orgs:
            return "detected"
        return None

    total_win_covs: list[float] = []
    unintended_peaks = 0
    unintended_reversals = 0
    planned_pauses = sum(1 for m in planned_marks if m.kind == "planned_pause")
    planned_reversals = sum(1 for m in planned_marks if m.kind == "planned_reversal")
    settling_steps = sum(1 for m in planned_marks if m.kind == "settling_step")
    candidate_window_count = 0
    evaluated_window_count = 0
    unavailable_windows = 0
    evaluated_origins: list[str | None] = []

    for w_idx, (t0, t1, anchor_t) in enumerate(windows):
        in_idx = [i for i, t in enumerate(timestamps) if t0 - 1e-6 <= t <= t1 + 1e-6]
        if not in_idx:
            continue
        candidate_window_count += 1
        cov = sum(1 for i in in_idx if pair_hip_eligible[i]) / len(in_idx)
        total_win_covs.append(cov)
        if cov < min_cov:
            unavailable_windows += 1
            continue

        # Partition window into contiguous eligible sub-intervals with enough duration for elapsed support
        win_subs: list[tuple[list[tuple[int, float, float]], list[tuple[int, float, float]]]] = []
        for iv in intervals:
            sub = [pt for pt in iv if t0 - 1e-6 <= pt[1] <= t1 + 1e-6]
            if len(sub) >= 2 and (sub[-1][1] - sub[0][1]) >= persistence_s - 1e-6:
                win_subs.append((iv, sub))

        if not win_subs:
            unavailable_windows += 1
            continue

        evaluated_window_count += 1
        for iv, sub in win_subs:
            sub_org = _sub_origin(sub)
            evaluated_origins.append(sub_org)

            # 1) Secondary speed peak check inside contiguous eligible interval
            running_min = abs(sub[0][2])
            for k, (_, tk, vk) in enumerate(sub):
                sp = abs(vk)
                if sp < running_min:
                    running_min = sp
                elif (sp - running_min) >= rise_thresh:
                    thresh = rise_thresh * 0.6
                    ok, support_end_t, support_dur = _has_elapsed_support(
                        sub,
                        k,
                        persistence_s,
                        eff_max_gap,
                        lambda val, rmin=running_min, th=thresh: (abs(val) - rmin) >= th,
                    )
                    if ok:
                        if not _episode_is_duplicate("speed_peak", tk, support_end_t):
                            ep_id = f"ep_speed_peak_{round(tk * 1000)}ms"
                            seen_episodes["speed_peak"].append((tk, support_end_t, ep_id))
                            is_settling_step = _overlaps_planned(
                                tk, support_end_t, "settling_step"
                            ) or _overlaps_planned(tk, support_end_t, "planned_pause")
                            if not is_settling_step:
                                unintended_peaks += 1
                            base_status = (
                                "user_confirmed"
                                if is_settling_step
                                else ("estimated" if sub_org == "propagated" else "measured")
                            )
                            cue_events.append(
                                CoupleEvent(
                                    event_id=f"cand_speed_peak_{round(tk * 1000)}ms",
                                    event_type="speed_cue",
                                    subject="pair",
                                    timestamp=round(tk, 4),
                                    end_timestamp=round(min(t1, support_end_t), 4),
                                    value=round(sp - running_min, 4),
                                    unit="pair_torso_lengths/s",
                                    status=base_status,  # type: ignore[arg-type]
                                    reason=(
                                        "planned_settling_step"
                                        if is_settling_step
                                        else "secondary_speed_peak"
                                    ),
                                    metadata={
                                        "cue_kind": (
                                            "planned_settling_step"
                                            if is_settling_step
                                            else "secondary_speed_peak"
                                        ),
                                        "episode_id": ep_id,
                                        "support_duration_s": round(support_dur, 4),
                                        "window_index": w_idx,
                                        **({"pose_origin": sub_org} if sub_org else {}),
                                    },
                                ).to_dict()
                            )
                        break

            # 2) Direction reversal check against phase-specific reference direction
            phase_sign = _phase_reference_sign(iv, sub, anchor_t)
            if phase_sign != 0.0:
                for k, (_, tk, vk) in enumerate(sub):
                    is_opp = (vk * phase_sign) < -stop_speed
                    prev_opp = k > 0 and (sub[k - 1][2] * phase_sign) < -stop_speed
                    if is_opp and not prev_opp:
                        ok, support_end_t, support_dur = _has_elapsed_support(
                            sub,
                            k,
                            persistence_s,
                            eff_max_gap,
                            lambda val, ps=phase_sign: (val * ps) < -stop_speed,
                        )
                        if ok:
                            if not _episode_is_duplicate("reversal", tk, support_end_t):
                                ep_id = f"ep_reversal_{round(tk * 1000)}ms"
                                seen_episodes["reversal"].append((tk, support_end_t, ep_id))
                                is_planned = _overlaps_planned(
                                    tk, support_end_t, "planned_reversal"
                                )
                                if not is_planned:
                                    unintended_reversals += 1
                                base_status = (
                                    "user_confirmed"
                                    if is_planned
                                    else ("estimated" if sub_org == "propagated" else "measured")
                                )
                                cue_events.append(
                                    CoupleEvent(
                                        event_id=f"cand_reversal_{round(tk * 1000)}ms",
                                        event_type="speed_cue",
                                        subject="pair",
                                        timestamp=round(tk, 4),
                                        end_timestamp=round(min(t1, support_end_t), 4),
                                        value=round(abs(vk), 4),
                                        unit="pair_torso_lengths/s",
                                        status=base_status,  # type: ignore[arg-type]
                                        reason=(
                                            "planned_reversal"
                                            if is_planned
                                            else "unintended_reversal"
                                        ),
                                        metadata={
                                            "cue_kind": (
                                                "planned_reversal"
                                                if is_planned
                                                else "unintended_reversal"
                                            ),
                                            "episode_id": ep_id,
                                            "support_duration_s": round(support_dur, 4),
                                            "phase_reference_sign": phase_sign,
                                            "window_index": w_idx,
                                            **({"pose_origin": sub_org} if sub_org else {}),
                                        },
                                    ).to_dict()
                                )
                            break

    mean_cov = float(np.mean(total_win_covs)) if total_win_covs else 0.0
    if evaluated_window_count == 0:
        reason = (
            "no_evaluated_transition_windows"
            if candidate_window_count == 0
            else "insufficient_transition_window_coverage"
        )
        return cue_events, {
            "status": "unavailable",
            "coverage": round(mean_cov, 4),
            "reason": reason,
            "secondary_speed_peaks": None,
            "unintended_reversals": None,
            "planned_pauses": planned_pauses,
            "planned_reversals": planned_reversals,
            "settling_steps": settling_steps,
            "candidate_window_count": candidate_window_count,
            "evaluated_window_count": 0,
            "unavailable_window_count": unavailable_windows,
            "eligible_count": 0,
            "coverage_denominator": candidate_window_count,
            "status_counts": _count_statuses([]),
            "origin_counts": _count_origins([]),
        }

    cue_statuses = [str(ev.get("status", "measured")) for ev in cue_events]
    cue_origins = [ev.get("metadata", {}).get("pose_origin") for ev in cue_events]
    if cue_statuses:
        summary_status, status_counts = _summarize_evidence_status(
            cue_statuses, eligible_count=len(cue_statuses)
        )
        # Reclassifying an observed kinematic transition via planned marks combines measured + user marks.
        if summary_status == "user_confirmed":
            summary_status = "mixed"
    else:
        win_statuses = [
            "estimated" if org == "propagated" else "measured"
            for org in (evaluated_origins or ["detected"])
        ]
        summary_status, status_counts = _summarize_evidence_status(
            win_statuses, eligible_count=evaluated_window_count
        )
    origin_counts = _count_origins(cue_origins if cue_origins else evaluated_origins)
    trans_summary = {
        "status": summary_status,
        "coverage": round(mean_cov, 4),
        "reason": "",
        "secondary_speed_peaks": unintended_peaks,
        "unintended_reversals": unintended_reversals,
        "planned_pauses": planned_pauses,
        "planned_reversals": planned_reversals,
        "settling_steps": settling_steps,
        "candidate_window_count": candidate_window_count,
        "evaluated_window_count": evaluated_window_count,
        "unavailable_window_count": unavailable_windows,
        "eligible_count": evaluated_window_count,
        "coverage_denominator": candidate_window_count,
        "status_counts": status_counts,
        "origin_counts": origin_counts,
    }
    return cue_events, trans_summary


def _turn_events_from_annotations_and_candidates(
    annotations: CoupleAnnotations,
    auto_turn_suggestions: bool,
    pivot_candidates: list[dict[str, Any]],
    *,
    turn_exit_match_window_s: float = 3.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build paired turn duration, exit timing, pivot behavior, and unmatched turn markers safely."""
    items = [item for item in annotations.items if not item.deleted]
    turn_items = [i for i in items if i.event_type in ("turn_start", "turn_end", "exit")]

    turns: list[dict[str, Any]] = []
    unmatched_records: list[dict[str, Any]] = []
    warnings: list[str] = []

    def _add_unmatched(item: CoupleAnnotationItem, reason: str) -> None:
        rec = {
            "event_id": f"unmatched_{item.annotation_id}",
            "annotation_id": item.annotation_id,
            "turn_id": item.turn_id,
            "event_type": item.event_type,
            "subject": item.subject,
            "timestamp": float(item.timestamp),
            "end_timestamp": float(item.end_timestamp) if item.end_timestamp is not None else None,
            "bracket_start": None,
            "bracket_end": None,
            "value": None,
            "unit": "",
            "status": item.status,
            "paired_event_id": None,
            "lag_ms": None,
            "beat_offset_ms": None,
            "beat_fraction": None,
            "reason": reason,
            "metadata": {
                **dict(item.metadata),
                "unmatched": True,
                "marker_event_type": item.event_type,
                "pivot_behavior": item.pivot_behavior,
                "label": item.label,
            },
        }
        unmatched_records.append(rec)

    def _build_turn_event(
        turn_id: str,
        st: CoupleAnnotationItem,
        en: CoupleAnnotationItem,
        exit_item: CoupleAnnotationItem | None,
        *,
        association: str,
        ambiguous_legacy: bool = False,
    ) -> dict[str, Any]:
        duration_s = en.timestamp - st.timestamp
        exit_diff_ms: float | None = None
        overlap = False
        if exit_item is not None:
            exit_diff_ms = (exit_item.timestamp - en.timestamp) * 1000.0
            overlap = exit_item.timestamp < en.timestamp
        marker_statuses = [st.status, en.status] + ([exit_item.status] if exit_item else [])
        ev_status, _ = _summarize_evidence_status(marker_statuses, eligible_count=len(marker_statuses))
        shared_pair = st.subject == "pair"
        return {
            "event_id": turn_id,
            "annotation_id": st.annotation_id,
            "turn_id": turn_id,
            "event_type": "turn",
            "subject": st.subject,
            "timestamp": float(st.timestamp),
            "end_timestamp": float(en.timestamp),
            "bracket_start": float(st.timestamp),
            "bracket_end": float(en.timestamp),
            "value": round(duration_s, 4),
            "unit": "seconds",
            "status": ev_status,
            "paired_event_id": exit_item.annotation_id if exit_item else None,
            "lag_ms": round(exit_diff_ms, 2) if exit_diff_ms is not None else None,
            "beat_offset_ms": None,
            "beat_fraction": None,
            "reason": "exit_occurred_during_turn" if overlap else "",
            "metadata": {
                "turn_id": turn_id,
                "association": association,
                "shared_pair_turn": shared_pair,
                "ambiguous_legacy_match": ambiguous_legacy,
                "start_annotation_id": st.annotation_id,
                "end_annotation_id": en.annotation_id,
                "exit_annotation_id": exit_item.annotation_id if exit_item else None,
                "turn_duration_s": round(duration_s, 4),
                "exit_timestamp": float(exit_item.timestamp) if exit_item else None,
                "exit_minus_turn_end_ms": round(exit_diff_ms, 2) if exit_diff_ms is not None else None,
                "overlap": overlap,
                "pivot_behavior_a": st.metadata.get("pivot_behavior_a", st.pivot_behavior or "unknown"),
                "pivot_behavior_b": st.metadata.get("pivot_behavior_b", en.pivot_behavior or "unknown"),
                "embrace_opening": st.label or en.label or "",
            },
        }

    # 1) Explicit turn_id groups
    by_turn_id: dict[str, list[CoupleAnnotationItem]] = {}
    legacy_items: list[CoupleAnnotationItem] = []
    for item in turn_items:
        if item.turn_id:
            by_turn_id.setdefault(item.turn_id, []).append(item)
        else:
            legacy_items.append(item)

    legacy_exits_pending: dict[str, list[CoupleAnnotationItem]] = {}

    for tid, group in by_turn_id.items():
        subjects = {g.subject for g in group}
        if len(subjects) > 1:
            warnings.append(f"Turn ID '{tid}' has mismatched marker subjects ({sorted(subjects)}).")
            for g in group:
                _add_unmatched(g, "turn_id_subject_mismatch")
            continue

        starts = sorted([g for g in group if g.event_type == "turn_start"], key=lambda g: g.timestamp)
        ends = sorted([g for g in group if g.event_type == "turn_end"], key=lambda g: g.timestamp)
        ex_list = sorted([g for g in group if g.event_type == "exit"], key=lambda g: g.timestamp)

        if len(starts) > 1:
            warnings.append(f"Turn ID '{tid}' has duplicate turn_start markers.")
            for extra in starts[1:]:
                _add_unmatched(extra, "duplicate_turn_id_marker")
            starts = starts[:1]
        if len(ends) > 1:
            warnings.append(f"Turn ID '{tid}' has duplicate turn_end markers.")
            for extra in ends[1:]:
                _add_unmatched(extra, "duplicate_turn_id_marker")
            ends = ends[:1]
        if len(ex_list) > 1:
            warnings.append(f"Turn ID '{tid}' has duplicate exit markers.")
            for extra in ex_list[1:]:
                _add_unmatched(extra, "duplicate_turn_id_marker")
            ex_list = ex_list[:1]

        if not starts and not ends:
            for g in ex_list:
                _add_unmatched(g, "unmatched_exit")
            continue
        if starts and not ends:
            _add_unmatched(starts[0], "missing_turn_end")
            for g in ex_list:
                _add_unmatched(g, "missing_turn_end_for_exit")
            continue
        if ends and not starts:
            _add_unmatched(ends[0], "missing_turn_start")
            for g in ex_list:
                _add_unmatched(g, "missing_turn_start_for_exit")
            continue

        st, en = starts[0], ends[0]
        if en.timestamp < st.timestamp:
            warnings.append(f"Turn ID '{tid}' has turn_end before turn_start.")
            _add_unmatched(st, "invalid_turn_marker_order")
            _add_unmatched(en, "invalid_turn_marker_order")
            for g in ex_list:
                _add_unmatched(g, "invalid_turn_marker_order")
            continue

        exit_item: CoupleAnnotationItem | None = None
        if ex_list:
            cand_ex = ex_list[0]
            if cand_ex.timestamp < st.timestamp - 0.10:
                warnings.append(f"Turn ID '{tid}' has exit before turn_start.")
                _add_unmatched(cand_ex, "invalid_exit_order")
            else:
                exit_item = cand_ex

        turns.append(
            _build_turn_event(tid, st, en, exit_item, association="turn_id")
        )

    # 2) Legacy markers without turn_id: strictly partitioned per subject
    legacy_subjects = sorted({item.subject for item in legacy_items})
    for subj in legacy_subjects:
        subj_starts = sorted(
            [i for i in legacy_items if i.subject == subj and i.event_type == "turn_start"],
            key=lambda i: i.timestamp,
        )
        subj_ends = sorted(
            [i for i in legacy_items if i.subject == subj and i.event_type == "turn_end"],
            key=lambda i: i.timestamp,
        )
        subj_exits = sorted(
            [i for i in legacy_items if i.subject == subj and i.event_type == "exit"],
            key=lambda i: i.timestamp,
        )

        # Detect nested/overlapping same-subject legacy starts before a closing turn_end
        combined_seq = sorted(
            [(s.timestamp, 0, s) for s in subj_starts] + [(e.timestamp, 1, e) for e in subj_ends],
            key=lambda row: (row[0], row[1]),
        )
        open_depth = 0
        ambiguous_legacy = False
        for _, kind_ord, _ in combined_seq:
            if kind_ord == 0:
                open_depth += 1
                if open_depth > 1:
                    ambiguous_legacy = True
            else:
                open_depth = max(0, open_depth - 1)
        if ambiguous_legacy:
            warnings.append(
                f"Overlapping legacy turn_start markers for subject '{subj}' without explicit turn_id."
            )

        used_ends: set[int] = set()
        subj_turns: list[dict[str, Any]] = []
        for idx_s, st in enumerate(subj_starts):
            end_idx = next(
                (j for j, en in enumerate(subj_ends) if j not in used_ends and en.timestamp >= st.timestamp),
                None,
            )
            if end_idx is None:
                _add_unmatched(st, "missing_turn_end")
                continue
            used_ends.add(end_idx)
            en = subj_ends[end_idx]
            tid = f"turn_{subj}_{idx_s}"
            turn_ev = _build_turn_event(
                tid,
                st,
                en,
                None,
                association="legacy_subject_sequence",
                ambiguous_legacy=ambiguous_legacy,
            )
            subj_turns.append(turn_ev)

        for j, en in enumerate(subj_ends):
            if j not in used_ends:
                _add_unmatched(en, "missing_turn_start")

        # Attach legacy exits strictly within the same subject
        legacy_exits_pending[subj] = subj_exits
        used_exits: set[int] = set()
        for t_ev in subj_turns:
            st_t = float(t_ev["timestamp"])
            en_t = float(t_ev["end_timestamp"])
            best_idx: int | None = None
            best_dt = math.inf
            for je, ex in enumerate(subj_exits):
                if je in used_exits:
                    continue
                if ex.timestamp >= st_t - 0.10:
                    dt = abs(ex.timestamp - en_t)
                    if dt < best_dt and dt <= float(turn_exit_match_window_s):
                        best_dt = dt
                        best_idx = je
            if best_idx is not None:
                used_exits.add(best_idx)
                ex_item = subj_exits[best_idx]
                exit_diff_ms = (ex_item.timestamp - en_t) * 1000.0
                overlap = ex_item.timestamp < en_t
                t_ev["paired_event_id"] = ex_item.annotation_id
                t_ev["lag_ms"] = round(exit_diff_ms, 2)
                if overlap:
                    t_ev["reason"] = "exit_occurred_during_turn"
                t_ev["metadata"]["exit_annotation_id"] = ex_item.annotation_id
                t_ev["metadata"]["exit_timestamp"] = float(ex_item.timestamp)
                t_ev["metadata"]["exit_minus_turn_end_ms"] = round(exit_diff_ms, 2)
                t_ev["metadata"]["overlap"] = overlap
                marker_statuses = [t_ev["status"], ex_item.status]
                t_ev["status"], _ = _summarize_evidence_status(
                    marker_statuses, eligible_count=len(marker_statuses)
                )
        for je, ex in enumerate(subj_exits):
            if je not in used_exits:
                _add_unmatched(ex, "unmatched_exit_no_matching_turn")

        turns.extend(subj_turns)

    if auto_turn_suggestions:
        for p_idx, piv in enumerate(pivot_candidates):
            p_start = float(piv.get("start_time", 0.0))
            p_end = float(piv.get("end_time", p_start))
            p_subj = str(piv.get("subject", "pair"))
            overlaps_user = any(
                (t.get("subject") in (p_subj, "pair") or p_subj == "pair")
                and t["timestamp"] - 0.3 <= p_end
                and t["end_timestamp"] + 0.3 >= p_start
                for t in turns
            )
            if not overlaps_user and p_end > p_start:
                auto_tid = f"auto_turn_{p_subj}_{p_idx}"
                turns.append(
                    {
                        "event_id": auto_tid,
                        "turn_id": auto_tid,
                        "event_type": "turn",
                        "subject": p_subj,
                        "timestamp": round(p_start, 4),
                        "end_timestamp": round(p_end, 4),
                        "bracket_start": round(p_start, 4),
                        "bracket_end": round(p_end, 4),
                        "value": round(p_end - p_start, 4),
                        "unit": "seconds",
                        "status": "estimated",
                        "paired_event_id": None,
                        "lag_ms": None,
                        "beat_offset_ms": None,
                        "beat_fraction": None,
                        "reason": "auto_turn_suggestion",
                        "metadata": {
                            "turn_id": auto_tid,
                            "shared_pair_turn": p_subj == "pair",
                            "turn_duration_s": round(p_end - p_start, 4),
                            "overlap": False,
                            "pivot_behavior_a": "unknown",
                            "pivot_behavior_b": "unknown",
                        },
                    }
                )

    durations = [float(t["value"]) for t in turns if t["value"] is not None]
    exit_timings = [
        float(t["metadata"]["exit_minus_turn_end_ms"])
        for t in turns
        if t["metadata"].get("exit_minus_turn_end_ms") is not None
    ]
    dur_med, dur_iqr = _median_iqr(durations)
    exit_med, exit_iqr = _median_iqr(exit_timings)
    turn_statuses = [str(t.get("status", "estimated")) for t in turns]
    summary_status, status_counts = _summarize_evidence_status(
        turn_statuses, eligible_count=len(turns)
    )
    unmatched_summaries = [
        {
            "annotation_id": rec["annotation_id"],
            "turn_id": rec["turn_id"],
            "event_type": rec["event_type"],
            "subject": rec["subject"],
            "timestamp": rec["timestamp"],
            "reason": rec["reason"],
        }
        for rec in unmatched_records
    ]
    turn_reason = ""
    if not turns:
        turn_reason = "unmatched_turn_markers_only" if unmatched_records else "no_turns_recorded"

    turn_summary = {
        "turn_count": len(turns),
        "overlap_count": sum(1 for t in turns if t["metadata"].get("overlap")),
        "median_turn_duration_s": dur_med,
        "iqr_turn_duration_s": dur_iqr,
        "median_exit_minus_turn_end_ms": exit_med,
        "iqr_exit_minus_turn_end_ms": exit_iqr,
        "status": summary_status,
        "reason": turn_reason,
        "status_counts": status_counts,
        "eligible_count": len(turns),
        "coverage_denominator": len(turns) + len(unmatched_records),
        "unmatched_count": len(unmatched_records),
        "unmatched_markers": unmatched_summaries,
        "warnings": warnings,
    }
    return turns + unmatched_records, turn_summary


def _extract_finite_series_samples(
    entries: list[dict[str, Any]] | list[tuple[float, float | None]],
    *,
    allow_propagated_estimates: bool = False,
) -> list[tuple[float, float]]:
    allowed_statuses = {"measured", "user_confirmed", ""}
    if allow_propagated_estimates:
        allowed_statuses.add("estimated")
    samples: list[tuple[float, float]] = []
    for item in entries:
        if isinstance(item, dict):
            if item.get("pose_origin") == "display_interpolated":
                continue
            status = item.get("status")
            if status is not None and status not in allowed_statuses:
                continue
            if status == "estimated" and item.get("pose_origin") not in ("propagated", None):
                continue
            raw_t = item.get("timestamp")
            raw_v = item.get("value")
        elif isinstance(item, tuple) and len(item) == 2:
            raw_t, raw_v = item
        else:
            continue
        if not isinstance(raw_t, (int, float)) or not isinstance(raw_v, (int, float)):
            continue
        tf = float(raw_t)
        vf = float(raw_v)
        if math.isfinite(tf) and math.isfinite(vf):
            samples.append((tf, vf))
    samples.sort(key=lambda pair: pair[0])
    return samples


def compute_torso_tilt_pearson_correlation(
    series_a: list[dict[str, Any]] | list[tuple[float, float | None]],
    series_b: list[dict[str, Any]] | list[tuple[float, float | None]],
    *,
    min_samples: int = 3,
    allow_propagated_estimates: bool = False,
) -> tuple[float | None, int, str]:
    """Compute Pearson correlation ``r`` between Dancer A and B torso tilts on aligned samples."""
    pts_a = _extract_finite_series_samples(
        series_a, allow_propagated_estimates=allow_propagated_estimates
    )
    pts_b = _extract_finite_series_samples(
        series_b, allow_propagated_estimates=allow_propagated_estimates
    )
    pairs: list[tuple[float, float]] = []
    ia = 0
    ib = 0
    while ia < len(pts_a) and ib < len(pts_b):
        ta, va = pts_a[ia]
        tb, vb = pts_b[ib]
        if abs(ta - tb) <= 1e-4 + 1e-9:
            pairs.append((va, vb))
            ia += 1
            ib += 1
        elif ta < tb:
            ia += 1
        else:
            ib += 1

    if len(pairs) < max(2, int(min_samples)):
        return (None, len(pairs), "insufficient_paired_samples")

    arr_a = np.array([va for va, _ in pairs], dtype=float)
    arr_b = np.array([vb for _, vb in pairs], dtype=float)
    da = arr_a - float(np.mean(arr_a))
    db = arr_b - float(np.mean(arr_b))
    norm_a = float(np.linalg.norm(da))
    norm_b = float(np.linalg.norm(db))
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return (None, len(pairs), "zero_variance")
    r_val = float(np.dot(da, db) / (norm_a * norm_b))
    r_clamped = max(-1.0, min(1.0, r_val))
    return (round(r_clamped, 4), len(pairs), "")


def _compute_summaries_from_events_and_series(
    *,
    series: dict[str, list[dict[str, Any]]],
    events: list[dict[str, Any]],
    trans_summary: dict[str, Any],
    turn_summary: dict[str, Any],
    coverage: dict[str, Any],
    baseline_passages: list[tuple[float, float]],
    allow_propagated_estimates: bool = False,
) -> dict[str, Any]:
    """Build structured summary metrics across all 5 Couple domains."""
    # 1. Coordination (inspect both A and paired B event statuses)
    coord: dict[str, Any] = {}
    for etype in ("start", "slow", "settle"):
        ev_type_all = [e for e in events if e.get("event_type") == etype]
        ev_by_id = {str(e.get("event_id")): e for e in ev_type_all}
        ev_a = [e for e in ev_type_all if e.get("subject") == "A"]
        ev_b = [e for e in ev_type_all if e.get("subject") == "B"]
        paired_a = [e for e in ev_a if e.get("lag_ms") is not None]
        lags = [float(e["lag_ms"]) for e in paired_a]
        med, iqr = _median_iqr(lags)
        unpaired = sum(1 for e in ev_type_all if e.get("lag_ms") is None)
        contributing_statuses: list[str] = []
        contributing_origins: list[str | None] = []
        for ea in paired_a:
            contributing_statuses.append(str(ea.get("status") or "measured"))
            contributing_origins.append((ea.get("metadata") or {}).get("pose_origin"))
            eb = ev_by_id.get(str(ea.get("paired_event_id") or ""))
            if eb is not None:
                contributing_statuses.append(str(eb.get("status") or "measured"))
                contributing_origins.append((eb.get("metadata") or {}).get("pose_origin"))
        lag_status, lag_status_counts = _summarize_evidence_status(
            contributing_statuses, eligible_count=len(lags)
        )
        lag_denom = max(len(ev_a), len(ev_b), len(lags))
        coord[f"{etype}_lag_ms"] = CoupleSummaryValue(
            measure=f"{etype}_lag_ms",
            subject="B-A",
            value=round(med, 2) if med is not None else None,
            iqr=round(iqr, 2) if iqr is not None else None,
            count=len(lags),
            unit="ms",
            status=lag_status,
            coverage=float(coverage.get("pair_hip_coverage", 0.0)),
            reason="" if lags else "no_paired_events",
            timestamp_or_bracket="",
            status_counts=lag_status_counts,
            origin_counts=_count_origins(contributing_origins),
            eligible_count=len(lags),
            coverage_denominator=lag_denom,
        ).to_dict()
        coord[f"{etype}_paired_count"] = len(lags)
        coord[f"{etype}_unpaired_count"] = unpaired

    # 2. Step and beat timing
    step_timing: dict[str, Any] = {}
    for subj in ("A", "B"):
        contacts = [e for e in events if e.get("event_type") == "contact" and e.get("subject") == subj]
        matched_contacts = [e for e in contacts if e.get("beat_offset_ms") is not None]
        offsets = [float(e["beat_offset_ms"]) for e in matched_contacts]
        fracs = [
            float(e["beat_fraction"])
            for e in matched_contacts
            if e.get("beat_fraction") is not None
        ]
        contacts_total = len(contacts)
        med_ms, iqr_ms = _median_iqr(offsets)
        med_fr, _ = _median_iqr(fracs)
        contact_statuses = [str(e.get("status") or "estimated") for e in contacts]
        beat_statuses = [
            str((e.get("metadata") or {}).get("beat_status"))
            for e in matched_contacts
            if (e.get("metadata") or {}).get("beat_status")
        ]
        combined_statuses = contact_statuses + (beat_statuses if matched_contacts else [])
        step_status, _step_status_counts = _summarize_evidence_status(
            combined_statuses, eligible_count=contacts_total
        )
        beat_status_agg, _ = _summarize_evidence_status(
            beat_statuses, eligible_count=len(beat_statuses)
        )
        step_timing[subj] = {
            "contact_count": contacts_total,
            "matched_count": len(offsets),
            "median_beat_offset_ms": round(med_ms, 2) if med_ms is not None else None,
            "iqr_beat_offset_ms": round(iqr_ms, 2) if iqr_ms is not None else None,
            "median_beat_fraction": round(med_fr, 4) if med_fr is not None else None,
            "status": step_status,
            "beat_status": beat_status_agg,
            "status_counts": _count_statuses(contact_statuses),
            "eligible_count": len(offsets) if beat_statuses else contacts_total,
            "coverage_denominator": contacts_total,
        }

    # 3. Posture & axes
    total_frames = max(
        int(coverage.get("frame_count") or 0),
        len(series.get("torso_tilt_a") or []),
        len(series.get("torso_tilt_b") or []),
    )
    allowed_posture_statuses = (
        {"measured", "user_confirmed", "estimated"}
        if allow_propagated_estimates
        else {"measured", "user_confirmed"}
    )

    def _angle_summary(key: str, subject: str, cov_key: str) -> dict[str, Any]:
        pts = series.get(key) or []
        eligible_pts = [
            p
            for p in pts
            if p.get("value") is not None
            and p.get("status") in allowed_posture_statuses
            and p.get("pose_origin") != "display_interpolated"
        ]
        vals = [float(p["value"]) for p in eligible_pts]
        med, iqr = _median_iqr(vals)
        base_change: float | None = None
        if baseline_passages and eligible_pts:
            bvals = [
                float(p["value"])
                for p in eligible_pts
                if any(t0 <= float(p["timestamp"]) <= t1 for t0, t1 in baseline_passages)
            ]
            if len(bvals) >= 3 and med is not None:
                ref = float(np.median(bvals))
                base_change = round(wrapped_angle_diff(med, ref), 2)

        all_statuses = [str(p.get("status") or "unavailable") for p in pts]
        all_origins = [p.get("pose_origin") for p in pts if p.get("pose_origin") is not None]
        elig_statuses = [str(p.get("status") or "measured") for p in eligible_pts]
        summary_st, _ = _summarize_evidence_status(elig_statuses, eligible_count=len(vals))
        all_status_counts = _count_statuses(all_statuses)
        all_origin_counts = _count_origins(all_origins)
        denom = max(total_frames, len(pts), 1)
        reason = ""
        if not vals:
            if any(p.get("reason") == "propagated_pose_excluded" for p in pts):
                reason = "propagated_poses_excluded"
            elif pts:
                reason = "no_eligible_frames"
            else:
                reason = "no_frames"
        return {
            "measure": key,
            "subject": subject,
            "median_deg": round(med, 2) if med is not None else None,
            "iqr_deg": round(iqr, 2) if iqr is not None else None,
            "baseline_change_deg": base_change,
            "sample_count": len(vals),
            "eligible_count": len(vals),
            "total_frame_count": len(pts) or total_frames,
            "coverage_denominator": denom,
            "coverage_denominator_label": f"{len(vals)}/{denom} frames",
            "coverage": float(coverage.get(cov_key, 0.0)),
            "status": summary_st,
            "status_counts": all_status_counts,
            "origin_counts": all_origin_counts,
            "reason": reason,
        }

    tilt_a_pts = series.get("torso_tilt_a") or []
    tilt_b_pts = series.get("torso_tilt_b") or []
    r_val, r_cnt, r_reason = compute_torso_tilt_pearson_correlation(
        tilt_a_pts,
        tilt_b_pts,
        allow_propagated_estimates=allow_propagated_estimates,
    )
    # Determine paired sample statuses/origins for Pearson r
    aligned_statuses: list[str] = []
    aligned_origins: list[str | None] = []
    map_b_by_t = {
        round(float(p["timestamp"]), 4): p
        for p in tilt_b_pts
        if isinstance(p, dict)
        and p.get("value") is not None
        and p.get("status") in allowed_posture_statuses
        and p.get("pose_origin") != "display_interpolated"
    }
    for pa in tilt_a_pts:
        if not isinstance(pa, dict):
            continue
        if (
            pa.get("value") is None
            or pa.get("status") not in allowed_posture_statuses
            or pa.get("pose_origin") == "display_interpolated"
        ):
            continue
        pb = map_b_by_t.get(round(float(pa["timestamp"]), 4))
        if pb is not None:
            aligned_statuses.extend([str(pa.get("status")), str(pb.get("status"))])
            aligned_origins.extend([pa.get("pose_origin"), pb.get("pose_origin")])

    if r_val is not None:
        pearson_st, pearson_status_counts = _summarize_evidence_status(
            aligned_statuses, eligible_count=r_cnt
        )
    elif 0 < r_cnt < 3:
        pearson_st = "insufficient_evidence"
        pearson_status_counts = _count_statuses(aligned_statuses)
    else:
        pearson_st = "unavailable"
        pearson_status_counts = _count_statuses(aligned_statuses)
    pair_denom = max(total_frames, len(tilt_a_pts), len(tilt_b_pts), 1)
    posture = {
        "torso_tilt_a": _angle_summary("torso_tilt_a", "A", "a_torso_coverage"),
        "torso_tilt_b": _angle_summary("torso_tilt_b", "B", "b_torso_coverage"),
        "couple_midpoint_tilt": _angle_summary("couple_midpoint_tilt", "pair", "pair_torso_coverage"),
        "torso_separation": _angle_summary("torso_separation", "pair", "pair_torso_coverage"),
        "torso_tilt_pearson_r": {
            "measure": "torso_tilt_pearson_r",
            "subject": "pair",
            "value": r_val,
            "sample_count": r_cnt,
            "eligible_count": r_cnt,
            "coverage_denominator": pair_denom,
            "coverage_denominator_label": f"{r_cnt}/{pair_denom} frames",
            "coverage": float(coverage.get("pair_torso_coverage", 0.0)),
            "unit": "r",
            "status": pearson_st,
            "status_counts": pearson_status_counts,
            "origin_counts": _count_origins(aligned_origins),
            "reason": r_reason,
        },
        "head_pitch_a": _angle_summary("head_pitch_a", "A", "a_head_coverage"),
        "head_pitch_b": _angle_summary("head_pitch_b", "B", "b_head_coverage"),
    }

    return {
        "coordination": coord,
        "step_timing": step_timing,
        "transitions": trans_summary,
        "posture": posture,
        "turns": turn_summary,
    }


def _apply_annotations_to_candidates(
    automated_candidates: list[dict[str, Any]],
    annotations: CoupleAnnotations,
    pivot_candidates: list[dict[str, Any]],
    series: dict[str, list[dict[str, Any]]],
    coverage: dict[str, Any],
    cfg: Config,
    *,
    allow_propagated_estimates: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge automated candidates with user confirmations, edits, deletions, beats, transitions, and turns."""
    by_source = {
        item.source_candidate_id: item
        for item in annotations.items
        if item.source_candidate_id
    }
    merged: list[dict[str, Any]] = []

    for raw_ev in automated_candidates:
        cid = str(raw_ev.get("event_id", ""))
        override = by_source.get(cid)
        if override is not None:
            if override.deleted:
                continue
            ev = dict(raw_ev)
            ev["timestamp"] = float(override.timestamp)
            if override.end_timestamp is not None:
                ev["end_timestamp"] = float(override.end_timestamp)
            ev["status"] = "user_confirmed"
            if override.label:
                ev.setdefault("metadata", {})["label"] = override.label
            merged.append(ev)
        else:
            merged.append(dict(raw_ev))

    # Add purely manual markers (not turn markers, which are handled via _turn_events_from_annotations_and_candidates)
    for item in annotations.items:
        if item.deleted or item.source_candidate_id:
            continue
        if item.event_type in ("turn_start", "turn_end", "exit"):
            continue
        merged.append(
            CoupleEvent(
                event_id=item.annotation_id,
                event_type=item.event_type,
                subject=item.subject,
                timestamp=float(item.timestamp),
                end_timestamp=item.end_timestamp,
                value=None,
                unit="",
                status=item.status,
                reason=item.label,
                metadata={**item.metadata, "pivot_behavior": item.pivot_behavior},
            ).to_dict()
        )

    # Re-pair coordination events
    break_ts = [float(p["timestamp"]) for p in (series.get("pair_breaks") or [])]
    paired = _pair_coordination_events(
        merged,
        pairing_window_s=float(cfg.dancer_analysis.pairing_window_s),
        break_timestamps=break_ts,
    )

    # Beat targets & offsets
    auto_beats = [
        float(e["timestamp"])
        for e in paired
        if e.get("event_type") == "musical_target" and e.get("status") == "estimated"
    ]
    confirmed_item_beats = [
        float(e["timestamp"])
        for e in paired
        if e.get("event_type") == "musical_target" and e.get("status") == "user_confirmed"
    ]
    beat_targets = _expand_beat_targets(auto_beats, annotations, confirmed_item_beats)
    _attach_beat_offsets(paired, beat_targets)

    # Re-evaluate transitions respecting planned marks
    vel_pts = series.get("velocity_pair") or []
    timestamps = [float(p["timestamp"]) for p in vel_pts]
    v_pair = [p["value"] for p in vel_pts]
    hip_ok = [bool(p.get("pair_hip_eligible", True)) for p in vel_pts]
    breaks = [bool(p.get("break_flag", False)) for p in vel_pts]
    v_origins = [p.get("pose_origin") for p in vel_pts]
    slow_evs = [e for e in paired if e.get("event_type") == "slow"]
    settle_evs = [e for e in paired if e.get("event_type") == "settle"]
    non_cue = [e for e in paired if e.get("event_type") != "speed_cue"]
    trans_cues, trans_summary = _evaluate_transitions(
        timestamps,
        v_pair,
        hip_ok,
        breaks,
        slow_evs,
        settle_evs,
        annotations,
        cfg,
        max_gap_s=float(cfg.dancer_analysis.max_observation_gap_s),
        v_origins=v_origins,
    )
    paired = non_cue + trans_cues

    # Turns & exit
    auto_turns_flag = bool(cfg.dancer_analysis.auto_turn_suggestions)
    turns, turn_summary = _turn_events_from_annotations_and_candidates(
        annotations,
        auto_turns_flag,
        pivot_candidates,
        turn_exit_match_window_s=float(cfg.dancer_analysis.turn_exit_match_window_s),
    )
    paired = [
        e for e in paired if e.get("event_type") not in ("turn", "turn_start", "turn_end", "exit")
    ] + turns
    paired.sort(key=lambda e: float(e.get("timestamp", 0.0)))

    summaries = _compute_summaries_from_events_and_series(
        series=series,
        events=paired,
        trans_summary=trans_summary,
        turn_summary=turn_summary,
        coverage=coverage,
        baseline_passages=annotations.baseline_passages,
        allow_propagated_estimates=allow_propagated_estimates,
    )
    return paired, summaries


_AXIS_OVERLAY_LANDMARKS = frozenset(
    {
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
        "nose",
        "left_ear",
        "right_ear",
    }
)


def run_couple_analysis(
    observations: list[tuple[int, FrameObservation]],
    cfg: Config,
    *,
    clip_in: float,
    clip_out: float,
    video_sha256: str = "",
    segment_index: int = 0,
    request: CoupleAnalysisRequest | dict[str, Any] | None = None,
    auto_decision: AutoDecisionResult | None = None,
    soundtrack_beats: list[float] | None = None,
    run_id: str | None = None,
) -> CoupleAnalysisResult:
    """Execute full Couple tracking, display interpolation, and 5-domain measurements."""
    req = CoupleAnalysisRequest.from_dict(request)
    da_cfg = cfg.dancer_analysis
    origin_policy: PoseOriginPolicy = (
        req.pose_origin_policy
        if req.pose_origin_policy in ("measured_only", "allow_propagated_estimates")
        else da_cfg.pose_origin_policy
    )
    decision = auto_decision or classify_opening_window(
        observations,
        cfg,
        clip_in=clip_in,
        clip_out=clip_out,
        requested_mode=req.requested_mode,
        forced_override=req.forced_override or req.requested_mode == "couple",
        analysis_region=req.analysis_region,
        anchors=req.annotations.anchors,
    )

    tracking: CoupleTrackingOutput = track_couple_frames(
        observations,
        cfg,
        analysis_region=req.analysis_region,
        anchors=req.annotations.anchors,
        baseline_passages=req.annotations.baseline_passages,
        pose_origin_policy=origin_policy,
    )
    frames = tracking.frames
    if not frames:
        raise ValueError("No valid video frames available for Couple analysis")

    # Apply display-only per-track 2D linear interpolation without mutating raw measurements
    timestamps = [pf.timestamp for pf in frames]
    frame_indices = [pf.frame_index for pf in frames]
    breaks_flag = [pf.shot_cut or pf.identity_break for pf in frames]

    def _empty_sample(t: float, fidx: int) -> TrackedDancerSample:
        return TrackedDancerSample(
            timestamp=t,
            frame_index=fidx,
            pose_index=None,
            raw_pose=None,
            display_pose={},
            world_pose=None,
            hip_px=None,
            shoulder_px=None,
            torso_length_px=None,
            torso_eligible=False,
            hip_eligible=False,
            head_eligible=False,
            view_usable=False,
            interpolated_display=True,
            quality_status="unavailable",
            pose_origin="display_interpolated",
            landmark_origins={},
        )

    disp_a = interpolate_tracked_dancer_poses(
        [pf.dancer_a for pf in frames],
        max_gap_s=float(da_cfg.display_interpolation_max_gap_s),
        confidence=float(cfg.confidence),
        break_flags=breaks_flag,
        timestamps=timestamps,
        frame_indices=frame_indices,
        sample_factory=_empty_sample,
    )
    disp_b = interpolate_tracked_dancer_poses(
        [pf.dancer_b for pf in frames],
        max_gap_s=float(da_cfg.display_interpolation_max_gap_s),
        confidence=float(cfg.confidence),
        break_flags=breaks_flag,
        timestamps=timestamps,
        frame_indices=frame_indices,
        sample_factory=_empty_sample,
    )
    frames = [
        replace(pf, dancer_a=sa, dancer_b=sb)
        for pf, sa, sb in zip(frames, disp_a, disp_b)
    ]

    # Travel axis and normalized displacement coordinates (RAW eligible hips only!)
    u_travel, travel_source = _infer_travel_axis(frames, req.annotations)
    scale_a = tracking.fixed_scale_a_px or tracking.fixed_pair_scale_px or 100.0
    scale_b = tracking.fixed_scale_b_px or tracking.fixed_pair_scale_px or 100.0
    scale_pair = tracking.fixed_pair_scale_px or 0.5 * (scale_a + scale_b)

    origin_a = next(
        (pf.dancer_a.hip_px for pf in frames if pf.dancer_a and pf.dancer_a.hip_eligible and pf.dancer_a.hip_px),
        None,
    )
    origin_b = next(
        (pf.dancer_b.hip_px for pf in frames if pf.dancer_b and pf.dancer_b.hip_eligible and pf.dancer_b.hip_px),
        None,
    )
    origin_pair = (
        (0.5 * (origin_a[0] + origin_b[0]), 0.5 * (origin_a[1] + origin_b[1]))
        if origin_a and origin_b
        else (origin_a or origin_b)
    )

    s_a: list[float | None] = []
    elig_a: list[bool] = []
    s_b: list[float | None] = []
    elig_b: list[bool] = []
    s_pair: list[float | None] = []
    elig_pair: list[bool] = []

    for pf in frames:
        if pf.dancer_a and pf.dancer_a.hip_eligible and pf.dancer_a.hip_px and origin_a:
            dx, dy = pf.dancer_a.hip_px[0] - origin_a[0], pf.dancer_a.hip_px[1] - origin_a[1]
            s_a.append((dx * u_travel[0] + dy * u_travel[1]) / scale_a)
            elig_a.append(True)
        else:
            s_a.append(None)
            elig_a.append(False)

        if pf.dancer_b and pf.dancer_b.hip_eligible and pf.dancer_b.hip_px and origin_b:
            dx, dy = pf.dancer_b.hip_px[0] - origin_b[0], pf.dancer_b.hip_px[1] - origin_b[1]
            s_b.append((dx * u_travel[0] + dy * u_travel[1]) / scale_b)
            elig_b.append(True)
        else:
            s_b.append(None)
            elig_b.append(False)

        if (
            pf.pair_hip_eligible
            and pf.dancer_a
            and pf.dancer_b
            and pf.dancer_a.hip_px
            and pf.dancer_b.hip_px
            and origin_pair
        ):
            hp = (
                0.5 * (pf.dancer_a.hip_px[0] + pf.dancer_b.hip_px[0]),
                0.5 * (pf.dancer_a.hip_px[1] + pf.dancer_b.hip_px[1]),
            )
            dx, dy = hp[0] - origin_pair[0], hp[1] - origin_pair[1]
            s_pair.append((dx * u_travel[0] + dy * u_travel[1]) / scale_pair)
            elig_pair.append(True)
        else:
            s_pair.append(None)
            elig_pair.append(False)

    vel_a = compute_centred_regression_velocity(
        timestamps,
        s_a,
        elig_a,
        breaks_flag,
        half_window_s=float(da_cfg.velocity_filter_window_s),
        max_gap_s=float(da_cfg.max_observation_gap_s),
    )
    vel_b = compute_centred_regression_velocity(
        timestamps,
        s_b,
        elig_b,
        breaks_flag,
        half_window_s=float(da_cfg.velocity_filter_window_s),
        max_gap_s=float(da_cfg.max_observation_gap_s),
    )
    vel_pair = compute_centred_regression_velocity(
        timestamps,
        s_pair,
        elig_pair,
        breaks_flag,
        half_window_s=float(da_cfg.velocity_filter_window_s),
        max_gap_s=float(da_cfg.max_observation_gap_s),
    )

    # §4.4 Torso tilts, couple midpoint axis, separation, head pitch & overlay frame contracts
    tilt_a_series: list[dict[str, Any]] = []
    tilt_b_series: list[dict[str, Any]] = []
    midpoint_series: list[dict[str, Any]] = []
    separation_series: list[dict[str, Any]] = []
    head_a_series: list[dict[str, Any]] = []
    head_b_series: list[dict[str, Any]] = []
    frame_overlays: list[dict[str, Any]] = []
    couple_axis_active: bool | None = None

    def _norm_from_canonical(pt: tuple[float, float], w: int, h: int) -> list[float]:
        return [round(pt[0] / max(1, w - 1), 6), round(-pt[1] / max(1, h - 1), 6)]

    def _pair_pose_origin(
        da: TrackedDancerSample | None,
        db: TrackedDancerSample | None,
    ) -> PoseOrigin | None:
        if da is None or db is None:
            return None
        orgs = (da.pose_origin, db.pose_origin)
        for cand in ("display_interpolated", "propagated", "crop_detected", "detected"):
            if cand in orgs:
                return cand  # type: ignore[return-value]
        return None

    def _sample_unavail_reason(sample: TrackedDancerSample | None, default_reason: str) -> str:
        if sample is None:
            return default_reason
        if sample.pose_origin == "propagated":
            return "propagated_pose_excluded"
        if sample.pose_origin == "display_interpolated":
            return "display_interpolated_only"
        return default_reason

    for pf in frames:
        t = float(pf.timestamp)
        ta: float | None = None
        tb: float | None = None
        org_a = pf.dancer_a.pose_origin if pf.dancer_a else None
        org_b = pf.dancer_b.pose_origin if pf.dancer_b else None
        org_pair = _pair_pose_origin(pf.dancer_a, pf.dancer_b)

        if pf.dancer_a and pf.dancer_a.torso_eligible and pf.dancer_a.hip_px and pf.dancer_a.shoulder_px:
            dx = pf.dancer_a.shoulder_px[0] - pf.dancer_a.hip_px[0]
            dy = pf.dancer_a.shoulder_px[1] - pf.dancer_a.hip_px[1]
            ta = math.degrees(math.atan2(dx, dy))
            st_a = "estimated" if org_a == "propagated" else "measured"
            tilt_a_series.append(
                {
                    "timestamp": t,
                    "value": round(ta, 3),
                    "status": st_a,
                    "reason": "propagated_estimate" if st_a == "estimated" else "",
                    "pose_origin": org_a,
                }
            )
        else:
            tilt_a_series.append(
                {
                    "timestamp": t,
                    "value": None,
                    "status": "unavailable",
                    "reason": _sample_unavail_reason(pf.dancer_a, "torso_occluded"),
                    "pose_origin": org_a,
                }
            )

        if pf.dancer_b and pf.dancer_b.torso_eligible and pf.dancer_b.hip_px and pf.dancer_b.shoulder_px:
            dx = pf.dancer_b.shoulder_px[0] - pf.dancer_b.hip_px[0]
            dy = pf.dancer_b.shoulder_px[1] - pf.dancer_b.hip_px[1]
            tb = math.degrees(math.atan2(dx, dy))
            st_b = "estimated" if org_b == "propagated" else "measured"
            tilt_b_series.append(
                {
                    "timestamp": t,
                    "value": round(tb, 3),
                    "status": st_b,
                    "reason": "propagated_estimate" if st_b == "estimated" else "",
                    "pose_origin": org_b,
                }
            )
        else:
            tilt_b_series.append(
                {
                    "timestamp": t,
                    "value": None,
                    "status": "unavailable",
                    "reason": _sample_unavail_reason(pf.dancer_b, "torso_occluded"),
                    "pose_origin": org_b,
                }
            )

        hip_only_norm: list[float] | None = None
        pair_hip_norm: list[float] | None = None
        pair_sh_norm: list[float] | None = None
        mid_tilt: float | None = None
        sep: float | None = None

        # Hysteresis check on estimated separation (metres; 1 torso length ≈ 0.50 m)
        hip_sep_m: float | None = None
        if pf.dancer_a and pf.dancer_b and pf.dancer_a.hip_px and pf.dancer_b.hip_px:
            torso_ref = max(float(scale_pair or 0.0), 1.0)
            hip_dist_px = math.hypot(
                pf.dancer_a.hip_px[0] - pf.dancer_b.hip_px[0],
                pf.dancer_a.hip_px[1] - pf.dancer_b.hip_px[1],
            )
            hip_sep_m = (hip_dist_px / torso_ref) * 0.50
            if couple_axis_active is None:
                couple_axis_active = hip_sep_m <= float(da_cfg.couple_axis_max_separation_m)
            elif couple_axis_active:
                if hip_sep_m > float(da_cfg.couple_axis_max_separation_m):
                    couple_axis_active = False
            elif hip_sep_m <= float(da_cfg.couple_axis_rejoin_separation_m):
                couple_axis_active = True

        draw_pair_axis = couple_axis_active is not False
        if pf.pair_torso_eligible and pf.dancer_a and pf.dancer_b:
            assert pf.dancer_a.hip_px and pf.dancer_b.hip_px and pf.dancer_a.shoulder_px and pf.dancer_b.shoulder_px
            hp = (
                0.5 * (pf.dancer_a.hip_px[0] + pf.dancer_b.hip_px[0]),
                0.5 * (pf.dancer_a.hip_px[1] + pf.dancer_b.hip_px[1]),
            )
            sp = (
                0.5 * (pf.dancer_a.shoulder_px[0] + pf.dancer_b.shoulder_px[0]),
                0.5 * (pf.dancer_a.shoulder_px[1] + pf.dancer_b.shoulder_px[1]),
            )
            mid_tilt = math.degrees(math.atan2(sp[0] - hp[0], sp[1] - hp[1]))
            assert ta is not None and tb is not None
            sep = abs(wrapped_angle_diff(ta, tb))
            st_pair = "estimated" if org_pair == "propagated" else "measured"
            reason_pair = "propagated_estimate" if st_pair == "estimated" else ""
            midpoint_series.append(
                {
                    "timestamp": t,
                    "value": round(mid_tilt, 3),
                    "status": st_pair,
                    "reason": reason_pair,
                    "pose_origin": org_pair,
                }
            )
            separation_series.append(
                {
                    "timestamp": t,
                    "value": round(sep, 3),
                    "status": st_pair,
                    "reason": reason_pair,
                    "pose_origin": org_pair,
                }
            )
            if draw_pair_axis:
                pair_hip_norm = _norm_from_canonical(hp, pf.width, pf.height)
                pair_sh_norm = _norm_from_canonical(sp, pf.width, pf.height)
        elif pf.pair_hip_eligible and pf.dancer_a and pf.dancer_b:
            assert pf.dancer_a.hip_px and pf.dancer_b.hip_px
            hp = (
                0.5 * (pf.dancer_a.hip_px[0] + pf.dancer_b.hip_px[0]),
                0.5 * (pf.dancer_a.hip_px[1] + pf.dancer_b.hip_px[1]),
            )
            if draw_pair_axis:
                hip_only_norm = _norm_from_canonical(hp, pf.width, pf.height)
            midpoint_series.append(
                {
                    "timestamp": t,
                    "value": None,
                    "status": "unavailable",
                    "reason": "shoulder_occluded_hip_only_travel_point",
                    "pose_origin": org_pair,
                }
            )
            separation_series.append(
                {
                    "timestamp": t,
                    "value": None,
                    "status": "unavailable",
                    "reason": "shoulder_occluded",
                    "pose_origin": org_pair,
                }
            )
        else:
            unavail_pair_reason = (
                "propagated_pose_excluded"
                if org_pair == "propagated"
                else (
                    "display_interpolated_only"
                    if "display_interpolated" in (org_a, org_b)
                    else "pair_unavailable"
                )
            )
            midpoint_series.append(
                {
                    "timestamp": t,
                    "value": None,
                    "status": "unavailable",
                    "reason": unavail_pair_reason,
                    "pose_origin": org_pair,
                }
            )
            separation_series.append(
                {
                    "timestamp": t,
                    "value": None,
                    "status": "unavailable",
                    "reason": unavail_pair_reason,
                    "pose_origin": org_pair,
                }
            )

        # Optional per-dancer head pitch from raw pose
        for d_sample, dest in ((pf.dancer_a, head_a_series), (pf.dancer_b, head_b_series)):
            d_org = d_sample.pose_origin if d_sample else None
            if d_sample and d_sample.raw_pose and d_sample.head_eligible:
                hp_res = head_horizontal_alignment(
                    d_sample.raw_pose, pf.width, pf.height, float(cfg.head_neutral_offset)
                )
                if hp_res is not None:
                    h_st = "estimated" if d_org == "propagated" else "measured"
                    dest.append(
                        {
                            "timestamp": t,
                            "value": round(float(hp_res[0]), 3),
                            "status": h_st,
                            "reason": "propagated_estimate" if h_st == "estimated" else "",
                            "pose_origin": d_org,
                        }
                    )
                    continue
            dest.append(
                {
                    "timestamp": t,
                    "value": None,
                    "status": "unavailable",
                    "reason": _sample_unavail_reason(d_sample, "head_landmarks_unavailable"),
                    "pose_origin": d_org,
                }
            )

        def _serialize_display_pose(d_sample: TrackedDancerSample | None) -> dict[str, Any] | None:
            if d_sample is None or not d_sample.display_pose:
                return None
            return {
                "landmarks": {
                    k: [round(lm.x, 5), round(lm.y, 5), round(lm.visibility or 0.0, 3)]
                    for k, lm in d_sample.display_pose.items()
                    if k in _AXIS_OVERLAY_LANDMARKS
                },
                "interpolated_display": bool(d_sample.interpolated_display),
                "quality_status": d_sample.quality_status,
                "pose_origin": d_sample.pose_origin,
                "landmark_origins": {
                    k: normalize_pose_origin(
                        d_sample.landmark_origins.get(k, lm.origin), d_sample.pose_origin
                    )
                    for k, lm in d_sample.display_pose.items()
                    if k in _AXIS_OVERLAY_LANDMARKS
                },
            }

        frame_overlays.append(
            {
                "frame_index": pf.frame_index,
                "timestamp": t,
                "dancer_a": _serialize_display_pose(pf.dancer_a),
                "dancer_b": _serialize_display_pose(pf.dancer_b),
                "pair_hip_norm": pair_hip_norm,
                "pair_shoulder_norm": pair_sh_norm,
                "hip_only_travel_point": hip_only_norm,
                "tilt_a_deg": round(ta, 2) if ta is not None and math.isfinite(ta) else None,
                "tilt_b_deg": round(tb, 2) if tb is not None and math.isfinite(tb) else None,
                "couple_midpoint_tilt_deg": (
                    round(mid_tilt, 2) if mid_tilt is not None and math.isfinite(mid_tilt) else None
                ),
                "torso_separation_deg": (
                    round(sep, 2)
                    if pf.pair_torso_eligible and sep is not None and math.isfinite(sep)
                    else None
                ),
                "travel_axis_unit": [round(u_travel[0], 5), round(u_travel[1], 5)],
                "min_landmark_visibility": float(da_cfg.overlay_min_landmark_visibility),
            }
        )

    overlay_tau_s = (
        float(cfg.overlay_smoothing)
        if abs(float(cfg.overlay_smoothing) - 0.20) > 1e-9
        and abs(float(da_cfg.overlay_smoothing_s) - 0.20) <= 1e-9
        else float(da_cfg.overlay_smoothing_s)
    )
    frame_overlays = smooth_couple_frame_overlays(
        frame_overlays,
        tau_s=overlay_tau_s,
        max_gap_s=float(da_cfg.axis_bridge_max_gap_s),
    )

    # §4.1 Motion event candidates
    a_origins = [pf.dancer_a.pose_origin if pf.dancer_a else None for pf in frames]
    b_origins = [pf.dancer_b.pose_origin if pf.dancer_b else None for pf in frames]
    pair_origins = [_pair_pose_origin(pf.dancer_a, pf.dancer_b) for pf in frames]
    motion_candidates = (
        _detect_motion_events_for_track(
            "A",
            timestamps,
            vel_a,
            breaks_flag,
            start_speed=float(da_cfg.start_speed),
            stop_speed=float(da_cfg.stop_speed),
            slowing_fraction=float(da_cfg.slowing_fraction),
            persistence_s=float(da_cfg.persistence_s),
            max_gap_s=float(da_cfg.max_observation_gap_s),
            sample_origins=a_origins,
        )
        + _detect_motion_events_for_track(
            "B",
            timestamps,
            vel_b,
            breaks_flag,
            start_speed=float(da_cfg.start_speed),
            stop_speed=float(da_cfg.stop_speed),
            slowing_fraction=float(da_cfg.slowing_fraction),
            persistence_s=float(da_cfg.persistence_s),
            max_gap_s=float(da_cfg.max_observation_gap_s),
            sample_origins=b_origins,
        )
    )

    # §4.2 Per-dancer apparent step contacts and pivot candidates from eligible poses via Movement engine
    w0, h0 = frames[0].width, frames[0].height
    span_t = max(0.05, timestamps[-1] - timestamps[0])
    est_fps = len(frames) / span_t if len(frames) > 1 else 30.0
    max_fidx = max(pf.frame_index for pf in frames) + 1
    step_candidates: list[CoupleEvent] = []
    pivot_candidates: list[dict[str, Any]] = []

    for subj, getter in (("A", lambda p: p.dancer_a), ("B", lambda p: p.dancer_b)):
        obs_list = [
            (
                pf.frame_index,
                FrameObservation(
                    pf.timestamp,
                    pf.width,
                    pf.height,
                    [d.raw_pose] if (d is not None and d.raw_pose is not None and d.hip_eligible) else [],
                    [d.world_pose] if (d is not None and d.world_pose is not None and d.hip_eligible and d.world_pose) else [],
                    [d.pose_origin] if (d is not None and d.raw_pose is not None and d.hip_eligible) else [],
                ),
            )
            for pf in frames
            for d in (getter(pf),)
        ]
        if any(o.poses for _, o in obs_list):
            mv = analyse_movement_observations(
                obs_list,
                width=w0,
                height=h0,
                fps=max(1.0, est_fps),
                frame_count=max_fidx,
                use_observation_timestamps=True,
            )
            for s_idx, step in enumerate(mv.get("steps") or []):
                step_candidates.append(
                    CoupleEvent(
                        event_id=f"cand_{subj}_contact_{s_idx}",
                        event_type="contact",
                        subject=subj,
                        timestamp=float(step["time"]),
                        value=float(step.get("confidence", 0.0)),
                        unit="confidence",
                        status="estimated",
                        metadata={
                            "side": step.get("side", "unknown"),
                            "x": step.get("x"),
                            "y": step.get("y"),
                            "frame": step.get("frame"),
                        },
                    )
                )
            for piv in mv.get("pivots") or []:
                pivot_candidates.append({**piv, "subject": subj})

    # Optional soundtrack beat candidates
    use_soundtrack = (
        bool(req.analyze_soundtrack)
        if req.analyze_soundtrack is not None
        else bool(da_cfg.analyze_soundtrack)
    )
    beat_candidates: list[CoupleEvent] = []
    if use_soundtrack and soundtrack_beats:
        for b_idx, bt in enumerate(sorted(soundtrack_beats)):
            beat_candidates.append(
                CoupleEvent(
                    event_id=f"cand_beat_{b_idx}",
                    event_type="musical_target",
                    subject="beat",
                    timestamp=float(bt),
                    status="estimated",
                    metadata={"source": "soundtrack"},
                )
            )

    automated_candidates = [e.to_dict() for e in (motion_candidates + step_candidates + beat_candidates)]

    series: dict[str, list[dict[str, Any]]] = {
        "velocity_a": [
            {
                "timestamp": float(pf.timestamp),
                "value": round(v, 4) if v is not None else None,
                "status": (
                    "unavailable"
                    if v is None
                    else ("estimated" if pf.dancer_a and pf.dancer_a.pose_origin == "propagated" else "measured")
                ),
                "pose_origin": pf.dancer_a.pose_origin if pf.dancer_a else None,
            }
            for pf, v in zip(frames, vel_a)
        ],
        "velocity_b": [
            {
                "timestamp": float(pf.timestamp),
                "value": round(v, 4) if v is not None else None,
                "status": (
                    "unavailable"
                    if v is None
                    else ("estimated" if pf.dancer_b and pf.dancer_b.pose_origin == "propagated" else "measured")
                ),
                "pose_origin": pf.dancer_b.pose_origin if pf.dancer_b else None,
            }
            for pf, v in zip(frames, vel_b)
        ],
        "velocity_pair": [
            {
                "timestamp": float(pf.timestamp),
                "value": round(v, 4) if v is not None else None,
                "pair_hip_eligible": bool(pf.pair_hip_eligible),
                "break_flag": bool(pf.shot_cut or pf.identity_break),
                "pose_origin": org_p,
                "pose_origin_a": pf.dancer_a.pose_origin if pf.dancer_a else None,
                "pose_origin_b": pf.dancer_b.pose_origin if pf.dancer_b else None,
            }
            for pf, v, org_p in zip(frames, vel_pair, pair_origins)
        ],
        "torso_tilt_a": tilt_a_series,
        "torso_tilt_b": tilt_b_series,
        "couple_midpoint_tilt": midpoint_series,
        "torso_separation": separation_series,
        "head_pitch_a": head_a_series,
        "head_pitch_b": head_b_series,
        "frame_pose_origins": [
            {
                "frame_index": pf.frame_index,
                "timestamp": float(pf.timestamp),
                "dancer_a": pf.dancer_a.pose_origin if pf.dancer_a else None,
                "dancer_b": pf.dancer_b.pose_origin if pf.dancer_b else None,
                "pair": org_p,
                "a_hip_eligible": bool(pf.dancer_a and pf.dancer_a.hip_eligible),
                "b_hip_eligible": bool(pf.dancer_b and pf.dancer_b.hip_eligible),
                "pair_hip_eligible": bool(pf.pair_hip_eligible),
                "a_torso_eligible": bool(pf.dancer_a and pf.dancer_a.torso_eligible),
                "b_torso_eligible": bool(pf.dancer_b and pf.dancer_b.torso_eligible),
                "pair_torso_eligible": bool(pf.pair_torso_eligible),
            }
            for pf, org_p in zip(frames, pair_origins)
        ],
        "pair_breaks": [{"timestamp": float(pf.timestamp)} for pf in frames if pf.shot_cut or pf.identity_break],
        "pivot_candidates": pivot_candidates,
    }

    total_n = max(1, len(frames))
    eligible_counts = {
        "a_hip": sum(1 for pf in frames if pf.dancer_a and pf.dancer_a.hip_eligible),
        "b_hip": sum(1 for pf in frames if pf.dancer_b and pf.dancer_b.hip_eligible),
        "pair_hip": sum(1 for pf in frames if pf.pair_hip_eligible),
        "a_torso": sum(1 for pf in frames if pf.dancer_a and pf.dancer_a.torso_eligible),
        "b_torso": sum(1 for pf in frames if pf.dancer_b and pf.dancer_b.torso_eligible),
        "pair_torso": sum(1 for pf in frames if pf.pair_torso_eligible),
        "a_head": sum(1 for pf in frames if pf.dancer_a and pf.dancer_a.head_eligible),
        "b_head": sum(1 for pf in frames if pf.dancer_b and pf.dancer_b.head_eligible),
    }
    strict_origins = {"detected", "crop_detected"}
    measured_only_coverage = {
        "a_hip_coverage": round(
            sum(1 for pf in frames if pf.dancer_a and pf.dancer_a.hip_eligible and pf.dancer_a.pose_origin in strict_origins) / total_n,
            4,
        ),
        "b_hip_coverage": round(
            sum(1 for pf in frames if pf.dancer_b and pf.dancer_b.hip_eligible and pf.dancer_b.pose_origin in strict_origins) / total_n,
            4,
        ),
        "pair_hip_coverage": round(
            sum(1 for pf, op in zip(frames, pair_origins) if pf.pair_hip_eligible and op in strict_origins) / total_n,
            4,
        ),
        "a_torso_coverage": round(
            sum(1 for pf in frames if pf.dancer_a and pf.dancer_a.torso_eligible and pf.dancer_a.pose_origin in strict_origins) / total_n,
            4,
        ),
        "b_torso_coverage": round(
            sum(1 for pf in frames if pf.dancer_b and pf.dancer_b.torso_eligible and pf.dancer_b.pose_origin in strict_origins) / total_n,
            4,
        ),
        "pair_torso_coverage": round(
            sum(1 for pf, op in zip(frames, pair_origins) if pf.pair_torso_eligible and op in strict_origins) / total_n,
            4,
        ),
    }
    coverage = {
        "frame_count": len(frames),
        "a_hip_coverage": round(eligible_counts["a_hip"] / total_n, 4),
        "b_hip_coverage": round(eligible_counts["b_hip"] / total_n, 4),
        "pair_hip_coverage": round(eligible_counts["pair_hip"] / total_n, 4),
        "a_torso_coverage": round(eligible_counts["a_torso"] / total_n, 4),
        "b_torso_coverage": round(eligible_counts["b_torso"] / total_n, 4),
        "pair_torso_coverage": round(eligible_counts["pair_torso"] / total_n, 4),
        "a_head_coverage": round(eligible_counts["a_head"] / total_n, 4),
        "b_head_coverage": round(eligible_counts["b_head"] / total_n, 4),
        "measurement_origin_policy": origin_policy,
        "denominator_explanation": f"eligible frames / {len(frames)} total frames in clip",
        "eligible_counts": eligible_counts,
        "measured_only_coverage": measured_only_coverage,
        "origin_counts": {
            "a": _count_origins(a_origins),
            "b": _count_origins(b_origins),
            "pair": _count_origins(pair_origins),
        },
    }

    auto_turns = (
        bool(req.auto_turn_suggestions)
        if req.auto_turn_suggestions is not None
        else bool(da_cfg.auto_turn_suggestions)
    )
    cfg_effective = cfg.model_copy(
        update={
            "dancer_analysis": cfg.dancer_analysis.model_copy(
                update={"auto_turn_suggestions": auto_turns}
            )
        }
    )

    events, summaries = _apply_annotations_to_candidates(
        automated_candidates,
        req.annotations,
        pivot_candidates,
        series,
        coverage,
        cfg_effective,
        allow_propagated_estimates=(origin_policy == "allow_propagated_estimates"),
    )

    post_diags = detect_post_window_scene_changes(frames, decision, cfg)
    diagnostics = list(decision.diagnostics) + list(tracking.diagnostics) + post_diags

    return CoupleAnalysisResult(
        run_id=run_id or str(uuid4()),
        revision_id=1,
        created_at=datetime.now(UTC).isoformat(),
        video_sha256=video_sha256,
        segment_index=segment_index,
        clip_in=float(clip_in),
        clip_out=float(clip_out),
        model_name=cfg.analysis_model,
        config_hash="",
        requested_mode=req.requested_mode,
        resolved_mode=decision.resolved_mode,
        auto_decision=decision,
        request=req,
        fixed_scale_a_px=tracking.fixed_scale_a_px,
        fixed_scale_b_px=tracking.fixed_scale_b_px,
        fixed_pair_scale_px=tracking.fixed_pair_scale_px,
        travel_axis_unit=u_travel,
        travel_axis_source=travel_source,  # type: ignore[arg-type]
        coverage=coverage,
        series=series,
        automated_candidates=automated_candidates,
        annotations=req.annotations.to_dict(),
        events=events,
        summaries=summaries,
        diagnostics=diagnostics,
        frame_overlays=frame_overlays,
    )


def rebuild_couple_result_with_annotations(
    couple_result: dict[str, Any],
    annotations_data: dict[str, Any] | CoupleAnnotations,
    cfg: Config,
) -> dict[str, Any]:
    """Recompute derived events and summaries from user edits without overwriting automated candidates."""
    annotations = (
        annotations_data
        if isinstance(annotations_data, CoupleAnnotations)
        else CoupleAnnotations.from_dict(annotations_data)
    )
    updated = dict(couple_result)
    series = dict(updated.get("series") or {})
    automated_candidates = [dict(c) for c in (updated.get("automated_candidates") or [])]
    pivot_candidates = list(series.get("pivot_candidates") or [])
    coverage = dict(updated.get("coverage") or {})
    stored_policy = (
        (updated.get("request") or {}).get("pose_origin_policy")
        or coverage.get("measurement_origin_policy")
        or cfg.dancer_analysis.pose_origin_policy
    )
    allow_prop = stored_policy == "allow_propagated_estimates"

    events, summaries = _apply_annotations_to_candidates(
        automated_candidates,
        annotations,
        pivot_candidates,
        series,
        coverage,
        cfg,
        allow_propagated_estimates=allow_prop,
    )
    updated["annotations"] = annotations.to_dict()
    updated["events"] = events
    updated["summaries"] = summaries
    updated["revision_id"] = int(updated.get("revision_id", 1)) + 1
    return updated
