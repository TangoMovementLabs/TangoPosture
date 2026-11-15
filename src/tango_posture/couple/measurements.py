"""Couple measurements across coordination, step timing, transitions, posture axes, and turns."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import numpy as np

from ..config import Config
from ..domain import FrameObservation
from ..geometry import head_horizontal_alignment
from ..movement import analyse_movement_observations, wrapped_angle_diff
from ..pose.interpolation import interpolate_tracked_dancer_poses
from .classification import classify_opening_window, detect_post_window_scene_changes
from .models import (
    AutoDecisionResult,
    CoupleAnalysisRequest,
    CoupleAnalysisResult,
    CoupleAnnotations,
    CoupleEvent,
    CoupleSummaryValue,
    PairedFrameSample,
    TrackedDancerSample,
)
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
) -> list[CoupleEvent]:
    """Detect start/restart, slowing, and settled events along one subject's travel speed curve."""
    events: list[CoupleEvent] = []
    # Partition into contiguous finite speed segments
    segments: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
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
        cur.append((float(t), abs(float(v))))
        last_t = float(t)
    if cur:
        segments.append(cur)

    for seg in segments:
        if len(seg) < 3:
            continue
        state = "IDLE"
        active_speeds: list[float] = []
        slowing_emitted = False

        for pos, (t, sp) in enumerate(seg):
            if state == "IDLE":
                if (
                    sp > start_speed
                    and pos > 0
                    and seg[pos - 1][1] <= start_speed
                    and _stays_above(seg, pos, start_speed, persistence_s)
                ):
                    t_prev, sp_prev = seg[pos - 1]
                    t_cross = _interpolate_crossing(t_prev, sp_prev, t, sp, start_speed)
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
                            status="measured",
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
                        and _stays_below(seg, pos, slow_thresh, persistence_s)
                    ):
                        t_prev, sp_prev = seg[pos - 1]
                        t_cross = _interpolate_crossing(t_prev, sp_prev, t, sp, slow_thresh)
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
                                status="measured",
                                metadata={"steady_median_speed": round(v_med, 4)},
                            )
                        )
                        slowing_emitted = True

                # Detect settle
                if (
                    pos > 0
                    and sp < stop_speed
                    and seg[pos - 1][1] >= stop_speed
                    and _stays_below(seg, pos, stop_speed, persistence_s)
                ):
                    t_prev, sp_prev = seg[pos - 1]
                    t_cross = _interpolate_crossing(t_prev, sp_prev, t, sp, stop_speed)
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
                            status="measured",
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


def _evaluate_transitions(
    timestamps: list[float],
    v_pair: list[float | None],
    pair_hip_eligible: list[bool],
    break_mask: list[bool],
    slow_events: list[dict[str, Any]],
    settle_events: list[dict[str, Any]],
    annotations: CoupleAnnotations,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate secondary speed peaks and direction reversals inside transition windows."""
    da_cfg = cfg.dancer_analysis
    persistence_s = float(da_cfg.persistence_s)
    rise_thresh = float(da_cfg.secondary_speed_rise)
    stop_speed = float(da_cfg.stop_speed)
    min_cov = float(da_cfg.transition_min_coverage)

    # Identify transition brackets [t_slow, t_end]
    windows: list[tuple[float, float]] = []
    for ev in sorted(slow_events, key=lambda e: float(e["timestamp"])):
        t0 = float(ev["timestamp"])
        after = [
            float(s["timestamp"])
            for s in settle_events
            if float(s["timestamp"]) > t0 + 0.05
        ]
        t1 = min(after) if after else min(timestamps[-1], t0 + 2.0) if timestamps else t0 + 2.0
        if t1 > t0:
            windows.append((t0, t1))

    if not windows and len(timestamps) >= 3:
        # Also inspect any interval between sign reversal or transition marks
        for mark in annotations.planned_transitions:
            t0 = float(mark.timestamp) - 0.5
            t1 = float(mark.window_end if mark.window_end is not None else mark.timestamp) + 0.5
            if t1 > t0:
                windows.append((t0, t1))

    # Determine dominant motion sign per phase for reversal check
    non_null_v = [float(v) for v in v_pair if v is not None and abs(v) > stop_speed]
    dom_sign = 1.0 if (sum(1.0 if v >= 0 else -1.0 for v in non_null_v) >= 0) else -1.0

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

    total_win_covs: list[float] = []
    unintended_peaks = 0
    unintended_reversals = 0
    planned_pauses = sum(1 for m in planned_marks if m.kind == "planned_pause")
    planned_reversals = sum(1 for m in planned_marks if m.kind == "planned_reversal")
    settling_steps = sum(1 for m in planned_marks if m.kind == "settling_step")
    unavailable_windows = 0

    for w_idx, (t0, t1) in enumerate(windows):
        in_idx = [i for i, t in enumerate(timestamps) if t0 - 1e-6 <= t <= t1 + 1e-6]
        if not in_idx:
            continue
        cov = sum(1 for i in in_idx if pair_hip_eligible[i]) / len(in_idx)
        total_win_covs.append(cov)
        if cov < min_cov:
            unavailable_windows += 1
            continue

        # Build valid speeds inside window
        win_pts = [(timestamps[i], v_pair[i], break_mask[i]) for i in in_idx if v_pair[i] is not None]
        if len(win_pts) < 3:
            continue

        # 1) Secondary speed peak check after slowing onset
        running_min = abs(float(win_pts[0][1]))  # type: ignore[arg-type]
        peak_found = False
        for k, (tk, vk, bk) in enumerate(win_pts):
            if bk:
                running_min = abs(float(vk))  # type: ignore[arg-type]
                continue
            sp = abs(float(vk))  # type: ignore[arg-type]
            if sp < running_min:
                running_min = sp
            elif not peak_found and (sp - running_min) >= rise_thresh:
                # Confirm sustained rise for persistence_s
                end_t = tk + persistence_s
                sustained = all(
                    abs(float(win_pts[m][1])) - running_min >= rise_thresh * 0.6  # type: ignore[arg-type]
                    for m in range(k, len(win_pts))
                    if win_pts[m][0] <= end_t
                )
                if sustained:
                    is_settling_step = _overlaps_planned(tk, end_t, "settling_step") or _overlaps_planned(
                        tk, end_t, "planned_pause"
                    )
                    if not is_settling_step:
                        unintended_peaks += 1
                    cue_events.append(
                        CoupleEvent(
                            event_id=f"cand_speed_peak_{w_idx}_{k}",
                            event_type="speed_cue",
                            subject="pair",
                            timestamp=round(tk, 4),
                            end_timestamp=round(min(t1, end_t), 4),
                            value=round(sp - running_min, 4),
                            unit="pair_torso_lengths/s",
                            status="user_confirmed" if is_settling_step else "measured",
                            reason="planned_settling_step" if is_settling_step else "secondary_speed_peak",
                            metadata={
                                "cue_kind": "planned_settling_step" if is_settling_step else "secondary_speed_peak",
                                "window_index": w_idx,
                            },
                        ).to_dict()
                    )
                    peak_found = True

        # 2) Direction reversal check
        in_rev_start: float | None = None
        for k, (tk, vk, bk) in enumerate(win_pts):
            if bk:
                in_rev_start = None
                continue
            vv = float(vk)  # type: ignore[arg-type]
            opposite = (vv * dom_sign) < -stop_speed
            if opposite:
                if in_rev_start is None:
                    in_rev_start = tk
                if (tk - in_rev_start) >= persistence_s - 1e-6:
                    is_planned = _overlaps_planned(in_rev_start, tk, "planned_reversal")
                    if not is_planned:
                        unintended_reversals += 1
                    cue_events.append(
                        CoupleEvent(
                            event_id=f"cand_reversal_{w_idx}_{k}",
                            event_type="speed_cue",
                            subject="pair",
                            timestamp=round(in_rev_start, 4),
                            end_timestamp=round(tk, 4),
                            value=round(abs(vv), 4),
                            unit="pair_torso_lengths/s",
                            status="user_confirmed" if is_planned else "measured",
                            reason="planned_reversal" if is_planned else "unintended_reversal",
                            metadata={
                                "cue_kind": "planned_reversal" if is_planned else "unintended_reversal",
                                "window_index": w_idx,
                            },
                        ).to_dict()
                    )
                    break
            else:
                in_rev_start = None

    # If user gave no slow event triggers but clip itself reverses direction, inspect whole clip once
    if not windows and len(timestamps) >= 3:
        cov = sum(1 for ok in pair_hip_eligible if ok) / len(timestamps)
        total_win_covs.append(cov)
        if cov < min_cov:
            unavailable_windows += 1
        else:
            in_rev_start = None
            for idx, (tk, vk, bk) in enumerate(zip(timestamps, v_pair, break_mask)):
                if vk is None or bk:
                    in_rev_start = None
                    continue
                opposite = (vk * dom_sign) < -stop_speed
                if opposite:
                    if in_rev_start is None:
                        in_rev_start = tk
                    if (tk - in_rev_start) >= persistence_s - 1e-6:
                        is_planned = _overlaps_planned(in_rev_start, tk, "planned_reversal")
                        if not is_planned:
                            unintended_reversals += 1
                        cue_events.append(
                            CoupleEvent(
                                event_id=f"cand_reversal_clip_{idx}",
                                event_type="speed_cue",
                                subject="pair",
                                timestamp=round(in_rev_start, 4),
                                end_timestamp=round(tk, 4),
                                value=round(abs(vk), 4),
                                unit="pair_torso_lengths/s",
                                status="user_confirmed" if is_planned else "measured",
                                reason="planned_reversal" if is_planned else "unintended_reversal",
                                metadata={
                                    "cue_kind": "planned_reversal" if is_planned else "unintended_reversal",
                                },
                            ).to_dict()
                        )
                        break
                else:
                    in_rev_start = None

    mean_cov = float(np.mean(total_win_covs)) if total_win_covs else 0.0
    summary_status = (
        "unavailable"
        if (total_win_covs and unavailable_windows == len(total_win_covs))
        else "measured"
    )
    trans_summary = {
        "status": summary_status,
        "coverage": round(mean_cov, 4),
        "reason": "insufficient_transition_coverage" if summary_status == "unavailable" else "",
        "secondary_speed_peaks": unintended_peaks if summary_status != "unavailable" else None,
        "unintended_reversals": unintended_reversals if summary_status != "unavailable" else None,
        "planned_pauses": planned_pauses,
        "planned_reversals": planned_reversals,
        "settling_steps": settling_steps,
    }
    return cue_events, trans_summary


def _turn_events_from_annotations_and_candidates(
    annotations: CoupleAnnotations,
    auto_turn_suggestions: bool,
    pivot_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build paired turn duration, exit timing, pivot behavior, and embrace opening events."""
    items = [item for item in annotations.items if not item.deleted]
    turn_starts = sorted(
        [i for i in items if i.event_type == "turn_start"],
        key=lambda i: i.timestamp,
    )
    turn_ends = sorted(
        [i for i in items if i.event_type == "turn_end"],
        key=lambda i: i.timestamp,
    )
    exits = sorted([i for i in items if i.event_type == "exit"], key=lambda i: i.timestamp)

    turns: list[dict[str, Any]] = []
    used_ends: set[int] = set()
    used_exits: set[int] = set()

    for idx_s, st in enumerate(turn_starts):
        end_idx = next(
            (j for j, en in enumerate(turn_ends) if j not in used_ends and en.timestamp >= st.timestamp),
            None,
        )
        if end_idx is None:
            continue
        used_ends.add(end_idx)
        en = turn_ends[end_idx]
        duration_s = en.timestamp - st.timestamp

        # Match closest exit after turn_start (prefer within +-2s of turn_end)
        exit_idx: int | None = None
        best_dt = math.inf
        for je, ex in enumerate(exits):
            if je in used_exits:
                continue
            if ex.timestamp >= st.timestamp - 0.10:
                dt = abs(ex.timestamp - en.timestamp)
                if dt < best_dt and dt <= 3.0:
                    best_dt = dt
                    exit_idx = je
        exit_item = exits[exit_idx] if exit_idx is not None else None
        if exit_idx is not None:
            used_exits.add(exit_idx)

        exit_diff_ms: float | None = None
        overlap = False
        if exit_item is not None:
            exit_diff_ms = (exit_item.timestamp - en.timestamp) * 1000.0
            overlap = exit_item.timestamp < en.timestamp

        turns.append(
            {
                "event_id": f"turn_{idx_s}",
                "event_type": "turn",
                "subject": st.subject,
                "timestamp": float(st.timestamp),
                "end_timestamp": float(en.timestamp),
                "bracket_start": float(st.timestamp),
                "bracket_end": float(en.timestamp),
                "value": round(duration_s, 4),
                "unit": "seconds",
                "status": "user_confirmed",
                "paired_event_id": exit_item.annotation_id if exit_item else None,
                "lag_ms": round(exit_diff_ms, 2) if exit_diff_ms is not None else None,
                "beat_offset_ms": None,
                "beat_fraction": None,
                "reason": "exit_occurred_during_turn" if overlap else "",
                "metadata": {
                    "turn_duration_s": round(duration_s, 4),
                    "exit_timestamp": float(exit_item.timestamp) if exit_item else None,
                    "exit_minus_turn_end_ms": round(exit_diff_ms, 2) if exit_diff_ms is not None else None,
                    "overlap": overlap,
                    "pivot_behavior_a": st.metadata.get("pivot_behavior_a", st.pivot_behavior or "unknown"),
                    "pivot_behavior_b": st.metadata.get("pivot_behavior_b", en.pivot_behavior or "unknown"),
                    "embrace_opening": st.label or en.label or "",
                },
            }
        )

    if auto_turn_suggestions:
        for p_idx, piv in enumerate(pivot_candidates):
            p_start = float(piv.get("start_time", 0.0))
            p_end = float(piv.get("end_time", p_start))
            overlaps_user = any(
                t["timestamp"] - 0.3 <= p_end and t["end_timestamp"] + 0.3 >= p_start
                for t in turns
            )
            if not overlaps_user and p_end > p_start:
                turns.append(
                    {
                        "event_id": f"auto_turn_{p_idx}",
                        "event_type": "turn",
                        "subject": str(piv.get("subject", "pair")),
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
    turn_summary = {
        "turn_count": len(turns),
        "overlap_count": sum(1 for t in turns if t["metadata"].get("overlap")),
        "median_turn_duration_s": dur_med,
        "iqr_turn_duration_s": dur_iqr,
        "median_exit_minus_turn_end_ms": exit_med,
        "iqr_exit_minus_turn_end_ms": exit_iqr,
        "status": "user_confirmed" if any(t["status"] == "user_confirmed" for t in turns) else (
            "estimated" if turns else "unavailable"
        ),
    }
    return turns, turn_summary


def _compute_summaries_from_events_and_series(
    *,
    series: dict[str, list[dict[str, Any]]],
    events: list[dict[str, Any]],
    trans_summary: dict[str, Any],
    turn_summary: dict[str, Any],
    coverage: dict[str, Any],
    baseline_passages: list[tuple[float, float]],
) -> dict[str, Any]:
    """Build structured summary metrics across all 5 Couple domains."""
    # 1. Coordination
    coord: dict[str, Any] = {}
    for etype in ("start", "slow", "settle"):
        ev_a = [e for e in events if e.get("event_type") == etype and e.get("subject") == "A"]
        lags = [float(e["lag_ms"]) for e in ev_a if e.get("lag_ms") is not None]
        med, iqr = _median_iqr(lags)
        unpaired = sum(
            1
            for e in events
            if e.get("event_type") == etype and e.get("lag_ms") is None
        )
        coord[f"{etype}_lag_ms"] = CoupleSummaryValue(
            measure=f"{etype}_lag_ms",
            subject="B-A",
            value=round(med, 2) if med is not None else None,
            iqr=round(iqr, 2) if iqr is not None else None,
            count=len(lags),
            unit="ms",
            status="measured" if lags else "unavailable",
            coverage=float(coverage.get("pair_hip_coverage", 0.0)),
            reason="" if lags else "no_paired_events",
            timestamp_or_bracket="",
        ).to_dict()
        coord[f"{etype}_paired_count"] = len(lags)
        coord[f"{etype}_unpaired_count"] = unpaired

    # 2. Step and beat timing
    step_timing: dict[str, Any] = {}
    for subj in ("A", "B"):
        offsets = [
            float(e["beat_offset_ms"])
            for e in events
            if e.get("event_type") == "contact" and e.get("subject") == subj and e.get("beat_offset_ms") is not None
        ]
        fracs = [
            float(e["beat_fraction"])
            for e in events
            if e.get("event_type") == "contact" and e.get("subject") == subj and e.get("beat_fraction") is not None
        ]
        contacts_total = sum(1 for e in events if e.get("event_type") == "contact" and e.get("subject") == subj)
        med_ms, iqr_ms = _median_iqr(offsets)
        med_fr, _ = _median_iqr(fracs)
        step_timing[subj] = {
            "contact_count": contacts_total,
            "matched_count": len(offsets),
            "median_beat_offset_ms": round(med_ms, 2) if med_ms is not None else None,
            "iqr_beat_offset_ms": round(iqr_ms, 2) if iqr_ms is not None else None,
            "median_beat_fraction": round(med_fr, 4) if med_fr is not None else None,
            "status": (
                "user_confirmed"
                if any(e.get("status") == "user_confirmed" for e in events if e.get("event_type") == "contact" and e.get("subject") == subj)
                else ("estimated" if contacts_total else "unavailable")
            ),
        }

    # 3. Posture & axes
    def _angle_summary(key: str, subject: str, cov_key: str) -> dict[str, Any]:
        pts = series.get(key) or []
        vals = [
            float(p["value"])
            for p in pts
            if p.get("value") is not None and p.get("status") in ("measured", "user_confirmed")
        ]
        med, iqr = _median_iqr(vals)
        base_change: float | None = None
        if baseline_passages and pts:
            bvals = [
                float(p["value"])
                for p in pts
                if p.get("value") is not None
                and any(t0 <= float(p["timestamp"]) <= t1 for t0, t1 in baseline_passages)
            ]
            if len(bvals) >= 3 and med is not None:
                ref = float(np.median(bvals))
                base_change = round(wrapped_angle_diff(med, ref), 2)
        return {
            "measure": key,
            "subject": subject,
            "median_deg": round(med, 2) if med is not None else None,
            "iqr_deg": round(iqr, 2) if iqr is not None else None,
            "baseline_change_deg": base_change,
            "sample_count": len(vals),
            "coverage": float(coverage.get(cov_key, 0.0)),
            "status": "measured" if vals else "unavailable",
        }

    posture = {
        "torso_tilt_a": _angle_summary("torso_tilt_a", "A", "a_torso_coverage"),
        "torso_tilt_b": _angle_summary("torso_tilt_b", "B", "b_torso_coverage"),
        "couple_midpoint_tilt": _angle_summary("couple_midpoint_tilt", "pair", "pair_torso_coverage"),
        "torso_separation": _angle_summary("torso_separation", "pair", "pair_torso_coverage"),
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
    slow_evs = [e for e in paired if e.get("event_type") == "slow"]
    settle_evs = [e for e in paired if e.get("event_type") == "settle"]
    non_cue = [e for e in paired if e.get("event_type") != "speed_cue"]
    trans_cues, trans_summary = _evaluate_transitions(
        timestamps, v_pair, hip_ok, breaks, slow_evs, settle_evs, annotations, cfg
    )
    paired = non_cue + trans_cues

    # Turns & exit
    auto_turns_flag = bool(cfg.dancer_analysis.auto_turn_suggestions)
    turns, turn_summary = _turn_events_from_annotations_and_candidates(
        annotations, auto_turns_flag, pivot_candidates
    )
    paired = [e for e in paired if e.get("event_type") != "turn"] + turns
    paired.sort(key=lambda e: float(e.get("timestamp", 0.0)))

    summaries = _compute_summaries_from_events_and_series(
        series=series,
        events=paired,
        trans_summary=trans_summary,
        turn_summary=turn_summary,
        coverage=coverage,
        baseline_passages=annotations.baseline_passages,
    )
    return paired, summaries


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

    def _norm_from_canonical(pt: tuple[float, float], w: int, h: int) -> list[float]:
        return [round(pt[0] / max(1, w - 1), 6), round(-pt[1] / max(1, h - 1), 6)]

    for pf in frames:
        t = float(pf.timestamp)
        ta: float | None = None
        tb: float | None = None
        if pf.dancer_a and pf.dancer_a.torso_eligible and pf.dancer_a.hip_px and pf.dancer_a.shoulder_px:
            dx = pf.dancer_a.shoulder_px[0] - pf.dancer_a.hip_px[0]
            dy = pf.dancer_a.shoulder_px[1] - pf.dancer_a.hip_px[1]
            ta = math.degrees(math.atan2(dx, dy))
            tilt_a_series.append({"timestamp": t, "value": round(ta, 3), "status": "measured", "reason": ""})
        else:
            tilt_a_series.append({"timestamp": t, "value": None, "status": "unavailable", "reason": "torso_occluded"})

        if pf.dancer_b and pf.dancer_b.torso_eligible and pf.dancer_b.hip_px and pf.dancer_b.shoulder_px:
            dx = pf.dancer_b.shoulder_px[0] - pf.dancer_b.hip_px[0]
            dy = pf.dancer_b.shoulder_px[1] - pf.dancer_b.hip_px[1]
            tb = math.degrees(math.atan2(dx, dy))
            tilt_b_series.append({"timestamp": t, "value": round(tb, 3), "status": "measured", "reason": ""})
        else:
            tilt_b_series.append({"timestamp": t, "value": None, "status": "unavailable", "reason": "torso_occluded"})

        hip_only_norm: list[float] | None = None
        pair_hip_norm: list[float] | None = None
        pair_sh_norm: list[float] | None = None

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
            midpoint_series.append({"timestamp": t, "value": round(mid_tilt, 3), "status": "measured", "reason": ""})
            separation_series.append({"timestamp": t, "value": round(sep, 3), "status": "measured", "reason": ""})
            pair_hip_norm = _norm_from_canonical(hp, pf.width, pf.height)
            pair_sh_norm = _norm_from_canonical(sp, pf.width, pf.height)
        elif pf.pair_hip_eligible and pf.dancer_a and pf.dancer_b:
            assert pf.dancer_a.hip_px and pf.dancer_b.hip_px
            hp = (
                0.5 * (pf.dancer_a.hip_px[0] + pf.dancer_b.hip_px[0]),
                0.5 * (pf.dancer_a.hip_px[1] + pf.dancer_b.hip_px[1]),
            )
            hip_only_norm = _norm_from_canonical(hp, pf.width, pf.height)
            midpoint_series.append(
                {
                    "timestamp": t,
                    "value": None,
                    "status": "unavailable",
                    "reason": "shoulder_occluded_hip_only_travel_point",
                }
            )
            separation_series.append({"timestamp": t, "value": None, "status": "unavailable", "reason": "shoulder_occluded"})
        else:
            midpoint_series.append({"timestamp": t, "value": None, "status": "unavailable", "reason": "pair_unavailable"})
            separation_series.append({"timestamp": t, "value": None, "status": "unavailable", "reason": "pair_unavailable"})

        # Optional per-dancer head pitch from raw pose
        for d_sample, dest in ((pf.dancer_a, head_a_series), (pf.dancer_b, head_b_series)):
            if d_sample and d_sample.raw_pose and d_sample.head_eligible:
                hp_res = head_horizontal_alignment(
                    d_sample.raw_pose, pf.width, pf.height, float(cfg.head_neutral_offset)
                )
                if hp_res is not None:
                    dest.append({"timestamp": t, "value": round(float(hp_res[0]), 3), "status": "measured", "reason": ""})
                    continue
            dest.append({"timestamp": t, "value": None, "status": "unavailable", "reason": "head_landmarks_unavailable"})

        def _serialize_display_pose(d_sample: TrackedDancerSample | None) -> dict[str, Any] | None:
            if d_sample is None or not d_sample.display_pose:
                return None
            return {
                "landmarks": {
                    k: [round(lm.x, 5), round(lm.y, 5), round(lm.visibility or 0.0, 3)]
                    for k, lm in d_sample.display_pose.items()
                    if k
                    in (
                        "left_shoulder",
                        "right_shoulder",
                        "left_hip",
                        "right_hip",
                        "left_ankle",
                        "right_ankle",
                        "nose",
                    )
                },
                "interpolated_display": bool(d_sample.interpolated_display),
                "quality_status": d_sample.quality_status,
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
                "travel_axis_unit": [round(u_travel[0], 5), round(u_travel[1], 5)],
            }
        )

    # §4.1 Motion event candidates
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
        )
    )

    # §4.2 Per-dancer apparent step contacts and pivot candidates from raw poses via existing Movement engine
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
                    [getter(pf).raw_pose] if (getter(pf) and getter(pf).raw_pose) else [],
                    [getter(pf).world_pose] if (getter(pf) and getter(pf).world_pose) else [],
                ),
            )
            for pf in frames
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
            {"timestamp": float(pf.timestamp), "value": round(v, 4) if v is not None else None}
            for pf, v in zip(frames, vel_a)
        ],
        "velocity_b": [
            {"timestamp": float(pf.timestamp), "value": round(v, 4) if v is not None else None}
            for pf, v in zip(frames, vel_b)
        ],
        "velocity_pair": [
            {
                "timestamp": float(pf.timestamp),
                "value": round(v, 4) if v is not None else None,
                "pair_hip_eligible": bool(pf.pair_hip_eligible),
                "break_flag": bool(pf.shot_cut or pf.identity_break),
            }
            for pf, v in zip(frames, vel_pair)
        ],
        "torso_tilt_a": tilt_a_series,
        "torso_tilt_b": tilt_b_series,
        "couple_midpoint_tilt": midpoint_series,
        "torso_separation": separation_series,
        "head_pitch_a": head_a_series,
        "head_pitch_b": head_b_series,
        "pair_breaks": [{"timestamp": float(pf.timestamp)} for pf in frames if pf.shot_cut or pf.identity_break],
        "pivot_candidates": pivot_candidates,
    }

    total_n = max(1, len(frames))
    coverage = {
        "frame_count": len(frames),
        "a_hip_coverage": round(sum(1 for pf in frames if pf.dancer_a and pf.dancer_a.hip_eligible) / total_n, 4),
        "b_hip_coverage": round(sum(1 for pf in frames if pf.dancer_b and pf.dancer_b.hip_eligible) / total_n, 4),
        "pair_hip_coverage": round(sum(1 for pf in frames if pf.pair_hip_eligible) / total_n, 4),
        "a_torso_coverage": round(sum(1 for pf in frames if pf.dancer_a and pf.dancer_a.torso_eligible) / total_n, 4),
        "b_torso_coverage": round(sum(1 for pf in frames if pf.dancer_b and pf.dancer_b.torso_eligible) / total_n, 4),
        "pair_torso_coverage": round(sum(1 for pf in frames if pf.pair_torso_eligible) / total_n, 4),
        "a_head_coverage": round(sum(1 for pf in frames if pf.dancer_a and pf.dancer_a.head_eligible) / total_n, 4),
        "b_head_coverage": round(sum(1 for pf in frames if pf.dancer_b and pf.dancer_b.head_eligible) / total_n, 4),
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

    events, summaries = _apply_annotations_to_candidates(
        automated_candidates,
        annotations,
        pivot_candidates,
        series,
        coverage,
        cfg,
    )
    updated["annotations"] = annotations.to_dict()
    updated["events"] = events
    updated["summaries"] = summaries
    updated["revision_id"] = int(updated.get("revision_id", 1)) + 1
    return updated
