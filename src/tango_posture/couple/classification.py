"""Opening-window Auto dancer-mode classifier and post-window scene-change diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..config import Config
from ..domain import FrameObservation
from .models import (
    AutoDecisionResult,
    DancerAnchor,
    DancerMode,
    PairedFrameSample,
    ResolvedDancerMode,
    TrackedDancerSample,
)
from .tracking import filter_and_deduplicate_poses, track_couple_frames


def _frame_weights(timestamps: list[float], win_in: float, win_out: float, max_gap_s: float) -> list[float]:
    n = len(timestamps)
    if n == 0 or win_out <= win_in:
        return []
    if n == 1:
        return [min(win_out - win_in, max_gap_s)]
    diffs = [timestamps[i + 1] - timestamps[i] for i in range(n - 1) if timestamps[i + 1] > timestamps[i]]
    typical_dt = float(np.median(diffs)) if diffs else max_gap_s
    edge_cap = max(typical_dt, min(max_gap_s, 0.25))
    weights: list[float] = []
    for i, t in enumerate(timestamps):
        left = 0.5 * (t - timestamps[i - 1]) if i > 0 else min(0.5 * typical_dt, max(0.0, t - win_in) + 0.5 * typical_dt)
        right = 0.5 * (timestamps[i + 1] - t) if i + 1 < n else min(0.5 * typical_dt, max(0.0, win_out - t))
        w = min(left, edge_cap) + min(right, edge_cap)
        weights.append(max(0.0, min(w, win_out - win_in)))
    total = sum(weights)
    target = max(0.0, win_out - win_in)
    if total > 0.0 and total > target:
        scale = target / total
        weights = [w * scale for w in weights]
    return weights


def _track_velocities(
    samples: list[TrackedDancerSample | None],
    fixed_scale_px: float | None,
    max_gap_s: float,
) -> list[tuple[float, float] | None]:
    """Compute instantaneous hip velocity vectors in normalized torso lengths/s for opening window."""
    n = len(samples)
    out: list[tuple[float, float] | None] = [None] * n
    scale = fixed_scale_px or 100.0
    if scale <= 0.0:
        return out
    for i, s in enumerate(samples):
        if s is None or s.hip_px is None or not s.hip_eligible:
            continue
        left_idx: int | None = None
        right_idx: int | None = None
        for j in range(i - 1, -1, -1):
            sj = samples[j]
            if sj is not None and sj.hip_px is not None and (s.timestamp - sj.timestamp) <= max_gap_s:
                left_idx = j
                break
            if sj is not None and (s.timestamp - sj.timestamp) > max_gap_s:
                break
        for j in range(i + 1, n):
            sj = samples[j]
            if sj is not None and sj.hip_px is not None and (sj.timestamp - s.timestamp) <= max_gap_s:
                right_idx = j
                break
            if sj is not None and (sj.timestamp - s.timestamp) > max_gap_s:
                break
        if left_idx is not None and right_idx is not None:
            sl, sr = samples[left_idx], samples[right_idx]
            assert sl is not None and sr is not None and sl.hip_px is not None and sr.hip_px is not None
            dt = sr.timestamp - sl.timestamp
            if dt > 1e-4:
                out[i] = ((sr.hip_px[0] - sl.hip_px[0]) / (dt * scale), (sr.hip_px[1] - sl.hip_px[1]) / (dt * scale))
        elif left_idx is not None:
            sl = samples[left_idx]
            assert sl is not None and sl.hip_px is not None
            dt = s.timestamp - sl.timestamp
            if dt > 1e-4:
                out[i] = ((s.hip_px[0] - sl.hip_px[0]) / (dt * scale), (s.hip_px[1] - sl.hip_px[1]) / (dt * scale))
        elif right_idx is not None:
            sr = samples[right_idx]
            assert sr is not None and sr.hip_px is not None
            dt = sr.timestamp - s.timestamp
            if dt > 1e-4:
                out[i] = ((sr.hip_px[0] - s.hip_px[0]) / (dt * scale), (sr.hip_px[1] - s.hip_px[1]) / (dt * scale))
    return out


def _longest_continuous_span(
    flags: list[bool],
    timestamps: list[float],
    weights: list[float],
    max_gap_s: float,
    break_flags: list[bool] | None = None,
) -> float:
    best = 0.0
    cur = 0.0
    last_true_t: float | None = None
    for i, ok in enumerate(flags):
        broken = bool(break_flags[i]) if break_flags is not None else False
        if not ok or broken:
            cur = 0.0
            last_true_t = None
            continue
        if last_true_t is not None and (timestamps[i] - last_true_t) > max_gap_s + 1e-9:
            cur = weights[i]
        else:
            cur += weights[i]
        last_true_t = timestamps[i]
        best = max(best, cur)
    return float(best)


def classify_opening_window(
    observations: list[tuple[int, FrameObservation]],
    cfg: Config,
    *,
    clip_in: float,
    clip_out: float,
    requested_mode: DancerMode = "auto",
    forced_override: bool = False,
    analysis_region: dict[str, float] | None = None,
    anchors: list[DancerAnchor] | None = None,
) -> AutoDecisionResult:
    """Classify the configured opening window into Single, Couple, or Needs selection."""
    da_cfg = cfg.dancer_analysis
    win_in = float(clip_in)
    configured_window = float(da_cfg.auto_detection_window_s)
    win_out = min(float(clip_in) + configured_window, float(clip_out))
    eff_duration = max(0.0, win_out - win_in)

    if eff_duration <= 1e-6:
        resolved: ResolvedDancerMode = (
            requested_mode if requested_mode in ("single", "couple") else "needs_selection"
        )
        return AutoDecisionResult(
            detector_version="couple_auto_v1",
            requested_mode=requested_mode,
            resolved_mode=resolved,
            forced_override=forced_override,
            configured_window_s=configured_window,
            effective_window_in=win_in,
            effective_window_out=win_out,
            effective_duration_s=eff_duration,
            primary_eligible_duration_s=0.0,
            primary_coverage=0.0,
            primary_moving_duration_s=0.0,
            second_coexistence_duration_s=0.0,
            second_joint_coverage=0.0,
            max_continuous_paired_span_s=0.0,
            max_co_movement_span_s=0.0,
            reason="empty_opening_window",
            explanation="Clip range has zero duration for Auto dancer detection.",
        )

    window_obs = [
        (idx, obs)
        for idx, obs in sorted(observations, key=lambda it: (it[1].timestamp, it[0]))
        if win_in - 1e-9 <= obs.timestamp < win_out - 1e-9
    ]

    if not window_obs:
        resolved = (
            requested_mode if requested_mode in ("single", "couple") else "needs_selection"
        )
        return AutoDecisionResult(
            detector_version="couple_auto_v1",
            requested_mode=requested_mode,
            resolved_mode=resolved,
            forced_override=forced_override,
            configured_window_s=configured_window,
            effective_window_in=win_in,
            effective_window_out=win_out,
            effective_duration_s=eff_duration,
            primary_eligible_duration_s=0.0,
            primary_coverage=0.0,
            primary_moving_duration_s=0.0,
            second_coexistence_duration_s=0.0,
            second_joint_coverage=0.0,
            max_continuous_paired_span_s=0.0,
            max_co_movement_span_s=0.0,
            reason="no_frames_in_window",
            explanation=f"No decoded frames fell inside [{win_in:.2f}s, {win_out:.2f}s).",
        )

    tracking = track_couple_frames(window_obs, cfg, analysis_region=analysis_region, anchors=anchors)
    frames = tracking.frames
    timestamps = [pf.timestamp for pf in frames]
    weights = _frame_weights(timestamps, win_in, win_out, float(da_cfg.max_observation_gap_s))

    # Also count deduplicated raw poses >= 2 per frame for transient passer detection
    obs_map = {idx: obs for idx, obs in window_obs}
    raw_multi_flags = [
        len(filter_and_deduplicate_poses(obs_map[pf.frame_index], cfg, analysis_region)) >= 2
        for pf in frames
    ]

    a_elig = [pf.dancer_a is not None and pf.dancer_a.hip_eligible for pf in frames]
    b_elig = [pf.dancer_b is not None and pf.dancer_b.hip_eligible for pf in frames]
    dur_a = sum(w for ok, w in zip(a_elig, weights) if ok)
    dur_b = sum(w for ok, w in zip(b_elig, weights) if ok)

    primary_is_b = dur_b > dur_a
    prim_elig = b_elig if primary_is_b else a_elig
    sec_elig = a_elig if primary_is_b else b_elig
    primary_dur = sum(w for ok, w in zip(prim_elig, weights) if ok)
    primary_cov = min(1.0, primary_dur / max(1e-6, eff_duration))

    paired_flags = [
        bool(a and b and not pf.identity_break and not pf.shot_cut)
        for a, b, pf in zip(a_elig, b_elig, frames)
    ]
    break_flags = [pf.identity_break or pf.shot_cut for pf in frames]
    joint_dur = sum(w for ok, w in zip(paired_flags, weights) if ok)
    joint_cov = min(1.0, joint_dur / max(1e-6, eff_duration))

    second_coexist_dur = sum(
        w
        for ok_prim, ok_sec, raw_m, w in zip(prim_elig, sec_elig, raw_multi_flags, weights)
        if (ok_prim and ok_sec) or raw_m
    )

    vel_a = _track_velocities(
        [pf.dancer_a for pf in frames],
        tracking.fixed_scale_a_px or tracking.fixed_pair_scale_px,
        float(da_cfg.max_observation_gap_s),
    )
    vel_b = _track_velocities(
        [pf.dancer_b for pf in frames],
        tracking.fixed_scale_b_px or tracking.fixed_pair_scale_px,
        float(da_cfg.max_observation_gap_s),
    )

    vel_prim = vel_b if primary_is_b else vel_a
    vel_sec = vel_a if primary_is_b else vel_b
    stop_speed = float(da_cfg.stop_speed)

    prim_moving_dur = sum(
        w
        for ok, v, w in zip(prim_elig, vel_prim, weights)
        if ok and v is not None and math.hypot(v[0], v[1]) > stop_speed
    )
    sec_moving_dur = sum(
        w
        for ok, v, w in zip(sec_elig, vel_sec, weights)
        if ok and v is not None and math.hypot(v[0], v[1]) > stop_speed
    )

    pair_scale = tracking.fixed_pair_scale_px or tracking.fixed_scale_a_px or tracking.fixed_scale_b_px or 100.0
    max_sep_px = float(da_cfg.together_max_hip_separation_torso_lengths) * pair_scale
    min_cos = float(da_cfg.together_direction_cosine_min)

    co_moving_flags: list[bool] = []
    for pf, paired, va, vb in zip(frames, paired_flags, vel_a, vel_b):
        if not paired or pf.dancer_a is None or pf.dancer_b is None or va is None or vb is None:
            co_moving_flags.append(False)
            continue
        ha, hb = pf.dancer_a.hip_px, pf.dancer_b.hip_px
        if ha is None or hb is None:
            co_moving_flags.append(False)
            continue
        sep_px = math.hypot(ha[0] - hb[0], ha[1] - hb[1])
        sp_a = math.hypot(va[0], va[1])
        sp_b = math.hypot(vb[0], vb[1])
        if sep_px > max_sep_px or sp_a <= stop_speed or sp_b <= stop_speed:
            co_moving_flags.append(False)
            continue
        cos_sim = (va[0] * vb[0] + va[1] * vb[1]) / max(1e-9, sp_a * sp_b)
        co_moving_flags.append(cos_sim >= min_cos)

    max_paired_span = _longest_continuous_span(
        paired_flags, timestamps, weights, float(da_cfg.max_observation_gap_s), break_flags
    )
    max_comove_span = _longest_continuous_span(
        co_moving_flags, timestamps, weights, float(da_cfg.max_observation_gap_s), break_flags
    )
    co_move_total_dur = sum(w for ok, w in zip(co_moving_flags, weights) if ok)

    # Precedence: Couple > Single > Needs selection
    couple_ok = (
        primary_cov >= float(da_cfg.primary_coverage_min)
        and primary_dur >= float(da_cfg.primary_duration_min_s)
        and max_paired_span >= float(da_cfg.second_sustained_min_s)
        and joint_cov >= float(da_cfg.second_joint_coverage_min)
        and (max_comove_span >= float(da_cfg.together_min_s) or co_move_total_dur >= float(da_cfg.together_min_s))
    )

    competing_candidate = (
        prim_moving_dur > 0.0 and sec_moving_dur >= 0.5 * prim_moving_dur
    )
    single_ok = (
        not couple_ok
        and primary_cov >= float(da_cfg.primary_coverage_min)
        and primary_dur >= float(da_cfg.primary_duration_min_s)
        and second_coexist_dur <= float(da_cfg.brief_second_max_s)
        and not competing_candidate
    )

    if forced_override or requested_mode in ("single", "couple"):
        resolved_mode: ResolvedDancerMode = requested_mode  # type: ignore[assignment]
        reason = f"user_selected_{requested_mode}"
        explanation = (
            f"Run explicitly configured for {requested_mode.title()} mode "
            f"(opening window {win_in:.1f}–{win_out:.1f}s showed {primary_cov:.0%} primary coverage, "
            f"{joint_cov:.0%} joint coverage)."
        )
    elif couple_ok:
        resolved_mode = "couple"
        reason = "sustained_paired_co_movement"
        explanation = (
            f"Detected Couple over [{win_in:.1f}s, {win_out:.1f}s): primary coverage {primary_cov:.0%}, "
            f"joint coverage {joint_cov:.0%}, continuous pair span {max_paired_span:.1f}s, "
            f"co-movement {max(max_comove_span, co_move_total_dur):.1f}s."
        )
    elif single_ok:
        resolved_mode = "single"
        reason = "single_foreground_dancer"
        explanation = (
            f"Detected Single dancer over [{win_in:.1f}s, {win_out:.1f}s): primary coverage {primary_cov:.0%} "
            f"({primary_dur:.1f}s), second person coexistence {second_coexist_dur:.2f}s <= {da_cfg.brief_second_max_s:.2f}s."
        )
    else:
        resolved_mode = "needs_selection"
        if primary_cov < float(da_cfg.primary_coverage_min) or primary_dur < float(da_cfg.primary_duration_min_s):
            reason = "insufficient_primary_coverage"
            explanation = (
                f"Opening window [{win_in:.1f}s, {win_out:.1f}s) had only {primary_cov:.0%} "
                f"({primary_dur:.1f}s) eligible primary observations; choose Single or Couple to proceed."
            )
        elif second_coexist_dur > float(da_cfg.brief_second_max_s) and not couple_ok:
            if competing_candidate:
                reason = "competing_single_candidates"
                explanation = (
                    f"Two distinct candidates appeared over [{win_in:.1f}s, {win_out:.1f}s) without "
                    f"meeting Couple co-movement criteria (joint {joint_cov:.0%}, co-movement {max(max_comove_span, co_move_total_dur):.1f}s)."
                )
            else:
                reason = "sustained_bystander_without_co_movement"
                explanation = (
                    f"Second person present for {second_coexist_dur:.1f}s (> {da_cfg.brief_second_max_s:.1f}s) "
                    f"without sustained paired Couple motion."
                )
        else:
            reason = "ambiguous_scene_evidence"
            explanation = (
                f"Opening window [{win_in:.1f}s, {win_out:.1f}s) did not conclusively match Single or Couple."
            )

    diagnostics = list(tracking.diagnostics)
    if eff_duration < configured_window - 1e-3:
        diagnostics.append(
            {
                "type": "short_clip_window",
                "timestamp": win_out,
                "message": (
                    f"Clip duration ({eff_duration:.2f}s) is shorter than configured Auto window "
                    f"({configured_window:.2f}s); inspected [{win_in:.2f}s, {win_out:.2f}s)."
                ),
            }
        )

    previews = []
    for p in tracking.candidate_previews:
        d_el = dur_a if p.candidate_id == "A" else dur_b
        p.eligible_duration_s = float(d_el)
        p.coverage = min(1.0, d_el / max(1e-6, eff_duration))
        previews.append(p)

    return AutoDecisionResult(
        detector_version="couple_auto_v1",
        requested_mode=requested_mode,
        resolved_mode=resolved_mode,
        forced_override=forced_override,
        configured_window_s=configured_window,
        effective_window_in=win_in,
        effective_window_out=win_out,
        effective_duration_s=eff_duration,
        primary_eligible_duration_s=float(primary_dur),
        primary_coverage=float(primary_cov),
        primary_moving_duration_s=float(prim_moving_dur),
        second_coexistence_duration_s=float(second_coexist_dur),
        second_joint_coverage=float(joint_cov),
        max_continuous_paired_span_s=float(max_paired_span),
        max_co_movement_span_s=float(max(max_comove_span, co_move_total_dur)),
        reason=reason,
        explanation=explanation,
        diagnostics=diagnostics,
        candidate_previews=previews,
    )


def detect_post_window_scene_changes(
    frames: list[PairedFrameSample],
    auto_decision: AutoDecisionResult,
    cfg: Config,
) -> list[dict[str, Any]]:
    """Record timestamped diagnostics if a second dancer enters/exits after the opening window."""
    if not frames:
        return []
    win_out = float(auto_decision.effective_window_out)
    post = [f for f in frames if f.timestamp >= win_out - 1e-9]
    if not post:
        return []
    da_cfg = cfg.dancer_analysis
    timestamps = [f.timestamp for f in post]
    weights = _frame_weights(
        timestamps,
        win_out,
        max(win_out + 1e-3, timestamps[-1] + 0.04),
        float(da_cfg.max_observation_gap_s),
    )
    breaks = [f.identity_break or f.shot_cut for f in post]
    diagnostics: list[dict[str, Any]] = []

    if auto_decision.resolved_mode == "single":
        both_flags = [
            f.dancer_a is not None and f.dancer_a.hip_eligible and f.dancer_b is not None and f.dancer_b.hip_eligible
            for f in post
        ]
        span = _longest_continuous_span(both_flags, timestamps, weights, float(da_cfg.max_observation_gap_s), breaks)
        if span >= float(da_cfg.second_sustained_min_s):
            first_t = next((f.timestamp for f, ok in zip(post, both_flags) if ok), win_out)
            diagnostics.append(
                {
                    "type": "post_window_second_dancer_entered",
                    "timestamp": float(first_t),
                    "message": (
                        f"A second dancer appeared for {span:.1f}s after the opening window at {first_t:.2f}s; "
                        "the run remained in Single mode as resolved."
                    ),
                }
            )
    elif auto_decision.resolved_mode == "couple":
        missing_partner = [
            (f.dancer_a is None or not f.dancer_a.hip_eligible)
            or (f.dancer_b is None or not f.dancer_b.hip_eligible)
            for f in post
        ]
        span = _longest_continuous_span(
            missing_partner, timestamps, weights, float(da_cfg.max_observation_gap_s), breaks
        )
        if span >= float(da_cfg.second_sustained_min_s):
            first_t = next((f.timestamp for f, ok in zip(post, missing_partner) if ok), win_out)
            diagnostics.append(
                {
                    "type": "post_window_partner_unobserved",
                    "timestamp": float(first_t),
                    "message": (
                        f"Only one tracked dancer was observed for {span:.1f}s starting at {first_t:.2f}s; "
                        "the run remained in Couple mode and reported unavailable pair intervals."
                    ),
                }
            )
    return diagnostics
