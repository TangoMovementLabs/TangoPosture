"""Overlay rendering for tracked Couple axes, display-interpolated gaps, steps, and turns."""

from __future__ import annotations

import copy
import math
from typing import Any

import cv2
import numpy as np

from ..overlay import TORSO_AXIS_EXTENSION

# BGR palette matching Qt's Format_BGR888 video frames
COLOR_DANCER_A_BGR = (11, 158, 245)      # Warm amber
COLOR_DANCER_B_BGR = (166, 184, 20)      # Teal/cyan
COLOR_COUPLE_MID_BGR = (252, 132, 192)   # Soft violet
COLOR_VERTICAL_REF_BGR = (255, 220, 0)   # Cyan-blue vertical reference
COLOR_HIP_ONLY_BGR = (120, 210, 255)     # Light amber for hip-only travel point
COLOR_EVENT_BGR = (90, 230, 140)         # Mint green for active turn/step cue


def _draw_dashed_line(
    image: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_px: int = 10,
    gap_px: int = 8,
) -> None:
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    length = math.hypot(dx, dy)
    if length < 1.0:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        end = min(length, pos + dash_px)
        q0 = (round(p0[0] + ux * pos), round(p0[1] + uy * pos))
        q1 = (round(p0[0] + ux * end), round(p0[1] + uy * end))
        cv2.line(image, q0, q1, color, thickness, cv2.LINE_AA)
        pos += dash_px + gap_px


def _to_px(x_norm: float, y_norm: float, width: int, height: int) -> tuple[int, int]:
    return (round(float(x_norm) * max(1, width - 1)), round(float(y_norm) * max(1, height - 1)))


def _lm_vis(coords: list[float] | None) -> float:
    if not coords or len(coords) < 2:
        return 0.0
    return float(coords[2]) if len(coords) >= 3 and coords[2] is not None else 1.0


def _dancer_midpoints_px(
    dancer_data: dict[str, Any] | None,
    width: int,
    height: int,
    *,
    min_vis: float = 0.5,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    if not isinstance(dancer_data, dict):
        return None, None
    landmarks: dict[str, list[float]] = dancer_data.get("landmarks") or {}
    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    lh = landmarks.get("left_hip")
    rh = landmarks.get("right_hip")
    hip_px: tuple[int, int] | None = None
    sh_px: tuple[int, int] | None = None
    if lh is not None and rh is not None and _lm_vis(lh) >= min_vis and _lm_vis(rh) >= min_vis:
        hip_px = _to_px(0.5 * (lh[0] + rh[0]), 0.5 * (lh[1] + rh[1]), width, height)
    if ls is not None and rs is not None and _lm_vis(ls) >= min_vis and _lm_vis(rs) >= min_vis:
        sh_px = _to_px(0.5 * (ls[0] + rs[0]), 0.5 * (ls[1] + rs[1]), width, height)
    return hip_px, sh_px


def _draw_extended_axis(
    image: np.ndarray,
    hip_px: tuple[int, int],
    sh_px: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
    *,
    interpolated: bool = False,
    tilt_deg: float | None = None,
    thickness: int = 3,
    label_y_offset: int = -12,
    extra_suffix: str = "",
) -> None:
    height, width = image.shape[:2]
    hx, hy = hip_px
    sx, sy = sh_px
    dx, dy = sx - hx, sy - hy
    length = math.hypot(dx, dy)
    if length < 1.0:
        return
    # Fixed vertical reference behind torso/couple axis for intuitive tilt comparison.
    v_top = (hx, max(0, round(hy - 1.30 * length)))
    v_bot = (hx, min(height - 1, round(hy + 0.25 * length)))
    _draw_dashed_line(image, v_top, v_bot, COLOR_VERTICAL_REF_BGR, thickness=1, dash_px=8, gap_px=8)

    ext_start = (
        round(hx - TORSO_AXIS_EXTENSION * dx),
        round(hy - TORSO_AXIS_EXTENSION * dy),
    )
    ext_end = (
        round(sx + TORSO_AXIS_EXTENSION * dx),
        round(sy + TORSO_AXIS_EXTENSION * dy),
    )
    if interpolated:
        _draw_dashed_line(image, ext_start, ext_end, color, thickness=thickness)
        cv2.circle(image, hip_px, 6, color, 2, cv2.LINE_AA)
        cv2.circle(image, sh_px, 6, color, 2, cv2.LINE_AA)
    else:
        cv2.line(image, ext_start, ext_end, color, thickness, cv2.LINE_AA)
        cv2.circle(image, hip_px, 5, color, -1, cv2.LINE_AA)
        cv2.circle(image, sh_px, 5, color, -1, cv2.LINE_AA)

    angle_val = tilt_deg
    if angle_val is None:
        angle_val = math.degrees(math.atan2(dx, hy - sy))
    angle_txt = f": {angle_val:+.1f} deg" if math.isfinite(angle_val) else ""
    est_suffix = " (est)" if interpolated else ""
    tag = f"{label}{angle_txt}{extra_suffix}{est_suffix}"
    (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
    anchor_x = max(8, min(ext_end[0] + 6, width - tw - 8))
    anchor_y = max(22, min(ext_end[1] + label_y_offset, height - 10))
    cv2.putText(image, tag, (anchor_x, anchor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)


def _draw_dancer(
    image: np.ndarray,
    dancer_data: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
    *,
    show_steps: bool,
    tilt_deg: float | None = None,
    min_vis: float = 0.5,
) -> None:
    """Draw strictly the Couple-mode torso axis geometry (shoulder/hip bars, vertical ref, torso axis)."""
    height, width = image.shape[:2]
    landmarks: dict[str, list[float]] = dancer_data.get("landmarks") or {}
    if not landmarks:
        return
    interpolated = bool(dancer_data.get("interpolated_display", False)) or (
        dancer_data.get("pose_origin") in ("propagated", "display_interpolated")
    )

    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    lh = landmarks.get("left_hip")
    rh = landmarks.get("right_hip")

    hip_px, sh_px = _dancer_midpoints_px(dancer_data, width, height, min_vis=min_vis)

    # Subtle shoulder and hip crossbars anchoring H_i and S_i
    for left_lm, right_lm in ((ls, rs), (lh, rh)):
        if left_lm is not None and right_lm is not None and _lm_vis(left_lm) >= min_vis and _lm_vis(right_lm) >= min_vis:
            p_l = _to_px(left_lm[0], left_lm[1], width, height)
            p_r = _to_px(right_lm[0], right_lm[1], width, height)
            if interpolated:
                _draw_dashed_line(image, p_l, p_r, color, thickness=1, dash_px=6, gap_px=6)
            else:
                cv2.line(image, p_l, p_r, color, 1, cv2.LINE_AA)
            cv2.circle(image, p_l, 3, color, 1 if interpolated else -1, cv2.LINE_AA)
            cv2.circle(image, p_r, 3, color, 1 if interpolated else -1, cv2.LINE_AA)

    if hip_px and sh_px:
        _draw_extended_axis(
            image,
            hip_px,
            sh_px,
            color,
            label,
            interpolated=interpolated,
            tilt_deg=tilt_deg,
            thickness=3,
        )
    elif hip_px:
        cv2.circle(image, hip_px, 6, color, 2, cv2.LINE_AA)
        anchor = (max(8, min(hip_px[0] + 8, width - 120)), max(22, hip_px[1] - 10))
        cv2.putText(
            image, f"{label} (hip)", anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA
        )
    _ = show_steps


def render_couple_frame_overlay(
    frame_bgr: np.ndarray,
    overlay_data: dict[str, Any] | None,
    *,
    show_dancer_a: bool = True,
    show_dancer_b: bool = True,
    show_couple_axis: bool = True,
    show_steps_turns: bool = True,
    dancer_a_name: str = "A",
    dancer_b_name: str = "B",
    active_events: list[dict[str, Any]] | None = None,
    min_vis: float | None = None,
) -> np.ndarray:
    """Draw Couple axis overlay (Dancer A/B torso axes, pair midpoint axis, and active events)."""
    out = frame_bgr.copy()
    if overlay_data is None:
        return out
    height, width = out.shape[:2]
    raw_vis = overlay_data.get("min_landmark_visibility", 0.5) if min_vis is None else min_vis
    resolved_min_vis = float(raw_vis if raw_vis is not None else 0.5)

    tilt_a = overlay_data.get("tilt_a_deg")
    tilt_b = overlay_data.get("tilt_b_deg")
    if show_dancer_a and overlay_data.get("dancer_a"):
        _draw_dancer(
            out,
            overlay_data["dancer_a"],
            COLOR_DANCER_A_BGR,
            dancer_a_name or "A",
            show_steps=show_steps_turns,
            tilt_deg=float(tilt_a) if isinstance(tilt_a, int | float) else None,
            min_vis=resolved_min_vis,
        )
    if show_dancer_b and overlay_data.get("dancer_b"):
        _draw_dancer(
            out,
            overlay_data["dancer_b"],
            COLOR_DANCER_B_BGR,
            dancer_b_name or "B",
            show_steps=show_steps_turns,
            tilt_deg=float(tilt_b) if isinstance(tilt_b, int | float) else None,
            min_vis=resolved_min_vis,
        )

    y_cursor = 28
    if show_steps_turns and active_events:
        for ev in active_events[:4]:
            etype = str(ev.get("event_type", ""))
            subj = str(ev.get("subject", ""))
            status = str(ev.get("status", ""))
            text = f"{etype.upper()} ({subj}) [{status}]"
            cv2.putText(
                out,
                text,
                (14, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLOR_EVENT_BGR,
                2,
                cv2.LINE_AA,
            )
            y_cursor += 22

    if show_couple_axis:
        pair_hip = overlay_data.get("pair_hip_norm")
        pair_sh = overlay_data.get("pair_shoulder_norm")
        hip_only = overlay_data.get("hip_only_travel_point")
        mid_tilt = overlay_data.get("couple_midpoint_tilt_deg")
        sep_deg = overlay_data.get("torso_separation_deg")
        if pair_hip and pair_sh:
            a_hip, a_sh = _dancer_midpoints_px(
                overlay_data.get("dancer_a"), width, height, min_vis=resolved_min_vis
            )
            b_hip, b_sh = _dancer_midpoints_px(
                overlay_data.get("dancer_b"), width, height, min_vis=resolved_min_vis
            )
            if a_hip and b_hip:
                _draw_dashed_line(out, a_hip, b_hip, COLOR_COUPLE_MID_BGR, thickness=1, dash_px=6, gap_px=6)
            if a_sh and b_sh:
                _draw_dashed_line(out, a_sh, b_sh, COLOR_COUPLE_MID_BGR, thickness=1, dash_px=6, gap_px=6)
            hp = _to_px(pair_hip[0], pair_hip[1], width, height)
            sp = _to_px(pair_sh[0], pair_sh[1], width, height)
            sep_suffix = f" (sep {float(sep_deg):.1f} deg)" if isinstance(sep_deg, int | float) else ""
            _draw_extended_axis(
                out,
                hp,
                sp,
                COLOR_COUPLE_MID_BGR,
                "Couple",
                interpolated=False,
                tilt_deg=float(mid_tilt) if isinstance(mid_tilt, int | float) else None,
                thickness=4,
                label_y_offset=18,
                extra_suffix=sep_suffix,
            )
        elif hip_only:
            hp = _to_px(hip_only[0], hip_only[1], width, height)
            cv2.circle(out, hp, 6, COLOR_HIP_ONLY_BGR, 2, cv2.LINE_AA)
            cv2.putText(
                out,
                "Hip-only midpoint",
                (max(8, hp[0] + 10), max(22, hp[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                COLOR_HIP_ONLY_BGR,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                out,
                "Pair axis unavailable on this frame (need both dancers)",
                (14, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                COLOR_HIP_ONLY_BGR,
                2,
                cv2.LINE_AA,
            )
    return out


_FOOT_OVERLAY_LANDMARKS = frozenset(
    {
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    }
)

_SMOOTHED_OVERLAY_LANDMARKS = (
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
)


def _zero_phase_ema_optional(
    timestamps: list[float],
    values: list[float | None],
    *,
    tau_s: float,
    max_gap_s: float,
) -> list[float | None]:
    """Bridge brief display gaps and apply zero-phase forward-backward EMA smoothing."""
    n = len(values)
    if n == 0:
        return []
    out: list[float | None] = [
        float(v) if v is not None and math.isfinite(float(v)) else None for v in values
    ]
    # Bridge brief bracketed display gaps linearly so lines don't blink on single-frame dropouts
    prev_valid: int | None = None
    i = 0
    while i < n:
        if out[i] is not None:
            if prev_valid is not None and i - prev_valid > 1:
                dt_gap = timestamps[i] - timestamps[prev_valid]
                if 0.0 < dt_gap <= max_gap_s + 1e-9:
                    v0 = out[prev_valid]
                    v1 = out[i]
                    assert v0 is not None and v1 is not None
                    for k in range(prev_valid + 1, i):
                        frac = (timestamps[k] - timestamps[prev_valid]) / max(dt_gap, 1e-9)
                        out[k] = v0 + frac * (v1 - v0)
            prev_valid = i
        i += 1

    if tau_s <= 1e-4:
        return out

    # Partition into contiguous finite runs separated by > smooth_gap_s or None
    smooth_gap_s = max(float(max_gap_s), 0.50)
    runs: list[list[int]] = []
    current_run: list[int] = []
    for idx, val in enumerate(out):
        if val is None:
            if current_run:
                runs.append(current_run)
                current_run = []
            continue
        if current_run and (timestamps[idx] - timestamps[current_run[-1]]) > smooth_gap_s + 1e-9:
            runs.append(current_run)
            current_run = []
        current_run.append(idx)
    if current_run:
        runs.append(current_run)

    safe_tau = max(float(tau_s), 1e-3)
    for run in runs:
        if len(run) < 2:
            continue
        fwd = [float(out[run[0]])]  # type: ignore[arg-type]
        for k in range(1, len(run)):
            dt = max(1e-4, timestamps[run[k]] - timestamps[run[k - 1]])
            alpha = 1.0 - math.exp(-dt / safe_tau)
            raw_v = float(out[run[k]])  # type: ignore[arg-type]
            fwd.append(fwd[-1] + alpha * (raw_v - fwd[-1]))
        bwd = [0.0] * len(run)
        bwd[-1] = fwd[-1]
        for k in range(len(run) - 2, -1, -1):
            dt = max(1e-4, timestamps[run[k + 1]] - timestamps[run[k]])
            alpha = 1.0 - math.exp(-dt / safe_tau)
            bwd[k] = bwd[k + 1] + alpha * (fwd[k] - bwd[k + 1])
        for k, idx in enumerate(run):
            out[idx] = bwd[k]
    return out


def smooth_couple_frame_overlays(
    frame_overlays: list[dict[str, Any]],
    *,
    tau_s: float = 0.16,
    max_gap_s: float = 0.50,
) -> list[dict[str, Any]]:
    """Apply display-only zero-phase smoothing and gap bridging to Couple overlay geometry."""
    if not frame_overlays:
        return []
    smoothed = copy.deepcopy(frame_overlays)
    n = len(smoothed)
    timestamps: list[float] = []
    for idx, fo in enumerate(smoothed):
        ts = fo.get("timestamp") if isinstance(fo, dict) else None
        t_val = float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) and math.isfinite(ts) else idx / 30.0
        if timestamps and t_val <= timestamps[-1]:
            t_val = timestamps[-1] + 1.0 / 30.0
        timestamps.append(t_val)

    for dancer_key in ("dancer_a", "dancer_b"):
        for lm_key in _SMOOTHED_OVERLAY_LANDMARKS:
            is_foot = lm_key in _FOOT_OVERLAY_LANDMARKS
            vis_floor = 0.25 if is_foot else 0.5
            lm_tau_s = min(float(tau_s), 0.02) if is_foot else float(tau_s)
            xs: list[float | None] = []
            ys: list[float | None] = []
            vis_list: list[float | None] = []
            for fo in smoothed:
                d_data = fo.get(dancer_key) if isinstance(fo, dict) else None
                lms = d_data.get("landmarks") if isinstance(d_data, dict) else None
                coords = lms.get(lm_key) if isinstance(lms, dict) else None
                if (
                    isinstance(coords, (list, tuple))
                    and len(coords) >= 2
                    and _lm_vis(list(coords)) >= vis_floor
                    and isinstance(coords[0], (int, float))
                    and isinstance(coords[1], (int, float))
                    and math.isfinite(float(coords[0]))
                    and math.isfinite(float(coords[1]))
                ):
                    xs.append(float(coords[0]))
                    ys.append(float(coords[1]))
                    vis_list.append(_lm_vis(list(coords)))
                else:
                    xs.append(None)
                    ys.append(None)
                    vis_list.append(None)

            sm_x = _zero_phase_ema_optional(timestamps, xs, tau_s=lm_tau_s, max_gap_s=max_gap_s)
            sm_y = _zero_phase_ema_optional(timestamps, ys, tau_s=lm_tau_s, max_gap_s=max_gap_s)
            for idx, fo in enumerate(smoothed):
                if sm_x[idx] is None or sm_y[idx] is None or not isinstance(fo, dict):
                    continue
                was_missing = xs[idx] is None
                d_data = fo.get(dancer_key)
                if not isinstance(d_data, dict):
                    d_data = {
                        "landmarks": {},
                        "interpolated_display": True,
                        "quality_status": "unavailable",
                        "pose_origin": "display_interpolated",
                        "landmark_origins": {},
                    }
                    fo[dancer_key] = d_data
                elif was_missing:
                    d_data["interpolated_display"] = True
                lms = d_data.setdefault("landmarks", {})
                lm_origins = d_data.setdefault("landmark_origins", {})
                if isinstance(lms, dict):
                    vis = vis_list[idx] if vis_list[idx] is not None else 0.75
                    lms[lm_key] = [round(sm_x[idx], 5), round(sm_y[idx], 5), round(vis, 3)]  # type: ignore[arg-type]
                    if was_missing and isinstance(lm_origins, dict):
                        lm_origins[lm_key] = "display_interpolated"

    # Populate missing pair midpoint endpoints from smoothed dancer landmarks, then smooth pair points
    def _dancer_mid_norm(d_data: Any, k1: str, k2: str) -> tuple[float, float] | None:
        if not isinstance(d_data, dict):
            return None
        lms = d_data.get("landmarks") or {}
        c1, c2 = lms.get(k1), lms.get(k2)
        if (
            isinstance(c1, (list, tuple))
            and isinstance(c2, (list, tuple))
            and len(c1) >= 2
            and len(c2) >= 2
            and _lm_vis(list(c1)) >= 0.5
            and _lm_vis(list(c2)) >= 0.5
        ):
            return (0.5 * (float(c1[0]) + float(c2[0])), 0.5 * (float(c1[1]) + float(c2[1])))
        return None

    for pair_key, k1, k2 in (
        ("pair_hip_norm", "left_hip", "right_hip"),
        ("pair_shoulder_norm", "left_shoulder", "right_shoulder"),
    ):
        pxs: list[float | None] = []
        pys: list[float | None] = []
        for fo in smoothed:
            pt = fo.get(pair_key) if isinstance(fo, dict) else None
            if (
                isinstance(pt, (list, tuple))
                and len(pt) >= 2
                and isinstance(pt[0], (int, float))
                and isinstance(pt[1], (int, float))
                and math.isfinite(float(pt[0]))
                and math.isfinite(float(pt[1]))
            ):
                pxs.append(float(pt[0]))
                pys.append(float(pt[1]))
            else:
                ma = _dancer_mid_norm(fo.get("dancer_a") if isinstance(fo, dict) else None, k1, k2)
                mb = _dancer_mid_norm(fo.get("dancer_b") if isinstance(fo, dict) else None, k1, k2)
                if ma is not None and mb is not None:
                    pxs.append(0.5 * (ma[0] + mb[0]))
                    pys.append(0.5 * (ma[1] + mb[1]))
                else:
                    pxs.append(None)
                    pys.append(None)
        sm_px = _zero_phase_ema_optional(timestamps, pxs, tau_s=tau_s, max_gap_s=max_gap_s)
        sm_py = _zero_phase_ema_optional(timestamps, pys, tau_s=tau_s, max_gap_s=max_gap_s)
        for idx, fo in enumerate(smoothed):
            if isinstance(fo, dict) and sm_px[idx] is not None and sm_py[idx] is not None:
                fo[pair_key] = [round(sm_px[idx], 6), round(sm_py[idx], 6)]  # type: ignore[arg-type]

    for angle_key in (
        "tilt_a_deg",
        "tilt_b_deg",
        "couple_midpoint_tilt_deg",
        "torso_separation_deg",
    ):
        raw_vals: list[float | None] = [
            float(fo[angle_key])
            if isinstance(fo, dict)
            and isinstance(fo.get(angle_key), (int, float))
            and not isinstance(fo.get(angle_key), bool)
            and math.isfinite(float(fo[angle_key]))
            else None
            for fo in smoothed
        ]
        sm_vals = _zero_phase_ema_optional(timestamps, raw_vals, tau_s=tau_s, max_gap_s=max_gap_s)
        for idx in range(n):
            val = sm_vals[idx]
            if isinstance(smoothed[idx], dict):
                smoothed[idx][angle_key] = round(val, 2) if val is not None else None

    return smoothed
