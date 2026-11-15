"""Overlay rendering for tracked Couple axes, display-interpolated gaps, steps, and turns."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# BGR palette matching Qt's Format_BGR888 video frames
COLOR_DANCER_A_BGR = (11, 158, 245)      # Warm amber
COLOR_DANCER_B_BGR = (166, 184, 20)      # Teal/cyan
COLOR_COUPLE_MID_BGR = (252, 132, 192)   # Soft violet
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


def _draw_dancer(
    image: np.ndarray,
    dancer_data: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
    *,
    show_steps: bool,
) -> None:
    height, width = image.shape[:2]
    landmarks: dict[str, list[float]] = dancer_data.get("landmarks") or {}
    if not landmarks:
        return
    interpolated = bool(dancer_data.get("interpolated_display", False))

    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    lh = landmarks.get("left_hip")
    rh = landmarks.get("right_hip")

    hip_px: tuple[int, int] | None = None
    sh_px: tuple[int, int] | None = None
    if lh and rh:
        hip_px = _to_px(0.5 * (lh[0] + rh[0]), 0.5 * (lh[1] + rh[1]), width, height)
    if ls and rs:
        sh_px = _to_px(0.5 * (ls[0] + rs[0]), 0.5 * (ls[1] + rs[1]), width, height)

    if hip_px and sh_px:
        if interpolated:
            _draw_dashed_line(image, hip_px, sh_px, color, thickness=3)
            cv2.circle(image, hip_px, 6, color, 2, cv2.LINE_AA)
            cv2.circle(image, sh_px, 6, color, 2, cv2.LINE_AA)
        else:
            cv2.line(image, hip_px, sh_px, color, 3, cv2.LINE_AA)
            cv2.circle(image, hip_px, 5, color, -1, cv2.LINE_AA)
            cv2.circle(image, sh_px, 5, color, -1, cv2.LINE_AA)

        anchor = (max(8, min(sh_px[0] + 8, width - 120)), max(22, sh_px[1] - 10))
        tag = f"{label} (est)" if interpolated else label
        cv2.putText(
            image, tag, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA
        )
    elif hip_px:
        cv2.circle(image, hip_px, 6, color, 2, cv2.LINE_AA)
        anchor = (max(8, min(hip_px[0] + 8, width - 120)), max(22, hip_px[1] - 10))
        cv2.putText(
            image, f"{label} (hip)", anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA
        )

    if show_steps:
        for side_key in ("left_ankle", "right_ankle"):
            ank = landmarks.get(side_key)
            if ank:
                ap = _to_px(ank[0], ank[1], width, height)
                cv2.circle(image, ap, 4, color, 1 if interpolated else -1, cv2.LINE_AA)


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
) -> np.ndarray:
    """Draw tracked Dancer A/B torso axes, couple midpoint axis, and active events onto ``frame_bgr``."""
    out = frame_bgr.copy()
    if overlay_data is None:
        return out
    height, width = out.shape[:2]

    if show_dancer_a and overlay_data.get("dancer_a"):
        _draw_dancer(
            out,
            overlay_data["dancer_a"],
            COLOR_DANCER_A_BGR,
            dancer_a_name or "A",
            show_steps=show_steps_turns,
        )
    if show_dancer_b and overlay_data.get("dancer_b"):
        _draw_dancer(
            out,
            overlay_data["dancer_b"],
            COLOR_DANCER_B_BGR,
            dancer_b_name or "B",
            show_steps=show_steps_turns,
        )

    if show_couple_axis:
        pair_hip = overlay_data.get("pair_hip_norm")
        pair_sh = overlay_data.get("pair_shoulder_norm")
        hip_only = overlay_data.get("hip_only_travel_point")
        if pair_hip and pair_sh:
            hp = _to_px(pair_hip[0], pair_hip[1], width, height)
            sp = _to_px(pair_sh[0], pair_sh[1], width, height)
            cv2.line(out, hp, sp, COLOR_COUPLE_MID_BGR, 3, cv2.LINE_AA)
            cv2.circle(out, hp, 5, COLOR_COUPLE_MID_BGR, -1, cv2.LINE_AA)
            cv2.circle(out, sp, 5, COLOR_COUPLE_MID_BGR, -1, cv2.LINE_AA)
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

    if show_steps_turns and active_events:
        y_cursor = 28
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
    return out
