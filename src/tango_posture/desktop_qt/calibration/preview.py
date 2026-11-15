"""Render approved calibration feedback on cached reference frames."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from ...calibration.models import ApprovedRange
from ...config import Config
from ...domain import FrameObservation, Landmark
from ...geometry import (
    axis_curve_alignment,
    head_horizontal_alignment,
    knee_bend_alignment,
    knee_geometry_kwargs,
    shoulder_geometry_kwargs,
    shoulder_tension_alignment,
    torso_axis_alignment,
)
from ...overlay import TORSO_AXIS_EXTENSION, screen_point
from ...quality import reliable

COLORS = {
    "WITHIN_REFERENCE": (80, 220, 80),
    "PENDING": (0, 190, 255),
    "DEVIATION": (0, 0, 255),
    "UNKNOWN": (180, 180, 180),
}
LABELS = {
    "head_pitch": "Head pitch",
    "head_forward": "Head forward",
    "head_inclination": "Head inclination",
    "shoulder_clearance": "Shoulder clearance",
    "hip_offset": "Hip offset",
    "torso_axis_tilt": "Torso axis",
    "trunk_inclination": "Trunk inclination",
    "knee_flexion": "Knee flexion",
}


def render_approved_feedback(
    image: np.ndarray,
    frame: FrameObservation,
    cfg: Config,
    targets: dict[str, ApprovedRange],
    rows: list[dict[str, Any]],
) -> np.ndarray:
    """Draw only approved, currently measurable geometry and timed feedback."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    pose = (
        {name: point for name, point in frame.poses[0].items() if reliable(point, cfg.confidence)}
        if len(frame.poses) == 1 else {}
    )

    def point(landmark: Landmark) -> tuple[int, int]:
        return screen_point(landmark.x, landmark.y, width, height, False)

    def line(a: Landmark, b: Landmark, color: tuple[int, int, int], thickness: int = 3) -> None:
        cv2.line(canvas, point(a), point(b), color, thickness, cv2.LINE_AA)

    def core_points(view: str) -> tuple[Landmark | None, Landmark | None, Landmark | None]:
        def midpoint(a: Landmark, b: Landmark) -> Landmark:
            return Landmark((a.x + b.x) / 2, (a.y + b.y) / 2)

        def preferred(joint: str) -> Landmark | None:
            return pose.get(f"{cfg.side}_{joint}") or pose.get(
                f"{'right' if cfg.side == 'left' else 'left'}_{joint}"
            )

        if view == "side":
            return (
                pose.get(f"{cfg.side}_ear"), pose.get(f"{cfg.side}_shoulder"),
                pose.get(f"{cfg.side}_hip"),
            )
        shoulders = [pose.get(f"{side}_shoulder") for side in ("left", "right")]
        hips = [pose.get(f"{side}_hip") for side in ("left", "right")]
        ears = [pose.get(f"{side}_ear") for side in ("left", "right")]
        if all(value is not None for value in (*shoulders, *hips)):
            ls, rs = shoulders
            lh, rh = hips
            assert ls is not None and rs is not None and lh is not None and rh is not None
            shoulder, hip = midpoint(ls, rs), midpoint(lh, rh)
            length = math.hypot((shoulder.x - hip.x) * frame.width, (shoulder.y - hip.y) * frame.height)
            span = math.hypot((ls.x - rs.x) * frame.width, (ls.y - rs.y) * frame.height)
            if length > 0 and span / length > 0.25 and all(ear is not None for ear in ears):
                left, right = ears
                assert left is not None and right is not None
                return midpoint(left, right), shoulder, hip
            return preferred("ear"), shoulder, hip
        return preferred("ear"), preferred("shoulder"), preferred("hip")

    for row in rows:
        key = row["metric"]
        if key not in targets or row["state"] == "UNKNOWN" or not pose:
            continue
        color = COLORS[row["state"]]
        if key == "head_pitch":
            head = head_horizontal_alignment(pose, frame.width, frame.height)
            if head is not None:
                angle, ear, nose = head
                ex, ey = point(ear)
                nx, ny = point(nose)
                direction = 1 if nx >= ex else -1
                length = max(40.0, math.hypot(nx - ex, ny - ey))
                cv2.line(canvas, (ex, ey), (round(ex + direction * length), ey),
                         (255, 220, 0), 2, cv2.LINE_AA)
                rad = math.radians(angle)
                cv2.line(canvas, (ex, ey),
                         (round(ex + direction * length * math.cos(rad)),
                          round(ey - length * math.sin(rad))), color, 3, cv2.LINE_AA)
        elif key == "shoulder_clearance":
            shoulder_info = shoulder_tension_alignment(
                pose, frame.width, frame.height, **shoulder_geometry_kwargs(cfg)
            )
            if shoulder_info is not None:
                _, _, shoulder_ear, joint = shoulder_info
                line(shoulder_ear, joint, color)
                x, y = point(joint)
                cv2.line(canvas, (x - 18, y), (x + 18, y), color, 3, cv2.LINE_AA)
        elif key == "hip_offset":
            hip_axis = axis_curve_alignment(pose, frame.width, frame.height, cfg.facing, cfg.side)
            if hip_axis is not None:
                _, _, back_head, hip_joint, ankle_joint, projection = hip_axis
                line(ankle_joint, back_head, (255, 220, 0), 2)
                line(projection, hip_joint, color)
                cv2.circle(canvas, point(hip_joint), 8, color, 2, cv2.LINE_AA)
        elif key == "torso_axis_tilt":
            torso_axis = torso_axis_alignment(pose, frame.width, frame.height, cfg.confidence)
            if torso_axis is not None:
                _, torso_shoulder, torso_hip = torso_axis
                hx, hy = point(torso_hip)
                sx, sy = point(torso_shoulder)
                length = math.hypot(sx - hx, sy - hy)
                cv2.line(canvas, (hx, max(0, round(hy - length))), (hx, hy),
                         (255, 220, 0), 2, cv2.LINE_AA)
                dx, dy = sx - hx, sy - hy
                axis_start = (round(hx - TORSO_AXIS_EXTENSION * dx),
                              round(hy - TORSO_AXIS_EXTENSION * dy))
                axis_end = (round(sx + TORSO_AXIS_EXTENSION * dx),
                            round(sy + TORSO_AXIS_EXTENSION * dy))
                cv2.line(canvas, axis_start, axis_end, color, 5, cv2.LINE_AA)
                cv2.line(canvas, axis_start, axis_end, (255, 140, 255), 2, cv2.LINE_AA)
        elif key == "knee_flexion":
            candidates = []
            for side in ("left", "right"):
                leg = {name: value for name, value in pose.items() if name in {
                    f"{side}_hip", f"{side}_knee", f"{side}_ankle",
                }}
                knee_info = knee_bend_alignment(
                    leg, frame.width, frame.height, side, **knee_geometry_kwargs(cfg)
                )
                if knee_info is not None and row.get("raw") is not None:
                    candidates.append((abs(knee_info[0] - row["raw"]), knee_info[1][0]))
            if candidates:
                _, (leg_hip, leg_knee, leg_ankle) = min(candidates, key=lambda item: item[0])
                line(leg_hip, leg_knee, color)
                line(leg_knee, leg_ankle, color)
                cv2.circle(canvas, point(leg_knee), 9, color, 2, cv2.LINE_AA)
        elif key in ("head_forward", "head_inclination", "trunk_inclination"):
            core_ear, core_shoulder, core_hip = core_points(targets[key].view)
            start = core_hip if key == "trunk_inclination" else core_ear
            if start is not None and core_shoulder is not None:
                line(start, core_shoulder, color)

    if rows:
        labels = []
        for row in rows:
            key = row["metric"]
            state = row["state"]
            value = row.get("smoothed")
            measured = "unavailable" if value is None else f"{value:.2f} {targets[key].unit}"
            status = {
                "WITHIN_REFERENCE": "IN RANGE",
                "PENDING": "PENDING",
                "DEVIATION": "ALERT",
                "UNKNOWN": "UNKNOWN",
            }[state]
            labels.append((f"{LABELS[key]}: {measured} [{status}]", COLORS[state]))
        longest = max(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
                      for label, _ in labels)
        scale = min(0.45, 0.45 * max(1, width * 0.4 - 10) / max(1, longest),
                    0.45 * max(1, height - 4) / (20 * len(labels)))
        line_height = max(1, round(20 * scale / 0.45))
        panel_width = min(width - 1, round(longest * scale / 0.45) + 10)
        cv2.rectangle(canvas, (0, 0), (panel_width, min(height - 1, line_height * len(labels) + 4)),
                      (25, 25, 25), -1)
        for index, (label, color) in enumerate(labels):
            cv2.putText(canvas, label, (4, min(height - 1, line_height * (index + 1))),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return canvas
