"""Qt-independent state machine, readiness gates, and guided plans for solo camera calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

from ..config import Config
from ..domain import FrameObservation, Landmark, MetricObservation
from ..quality import reliable
from .measurements import measure_ranges, view_eligible
from .models import METRIC_DETAILS, Passage
from .statistics import repeat_check, summarize_examples

SessionState = Literal[
    "Preview",
    "Prompt",
    "Wait for view and pose",
    "Settle",
    "Capture",
    "Evaluate",
    "Wait for take",
    "Capture take",
    "Review",
    "Cancelled",
    "Camera error",
]

GuideView = Literal["front", "side", "uncertain"]
MetricOutcomeStatus = Literal["Ready for review", "Needs retry", "Unavailable"]


@dataclass(frozen=True)
class GuidePhaseSpec:
    """Definition of a single capture phase in a guided calibration plan."""

    phase_id: str
    plan_kind: Literal["quick", "check", "walking"]
    view: Literal["front", "side"]
    repetition: int
    instruction: str
    role: Literal["neutral", "changed", "walking"] = "neutral"
    check_id: str | None = None
    metrics: tuple[str, ...] = ()
    requires_stillness: bool = True
    turn_direction: Literal["camera", "left", "right"] | None = None
    hold_window_s: float | None = None
    min_usable_hold_s: float | None = None


@dataclass(frozen=True)
class CheckSpec:
    """Definition of a guided individual measurement check."""

    check_id: str
    title: str
    view: Literal["front", "side"]
    metrics: tuple[str, ...]
    changed_instruction: str
    neutral_instruction: str
    expected_sign: int | None = None  # -1 = decrease expected, +1 = increase expected, None = consistent direction


CHECK_SPECS: dict[str, CheckSpec] = {
    "head_pitch": CheckSpec(
        check_id="head_pitch",
        title="Head pitch · side",
        view="side",
        metrics=("head_pitch",),
        neutral_instruction="Turn left or right so you are side-on to the camera. Take up your usual tango stance, looking ahead.",
        changed_instruction="Lower your gaze a little. Keep your chest facing the same way.",
        expected_sign=-1,
    ),
    "head_position": CheckSpec(
        check_id="head_position",
        title="Head position · side",
        view="side",
        metrics=("head_forward", "head_inclination"),
        neutral_instruction="Turn left or right so you are side-on to the camera. Take up your usual tango stance with your head balanced.",
        changed_instruction="Bring your head slightly forward while keeping your gaze level. Return to your usual position when prompted.",
        expected_sign=None,
    ),
    "shoulder_clearance": CheckSpec(
        check_id="shoulder_clearance",
        title="Shoulders · front",
        view="front",
        metrics=("shoulder_clearance",),
        neutral_instruction="Turn to the camera with both shoulders relaxed and settled.",
        changed_instruction="Raise both shoulders gently. Keep looking ahead. Let both shoulders settle for the next hold.",
        expected_sign=-1,
    ),
    "torso_axis_tilt": CheckSpec(
        check_id="torso_axis_tilt",
        title="Torso axis · front",
        view="front",
        metrics=("torso_axis_tilt",),
        neutral_instruction="Turn to the camera in your upright ready-to-walk stance.",
        changed_instruction="Let your torso incline a little toward the same side each time. Then return to your usual stance.",
        expected_sign=1,
    ),
    "trunk_inclination": CheckSpec(
        check_id="trunk_inclination",
        title="Trunk inclination · side",
        view="side",
        metrics=("trunk_inclination",),
        neutral_instruction="Turn left or right so you are side-on to the camera. Take up your usual tango stance.",
        changed_instruction="Incline your trunk forward a little, then return to your usual stance.",
        expected_sign=None,
    ),
    "hip_offset": CheckSpec(
        check_id="hip_offset",
        title="Hip position · side",
        view="side",
        metrics=("hip_offset",),
        neutral_instruction="Turn left or right so you are side-on to the camera. Take up your usual tango stance with ankles visible.",
        changed_instruction="If comfortable, shift your hips forward slightly, then return.",
        expected_sign=None,
    ),
    "knee_flexion": CheckSpec(
        check_id="knee_flexion",
        title="Knee flexion · side",
        view="side",
        metrics=("knee_flexion",),
        neutral_instruction="Turn left or right so you are side-on to the camera. Stand in your usual soft-knee tango stance.",
        changed_instruction="Gently straighten your knees a little, keeping them comfortable.",
        expected_sign=-1,
    ),
}


def build_quick_setup_phases() -> list[GuidePhaseSpec]:
    """Return the default four-hold Quick Setup plan."""
    front_metrics = ("shoulder_clearance", "torso_axis_tilt")
    side_metrics = (
        "head_pitch",
        "head_forward",
        "head_inclination",
        "trunk_inclination",
        "hip_offset",
        "knee_flexion",
    )
    return [
        GuidePhaseSpec(
            phase_id="front_a",
            plan_kind="quick",
            view="front",
            repetition=1,
            instruction="Turn to the camera in your ready-to-walk stance.",
            role="neutral",
            metrics=front_metrics,
            turn_direction="camera",
        ),
        GuidePhaseSpec(
            phase_id="front_b",
            plan_kind="quick",
            view="front",
            repetition=2,
            instruction="Relax, then turn to the camera in the same stance.",
            role="neutral",
            metrics=front_metrics,
            turn_direction="camera",
        ),
        GuidePhaseSpec(
            phase_id="side_a",
            plan_kind="quick",
            view="side",
            repetition=1,
            instruction="Turn left so you are side-on to the camera.",
            role="neutral",
            metrics=side_metrics,
            turn_direction="left",
            hold_window_s=1.5,
            min_usable_hold_s=1.0,
        ),
        GuidePhaseSpec(
            phase_id="side_b",
            plan_kind="quick",
            view="side",
            repetition=2,
            instruction="Turn right so you are side-on to the camera.",
            role="neutral",
            metrics=side_metrics,
            turn_direction="right",
            hold_window_s=1.5,
            min_usable_hold_s=1.0,
        ),
    ]


def build_check_phases(check_ids: list[str] | tuple[str, ...]) -> list[GuidePhaseSpec]:
    """Build 4-phase neutral A -> changed A -> neutral B -> changed B sequences for selected checks."""
    phases: list[GuidePhaseSpec] = []
    for cid in check_ids:
        spec = CHECK_SPECS[cid]
        td: Literal["camera", "left", "right"] | None = "camera" if spec.view == "front" else None
        phases.extend([
            GuidePhaseSpec(
                phase_id=f"{cid}_neutral_a",
                plan_kind="check",
                view=spec.view,
                repetition=1,
                instruction=spec.neutral_instruction,
                role="neutral",
                check_id=cid,
                metrics=spec.metrics,
                turn_direction=td,
            ),
            GuidePhaseSpec(
                phase_id=f"{cid}_changed_a",
                plan_kind="check",
                view=spec.view,
                repetition=1,
                instruction=spec.changed_instruction,
                role="changed",
                check_id=cid,
                metrics=spec.metrics,
                turn_direction=td,
            ),
            GuidePhaseSpec(
                phase_id=f"{cid}_neutral_b",
                plan_kind="check",
                view=spec.view,
                repetition=2,
                instruction=f"Return to neutral: {spec.neutral_instruction}",
                role="neutral",
                check_id=cid,
                metrics=spec.metrics,
                turn_direction=td,
            ),
            GuidePhaseSpec(
                phase_id=f"{cid}_changed_b",
                plan_kind="check",
                view=spec.view,
                repetition=2,
                instruction=f"Repeat: {spec.changed_instruction}",
                role="changed",
                check_id=cid,
                metrics=spec.metrics,
                turn_direction=td,
            ),
        ])
    return phases


def build_walking_phases() -> list[GuidePhaseSpec]:
    """Build the two-pass Solo Walking plan."""
    walking_metrics = tuple(METRIC_DETAILS.keys())
    return [
        GuidePhaseSpec(
            phase_id="walk_1",
            plan_kind="walking",
            view="side",
            repetition=1,
            instruction="Turn left so you are side-on to the camera, then take up to four natural tango steps across the frame, keeping your shoes visible. Stop at the end.",
            role="walking",
            metrics=walking_metrics,
            requires_stillness=False,
            turn_direction="left",
        ),
        GuidePhaseSpec(
            phase_id="walk_2",
            plan_kind="walking",
            view="side",
            repetition=2,
            instruction="Turn right so you are side-on to the camera, then take up to four natural tango steps across the frame, keeping your shoes visible. Stop at the end.",
            role="walking",
            metrics=walking_metrics,
            requires_stillness=False,
            turn_direction="right",
        ),
    ]


@dataclass(frozen=True)
class AcceptedPassageRecord:
    """Ledger entry for a single accepted capture phase."""

    phase_id: str
    plan_kind: Literal["quick", "check", "walking"]
    view: Literal["front", "side"]
    repetition: int
    role: Literal["neutral", "changed", "walking"]
    passage: Passage
    check_id: str | None = None
    eligible_metrics: dict[str, float] = field(default_factory=dict)  # metric -> usable seconds
    unavailable_reasons: dict[str, str] = field(default_factory=dict)
    usable: bool = True
    detected_side: Literal["left", "right"] | None = None
    detected_facing: Literal["left", "right"] | None = None


@dataclass(frozen=True)
class GateEvaluation:
    """Per-frame evaluation of dancer tracking, view, framing, and metric eligibility."""

    dancer_status: Literal["ok", "no_dancer", "multiple_dancers", "identity_lost"]
    detected_view: GuideView
    view_matches: bool
    framing_ok: bool
    hint: str
    torso_midpoint: tuple[float, float] | None
    torso_length_px: float | None
    observation: MetricObservation | None


def _in_frame(point: Landmark) -> bool:
    return 0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0


def _pose_torso_geometry(
    pose: dict[str, Landmark], width: int, height: int, confidence: float,
) -> tuple[tuple[float, float], float] | None:
    shoulders = [
        p
        for k in ("left_shoulder", "right_shoulder")
        if (p := pose.get(k)) is not None and reliable(p, confidence) and _in_frame(p)
    ]
    hips = [
        p
        for k in ("left_hip", "right_hip")
        if (p := pose.get(k)) is not None and reliable(p, confidence) and _in_frame(p)
    ]
    if not shoulders or not hips:
        return None
    smx = sum(p.x for p in shoulders) / len(shoulders) * width
    smy = sum(p.y for p in shoulders) / len(shoulders) * height
    hmx = sum(p.x for p in hips) / len(hips) * width
    hmy = sum(p.y for p in hips) / len(hips) * height
    torso_px = math.hypot(smx - hmx, smy - hmy)
    if torso_px <= 1e-6:
        return None
    mid = ((smx + hmx) * 0.5, (smy + hmy) * 0.5)
    return mid, torso_px


def _detect_side_orientation(
    pose: dict[str, Landmark],
    cfg: Config,
) -> tuple[Literal["left", "right"], Literal["left", "right"], Landmark] | None:
    """Return (detected_side, detected_facing, chosen_ear) from reliable profile landmarks."""
    head_conf = min(cfg.confidence, 0.45)
    nose = pose.get("nose")
    if nose is None or not reliable(nose, head_conf) or not _in_frame(nose):
        return None
    candidates: list[tuple[Literal["left", "right"], Landmark]] = []
    for side in ("left", "right"):
        pt = pose.get(f"{side}_ear")
        if pt is not None and reliable(pt, head_conf) and _in_frame(pt):
            candidates.append((side, pt))  # type: ignore[arg-type]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            abs(nose.x - item[1].x) >= 0.010,
            item[1].visibility if item[1].visibility is not None else 0.0,
            item[0] == cfg.side,
        ),
        reverse=True,
    )
    ear_side, ear = candidates[0]
    dx = nose.x - ear.x
    detected_facing: Literal["left", "right"] = "right" if dx > 0 else "left"
    other_side: Literal["left", "right"] = "right" if ear_side == "left" else "left"

    def _side_torso_ok(s: Literal["left", "right"]) -> bool:
        sh = pose.get(f"{s}_shoulder")
        hip = pose.get(f"{s}_hip")
        return (
            sh is not None and reliable(sh, cfg.confidence) and _in_frame(sh)
            and hip is not None and reliable(hip, cfg.confidence) and _in_frame(hip)
        )

    if _side_torso_ok(ear_side):
        detected_side: Literal["left", "right"] = ear_side
    elif _side_torso_ok(other_side):
        detected_side = other_side
    else:
        detected_side = ear_side
    return detected_side, detected_facing, ear


def prepare_side_measurement_frame(frame: FrameObservation, cfg: Config) -> FrameObservation:
    """Repair occluded far-side torso joints in side view so near-side measurements stay valid."""
    if len(frame.poses) != 1:
        return frame
    pose = frame.poses[0]
    near_side = cfg.side
    far_side: Literal["left", "right"] = "right" if near_side == "left" else "left"
    near_sh = pose.get(f"{near_side}_shoulder")
    near_hip = pose.get(f"{near_side}_hip")
    if (
        near_sh is None or not reliable(near_sh, cfg.confidence) or not _in_frame(near_sh)
        or near_hip is None or not reliable(near_hip, cfg.confidence) or not _in_frame(near_hip)
    ):
        return frame

    torso_px = math.hypot(
        (near_sh.x - near_hip.x) * frame.width,
        (near_sh.y - near_hip.y) * frame.height,
    )
    updated_pose: dict[str, Landmark] | None = None
    for joint, near_pt in (("shoulder", near_sh), ("hip", near_hip)):
        far_key = f"{far_side}_{joint}"
        far_pt = pose.get(far_key)
        far_within_profile = (
            far_pt is not None
            and math.isfinite(far_pt.x)
            and math.isfinite(far_pt.y)
            and _in_frame(far_pt)
            and torso_px > 1e-6
            and math.hypot((far_pt.x - near_pt.x) * frame.width, (far_pt.y - near_pt.y) * frame.height)
            <= cfg.max_view_width * torso_px
        )
        if far_pt is not None and reliable(far_pt, cfg.confidence) and _in_frame(far_pt) and far_within_profile:
            continue
        if updated_pose is None:
            updated_pose = dict(pose)
        if far_within_profile and far_pt is not None:
            updated_pose[far_key] = Landmark(
                x=far_pt.x,
                y=far_pt.y,
                visibility=max(far_pt.visibility or 0.0, cfg.confidence),
                presence=max(far_pt.presence or 0.0, cfg.confidence),
                z=far_pt.z,
            )
        else:
            updated_pose[far_key] = near_pt

    head_floor = min(cfg.confidence, 0.45)
    for lm_key in ("nose", f"{near_side}_ear"):
        pt = pose.get(lm_key)
        if (
            pt is not None
            and _in_frame(pt)
            and reliable(pt, head_floor)
            and not reliable(pt, cfg.confidence)
        ):
            if updated_pose is None:
                updated_pose = dict(pose)
            updated_pose[lm_key] = Landmark(
                x=pt.x,
                y=pt.y,
                visibility=max(pt.visibility or 0.0, cfg.confidence),
                presence=max(pt.presence or 0.0, cfg.confidence),
                z=pt.z,
            )
    near_ankle_key = f"{near_side}_ankle"
    far_ankle_key = f"{far_side}_ankle"
    near_ankle = pose.get(near_ankle_key)
    if near_ankle is not None and _in_frame(near_ankle) and reliable(near_ankle, head_floor):
        ankle_key, ankle_pt = near_ankle_key, near_ankle
    else:
        far_ankle = pose.get(far_ankle_key)
        if far_ankle is not None and _in_frame(far_ankle) and reliable(far_ankle, head_floor):
            ankle_key, ankle_pt = far_ankle_key, far_ankle
        else:
            ankle_key, ankle_pt = "", None
    if ankle_pt is not None and not reliable(ankle_pt, cfg.confidence):
        if updated_pose is None:
            updated_pose = dict(pose)
        updated_pose[ankle_key] = Landmark(
            x=ankle_pt.x,
            y=ankle_pt.y,
            visibility=max(ankle_pt.visibility or 0.0, cfg.confidence),
            presence=max(ankle_pt.presence or 0.0, cfg.confidence),
            z=ankle_pt.z,
        )

    if updated_pose is None:
        return frame
    return FrameObservation(
        timestamp=frame.timestamp,
        width=frame.width,
        height=frame.height,
        poses=[updated_pose],
        world_poses=list(frame.world_poses),
    )


def _turn_hint(
    requested_view: Literal["front", "side"] | None,
    turn_direction: Literal["camera", "left", "right"] | None,
) -> str:
    if requested_view == "front" or turn_direction == "camera":
        return "Turn to the camera with your head and shoulders level."
    if turn_direction == "left":
        return "Turn left so you are side-on to the camera."
    if turn_direction == "right":
        return "Turn right so you are side-on to the camera."
    if requested_view == "side":
        return "Turn left or right so you are side-on to the camera."
    return "Turn to the camera, or turn left or right so you are side-on."


def classify_guide_view(
    frame: FrameObservation,
    cfg: Config,
    requested_view: Literal["front", "side"] | None = None,
    *,
    strict_facing: bool = True,
    turn_direction: Literal["camera", "left", "right"] | None = None,
) -> tuple[GuideView, str]:
    """Classify the single-dancer view for guided capture using strict facial and torso cues."""
    if len(frame.poses) != 1:
        return "uncertain", "Ensure exactly one dancer is visible."
    pose = frame.poses[0]
    conf = cfg.confidence
    geom = _pose_torso_geometry(pose, frame.width, frame.height, conf)
    if geom is None:
        return "uncertain", "Step back so shoulders and hips are clearly visible."
    _, torso_px = geom

    ls, rs = pose.get("left_shoulder"), pose.get("right_shoulder")
    lh, rh = pose.get("left_hip"), pose.get("right_hip")
    nose = pose.get("nose")
    le = pose.get("left_ear")
    re = pose.get("right_ear")

    obs = measure_ranges(frame, cfg)
    vw = obs.view_width
    if vw is None or not math.isfinite(vw):
        sh_span = (
            math.hypot((ls.x - rs.x) * frame.width, (ls.y - rs.y) * frame.height)
            if ls is not None and rs is not None and reliable(ls, conf) and reliable(rs, conf) and _in_frame(ls) and _in_frame(rs)
            else 0.0
        )
        hip_span = (
            math.hypot((lh.x - rh.x) * frame.width, (lh.y - rh.y) * frame.height)
            if lh is not None and rh is not None and reliable(lh, conf) and reliable(rh, conf) and _in_frame(lh) and _in_frame(rh)
            else 0.0
        )
        vw = max(sh_span, hip_span) / torso_px

    front_cutoff = cfg.max_view_width + cfg.solo_camera.front_view_width_margin

    # Front check: width above front_cutoff + front anatomical order + centered nose
    if vw > front_cutoff:
        all_torso = all(
            pt is not None and reliable(pt, conf) and _in_frame(pt)
            for pt in (ls, rs, lh, rh)
        )
        nose_ok = nose is not None and reliable(nose, conf) and _in_frame(nose)
        if all_torso and nose_ok:
            assert ls and rs and lh and rh and nose
            anatomical_front = ls.x > rs.x and lh.x > rh.x
            if anatomical_front:
                ear_conf = min(conf, 0.40)
                if (
                    le is not None and reliable(le, ear_conf) and _in_frame(le)
                    and re is not None and reliable(re, ear_conf) and _in_frame(re)
                ):
                    centered = min(le.x, re.x) <= nose.x <= max(le.x, re.x)
                else:
                    centered = min(ls.x, rs.x) <= nose.x <= max(ls.x, rs.x)
                if centered:
                    hint = "" if requested_view in (None, "front") else _turn_hint(requested_view, turn_direction)
                    return "front", hint
        if requested_view == "side":
            return "uncertain", _turn_hint(requested_view, turn_direction)
        return "uncertain", "Turn to the camera with your head and shoulders level."

    # Side check: width <= max_view_width (or buffer zone with clear profile head) + ear-to-nose horizontal sign
    orient = _detect_side_orientation(pose, cfg)
    allow_buffer_side = (
        requested_view == "side"
        and vw <= front_cutoff
        and orient is not None
    )
    if vw <= cfg.max_view_width or allow_buffer_side:
        if orient is None:
            return "uncertain", "Keep your profile head (nose and ear) visible in the frame."
        _, _, ear = orient
        assert nose is not None
        dx = nose.x - ear.x
        if abs(dx) < 0.010:
            return "uncertain", _turn_hint(requested_view or "side", turn_direction)
        if requested_view == "side" and turn_direction in ("left", "right"):
            # Turning to your own left faces the right side of the unmirrored image, and vice versa.
            expected_sign = 1.0 if turn_direction == "left" else -1.0
            if dx * expected_sign <= 0:
                return "uncertain", _turn_hint(requested_view, turn_direction)
        elif strict_facing:
            expected_sign = 1.0 if cfg.facing == "right" else -1.0
            if dx * expected_sign <= 0:
                turn_word = "left" if cfg.facing == "right" else "right"
                return "uncertain", f"Turn {turn_word} so you are side-on to the camera (Face {cfg.facing})."
        hint = "" if requested_view in (None, "side") else _turn_hint(requested_view, turn_direction)
        return "side", hint

    # Ambiguous buffer zone between max_view_width and front_cutoff
    return "uncertain", _turn_hint(requested_view, turn_direction)


def check_full_body_framing(frame: FrameObservation, cfg: Config) -> tuple[bool, str]:
    """Require one tracked dancer, sufficient torso size, and reliable head + ankle/shoe landmarks."""
    if len(frame.poses) == 0:
        return False, "No dancer detected. Step into view."
    if len(frame.poses) > 1:
        return False, "More than one person or reflection detected. Clear the scene."
    pose = frame.poses[0]
    geom = _pose_torso_geometry(pose, frame.width, frame.height, cfg.confidence)
    if geom is None:
        return False, "Step back until shoulders and hips fit inside the frame."
    _, torso_px = geom
    if torso_px / max(1, frame.height) < cfg.min_torso:
        return False, "Step closer so your torso is large enough to track."
    nose = pose.get("nose")
    if nose is None or not reliable(nose, cfg.confidence) or not _in_frame(nose):
        return False, "Adjust camera height or step back so your head is fully visible."
    ankles_ok = any(
        (pt := pose.get(k)) is not None and reliable(pt, cfg.confidence) and _in_frame(pt)
        for k in ("left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index")
    )
    if not ankles_ok:
        return False, "Step back or tilt camera slightly so your shoes/ankles fit in frame."
    return True, "Full body framed and ready."


def accumulate_usable_duration(
    samples: list[tuple[float, float | None]],
    max_gap: float,
) -> float:
    """Accumulate duration only between adjacent eligible observations within max_gap."""
    if len(samples) < 2 or max_gap <= 0:
        return 0.0
    total = 0.0
    for (t_a, v_a), (t_b, v_b) in pairwise(samples):
        if t_b <= t_a:
            continue
        if v_a is None or v_b is None or not math.isfinite(v_a) or not math.isfinite(v_b):
            continue
        dt = t_b - t_a
        if 0.0 < dt <= max_gap:
            total += dt
    return total


def check_neutral_pair_comparable(
    samples: list[tuple[float, float | None]],
    hold_a: Passage,
    hold_b: Passage,
    max_gap: float,
    *,
    is_wrapped: bool = False,
) -> tuple[bool, str]:
    """Check whether two independent neutral holds agree within 2x max spread."""
    try:
        stat_a = summarize_examples(samples, [hold_a], max_gap)
        stat_b = summarize_examples(samples, [hold_b], max_gap)
    except ValueError as err:
        return False, str(err)
    med_a = float(stat_a["median"])
    med_b = float(stat_b["median"])
    diff = abs(med_b - med_a)
    if is_wrapped:
        diff = abs(((med_b - med_a + 180.0) % 360.0) - 180.0)
    noise = max(float(stat_a["p90"]) - float(stat_a["p10"]), float(stat_b["p90"]) - float(stat_b["p10"]))
    if diff > max(2.0 * noise, 1e-6):
        return False, (
            f"The two neutral holds differed by {diff:.2f} (spread tolerance {2.0 * noise:.2f}). "
            "Retry the view or select one hold for a Comparison only range."
        )
    return True, "Neutral holds are consistent."


def evaluate_metric_repeat_check(
    samples: list[tuple[float, float | None]],
    holds_4: list[Passage],
    max_gap: float,
    *,
    expected_sign: int | None = None,
) -> tuple[bool, str]:
    """Run repeat_check() on neutral A, changed A, neutral B, changed B plus optional sign check."""
    ok, msg = repeat_check(samples, holds_4, max_gap)
    if not ok:
        return False, msg
    if expected_sign is not None:
        stats = [summarize_examples(samples, [h], max_gap) for h in holds_4]
        change = stats[1]["median"] - stats[0]["median"]
        repeated = stats[3]["median"] - stats[2]["median"]
        if change * expected_sign <= 0 or repeated * expected_sign <= 0:
            direction = "decrease" if expected_sign < 0 else "increase"
            return False, f"Expected a consistent {direction} during the deliberate change."
    return True, msg


class SoloCameraSession:
    """Qt-independent state machine managing solo camera calibration capture and gating."""

    def __init__(
        self,
        cfg: Config,
        *,
        now_monotonic: float = 0.0,
        auto_detect_side_facing: bool = True,
    ) -> None:
        self.cfg = cfg.model_copy(deep=True)
        self.auto_detect_side_facing = auto_detect_side_facing
        self.state: SessionState = "Preview"
        self.active_prompt_id: int = 0
        self.current_instruction: str = ""
        self.hint_message: str = ""
        self.error_message: str = ""

        self._phases: list[GuidePhaseSpec] = []
        self._phase_index: int = 0
        self._accepted_passages: dict[str, AcceptedPassageRecord] = {}

        # Monotonic timers & capture state
        self._last_frame_mono: float = now_monotonic
        self._wait_started_mono: float = now_monotonic
        self._settle_started_mono: float | None = None
        self._settle_anchor_mid: tuple[float, float] | None = None
        self._last_capture_ts: float | None = None
        self._tracked_midpoint: tuple[float, float] | None = None
        self._tracked_torso_px: float | None = None

        # Active phase captured samples: metric -> list[(timestamp, value_or_none)]
        self._phase_start_ts: float | None = None
        self._phase_end_ts: float | None = None
        self._phase_samples: dict[str, list[tuple[float, float | None]]] = {}
        self._phase_last_bad_reason: dict[str, str] = {}
        self._take_active: bool = False
        self._capture_view_lost_since: float | None = None

    @property
    def current_phase(self) -> GuidePhaseSpec | None:
        if 0 <= self._phase_index < len(self._phases):
            return self._phases[self._phase_index]
        return None

    @property
    def accepted_passages(self) -> dict[str, AcceptedPassageRecord]:
        return dict(self._accepted_passages)

    def effective_max_gap(self) -> float:
        return max(self.cfg.solo_camera.max_usable_gap_s, self.cfg.max_gap)

    def start_plan(
        self,
        phases: list[GuidePhaseSpec],
        *,
        now_monotonic: float = 0.0,
        speak_enabled: bool | None = None,
    ) -> int:
        self._phases = list(phases)
        self._phase_index = 0
        self._last_frame_mono = now_monotonic
        self._tracked_midpoint = None
        self._tracked_torso_px = None
        if not self._phases:
            self.state = "Review"
            return self.active_prompt_id
        return self._enter_prompt(now_monotonic=now_monotonic, speak_enabled=speak_enabled)

    def _enter_prompt(self, *, now_monotonic: float, speak_enabled: bool | None = None) -> int:
        phase = self.current_phase
        if phase is None:
            self.state = "Review"
            return self.active_prompt_id
        self.active_prompt_id += 1
        self.current_instruction = phase.instruction
        self.hint_message = ""
        self._reset_attempt_buffers()
        use_speech = self.cfg.solo_camera.speak_instructions if speak_enabled is None else speak_enabled
        if use_speech:
            self.state = "Prompt"
        else:
            self._enter_waiting(now_monotonic=now_monotonic)
        return self.active_prompt_id

    def _enter_waiting(self, *, now_monotonic: float) -> None:
        phase = self.current_phase
        self._wait_started_mono = now_monotonic
        self._settle_started_mono = None
        self._settle_anchor_mid = None
        self._take_active = False
        if phase is not None and not phase.requires_stillness:
            self.state = "Wait for take"
        else:
            self.state = "Wait for view and pose"

    def _reset_attempt_buffers(self) -> None:
        self._phase_start_ts = None
        self._phase_end_ts = None
        self._phase_samples = {}
        self._phase_last_bad_reason = {}
        self._settle_started_mono = None
        self._settle_anchor_mid = None
        self._capture_view_lost_since = None

    def on_speech_finished(self, prompt_id: int, *, now_monotonic: float = 0.0, failed: bool = False) -> None:
        if prompt_id != self.active_prompt_id or self.state != "Prompt":
            return
        if failed:
            self.hint_message = "Speech unavailable; continuing with on-screen text."
        self._enter_waiting(now_monotonic=now_monotonic)

    def start_walking_take(self, *, now_monotonic: float = 0.0) -> None:
        if self.state != "Wait for take":
            return
        self._reset_attempt_buffers()
        self._take_active = True
        self._wait_started_mono = now_monotonic
        self.state = "Capture take"

    def end_walking_take(self) -> AcceptedPassageRecord | None:
        if self.state != "Capture take":
            return None
        self._take_active = False
        return self._finish_current_phase()

    def retry_phase(self, *, now_monotonic: float = 0.0, speak_enabled: bool | None = None) -> int:
        self._tracked_midpoint = None
        self._tracked_torso_px = None
        return self._enter_prompt(now_monotonic=now_monotonic, speak_enabled=speak_enabled)

    def skip_view(self, *, now_monotonic: float = 0.0, speak_enabled: bool | None = None) -> int:
        phase = self.current_phase
        if phase is None:
            self.state = "Review"
            return self.active_prompt_id
        skip_view_name = phase.view
        while self._phase_index < len(self._phases) and self._phases[self._phase_index].view == skip_view_name:
            self._phase_index += 1
        if self._phase_index >= len(self._phases):
            self.state = "Review"
            return self.active_prompt_id
        return self._enter_prompt(now_monotonic=now_monotonic, speak_enabled=speak_enabled)

    def cancel(self) -> None:
        self.state = "Cancelled"

    def set_camera_error(self, message: str) -> None:
        self.error_message = message
        self.state = "Camera error"

    def check_watchdog(self, now_monotonic: float) -> bool:
        """Return True if a camera stall was triggered."""
        if self.state in ("Preview", "Review", "Cancelled", "Camera error"):
            return False
        if now_monotonic - self._last_frame_mono > self.cfg.solo_camera.camera_stall_timeout_s:
            self.set_camera_error("Camera stopped responding")
            return True
        if (
            self.state == "Wait for view and pose"
            and now_monotonic - self._wait_started_mono >= self.cfg.solo_camera.retry_hint_delay_s
            and not self.hint_message
        ):
            phase = self.current_phase
            if phase is not None:
                if phase.view == "front" or phase.turn_direction == "camera":
                    self.hint_message = (
                        "Still waiting for front view: turn to the camera and check framing and lighting."
                    )
                elif phase.turn_direction == "left":
                    self.hint_message = (
                        "Still waiting for side view: turn left so you are side-on to the camera."
                    )
                elif phase.turn_direction == "right":
                    self.hint_message = (
                        "Still waiting for side view: turn right so you are side-on to the camera."
                    )
                else:
                    self.hint_message = (
                        "Still waiting for side view: turn left or right so you are side-on to the camera."
                    )
        return False

    def evaluate_frame(self, frame: FrameObservation) -> GateEvaluation:
        """Evaluate dancer continuity, guide view, and metric observations for one frame."""
        if len(frame.poses) == 0:
            self._tracked_midpoint = None
            return GateEvaluation("no_dancer", "uncertain", False, False, "No dancer visible.", None, None, None)
        if len(frame.poses) > 1:
            self._tracked_midpoint = None
            return GateEvaluation(
                "multiple_dancers",
                "uncertain",
                False,
                False,
                "Second person or reflection detected. Clear the scene.",
                None,
                None,
                None,
            )
        geom = _pose_torso_geometry(frame.poses[0], frame.width, frame.height, self.cfg.confidence)
        if geom is None:
            self._tracked_midpoint = None
            return GateEvaluation(
                "identity_lost",
                "uncertain",
                False,
                False,
                "Shoulders or hips are out of frame or low confidence.",
                None,
                None,
                None,
            )
        mid, torso_px = geom
        if torso_px / max(1, frame.height) < self.cfg.min_torso:
            return GateEvaluation(
                "ok",
                "uncertain",
                False,
                False,
                "Step closer so your torso fills enough of the frame.",
                mid,
                torso_px,
                None,
            )
        if self._tracked_midpoint is not None:
            jump = math.hypot(mid[0] - self._tracked_midpoint[0], mid[1] - self._tracked_midpoint[1])
            if jump / max(torso_px, 1e-6) > 0.75:
                self._tracked_midpoint = mid
                self._tracked_torso_px = torso_px
                return GateEvaluation(
                    "identity_lost",
                    "uncertain",
                    False,
                    False,
                    "Dancer position jumped unexpectedly; re-establishing track.",
                    mid,
                    torso_px,
                    None,
                )
        self._tracked_midpoint = mid
        self._tracked_torso_px = torso_px

        phase = self.current_phase
        req_view = phase.view if phase is not None else None
        detected_view, view_hint = classify_guide_view(
            frame,
            self.cfg,
            requested_view=req_view,
            strict_facing=not self.auto_detect_side_facing,
            turn_direction=phase.turn_direction if phase is not None else None,
        )
        if self.auto_detect_side_facing and detected_view == "side":
            orient = _detect_side_orientation(frame.poses[0], self.cfg)
            if orient is not None:
                detected_side, detected_facing, _ = orient
                if self.cfg.facing != detected_facing or self.cfg.side != detected_side:
                    self.cfg = self.cfg.model_copy(
                        update={"facing": detected_facing, "side": detected_side}
                    )
        view_matches = req_view is not None and detected_view == req_view
        eval_frame = (
            prepare_side_measurement_frame(frame, self.cfg)
            if detected_view == "side"
            else frame
        )
        obs = measure_ranges(eval_frame, self.cfg)
        framing_ok = view_matches or (req_view is None and check_full_body_framing(frame, self.cfg)[0])
        return GateEvaluation("ok", detected_view, view_matches, framing_ok, view_hint, mid, torso_px, obs)

    def process_frame(
        self,
        frame: FrameObservation,
        *,
        now_monotonic: float = 0.0,
        speak_enabled: bool | None = None,
    ) -> GateEvaluation:
        """Advance the state machine with a new camera frame observation."""
        self._last_frame_mono = now_monotonic
        # Exclude duplicate or non-increasing capture timestamps
        if self._last_capture_ts is not None and frame.timestamp <= self._last_capture_ts:
            return self.evaluate_frame(frame)
        self._last_capture_ts = frame.timestamp

        gate = self.evaluate_frame(frame)
        phase = self.current_phase
        if phase is None or self.state in ("Preview", "Prompt", "Review", "Cancelled", "Camera error", "Evaluate"):
            return gate

        if gate.hint:
            self.hint_message = gate.hint

        # Ambiguity / bystander / identity loss restarts active hold/settle
        if gate.dancer_status in ("multiple_dancers", "identity_lost", "no_dancer"):
            if self.state in ("Settle", "Capture"):
                self._reset_attempt_buffers()
                self.state = "Wait for view and pose"
            return gate

        if self.state == "Wait for take":
            return gate

        if self.state == "Capture take":
            if not gate.view_matches or gate.observation is None:
                return gate
            self._record_phase_observation(phase, frame.timestamp, gate.observation)
            assert self._phase_start_ts is not None
            if frame.timestamp - self._phase_start_ts >= self.cfg.solo_camera.walking_take_limit_s:
                self._finish_current_phase(now_monotonic=now_monotonic, speak_enabled=speak_enabled)
            return gate

        if self.state == "Wait for view and pose":
            if gate.view_matches and gate.torso_midpoint is not None:
                self.state = "Settle"
                self._settle_started_mono = now_monotonic
                self._settle_anchor_mid = gate.torso_midpoint
                self.hint_message = "Hold steady…"
            return gate

        if self.state == "Settle":
            if not gate.view_matches or gate.torso_midpoint is None or gate.torso_length_px is None:
                self._settle_started_mono = None
                self._settle_anchor_mid = None
                self.state = "Wait for view and pose"
                return gate
            assert self._settle_anchor_mid is not None and self._settle_started_mono is not None
            drift = math.hypot(
                gate.torso_midpoint[0] - self._settle_anchor_mid[0],
                gate.torso_midpoint[1] - self._settle_anchor_mid[1],
            ) / max(gate.torso_length_px, 1e-6)
            if drift > self.cfg.solo_camera.settle_drift_torso_lengths:
                # Movement resumed: re-arm settling
                self._settle_started_mono = now_monotonic
                self._settle_anchor_mid = gate.torso_midpoint
                self.hint_message = "Movement detected; hold still to begin."
                return gate
            if now_monotonic - self._settle_started_mono >= self.cfg.solo_camera.stable_time_s:
                self._reset_attempt_buffers()
                self.state = "Capture"
                self.hint_message = "Capturing…"
                if gate.observation is not None:
                    self._record_phase_observation(phase, frame.timestamp, gate.observation)
            return gate

        if self.state == "Capture":
            target_hold_s = (
                phase.hold_window_s
                if (phase.hold_window_s is not None and phase.hold_window_s > 0)
                else self.cfg.solo_camera.hold_window_s
            )
            if not gate.view_matches or gate.observation is None:
                if self._capture_view_lost_since is None:
                    self._capture_view_lost_since = now_monotonic
                if now_monotonic - self._capture_view_lost_since > 0.35:
                    # Prolonged loss of intended view discards active attempt and re-arms
                    self._reset_attempt_buffers()
                    self.state = "Wait for view and pose"
                    self.hint_message = gate.hint or "View lost; returning to waiting."
                return gate
            self._capture_view_lost_since = None
            self._record_phase_observation(phase, frame.timestamp, gate.observation)
            assert self._phase_start_ts is not None
            if frame.timestamp - self._phase_start_ts >= target_hold_s:
                self._finish_current_phase(now_monotonic=now_monotonic, speak_enabled=speak_enabled)
            return gate

        return gate

    def _record_phase_observation(
        self,
        phase: GuidePhaseSpec,
        timestamp: float,
        obs: MetricObservation,
    ) -> None:
        if self._phase_start_ts is None:
            self._phase_start_ts = timestamp
        self._phase_end_ts = timestamp
        side_cutoff = self.cfg.max_view_width + self.cfg.solo_camera.front_view_width_margin
        scoring_view_ok = (
            view_eligible(
                phase.view,
                obs.view_width,
                side_cutoff if phase.view == "side" else self.cfg.max_view_width,
            )
            if obs.view_width is not None
            else True
        )
        for key in phase.metrics:
            bucket = self._phase_samples.setdefault(key, [])
            qual = obs.quality.get(key)
            val = obs.values.get(key)
            if not scoring_view_ok:
                bucket.append((timestamp, None))
                self._phase_last_bad_reason[key] = f"Not eligible for {phase.view} view"
            elif qual is not None and qual.valid and val is not None and math.isfinite(val):
                bucket.append((timestamp, float(val)))
            else:
                bucket.append((timestamp, None))
                self._phase_last_bad_reason[key] = qual.reason if qual and qual.reason else "Landmark unavailable"

    def _finish_current_phase(
        self,
        *,
        now_monotonic: float = 0.0,
        speak_enabled: bool | None = None,
    ) -> AcceptedPassageRecord | None:
        phase = self.current_phase
        if phase is None:
            return None
        self.state = "Evaluate"
        max_gap = self.effective_max_gap()
        target_hold_s = (
            phase.hold_window_s
            if (phase.hold_window_s is not None and phase.hold_window_s > 0)
            else self.cfg.solo_camera.hold_window_s
        )
        min_usable = (
            min(phase.min_usable_hold_s, target_hold_s)
            if (phase.min_usable_hold_s is not None and phase.min_usable_hold_s > 0)
            else min(self.cfg.solo_camera.min_usable_hold_s, target_hold_s)
        )
        eligible_metrics: dict[str, float] = {}
        reasons: dict[str, str] = {}
        for key in phase.metrics:
            samples = self._phase_samples.get(key, [])
            dur = accumulate_usable_duration(samples, max_gap)
            if dur >= min_usable:
                eligible_metrics[key] = dur
            else:
                reasons[key] = self._phase_last_bad_reason.get(
                    key,
                    f"Usable duration {dur:.2f}s below {min_usable:.2f}s minimum",
                )
        start_ts = self._phase_start_ts if self._phase_start_ts is not None else 0.0
        end_ts = self._phase_end_ts if self._phase_end_ts is not None and self._phase_end_ts > start_ts else start_ts + 0.05
        record = AcceptedPassageRecord(
            phase_id=phase.phase_id,
            plan_kind=phase.plan_kind,
            view=phase.view,
            repetition=phase.repetition,
            role=phase.role,
            passage=Passage(start=start_ts, end=end_ts),
            check_id=phase.check_id,
            eligible_metrics=eligible_metrics,
            unavailable_reasons=reasons,
            usable=bool(eligible_metrics),
            detected_side=self.cfg.side,
            detected_facing=self.cfg.facing,
        )
        if not record.usable:
            # Failed hold: keep previous accepted holds, re-arm current phase
            self.hint_message = "Hold had insufficient usable footage; retrying this phase."
            self._enter_waiting(now_monotonic=now_monotonic)
            return record

        self._accepted_passages[phase.phase_id] = record
        self._phase_index += 1
        if self._phase_index >= len(self._phases):
            self.state = "Review"
        else:
            self._enter_prompt(now_monotonic=now_monotonic, speak_enabled=speak_enabled)
        return record
