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
        neutral_instruction="Take up your usual tango stance, looking ahead.",
        changed_instruction="Lower your gaze a little. Keep your chest facing the same way.",
        expected_sign=-1,
    ),
    "head_position": CheckSpec(
        check_id="head_position",
        title="Head position · side",
        view="side",
        metrics=("head_forward", "head_inclination"),
        neutral_instruction="Take up your usual tango stance with your head balanced.",
        changed_instruction="Bring your head slightly forward while keeping your gaze level. Return to your usual position when prompted.",
        expected_sign=None,
    ),
    "shoulder_clearance": CheckSpec(
        check_id="shoulder_clearance",
        title="Shoulders · front",
        view="front",
        metrics=("shoulder_clearance",),
        neutral_instruction="Face the camera with both shoulders relaxed and settled.",
        changed_instruction="Raise both shoulders gently. Keep looking ahead. Let both shoulders settle for the next hold.",
        expected_sign=-1,
    ),
    "torso_axis_tilt": CheckSpec(
        check_id="torso_axis_tilt",
        title="Torso axis · front",
        view="front",
        metrics=("torso_axis_tilt",),
        neutral_instruction="Face the camera in your upright ready-to-walk stance.",
        changed_instruction="Let your torso incline a little toward the same side each time. Then return to your usual stance.",
        expected_sign=1,
    ),
    "trunk_inclination": CheckSpec(
        check_id="trunk_inclination",
        title="Trunk inclination · side",
        view="side",
        metrics=("trunk_inclination",),
        neutral_instruction="Take up your usual side-on tango stance.",
        changed_instruction="Incline your trunk forward a little, then return to your usual stance.",
        expected_sign=None,
    ),
    "hip_offset": CheckSpec(
        check_id="hip_offset",
        title="Hip position · side",
        view="side",
        metrics=("hip_offset",),
        neutral_instruction="Take up your usual side-on tango stance with ankles visible.",
        changed_instruction="If comfortable, shift your hips forward slightly, then return.",
        expected_sign=None,
    ),
    "knee_flexion": CheckSpec(
        check_id="knee_flexion",
        title="Knee flexion · side",
        view="side",
        metrics=("knee_flexion",),
        neutral_instruction="Stand in your usual soft-knee tango stance.",
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
            instruction="Face the camera. Stand as you do when you’re ready to walk. Look ahead and let your shoulders settle.",
            role="neutral",
            metrics=front_metrics,
        ),
        GuidePhaseSpec(
            phase_id="front_b",
            plan_kind="quick",
            view="front",
            repetition=2,
            instruction="Relax for a moment. Now return to the same ready-to-walk stance.",
            role="neutral",
            metrics=front_metrics,
        ),
        GuidePhaseSpec(
            phase_id="side_a",
            plan_kind="quick",
            view="side",
            repetition=1,
            instruction="Turn side-on in the direction shown. Keep the camera where it is. Take up your usual tango stance.",
            role="neutral",
            metrics=side_metrics,
        ),
        GuidePhaseSpec(
            phase_id="side_b",
            plan_kind="quick",
            view="side",
            repetition=2,
            instruction="Relax, then return to the same stance. Keep looking in the same direction.",
            role="neutral",
            metrics=side_metrics,
        ),
    ]


def build_check_phases(check_ids: list[str] | tuple[str, ...]) -> list[GuidePhaseSpec]:
    """Build 4-phase neutral A -> changed A -> neutral B -> changed B sequences for selected checks."""
    phases: list[GuidePhaseSpec] = []
    for cid in check_ids:
        spec = CHECK_SPECS[cid]
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
            instruction="Take up to four natural tango steps across the frame, keeping your shoes visible. Stop at the end.",
            role="walking",
            metrics=walking_metrics,
            requires_stillness=False,
        ),
        GuidePhaseSpec(
            phase_id="walk_2",
            plan_kind="walking",
            view="side",
            repetition=2,
            instruction="Take up to four natural tango steps across the frame, keeping your shoes visible. Stop at the end.",
            role="walking",
            metrics=walking_metrics,
            requires_stillness=False,
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
    keys = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    pts = [pose.get(k) for k in keys]
    if any(p is None or not reliable(p, confidence) or not _in_frame(p) for p in pts):
        return None
    ls, rs, lh, rh = (pts[0], pts[1], pts[2], pts[3])
    assert ls and rs and lh and rh
    smx = (ls.x + rs.x) * 0.5 * width
    smy = (ls.y + rs.y) * 0.5 * height
    hmx = (lh.x + rh.x) * 0.5 * width
    hmy = (lh.y + rh.y) * 0.5 * height
    torso_px = math.hypot(smx - hmx, smy - hmy)
    if torso_px <= 1e-6:
        return None
    mid = ((smx + hmx) * 0.5, (smy + hmy) * 0.5)
    return mid, torso_px


def classify_guide_view(
    frame: FrameObservation,
    cfg: Config,
    requested_view: Literal["front", "side"] | None = None,
    *,
    strict_facing: bool = True,
) -> tuple[GuideView, str]:
    """Classify the single-dancer view for guided capture using strict facial and torso cues."""
    if len(frame.poses) != 1:
        return "uncertain", "Ensure exactly one dancer is visible."
    pose = frame.poses[0]
    conf = cfg.confidence
    for key in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        pt = pose.get(key)
        if pt is None or not reliable(pt, conf) or not _in_frame(pt):
            return "uncertain", "Step back so shoulders and hips are clearly visible."

    ls, rs = pose["left_shoulder"], pose["right_shoulder"]
    lh, rh = pose["left_hip"], pose["right_hip"]
    nose = pose.get("nose")
    le = pose.get("left_ear")
    re = pose.get("right_ear")

    obs = measure_ranges(frame, cfg)
    vw = obs.view_width
    if vw is None or not math.isfinite(vw):
        smx = (ls.x + rs.x) * 0.5 * frame.width
        smy = (ls.y + rs.y) * 0.5 * frame.height
        hmx = (lh.x + rh.x) * 0.5 * frame.width
        hmy = (lh.y + rh.y) * 0.5 * frame.height
        torso_px = math.hypot(smx - hmx, smy - hmy)
        if torso_px <= 1e-6:
            return "uncertain", "Move to a clearer position so torso width can be measured."
        sh_span = math.hypot((ls.x - rs.x) * frame.width, (ls.y - rs.y) * frame.height)
        hip_span = math.hypot((lh.x - rh.x) * frame.width, (lh.y - rh.y) * frame.height)
        vw = max(sh_span, hip_span) / torso_px

    front_cutoff = cfg.max_view_width + cfg.solo_camera.front_view_width_margin

    # Front check: width above front_cutoff + front anatomical order + nose between ears
    if vw > front_cutoff:
        anatomical_front = ls.x > rs.x and lh.x > rh.x
        if (
            anatomical_front
            and nose is not None and reliable(nose, conf) and _in_frame(nose)
            and le is not None and reliable(le, conf) and _in_frame(le)
            and re is not None and reliable(re, conf) and _in_frame(re)
        ):
            ear_min = min(le.x, re.x)
            ear_max = max(le.x, re.x)
            if ear_min <= nose.x <= ear_max:
                return "front", ""
        return "uncertain", "Face the camera squarely with your head and shoulders level."

    # Side check: width <= max_view_width + ear-to-nose horizontal sign matches facing
    if vw <= cfg.max_view_width:
        ear_key = f"{cfg.side}_ear"
        ear = pose.get(ear_key)
        if ear is None or not reliable(ear, conf) or not _in_frame(ear):
            # Fallback to whichever ear is reliable if primary ear momentarily occluded
            other_key = "right_ear" if cfg.side == "left" else "left_ear"
            other = pose.get(other_key)
            if other is not None and reliable(other, conf) and _in_frame(other):
                ear = other
        if nose is None or not reliable(nose, conf) or not _in_frame(nose) or ear is None:
            return "uncertain", "Keep your profile head (nose and ear) visible in the frame."
        dx = nose.x - ear.x
        if abs(dx) < 0.015:
            return "uncertain", "Turn more clearly side-on so your profile direction is clear."
        if not strict_facing:
            return "side", ""
        expected_sign = 1.0 if cfg.facing == "right" else -1.0
        if dx * expected_sign > 0:
            return "side", ""
        return "uncertain", f"Face {cfg.facing} in the recorded image as shown."

    # Ambiguous buffer zone between max_view_width and front_cutoff
    if requested_view == "front":
        return "uncertain", "Turn more squarely toward the camera."
    if requested_view == "side":
        return "uncertain", "Turn more fully side-on to the camera."
    return "uncertain", "Turn clearly front-on or side-on."


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
        auto_detect_side_facing: bool = False,
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

    @property
    def current_phase(self) -> GuidePhaseSpec | None:
        if 0 <= self._phase_index < len(self._phases):
            return self._phases[self._phase_index]
        return None

    @property
    def accepted_passages(self) -> dict[str, AcceptedPassageRecord]:
        return dict(self._accepted_passages)

    def effective_max_gap(self) -> float:
        return min(self.cfg.solo_camera.max_usable_gap_s, self.cfg.max_gap)

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
                self.hint_message = (
                    f"Still waiting for {phase.view} view. Check framing, facing, and lighting."
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
        )
        if self.auto_detect_side_facing and detected_view == "side":
            pose = frame.poses[0]
            nose = pose.get("nose")
            le = pose.get("left_ear")
            re = pose.get("right_ear")
            ear = le if (le is not None and reliable(le, self.cfg.confidence)) else re
            if nose is not None and ear is not None:
                detected_facing: Literal["left", "right"] = "right" if (nose.x - ear.x) > 0 else "left"
                detected_side: Literal["left", "right"] = (
                    "left"
                    if (le is not None and (re is None or (le.visibility or 0.0) >= (re.visibility or 0.0)))
                    else "right"
                )
                if self.cfg.facing != detected_facing or self.cfg.side != detected_side:
                    self.cfg = self.cfg.model_copy(
                        update={"facing": detected_facing, "side": detected_side}
                    )
        view_matches = req_view is not None and detected_view == req_view
        obs = measure_ranges(frame, self.cfg)
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
            if not gate.view_matches or gate.observation is None:
                # Prolonged loss of intended view discards active attempt and re-arms
                self._reset_attempt_buffers()
                self.state = "Wait for view and pose"
                self.hint_message = gate.hint or "View lost; returning to waiting."
                return gate
            self._record_phase_observation(phase, frame.timestamp, gate.observation)
            assert self._phase_start_ts is not None
            if frame.timestamp - self._phase_start_ts >= self.cfg.solo_camera.hold_window_s:
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
        scoring_view_ok = (
            view_eligible(phase.view, obs.view_width, self.cfg.max_view_width)
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
        min_usable = self.cfg.solo_camera.min_usable_hold_s
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
