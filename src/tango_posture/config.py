import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .normalization_models import MeasurementNormalizationConfig


class Threshold(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    entry: float = Field(gt=0)
    clear: float = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> "Threshold":
        if self.clear >= self.entry:
            raise ValueError("clear threshold must be below entry threshold")
        return self


class RangeDefault(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    lower: float
    upper: float
    unit: str
    view: Literal["side", "front", "all"]

    @model_validator(mode="after")
    def ordered(self) -> "RangeDefault":
        if self.lower > self.upper:
            raise ValueError("Range minimum must not exceed maximum")
        return self


class AnalysisRegion(BaseModel):
    """Normalized [0, 1] image region restricting candidate dancer detection."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x_min: float = Field(default=0.0, ge=0.0, le=1.0)
    y_min: float = Field(default=0.0, ge=0.0, le=1.0)
    x_max: float = Field(default=1.0, ge=0.0, le=1.0)
    y_max: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def ordered(self) -> "AnalysisRegion":
        if self.x_min >= self.x_max:
            raise ValueError("analysis region x_min must be below x_max")
        if self.y_min >= self.y_max:
            raise ValueError("analysis region y_min must be below y_max")
        return self


class DancerAnalysisConfig(BaseModel):
    """Configurable controls for Single, Couple, and Auto opening-window analysis."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    default_dancer_mode: Literal["auto", "single", "couple"] = "auto"
    auto_detection_window_s: float = Field(default=10.0, gt=0.0)
    primary_coverage_min: float = Field(default=0.60, gt=0.0, le=1.0)
    primary_duration_min_s: float = Field(default=2.0, gt=0.0)
    second_joint_coverage_min: float = Field(default=0.45, gt=0.0, le=1.0)
    second_sustained_min_s: float = Field(default=1.5, gt=0.0)
    brief_second_max_s: float = Field(default=0.5, ge=0.0)
    together_min_s: float = Field(default=0.4, gt=0.0)
    together_max_hip_separation_torso_lengths: float = Field(default=2.0, gt=0.0)
    together_direction_cosine_min: float = Field(default=0.5, ge=-1.0, le=1.0)
    identity_max_innovation_torso_lengths: float = Field(default=0.75, gt=0.0)
    identity_margin_torso_lengths: float = Field(default=0.15, ge=0.0)
    max_observation_gap_s: float = Field(default=0.20, gt=0.0)
    display_interpolation_max_gap_s: float = Field(default=0.13, ge=0.0)
    velocity_filter_window_s: float = Field(default=0.15, gt=0.0)
    persistence_s: float = Field(default=0.15, gt=0.0)
    pairing_window_s: float = Field(default=0.50, gt=0.0)
    start_speed: float = Field(default=0.20, gt=0.0)
    stop_speed: float = Field(default=0.10, ge=0.0)
    slowing_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    transition_min_coverage: float = Field(default=0.75, gt=0.0, le=1.0)
    secondary_speed_rise: float = Field(default=0.12, gt=0.0)
    analyze_soundtrack: bool = False
    auto_turn_suggestions: bool = False

    @model_validator(mode="after")
    def validated_thresholds(self) -> "DancerAnalysisConfig":
        if self.brief_second_max_s >= self.second_sustained_min_s:
            raise ValueError("brief_second_max_s must be below second_sustained_min_s")
        if self.stop_speed >= self.start_speed:
            raise ValueError("stop_speed must be below start_speed")
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    camera: int = Field(default=0, ge=0)
    width: int = Field(default=640, gt=0)
    height: int = Field(default=480, gt=0)
    model_path: str = "models/pose_landmarker_lite.task"
    analysis_model: Literal["lite", "full", "heavy"] = "lite"
    data_dir: str = "data"
    side: Literal["left", "right"] = "left"
    facing: Literal["left", "right"] = "right"
    capture_profile: str = "default"
    mirror: bool = False
    confidence: float = Field(default=0.70, ge=0, le=1)
    min_torso: float = Field(default=0.12, gt=0)
    min_head: float = Field(default=0.04, gt=0)
    max_view_width: float = Field(default=0.45, gt=0)
    calibration_suggest_lower_pct: float = Field(default=10.0, ge=0.0, lt=100.0)
    calibration_suggest_upper_pct: float = Field(default=90.0, gt=0.0, le=100.0)
    calibration_min_usable_samples: int = Field(default=2, ge=2)
    view_change_tolerance: float = Field(default=0.15, gt=0)
    smoothing: float = Field(default=0.25, gt=0)
    overlay_smoothing: float = Field(default=0.20, ge=0)
    min_filter_time_constant: float = Field(default=0.0001, gt=0.0)
    entry_duration: float = Field(default=1.5, gt=0)
    clear_duration: float = Field(default=0.75, gt=0)
    max_gap: float = Field(default=0.5, gt=0)
    calibration_duration: float = Field(default=12, gt=0)
    calibration_valid_duration: float = Field(default=9, gt=0)
    calibration_samples: int = Field(default=60, ge=2)
    offset_mad: float = Field(default=0.03, gt=0)
    angle_mad: float = Field(default=4, gt=0)
    head_neutral_offset: float = Field(default=-12.0)
    head_down_duration: float = Field(default=3.0, gt=0)
    head_down_angle: float = Field(default=-15.0, lt=0)
    beat_tracking: bool = True
    tango_bpm: float = Field(default=120.0, gt=20, le=240)
    beat_analysis_seconds: float = Field(default=30.0, gt=0)
    auto_record_start_delay: float = Field(default=1.0, ge=0, le=60)
    auto_record_stop_delay: float = Field(default=3.0, ge=0, le=120)
    max_recordings: int = Field(default=10, ge=1, le=1000)
    role_assign_ms: float = Field(default=300.0, ge=0)
    role_hold_ms: float = Field(default=5000.0, ge=0)
    shoulder_tension_duration: float = Field(default=1.0, gt=0)
    shoulder_raise_threshold: float = Field(default=10.0, gt=0)
    shoulder_neutral_clearance: float = Field(default=1.05, gt=0)
    shoulder_visibility_floor: float = Field(default=0.50, ge=0.0, le=1.0)
    shoulder_ear_hip_scale_factor: float = Field(default=0.26, gt=0.0)
    shoulder_min_ear_hip_px: float = Field(default=20.0, gt=0.0)
    shoulder_nose_ear_scale_factor: float = Field(default=0.68, gt=0.0)
    shoulder_ear_ear_scale_factor: float = Field(default=0.65, gt=0.0)
    shoulder_width_scale_factor: float = Field(default=0.28, gt=0.0)
    shoulder_min_ref_scale_px: float = Field(default=8.0, gt=0.0)
    shoulder_baseline_clamp_floor: float = Field(default=0.20, gt=0.0)
    shoulder_adaptive_max_samples: int = Field(default=600, ge=1)
    shoulder_adaptive_min_samples: int = Field(default=15, ge=1)
    shoulder_adaptive_min_clearance: float = Field(default=0.65, gt=0.0)
    shoulder_adaptive_max_clearance: float = Field(default=1.50, gt=0.0)
    shoulder_adaptive_percentile: float = Field(default=75.0, gt=0.0, le=100.0)
    shoulder_recovery_min_pct: float = Field(default=4.0, ge=0.0)
    shoulder_recovery_ratio: float = Field(default=0.65, gt=0.0, le=1.0)
    allow_all_angles: bool = False
    embrace_closed_threshold: float = Field(default=0.72, gt=0, le=1.0)
    embrace_hysteresis_seconds: float = Field(default=1.0, ge=0)
    closed_embrace_min_knee_bend: float = Field(default=8.0, ge=0)
    knee_straight_duration: float = Field(default=1.0, gt=0)
    knee_recovery_margin: float = Field(default=2.5, ge=0.0)
    knee_visibility_floor: float = Field(default=0.35, ge=0.0, le=1.0)
    knee_min_segment_px: float = Field(default=10.0, gt=0.0)
    knee_tiebreak_other_max_vis: float = Field(default=0.60, ge=0.0, le=1.0)
    knee_tiebreak_min_advantage: float = Field(default=0.15, ge=0.0, le=1.0)
    slow_tango_max_bpm: float = Field(default=115.0, gt=0, le=240)
    ankle_high_lift_ratio: float = Field(default=0.06, gt=0, le=1.0)
    max_consecutive_high_ankle_steps: int = Field(default=2, ge=1)
    axis_curve_entry: float = Field(default=0.06, gt=0)
    axis_curve_clear: float = Field(default=0.035, ge=0)
    axis_curve_duration: float = Field(default=1.5, gt=0)
    axis_curve_clear_duration: float = Field(default=0.75, gt=0)
    axis_neutral_offset: float = Field(default=0.0)
    # Accept older TOML files; neither replay flag has a runtime consumer.
    replay_audio: bool = False
    replay_auto_analyse_movement: bool = True
    replay_step_history_seconds: float = Field(default=5.0, ge=0.5, le=30.0)
    replay_joint_interpolation: Literal["linear", "quadratic"] = "linear"
    replay_joint_max_gap: float = Field(default=3.0, ge=0)
    replay_recovery_auto: bool = False
    replay_recovery_max_gap_s: float = Field(default=0.75, ge=0.05, le=2.00)
    replay_recovery_context_s: float = Field(default=0.50, ge=0.10, le=2.00)
    replay_pivot_interpolation_max_gap_s: float = Field(default=0.20, ge=0.0, le=1.00)
    replay_show_recovered_joints: bool = True
    replay_recovery_budget_s: float = Field(default=12.0, ge=1.0, le=60.0)
    calibration_defaults: dict[str, RangeDefault] = Field(default_factory=dict)
    measurement_normalization: MeasurementNormalizationConfig = Field(
        default_factory=MeasurementNormalizationConfig
    )
    dancer_analysis: DancerAnalysisConfig = Field(default_factory=DancerAnalysisConfig)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = Field(default="llama3.2", min_length=1)
    ollama_timeout: float = Field(default=120.0, gt=0, le=1800)
    ollama_temperature: float = Field(default=0.2, ge=0, le=2)
    ollama_num_ctx: int = Field(default=16384, ge=1024, le=131072)
    ollama_num_predict: int = Field(default=2048, ge=128, le=16384)
    ollama_max_report_chars: int = Field(default=40000, ge=1000)
    ollama_instruction: str = (
        "Explain the supported observations and suggest three focused practice exercises."
    )
    head_forward: Threshold = Threshold(entry=0.08, clear=0.05)
    head_inclination: Threshold = Threshold(entry=10, clear=6)
    trunk_inclination: Threshold = Threshold(entry=8, clear=5)
    axis_curve: Threshold | None = None

    @model_validator(mode="after")
    def durations(self) -> "Config":
        if self.calibration_defaults:
            from .calibration.models import METRIC_DETAILS

            for key, target in self.calibration_defaults.items():
                if key not in METRIC_DETAILS or target.unit != METRIC_DETAILS[key][2]:
                    raise ValueError(f"Unknown default metric or incompatible unit: {key}")
        if self.calibration_valid_duration > self.calibration_duration:
            raise ValueError("valid calibration duration exceeds capture window")
        if self.calibration_suggest_lower_pct >= self.calibration_suggest_upper_pct:
            raise ValueError("calibration_suggest_lower_pct must be below calibration_suggest_upper_pct")
        if self.shoulder_adaptive_min_samples > self.shoulder_adaptive_max_samples:
            raise ValueError("shoulder_adaptive_min_samples must not exceed shoulder_adaptive_max_samples")
        if self.shoulder_adaptive_min_clearance >= self.shoulder_adaptive_max_clearance:
            raise ValueError("shoulder_adaptive_min_clearance must be below shoulder_adaptive_max_clearance")
        if self.axis_curve is not None:
            if "axis_curve_entry" not in self.model_fields_set:
                self.axis_curve_entry = self.axis_curve.entry
            if "axis_curve_clear" not in self.model_fields_set:
                self.axis_curve_clear = self.axis_curve.clear
        if self.axis_curve_clear >= self.axis_curve_entry:
            raise ValueError("axis_curve clear threshold must be below entry threshold")
        if self.axis_curve is None:
            self.axis_curve = Threshold(entry=self.axis_curve_entry, clear=self.axis_curve_clear)
        return self


def parse_bool(value: str) -> bool:
    if value.lower() not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
        raise ValueError("expected true or false")
    return value.lower() in ("true", "1", "yes", "on")


SETTING_GROUPS = {
    "Camera & view": (
        "side",
        "facing",
        "capture_profile",
        "allow_all_angles",
    ),
    "Files": ("model_path", "analysis_model", "data_dir"),
    "Quality & calibration": (
        "confidence",
        "min_torso",
        "min_head",
        "max_view_width",
        "calibration_suggest_lower_pct",
        "calibration_suggest_upper_pct",
        "calibration_min_usable_samples",
    ),
    "Tracking": (
        "smoothing",
        "overlay_smoothing",
        "min_filter_time_constant",
        "entry_duration",
        "clear_duration",
        "max_gap",
    ),
    "Head & shoulders": (
        "head_neutral_offset",
        "head_down_duration",
        "head_down_angle",
        "shoulder_tension_duration",
        "shoulder_raise_threshold",
        "shoulder_neutral_clearance",
        "shoulder_visibility_floor",
        "shoulder_ear_hip_scale_factor",
        "shoulder_min_ear_hip_px",
        "shoulder_nose_ear_scale_factor",
        "shoulder_ear_ear_scale_factor",
        "shoulder_width_scale_factor",
        "shoulder_min_ref_scale_px",
        "shoulder_baseline_clamp_floor",
        "shoulder_adaptive_max_samples",
        "shoulder_adaptive_min_samples",
        "shoulder_adaptive_min_clearance",
        "shoulder_adaptive_max_clearance",
        "shoulder_adaptive_percentile",
        "shoulder_recovery_min_pct",
        "shoulder_recovery_ratio",
    ),
    "Embrace & knees": (
        "role_assign_ms",
        "role_hold_ms",
        "embrace_closed_threshold",
        "embrace_hysteresis_seconds",
        "closed_embrace_min_knee_bend",
        "knee_straight_duration",
        "knee_recovery_margin",
        "knee_visibility_floor",
        "knee_min_segment_px",
        "knee_tiebreak_other_max_vis",
        "knee_tiebreak_min_advantage",
    ),
    "Torso & hip offset": (
        "axis_curve_entry",
        "axis_curve_clear",
        "axis_curve_duration",
        "axis_curve_clear_duration",
        "axis_neutral_offset",
    ),
    "Slow tango": (
        "slow_tango_max_bpm",
        "ankle_high_lift_ratio",
        "max_consecutive_high_ankle_steps",
    ),
    "Music": (
        "tango_bpm",
    ),
    "Replay movement": (
        "replay_step_history_seconds",
        "replay_joint_interpolation",
        "replay_joint_max_gap",
        "replay_recovery_auto",
        "replay_recovery_max_gap_s",
        "replay_recovery_context_s",
        "replay_pivot_interpolation_max_gap_s",
        "replay_show_recovered_joints",
        "replay_recovery_budget_s",
    ),
    "Analysis": (
        "dancer_analysis.default_dancer_mode",
        "dancer_analysis.auto_detection_window_s",
        "dancer_analysis.primary_coverage_min",
        "dancer_analysis.primary_duration_min_s",
        "dancer_analysis.second_joint_coverage_min",
        "dancer_analysis.second_sustained_min_s",
        "dancer_analysis.brief_second_max_s",
        "dancer_analysis.together_min_s",
        "dancer_analysis.together_max_hip_separation_torso_lengths",
        "dancer_analysis.together_direction_cosine_min",
        "dancer_analysis.identity_max_innovation_torso_lengths",
        "dancer_analysis.identity_margin_torso_lengths",
        "dancer_analysis.max_observation_gap_s",
        "dancer_analysis.display_interpolation_max_gap_s",
        "dancer_analysis.velocity_filter_window_s",
        "dancer_analysis.persistence_s",
        "dancer_analysis.pairing_window_s",
        "dancer_analysis.start_speed",
        "dancer_analysis.stop_speed",
        "dancer_analysis.slowing_fraction",
        "dancer_analysis.transition_min_coverage",
        "dancer_analysis.secondary_speed_rise",
        "dancer_analysis.analyze_soundtrack",
        "dancer_analysis.auto_turn_suggestions",
    ),
    "Review & Ollama": (
        "ollama_base_url",
        "ollama_model",
        "ollama_timeout",
        "ollama_temperature",
        "ollama_num_ctx",
        "ollama_num_predict",
        "ollama_max_report_chars",
        "ollama_instruction",
    ),
}

SETTING_MAP: dict[str, tuple[str, Callable]] = {}


def _setting_parser(field: str) -> Callable:
    if "." in field:
        root, sub = field.split(".", 1)
        if root == "dancer_analysis":
            kind = DancerAnalysisConfig.model_fields[sub].annotation
        else:
            kind = float
    else:
        kind = Config.model_fields[field].annotation
    if kind is bool:
        return parse_bool
    if kind is int:
        return int
    if kind is float:
        return float
    return str


for _fields in SETTING_GROUPS.values():
    for _field in _fields:
        SETTING_MAP[_field] = (_field, _setting_parser(_field))

SETTING_DESCRIPTIONS: dict[str, str] = {
    "side": "Anatomical side selected for side-view measurements.",
    "facing": "Direction faced in the original, unmirrored image.",
    "capture_profile": "Name identifying the camera and capture setup.",
    "allow_all_angles": "Enables the experimental measurement path for additional view angles.",
    "model_path": "Location of the pinned Lite model used when offline analysis selects Lite.",
    "analysis_model": (
        "Offline analysis model: lite (default), full, or heavy. "
        "Missing models download when analysis starts; live coaching remains Lite."
    ),
    "data_dir": "Directory for the local database and practice artifacts.",
    "confidence": "Minimum landmark confidence used by quality checks.",
    "min_torso": "Minimum visible torso length as a fraction of image height.",
    "min_head": "Minimum visible shoulder-to-ear segment as a fraction of image height.",
    "max_view_width": "Maximum body-width-to-torso ratio accepted for a side view.",
    "calibration_suggest_lower_pct": "Lower percentile used by the calibration editor's middle-range suggestion.",
    "calibration_suggest_upper_pct": "Upper percentile used by the calibration editor's middle-range suggestion.",
    "calibration_min_usable_samples": "Minimum usable frames required to summarize and approve a calibration passage.",
    "smoothing": "Time constant for smoothing measurements over time.",
    "overlay_smoothing": "Smoothing time constant for rendered landmark lines and segments.",
    "min_filter_time_constant": "Minimum filter time constant floor (seconds) for descriptive knee/shoulder smoothing.",
    "entry_duration": "Time a threshold must be crossed before confirming a deviation.",
    "clear_duration": "Time a value must remain below its clear threshold before closing an event.",
    "max_gap": "Maximum observation gap treated as continuous tracking.",
    "head_neutral_offset": "Offset used by the separate ear-to-nose horizontal-angle cue.",
    "head_down_duration": "Duration required before showing the head-angle warning.",
    "head_down_angle": "Ear-to-nose image-angle boundary for that warning.",
    "shoulder_tension_duration": "Duration required before showing the shoulder cue.",
    "shoulder_raise_threshold": "Shoulder-elevation threshold above the relaxed baseline.",
    "shoulder_neutral_clearance": "Fallback clearance used when no captured shoulder baseline is available.",
    "shoulder_visibility_floor": "Helper landmark visibility floor checked in shoulder clearance geometry.",
    "shoulder_ear_hip_scale_factor": "Primary anatomical scale multiplier applied to vertical ear-to-hip separation.",
    "shoulder_min_ear_hip_px": "Minimum vertical ear-to-hip pixel separation required for the primary scale candidate.",
    "shoulder_nose_ear_scale_factor": "Fallback scale multiplier applied to 2D nose-to-ear pixel distance.",
    "shoulder_ear_ear_scale_factor": "Fallback scale multiplier applied to horizontal ear-to-ear pixel distance.",
    "shoulder_width_scale_factor": "Fallback scale multiplier applied to horizontal shoulder-to-shoulder pixel distance.",
    "shoulder_min_ref_scale_px": "Minimum shoulder reference scale S in pixels; smaller values mark clearance unavailable.",
    "shoulder_baseline_clamp_floor": "Minimum floor for effective neutral shoulder clearance B in fallback elevation percentages.",
    "shoulder_adaptive_max_samples": "Maximum qualifying relaxed-clearance samples collected for the adaptive shoulder baseline window.",
    "shoulder_adaptive_min_samples": "Minimum qualifying samples required before the adaptive shoulder baseline takes effect.",
    "shoulder_adaptive_min_clearance": "Lower raw clearance ratio bound for samples admitted into the adaptive baseline window.",
    "shoulder_adaptive_max_clearance": "Upper raw clearance ratio bound for samples admitted into the adaptive baseline window.",
    "shoulder_adaptive_percentile": "Percentile of adaptive baseline samples compared against the starting neutral clearance.",
    "shoulder_recovery_min_pct": "Minimum percentage floor for the fallback shoulder tension recovery hysteresis boundary.",
    "shoulder_recovery_ratio": "Fraction of shoulder_raise_threshold used for the fallback shoulder recovery boundary.",
    "role_assign_ms": "Time an arm pattern must persist before assigning a role label.",
    "role_hold_ms": "Time to retain a role label after the pattern disappears.",
    "embrace_closed_threshold": "Boundary for the heuristic open/closed embrace label.",
    "embrace_hysteresis_seconds": "Hysteresis duration used to stabilize embrace classification.",
    "closed_embrace_min_knee_bend": "Minimum knee flexion used by the straight-knee cue in closed embrace.",
    "knee_straight_duration": "Duration required before showing the knee cue.",
    "knee_recovery_margin": "Degrees above closed_embrace_min_knee_bend required for fallback knee recovery hysteresis.",
    "knee_visibility_floor": "Helper landmark visibility floor checked for hip, knee, and ankle in knee flexion geometry.",
    "knee_min_segment_px": "Minimum pixel length required for both hip-to-knee and ankle-to-knee leg segments.",
    "knee_tiebreak_other_max_vis": "Other-leg visibility ceiling below which the preferred-side knee tie-break can trigger.",
    "knee_tiebreak_min_advantage": "Minimum visibility advantage required for the preferred leg to win the knee tie-break.",
    "axis_curve_entry": "Posterior hip offset that starts a pending axis event.",
    "axis_curve_clear": "Lower offset at which a confirmed axis event can clear.",
    "axis_curve_duration": "Time above the entry threshold required to confirm an axis event.",
    "axis_curve_clear_duration": "Time below the clear threshold required to close an axis event.",
    "axis_neutral_offset": "Fallback neutral offset when no captured reference is available.",
    "slow_tango_max_bpm": "Tempo ceiling for the slow-tango ankle-lift cue.",
    "ankle_high_lift_ratio": "Ankle lift threshold relative to estimated leg length.",
    "max_consecutive_high_ankle_steps": "Consecutive high-lift steps required to trigger the cue.",
    "tango_bpm": (
        "Fallback tempo when analysis has no soundtrack BPM. "
        "Compared with the separate slow-tango ceiling for the ankle-lift cue."
    ),
    "replay_step_history_seconds": (
        "Replay media-time window (0.5 to 30 seconds) over which observed step footprints fade out."
    ),
    "replay_joint_interpolation": (
        "Display-only Review joint gap filling: linear by default, or quadratic with a linear fallback. "
        "Raw calibration/scoring evidence is unchanged."
    ),
    "replay_joint_max_gap": (
        "Maximum bracketed joint gap filled in Review, in seconds; default 3.0, and 0 disables filling. "
        "Not the measurement tracking gap limit."
    ),
    "replay_recovery_auto": (
        "Run display-only backward pose recovery automatically on short lower-limb dropouts during analysis."
    ),
    "replay_recovery_max_gap_s": (
        "Maximum lower-limb dropout duration in seconds eligible for backward tracking recovery."
    ),
    "replay_recovery_context_s": (
        "Seconds of context included before and after each gap window for backward pose recovery."
    ),
    "replay_pivot_interpolation_max_gap_s": (
        "Maximum leg-joint interpolation gap in seconds allowed during a pivot when forward and backward passes miss."
    ),
    "replay_show_recovered_joints": (
        "Display recovered backward-pass and interpolated joints on Review skeletons without changing measurements."
    ),
    "replay_recovery_budget_s": (
        "Maximum cumulative window seconds recovered automatically per clip."
    ),
    "dancer_analysis.default_dancer_mode": (
        "Default dancer mode for new runs: auto (detect from opening window), single, or couple."
    ),
    "dancer_analysis.auto_detection_window_s": (
        "Duration in seconds from clip_in inspected when resolving Auto dancer mode."
    ),
    "dancer_analysis.primary_coverage_min": (
        "Minimum fraction of the opening window with eligible observations for the primary subject."
    ),
    "dancer_analysis.primary_duration_min_s": (
        "Minimum eligible primary-subject seconds required in the opening window."
    ),
    "dancer_analysis.second_joint_coverage_min": (
        "Minimum fraction of the opening window where both tracked dancers are jointly eligible for Couple."
    ),
    "dancer_analysis.second_sustained_min_s": (
        "Minimum continuous paired tracking span (seconds) required to qualify as a Couple in Auto mode."
    ),
    "dancer_analysis.brief_second_max_s": (
        "Maximum total second-person coexistence (seconds) still accepted as Single in Auto mode."
    ),
    "dancer_analysis.together_min_s": (
        "Minimum co-movement span (seconds) where both tracked dancers move coherently in the same region."
    ),
    "dancer_analysis.together_max_hip_separation_torso_lengths": (
        "Maximum inter-dancer hip midpoint distance in pair torso lengths for co-movement qualification."
    ),
    "dancer_analysis.together_direction_cosine_min": (
        "Minimum cosine similarity of hip velocities required for two dancers to qualify as moving together."
    ),
    "dancer_analysis.identity_max_innovation_torso_lengths": (
        "Gating distance ceiling in torso lengths between predicted and observed hip midpoints per step."
    ),
    "dancer_analysis.identity_margin_torso_lengths": (
        "Minimum cost margin in torso lengths required over the second-best A/B assignment; otherwise unknown."
    ),
    "dancer_analysis.max_observation_gap_s": (
        "Maximum observation gap in seconds before continuous tracking spans and velocity windows break."
    ),
    "dancer_analysis.display_interpolation_max_gap_s": (
        "Maximum per-track gap in seconds for linear display-only 2D drawing interpolation."
    ),
    "dancer_analysis.velocity_filter_window_s": (
        "Half-window in seconds for centred linear-regression travel velocity estimation."
    ),
    "dancer_analysis.persistence_s": (
        "Minimum duration in seconds a start, slowing, or settled speed condition must persist."
    ),
    "dancer_analysis.pairing_window_s": (
        "Maximum interval in seconds for pairing Dancer A and Dancer B coordination events in the same phase."
    ),
    "dancer_analysis.start_speed": (
        "Travel speed threshold in normalized torso lengths per second for start or restart onset."
    ),
    "dancer_analysis.stop_speed": (
        "Travel speed ceiling in normalized torso lengths per second below which a dancer is settled."
    ),
    "dancer_analysis.slowing_fraction": (
        "Fractional speed drop from steady-phase median required to trigger a slowing event."
    ),
    "dancer_analysis.transition_min_coverage": (
        "Minimum measured pair coverage required in a transition window to summarize smoothness."
    ),
    "dancer_analysis.secondary_speed_rise": (
        "Minimum rebound speed rise in pair torso lengths per second during slowing to count a secondary peak."
    ),
    "dancer_analysis.analyze_soundtrack": (
        "Extract optional soundtrack beat candidates for step timing alignment by default."
    ),
    "dancer_analysis.auto_turn_suggestions": (
        "Generate optional estimated turn marker suggestions when supported by torso rotation cues."
    ),
    "ollama_base_url": "Address of the local or reachable Ollama server.",
    "ollama_model": "Model name installed on that Ollama server.",
    "ollama_timeout": "Maximum wait for a local model request.",
    "ollama_temperature": "Controls response variation.",
    "ollama_num_ctx": "Requested model context capacity in tokens.",
    "ollama_num_predict": "Maximum generated response length in tokens.",
    "ollama_max_report_chars": "Maximum report text sent for local analysis.",
    "ollama_instruction": "Additional instruction accompanying a report.",
}


def set_setting(data: dict, key: str, value: str) -> None:
    field, parse = SETTING_MAP[key]
    try:
        parsed = parse(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{key}: {error}") from error
    target = data
    parts = field.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = parsed


def setting_value(cfg: Config, field: str):
    value = cfg.model_dump()
    for part in field.split("."):
        value = value[part]
    return value


def export_toml(cfg: Config) -> str:
    fields = {field.split(".")[0] for field in SETTING_MAP} | {
        "calibration_defaults",
        "measurement_normalization",
    }
    data = cfg.model_dump(include=fields, exclude_none=True)
    lines: list[str] = []

    def _strip_none(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _strip_none(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [_strip_none(v) for v in value if v is not None]
        return value

    def _is_table_list(value: Any) -> bool:
        return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)

    def _emit_table(values: dict[str, Any], path: tuple[str, ...] = ()) -> None:
        scalars = {
            key: value for key, value in values.items()
            if not isinstance(value, dict) and not _is_table_list(value)
        }
        dicts = {key: value for key, value in values.items() if isinstance(value, dict)}
        table_lists = {key: value for key, value in values.items() if _is_table_list(value)}
        for key, value in scalars.items():
            lines.append(f"{key} = {json.dumps(value)}")
        for key, value in dicts.items():
            child = (*path, key)
            lines.extend(("", f"[{'.'.join(child)}]"))
            _emit_table(value, child)
        for key, items in table_lists.items():
            child = (*path, key)
            for item in items:
                lines.extend(("", f"[[{'.'.join(child)}]]"))
                _emit_table(item, child)

    _emit_table(_strip_none(data))
    return "\n".join(lines) + "\n"


ENV_MAP: dict[str, tuple[str, Callable[[str], float | int | bool]]] = {
    "HEAD_NEUTRAL_OFFSET_DEGREES": ("head_neutral_offset", float),
    "HEAD_DOWN_DURATION_SECONDS": ("head_down_duration", float),
    "HEAD_DOWN_ANGLE_DEGREES": ("head_down_angle", float),
    "BEAT_TRACKING_ENABLED": (
        "beat_tracking",
        lambda s: s.lower() not in ("0", "false", "no", "off"),
    ),
    "TANGO_BPM": ("tango_bpm", float),
    "BEAT_ANALYSIS_SECONDS": ("beat_analysis_seconds", float),
    "ROLE_ASSIGN_MS": ("role_assign_ms", float),
    "ROLE_HOLD_MS": ("role_hold_ms", float),
    "SHOULDER_TENSION_DURATION_SECONDS": ("shoulder_tension_duration", float),
    "SHOULDER_RAISE_THRESHOLD_PCT": ("shoulder_raise_threshold", float),
    "SHOULDER_NEUTRAL_CLEARANCE": ("shoulder_neutral_clearance", float),
    "ALLOW_ALL_ANGLES": (
        "allow_all_angles",
        lambda s: s.lower() not in ("0", "false", "no", "off"),
    ),
    "EMBRACE_CLOSED_THRESHOLD": ("embrace_closed_threshold", float),
    "EMBRACE_HYSTERESIS_SECONDS": ("embrace_hysteresis_seconds", float),
    "CLOSED_EMBRACE_MIN_KNEE_BEND_DEGREES": ("closed_embrace_min_knee_bend", float),
    "KNEE_STRAIGHT_DURATION_SECONDS": ("knee_straight_duration", float),
    "AXIS_CURVE_ENTRY": ("axis_curve_entry", float),
    "AXIS_CURVE_CLEAR": ("axis_curve_clear", float),
    "AXIS_CURVE_DURATION_SECONDS": ("axis_curve_duration", float),
    "AXIS_CURVE_CLEAR_DURATION_SECONDS": ("axis_curve_clear_duration", float),
    "AXIS_NEUTRAL_OFFSET": ("axis_neutral_offset", float),
    "SMOOTHING_SECONDS": ("smoothing", float),
    "OVERLAY_SMOOTHING_SECONDS": ("overlay_smoothing", float),
    "SLOW_TANGO_MAX_BPM": ("slow_tango_max_bpm", float),
    "ANKLE_HIGH_LIFT_RATIO": ("ankle_high_lift_ratio", float),
    "MAX_CONSECUTIVE_HIGH_ANKLE_STEPS": ("max_consecutive_high_ankle_steps", int),
}


def startup_config_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    settings = Path("settings.toml")
    return settings if settings.exists() else None


def load(
    path: Path | None,
    overrides: dict,
    env_path: Path | None = None,
) -> Config:
    data: dict = {}
    if env_path is not None and env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = (part.strip() for part in line.split("=", 1))
            if key in ENV_MAP:
                field_name, parser_fn = ENV_MAP[key]
                data[field_name] = parser_fn(val.strip("'\""))
    if path:
        data.update(tomllib.loads(path.read_text()))
    data.update({k: v for k, v in overrides.items() if v is not None})
    return Config.model_validate(data)
