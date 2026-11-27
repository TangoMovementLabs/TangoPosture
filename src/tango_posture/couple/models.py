"""Domain dataclasses and JSON contracts for Couple analysis and Auto dancer mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..domain import Landmark, PoseOrigin

DancerMode = Literal["auto", "single", "couple"]
ResolvedDancerMode = Literal["single", "couple", "needs_selection"]
SampleQualityStatus = Literal["measured", "estimated", "user_confirmed", "unavailable"]
SummaryEvidenceStatus = Literal[
    "measured",
    "estimated",
    "user_confirmed",
    "mixed",
    "insufficient_evidence",
    "unavailable",
]
PoseOriginPolicy = Literal["measured_only", "allow_propagated_estimates"]
DancerId = Literal["A", "B"]
PivotFootBehavior = Literal["appears_planted", "rotates_in_place", "repositions", "unknown"]
TransitionMarkKind = Literal["planned_pause", "planned_reversal", "settling_step"]
AnnotationEventType = Literal[
    "start",
    "slow",
    "settle",
    "contact",
    "musical_target",
    "speed_cue",
    "turn_start",
    "turn_end",
    "exit",
    "embrace_opening",
]


@dataclass
class DancerAnchor:
    """Explicit user anchor pinning a dancer identity at a frame or timestamp."""

    dancer_id: Literal["A", "B", "single"]
    timestamp: float
    frame_index: int | None = None
    hip_xy: tuple[float, float] | None = None
    pose_index: int | None = None
    display_name: str = ""
    role_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dancer_id": self.dancer_id,
            "timestamp": float(self.timestamp),
            "frame_index": self.frame_index,
            "hip_xy": list(self.hip_xy) if self.hip_xy is not None else None,
            "pose_index": self.pose_index,
            "display_name": self.display_name,
            "role_label": self.role_label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DancerAnchor:
        hip_raw = data.get("hip_xy")
        hip_xy = (
            (float(hip_raw[0]), float(hip_raw[1]))
            if isinstance(hip_raw, (list, tuple)) and len(hip_raw) == 2
            else None
        )
        return cls(
            dancer_id=data["dancer_id"],
            timestamp=float(data.get("timestamp", 0.0)),
            frame_index=int(data["frame_index"]) if data.get("frame_index") is not None else None,
            hip_xy=hip_xy,
            pose_index=int(data["pose_index"]) if data.get("pose_index") is not None else None,
            display_name=str(data.get("display_name", "")),
            role_label=str(data.get("role_label", "")),
        )


@dataclass
class PlannedTransitionMark:
    """User-marked planned pause, reversal, or settling step during transitions."""

    mark_id: str
    timestamp: float
    window_end: float | None = None
    kind: TransitionMarkKind = "planned_reversal"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mark_id": self.mark_id,
            "timestamp": float(self.timestamp),
            "window_end": float(self.window_end) if self.window_end is not None else None,
            "kind": self.kind,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlannedTransitionMark:
        end_raw = data.get("window_end")
        return cls(
            mark_id=str(data["mark_id"]),
            timestamp=float(data["timestamp"]),
            window_end=float(end_raw) if end_raw is not None else None,
            kind=data.get("kind", "planned_reversal"),
            note=str(data.get("note", "")),
        )


@dataclass
class CoupleAnnotationItem:
    """Editable user confirmation, edit, deletion, or manual marker."""

    annotation_id: str
    event_type: AnnotationEventType
    timestamp: float
    end_timestamp: float | None = None
    subject: Literal["A", "B", "pair", "beat"] = "pair"
    status: SampleQualityStatus = "user_confirmed"
    source_candidate_id: str | None = None
    deleted: bool = False
    pivot_behavior: PivotFootBehavior | None = None
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    turn_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "event_type": self.event_type,
            "timestamp": float(self.timestamp),
            "end_timestamp": float(self.end_timestamp) if self.end_timestamp is not None else None,
            "subject": self.subject,
            "status": self.status,
            "source_candidate_id": self.source_candidate_id,
            "deleted": bool(self.deleted),
            "pivot_behavior": self.pivot_behavior,
            "label": self.label,
            "metadata": dict(self.metadata),
            "turn_id": self.turn_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoupleAnnotationItem:
        end_raw = data.get("end_timestamp")
        raw_turn_id = data.get("turn_id")
        turn_id = str(raw_turn_id).strip() if isinstance(raw_turn_id, str) and raw_turn_id.strip() else None
        return cls(
            annotation_id=str(data["annotation_id"]),
            event_type=data["event_type"],
            timestamp=float(data["timestamp"]),
            end_timestamp=float(end_raw) if end_raw is not None else None,
            subject=data.get("subject", "pair"),
            status=data.get("status", "user_confirmed"),
            source_candidate_id=data.get("source_candidate_id"),
            deleted=bool(data.get("deleted", False)),
            pivot_behavior=data.get("pivot_behavior"),
            label=str(data.get("label", "")),
            metadata=dict(data.get("metadata") or {}),
            turn_id=turn_id,
        )


@dataclass
class CoupleAnnotations:
    """User annotations layered on top of automated couple evidence without destroying it."""

    dancer_a_name: str = "Dancer A"
    dancer_b_name: str = "Dancer B"
    dancer_a_role: str = ""
    dancer_b_role: str = ""
    anchors: list[DancerAnchor] = field(default_factory=list)
    travel_direction_points: list[tuple[float, float]] = field(default_factory=list)
    baseline_passages: list[tuple[float, float]] = field(default_factory=list)
    manual_beats: list[float] = field(default_factory=list)
    beat_subdivision: int = 1
    beat_offset_s: float = 0.0
    planned_transitions: list[PlannedTransitionMark] = field(default_factory=list)
    items: list[CoupleAnnotationItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dancer_a_name": self.dancer_a_name,
            "dancer_b_name": self.dancer_b_name,
            "dancer_a_role": self.dancer_a_role,
            "dancer_b_role": self.dancer_b_role,
            "anchors": [a.to_dict() for a in self.anchors],
            "travel_direction_points": [list(p) for p in self.travel_direction_points],
            "baseline_passages": [list(b) for b in self.baseline_passages],
            "manual_beats": [float(t) for t in self.manual_beats],
            "beat_subdivision": int(self.beat_subdivision),
            "beat_offset_s": float(self.beat_offset_s),
            "planned_transitions": [m.to_dict() for m in self.planned_transitions],
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CoupleAnnotations:
        if not data:
            return cls()
        return cls(
            dancer_a_name=str(data.get("dancer_a_name", "Dancer A")),
            dancer_b_name=str(data.get("dancer_b_name", "Dancer B")),
            dancer_a_role=str(data.get("dancer_a_role", "")),
            dancer_b_role=str(data.get("dancer_b_role", "")),
            anchors=[DancerAnchor.from_dict(a) for a in (data.get("anchors") or [])],
            travel_direction_points=[
                (float(p[0]), float(p[1]))
                for p in (data.get("travel_direction_points") or [])
                if isinstance(p, (list, tuple)) and len(p) == 2
            ],
            baseline_passages=[
                (float(b[0]), float(b[1]))
                for b in (data.get("baseline_passages") or [])
                if isinstance(b, (list, tuple)) and len(b) == 2
            ],
            manual_beats=[float(t) for t in (data.get("manual_beats") or [])],
            beat_subdivision=max(1, int(data.get("beat_subdivision", 1))),
            beat_offset_s=float(data.get("beat_offset_s", 0.0)),
            planned_transitions=[
                PlannedTransitionMark.from_dict(m) for m in (data.get("planned_transitions") or [])
            ],
            items=[CoupleAnnotationItem.from_dict(i) for i in (data.get("items") or [])],
        )


@dataclass
class CoupleAnalysisRequest:
    """Explicit analysis request parameters passed into the offline pipeline."""

    requested_mode: DancerMode = "auto"
    forced_override: bool = False
    analysis_region: dict[str, float] | None = None
    analyze_soundtrack: bool | None = None
    auto_turn_suggestions: bool | None = None
    annotations: CoupleAnnotations = field(default_factory=CoupleAnnotations)
    pose_origin_policy: PoseOriginPolicy | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "forced_override": bool(self.forced_override),
            "analysis_region": dict(self.analysis_region) if self.analysis_region else None,
            "analyze_soundtrack": self.analyze_soundtrack,
            "auto_turn_suggestions": self.auto_turn_suggestions,
            "annotations": self.annotations.to_dict(),
            "pose_origin_policy": self.pose_origin_policy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | CoupleAnalysisRequest | None) -> CoupleAnalysisRequest:
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict) or not data:
            return cls()
        raw_mode = str(data.get("requested_mode", "auto")).lower()
        mode: DancerMode = (
            raw_mode if raw_mode in ("auto", "single", "couple") else "auto"  # type: ignore[assignment]
        )
        region_raw = data.get("analysis_region")
        region = (
            {k: float(v) for k, v in region_raw.items()}
            if isinstance(region_raw, dict)
            else None
        )
        raw_policy = data.get("pose_origin_policy")
        policy: PoseOriginPolicy | None = (
            raw_policy
            if raw_policy in ("measured_only", "allow_propagated_estimates")
            else None  # type: ignore[assignment]
        )
        return cls(
            requested_mode=mode,
            forced_override=bool(data.get("forced_override", False)),
            analysis_region=region,
            analyze_soundtrack=(
                bool(data["analyze_soundtrack"])
                if data.get("analyze_soundtrack") is not None
                else None
            ),
            auto_turn_suggestions=(
                bool(data["auto_turn_suggestions"])
                if data.get("auto_turn_suggestions") is not None
                else None
            ),
            annotations=CoupleAnnotations.from_dict(data.get("annotations")),
            pose_origin_policy=policy,
        )


@dataclass
class CandidatePreview:
    """Representative candidate subject in the opening window for manual disambiguation."""

    candidate_id: str
    pose_index: int
    timestamp: float
    frame_index: int
    hip_xy: tuple[float, float]
    shoulder_xy: tuple[float, float] | None
    torso_length_px: float | None
    bbox: tuple[float, float, float, float]
    eligible_duration_s: float
    coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "pose_index": int(self.pose_index),
            "timestamp": float(self.timestamp),
            "frame_index": int(self.frame_index),
            "hip_xy": [float(self.hip_xy[0]), float(self.hip_xy[1])],
            "shoulder_xy": (
                [float(self.shoulder_xy[0]), float(self.shoulder_xy[1])]
                if self.shoulder_xy is not None
                else None
            ),
            "torso_length_px": (
                float(self.torso_length_px) if self.torso_length_px is not None else None
            ),
            "bbox": [float(v) for v in self.bbox],
            "eligible_duration_s": float(self.eligible_duration_s),
            "coverage": float(self.coverage),
        }


@dataclass
class AutoDecisionResult:
    """Full provenance and evidence for Single/Couple/Needs selection decision."""

    detector_version: str
    requested_mode: DancerMode
    resolved_mode: ResolvedDancerMode
    forced_override: bool
    configured_window_s: float
    effective_window_in: float
    effective_window_out: float
    effective_duration_s: float
    primary_eligible_duration_s: float
    primary_coverage: float
    primary_moving_duration_s: float
    second_coexistence_duration_s: float
    second_joint_coverage: float
    max_continuous_paired_span_s: float
    max_co_movement_span_s: float
    reason: str
    explanation: str
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    candidate_previews: list[CandidatePreview] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_version": self.detector_version,
            "requested_mode": self.requested_mode,
            "resolved_mode": self.resolved_mode,
            "forced_override": bool(self.forced_override),
            "configured_window_s": float(self.configured_window_s),
            "effective_window_in": float(self.effective_window_in),
            "effective_window_out": float(self.effective_window_out),
            "effective_duration_s": float(self.effective_duration_s),
            "primary_eligible_duration_s": float(self.primary_eligible_duration_s),
            "primary_coverage": float(self.primary_coverage),
            "primary_moving_duration_s": float(self.primary_moving_duration_s),
            "second_coexistence_duration_s": float(self.second_coexistence_duration_s),
            "second_joint_coverage": float(self.second_joint_coverage),
            "max_continuous_paired_span_s": float(self.max_continuous_paired_span_s),
            "max_co_movement_span_s": float(self.max_co_movement_span_s),
            "reason": self.reason,
            "explanation": self.explanation,
            "diagnostics": [dict(d) for d in self.diagnostics],
            "candidate_previews": [c.to_dict() for c in self.candidate_previews],
        }


@dataclass
class TrackedDancerSample:
    """Single-frame tracked state for Dancer A or Dancer B."""

    timestamp: float
    frame_index: int
    pose_index: int | None
    raw_pose: dict[str, Landmark] | None
    display_pose: dict[str, Landmark] | None
    world_pose: dict[str, Landmark] | None
    hip_px: tuple[float, float] | None
    shoulder_px: tuple[float, float] | None
    torso_length_px: float | None
    torso_eligible: bool
    hip_eligible: bool
    head_eligible: bool
    view_usable: bool
    interpolated_display: bool = False
    innovation_torso_lengths: float | None = None
    assignment_margin_torso_lengths: float | None = None
    quality_status: SampleQualityStatus = "unavailable"
    pose_origin: PoseOrigin = "detected"
    landmark_origins: dict[str, PoseOrigin] = field(default_factory=dict)


@dataclass
class PairedFrameSample:
    """Synchronized frame observation pairing Track A and Track B."""

    timestamp: float
    frame_index: int
    width: int
    height: int
    dancer_a: TrackedDancerSample | None
    dancer_b: TrackedDancerSample | None
    pair_torso_eligible: bool
    pair_hip_eligible: bool
    shot_cut: bool = False
    identity_break: bool = False


@dataclass
class CoupleEvent:
    """Discrete timeline event (measured, estimated, user-confirmed, or unavailable)."""

    event_id: str
    event_type: str
    subject: str
    timestamp: float
    end_timestamp: float | None = None
    bracket_start: float | None = None
    bracket_end: float | None = None
    value: float | None = None
    unit: str = ""
    status: SampleQualityStatus = "measured"
    paired_event_id: str | None = None
    lag_ms: float | None = None
    beat_offset_ms: float | None = None
    beat_fraction: float | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    turn_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoupleSummaryValue:
    """Aggregated Couple summary measure with explicit status and coverage."""

    measure: str
    subject: str
    value: float | None
    iqr: float | None
    count: int
    unit: str
    status: SummaryEvidenceStatus
    coverage: float
    reason: str = ""
    timestamp_or_bracket: str = ""
    status_counts: dict[str, int] = field(default_factory=dict)
    origin_counts: dict[str, int] = field(default_factory=dict)
    eligible_count: int | None = None
    coverage_denominator: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoupleAnalysisResult:
    """Versioned Couple analysis result persisted in analytics_XX.json."""

    run_id: str
    created_at: str
    video_sha256: str
    segment_index: int
    clip_in: float
    clip_out: float
    model_name: str
    config_hash: str
    requested_mode: DancerMode
    resolved_mode: ResolvedDancerMode
    auto_decision: AutoDecisionResult
    request: CoupleAnalysisRequest
    fixed_scale_a_px: float | None
    fixed_scale_b_px: float | None
    fixed_pair_scale_px: float | None
    travel_axis_unit: tuple[float, float]
    travel_axis_source: Literal["user_defined", "inferred_dominant"]
    coverage: dict[str, Any]
    series: dict[str, list[dict[str, Any]]]
    automated_candidates: list[dict[str, Any]]
    annotations: dict[str, Any]
    events: list[dict[str, Any]]
    summaries: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    frame_overlays: list[dict[str, Any]]
    schema_version: int = 1
    detector_version: str = "couple_engine_v1"
    revision_id: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "detector_version": self.detector_version,
            "run_id": self.run_id,
            "revision_id": int(self.revision_id),
            "created_at": self.created_at,
            "video_sha256": self.video_sha256,
            "segment_index": int(self.segment_index),
            "clip_in": float(self.clip_in),
            "clip_out": float(self.clip_out),
            "model_name": self.model_name,
            "config_hash": self.config_hash,
            "requested_mode": self.requested_mode,
            "resolved_mode": self.resolved_mode,
            "auto_decision": self.auto_decision.to_dict(),
            "request": self.request.to_dict(),
            "fixed_scale_a_px": (
                float(self.fixed_scale_a_px) if self.fixed_scale_a_px is not None else None
            ),
            "fixed_scale_b_px": (
                float(self.fixed_scale_b_px) if self.fixed_scale_b_px is not None else None
            ),
            "fixed_pair_scale_px": (
                float(self.fixed_pair_scale_px) if self.fixed_pair_scale_px is not None else None
            ),
            "travel_axis_unit": [
                float(self.travel_axis_unit[0]),
                float(self.travel_axis_unit[1]),
            ],
            "travel_axis_source": self.travel_axis_source,
            "coverage": dict(self.coverage),
            "series": dict(self.series),
            "automated_candidates": [dict(c) for c in self.automated_candidates],
            "annotations": dict(self.annotations),
            "events": [dict(e) for e in self.events],
            "summaries": dict(self.summaries),
            "diagnostics": [dict(d) for d in self.diagnostics],
            "frame_overlays": [dict(f) for f in self.frame_overlays],
        }
