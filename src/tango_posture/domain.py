"""Records shared by the pure pipeline, adapters and persistence."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol
from uuid import uuid4

METRICS = ("head_forward", "head_inclination", "trunk_inclination")
METRIC_VERSION = "1.0"
UNITS = dict(zip(METRICS, ("torso_lengths", "degrees", "degrees")))

PoseOrigin = Literal["detected", "crop_detected", "propagated", "display_interpolated"]
VALID_POSE_ORIGINS: frozenset[str] = frozenset(
    {"detected", "crop_detected", "propagated", "display_interpolated"}
)


def normalize_pose_origin(value: Any, default: PoseOrigin = "detected") -> PoseOrigin:
    """Return a valid ``PoseOrigin`` string, falling back cleanly for legacy values."""
    if isinstance(value, str) and value in VALID_POSE_ORIGINS:
        return value  # type: ignore[return-value]
    return default


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float | None = None
    presence: float | None = None
    z: float | None = None
    origin: PoseOrigin = "detected"


def infer_pose_origin(
    pose: Mapping[str, Landmark] | None,
    default: PoseOrigin = "detected",
) -> PoseOrigin:
    """Infer pose-level origin from constituent landmarks without mutating geometry."""
    if not pose:
        return default
    origins = {
        lm.origin
        for lm in pose.values()
        if getattr(lm, "origin", None) in VALID_POSE_ORIGINS
    }
    if not origins:
        return default
    if origins == {"display_interpolated"}:
        return "display_interpolated"
    for cand in ("propagated", "crop_detected", "detected", "display_interpolated"):
        if cand in origins:
            return cand  # type: ignore[return-value]
    return default


def tag_pose_origin(pose: Mapping[str, Landmark], origin: PoseOrigin) -> dict[str, Landmark]:
    """Return a shallow copy of ``pose`` with every landmark stamped with ``origin``."""
    return {name: replace(lm, origin=origin) for name, lm in pose.items()}


@dataclass(frozen=True)
class FrameObservation:
    timestamp: float
    width: int
    height: int
    poses: list[dict[str, Landmark]]
    world_poses: list[dict[str, Landmark]] = field(default_factory=list)
    pose_origins: list[PoseOrigin] = field(default_factory=list)

    def pose_origin_at(self, index: int) -> PoseOrigin:
        """Return explicit pose origin at ``index``, or infer from landmarks if omitted."""
        if 0 <= index < len(self.pose_origins):
            return normalize_pose_origin(self.pose_origins[index])
        if 0 <= index < len(self.poses):
            return infer_pose_origin(self.poses[index])
        return "detected"


@dataclass(frozen=True)
class QualityResult:
    valid: bool
    reason: str = ""
    view_width: float | None = None


@dataclass(frozen=True)
class MetricObservation:
    timestamp: float
    values: dict[str, float | None]
    quality: dict[str, QualityResult]
    view_width: float | None = None


@dataclass
class Calibration:
    id: str
    created_at_utc: str
    name: str
    profile: dict[str, Any]
    baseline: dict[str, float]
    quality: dict[str, Any]
    model_hash: str
    metric_version: str = METRIC_VERSION
    review: dict[str, Any] = field(default_factory=lambda: {"label": "self-selected"})


@dataclass
class DeviationEvent:
    metric: str
    onset: float
    detection: float
    peak_deviation: float
    end: float | None = None
    end_reason: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class SessionSummary:
    active_duration: float
    pause_duration: float
    metrics: dict[str, dict[str, Any]]
    schema_version: int = 1


class PoseEstimator(Protocol):
    def estimate(self, image: Any, timestamp: float) -> FrameObservation: ...
    def close(self) -> None: ...
