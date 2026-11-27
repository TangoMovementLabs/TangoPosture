"""Pure Pydantic schema models for Review measurement normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_METRIC_KEYS: frozenset[str] = frozenset({
    "head_pitch",
    "head_forward",
    "head_inclination",
    "shoulder_clearance",
    "hip_offset",
    "torso_axis_tilt",
    "trunk_inclination",
    "knee_flexion",
})


def require_finite(value: Any, name: str = "Value") -> float:
    """Validate and convert `value` to a finite float."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def clean_toml_dict(value: Any) -> Any:
    """Recursively drop None values so dictionaries remain TOML-serializable."""
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            cleaned_item = clean_toml_dict(item)
            if cleaned_item is not None:
                cleaned[str(key)] = cleaned_item
        return cleaned
    if isinstance(value, list):
        return [clean_toml_dict(item) for item in value if item is not None]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite floating point value is not allowed")
    return value


class HoldWindow(BaseModel):
    """Time interval `[start, end)` in seconds used for hold normalization."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, from_attributes=True,
    )

    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> HoldWindow:
        if self.end <= self.start:
            raise ValueError("Hold window end must exceed start")
        return self


class HoldWindowStats(BaseModel):
    """Summary statistics computed over real frames inside a hold window."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    start_seconds: float
    end_seconds: float
    valid_frames: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    median: float
    mad: float | None = None
    q10: float | None = None
    q90: float | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> HoldWindowStats:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Hold window end must exceed start")
        if self.valid_frames > self.total_frames:
            raise ValueError("Valid frame count cannot exceed total frames")
        return self


class HoldSourceIdentity(BaseModel):
    """Content-bound identity metadata for a reference or practice recording."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    archive_path: str | None = None
    archive_name: str | None = None
    segment_index: int = Field(default=0, ge=0)
    raw_media_sha256: str | None = None
    media_sha256: str | None = None
    capture_identity: dict[str, Any] = Field(default_factory=dict)
    settings_fingerprint: str | None = None
    frames_fingerprint: str | None = None
    source_label: str | None = None

    @field_validator("capture_identity", mode="before")
    @classmethod
    def _strip_none_capture_values(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("capture_identity must be a mapping")
        return clean_toml_dict(value)


class MetricNormalizationRule(BaseModel):
    """Additive normalization rule for a single calibration metric."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    method: Literal["manual", "matched_holds"] = "manual"
    offset: float = 0.0
    show_calculation_in_review: bool = True
    max_verification_error: float | None = Field(default=None, gt=0.0)
    fit_reference_window: HoldWindow | None = None
    fit_practice_window: HoldWindow | None = None
    check_reference_window: HoldWindow | None = None
    check_practice_window: HoldWindow | None = None
    fit_reference_stats: HoldWindowStats | None = None
    fit_practice_stats: HoldWindowStats | None = None
    check_reference_stats: HoldWindowStats | None = None
    check_practice_stats: HoldWindowStats | None = None
    verification_error: float | None = Field(default=None, ge=0.0)
    verified: bool = False
    status_message: str = ""

    @field_validator("offset", mode="before")
    @classmethod
    def _validate_offset(cls, value: Any) -> float:
        return require_finite(value, "Normalization offset")


class NormalizationPairRule(BaseModel):
    """Normalization configuration bound to one calibration profile and practice recording."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(min_length=1)
    name: str = ""
    enabled: bool = True
    calibration_id: str = Field(min_length=1)
    calibration_name: str = ""
    profile_content_digest: str = Field(min_length=1)
    reference_source: HoldSourceIdentity = Field(default_factory=HoldSourceIdentity)
    practice_source: HoldSourceIdentity = Field(default_factory=HoldSourceIdentity)
    metrics: dict[str, MetricNormalizationRule] = Field(default_factory=dict)
    created_at_utc: str = ""
    updated_at_utc: str = ""

    @field_validator("metrics")
    @classmethod
    def _validate_metric_keys(
        cls, value: dict[str, MetricNormalizationRule],
    ) -> dict[str, MetricNormalizationRule]:
        unknown = set(value) - _METRIC_KEYS
        if unknown:
            raise ValueError(f"Unknown calibration metrics in normalization rule: {sorted(unknown)}")
        return value


class MeasurementNormalizationConfig(BaseModel):
    """Top-level configuration section for Review measurement normalization."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    min_hold_samples: int = Field(default=8, ge=1)
    min_hold_coverage: float = Field(default=0.6, ge=0.0, le=1.0)
    default_max_verification_error: float = Field(default=1.0, gt=0.0)
    pairs: list[NormalizationPairRule] = Field(default_factory=list)

    @field_validator("pairs")
    @classmethod
    def _validate_unique_pair_ids(
        cls, value: list[NormalizationPairRule],
    ) -> list[NormalizationPairRule]:
        seen: set[str] = set()
        for pair in value:
            if pair.id in seen:
                raise ValueError(f"Duplicate normalization pair id: {pair.id}")
            seen.add(pair.id)
        return value

    def pair_by_id(self, pair_id: str) -> NormalizationPairRule | None:
        """Return the configured pair rule matching `pair_id`, if present."""
        for pair in self.pairs:
            if pair.id == pair_id:
                return pair
        return None
