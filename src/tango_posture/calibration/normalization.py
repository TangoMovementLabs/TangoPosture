"""Per-metric measurement normalization models, hold verification, and applicability checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence
from zipfile import BadZipFile, ZipFile

import numpy as np

from ..domain import FrameObservation, Landmark
from ..geometry import wrapped
from ..normalization_models import (
    HoldSourceIdentity,
    HoldWindow,
    HoldWindowStats,
    MeasurementNormalizationConfig,
    MetricNormalizationRule,
    NormalizationPairRule,
    clean_toml_dict as _clean_toml_dict,
    require_finite as _require_finite,
)

if TYPE_CHECKING:
    from ..config import Config, RangeDefault
    from .models import ApprovedRange, Passage, RangeProfile


_CYCLIC_METRICS: frozenset[str] = frozenset({"head_inclination", "trunk_inclination"})


NORMALIZATION_VERSION = "measurement-normalization/1.0"

MEASUREMENT_AFFECTING_FIELDS: tuple[str, ...] = (
    "confidence",
    "side",
    "facing",
    "allow_all_angles",
    "max_view_width",
    "view_change_tolerance",
    "min_torso",
    "min_head",
    "shoulder_visibility_floor",
    "shoulder_ear_hip_scale_factor",
    "shoulder_min_ear_hip_px",
    "shoulder_nose_ear_scale_factor",
    "shoulder_ear_ear_scale_factor",
    "shoulder_width_scale_factor",
    "shoulder_min_ref_scale_px",
    "shoulder_baseline_clamp_floor",
    "knee_visibility_floor",
    "knee_min_segment_px",
    "knee_tiebreak_other_max_vis",
    "knee_tiebreak_min_advantage",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PairApplicability:
    """Applicability status of a normalization pair against current data."""

    eligible: bool
    status: Literal["applied", "disabled", "needs_recheck", "no_pair"]
    reason: str = ""
    pair: NormalizationPairRule | None = None


@dataclass(frozen=True)
class MatchedHoldResult:
    """Outcome of computing and checking a matched-hold offset for one metric."""

    rule: MetricNormalizationRule
    eligible: bool
    reason: str = ""


@dataclass(frozen=True)
class NormalizedMetricValue:
    """Single-frame normalization outcome for one metric."""

    observed_raw: float | None
    normalization_offset: float | None
    reference_scale_raw: float | None
    evaluator_input: float | None
    normalization_status: str
    status_detail: str = ""


def make_pair_rule_id(calibration_id: str, raw_media_sha256: str | None, fallback_seed: str = "") -> str:
    """Build a deterministic short identifier for a calibration + practice recording pair."""
    payload = json.dumps(
        {
            "calibration_id": calibration_id,
            "raw_media_sha256": raw_media_sha256 or "",
            "fallback": "" if raw_media_sha256 else fallback_seed,
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"pair-{digest}"


def capture_identity_dict(capture: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return canonical non-null capture orientation/size fields for identity comparisons."""
    if not isinstance(capture, Mapping):
        return {}
    allowed = ("side", "facing", "width", "height", "aspect_ratio", "capture_profile")
    raw = {key: capture[key] for key in allowed if key in capture and capture[key] is not None}
    return _clean_toml_dict(raw)


def profile_content_digest(profile: RangeProfile) -> str:
    """Hash approved ranges and measurement contract without archive filenames or names."""
    metrics_payload = {
        key: {
            "lower": round(float(target.lower), 9),
            "upper": round(float(target.upper), 9),
            "unit": target.unit,
            "view": target.view,
            "approved": bool(target.approved),
        }
        for key, target in sorted(profile.metrics.items())
    }
    payload = {
        "id": profile.id,
        "measurement_version": profile.measurement_version,
        "model_hash": profile.model_hash,
        "capture_profile": capture_identity_dict(profile.capture_profile),
        "metrics": metrics_payload,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def settings_fingerprint(cfg: Config) -> str:
    """Hash config fields that affect raw pose/metric computation."""
    payload = {field: getattr(cfg, field) for field in MEASUREMENT_AFFECTING_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return round(numeric, 6) if math.isfinite(numeric) else None


def frames_fingerprint(observations: Sequence[FrameObservation]) -> str:
    """Hash raw frame dimensions, timestamps, and pose coordinates in order."""
    hasher = hashlib.sha256()
    for obs in observations:
        poses_payload = []
        for pose in obs.poses:
            poses_payload.append({
                name: [
                    _round_optional(lm.x),
                    _round_optional(lm.y),
                    _round_optional(lm.z),
                    _round_optional(lm.visibility),
                    _round_optional(lm.presence),
                ]
                for name, lm in sorted(pose.items())
            })
        record = json.dumps(
            [
                _round_optional(obs.timestamp),
                int(obs.width),
                int(obs.height),
                poses_payload,
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        hasher.update(record)
        hasher.update(b"\n")
    return hasher.hexdigest()


def _parse_raw_pose(raw_pose: Any) -> dict[str, Landmark]:
    if not isinstance(raw_pose, Mapping) or not raw_pose:
        return {}
    pose: dict[str, Landmark] = {}
    for name, coords in raw_pose.items():
        if not isinstance(coords, (list, tuple)) or len(coords) < 3:
            continue
        if coords[0] is None or coords[1] is None:
            continue
        pose[str(name)] = Landmark(
            x=float(coords[0]),
            y=float(coords[1]),
            z=float(coords[2]) if coords[2] is not None else None,
            visibility=float(coords[3]) if len(coords) >= 4 and coords[3] is not None else None,
            presence=float(coords[4]) if len(coords) >= 5 and coords[4] is not None else None,
        )
    return pose


def extract_coach_observations(data: Mapping[str, Any]) -> list[FrameObservation]:
    """Extract ordered raw FrameObservations from cached `coach_data`, ignoring display poses."""
    frames = data.get("frames")
    if not isinstance(frames, Mapping) or not frames:
        return []
    width = data.get("width")
    height = data.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        capture = data.get("capture_profile") if isinstance(data.get("capture_profile"), Mapping) else {}
        width = capture.get("width")
        height = capture.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        return []
    observations: list[FrameObservation] = []
    for index in sorted(frames, key=int):
        entry = frames[index]
        if not isinstance(entry, Mapping):
            continue
        ts = float(entry.get("timestamp", 0.0))
        if entry.get("interpolated") is True:
            observations.append(FrameObservation(ts, width, height, []))
            continue
        raw_poses = entry.get("measurement_poses")
        if raw_poses is None and "display_poses" not in entry:
            raw_poses = entry.get("poses")
        if raw_poses is None and "pose" in entry and "display_poses" not in entry:
            raw_poses = [entry.get("pose")]
        poses: list[dict[str, Landmark]] = []
        if isinstance(raw_poses, list):
            for raw_pose in raw_poses:
                parsed = _parse_raw_pose(raw_pose)
                if parsed:
                    poses.append(parsed)
        observations.append(FrameObservation(ts, width, height, poses))
    return observations


def _read_selected_dancer(path: Path) -> int | None:
    try:
        if path.is_dir():
            ref_file = path / "reference.json"
            if not ref_file.is_file():
                return None
            meta = json.loads(ref_file.read_text(encoding="utf-8"))
        elif path.is_file():
            with ZipFile(path) as archive:
                if "reference.json" not in archive.namelist():
                    return None
                meta = json.loads(archive.read("reference.json"))
        else:
            return None
        if isinstance(meta, dict) and type(meta.get("selected_dancer")) is int:
            return int(meta["selected_dancer"])
    except (OSError, ValueError, TypeError, BadZipFile):
        return None
    return None


def resolve_raw_media_sha256(
    archive_path: Path,
    segment_index: int = 0,
    analytics: Mapping[str, Any] | None = None,
) -> str | None:
    """Hash raw segment video media inside `.tp`/`.tpr`, or recover from provenance metadata."""
    from ..archive import _archive_lock, _stream_member_sha256, file_sha256, read_manifest

    if archive_path.is_file() and archive_path.suffix.lower() not in {".tp", ".tpr", ".tpc"}:
        return file_sha256(archive_path)
    try:
        if archive_path.exists():
            if archive_path.is_dir():
                video_member = f"video_{segment_index:02d}.mp4"
                candidate = archive_path / video_member
                if candidate.is_file():
                    return file_sha256(candidate)
            else:
                manifest = read_manifest(archive_path)
                segments = manifest.get("segments") or []
                if 0 <= segment_index < len(segments):
                    video_name = segments[segment_index].get("video")
                    if video_name:
                        with _archive_lock(archive_path), ZipFile(archive_path) as zf:
                            if video_name in zf.namelist():
                                return _stream_member_sha256(zf, video_name)
    except (OSError, ValueError, BadZipFile):
        pass
    if isinstance(analytics, Mapping):
        for container in (analytics, analytics.get("coach_data"), analytics.get("source_provenance")):
            if isinstance(container, Mapping):
                for key in ("raw_media_sha256", "media_sha256", "video_sha256"):
                    val = container.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
    return None


def resolve_profile_reference_source(
    profile: RangeProfile,
    profile_path: Path,
    cfg: Config,
    *,
    override_path: Path | None = None,
) -> Path:
    """Find the source `.tpr` archive for `profile`, honoring explicit user override when valid."""
    from ..archive import file_sha256, read_manifest
    from ..references.archives import reference_directory
    from .evidence import cached_evidence

    if override_path is not None:
        if not override_path.is_file():
            raise FileNotFoundError(f"Reference source not found: {override_path}")
        read_manifest(override_path)
        ev = cached_evidence(override_path, 0)
        if ev.model_hash != profile.model_hash:
            raise ValueError("Selected reference source uses a different measurement model")
        if any(
            ev.capture_profile.get(k) != profile.capture_profile.get(k)
            for k in ("side", "facing")
        ):
            raise ValueError("Selected reference source side/facing does not match calibration profile")
        return override_path

    archive_name = str(profile.source.get("archive") or "").strip()
    archive_hash = str(profile.source.get("archive_sha256") or "").strip()
    search_dirs: list[Path] = []
    for directory in (reference_directory(cfg.data_dir), profile_path.parent.parent / "references"):
        if directory not in search_dirs:
            search_dirs.append(directory)
    if archive_name:
        for directory in search_dirs:
            candidate = directory / archive_name
            if candidate.is_file() and (not archive_hash or file_sha256(candidate) == archive_hash):
                return candidate
    if archive_hash:
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            for candidate in sorted(directory.glob("*.tpr")):
                if file_sha256(candidate) == archive_hash:
                    return candidate
    raise FileNotFoundError(
        f"Source reference '{archive_name or 'unknown'}' was not found in "
        f"{reference_directory(cfg.data_dir)}. Use 'Locate reference source…' or keep the .tpr archive in Library."
    )


def select_archive_frames(
    archive_path: Path,
    segment_index: int,
    cfg: Config,
    *,
    prefer_coach_frames: bool = False,
) -> tuple[list[FrameObservation], dict[str, Any]]:
    """Load real, non-interpolated FrameObservations and capture/analytics metadata for a segment."""
    from ..archive import read_analytics
    from .evidence import cached_evidence, dancer_frames

    analytics = read_analytics(archive_path, segment_index) or {}
    coach = analytics.get("coach_data")
    if prefer_coach_frames and isinstance(coach, Mapping) and coach.get("frames"):
        frames = extract_coach_observations(coach)
        capture = analytics.get("capture_profile") or coach.get("capture_profile") or {}
        if frames:
            return frames, {
                "analytics": analytics,
                "capture_profile": capture,
                "raw_media_sha256": resolve_raw_media_sha256(archive_path, segment_index, analytics),
                "model_hash": analytics.get("model_hash") or coach.get("model_hash"),
            }
    evidence = cached_evidence(archive_path, segment_index)
    selected_dancer = _read_selected_dancer(archive_path)
    multi_dancer = any(len(frame.poses) > 1 for frame in evidence.frames)
    if selected_dancer is not None or multi_dancer:
        frames = dancer_frames(evidence, selected_dancer if selected_dancer is not None else 0, cfg.confidence)
    else:
        frames = evidence.frames
    return frames, {
        "analytics": analytics,
        "capture_profile": evidence.capture_profile,
        "raw_media_sha256": resolve_raw_media_sha256(archive_path, segment_index, analytics),
        "model_hash": evidence.model_hash,
    }


def build_source_identity(
    *,
    archive_path: Path | None,
    segment_index: int,
    cfg: Config,
    observations: Sequence[FrameObservation],
    capture_profile: Mapping[str, Any] | None,
    raw_media_sha256: str | None,
    source_label: str | None = None,
) -> HoldSourceIdentity:
    """Construct a complete content-bound HoldSourceIdentity for persistence and applicability checks."""
    return HoldSourceIdentity(
        archive_path=str(archive_path) if archive_path is not None else None,
        archive_name=archive_path.name if archive_path is not None else None,
        segment_index=segment_index,
        raw_media_sha256=raw_media_sha256,
        media_sha256=raw_media_sha256,
        capture_identity=capture_identity_dict(capture_profile),
        settings_fingerprint=settings_fingerprint(cfg),
        frames_fingerprint=frames_fingerprint(observations),
        source_label=source_label,
    )


def windows_disjoint(a: HoldWindow | Passage, b: HoldWindow | Passage) -> bool:
    """Return True when hold windows `a` and `b` do not overlap (`[start, end)`)."""
    return a.end <= b.start + 1e-9 or b.end <= a.start + 1e-9


def _summarize_values(values: list[float], metric: str) -> tuple[float, float | None, float | None, float | None]:
    arr = np.asarray(values, dtype=float)
    if metric in _CYCLIC_METRICS:
        radians = np.deg2rad(arr)
        mean_angle = math.degrees(math.atan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians)))))
        deltas = np.asarray([wrapped(v - mean_angle) for v in values], dtype=float)
        q10_d, med_d, q90_d = map(float, np.quantile(deltas, [0.1, 0.5, 0.9]))
        median = float(wrapped(mean_angle + med_d))
        mad = float(np.median(np.abs(deltas - med_d)))
        return median, mad, float(wrapped(mean_angle + q10_d)), float(wrapped(mean_angle + q90_d))
    q10, median, q90 = map(float, np.quantile(arr, [0.1, 0.5, 0.9]))
    mad = float(np.median(np.abs(arr - median)))
    return median, mad, q10, q90


def measure_hold_window(
    frames: Sequence[FrameObservation],
    cfg: Config,
    metric: str,
    target: ApprovedRange | RangeDefault,
    window: HoldWindow | Passage,
    *,
    min_samples: int = 8,
    min_coverage: float = 0.6,
) -> tuple[HoldWindowStats | None, str]:
    """Measure one hold window on real frames without interpolation."""
    from .measurements import measure_targets

    selected = [obs for obs in frames if window.start <= obs.timestamp < window.end]
    total_frames = len(selected)
    if total_frames == 0:
        return None, f"No frames found inside {window.start:.2f}s–{window.end:.2f}s"
    valid_values: list[float] = []
    targets = {metric: target}
    for obs in selected:
        measured = measure_targets(obs, cfg, targets)
        quality = measured.quality.get(metric)
        val = measured.values.get(metric)
        if quality is not None and quality.valid and val is not None and math.isfinite(val):
            valid_values.append(float(val))
    valid_frames = len(valid_values)
    coverage = valid_frames / total_frames
    if valid_frames == 0:
        return (
            HoldWindowStats(
                start_seconds=window.start,
                end_seconds=window.end,
                valid_frames=0,
                total_frames=total_frames,
                coverage_ratio=0.0,
                median=0.0,
            ),
            f"0/{total_frames} valid frames in {window.start:.2f}s–{window.end:.2f}s",
        )
    median, mad, q10, q90 = _summarize_values(valid_values, metric)
    stats = HoldWindowStats(
        start_seconds=window.start,
        end_seconds=window.end,
        valid_frames=valid_frames,
        total_frames=total_frames,
        coverage_ratio=coverage,
        median=median,
        mad=mad,
        q10=q10,
        q90=q90,
    )
    if valid_frames < min_samples:
        return stats, f"Only {valid_frames} valid frames (minimum {min_samples})"
    if coverage + 1e-9 < min_coverage:
        return stats, f"Hold coverage {coverage * 100:.0f}% is below {min_coverage * 100:.0f}%"
    return stats, ""


def metric_offset_between(metric: str, reference_value: float, practice_value: float) -> float:
    """Compute signed offset `b` mapping `practice_value + b -> reference_value`."""
    diff = float(reference_value) - float(practice_value)
    return float(wrapped(diff)) if metric in _CYCLIC_METRICS else diff


def metric_error(metric: str, a: float, b: float) -> float:
    """Distance between two metric values on `metric`'s domain."""
    diff = float(a) - float(b)
    return abs(float(wrapped(diff))) if metric in _CYCLIC_METRICS else abs(diff)


def in_physical_domain(metric: str, value: float) -> bool:
    """Return True if `value` obeys physical domain bounds for `metric`."""
    if not math.isfinite(value):
        return False
    if metric == "knee_flexion":
        return 0.0 <= value <= 180.0
    if metric == "shoulder_clearance":
        return value >= 0.0
    if metric in ("head_pitch", "torso_axis_tilt"):
        return -180.0 <= value <= 180.0
    return True


def apply_metric_offset(metric: str, observed_raw: float, offset: float) -> float:
    """Apply additive adjustment `offset` cleanly for linear or cyclic metrics."""
    raw = _require_finite(observed_raw, "Observed raw value")
    adj = _require_finite(offset, "Offset")
    if metric in _CYCLIC_METRICS:
        return float(wrapped(raw + adj))
    return raw + adj


def evaluate_matched_holds(
    *,
    metric: str,
    target: ApprovedRange | RangeDefault,
    rule: MetricNormalizationRule,
    reference_frames: Sequence[FrameObservation],
    practice_frames: Sequence[FrameObservation],
    cfg: Config,
) -> MatchedHoldResult:
    """Fit additive offset on Fit holds and verify on separate Check holds."""
    norm_cfg = cfg.measurement_normalization
    min_samples = norm_cfg.min_hold_samples
    min_coverage = norm_cfg.min_hold_coverage
    tol = rule.max_verification_error if rule.max_verification_error is not None else norm_cfg.default_max_verification_error

    fit_ref = rule.fit_reference_window
    fit_prac = rule.fit_practice_window
    check_ref = rule.check_reference_window
    check_prac = rule.check_practice_window
    if fit_ref is None or fit_prac is None or check_ref is None or check_prac is None:
        updated = rule.model_copy(update={
            "method": "matched_holds",
            "offset": 0.0,
            "verified": False,
            "verification_error": None,
            "status_message": "Choose separate Fit and Check holds on both recordings",
        })
        return MatchedHoldResult(updated, False, updated.status_message)

    if not windows_disjoint(fit_ref, check_ref):
        updated = rule.model_copy(update={
            "method": "matched_holds",
            "offset": 0.0,
            "verified": False,
            "verification_error": None,
            "status_message": "Reference Fit and Check holds must not overlap",
        })
        return MatchedHoldResult(updated, False, updated.status_message)
    if not windows_disjoint(fit_prac, check_prac):
        updated = rule.model_copy(update={
            "method": "matched_holds",
            "offset": 0.0,
            "verified": False,
            "verification_error": None,
            "status_message": "Practice Fit and Check holds must not overlap",
        })
        return MatchedHoldResult(updated, False, updated.status_message)

    fit_ref_stats, fit_ref_err = measure_hold_window(
        reference_frames, cfg, metric, target, fit_ref, min_samples=min_samples, min_coverage=min_coverage,
    )
    fit_prac_stats, fit_prac_err = measure_hold_window(
        practice_frames, cfg, metric, target, fit_prac, min_samples=min_samples, min_coverage=min_coverage,
    )
    check_ref_stats, check_ref_err = measure_hold_window(
        reference_frames, cfg, metric, target, check_ref, min_samples=min_samples, min_coverage=min_coverage,
    )
    check_prac_stats, check_prac_err = measure_hold_window(
        practice_frames, cfg, metric, target, check_prac, min_samples=min_samples, min_coverage=min_coverage,
    )

    stats_updates = {
        "fit_reference_stats": fit_ref_stats,
        "fit_practice_stats": fit_prac_stats,
        "check_reference_stats": check_ref_stats,
        "check_practice_stats": check_prac_stats,
    }
    for label, stats, reason in (
        ("Reference Fit", fit_ref_stats, fit_ref_err),
        ("Practice Fit", fit_prac_stats, fit_prac_err),
        ("Reference Check", check_ref_stats, check_ref_err),
        ("Practice Check", check_prac_stats, check_prac_err),
    ):
        if reason or stats is None or stats.valid_frames < min_samples or stats.coverage_ratio + 1e-9 < min_coverage:
            msg = f"{label} hold: {reason or 'Hold quality gate not met'}"
            updated = rule.model_copy(update={
                **stats_updates,
                "method": "matched_holds",
                "offset": 0.0,
                "verified": False,
                "verification_error": None,
                "status_message": msg,
            })
            return MatchedHoldResult(updated, False, msg)

    assert fit_ref_stats and fit_prac_stats and check_ref_stats and check_prac_stats
    offset = metric_offset_between(metric, fit_ref_stats.median, fit_prac_stats.median)
    corrected_check = apply_metric_offset(metric, check_prac_stats.median, offset)
    if not in_physical_domain(metric, corrected_check):
        msg = "Adjustment unavailable: corrected check hold falls outside physical domain"
        updated = rule.model_copy(update={
            **stats_updates,
            "method": "matched_holds",
            "offset": offset,
            "verified": False,
            "verification_error": None,
            "status_message": msg,
        })
        return MatchedHoldResult(updated, False, msg)

    err = metric_error(metric, corrected_check, check_ref_stats.median)
    verified = err <= tol + 1e-9
    unit = target.unit
    unit_suffix = "°" if unit == "degrees" else f" {unit}"
    if verified:
        msg = f"Verified on check hold (error {err:.2f}{unit_suffix} <= {tol:.2f}{unit_suffix})"
    else:
        msg = f"Check error {err:.2f}{unit_suffix} exceeds maximum {tol:.2f}{unit_suffix}"
    updated = rule.model_copy(update={
        **stats_updates,
        "method": "matched_holds",
        "offset": offset,
        "verification_error": err,
        "verified": verified,
        "status_message": msg,
    })
    return MatchedHoldResult(updated, verified, msg)


def _paths_equal(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    try:
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except OSError:
        return a == b


def _pair_matches_practice(
    pair: NormalizationPairRule,
    current_identity: HoldSourceIdentity,
) -> bool:
    stored = pair.practice_source
    if stored.raw_media_sha256 and current_identity.raw_media_sha256:
        return stored.raw_media_sha256 == current_identity.raw_media_sha256
    if stored.frames_fingerprint and current_identity.frames_fingerprint:
        if stored.frames_fingerprint == current_identity.frames_fingerprint:
            return True
    if stored.raw_media_sha256 is None and current_identity.raw_media_sha256 is None:
        if stored.segment_index == current_identity.segment_index and _paths_equal(
            stored.archive_path, current_identity.archive_path
        ):
            return True
    return False


def find_matching_pair(
    cfg: Config,
    calibration_id: str,
    practice_identity: HoldSourceIdentity,
) -> NormalizationPairRule | None:
    """Locate the stored pair rule targeting `(calibration_id, practice_identity)`."""
    candidates = [
        pair
        for pair in cfg.measurement_normalization.pairs
        if pair.calibration_id == calibration_id and _pair_matches_practice(pair, practice_identity)
    ]
    if not candidates:
        return None
    enabled_candidates = [pair for pair in candidates if pair.enabled]
    pool = enabled_candidates or candidates
    return max(pool, key=lambda pair: (pair.updated_at_utc, pair.created_at_utc, pair.id))


def evaluate_pair_applicability(
    cfg: Config,
    profile: RangeProfile,
    practice_identity: HoldSourceIdentity,
) -> PairApplicability:
    """Check whether any stored normalization pair safely applies to this comparison."""
    norm_cfg = cfg.measurement_normalization
    if not norm_cfg.enabled:
        pair = find_matching_pair(cfg, profile.id, practice_identity)
        return PairApplicability(False, "disabled", "Measurement normalization is disabled in Settings", pair)

    pair = find_matching_pair(cfg, profile.id, practice_identity)
    if pair is None:
        return PairApplicability(False, "no_pair", "No normalization pair configured for this comparison", None)
    if not pair.enabled:
        return PairApplicability(False, "disabled", "Normalization is turned off for this calibration and recording pair", pair)

    if pair.profile_content_digest != profile_content_digest(profile):
        return PairApplicability(
            False, "needs_recheck",
            "Calibration profile bounds or settings changed since this offset was saved.",
            pair,
        )

    expected_settings_fp = practice_identity.settings_fingerprint or settings_fingerprint(cfg)
    if pair.practice_source.settings_fingerprint and pair.practice_source.settings_fingerprint != expected_settings_fp:
        return PairApplicability(
            False, "needs_recheck",
            "Measurement settings changed since this offset was saved.",
            pair,
        )
    if pair.reference_source.settings_fingerprint and pair.reference_source.settings_fingerprint != expected_settings_fp:
        return PairApplicability(
            False, "needs_recheck",
            "Measurement settings changed since reference holds were measured.",
            pair,
        )

    if pair.practice_source.raw_media_sha256:
        if not practice_identity.raw_media_sha256:
            return PairApplicability(
                False, "needs_recheck",
                "Raw practice media fingerprint is unavailable for verification.",
                pair,
            )
        if pair.practice_source.raw_media_sha256 != practice_identity.raw_media_sha256:
            return PairApplicability(
                False, "needs_recheck",
                "Practice media no longer matches this normalization rule.",
                pair,
            )

    if (
        pair.practice_source.frames_fingerprint
        and practice_identity.frames_fingerprint
        and pair.practice_source.frames_fingerprint != practice_identity.frames_fingerprint
    ):
        return PairApplicability(
            False, "needs_recheck",
            "Practice recording landmarks changed after reanalysis.",
            pair,
        )

    for key in ("side", "facing"):
        stored_val = pair.practice_source.capture_identity.get(key)
        current_val = practice_identity.capture_identity.get(key)
        if stored_val is not None and current_val is not None and stored_val != current_val:
            return PairApplicability(
                False, "needs_recheck",
                "Capture orientation does not match the normalized recording.",
                pair,
            )

    return PairApplicability(True, "applied", "", pair)


def _matched_rule_passes_current_gates(
    rule: MetricNormalizationRule,
    norm_cfg: MeasurementNormalizationConfig,
) -> tuple[bool, str]:
    if not rule.verified or rule.verification_error is None:
        return False, rule.status_message or "Matched holds have not passed verification"
    tol = rule.max_verification_error if rule.max_verification_error is not None else norm_cfg.default_max_verification_error
    if rule.verification_error > tol + 1e-9:
        return False, f"Verification error {rule.verification_error:.2f} exceeds {tol:.2f}"
    fit_ref = rule.fit_reference_window
    check_ref = rule.check_reference_window
    fit_prac = rule.fit_practice_window
    check_prac = rule.check_practice_window
    if fit_ref is None or check_ref is None or fit_prac is None or check_prac is None:
        return False, "Fit and Check hold windows must all be defined"
    if not windows_disjoint(fit_ref, check_ref) or not windows_disjoint(fit_prac, check_prac):
        return False, "Fit and Check holds must not overlap"
    for stats in (
        rule.fit_reference_stats,
        rule.fit_practice_stats,
        rule.check_reference_stats,
        rule.check_practice_stats,
    ):
        if stats is None:
            return False, "Hold quality statistics are missing; recompute matched holds"
        if stats.valid_frames < norm_cfg.min_hold_samples:
            return False, f"Hold sample count {stats.valid_frames} is below minimum {norm_cfg.min_hold_samples}"
        if stats.coverage_ratio + 1e-9 < norm_cfg.min_hold_coverage:
            return False, f"Hold coverage {stats.coverage_ratio * 100:.0f}% is below {norm_cfg.min_hold_coverage * 100:.0f}%"
    return True, ""


def normalize_metric_observation(
    *,
    metric: str,
    observed_raw: float | None,
    quality_valid: bool,
    rule: MetricNormalizationRule | None,
    applicability: PairApplicability,
    norm_cfg: MeasurementNormalizationConfig,
) -> NormalizedMetricValue:
    """Resolve one frame's raw value, applied adjustment, and Evaluator input."""
    if not quality_valid or observed_raw is None or not math.isfinite(observed_raw):
        return NormalizedMetricValue(
            observed_raw=None,
            normalization_offset=None,
            reference_scale_raw=None,
            evaluator_input=None,
            normalization_status="no_observation",
            status_detail="",
        )
    raw = float(observed_raw)
    if applicability.status in ("no_pair", "disabled"):
        if applicability.pair is not None and rule is not None and not rule.enabled:
            return NormalizedMetricValue(raw, None, raw, raw, "direct_comparison", "")
        return NormalizedMetricValue(raw, None, raw, raw, "unadjusted", "")

    if applicability.status == "needs_recheck":
        return NormalizedMetricValue(
            observed_raw=raw,
            normalization_offset=None,
            reference_scale_raw=None,
            evaluator_input=raw,
            normalization_status="needs_recheck",
            status_detail=applicability.reason,
        )

    if rule is None or not rule.enabled:
        return NormalizedMetricValue(raw, None, raw, raw, "direct_comparison", "")

    if rule.method == "matched_holds":
        ok, detail = _matched_rule_passes_current_gates(rule, norm_cfg)
        if not ok:
            return NormalizedMetricValue(
                observed_raw=raw,
                normalization_offset=None,
                reference_scale_raw=None,
                evaluator_input=raw,
                normalization_status="needs_recheck",
                status_detail=detail,
            )

    candidate = apply_metric_offset(metric, raw, rule.offset)
    if not in_physical_domain(metric, candidate):
        return NormalizedMetricValue(
            observed_raw=raw,
            normalization_offset=rule.offset,
            reference_scale_raw=None,
            evaluator_input=raw,
            normalization_status="unavailable",
            status_detail="Adjusted value falls outside metric physical domain",
        )

    return NormalizedMetricValue(
        observed_raw=raw,
        normalization_offset=rule.offset,
        reference_scale_raw=candidate,
        evaluator_input=candidate,
        normalization_status="applied",
        status_detail=rule.status_message,
    )


def utc_timestamp_now() -> str:
    """Public helper returning an ISO-8601 UTC timestamp with Z suffix."""
    return _utc_now()
