"""Read-only projection of sessions and archive analytics with calibration comparison."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile

from ...archive import practice_directory, read_analytics, read_manifest
from ...calibration.evidence import Evidence, dancer_frames
from ...calibration.measurements import metric_samples
from ...calibration.models import (
    CYCLIC_METRICS,
    METRIC_DETAILS,
    RANGE_VERSION,
    ApprovedRange,
    RangeProfile,
    profile_from_calibration,
)
from ...config import Config, RangeDefault
from ...domain import METRIC_VERSION, METRICS, Calibration, FrameObservation
from ...geometry import wrapped
from ...references.coach import _coach_observation
from ...temporal import Evaluator

Point = tuple[float, float | None]


@dataclass(frozen=True)
class MetricData:
    points: list[Point]
    segments: list[list[tuple[float, float]]]
    summary: dict[str, Any]
    event_count: int
    open_event_count: int


@dataclass(frozen=True)
class PersistedEvent:
    identifier: str
    metric: str
    onset: float
    detection: float
    end: float | None
    peak_deviation: float | None
    end_reason: str | None


@dataclass(frozen=True)
class SessionData:
    identifier: str
    status: str
    metrics: dict[str, MetricData]
    active_duration: float | None
    max_gap: float | None
    origin_timestamp: float
    events: list[PersistedEvent]


@dataclass(frozen=True)
class MetricComparison:
    data: MetricData
    bounds: tuple[float, float] | None
    reference_line: float | None
    states: dict[float, str]
    unit: str
    calibration_label: str = ""
    note: str = ""


@dataclass(frozen=True)
class ComparisonResult:
    calibration_name: str
    metrics: dict[str, MetricComparison]
    events: list[PersistedEvent]
    note: str = ""


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _object(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("Invalid persisted session metadata.")
    return parsed


def _connect(data_dir: Path) -> sqlite3.Connection:
    path = data_dir / "tango.sqlite3"
    if not path.is_file():
        raise ValueError("No local session database. Save a practice session first.")
    db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        if db.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise ValueError("Unsupported local session database schema (expected version 1).")
    except Exception:
        db.close()
        raise
    return db


def _segments_from_points(
    points: list[Point], max_gap: float | None
) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    segment: list[tuple[float, float]] = []
    for timestamp, value in points:
        if value is None:
            segment = []
            continue
        if not segment or max_gap is None or timestamp - segment[-1][0] > max_gap:
            segment = []
            segments.append(segment)
        segment.append((timestamp, value))
    return segments


def _interval_summary(
    points: list[Point],
    max_gap: float | None,
    states: dict[float, str] | None = None,
) -> dict[str, Any]:
    valid_duration = 0.0
    unknown_duration = 0.0
    within_duration = 0.0
    pending_duration = 0.0
    deviation_duration = 0.0
    for (t0, v0), (t1, v1) in pairwise(points):
        dt = t1 - t0
        if dt <= 0 or (max_gap is not None and dt > max_gap):
            continue
        if v0 is not None and v1 is not None:
            valid_duration += dt
            if states is not None:
                state = states.get(t1) or states.get(t0)
                if state == "WITHIN_REFERENCE":
                    within_duration += dt
                elif state == "PENDING":
                    pending_duration += dt
                elif state == "DEVIATION":
                    deviation_duration += dt
        else:
            unknown_duration += dt
    active = valid_duration + unknown_duration
    samples = sum(1 for _, value in points if value is not None)
    summary: dict[str, Any] = {
        "valid_duration": valid_duration,
        "unknown_duration": unknown_duration,
        "valid_coverage": (valid_duration / active) if active > 0 else None,
        "samples": samples,
    }
    if states is not None:
        if valid_duration > 0:
            summary["within_reference_ratio"] = within_duration / valid_duration
            summary["within_reference_duration"] = within_duration
            summary["pending_duration"] = pending_duration
            summary["deviation_duration"] = deviation_duration
        elif samples > 0:
            within_count = sum(
                1 for ts, val in points if val is not None and states.get(ts) == "WITHIN_REFERENCE"
            )
            summary["within_reference_ratio"] = within_count / samples
    return summary


def _smooth_points(
    points: list[Point],
    cfg: Config,
    max_gap: float | None,
) -> list[Point]:
    if cfg.smoothing <= cfg.min_filter_time_constant:
        return list(points)
    smoothed_points: list[Point] = []
    smoothed: float | None = None
    last_valid: float | None = None
    for timestamp, raw in points:
        if last_valid is not None and max_gap is not None and timestamp - last_valid > max_gap:
            smoothed = None
            last_valid = None
        if raw is None or not math.isfinite(raw):
            smoothed_points.append((timestamp, None))
            continue
        if smoothed is None or last_valid is None:
            smoothed = raw
        else:
            dt = max(0.0, timestamp - last_valid)
            alpha = 1.0 - math.exp(-dt / max(cfg.smoothing, cfg.min_filter_time_constant))
            smoothed += alpha * (raw - smoothed)
        last_valid = timestamp
        smoothed_points.append((timestamp, smoothed))
    return smoothed_points


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


def _extract_archive_observations(
    path: Path,
    payload: dict[str, Any],
    cfg: Config,
    origin: float,
) -> tuple[list[FrameObservation], dict[str, Any], str | None]:
    coach = payload.get("coach_data")
    capture = (
        payload.get("capture_profile")
        or (coach.get("capture_profile") if isinstance(coach, dict) else None)
        or {}
    )
    if not isinstance(capture, dict):
        capture = {}
    model_hash = payload.get("model_hash") or (
        coach.get("model_hash") if isinstance(coach, dict) else None
    )
    if not isinstance(model_hash, str):
        model_hash = None

    entries = coach.get("frames", {}) if isinstance(coach, dict) else {}
    if not entries:
        entries = {str(i): entry for i, entry in enumerate(payload.get("frames") or [])}
    if not isinstance(entries, dict) or not entries:
        return [], capture, model_hash

    has_poses = any(
        isinstance(entry, dict)
        and any(entry.get(key) for key in ("measurement_poses", "poses", "pose"))
        for entry in entries.values()
    )
    if not has_poses:
        return [], capture, model_hash

    raw_width = (coach.get("width") if isinstance(coach, dict) else None) or capture.get("width")
    raw_height = (coach.get("height") if isinstance(coach, dict) else None) or capture.get("height")
    width = raw_width if type(raw_width) is int and raw_width > 0 else cfg.width
    height = raw_height if type(raw_height) is int and raw_height > 0 else cfg.height

    timestamps = payload.get("frame_timestamps") or []
    observations: list[FrameObservation] = []
    last_ts: float | None = None
    for frame_key, entry in sorted(entries.items(), key=lambda item: int(item[0])):
        if not isinstance(entry, dict):
            continue
        frame_index = int(frame_key)
        ts = (
            finite_number(timestamps[frame_index])
            if frame_index < len(timestamps)
            else finite_number(entry.get("timestamp"))
        )
        if ts is None:
            continue
        if last_ts is not None and ts <= last_ts:
            continue
        last_ts = ts
        obs = _coach_observation(entry, width, height, require_measurements=False)
        observations.append(FrameObservation(ts, width, height, obs.poses))

    selected_dancer = _read_selected_dancer(path)
    if (
        selected_dancer is not None
        and observations
        and any(len(frame.poses) > 1 for frame in observations)
    ):
        gap = finite_number(payload.get("max_gap"))
        evidence = Evidence(
            observations,
            model_hash or ("0" * 64),
            capture,
            {"archive": path.name},
            gap if gap is not None and gap > 0 else cfg.max_gap,
        )
        associated = dancer_frames(evidence, selected_dancer, cfg.confidence)
        if any(frame.poses for frame in associated):
            observations = associated

    return observations, capture, model_hash


def list_sessions(data_dir: Path) -> list[tuple[str, str]]:
    with closing(_connect(data_dir)) as db:
        return [
            (row["id"], f"{row['start_utc'] or 'Unknown date'} · {row['status']} · {row['id']}")
            for row in db.execute("SELECT id,start_utc,status FROM sessions ORDER BY rowid DESC")
        ]


def load_session(data_dir: Path, identifier: str) -> SessionData:
    """Keep persisted deltas and summaries, splitting lines at unknown/long gaps."""
    with closing(_connect(data_dir)) as db:
        # A consistent snapshot also prevents mixing a summary with newer observations.
        db.execute("BEGIN")
        row = db.execute(
            "SELECT s.*, c.metric_version FROM sessions s "
            "LEFT JOIN calibrations c ON c.id=s.calibration_id WHERE s.id=?",
            (identifier,),
        ).fetchone()
        if row is None:
            raise ValueError("Selected session is no longer available.")
        if row["status"] == "running":
            raise ValueError("Selected session is not finalized. Save or finish it first.")
        versions = _object(row["versions"])
        if (
            row["metric_version"] != METRIC_VERSION
            or versions.get("metric_version", METRIC_VERSION) != METRIC_VERSION
        ):
            raise ValueError("Selected session has incompatible or unknown metric semantics.")
        summary = _object(row["summary"])
        if summary and summary.get("schema_version") != 1:
            raise ValueError("Unsupported persisted summary schema.")
        config = _object(row["config"])
        max_gap = finite_number(config.get("max_gap"))
        if max_gap is not None and max_gap <= 0:
            max_gap = None
        rows = db.execute(
            "SELECT timestamp,metric,delta,quality_state,state FROM observations "
            "WHERE session_id=? ORDER BY timestamp,metric",
            (identifier,),
        ).fetchall()
        events = db.execute(
            "SELECT id,metric,onset,detection,end,peak_deviation,end_reason "
            "FROM events WHERE session_id=? ORDER BY onset,id",
            (identifier,),
        ).fetchall()

    return project_session(
        identifier, row["status"], rows, events, summary, max_gap,
    )


def project_session(
    identifier: str, status: str, rows: list[Any], events: list[Any],
    summary: dict[str, Any], max_gap: float | None,
) -> SessionData:
    if any(finite_number(observation["timestamp"]) is None for observation in rows):
        raise ValueError("Selected session contains an invalid observation timestamp.")
    rows = sorted(rows, key=lambda observation: (observation["timestamp"], observation["metric"]))
    if not rows:
        raise ValueError("Selected session has no compatible persisted metric observations.")
    origin = finite_number(rows[0]["timestamp"])
    if origin is None:
        raise ValueError("Selected session contains an invalid observation timestamp.")
    if summary.get("measurement_version", RANGE_VERSION) != RANGE_VERSION:
        raise ValueError("Incompatible range measurement version")
    keys = list(dict.fromkeys([*METRICS, *(row["metric"] for row in rows)]))
    if any(key not in METRIC_DETAILS for key in keys):
        raise ValueError("Selected session has no compatible persisted observations: unsupported metric")
    points: dict[str, list[Point]] = {metric: [] for metric in keys}
    for observation in rows:
        timestamp = finite_number(observation["timestamp"])
        own = points[observation["metric"]]
        if timestamp is None or (own and timestamp - origin <= own[-1][0]):
            raise ValueError(
                "Selected session contains invalid or duplicate observation timestamps."
            )
        delta = finite_number(observation["delta"])
        if observation["quality_state"] != "VALID" or observation["state"] == "UNKNOWN":
            delta = None
        own.append((timestamp - origin, delta))

    metric_summaries = summary.get("metrics", {})
    if not isinstance(metric_summaries, dict):
        raise TypeError("Invalid persisted metric summaries.")
    persisted_events = []
    for event in events:
        onset = finite_number(event["onset"])
        detection = finite_number(event["detection"])
        end = finite_number(event["end"])
        peak = finite_number(event["peak_deviation"])
        if onset is None or detection is None or detection < onset:
            raise ValueError("Selected session contains an invalid persisted event timeline.")
        if end is not None and end < detection:
            raise ValueError("Selected session contains an invalid persisted event timeline.")
        persisted_events.append(
            PersistedEvent(
                str(event["id"]),
                str(event["metric"]),
                onset,
                detection,
                end,
                peak,
                event["end_reason"] if isinstance(event["end_reason"], str) else None,
            )
        )
    counts: dict[str, tuple[int, int]] = {}
    for event in persisted_events:
        total, opened = counts.get(event.metric, (0, 0))
        counts[event.metric] = total + 1, opened + int(event.end is None)
    metrics = {}
    for metric, own in points.items():
        segments = _segments_from_points(own, max_gap)
        details = metric_summaries.get(metric, {})
        if not isinstance(details, dict):
            raise TypeError("Invalid persisted metric summary.")
        count, open_count = counts.get(metric, (0, 0))
        metrics[metric] = MetricData(own, segments, details, count, open_count)
    return SessionData(
        identifier,
        status,
        metrics,
        finite_number(summary.get("active_duration")),
        max_gap,
        origin,
        persisted_events,
    )


def session_payload(data_dir: Path, identifier: str) -> dict[str, Any]:
    dataset = load_session(data_dir, identifier)
    with closing(_connect(data_dir)) as db:
        row = db.execute("SELECT summary FROM sessions WHERE id=?", (identifier,)).fetchone()
        observations = [
            dict(item) for item in db.execute(
                "SELECT timestamp,metric,delta,quality_state,state FROM observations "
                "WHERE session_id=? ORDER BY timestamp,metric", (identifier,),
            )
        ]
    return {
        "id": identifier, "status": dataset.status,
        "observations": observations, "summary": _object(row["summary"]),
        "max_gap": dataset.max_gap,
        "events": [
            {"id": event.identifier, "metric": event.metric, "onset": event.onset,
             "detection": event.detection, "end": event.end,
             "peak_deviation": event.peak_deviation, "end_reason": event.end_reason}
            for event in dataset.events
        ],
    }


def load_archive_session(path: Path, index: int) -> SessionData:
    payload = read_analytics(path, index)
    if payload is None:
        raise ValueError("This recording has no stored analytics. Analyze it in Review.")
    session = payload.get("session")
    if not isinstance(session, dict):
        raise TypeError("No reference-relative analytics were stored for this recording.")
    if not isinstance(session.get("status"), str) or session["status"] == "running":
        raise ValueError("Archived session is not finalized or has an invalid status")
    summary = session.get("summary") or {}
    if not isinstance(summary, dict) or summary.get("schema_version", 1) != 1:
        raise ValueError("Unsupported archived analytics summary")
    rows = session.get("observations")
    events = session.get("events") or []
    if not isinstance(rows, list) or not isinstance(events, list):
        raise TypeError("Invalid archived analytics observations or events")
    return project_session(
        str(session.get("id") or f"{path.name}:{index}"),
        str(session.get("status", "completed")), rows, events, summary,
        gap if (gap := finite_number(session.get("max_gap"))) is not None and gap > 0 else None,
    )


FRAME_METRICS = {
    "head_angle": ("Head alignment", "degrees", 1.0),
    "shoulder_elev_pct": ("Shoulder height", "% raised", 1.0),
    "knee_bend_deg": ("Knee bend", "degrees", 1.0),
    "axis_curve_smoothed": ("Torso / hip offset", "%", 100.0),
}


def load_archive_dataset(path: Path, index: int, cfg: Config | None = None) -> SessionData:
    effective_cfg = cfg if cfg is not None else Config()
    payload = read_analytics(path, index)
    if payload is None:
        raise ValueError("No stored analytics for this recording. Analyze it in Review.")
    has_session = isinstance(payload.get("session"), dict)
    if has_session:
        dataset = load_archive_session(path, index)
    else:
        dataset = SessionData(
            f"{path.name}:{index:02d}", "archived",
            {key: MetricData([], [], {}, 0, 0) for key in METRICS},
            None, finite_number(payload.get("max_gap")), 0.0, [],
        )
    coach = payload.get("coach_data")
    entries = coach.get("frames", {}) if isinstance(coach, dict) else {}
    if not entries:
        entries = {str(i): entry for i, entry in enumerate(payload.get("frames") or [])}
    timestamps = payload.get("frame_timestamps") or []
    origin = (
        dataset.origin_timestamp if has_session
        else float(timestamps[0]) if timestamps else dataset.origin_timestamp
    )
    for key, (_, _, scale) in FRAME_METRICS.items():
        points: list[Point] = []
        for frame, entry in sorted(entries.items(), key=lambda item: int(item[0])):
            if not isinstance(entry, dict):
                raise TypeError("Invalid archived frame analytics")
            frame_index = int(frame)
            timestamp = (
                finite_number(timestamps[frame_index]) if frame_index < len(timestamps)
                else finite_number(entry.get("timestamp"))
            )
            value = finite_number(entry.get(key))
            if key == "axis_curve_smoothed" and value is None:
                value = finite_number(entry.get("axis_curve_ratio"))
            if timestamp is not None:
                points.append((timestamp - origin, value * scale if value is not None else None))
        segments = _segments_from_points(points, dataset.max_gap)
        dataset.metrics[key] = MetricData(points, segments, {}, 0, 0)

    observations, capture, _ = _extract_archive_observations(path, payload, effective_cfg, origin)
    if observations:
        eval_cfg = effective_cfg.model_copy(
            update={
                key: capture[key]
                for key in ("side", "facing")
                if capture.get(key) in ("left", "right")
            }
        )
        raw_series = metric_samples(observations, eval_cfg, view="all")
        for key in METRIC_DETAILS:
            if has_session and key in dataset.metrics and dataset.metrics[key].points:
                continue
            rel_points: list[Point] = [(t - origin, val) for t, val in raw_series[key]]
            smoothed = _smooth_points(rel_points, eval_cfg, dataset.max_gap)
            segments = _segments_from_points(smoothed, dataset.max_gap)
            summary = _interval_summary(smoothed, dataset.max_gap)
            summary["source"] = "observations"
            dataset.metrics[key] = MetricData(smoothed, segments, summary, 0, 0)

    active_duration = dataset.active_duration
    if active_duration is None:
        if len(timestamps) >= 2:
            active_duration = max(0.0, float(timestamps[-1]) - float(timestamps[0]))
        elif len(observations) >= 2:
            active_duration = max(0.0, observations[-1].timestamp - observations[0].timestamp)

    if not has_session:
        return SessionData(
            dataset.identifier, dataset.status, dataset.metrics, active_duration,
            dataset.max_gap, origin, dataset.events,
        )
    return SessionData(
        dataset.identifier, dataset.status, dataset.metrics, active_duration,
        dataset.max_gap, dataset.origin_timestamp, dataset.events,
    )


def _evaluate_series(
    metric: str,
    raw_points: list[Point],
    eval_cfg: Config,
    origin: float,
    max_gap: float | None,
    *,
    bounds: tuple[float, float] | None = None,
    baseline: float = 0.0,
    pass_through: bool = False,
) -> tuple[MetricData, dict[float, str], list[PersistedEvent]]:
    if pass_through:
        run_cfg = eval_cfg.model_copy(update={"smoothing": eval_cfg.min_filter_time_constant})
    else:
        run_cfg = eval_cfg
    if max_gap is not None and max_gap > 0:
        run_cfg = run_cfg.model_copy(update={"max_gap": max_gap})
    evaluator = Evaluator(metric, baseline, run_cfg, bounds=bounds)
    smoothed_points: list[Point] = []
    states: dict[float, str] = {}
    last_t: float | None = None
    for rel_t, raw in raw_points:
        if last_t is not None and rel_t <= last_t:
            continue
        last_t = rel_t
        row = evaluator.update(rel_t, raw)
        smoothed = finite_number(row["smoothed"])
        smoothed_points.append((rel_t, smoothed))
        states[rel_t] = str(row["state"])
    segments = _segments_from_points(smoothed_points, max_gap)
    events = [
        PersistedEvent(
            event.id,
            metric,
            origin + event.onset,
            origin + event.detection,
            origin + event.end if event.end is not None else None,
            event.peak_deviation,
            event.end_reason,
        )
        for event in evaluator.events
    ]
    open_count = sum(1 for ev in events if ev.end is None)
    summary = _interval_summary(smoothed_points, max_gap, states)
    data = MetricData(smoothed_points, segments, summary, len(events), open_count)
    return data, states, events


def _frame_metric_targets(
    targets: Mapping[str, RangeDefault | ApprovedRange],
    cfg: Config,
    *,
    is_profile: bool,
) -> dict[str, tuple[tuple[float, float] | None, float | None, str]]:
    mapped: dict[str, tuple[tuple[float, float] | None, float | None, str]] = {}

    def _ref(target: RangeDefault | ApprovedRange) -> float:
        if isinstance(target, ApprovedRange):
            return target.median
        return (target.lower + target.upper) / 2.0

    if "head_pitch" in targets:
        target = targets["head_pitch"]
        mapped["head_angle"] = ((target.lower, target.upper), _ref(target), "head_pitch")
    if "knee_flexion" in targets:
        target = targets["knee_flexion"]
        mapped["knee_bend_deg"] = ((target.lower, target.upper), _ref(target), "knee_flexion")
    if "hip_offset" in targets:
        target = targets["hip_offset"]
        mapped["axis_curve_smoothed"] = (
            (target.lower * 100.0, target.upper * 100.0),
            _ref(target) * 100.0,
            "hip_offset",
        )
    if "shoulder_clearance" in targets:
        target = targets["shoulder_clearance"]
        base = max(cfg.shoulder_baseline_clamp_floor, float(cfg.shoulder_neutral_clearance))
        pct_a = ((base - target.upper) / base) * 100.0
        pct_b = ((base - target.lower) / base) * 100.0
        mapped["shoulder_elev_pct"] = (
            (min(pct_a, pct_b), max(pct_a, pct_b)),
            ((base - _ref(target)) / base) * 100.0,
            "shoulder_clearance",
        )
    elif not is_profile and "shoulder_clearance" not in targets:
        # Built-in Settings thresholds don't calibrate shoulder clearance without a profile.
        pass
    return mapped


def _metric_unit(key: str) -> str:
    if key in METRIC_DETAILS:
        return METRIC_DETAILS[key][2]
    if key in FRAME_METRICS:
        return FRAME_METRICS[key][1]
    return ""


def compare_archive_with_calibration(
    path: Path,
    index: int,
    dataset: SessionData,
    cfg: Config,
    calibration: RangeProfile | Calibration | str | None,
) -> ComparisonResult:
    """Compare archive observations/series against a selected calibration reference."""
    if calibration is None:
        return ComparisonResult(
            calibration_name="No calibration reference",
            metrics={
                key: MetricComparison(
                    data=data,
                    bounds=None,
                    reference_line=None,
                    states={},
                    unit=_metric_unit(key),
                    calibration_label="No calibration reference",
                )
                for key, data in dataset.metrics.items()
            },
            events=list(dataset.events),
        )

    payload = read_analytics(path, index) or {}
    observations, capture, model_hash = _extract_archive_observations(
        path, payload, cfg, dataset.origin_timestamp
    )
    notes: list[str] = []

    if calibration == "__defaults__":
        calibration_name = "Settings default ranges"
        if cfg.calibration_defaults:
            return _compare_range_targets(
                dataset, observations, capture, model_hash,
                cfg, cfg.calibration_defaults, calibration_name,
                expected_model_hash=None, expected_capture=None,
            )
        # Fallback to built-in Settings thresholds where 0-centered / unambiguous limits exist.
        metrics: dict[str, MetricComparison] = {}
        events: list[PersistedEvent] = []
        base_clearance = max(cfg.shoulder_baseline_clamp_floor, float(cfg.shoulder_neutral_clearance))
        builtin_bounds: dict[str, tuple[tuple[float, float] | None, float | None]] = {
            "hip_offset": (
                (cfg.axis_neutral_offset - cfg.axis_curve_entry,
                 cfg.axis_neutral_offset + cfg.axis_curve_entry),
                cfg.axis_neutral_offset,
            ),
            "axis_curve_smoothed": (
                ((cfg.axis_neutral_offset - cfg.axis_curve_entry) * 100.0,
                 (cfg.axis_neutral_offset + cfg.axis_curve_entry) * 100.0),
                cfg.axis_neutral_offset * 100.0,
            ),
            "shoulder_elev_pct": ((0.0, cfg.shoulder_raise_threshold), 0.0),
            "shoulder_clearance": (None, base_clearance),
            "head_pitch": (None, cfg.head_neutral_offset),
            "head_angle": (None, cfg.head_neutral_offset),
            "knee_flexion": (None, cfg.closed_embrace_min_knee_bend),
            "knee_bend_deg": (None, cfg.closed_embrace_min_knee_bend),
        }
        eval_cfg = cfg.model_copy(
            update={
                k: capture[k] for k in ("side", "facing") if capture.get(k) in ("left", "right")
            }
        )
        raw_all = metric_samples(observations, eval_cfg, view="all") if observations else {}
        for key in (*METRIC_DETAILS, *FRAME_METRICS):
            spec = builtin_bounds.get(key)
            bounds = spec[0] if spec else None
            ref_line = spec[1] if spec else None
            if raw_all.get(key):
                rel_raw = [(t - dataset.origin_timestamp, v) for t, v in raw_all[key]]
                if bounds is not None:
                    data, states, metric_events = _evaluate_series(
                        key, rel_raw, eval_cfg, dataset.origin_timestamp, dataset.max_gap,
                        bounds=bounds,
                    )
                    events.extend(metric_events)
                else:
                    smoothed = _smooth_points(rel_raw, eval_cfg, dataset.max_gap)
                    segments = _segments_from_points(smoothed, dataset.max_gap)
                    data = MetricData(smoothed, segments, _interval_summary(smoothed, dataset.max_gap), 0, 0)
                    states = {}
                metrics[key] = MetricComparison(
                    data=data, bounds=bounds, reference_line=ref_line, states=states,
                    unit=_metric_unit(key), calibration_label=calibration_name,
                )
            elif key in dataset.metrics:
                existing = dataset.metrics[key]
                if bounds is not None and existing.segments:
                    data, states, metric_events = _evaluate_series(
                        key, existing.points, eval_cfg, dataset.origin_timestamp, dataset.max_gap,
                        bounds=bounds, pass_through=True,
                    )
                    events.extend(metric_events)
                else:
                    data = existing
                    states = {}
                metrics[key] = MetricComparison(
                    data=data, bounds=bounds, reference_line=ref_line, states=states,
                    unit=_metric_unit(key), calibration_label=calibration_name,
                )
        events.sort(key=lambda item: (item.onset, item.metric))
        return ComparisonResult(
            calibration_name=calibration_name,
            metrics=metrics,
            events=events,
            note=(
                "No [calibration_defaults] saved in Settings; showing built-in Settings reference "
                "lines/limits where defined. Select a .tpc profile for full calibrated ranges."
            ),
        )

    if isinstance(calibration, RangeProfile):
        profile: RangeProfile | None = calibration
    elif isinstance(calibration, Calibration):
        profile = profile_from_calibration(calibration)
    else:
        raise TypeError(f"Unsupported calibration reference: {calibration}")
    if profile is not None:
        return _compare_range_targets(
            dataset, observations, capture, model_hash,
            cfg, profile.metrics, profile.name,
            expected_model_hash=profile.model_hash,
            expected_capture=profile.capture_profile,
        )

    # Legacy Calibration (neutral baseline dict for head_forward, head_inclination, trunk_inclination)
    assert isinstance(calibration, Calibration)
    cal_name = calibration.name or "Legacy baseline"
    cal_capture = calibration.profile if isinstance(calibration.profile, dict) else {}
    if model_hash and calibration.model_hash and model_hash != calibration.model_hash:
        notes.append("pose model differs from legacy calibration")
    for k in ("side", "facing"):
        if capture.get(k) and cal_capture.get(k) and capture.get(k) != cal_capture.get(k):
            notes.append(f"recording {k}={capture.get(k)} differs from calibration {k}={cal_capture.get(k)}")
            break

    eval_cfg = cfg.model_copy(
        update={
            k: capture[k] if capture.get(k) in ("left", "right") else cal_capture.get(k, getattr(cfg, k))
            for k in ("side", "facing")
        }
    )
    raw_all = metric_samples(observations, eval_cfg, view="all") if observations else {}
    metrics = {}
    events = []
    for key in (*METRIC_DETAILS, *FRAME_METRICS):
        if key in METRICS and key in calibration.baseline:
            baseline = float(calibration.baseline[key])
            th = getattr(eval_cfg, key)
            bounds = (baseline - th.entry, baseline + th.entry)
            if raw_all.get(key):
                rel_raw = [(t - dataset.origin_timestamp, v) for t, v in raw_all[key]]
                data, states, metric_events = _evaluate_series(
                    key, rel_raw, eval_cfg, dataset.origin_timestamp, dataset.max_gap,
                    bounds=None, baseline=baseline,
                )
                events.extend(metric_events)
                metrics[key] = MetricComparison(
                    data=data, bounds=bounds, reference_line=baseline, states=states,
                    unit=_metric_unit(key), calibration_label=cal_name,
                )
            elif key in dataset.metrics:
                existing = dataset.metrics[key]
                # Session payloads store deltas relative to session baseline (0.0 center).
                rel_bounds = (-th.entry, th.entry)
                data, states, metric_events = _evaluate_series(
                    key, existing.points, eval_cfg, dataset.origin_timestamp, dataset.max_gap,
                    bounds=None, baseline=0.0, pass_through=True,
                )
                events.extend(metric_events)
                metrics[key] = MetricComparison(
                    data=data, bounds=rel_bounds, reference_line=0.0, states=states,
                    unit=_metric_unit(key), calibration_label=cal_name,
                )
        elif key in dataset.metrics:
            metrics[key] = MetricComparison(
                data=dataset.metrics[key], bounds=None, reference_line=None, states={},
                unit=_metric_unit(key), calibration_label="Not in legacy calibration",
            )
    events.sort(key=lambda item: (item.onset, item.metric))
    note_text = f"Legacy baseline comparison ({'; '.join(notes)})." if notes else ""
    return ComparisonResult(calibration_name=cal_name, metrics=metrics, events=events, note=note_text)


def _compare_range_targets(
    dataset: SessionData,
    observations: list[FrameObservation],
    capture: dict[str, Any],
    model_hash: str | None,
    cfg: Config,
    targets: Mapping[str, RangeDefault | ApprovedRange],
    calibration_name: str,
    *,
    expected_model_hash: str | None,
    expected_capture: dict[str, Any] | None,
) -> ComparisonResult:
    notes: list[str] = []
    if expected_model_hash and model_hash and model_hash != expected_model_hash:
        notes.append("recording and calibration use different pose models")
    if isinstance(expected_capture, dict):
        mismatches = [
            f"{k}: recording={capture.get(k)} vs calibration={expected_capture.get(k)}"
            for k in ("side", "facing")
            if capture.get(k) and expected_capture.get(k) and capture.get(k) != expected_capture.get(k)
        ]
        if mismatches:
            notes.append("; ".join(mismatches))

    eval_cfg = cfg.model_copy(
        update={
            k: capture[k]
            if capture.get(k) in ("left", "right")
            else (
                expected_capture.get(k, getattr(cfg, k))
                if isinstance(expected_capture, dict)
                else getattr(cfg, k)
            )
            for k in ("side", "facing")
        }
    )

    samples_by_view: dict[str, dict[str, list[tuple[float, float | None]]]] = {}
    if observations:
        views: set[Literal["side", "front", "all"]] = {"all"}
        for target in targets.values():
            views.add(target.view)
        for view in views:
            samples_by_view[view] = metric_samples(observations, eval_cfg, view=view)

    metrics: dict[str, MetricComparison] = {}
    events: list[PersistedEvent] = []
    fallback_metrics: list[str] = []

    for key in METRIC_DETAILS:
        range_target = targets.get(key)
        existing = dataset.metrics.get(key)
        if range_target is None:
            if existing is not None:
                metrics[key] = MetricComparison(
                    data=existing,
                    bounds=None,
                    reference_line=None,
                    states={},
                    unit=_metric_unit(key),
                    calibration_label="Skipped in calibration",
                )
            continue

        bounds = (range_target.lower, range_target.upper)
        ref_line = (
            range_target.median
            if isinstance(range_target, ApprovedRange)
            else (range_target.lower + range_target.upper) / 2.0
        )

        if observations:
            view_series = samples_by_view[range_target.view][key]
            used_fallback = False
            if range_target.view != "all":
                all_series = samples_by_view["all"][key]
                combined: list[tuple[float, float | None]] = []
                for (t, v_view), (_, v_all) in zip(view_series, all_series, strict=True):
                    if v_view is None and v_all is not None:
                        used_fallback = True
                        combined.append((t, v_all))
                    else:
                        combined.append((t, v_view))
                series = combined
                if used_fallback:
                    fallback_metrics.append(key)
            else:
                series = view_series
            center = (range_target.lower + range_target.upper) / 2.0
            rel_raw: list[Point] = [
                (
                    t - dataset.origin_timestamp,
                    (center + wrapped(v - center))
                    if (v is not None and key in CYCLIC_METRICS)
                    else v,
                )
                for t, v in series
            ]
            data, states, metric_events = _evaluate_series(
                key, rel_raw, eval_cfg, dataset.origin_timestamp, dataset.max_gap, bounds=bounds
            )
            events.extend(metric_events)
            note = "Measured across all camera angles (recording differed from target view)." if used_fallback else ""
            metrics[key] = MetricComparison(
                data=data,
                bounds=bounds,
                reference_line=ref_line,
                states=states,
                unit=_metric_unit(key),
                calibration_label=calibration_name,
                note=note,
            )
        elif existing is not None and existing.segments:
            data, states, metric_events = _evaluate_series(
                key, existing.points, eval_cfg, dataset.origin_timestamp, dataset.max_gap,
                bounds=bounds, pass_through=True,
            )
            events.extend(metric_events)
            metrics[key] = MetricComparison(
                data=data,
                bounds=bounds,
                reference_line=ref_line,
                states=states,
                unit=_metric_unit(key),
                calibration_label=calibration_name,
            )
        elif existing is not None:
            metrics[key] = MetricComparison(
                data=existing,
                bounds=bounds,
                reference_line=ref_line,
                states={},
                unit=_metric_unit(key),
                calibration_label=calibration_name,
            )

    frame_targets = _frame_metric_targets(targets, eval_cfg, is_profile=True)
    for key in FRAME_METRICS:
        existing = dataset.metrics.get(key)
        if existing is None:
            continue
        mapped = frame_targets.get(key)
        if mapped is None:
            metrics[key] = MetricComparison(
                data=existing,
                bounds=None,
                reference_line=None,
                states={},
                unit=_metric_unit(key),
                calibration_label="Skipped in calibration",
            )
            continue
        f_bounds, f_ref, _ = mapped
        if existing.segments and f_bounds is not None:
            data, states, _ = _evaluate_series(
                key, existing.points, eval_cfg, dataset.origin_timestamp, dataset.max_gap,
                bounds=f_bounds, pass_through=True,
            )
        else:
            data = existing
            states = {}
        metrics[key] = MetricComparison(
            data=data,
            bounds=f_bounds,
            reference_line=f_ref,
            states=states,
            unit=_metric_unit(key),
            calibration_label=calibration_name,
        )

    if fallback_metrics:
        notes.append(
            "used all-angle measurement for "
            + ", ".join(METRIC_DETAILS[m][1] for m in fallback_metrics)
            + " because recording view did not match approved view"
        )
    events.sort(key=lambda item: (item.onset, item.metric))
    return ComparisonResult(
        calibration_name=calibration_name,
        metrics=metrics,
        events=events,
        note=("Note: " + "; ".join(notes) + ".") if notes else "",
    )


def recordings_for_session(data_dir: Path, identifier: str) -> list[tuple[Path, int, list[float]]]:
    """Find archived recording segments whose report identifies this persisted session."""
    folder = practice_directory(str(data_dir))
    if not folder.is_dir():
        return []
    session_line = re.compile(rf"^\| Session ID \| `{re.escape(identifier)}` \|$", re.MULTILINE)
    matches: list[tuple[Path, int, list[float]]] = []
    for path in sorted(folder.glob("*.tp"), key=lambda item: item.stat().st_mtime_ns, reverse=True):
        try:
            manifest = read_manifest(path)
            with ZipFile(path) as archive:
                for index, segment in enumerate(manifest["segments"]):
                    report = archive.read(segment["report"]).decode("utf-8")
                    if not session_line.search(report):
                        continue
                    metadata = json.loads(archive.read(segment["metadata"]))
                    if not isinstance(metadata, dict):
                        continue
                    timestamps = metadata.get("frame_timestamps", [])
                    if (
                        isinstance(timestamps, list)
                        and timestamps
                        and all(
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(value)
                            for value in timestamps
                        )
                        and all(
                            earlier <= later for earlier, later in pairwise(timestamps)
                        )
                    ):
                        matches.append((path.resolve(), index, [float(value) for value in timestamps]))
        except (OSError, ValueError, KeyError, TypeError, UnicodeError, BadZipFile):
            continue
    return matches
