"""CSV export for Couple summary measures and discrete timeline events."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

CSV_COLUMNS = [
    "row_type",
    "recording",
    "segment",
    "run_id",
    "measure",
    "subject",
    "value",
    "unit",
    "status",
    "timestamp_or_bracket",
    "coverage_or_reason",
    "turn_id",
    "pose_origin",
    "eligible_count",
    "coverage_denominator",
    "evaluated_window_count",
    "status_counts",
]


def _fmt_val(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.4f}".rstrip("0").rstrip(".") or "0"
    return str(val)


def _fmt_status_counts(val: Any) -> str:
    if isinstance(val, dict) and val:
        return json.dumps(val, sort_keys=True, separators=(",", ":"))
    return ""


def export_couple_csv(
    couple_result: dict[str, Any],
    recording_name: str,
    segment_index: int,
) -> str:
    """Render Couple summary rows and timeline event rows into RFC-4180 CSV text."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_COLUMNS)

    run_id = str(couple_result.get("run_id", ""))
    summaries = couple_result.get("summaries") or {}
    coverage = couple_result.get("coverage") or {}
    coord = summaries.get("coordination") or {}

    for key in ("start_lag_ms", "slow_lag_ms", "settle_lag_ms"):
        item = coord.get(key)
        if isinstance(item, dict):
            writer.writerow(
                [
                    "summary",
                    recording_name,
                    segment_index,
                    run_id,
                    key,
                    item.get("subject", "B-A"),
                    _fmt_val(item.get("value")),
                    item.get("unit", "ms"),
                    item.get("status", "unavailable"),
                    item.get("timestamp_or_bracket", ""),
                    item.get("reason") or f"coverage={item.get('coverage', 0.0)}",
                    "",
                    "",
                    _fmt_val(item.get("eligible_count")),
                    _fmt_val(item.get("coverage_denominator")),
                    "",
                    _fmt_status_counts(item.get("status_counts")),
                ]
            )

    step_timing = summaries.get("step_timing") or {}
    for subj in ("A", "B"):
        st = step_timing.get(subj) or {}
        writer.writerow(
            [
                "summary",
                recording_name,
                segment_index,
                run_id,
                "median_beat_offset_ms",
                subj,
                _fmt_val(st.get("median_beat_offset_ms")),
                "ms",
                st.get("status", "unavailable"),
                "",
                f"contacts={st.get('contact_count', 0)},matched={st.get('matched_count', 0)}",
                "",
                "",
                _fmt_val(st.get("eligible_count")),
                _fmt_val(st.get("coverage_denominator")),
                "",
                _fmt_status_counts(st.get("status_counts")),
            ]
        )

    trans = summaries.get("transitions") or {}
    for mkey in ("secondary_speed_peaks", "unintended_reversals", "planned_pauses", "planned_reversals"):
        writer.writerow(
            [
                "summary",
                recording_name,
                segment_index,
                run_id,
                mkey,
                "pair",
                _fmt_val(trans.get(mkey)),
                "count",
                trans.get("status", "unavailable"),
                "",
                trans.get("reason") or f"coverage={trans.get('coverage', 0.0)}",
                "",
                "",
                _fmt_val(trans.get("eligible_count")),
                _fmt_val(trans.get("coverage_denominator")),
                _fmt_val(trans.get("evaluated_window_count")),
                _fmt_status_counts(trans.get("status_counts")),
            ]
        )

    posture = summaries.get("posture") or {}
    for pkey in (
        "torso_tilt_a",
        "torso_tilt_b",
        "couple_midpoint_tilt",
        "torso_separation",
        "torso_tilt_pearson_r",
        "head_pitch_a",
        "head_pitch_b",
    ):
        item = posture.get(pkey) or {}
        if pkey == "torso_tilt_pearson_r":
            val = item.get("value")
            unit = item.get("unit", "r")
        else:
            val = item.get("median_deg")
            unit = "degrees"
        writer.writerow(
            [
                "summary",
                recording_name,
                segment_index,
                run_id,
                pkey,
                item.get("subject", "pair"),
                _fmt_val(val),
                unit,
                item.get("status", "unavailable"),
                "",
                item.get("reason") or f"coverage={item.get('coverage', 0.0)}",
                "",
                "",
                _fmt_val(item.get("eligible_count")),
                _fmt_val(item.get("coverage_denominator")),
                "",
                _fmt_status_counts(item.get("status_counts")),
            ]
        )

    turns = summaries.get("turns") or {}
    writer.writerow(
        [
            "summary",
            recording_name,
            segment_index,
            run_id,
            "median_turn_duration_s",
            "pair",
            _fmt_val(turns.get("median_turn_duration_s")),
            "seconds",
            turns.get("status", "unavailable"),
            "",
            f"turns={turns.get('turn_count', 0)},overlaps={turns.get('overlap_count', 0)},unmatched={turns.get('unmatched_count', 0)}",
            "",
            "",
            _fmt_val(turns.get("eligible_count")),
            _fmt_val(turns.get("coverage_denominator")),
            "",
            _fmt_status_counts(turns.get("status_counts")),
        ]
    )
    writer.writerow(
        [
            "summary",
            recording_name,
            segment_index,
            run_id,
            "median_exit_minus_turn_end_ms",
            "pair",
            _fmt_val(turns.get("median_exit_minus_turn_end_ms")),
            "ms",
            turns.get("status", "unavailable"),
            "",
            f"pair_hip_coverage={coverage.get('pair_hip_coverage', 0.0)}",
            "",
            "",
            _fmt_val(turns.get("eligible_count")),
            _fmt_val(turns.get("coverage_denominator")),
            "",
            _fmt_status_counts(turns.get("status_counts")),
        ]
    )

    for ev in couple_result.get("events") or []:
        t0 = ev.get("bracket_start") if ev.get("bracket_start") is not None else ev.get("timestamp")
        t1 = ev.get("bracket_end") if ev.get("bracket_end") is not None else ev.get("end_timestamp")
        bracket = f"{_fmt_val(t0)}..{_fmt_val(t1)}" if t1 is not None else _fmt_val(ev.get("timestamp"))
        val = (
            ev.get("lag_ms")
            if ev.get("lag_ms") is not None
            else (
                ev.get("beat_offset_ms")
                if ev.get("beat_offset_ms") is not None
                else ev.get("value")
            )
        )
        unit = (
            "ms"
            if ev.get("lag_ms") is not None or ev.get("beat_offset_ms") is not None
            else ev.get("unit", "")
        )
        meta = ev.get("metadata") or {}
        ev_turn_id = ev.get("turn_id") or (meta.get("turn_id") if isinstance(meta, dict) else "") or ""
        ev_origin = meta.get("pose_origin", "") if isinstance(meta, dict) else ""
        writer.writerow(
            [
                "event",
                recording_name,
                segment_index,
                run_id,
                ev.get("event_type", ""),
                ev.get("subject", ""),
                _fmt_val(val),
                unit,
                ev.get("status", "measured"),
                bracket,
                ev.get("reason", ""),
                ev_turn_id,
                ev_origin,
                "",
                "",
                "",
                "",
            ]
        )

    return buf.getvalue()
