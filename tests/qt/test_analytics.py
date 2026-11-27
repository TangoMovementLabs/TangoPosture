"""Portable recording analytics load from indexed archives, without a database."""

import csv
import json
import socket
import sqlite3
import xml.etree.ElementTree as ET
from itertools import pairwise
from zipfile import ZipFile

import pytest
from PySide6.QtWidgets import QApplication

from tango_posture.archive import append_segment, read_analytics, store_analytics
from tango_posture.calibration.archives import save_range_profile
from tango_posture.calibration.baseline import profile
from tango_posture.calibration.measurements import measure_ranges
from tango_posture.calibration.models import (
    METRIC_DETAILS,
    ApprovedRange,
    Passage,
    RangeProfile,
)
from tango_posture.config import Config, RangeDefault
from tango_posture.desktop_qt.analytics.data import (
    FRAME_METRICS,
    _evaluate_series,
    compare_archive_with_calibration,
    load_archive_dataset,
    load_session,
    session_payload,
)
from tango_posture.desktop_qt.components.metric_chart import STATE_COLORS, MetricChart
from tango_posture.desktop_qt.router import Route, Router
from tango_posture.desktop_qt.screens.analytics import AnalyticsScreen
from tango_posture.desktop_qt.service_bridge import ServiceBridge
from tango_posture.domain import (
    METRICS,
    FrameObservation,
    Landmark,
    MetricObservation,
    QualityResult,
)
from tango_posture.models import MANIFEST
from tango_posture.references.archives import mark_reference, reference_directory
from tango_posture.references.coach import _serialise_pose
from tango_posture.session import Session
from tango_posture.storage import Storage


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(["-platform", "offscreen"])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Analytics must not use the network")
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def payload(tmp_path, reference, cfg):
    storage = Storage(tmp_path)
    storage.save_calibration(reference)
    session = Session(reference, cfg, storage, 10.0, "fixture", {"metric_version": "1.0"})
    for timestamp in (10.0, 10.1, 10.2, 10.3, 11.0, 11.1):
        valid = timestamp != 10.2
        session.update(MetricObservation(
            timestamp,
            {metric: reference.baseline[metric] - 0.2 if valid else None for metric in METRICS},
            {metric: QualityResult(valid, "" if valid else "occluded") for metric in METRICS},
        ))
    summary = session.finish(11.2)
    data = session_payload(tmp_path, session.id)
    stored = load_session(tmp_path, session.id)
    storage.close()
    return data, stored, summary


def make_archive(root, payload, *, count=1):
    path = root / "practice" / "practice.tp"
    report = root / "report.md"
    report.write_text(f"| Session ID | `{payload['id']}` |\n")
    for _ in range(count):
        append_segment(
            path, report, None, None, [10.0, 10.1, 10.2, 10.3, 11.0, 11.1],
            analytics={"session": payload},
        )
    return path


def test_missing_archives_does_not_create_storage(qapp, tmp_path):
    root = tmp_path / "not-created"
    screen = AnalyticsScreen(ServiceBridge(Config(data_dir=str(root))), Router())
    assert not root.exists()
    assert "No Library practice archives" in screen.state_label.text()
    assert screen.dataset is None
    assert not screen.align_btn.isEnabled()
    screen.close()


def test_indexed_analytics_survive_portable_copy_without_database(tmp_path, payload, monkeypatch):
    data, original, summary = payload
    path = make_archive(tmp_path, data, count=2)
    copied = tmp_path / "portable.tp"
    copied.write_bytes(path.read_bytes())
    def no_database(*args):
        pytest.fail("Displaying archived analytics must not read the database")
    monkeypatch.setattr('tango_posture.desktop_qt.analytics.data._connect', no_database)
    for index in (0, 1):
        loaded = load_archive_dataset(copied, index)
        assert loaded.identifier == original.identifier
        for metric in METRICS:
            assert loaded.metrics[metric] == original.metrics[metric]
            assert [len(segment) for segment in loaded.metrics[metric].segments] == [2, 1, 2]
        assert loaded.active_duration == summary["active_duration"]
    with ZipFile(copied) as archive:
        assert {"analytics_00.json", "analytics_01.json"} <= set(archive.namelist())


def test_screen_selects_archive_recording_and_seeks_its_timestamp(qapp, tmp_path, payload):
    data, _, _ = payload
    path = make_archive(tmp_path, data, count=2)
    router = Router()
    navigations = []
    router.route_changed.connect(lambda route, payload: navigations.append((route, payload)))
    screen = AnalyticsScreen(ServiceBridge(Config(data_dir=str(tmp_path))), router)
    assert screen.dataset is None
    screen.session_combo.setCurrentIndex(2)
    assert screen.dataset is not None
    screen._seek_to_recording(10.1)
    assert navigations[-1][1] == {
        "archive_path": path.resolve(), "segment_index": 1, "timestamp": 10.1,
    }
    assert "Delta samples: 5/6" in screen.coverage_labels["head_forward"].text()
    screen.close()


def test_saved_frame_measurements_display_without_reference(qapp, tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# Imported clip\n")
    path = tmp_path / "practice" / "import.tp"
    append_segment(path, report, None, None, [0.0, 0.1, 0.2], analytics={
        "max_gap": 0.5,
        "frame_timestamps": [0.0, 0.1, 0.2],
        "coach_data": {"frames": {
            "0": {"head_angle": -5, "knee_bend_deg": 8, "axis_curve_smoothed": 0.02},
            "1": {"head_angle": None, "knee_bend_deg": None},
            "2": {"head_angle": -4, "knee_bend_deg": 10},
        }},
    })
    screen = AnalyticsScreen(ServiceBridge(Config(data_dir=str(tmp_path))), Router())
    screen.session_combo.setCurrentIndex(1)
    assert screen.dataset is not None
    assert screen.dataset.metrics["head_angle"].points == [(0, -5), (0.1, None), (0.2, -4)]
    assert [len(part) for part in screen.dataset.metrics["head_angle"].segments] == [1, 1]
    assert screen.dataset.metrics["axis_curve_smoothed"].points[0] == (0, 2)
    assert not screen.charts["head_forward"].isEnabled()
    assert "Stored measurement samples: 2/3" in screen.coverage_labels["knee_bend_deg"].text()
    screen.close()


def test_reanalysis_preserves_other_indices_and_original_session(tmp_path, payload):
    data, _, _ = payload
    path = make_archive(tmp_path, data, count=2)
    before = read_analytics(path, 1)
    store_analytics(path, 0, {"coach_data": {"frames": {"0": {"head_angle": 3}}}})
    assert read_analytics(path, 1) == before
    assert read_analytics(path, 0)["session"] == data


def test_unanalyzed_archive_shows_actionable_status(qapp, tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# Old recording\n")
    append_segment(tmp_path / "practice" / "old.tp", report, None, None, [])
    screen = AnalyticsScreen(ServiceBridge(Config(data_dir=str(tmp_path))), Router())
    screen.session_combo.setCurrentIndex(1)
    assert screen.dataset is None
    assert "Analyze it in Review" in screen.state_label.text()
    screen.close()


@pytest.mark.parametrize(("statement", "reason"), [
    ("UPDATE sessions SET status='running'", "not finalized"),
    ("UPDATE calibrations SET metric_version='2.0'", "metric semantics"),
    ('UPDATE sessions SET versions=\'{"metric_version":"2.0"}\'', "metric semantics"),
    ('UPDATE sessions SET summary=\'{"schema_version":2}\'', "summary schema"),
    ("UPDATE sessions SET config='[]'", "Invalid persisted session metadata"),
    ("UPDATE observations SET metric='legacy_head_angle'", "no compatible persisted"),
    ("UPDATE observations SET timestamp=NULL", "invalid observation timestamp"),
    ("UPDATE observations SET timestamp=10", "duplicate observation timestamps"),
])
def test_migration_rejects_incompatible_database_evidence(tmp_path, payload, statement, reason):
    data, _, _ = payload
    with sqlite3.connect(tmp_path / "tango.sqlite3") as db:
        db.execute(statement)
    with pytest.raises((ValueError, TypeError), match=reason):
        session_payload(tmp_path, data["id"])


def test_legacy_projection_keeps_unknown_nonfinite_and_zero_distinct(tmp_path, payload):
    data, _, _ = payload
    with sqlite3.connect(tmp_path / "tango.sqlite3") as db:
        db.execute("UPDATE observations SET quality_state='UNKNOWN' WHERE timestamp=10.1")
        db.execute("UPDATE observations SET delta=? WHERE timestamp=11.0", (float("inf"),))
        db.execute("UPDATE observations SET delta=0 WHERE timestamp=11.1")
    for metric in load_session(tmp_path, data["id"]).metrics.values():
        assert metric.points[1][1] is None
        assert metric.points[4][1] is None
        assert metric.points[5][1] == 0
        assert [len(segment) for segment in metric.segments] == [1, 1, 1]


def _full_pose_observation(timestamp: float, *, ear_dx: float = 0.0, shoulder_spread: float = 10.0):
    width, height = 640, 480
    coords = {
        "nose": (350 + ear_dx, 90),
        "left_ear": (320 + ear_dx, 100),
        "right_ear": (320 + shoulder_spread + ear_dx, 100),
        "left_shoulder": (320, 160),
        "right_shoulder": (320 + shoulder_spread, 160),
        "left_hip": (320, 320),
        "right_hip": (320 + shoulder_spread, 320),
        "left_knee": (335, 385),
        "right_knee": (335 + shoulder_spread, 385),
        "left_ankle": (320, 450),
        "right_ankle": (320 + shoulder_spread, 450),
    }
    return FrameObservation(
        timestamp,
        width,
        height,
        [{k: Landmark(x / width, y / height, 0.99, 0.99) for k, (x, y) in coords.items()}],
    )


def _make_pose_archive(
    archive_path,
    report_path,
    timestamps,
    observations,
    capture_profile,
    *,
    is_reference: bool = False,
):
    coach_frames = {}
    for index, (ts, obs) in enumerate(zip(timestamps, observations, strict=True)):
        raw_pose = _serialise_pose(obs.poses[0])
        coach_frames[str(index)] = {
            "timestamp": ts,
            "poses": [raw_pose],
            "measurement_poses": [raw_pose],
            "head_angle": -6.0 + index * 0.5,
            "shoulder_elev_pct": 2.0 + index * 0.5,
            "knee_bend_deg": 18.0 + index * 0.2,
            "axis_curve_smoothed": 0.01 * index,
        }
    report_path.write_text("# Recording\n", encoding="utf-8")
    target_path = archive_path.with_suffix(".tp") if is_reference else archive_path
    append_segment(
        target_path,
        report_path,
        None,
        None,
        timestamps,
        analytics={
            "max_gap": 0.5,
            "model_hash": MANIFEST["sha256"],
            "capture_profile": capture_profile,
            "frame_timestamps": timestamps,
            "coach_data": {
                "width": 640,
                "height": 480,
                "model_hash": MANIFEST["sha256"],
                "capture_profile": capture_profile,
                "frames": coach_frames,
            },
        },
    )
    if is_reference:
        mark_reference(target_path, "Teacher Demo", {"camera_view": capture_profile})
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.replace(archive_path)


def _full_range_profile(cfg: Config, capture_profile: dict, *, baseline_obs: FrameObservation):
    measured = measure_ranges(baseline_obs, cfg.model_copy(update={"allow_all_angles": True}))
    metrics = {}
    for key, (_part, _label, unit) in METRIC_DETAILS.items():
        val = float(measured.values[key] or 0.0)
        tol = 0.05 if key in ("head_forward", "hip_offset", "shoulder_clearance") else 5.0
        metrics[key] = ApprovedRange(
            lower=val - tol,
            upper=val + tol,
            unit=unit,
            view="side",
            passages=[Passage(start=0.0, end=2.0)],
            median=val,
            p10=val - tol * 0.5,
            p90=val + tol * 0.5,
            valid_duration=2.0,
            selected_duration=2.0,
            samples=10,
            inside_fraction=1.0,
        )
    return RangeProfile(
        name="Full Posture Reference",
        context="Solo walking",
        model_hash=MANIFEST["sha256"],
        capture_profile=capture_profile,
        source={"archive": "ref.tpr"},
        metrics=metrics,
    )


def test_analytics_loads_tpr_and_tp_with_all_metrics_and_tpc_comparison(qapp, tmp_path):
    cfg = Config(
        data_dir=str(tmp_path),
        entry_duration=0.4,
        clear_duration=0.2,
        smoothing=0.0001,
        max_gap=0.5,
    )
    capture_profile = profile(cfg, 640, 480, "video")
    timestamps = [round(i * 0.2, 2) for i in range(10)]
    observations = [
        _full_pose_observation(ts, ear_dx=0.0 if i < 4 else 40.0)
        for i, ts in enumerate(timestamps)
    ]

    report_path = tmp_path / "report.md"
    tp_path = tmp_path / "practice" / "Solo.tp"
    tpr_path = reference_directory(str(tmp_path)) / "RefDemo.tpr"
    _make_pose_archive(tp_path, report_path, timestamps, observations, capture_profile)
    _make_pose_archive(
        tpr_path, report_path, timestamps, observations, capture_profile, is_reference=True
    )

    cal_profile = _full_range_profile(cfg, capture_profile, baseline_obs=observations[0])
    cal_path = tmp_path / "calibrations" / "Full.tpc"
    save_range_profile(cal_path, cal_profile)

    screen = AnalyticsScreen(ServiceBridge(cfg), Router())
    labels = [screen.session_combo.itemText(i) for i in range(screen.session_combo.count())]
    assert any("(Reference .tpr)" in label for label in labels)
    assert any("Solo" in label and ".tpr" not in label for label in labels)

    tpr_index = next(
        i for i in range(screen.session_combo.count())
        if "(Reference .tpr)" in screen.session_combo.itemText(i)
    )
    screen.session_combo.setCurrentIndex(tpr_index)
    assert screen.dataset is not None
    assert "Reference (.tpr)" in screen.state_label.text()

    for metric in (*METRIC_DETAILS, *FRAME_METRICS):
        assert screen.charts[metric].isEnabled(), f"Expected chart {metric} to be enabled"
        assert screen.export_buttons[metric].isEnabled()
    assert screen.export_charts_btn.isEnabled()

    cal_idx = screen.calibration_combo.findData(str(cal_path.resolve()))
    assert cal_idx > 0
    screen.calibration_combo.setCurrentIndex(cal_idx)

    head_chart = screen.charts["head_forward"]
    target = cal_profile.metrics["head_forward"]
    assert head_chart.bounds == (target.lower, target.upper)
    assert head_chart.reference_line == pytest.approx(target.median)
    assert "WITHIN_REFERENCE" in head_chart.states.values()
    assert "PENDING" in head_chart.states.values()
    assert "DEVIATION" in head_chart.states.values()
    assert not hasattr(screen, "event_table")
    assert "Range:" in screen.coverage_labels["head_forward"].text()
    assert "Ref:" in screen.coverage_labels["head_forward"].text()
    assert "Within range:" in screen.coverage_labels["head_forward"].text()

    # Exercise painting on offscreen widget + image render
    rendered = head_chart.render_to_image(title="Head", subtitle="Sub")
    assert rendered.width() == 920 and rendered.height() == 240
    screen.close()


def test_analytics_view_mismatch_falls_back_to_all_angles_and_supports_defaults(qapp, tmp_path):
    cfg = Config(
        data_dir=str(tmp_path),
        smoothing=0.0001,
        max_gap=0.5,
        calibration_defaults={
            "head_pitch": RangeDefault(lower=-20.0, upper=20.0, unit="degrees", view="all"),
        },
    )
    capture_profile = profile(cfg, 640, 480, "video")
    timestamps = [0.0, 0.2, 0.4, 0.6]
    # Mixed side + frontal view frames (wide shoulder spread on 2 frames) vs side target view
    observations = [
        _full_pose_observation(ts, shoulder_spread=10.0 if i % 2 == 0 else 160.0)
        for i, ts in enumerate(timestamps)
    ]
    ext_tpr = tmp_path / "external" / "FrontalClip.tpr"
    _make_pose_archive(
        ext_tpr, tmp_path / "rep.md", timestamps, observations, capture_profile, is_reference=True
    )
    cal_profile = _full_range_profile(
        cfg,
        capture_profile,
        baseline_obs=_full_pose_observation(0.0, shoulder_spread=10.0),
    )
    cal_path = tmp_path / "calibrations" / "SideOnly.tpc"
    save_range_profile(cal_path, cal_profile)

    screen = AnalyticsScreen(ServiceBridge(cfg), Router())
    screen.load_archive(ext_tpr, calibration_path=cal_path)
    assert screen.dataset is not None
    assert "all-angle measurement" in screen.state_label.text()
    assert screen.charts["head_forward"].isEnabled()
    head_points = screen.charts["head_forward"].data.points
    assert len(head_points) == len(timestamps)
    assert all(val is not None for _, val in head_points)

    defaults_idx = screen.calibration_combo.findData("__defaults__")
    screen.calibration_combo.setCurrentIndex(defaults_idx)
    assert screen.charts["head_pitch"].bounds == (-20.0, 20.0)
    assert screen.charts["head_pitch"].reference_line == pytest.approx(0.0)
    assert screen.charts["head_angle"].bounds == (-20.0, 20.0)
    screen.close()


def test_analytics_export_individual_and_all_charts_to_files(qapp, tmp_path):
    cfg = Config(data_dir=str(tmp_path), smoothing=0.0001, max_gap=0.5)
    capture_profile = profile(cfg, 640, 480, "video")
    timestamps = [0.0, 0.2, 0.4, 0.6]
    observations = [_full_pose_observation(ts, ear_dx=i * 5.0) for i, ts in enumerate(timestamps)]
    tpr_path = reference_directory(str(tmp_path)) / "ExportRef.tpr"
    _make_pose_archive(
        tpr_path, tmp_path / "rep.md", timestamps, observations, capture_profile, is_reference=True
    )
    cal_profile = _full_range_profile(cfg, capture_profile, baseline_obs=observations[0])
    cal_path = tmp_path / "calibrations" / "ExportCal.tpc"
    save_range_profile(cal_path, cal_profile)

    screen = AnalyticsScreen(ServiceBridge(cfg), Router())
    screen.load_archive(tpr_path, calibration_path=cal_path)

    out_dir = tmp_path / "exports"
    single_png = screen.export_chart("head_pitch", out_dir / "head_pitch.png")
    single_svg = screen.export_chart("head_pitch", out_dir / "head_pitch.svg")
    single_csv = screen.export_chart("head_pitch", out_dir / "head_pitch.csv")
    single_json = screen.export_chart("head_pitch", out_dir / "head_pitch.json")

    assert single_png is not None and single_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert single_svg is not None
    svg_root = ET.fromstring(single_svg.read_text(encoding="utf-8"))
    assert svg_root.tag.endswith("svg")
    svg_text = single_svg.read_text(encoding="utf-8")
    assert "Head pitch" in svg_text
    assert "Min " in svg_text and "Max " in svg_text and "Ref " in svg_text

    assert single_csv is not None
    with single_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(timestamps)
    assert set(rows[0]) == {
        "timestamp_s", "value", "state", "min_bound", "max_bound", "reference_line"
    }

    assert single_json is not None
    single_payload = json.loads(single_json.read_text(encoding="utf-8"))
    assert single_payload["metric"] == "head_pitch"
    assert len(single_payload["samples"]) == len(timestamps)

    all_png = screen.export_all_charts(out_dir / "all_charts.png")
    all_svg = screen.export_all_charts(out_dir / "all_charts.svg")
    all_csv = screen.export_all_charts(out_dir / "all_charts.csv")
    all_json = screen.export_all_charts(out_dir / "all_charts.json")

    assert all_png is not None and all_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert all_svg is not None
    all_svg_root = ET.fromstring(all_svg.read_text(encoding="utf-8"))
    assert all_svg_root.tag.endswith("svg")
    all_svg_text = all_svg.read_text(encoding="utf-8")
    assert "Knee" in all_svg_text and "Head pitch" in all_svg_text

    assert all_csv is not None
    with all_csv.open(encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    exported_metrics = {row["metric"] for row in all_rows}
    assert set(METRIC_DETAILS) <= exported_metrics
    assert set(FRAME_METRICS) <= exported_metrics

    assert all_json is not None
    all_payload = json.loads(all_json.read_text(encoding="utf-8"))
    assert all_payload["calibration"] == "Full Posture Reference"
    assert set(METRIC_DETAILS) <= set(all_payload["metrics"])
    assert set(FRAME_METRICS) <= set(all_payload["metrics"])
    screen.close()


def test_analytics_defaults_and_tpc_include_head_forward_inclination_and_trunk(tmp_path):
    cfg = Config(data_dir=str(tmp_path), smoothing=0.0001, max_gap=0.5)
    capture_profile = profile(cfg, 640, 480, "video")
    timestamps = [0.0, 0.2, 0.4, 0.6]
    observations = [_full_pose_observation(ts, ear_dx=i * 8.0) for i, ts in enumerate(timestamps)]
    tpr_path = reference_directory(str(tmp_path)) / "DefaultsRef.tpr"
    _make_pose_archive(
        tpr_path, tmp_path / "rep.md", timestamps, observations, capture_profile, is_reference=True
    )
    dataset = load_archive_dataset(tpr_path, 0, cfg)

    defaults_comp = compare_archive_with_calibration(tpr_path, 0, dataset, cfg, "__defaults__")
    assert defaults_comp.metrics["head_forward"].bounds == (-cfg.head_forward.entry, cfg.head_forward.entry)
    assert defaults_comp.metrics["head_forward"].reference_line == 0.0
    assert defaults_comp.metrics["head_inclination"].bounds == (
        -cfg.head_inclination.entry,
        cfg.head_inclination.entry,
    )
    assert defaults_comp.metrics["head_inclination"].reference_line == 0.0
    assert defaults_comp.metrics["trunk_inclination"].bounds == (
        -cfg.trunk_inclination.entry,
        cfg.trunk_inclination.entry,
    )
    assert defaults_comp.metrics["trunk_inclination"].reference_line == 0.0

    cal_profile = _full_range_profile(cfg, capture_profile, baseline_obs=observations[0])
    profile_comp = compare_archive_with_calibration(tpr_path, 0, dataset, cfg, cal_profile)
    for key in ("head_forward", "head_inclination", "trunk_inclination"):
        target = cal_profile.metrics[key]
        assert profile_comp.metrics[key].bounds == (target.lower, target.upper)
        assert profile_comp.metrics[key].reference_line == pytest.approx(target.median)


def test_analytics_in_range_points_and_boundary_segments_are_not_red_or_orange(qapp):
    cfg = Config(smoothing=0.0001, entry_duration=0.4, clear_duration=0.6, max_gap=0.5)
    bounds = (-0.10, 0.10)
    # Leaves range long enough to trigger DEVIATION (0.8s..1.4s), then re-enters inside [-0.10, 0.10]
    raw_points = [
        (0.0, 0.0),
        (0.2, 0.05),
        (0.4, 0.15),
        (0.6, 0.18),
        (0.8, 0.20),
        (1.0, 0.22),
        (1.2, 0.04),  # re-entered while clear_duration hysteresis is still running
        (1.4, 0.02),
    ]
    data, states, _ = _evaluate_series(
        "head_forward", raw_points, cfg, 0.0, cfg.max_gap, bounds=bounds
    )
    for ts, val in data.points:
        assert val is not None
        if bounds[0] <= val <= bounds[1]:
            assert states[ts] == "WITHIN_REFERENCE"
        else:
            assert states[ts] in ("PENDING", "DEVIATION")

    chart = MetricChart()
    chart.set_data(data, 1.4, 0.0, bounds=bounds, reference_line=0.0, states=states)
    assert len(data.segments) == 1
    for (t0, v0), (t1, v1) in zip(data.segments[0][:-1], data.segments[0][1:], strict=True):
        pieces = chart._segment_pieces(t0, v0, t1, v1)
        assert pieces
        for _ta, va, _tb, vb, color in pieces:
            vm = 0.5 * (va + vb)
            if bounds[0] + 1e-6 < vm < bounds[1] - 1e-6:
                assert color == STATE_COLORS["WITHIN_REFERENCE"]
            elif vm > bounds[1] + 1e-6 or vm < bounds[0] - 1e-6:
                assert color in (STATE_COLORS["PENDING"], STATE_COLORS["DEVIATION"])

    svg = chart.svg_fragment(title="Head Forward", subtitle="Calibration")
    root = ET.fromstring(svg)
    y_limits = chart._y_range()
    assert y_limits is not None
    low_y, high_y = y_limits
    plot_top = 46.0 + 12.0
    plot_height = (240.0 - 46.0) - 50.0
    plot_bottom = plot_top + plot_height
    y_upper_bound = plot_bottom - (bounds[1] - low_y) / (high_y - low_y) * plot_height
    y_lower_bound = plot_bottom - (bounds[0] - low_y) / (high_y - low_y) * plot_height
    band_min_y = min(y_upper_bound, y_lower_bound)
    band_max_y = max(y_upper_bound, y_lower_bound)
    for line in root.findall(".//line"):
        stroke = line.attrib.get("stroke", "")
        if stroke in (STATE_COLORS["DEVIATION"], STATE_COLORS["PENDING"]):
            y1 = float(line.attrib["y1"])
            y2 = float(line.attrib["y2"])
            assert not (band_min_y + 0.5 < y1 < band_max_y - 0.5 and band_min_y + 0.5 < y2 < band_max_y - 0.5)
    chart.close()


def test_metric_chart_hover_shows_timestamp_and_value_and_settings_refresh(
    qapp, tmp_path, monkeypatch
):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from tango_posture.desktop_qt.analytics.data import MetricData
    from tango_posture.desktop_qt.shell import MainWindow

    monkeypatch.chdir(tmp_path)

    # 1. Hovering over a MetricChart shows timestamp (s) and value in hover hint & tooltip
    chart = MetricChart()
    chart.resize(640, 240)
    points = [(0.0, 5.0), (1.0, 12.5), (2.0, 7.0)]
    chart.set_data(
        MetricData(points, [points], {}, 0, 0),
        time_end=2.0,
        origin_timestamp=10.0,
        unit="degrees",
    )
    assert chart.hasMouseTracking()
    plot_left = 62.0
    plot_right = float(chart.width() - 20)
    mid_pos = QPointF(plot_left + 0.5 * (plot_right - plot_left), 100.0)
    move_ev = QMouseEvent(
        QEvent.Type.MouseMove,
        mid_pos,
        chart.mapToGlobal(mid_pos),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    chart.mouseMoveEvent(move_ev)
    assert "11.00 s" in chart.hover_hint_text
    assert "12.5 degrees" in chart.hover_hint_text
    assert chart.toolTip() == chart.hover_hint_text
    # Ensure paint with hover badge does not crash
    img = chart.grab()
    assert img.width() == 640

    chart.leaveEvent(QEvent(QEvent.Type.Leave))
    assert chart.hover_hint_text == ""
    assert chart.toolTip() == ""
    chart.close()

    # 2. Updating & saving Settings persists config.toml + settings.toml and refreshes Analytics charts
    cfg = Config(data_dir=str(tmp_path / "data"), smoothing=0.0001, max_gap=0.5)
    capture_profile = profile(cfg, 640, 480, "video")
    timestamps = [round(i * 0.2, 2) for i in range(8)]
    observations = [_full_pose_observation(ts, ear_dx=(i % 2) * 35.0) for i, ts in enumerate(timestamps)]
    tpr_path = reference_directory(cfg.data_dir) / "SettingsRefreshRef.tpr"
    _make_pose_archive(
        tpr_path, tmp_path / "rep.md", timestamps, observations, capture_profile, is_reference=True
    )

    window = MainWindow(cfg)
    try:
        analytics = window.screens[Route.ANALYTICS] if False else None  # typing resolved at runtime
        window.router.navigate(Route.ANALYTICS)
        qapp.processEvents()
        analytics = window.screens[Route.ANALYTICS]
        assert isinstance(analytics, AnalyticsScreen)
        idx = next(
            i for i in range(analytics.session_combo.count())
            if "(Reference .tpr)" in analytics.session_combo.itemText(i)
        )
        analytics.session_combo.setCurrentIndex(idx)
        qapp.processEvents()

        before_vals = [v for _, v in analytics.charts["head_forward"].data.points if v is not None]
        before_jitter = sum(abs(b - a) for a, b in pairwise(before_vals))

        window.router.navigate(Route.SETTINGS)
        qapp.processEvents()
        settings = window.screens[Route.SETTINGS]
        settings.smooth_spin.setValue(0.60)
        settings.save_config_btn.click()
        qapp.processEvents()

        assert (tmp_path / "config.toml").is_file()
        assert (tmp_path / "settings.toml").is_file()
        assert window.bridge.config.smoothing == pytest.approx(0.60)

        after_vals = [v for _, v in analytics.charts["head_forward"].data.points if v is not None]
        after_jitter = sum(abs(b - a) for a, b in pairwise(after_vals))
        assert after_jitter < before_jitter * 0.7
    finally:
        window.close()
