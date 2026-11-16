import json
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import frame
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from tango_posture.archive import append_segment, read_analytics, read_manifest, store_analytics
from tango_posture.calibration.archives import read_range_profile
from tango_posture.calibration.baseline import profile
from tango_posture.calibration.evidence import single_dancer_anchors
from tango_posture.calibration.measurements import measure_ranges
from tango_posture.calibration.models import METRIC_DETAILS, ApprovedRange, Passage
from tango_posture.config import Config, export_toml
from tango_posture.desktop_qt.calibration.editor import CalibrationEditor
from tango_posture.desktop_qt.calibration.guided import GuidedCalibration
from tango_posture.desktop_qt.calibration.preview import COLORS, render_approved_feedback
from tango_posture.desktop_qt.router import Router
from tango_posture.desktop_qt.screens.library import LibraryScreen
from tango_posture.desktop_qt.screens.settings import SettingsScreen
from tango_posture.desktop_qt.service_bridge import ServiceBridge
from tango_posture.domain import Landmark
from tango_posture.geometry import torso_axis_alignment
from tango_posture.models import MANIFEST
from tango_posture.overlay import screen_point
from tango_posture.references.archives import promote_reference


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(["-platform", "offscreen"])


@pytest.fixture
def cached_reference(tmp_path):
    video, report, practice = (tmp_path / name for name in ("video.mp4", "report.md", "practice.tp"))
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (640, 480))
    assert writer.isOpened()
    entries = []
    for index in range(40):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        pose = frame(index / 10).poses[0]
        entries.append({"poses": [
            {name: [p.x, p.y, p.z, p.visibility, p.presence] for name, p in pose.items()}
        ]})
    writer.release()
    report.write_text("# Cached source\n")
    append_segment(practice, report, video, None, [i / 10 for i in range(40)], analytics={
        "model_hash": MANIFEST["sha256"],
        "capture_profile": profile(Config(), 640, 480, "video"),
        "frames": entries, "max_gap": 0.5,
    })
    reference = tmp_path / "references" / "Reference.tpr"
    promote_reference(practice, reference, "Teacher reference")
    return reference


def test_shared_editor_reuses_cache_and_saves_independent_profiles(qapp, tmp_path, cached_reference, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Range editing must not rerun inference")
    monkeypatch.setattr("tango_posture.app.MediaPipeEstimator", forbidden)
    cfg = Config(data_dir=str(tmp_path))
    saved = []
    for name in ("Solo walking", "Close embrace"):
        editor = CalibrationEditor(cached_reference, cfg)
        editor.show()
        qapp.processEvents()
        assert editor._stats is not None
        assert not editor.live_check.isEnabled()
        editor._save()
        assert editor.saved_path is None
        editor.passages_input.setText("0-1; 2-3")
        editor.lower_spin.setValue(-100)
        editor.upper_spin.setValue(100)
        editor._approve()
        editor.name_input.setText(name)
        editor.context_combo.setCurrentText(name)
        editor._save()
        assert editor.saved_path is not None
        saved.append(read_range_profile(editor.saved_path))
    assert saved[0].id != saved[1].id
    assert saved[0].metrics["head_pitch"].lower == -100
    assert len(saved[0].metrics["head_pitch"].passages) == 2
    assert not saved[0].metrics["head_pitch"].camera_check_passed


def test_editor_uses_cached_review_layers_and_interpolation(qapp, cached_reference, monkeypatch):
    analytics = read_analytics(cached_reference, 0)
    entries = analytics["frames"]
    coach = {
        "width": 640, "height": 480,
        "frames": {
            str(index): {
                "timestamp": index / 10,
                "measurement_poses": [] if index == 2 else entry["poses"],
                "message": "BEND YOUR KNEES",
            }
            for index, entry in enumerate(entries)
        },
    }
    movement = {"steps": [{"frame": 1}], "pivots": [{"start_frame": 1, "end_frame": 5}]}
    store_analytics(cached_reference, 0, {"coach_data": coach, "movement_data": movement})
    before = cached_reference.read_bytes()
    seen = {}

    def forbidden(*args, **kwargs):
        raise AssertionError("Calibration must not rerun inference")

    def skeleton(image, data, index, cfg, **kwargs):
        seen["entry"] = data["frames"][str(index)]
        seen["interpolation"] = cfg.replay_joint_interpolation
        seen["lines_only"] = kwargs["lines_only"]
        return image

    def movement_overlay(image, data, index, fps, **kwargs):
        seen["movement"] = data
        seen["movement_options"] = kwargs
        return image

    monkeypatch.setattr("tango_posture.references.pipeline.MediaPipeEstimator", forbidden)
    monkeypatch.setattr(
        "tango_posture.desktop_qt.calibration.editor.render_coach_frame_overlay", skeleton
    )
    monkeypatch.setattr(
        "tango_posture.desktop_qt.calibration.editor.render_movement_overlay", movement_overlay
    )
    editor = CalibrationEditor(cached_reference, Config(
        replay_joint_interpolation="quadratic", replay_step_history_seconds=2.5,
    ))
    try:
        editor._seek_frame(2)
        assert seen["entry"]["measurement_poses"] == []
        assert seen["entry"]["display_poses"]
        assert seen["interpolation"] == "quadratic"
        assert seen["lines_only"]
        assert seen["movement"] == movement
        assert seen["movement_options"]["show_pivots"]
        assert seen["movement_options"]["show_steps"]
        assert seen["movement_options"]["step_history_seconds"] == 2.5
        assert seen["movement_options"]["frame_timestamp"] == pytest.approx(0.2)
        assert not editor.evidence.frames[2].poses
        assert cached_reference.read_bytes() == before
    finally:
        editor.reject()


def test_approved_feedback_tracks_all_metrics_and_draft_lifecycle(qapp, cached_reference, monkeypatch):
    seen = []

    def render(image, observation, cfg, targets, rows):
        seen.append((set(targets), {row["metric"]: row["state"] for row in rows}))
        return render_approved_feedback(image, observation, cfg, targets, rows)

    monkeypatch.setattr("tango_posture.desktop_qt.calibration.editor.render_approved_feedback", render)
    editor = CalibrationEditor(cached_reference, Config(
        smoothing=0.0001, entry_duration=0.2, clear_duration=0.2,
    ))
    try:
        assert not seen
        assert editor._approved_coach_data is None
        value = editor._samples["head_pitch"][0][1]
        assert value is not None
        editor.lower_spin.setValue(-100)
        editor.upper_spin.setValue(value - 5)
        editor._approve()
        assert seen[-1] == ({"head_pitch"}, {"head_pitch": "PENDING"})
        editor.scrubber.setValue(3)
        assert seen[-1][1]["head_pitch"] == "DEVIATION"
        assert editor.chart.states[0.0] == "PENDING"
        assert editor.chart.states[0.3] == "DEVIATION"
        editor.scrubber.setValue(0)
        assert seen[-1][1]["head_pitch"] == "PENDING"

        editor.step_combo.setCurrentText("Shoulders")
        editor.lower_spin.setValue(-100)
        editor.upper_spin.setValue(100)
        editor._approve()
        assert seen[-1][0] == {"head_pitch", "shoulder_clearance"}
        assert seen[-1][1]["shoulder_clearance"] == "WITHIN_REFERENCE"

        editor.step_combo.setCurrentText("Head")
        original = editor.approved["head_pitch"].upper
        editor.upper_spin.setValue(100)
        assert not editor.chart.states
        editor.scrubber.setValue(3)
        assert editor.approved["head_pitch"].upper == original
        assert seen[-1][1]["head_pitch"] == "DEVIATION"
        editor._approve()
        assert seen[-1][1]["head_pitch"] == "WITHIN_REFERENCE"
        editor._skip()
        assert seen[-1][0] == {"shoulder_clearance"}
        editor._restart_track()
        assert not editor.approved
        assert editor._approved_coach_data is None
    finally:
        editor.reject()


def test_approved_feedback_stops_when_dancer_tracking_is_lost(qapp, cached_reference, monkeypatch):
    analytics = read_analytics(cached_reference, 0)
    entries = analytics["frames"]
    entries[2]["poses"] = []
    store_analytics(cached_reference, 0, {"frames": entries})
    seen = []

    def render(image, observation, cfg, targets, rows):
        seen.append((observation, rows))
        return render_approved_feedback(image, observation, cfg, targets, rows)

    monkeypatch.setattr("tango_posture.desktop_qt.calibration.editor.render_approved_feedback", render)
    editor = CalibrationEditor(cached_reference, Config(entry_duration=0.1))
    try:
        editor._approve()
        editor.scrubber.setValue(2)
        assert not seen[-1][0].poses
        assert seen[-1][1][0]["state"] == "UNKNOWN"
        editor.scrubber.setValue(3)
        assert not seen[-1][0].poses
        assert seen[-1][1][0]["state"] == "UNKNOWN"
    finally:
        editor.reject()


def test_approved_feedback_times_both_limits_and_recovery(qapp, cached_reference):
    entries = read_analytics(cached_reference, 0)["frames"]
    for index, entry in enumerate(entries):
        y = 100 if index < 4 else 90 if index < 8 else 70 if index < 12 else 90
        entry["poses"][0]["nose"][1] = y / 480 + 0.0000123
    store_analytics(cached_reference, 0, {"frames": entries})
    editor = CalibrationEditor(cached_reference, Config(
        smoothing=0.0001, entry_duration=0.2, clear_duration=0.2,
    ))
    try:
        editor.lower_spin.setValue(25)
        editor.upper_spin.setValue(45)
        editor._approve()
        compared = editor._approved_coach_data
        assert compared is not None
        rows = [compared["frames"][str(index)]["calibration_rows"][0] for index in range(15)]
        assert rows[0]["raw"] == pytest.approx(editor._samples["head_pitch"][0][1], abs=1e-10)
        assert [rows[i]["state"] for i in (0, 2, 4, 6, 8, 10, 12, 14)] == [
            "PENDING", "DEVIATION", "DEVIATION", "WITHIN_REFERENCE",
            "PENDING", "DEVIATION", "DEVIATION", "WITHIN_REFERENCE",
        ]
    finally:
        editor.reject()


@pytest.mark.parametrize("key", list(METRIC_DETAILS))
def test_approved_metric_geometry_and_unknown_abstention(key, monkeypatch):
    observation = frame()
    pose = observation.poses[0]
    pose["left_knee"] = Landmark(315 / 640, 380 / 480, 0.99, 0.99)
    pose["right_knee"] = Landmark(330 / 640, 380 / 480, 0.99, 0.99)
    cfg = Config()
    value = measure_ranges(observation, cfg).values[key]
    assert value is not None
    target = ApprovedRange(
        lower=-100, upper=100, unit=METRIC_DETAILS[key][2], view="side",
        passages=[Passage(start=0, end=1)], median=0, p10=0, p90=0,
        valid_duration=1, selected_duration=1, samples=2, inside_fraction=1,
    )
    drawn = []
    original_line = cv2.line

    def record_line(image, start, end, color, *args):
        drawn.append((start, end, color))
        return original_line(image, start, end, color, *args)

    monkeypatch.setattr(cv2, "line", record_line)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    row = {"metric": key, "state": "DEVIATION", "raw": value, "smoothed": value}
    render_approved_feedback(image, observation, cfg, {key: target}, [row])
    assert any(color == COLORS["DEVIATION"] for _, _, color in drawn)
    if key == "head_pitch":
        assert any(start[1] == end[1] and color == (255, 220, 0)
                   for start, end, color in drawn)
    if key == "torso_axis_tilt":
        axis = torso_axis_alignment(pose, 640, 480, cfg.confidence)
        assert axis is not None
        _, shoulder, hip = axis
        sx, sy = screen_point(shoulder.x, shoulder.y, 640, 480, False)
        hx, hy = screen_point(hip.x, hip.y, 640, 480, False)
        assert (
            (round(hx - 0.25 * (sx - hx)), round(hy - 0.25 * (sy - hy))),
            (round(sx + 0.25 * (sx - hx)), round(sy + 0.25 * (sy - hy))),
            (255, 140, 255),
        ) in drawn
    drawn.clear()
    row["state"] = "UNKNOWN"
    render_approved_feedback(image, observation, cfg, {key: target}, [row])
    assert not drawn


def test_approved_assessments_fit_small_video_and_show_every_metric(monkeypatch):
    observation = frame()
    target = ApprovedRange(
        lower=-100, upper=100, unit="degrees", view="side",
        passages=[Passage(start=0, end=1)], median=0, p10=0, p90=0,
        valid_duration=1, selected_duration=1, samples=2, inside_fraction=1,
    )
    targets = {key: target.model_copy(update={"unit": details[2]})
               for key, details in METRIC_DETAILS.items()}
    rows = [{"metric": key, "state": "UNKNOWN", "smoothed": None} for key in targets]
    labels = []
    original_text = cv2.putText

    def record_text(image, text, origin, font, scale, color, thickness, line_type):
        labels.append((text, origin, cv2.getTextSize(text, font, scale, thickness)[0]))
        return original_text(image, text, origin, font, scale, color, thickness, line_type)

    monkeypatch.setattr(cv2, "putText", record_text)
    render_approved_feedback(np.zeros((120, 240, 3), dtype=np.uint8),
                             observation, Config(), targets, rows)
    assert len(labels) == len(METRIC_DETAILS)
    assert all("[UNKNOWN]" in text and x + size[0] < 240 and y < 120
               for text, (x, y), size in labels)


def test_editor_chart_shows_limits_and_example_intervals(qapp, cached_reference):
    editor = CalibrationEditor(cached_reference, Config())
    try:
        editor.resize(1180, 800)
        editor.show()
        editor.passages_input.setText("0-1; 2-3")
        editor.lower_spin.setValue(-100)
        editor.upper_spin.setValue(100)
        editor._update_stats()
        qapp.processEvents()
        assert editor.chart.bounds == (-100, 100)
        assert editor.chart.intervals == [(0, 1), (2, 3)]
        image = editor.chart.grab().toImage()
        amber_pixels = sum(
            1 for y in range(image.height()) for x in range(image.width())
            if (color := image.pixelColor(x, y)).red() > 180
            and color.green() > 120 and color.blue() < 130
        )
        assert amber_pixels > 20
        width = image.width() - 82
        inside_x = round(62 + 0.5 / editor.chart.time_end * width)
        outside_x = round(62 + 1.5 / editor.chart.time_end * width)
        assert image.pixelColor(inside_x, 30) != image.pixelColor(outside_x, 30)
        editor._approve()
        editor.metric_combo.setCurrentIndex(1)
        editor.metric_combo.setCurrentIndex(0)
        assert editor.chart.bounds == (-100, 100)
        assert editor.chart.intervals == [(0, 1), (2, 3)]
    finally:
        editor.reject()


def test_calibration_chart_playhead_tracks_playback_and_seeking(qapp, tmp_path, cached_reference):
    excerpt = tmp_path / "references" / "ExcerptForPlayhead.tpr"
    promote_reference(
        tmp_path / "practice.tp", excerpt, "Excerpt", start_frame=5, end_frame=25,
    )
    editor = CalibrationEditor(excerpt, Config())
    try:
        editor.show()
        qapp.processEvents()
        assert editor.chart.origin_timestamp == pytest.approx(0.5)
        assert editor.chart.playhead_timestamp == pytest.approx(0.5)
        editor.play_button.click()
        assert editor.play_timer.isActive()
        editor._advance_frame()
        assert editor.chart.playhead_timestamp == pytest.approx(0.6)
        editor.play_button.click()
        editor.scrubber.setValue(12)
        assert editor.chart.playhead_timestamp == pytest.approx(1.7)
        editor.scrubber.setValue(3)
        assert editor.chart.playhead_timestamp == pytest.approx(0.8)
        editor.metric_combo.setCurrentIndex(1)
        assert editor.chart.playhead_timestamp == pytest.approx(0.8)
        qapp.processEvents()
        image = editor.chart.grab().toImage()
        x = round(62 + (0.8 - 0.5) / editor.chart.time_end * (image.width() - 82))
        assert any(
            (color := image.pixelColor(px, 40)).red() > 200
            and color.green() < 150 and color.blue() > 160
            for px in range(x - 2, x + 3)
        )
        target_x = round(62 + 0.5 * (editor.chart.width() - 82))
        QTest.mouseClick(editor.chart, Qt.MouseButton.LeftButton, pos=QPoint(target_x, 40))
        assert editor.scrubber.value() in (9, 10)
        assert editor.chart.playhead_timestamp == pytest.approx(
            editor.evidence.frames[editor.scrubber.value()].timestamp
        )
    finally:
        editor.reject()


def test_pivot_context_is_saved_without_changing_approved_ranges(qapp, tmp_path, cached_reference):
    editor = CalibrationEditor(cached_reference, Config(data_dir=str(tmp_path)))
    try:
        editor._approve()
        approved = editor.approved["head_pitch"].model_dump()
        editor.context_combo.setCurrentText("Pivot")
        editor._save()
        profile = read_range_profile(editor.saved_path)
        assert profile.context == "Pivot"
        assert profile.metrics["head_pitch"].model_dump() == approved
    finally:
        editor.reject()


def test_guided_pivot_context_does_not_start_walking_drill(qapp, tmp_path):
    dialog = GuidedCalibration(ServiceBridge(Config(data_dir=str(tmp_path))))
    try:
        dialog.context.setCurrentText("Pivot")
        assert dialog.context.currentText() == "Pivot"
        dialog._step = 4
        dialog._next()
        assert not dialog._walking
        assert dialog._review_requested
    finally:
        dialog.reject()


def test_editor_draws_cached_movement_without_running_analysis(qapp, cached_reference):
    movement = {
        "width": 640, "height": 480,
        "steps": [{"frame": 0, "time": 0.0, "hx": 170, "hy": 420,
                   "tx": 200, "ty": 370, "side": "left"}],
        "pivots": [{"start_frame": 0, "end_frame": 5, "direction": "clockwise"}],
        "hip_frames": [{"frame": 0, "lx": 250, "ly": 250,
                        "rx": 390, "ry": 250, "scale": 100}],
    }
    store_analytics(cached_reference, 0, {"movement_data": movement})
    editor = CalibrationEditor(cached_reference, Config())
    try:
        editor._seek_frame(0)
        with_movement = editor.stage._image.copy()
        editor._movement_data = None
        editor._seek_frame(0)
        assert editor.stage._image != with_movement
    finally:
        editor.reject()


def test_calibration_defaults_require_confirmation_and_round_trip_through_settings(
    qapp, tmp_path, cached_reference, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    cfg = Config(data_dir=str(tmp_path), camera=2, tango_bpm=135, calibration_defaults={
        "head_forward": {"lower": 0.01, "upper": 0.2, "unit": "torso_lengths", "view": "side"},
    })
    bridge = ServiceBridge(cfg)
    settings = SettingsScreen(bridge, Router())
    destination = tmp_path / "settings.toml"
    destination.write_text(export_toml(cfg))
    before = destination.read_bytes()
    editor = CalibrationEditor(cached_reference, cfg, bridge=bridge)
    try:
        editor.lower_spin.setValue(-23.123456)
        editor.upper_spin.setValue(17.654321)
        editor._approve()
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
        editor._use_defaults()
        assert destination.read_bytes() == before
        assert bridge.config == cfg
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
        editor.defaults_button.click()
        saved = Config.model_validate(tomllib.loads(destination.read_text()))
        assert export_toml(saved) == export_toml(bridge.config)
        assert "camera" not in tomllib.loads(destination.read_text())
        assert bridge.config.camera == 2 and saved.tango_bpm == 135
        assert set(saved.calibration_defaults) == {"head_pitch"}
        approved = editor.approved["head_pitch"]
        target = saved.calibration_defaults["head_pitch"]
        assert (target.lower, target.upper, target.unit, target.view) == (
            approved.lower, approved.upper, approved.unit, approved.view,
        )
        enabled, lower, upper, _view = settings.range_widgets["head_pitch"]
        assert enabled.isChecked() and lower.value() == target.lower
        assert not settings.range_widgets["head_forward"][0].isChecked()
        lower.setValue(-15)
        upper.setValue(12.5)
        settings.field_widgets["replay_joint_interpolation"].setCurrentText("quadratic")
        settings.field_widgets["replay_joint_max_gap"].setValue(1.25)
        settings.save_toml_btn.click()
        saved = Config.model_validate(tomllib.loads(destination.read_text()))
        assert saved.calibration_defaults["head_pitch"].lower == -15
        assert saved.calibration_defaults["head_pitch"].upper == 12.5
        assert saved.replay_joint_interpolation == "quadratic"
        assert saved.replay_joint_max_gap == 1.25
        assert bridge.config.camera == 2 and saved.tango_bpm == 135
    finally:
        editor.reject()
        settings.close()


def test_failed_default_settings_save_preserves_current_config_and_file(
    qapp, tmp_path, cached_reference, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    cfg = Config(data_dir=str(tmp_path))
    bridge = ServiceBridge(cfg)
    messages = []
    bridge.status_message.connect(messages.append)
    destination = tmp_path / "settings.toml"
    destination.write_text(export_toml(cfg))
    before = destination.read_bytes()
    editor = CalibrationEditor(cached_reference, cfg, bridge=bridge)
    try:
        editor._approve()
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)

        def fail(*args):
            raise OSError("Disk full")

        monkeypatch.setattr(bridge, "save_settings", fail)
        editor._use_defaults()
        assert bridge.config == cfg
        assert destination.read_bytes() == before
        assert "Could not save calibration defaults: Disk full" in messages
    finally:
        editor.reject()


def test_library_distinguishes_profiles_from_replayable_references(qapp, tmp_path, cached_reference):
    cfg = Config(data_dir=str(tmp_path))
    editor = CalibrationEditor(cached_reference, cfg)
    editor._approve()
    editor._save()
    bridge = ServiceBridge(cfg)
    assert not bridge.list_references()[0].scoring_ready
    screen = LibraryScreen(bridge, Router())
    paths = {entry["path"].suffix for entry in screen.practices}
    assert {".tpc", ".tpr"} <= paths
    row = next(i for i in range(screen.table.rowCount()) if screen.table.item(i, 4).text() == ".tpc")
    buttons = screen.table.cellWidget(row, 3).findChildren(QPushButton)
    assert buttons[0].text() == "Profile details"
    assert all(button.text() != "Open in Review" for button in buttons)
    assert read_manifest(editor.saved_path)["segments"] == []
    screen.close()


def test_reference_editor_approves_pink_axis_and_saves_it_as_settings_defaults(
    qapp, tmp_path, cached_reference, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    cfg = Config(data_dir=str(tmp_path))
    bridge = ServiceBridge(cfg)
    editor = CalibrationEditor(cached_reference, cfg, bridge=bridge)
    try:
        editor.step_combo.setCurrentText("Torso")
        assert editor.metric_combo.currentData() == "torso_axis_tilt"
        assert "pink line" in editor.metric_combo.currentText()
        assert "0 = upright" in editor.metric_hint.text()
        assert editor.metric_combo.findData("trunk_inclination") >= 0
        assert editor._stats["median"] == pytest.approx(0)
        editor.view_combo.setCurrentText("all")
        editor.lower_spin.setValue(0)
        editor.upper_spin.setValue(5)
        editor._approve()
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
        editor._use_defaults()
        saved_settings = Config.model_validate(tomllib.loads((tmp_path / "settings.toml").read_text()))
        assert saved_settings.calibration_defaults["torso_axis_tilt"].upper == 5
        editor._save()
        target = read_range_profile(editor.saved_path)
        assert set(target.metrics) == {"torso_axis_tilt"}
        assert target.metrics["torso_axis_tilt"].lower == 0
        assert target.metrics["torso_axis_tilt"].upper == 5
        assert target.metrics["torso_axis_tilt"].unit == "degrees"
        assert not target.metrics["torso_axis_tilt"].camera_check_passed
    finally:
        editor.reject()


def test_guided_dialog_never_opens_camera_until_requested(qapp, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Opening the dialog must not open camera/model")
    monkeypatch.setattr('tango_posture.desktop_qt.calibration.guided.QtPracticeWorker', forbidden)
    dialog = GuidedCalibration(ServiceBridge(Config(data_dir=str(tmp_path))))
    assert dialog._thread is None
    assert not dialog.start.isEnabled()
    assert dialog.preview.isEnabled()
    dialog.reject()


def test_explicit_dancer_restart_is_confirmed_and_persisted(qapp, tmp_path, cached_reference):
    from zipfile import ZipFile

    editor = CalibrationEditor(cached_reference, Config(data_dir=str(tmp_path)))
    editor._approve()
    editor.scrubber.setValue(20)
    editor._restart_track()
    assert not editor.approved
    before = cached_reference.read_bytes()
    editor._save_selection()
    assert cached_reference.read_bytes() == before
    editor.confirm_identity.setChecked(True)
    editor._save_selection()
    with ZipFile(cached_reference) as archive:
        descriptor = json.loads(archive.read("reference.json"))
    assert descriptor["dancer_anchors"] == [{"frame": 0, "dancer": 0}, {"frame": 20, "dancer": 0}]
    editor.reject()
    reopened = CalibrationEditor(cached_reference, Config(data_dir=str(tmp_path)))
    assert reopened._anchors == {0: 0, 20: 0}
    reopened.reject()


def test_promoted_excerpt_has_matching_video_and_cached_frame_indices(qapp, tmp_path, cached_reference):
    source = tmp_path / "practice.tp"
    before = source.read_bytes()
    destination = tmp_path / "references" / "Excerpt.tpr"
    promote_reference(source, destination, "Walking excerpt", start_frame=5, end_frame=15)
    analytics = read_analytics(destination, 0)
    assert len(analytics["frames"]) == 10
    assert analytics["frame_timestamps"] == pytest.approx([index / 10 for index in range(5, 15)])
    editor = CalibrationEditor(destination, Config(data_dir=str(tmp_path)))
    assert editor._reader.count == len(editor.evidence.frames) == 10
    editor.reject()
    assert source.read_bytes() == before


@pytest.mark.parametrize("responsive", [False, True])
def test_guided_holds_require_a_repeatable_change_before_continuing(qapp, tmp_path, responsive):
    from dataclasses import replace

    dialog = GuidedCalibration(ServiceBridge(Config(data_dir=str(tmp_path))))
    for index in range(240):
        observation = frame(index / 10)
        if responsive and (8 <= observation.timestamp < 13 or 18 <= observation.timestamp < 23):
            point = observation.poses[0]["nose"]
            observation.poses[0]["nose"] = replace(point, x=point.x + 0.06)
        dialog._frames.append(observation)
    dialog._latest = 0
    dialog._begin()
    for timestamp in (3, 8, 13, 18, 23):
        dialog._latest = timestamp
        dialog._poll()
    assert len(dialog._holds["Head"]) == 4
    assert dialog.next.isEnabled() is responsive
    assert dialog.retry.isEnabled() and dialog.skip.isEnabled()
    dialog.context.setCurrentText("Solo walking")
    dialog._step = 4
    dialog._next()
    assert dialog._walking
    dialog._latest = 33
    dialog._poll()
    assert dialog._review_requested
    assert dialog._holds["Walking"][0].start == 25
    dialog.reject()


def test_editing_existing_tpc_profile_preserves_and_adds_metrics(qapp, tmp_path):
    from tango_posture.desktop_qt.screens.library import LibraryScreen, ProfileDetailsDialog

    ref_dir = tmp_path / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    video, report, practice = (tmp_path / name for name in ("v.mp4", "r.md", "p.tp"))
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (640, 480))
    assert writer.isOpened()
    entries = []
    for index in range(25):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        pose = dict(frame(index / 10).poses[0])
        pose["left_knee"] = Landmark(0.52 + 0.005 * (index % 3), 385 / 480, 0.99, 0.99)
        pose["right_knee"] = Landmark(0.53 + 0.005 * (index % 3), 385 / 480, 0.99, 0.99)
        entries.append({"poses": [
            {name: [p.x, p.y, p.z, p.visibility, p.presence] for name, p in pose.items()}
        ]})
    writer.release()
    report.write_text("# Source\n")
    append_segment(practice, report, video, None, [i / 10 for i in range(25)], analytics={
        "model_hash": MANIFEST["sha256"],
        "capture_profile": profile(Config(), 640, 480, "video"),
        "frames": entries, "max_gap": 0.5,
    })
    reference = ref_dir / "Teacher.tpr"
    promote_reference(practice, reference, "Teacher")

    cfg = Config(data_dir=str(tmp_path))
    creator = CalibrationEditor(reference, cfg)
    creator.name_input.setText("Editable Profile")
    creator._approve()  # head_pitch
    creator._save()
    saved_path = creator.saved_path
    assert saved_path is not None
    initial = read_range_profile(saved_path)
    assert set(initial.metrics) == {"head_pitch"}

    # Reopen the .tpc file directly and add knee_flexion while keeping head_pitch.
    editor = CalibrationEditor(saved_path, cfg)
    try:
        assert set(editor.approved) == {"head_pitch"}
        assert "Edit approved" in editor.windowTitle()
        editor.step_combo.setCurrentText("Knees")
        assert editor.metric_combo.currentData() == "knee_flexion"
        editor._approve()
        editor._save()
    finally:
        if editor.isVisible():
            editor.reject()

    updated = read_range_profile(saved_path)
    assert updated.id == initial.id
    assert set(updated.metrics) == {"head_pitch", "knee_flexion"}

    # Library row has "Edit calibration" and ProfileDetailsDialog has "Edit calibration…".
    screen = LibraryScreen(ServiceBridge(cfg), Router())
    try:
        row = next(i for i in range(screen.table.rowCount()) if screen.table.item(i, 4).text() == ".tpc")
        buttons = screen.table.cellWidget(row, 3).findChildren(QPushButton)
        assert any(btn.text() == "Edit calibration" for btn in buttons)
        called = []
        dlg = ProfileDetailsDialog(
            updated,
            saved_path,
            screen,
            on_edit_calibration=lambda: called.append(True),
        )
        dlg.edit_calibration_btn.click()
        assert called == [True]
    finally:
        screen.close()


def test_settings_screen_has_expandable_guides_and_worked_examples(qapp, tmp_path):
    from PySide6.QtWidgets import QLabel

    from tango_posture.config import SETTING_MAP

    cfg = Config(data_dir=str(tmp_path))
    screen = SettingsScreen(ServiceBridge(cfg), Router())
    try:
        expected_keys = set(SETTING_MAP) | set(METRIC_DETAILS)
        assert expected_keys <= set(screen.help_toggles)
        assert expected_keys <= set(screen.help_panels)

        # Collapsed by default.
        assert all(not btn.isChecked() for btn in screen.help_toggles.values())
        assert all(panel.isHidden() for panel in screen.help_panels.values())

        # Expand single parameter guide and verify worked example is shown.
        screen.help_toggles["knee_visibility_floor"].click()
        qapp.processEvents()
        assert screen.help_toggles["knee_visibility_floor"].isChecked()
        assert not screen.help_panels["knee_visibility_floor"].isHidden()
        panel_text = "\n".join(
            lbl.text() for lbl in screen.help_panels["knee_visibility_floor"].findChildren(QLabel)
        )
        assert "Example:" in panel_text

        # Collapse it back.
        screen.help_toggles["knee_visibility_floor"].click()
        assert screen.help_panels["knee_visibility_floor"].isHidden()

        # Expand All / Collapse All works for every parameter and Calibration default.
        screen.toggle_all_guides_btn.click()
        assert all(btn.isChecked() for btn in screen.help_toggles.values())
        assert all(not panel.isHidden() for panel in screen.help_panels.values())
        screen.toggle_all_guides_btn.click()
        assert all(not btn.isChecked() for btn in screen.help_toggles.values())
        assert all(panel.isHidden() for panel in screen.help_panels.values())
    finally:
        screen.close()


def test_editing_tpc_commits_draft_metrics_hints_approve_all_and_auto_links(qapp, tmp_path):
    from tango_posture.desktop_qt.screens.references import ReferencesScreen

    ref_dir = tmp_path / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    video, report, practice = (tmp_path / name for name in ("v.mp4", "r.md", "p.tp"))
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (640, 480))
    assert writer.isOpened()
    entries = []
    for index in range(30):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        if index == 12:
            entries.append({"poses": []})
            continue
        pose = dict(frame(index / 10).poses[0])
        pose["left_knee"] = Landmark(0.52 + 0.005 * (index % 3), 385 / 480, 0.99, 0.99)
        pose["right_knee"] = Landmark(0.53 + 0.005 * (index % 3), 385 / 480, 0.99, 0.99)
        entries.append({"poses": [
            {name: [p.x, p.y, p.z, p.visibility, p.presence] for name, p in pose.items()}
        ]})
    writer.release()
    report.write_text("# Source\n")
    append_segment(practice, report, video, None, [i / 10 for i in range(30)], analytics={
        "model_hash": MANIFEST["sha256"],
        "capture_profile": profile(Config(), 640, 480, "video"),
        "frames": entries, "max_gap": 0.5,
    })
    reference = ref_dir / "SoloDrop.tpr"
    promote_reference(practice, reference, "Solo Dropout")

    cfg = Config(data_dir=str(tmp_path))
    creator = CalibrationEditor(reference, cfg)
    try:
        creator.name_input.setText("Draft Auto Test")
        creator.view_combo.setCurrentText("all")
        creator._approve()  # head_pitch
        creator._save()
        tpc_path = creator.saved_path
        assert tpc_path is not None
    finally:
        if creator.isVisible():
            creator.reject()

    editor = CalibrationEditor(tpc_path, cfg)
    try:
        assert editor.view_combo.currentText() == "all"
        assert editor.auto_link_button.isEnabled()
        editor.auto_link_button.click()
        assert editor._anchors == {0: 0, 13: 0}
        assert set(editor.approved) == {"head_pitch"}

        # Restarting track while editing an existing .tpc preserves approved metrics.
        editor.scrubber.setValue(13)
        editor._restart_track()
        assert "head_pitch" in editor.approved

        # Interactively choosing head_forward and head_inclination without clicking Approve
        # shows explanatory hints and commits both drafts on Save.
        hf_idx = editor.metric_combo.findData("head_forward")
        editor.metric_combo.setCurrentIndex(hf_idx)
        editor.metric_combo.activated.emit(hf_idx)
        assert "ear–shoulder" in editor.metric_hint.text()
        editor.lower_spin.setValue(editor.lower_spin.value() - 0.01)

        hi_idx = editor.metric_combo.findData("head_inclination")
        editor.metric_combo.setCurrentIndex(hi_idx)
        editor.metric_combo.activated.emit(hi_idx)
        assert "degrees" in editor.metric_hint.text()
        editor._save()
    finally:
        if editor.isVisible():
            editor.reject()

    saved = read_range_profile(tpc_path)
    assert {"head_pitch", "head_forward", "head_inclination"} <= set(saved.metrics)

    # Approve all remaining metrics via one click and verify ReferencesScreen lists them.
    editor2 = CalibrationEditor(tpc_path, cfg)
    try:
        editor2.approve_all_button.click()
        assert len(editor2.approved) == len(METRIC_DETAILS)
        assert "✓" in editor2.metric_combo.currentText()
        editor2._save()
    finally:
        if editor2.isVisible():
            editor2.reject()

    refs_screen = ReferencesScreen(ServiceBridge(cfg), Router())
    try:
        assert refs_screen.table.rowCount() >= 1
        detail_text = refs_screen.detail_meta.text()
        assert "Approved metrics" in detail_text
        assert "Head-forward offset" in detail_text
        assert "Head inclination" in detail_text
    finally:
        refs_screen.close()


def test_calibration_editor_trims_trailing_static_outro_and_zooms_chart(qapp, tmp_path):
    video, report, practice = (
        tmp_path / name for name in ("outro.mp4", "outro.md", "outro.tp")
    )
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (640, 480))
    assert writer.isOpened()

    entries = []
    # Moving frames 0..24 (0.0s..2.4s), with a single-frame extreme knee bend outlier on frame 12.
    for index in range(25):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        shift = 0.004 * index
        pose = dict(frame(index / 10).poses[0])
        if index == 12:
            pose["left_knee"] = Landmark(0.75, 0.55, 0.99, 0.99)
        else:
            pose["left_knee"] = Landmark(0.52 + 0.005 * (index % 3), 385 / 480, 0.99, 0.99)
        pose["right_knee"] = Landmark(0.53 + 0.005 * (index % 3), 385 / 480, 0.99, 0.99)
        entries.append({
            "poses": [
                {
                    name: [p.x + shift, p.y, p.z, p.visibility, p.presence]
                    for name, p in pose.items()
                }
            ]
        })

    # Empty dropout gap frames 25..27 (2.5s..2.7s).
    for _ in range(25, 28):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        entries.append({"poses": []})

    # Frozen static poster outro frames 28..39 (2.8s..3.9s).
    static_pose = {
        name: [p.x + 0.25, p.y, p.z, p.visibility, p.presence]
        for name, p in frame(2.8).poses[0].items()
    }
    for _ in range(28, 40):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        entries.append({"poses": [dict(static_pose)]})

    writer.release()
    report.write_text("# Source\n")
    append_segment(
        practice,
        report,
        video,
        None,
        [i / 10 for i in range(40)],
        analytics={
            "model_hash": MANIFEST["sha256"],
            "capture_profile": profile(Config(), 640, 480, "video"),
            "frames": entries,
            "max_gap": 0.5,
        },
    )
    ref_dir = tmp_path / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    reference = ref_dir / "OutroRef.tpr"
    promote_reference(practice, reference, "OutroRef")

    cfg = Config(data_dir=str(tmp_path))
    editor = CalibrationEditor(reference, cfg)
    try:
        editor.resize(1100, 760)
        editor.show()
        qapp.processEvents()

        assert single_dancer_anchors(editor.evidence, cfg.confidence) == {0: 0}
        assert editor.chart.time_end == pytest.approx(2.4)
        assert "Track stopped" not in editor.status_label.text()

        # Switching to Knees exercises outlier-resistant Y-axis & distribution rendering.
        editor.step_combo.setCurrentText("Knees")
        assert editor.metric_combo.currentData() == "knee_flexion"
        qapp.processEvents()
        assert not editor.chart.grab().isNull()

        # Zoom to 2x: view window halves and chart click seeks inside the zoomed window.
        editor.chart_zoom_combo.setCurrentIndex(1)
        qapp.processEvents()
        assert editor.chart.x_zoom == pytest.approx(2.0)
        start, end = editor.chart._view_window()
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(1.2)

        center = QPoint(editor.chart.width() // 2, editor.chart.height() // 2)
        QTest.mouseClick(editor.chart, Qt.MouseButton.LeftButton, pos=center)
        assert 0.3 <= editor.scrubber.value() / 10.0 <= 0.9

        # Scrubbing into the second half auto-pans the zoomed window and seeks accurately.
        editor.scrubber.setValue(20)
        qapp.processEvents()
        assert editor.chart.playhead_timestamp == pytest.approx(2.0)
        start2, end2 = editor.chart._view_window()
        assert end2 - start2 == pytest.approx(1.2)
        assert end2 == pytest.approx(2.4)
        QTest.mouseClick(editor.chart, Qt.MouseButton.LeftButton, pos=center)
        assert 1.4 <= editor.scrubber.value() / 10.0 <= 2.1

        # Reset zoom back to Fit (1x).
        editor.chart_zoom_combo.setCurrentIndex(0)
        qapp.processEvents()
        assert editor.chart.x_zoom == pytest.approx(1.0)
        assert editor.chart._view_window() == pytest.approx((0.0, 2.4))
    finally:
        if editor.isVisible():
            editor.reject()


def test_edit_calibration_resolves_renamed_reference_and_works_in_review_and_analytics(
    qapp, tmp_path, cached_reference, monkeypatch
):
    from tango_posture.archive import rename_archive
    from tango_posture.desktop_qt.calibration import editor as editor_mod
    from tango_posture.desktop_qt.screens.analytics import AnalyticsScreen
    from tango_posture.desktop_qt.screens.review import ReviewScreen

    ref_dir = tmp_path / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    tpr_copy = ref_dir / cached_reference.name
    tpr_copy.write_bytes(cached_reference.read_bytes())

    cfg = Config(data_dir=str(tmp_path))
    creator = CalibrationEditor(tpr_copy, cfg)
    try:
        creator.name_input.setText("Giro baseline")
        creator._approve()
        creator._save()
        tpc_path = creator.saved_path
        assert tpc_path is not None
    finally:
        if creator.isVisible():
            creator.reject()

    # Renaming the .tpr changes its outer ZIP SHA-256; Edit calibration still opens it directly.
    rename_archive(tpr_copy, "Renamed Teacher Reference")
    reopened = CalibrationEditor(tpc_path, cfg)
    try:
        assert "Edit approved" in reopened.windowTitle()
        assert set(reopened.approved) == {"head_pitch"}
    finally:
        if reopened.isVisible():
            reopened.reject()

    opened_paths: list[Path] = []

    def fake_open_editor(path: Path, _cfg: Config, _parent, **_kwargs) -> Path | None:
        opened_paths.append(path)
        return path

    monkeypatch.setattr(editor_mod, "open_profile_calibration_editor", fake_open_editor)

    review = ReviewScreen(ServiceBridge(cfg), Router())
    analytics = AnalyticsScreen(ServiceBridge(cfg), Router())
    try:
        assert not review.edit_calibration_btn.isEnabled()
        rev_idx = review.reference_combo.findData(str(tpc_path.resolve()))
        if rev_idx < 0:
            rev_idx = review.reference_combo.findData(str(tpc_path))
        assert rev_idx > 0
        review.reference_combo.setCurrentIndex(rev_idx)
        assert review.edit_calibration_btn.isEnabled()
        review.edit_calibration_btn.click()
        assert opened_paths[-1].resolve() == tpc_path.resolve()

        assert not analytics.edit_calibration_btn.isEnabled()
        def_idx = analytics.calibration_combo.findData("__defaults__")
        analytics.calibration_combo.setCurrentIndex(def_idx)
        assert not analytics.edit_calibration_btn.isEnabled()
        ana_idx = analytics.calibration_combo.findData(str(tpc_path.resolve()))
        assert ana_idx > 0
        analytics.calibration_combo.setCurrentIndex(ana_idx)
        assert analytics.edit_calibration_btn.isEnabled()
        analytics.edit_calibration_btn.click()
        assert len(opened_paths) == 2
        assert opened_paths[-1].resolve() == tpc_path.resolve()
    finally:
        review.close()
        analytics.close()


def test_opening_portrait_giro_reference_auto_links_and_switches_to_all_view(qapp, tmp_path):
    width, height = 608, 1080
    video, report, practice = (
        tmp_path / name for name in ("giro.mp4", "giro.md", "giro.tp")
    )
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (width, height))
    assert writer.isOpened()

    coach_frames = {}
    for index in range(20):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
        if index in (10, 11):
            coach_frames[str(index)] = {"timestamp": index / 10.0, "measurement_poses": []}
            continue
        shift = 2.0 * index
        # First 5 frames (0..4) are side-on (10 px shoulder span); remaining valid frames rotate frontal (70 px).
        right_offset = 10.0 if index < 5 else 70.0
        coords = {
            "nose": (300.0 + shift, 300.0),
            "left_ear": (300.0 + shift, 330.0),
            "right_ear": (300.0 + shift + right_offset, 330.0),
            "left_shoulder": (300.0 + shift, 360.0),
            "right_shoulder": (300.0 + shift + right_offset, 360.0),
            "left_hip": (300.0 + shift, 475.0),
            "right_hip": (300.0 + shift + right_offset, 475.0),
            "left_ankle": (300.0 + shift, 700.0),
            "right_ankle": (300.0 + shift + right_offset, 700.0),
        }
        pose = {
            name: [x / width, y / height, 0.0, 0.99, 0.99]
            for name, (x, y) in coords.items()
        }
        coach_frames[str(index)] = {
            "timestamp": index / 10.0,
            "measurement_poses": [pose],
        }
    writer.release()

    report.write_text("# Giro Source\n")
    append_segment(
        practice,
        report,
        video,
        None,
        [i / 10.0 for i in range(20)],
        analytics={
            "model_hash": MANIFEST["sha256"],
            "capture_profile": profile(Config(), width, height, "video"),
            "coach_data": {
                "width": width,
                "height": height,
                "model_hash": MANIFEST["sha256"],
                "frames": coach_frames,
            },
            "max_gap": 0.5,
        },
    )
    ref_dir = tmp_path / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    reference = ref_dir / "PortraitGiro.tpr"
    promote_reference(practice, reference, "Portrait Giro")

    cfg = Config(data_dir=str(tmp_path))
    editor = CalibrationEditor(reference, cfg)
    try:
        editor.show()
        qapp.processEvents()
        # Opened immediately at the initial metric without touching the step/metric combo box:
        # 1) single dancer auto-linked across the 1.0s-1.1s dropout gap
        assert editor._anchors == {0: 0, 12: 0}
        assert editor.confirm_identity.isChecked()
        # 2) switched to 'all' camera view because side view covers < 50% of valid samples
        assert editor.view_combo.currentText() == "all"
        # 3) full portrait timeline plotted from 0.0s to 1.9s with no dropout warning
        assert editor.chart.time_end == pytest.approx(1.9)
        assert "Track stopped" not in editor.status_label.text()
        valid_points = [val for _, val in editor.chart.data.points if val is not None]
        assert len(valid_points) == 18
    finally:
        if editor.isVisible():
            editor.reject()



