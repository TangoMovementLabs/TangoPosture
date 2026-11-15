import json
import tomllib

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
