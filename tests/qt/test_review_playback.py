"""Offscreen replay tests using generated local media, never a camera."""

import json
import threading
import time
from types import SimpleNamespace
from zipfile import ZipFile

import cv2
import numpy as np
import pytest
from conftest import frame
from PySide6.QtCore import QCoreApplication, QEvent, QUrl
from PySide6.QtWidgets import QApplication, QStackedWidget, QWidget

from tango_posture.archive import (
    append_segment,
    file_sha256,
    read_analytics,
    read_manifest,
    store_analytics,
)
from tango_posture.calibration.archives import save_range_profile
from tango_posture.calibration.baseline import profile
from tango_posture.calibration.models import ApprovedRange, Passage, RangeProfile
from tango_posture.config import Config
from tango_posture.desktop_qt.library.archives import archive_report
from tango_posture.desktop_qt.review.analysis import ReviewAnalysis, analyse_review_clip
from tango_posture.desktop_qt.review.playback import ReviewPlayback
from tango_posture.desktop_qt.router import Router
from tango_posture.desktop_qt.screens.review import ReviewScreen
from tango_posture.desktop_qt.service_bridge import ServiceBridge
from tango_posture.models import MANIFEST, model_manifest
from tango_posture.movement import render_floor_guide
from tango_posture.references.archives import promote_reference
from tango_posture.references.coach import _serialise_pose, render_coach_frame_overlay
from tango_posture.review import write_metadata


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(["-platform", "offscreen"])


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "sequence.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(12):
            writer.write(np.full((48, 64, 3), index * 15, np.uint8))
    finally:
        writer.release()
    return path


@pytest.fixture
def screen(qapp):
    widget = ReviewScreen(ServiceBridge(Config()), Router())
    yield widget
    widget.close()


def make_archive(tmp_path, video):
    path = tmp_path / "practice.tp"
    report = tmp_path / "report.md"
    report.write_text("# Stored report\nActual recorded findings.\n", encoding="utf-8")
    append_segment(path, report, video, video, [0.0, 0.1])
    return path, report


def test_empty_review_has_no_fake_playback_or_insights(screen):
    assert screen.duration_sec == 0
    assert screen.time_label.text() == "00:00.0 / 00:00.0"
    assert not screen.play_btn.isEnabled()
    assert not screen.scrubber.isEnabled()
    assert not screen.compare_cb.isEnabled()
    assert not screen.analytics_btn.isEnabled()
    assert not screen.stream_combo.isEnabled()
    assert screen.report_view.toPlainText() == ""
    assert all(value.text() == "Not observed" for value in screen.metric_values.values())
    assert "when a frame is analyzed" in screen.alerts_text.text()
    screen._toggle_playback()
    assert not screen.is_playing
    assert not screen.timer.isActive()


def test_review_calibration_picker_compares_cached_ranges_without_settings_fallback(
    qapp, tmp_path, video, monkeypatch,
):
    cfg = Config(data_dir=str(tmp_path), entry_duration=0.1, smoothing=0.0001)
    target = RangeProfile(
        name="Head only", context="Walking", model_hash=MANIFEST["sha256"],
        capture_profile=profile(cfg, 64, 48, "video"), source={"archive": "reference.tpr"},
        metrics={"head_pitch": ApprovedRange(
            lower=-180, upper=180, unit="degrees", view="all",
            passages=[Passage(start=0, end=1)], median=0, p10=-10, p90=10,
            valid_duration=1, selected_duration=1, samples=10, inside_fraction=1,
        )},
    )
    calibration = tmp_path / "calibrations" / "Head.tpc"
    save_range_profile(calibration, target)
    raw_pose = _serialise_pose(frame().poses[0])
    raw_pose["nose"] = [0.65, 0.05, 0, 0.99, 0.99]
    data = {
        "width": 64, "height": 48, "model_hash": MANIFEST["sha256"],
        "capture_profile": target.capture_profile, "frames": {
            str(index): {
                "timestamp": index / 10, "message": "BEND YOUR KNEES",
                "poses": [raw_pose],
                "measurement_poses": [raw_pose],
            }
            for index in range(3)
        },
    }
    widget = ReviewScreen(ServiceBridge(cfg), Router())
    try:
        widget.load_archive(video)
        widget._coach_data = data
        widget._update_calibration_comparison()
        widget._show_frame(2)
        assert widget.alerts_text.text() == "Select a calibration profile to see posture alerts."
        widget.reference_combo.setCurrentIndex(widget.reference_combo.findData(str(calibration.resolve())))
        assert widget.alerts_text.text() == "No active posture alert on this frame."
        assert widget.metric_values["shoulder_elev_pct"].text() == "Not in profile"
        assert "Head only" in widget.insights_text.text()
        assert "normalized_clearance" not in widget.insights_text.text()
        target.metrics["head_pitch"].lower = -180
        target.metrics["head_pitch"].upper = -170
        second = tmp_path / "calibrations" / "Narrow.tpc"
        save_range_profile(second, target.model_copy(update={"id": "narrow", "name": "Narrow"}))
        widget.refresh()
        widget.reference_combo.setCurrentIndex(widget.reference_combo.findData(str(second.resolve())))
        assert "outside approved range" in widget.alerts_text.text()
        assert "BEND YOUR KNEES" not in widget.alerts_text.text()
        monkeypatch.setattr(
            "tango_posture.desktop_qt.screens.review.analyse_review_clip",
            lambda *args, **kwargs: ReviewAnalysis(data, {"pivots": [], "steps": []}),
        )
        widget._start_posture_analysis()
        deadline = time.monotonic() + 3
        while widget._analysis_thread is not None and time.monotonic() < deadline:
            qapp.processEvents()
            widget._poll_posture_analysis()
            time.sleep(0.01)
        assert widget._analysis_thread is None
        widget._show_frame(2)
        assert "outside approved range" in widget.alerts_text.text()
        assert "BEND YOUR KNEES" not in widget.alerts_text.text()
        defaults = Config.model_validate({
            **cfg.model_dump(exclude={"axis_curve"}),
            "calibration_defaults": {
                "head_pitch": {"lower": -180, "upper": 180, "unit": "degrees", "view": "all"},
            },
        })
        widget.bridge.update_config(defaults)
        assert "outside approved range" in widget.alerts_text.text()
        widget.reference_combo.setCurrentIndex(0)
        widget._show_frame(2)
        assert widget.alerts_text.text() == "Select a calibration profile to see posture alerts."
        assert "without posture coaching" in widget.insights_text.text()
        widget.reference_combo.setCurrentIndex(widget.reference_combo.findData(str(second.resolve())))
        widget.bridge.update_config(cfg)
        widget._coach_data["model_hash"] = "0" * 64
        rendering = []
        monkeypatch.setattr(
            "tango_posture.desktop_qt.screens.review.render_coach_frame_overlay",
            lambda image, *args, **kwargs: rendering.append(kwargs["lines_only"]) or image,
        )
        widget._calibration_changed()
        assert "different measurement models" in widget.alerts_text.text()
        assert widget._display_coach_data() is widget._coach_data
        assert rendering[-1]
        widget.reference_combo.setCurrentIndex(0)
        assert widget.alerts_text.text() == "Select a calibration profile to see posture alerts."
        widget._release()
        assert widget._compared_coach_data is None
    finally:
        widget.close()


def test_review_approved_pink_axis_uses_old_raw_cache_and_abstains_on_missing_hip(
    qapp, tmp_path, video,
):
    cfg = Config(data_dir=str(tmp_path), entry_duration=0.1, smoothing=0.0001)
    target = RangeProfile(
        name="Pink axis", context="Walking", model_hash=MANIFEST["sha256"],
        capture_profile=profile(cfg, 64, 48, "video"), source={"archive": "reference.tpr"},
        metrics={"torso_axis_tilt": ApprovedRange(
            lower=0, upper=5, unit="degrees", view="all",
            passages=[Passage(start=0, end=1)], median=0, p10=0, p90=5,
            valid_duration=1, selected_duration=1, samples=10, inside_fraction=1,
        )},
    )
    calibration = tmp_path / "calibrations" / "Pink.tpc"
    save_range_profile(calibration, target)
    raw_pose = {
        "left_shoulder": [0.6, 0.2, 0, 0.99, 0.99],
        "right_shoulder": [0.7, 0.2, 0, 0.99, 0.99],
        "left_hip": [0.4, 0.8, 0, 0.99, 0.99],
        "right_hip": [0.5, 0.8, 0, 0.99, 0.99],
    }
    missing_hip = {key: value for key, value in raw_pose.items() if key != "right_hip"}
    data = {
        "width": 64, "height": 48, "model_hash": MANIFEST["sha256"],
        "capture_profile": target.capture_profile, "frames": {
            str(index): {
                "timestamp": index / 10, "poses": [pose], "measurement_poses": [pose],
            }
            for index, pose in enumerate([raw_pose, raw_pose, raw_pose, missing_hip])
        },
    }
    widget = ReviewScreen(ServiceBridge(cfg), Router())
    try:
        widget.load_archive(video)
        widget._coach_data = data
        widget.reference_combo.setCurrentIndex(
            widget.reference_combo.findData(str(calibration.resolve())),
        )
        widget._show_frame(2)
        assert widget.metric_values["torso_axis_tilt"].text() == "24.1°"
        assert widget.metric_ranges["torso_axis_tilt"].text() == "Approved 0.0° to 5.0°"
        assert "Torso axis tilt (pink line)" in widget.alerts_text.text()
        assert "outside approved range" in widget.alerts_text.text()
        assert widget.metric_values["head_angle"].text() == "Not in profile"
        widget._show_frame(3)
        assert widget.metric_values["torso_axis_tilt"].text() == "Not measured"
        assert "tracking/view" in widget.alerts_text.text()
        assert "right_hip" not in data["frames"]["3"]["measurement_poses"][0]
    finally:
        widget.close()


def test_review_axis_toggle_uses_raw_landmarks_without_coaching(screen, video, monkeypatch):
    screen.load_archive(video)
    pose = {
        "left_shoulder": [0.55, 0.2, 0, 0.99, 0.99],
        "right_shoulder": [0.75, 0.2, 0, 0.99, 0.99],
        "left_hip": [0.35, 0.8, 0, 0.99, 0.99],
        "right_hip": [0.55, 0.8, 0, 0.99, 0.99],
    }
    missing_hip = {key: point for key, point in pose.items() if key != "right_hip"}
    screen._coach_data = {"width": 64, "height": 48, "frames": {
        "0": {"timestamp": 0.0, "measurement_poses": [pose]},
        "1": {"timestamp": 0.1, "measurement_poses": [missing_hip], "display_poses": [pose]},
        "2": {"timestamp": 0.2, "poses": [pose]},
    }}
    screen._update_calibration_comparison()
    screen.show_coach_cb.setChecked(False)
    screen.show_pivots_cb.setChecked(False)
    screen.show_steps_cb.setChecked(False)
    screen._show_frame(0)
    original = screen.primary_stage._image.copy()
    assert not screen.show_torso_axis_cb.isChecked()

    text = []
    put_text = cv2.putText

    def record_text(image, label, *args, **kwargs):
        text.append(label)
        return put_text(image, label, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)
    screen.show_torso_axis_cb.setChecked(True)
    assert screen.primary_stage._image != original
    assert any(label.startswith("Torso axis:") and "deg" in label for label in text)
    assert screen.alerts_text.text() == "Select a calibration profile to see posture alerts."
    screen.show_torso_axis_cb.setChecked(False)
    assert screen.primary_stage._image == original

    screen.scrubber.setValue(1)
    without_axis = screen.primary_stage._image.copy()
    text.clear()
    screen.show_torso_axis_cb.setChecked(True)
    assert screen.primary_stage._image == without_axis
    assert not text

    screen.show_torso_axis_cb.setChecked(False)
    screen.scrubber.setValue(2)
    without_axis = screen.primary_stage._image.copy()
    screen.show_torso_axis_cb.setChecked(True)
    assert screen.primary_stage._image == without_axis
    assert "Analyze the source video again" in screen.axis_status.text()

    screen.scrubber.setValue(0)
    text.clear()
    screen.lines_only_cb.setChecked(True)
    assert not text


def test_review_interpolation_settings_refresh_display_without_changing_raw_alerts(
    screen, video, monkeypatch,
):
    displayed = []

    def render(image, data, frame_index, cfg, **kwargs):
        assert kwargs["lines_only"]
        displayed.append(data["frames"][str(frame_index)])
        return image

    monkeypatch.setattr("tango_posture.desktop_qt.screens.review.render_coach_frame_overlay", render)
    screen.load_archive(video)
    screen._coach_data = {"width": 64, "height": 48, "frames": {
        str(index): {
            "timestamp": index / 10, "message": "Tracking unavailable",
            "measurement_poses": [
                {"left_hip": [x, 0.5, 0, 1, 1]} if x is not None else {},
            ],
        }
        for index, x in enumerate((0.1, 0.2, None, 1.0))
    }}
    screen.refresh()
    screen._show_frame(2)
    assert displayed[-1]["display_poses"][0]["left_hip"][0] == pytest.approx(0.6)
    assert screen.alerts_text.text() == "Select a calibration profile to see posture alerts."
    screen.bridge.update_config(screen.bridge.config.model_copy(update={
        "replay_joint_interpolation": "quadratic",
    }))
    screen._show_frame(2)
    assert displayed[-1]["display_poses"][0]["left_hip"][0] == pytest.approx(0.5)
    assert screen.alerts_text.text() == "Select a calibration profile to see posture alerts."
    assert screen._coach_data["frames"]["2"]["measurement_poses"] == [{}]
    screen.bridge.update_config(screen.bridge.config.model_copy(update={"replay_joint_max_gap": 0}))
    screen._show_frame(2)
    assert displayed[-1]["measurement_poses"] == [{}]
    assert "display_poses" not in displayed[-1]


@pytest.mark.parametrize("archived", [False, True])
def test_review_posture_analysis_runs_off_ui_thread_and_syncs_overlay(
    screen, video, monkeypatch, qapp, tmp_path, archived
):
    observed = {}
    coach_data = {
        "width": 64,
        "height": 48,
        "frames": {
            "0": {
                "head_angle": -8.0,
                "shoulder_elev_pct": 24.0,
                "axis_curve_smoothed": 0.12,
                "knee_bend_deg": 14.5,
                "message": "BEND YOUR KNEES",
                "poses": [{"nose": [0.5, 0.1, 0.0]}],
                "notes": ["BEND YOUR KNEES", "Knees: STRAIGHT (14.5 deg)"],
            }
        },
    }

    def analyze(path, cfg, *, selected_out, progress_callback, cancel_check):
        observed["thread"] = threading.get_ident()
        observed["path"] = path
        observed["source_bytes"] = path.read_bytes()
        observed["selected_out"] = selected_out
        observed["cancel_check"] = cancel_check
        progress_callback(50, 100, "Analyzing posture")
        if archived:
            coach_data["frames"].update({str(i): {} for i in range(1, 12)})
            result = SimpleNamespace(
                frame_log=[{"timestamp": i / 10} for i in range(12)],
                analysis_mode="movement", selected_out=selected_out, title="Test",
                source_sha256=file_sha256(path),
                readiness=SimpleNamespace(to_dict=dict), coverage={},
                soundtrack=SimpleNamespace(to_metadata_dict=dict),
                manifest=SimpleNamespace(capture_profile={}),
            )
            return ReviewAnalysis(coach_data, {"pivots": [], "steps": []}, result=result)
        return ReviewAnalysis(coach_data, {"pivots": [], "steps": []})

    def render(image, data, frame_index, cfg, **kwargs):
        observed["overlay_frame"] = frame_index
        observed["lines_only"] = kwargs["lines_only"]
        image[0, 0] = (0, 0, 255)
        return image

    def render_movement(image, data, frame_index, fps, **kwargs):
        observed["movement_options"] = (kwargs["show_pivots"], kwargs["show_steps"])
        return image

    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.review.analyse_review_clip", analyze
    )
    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.review.render_coach_frame_overlay", render
    )
    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.review.render_movement_overlay", render_movement
    )
    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.review.render_review_report",
        lambda *args: "# Updated analysis report",
    )
    path = video
    if archived:
        report = tmp_path / "report.md"
        report.write_text("# Archived practice\n")
        path = tmp_path / "practice.tp"
        append_segment(path, report, video, None, [index / 10 for index in range(12)])
    screen.load_archive(path)
    screen._start_posture_analysis()

    deadline = time.monotonic() + 10
    while screen._analysis_thread is not None and time.monotonic() < deadline:
        qapp.processEvents()
        screen._poll_posture_analysis()
        time.sleep(0.01)

    assert screen._analysis_thread is None
    assert observed["thread"] != threading.get_ident()
    assert observed["source_bytes"] == video.read_bytes()
    assert observed["selected_out"] == pytest.approx(1.2)
    assert not observed["cancel_check"]()
    assert observed["overlay_frame"] == 0
    assert screen.metric_values["head_angle"].text() == "-8.0°"
    assert screen.metric_values["shoulder_elev_pct"].text() == "+24.0%"
    assert screen.metric_values["axis_curve_smoothed"].text() == "+12.0%"
    assert screen.metric_values["knee_bend_deg"].text() == "14.5°"
    assert observed["lines_only"]
    assert observed["movement_options"] == (True, True)
    assert screen.alerts_text.text() == "Select a calibration profile to see posture alerts."
    assert "Knees: STRAIGHT" not in screen.insights_text.text()
    assert screen.primary_stage._image.pixelColor(0, 0).red() == 255
    assert screen.splitter.count() == 2
    assert screen.inspector.maximumWidth() == 420
    if archived:
        assert read_analytics(path, 0)["coach_data"] == coach_data
        screen.load_archive(path)
        assert screen.metric_values["head_angle"].text() == "-8.0°"
        assert screen._movement_data["pivots"] == []
        assert screen._movement_data["steps"] == []
    else:
        assert "Latest Review posture" in screen.report_view.toPlainText()
        assert "not saved with the standalone source" in screen.report_view.toPlainText()


@pytest.mark.parametrize("kind", ["tp", "tpr"])
def test_reference_reanalysis_replaces_cached_results_and_survives_reload(
    screen, video, tmp_path, monkeypatch, qapp, kind,
):
    practice = tmp_path / "source.tp"
    report = tmp_path / "teacher.md"
    report.write_text("# Original reference report")
    append_segment(practice, report, video, None, [index / 10 for index in range(12)])
    if kind == "tpr":
        reference = tmp_path / "teacher.tpr"
        promote_reference(practice, reference, "Teacher")
    else:
        reference = practice
        append_segment(practice, report, video, None, [index / 10 for index in range(12)])
    with ZipFile(reference, "a") as archive:
        archive.writestr("overlay_00_lines.mp4", b"old lines")
        archive.writestr("overlay_00.notes.jsonl", '{"old": true}\n')
    with ZipFile(reference) as archive:
        raw_video = archive.read("video_00.mp4")
        other = {
            name: archive.read(name) for name in archive.namelist() if "_01." in name
        }
    store_analytics(reference, 0, {
        "frames": [{"old": "legacy landmarks"}],
        "coach_data": {"frames": {"0": {"head_angle": -1}}},
        "movement_data": {"steps": [], "pivots": [], "tag": "old"},
        "source_provenance": {"origin": "original"},
    })
    from tango_posture.references.pipeline import analyze_clip

    values = iter((1, 2))

    def analyze(path, cfg, *, selected_out, progress_callback, cancel_check):
        value = next(values)

        class Estimator:
            def estimate(self, image, timestamp):
                return frame(timestamp, forward=value * 0.3)

            def close(self):
                pass

        result = analyze_clip(
            video_path=path, cfg=cfg, selected_out=selected_out, analysis_mode="movement",
            analyse_soundtrack=False, estimator_factory=Estimator,
            progress_callback=progress_callback, cancel_check=cancel_check,
        )
        digest = model_manifest("lite" if value == 1 else "heavy")["sha256"]
        result.performance["model_hash"] = digest
        result.coach_data["model_hash"] = digest
        result.coach_data["frames"]["0"]["run_tag"] = value
        if value == 2:
            result.movement_data["steps"] = [{
                "frame": 0, "time": 0.0, "side": "left", "confidence": 0.9,
                "hx": 15, "hy": 35, "tx": 25, "ty": 35, "ax": 20, "ay": 35,
            }]
        result.movement_data["tag"] = value
        return ReviewAnalysis(result.coach_data, result.movement_data, result.manifest, result)

    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.review.analyse_review_clip",
        analyze,
    )
    screen.load_archive(reference)
    previous_overlay = None
    previous_lines = None
    previous_report = None
    previous_pixels = {}
    for value in (1, 2):
        screen.analytics_btn.click()
        deadline = time.monotonic() + 15
        while screen._analysis_thread is not None and time.monotonic() < deadline:
            qapp.processEvents()
            screen._poll_posture_analysis()
            time.sleep(0.01)
        assert screen._analysis_thread is None
        stored = read_analytics(reference, 0)
        assert stored["coach_data"]["frames"]["0"]["run_tag"] == value
        assert stored["movement_data"]["tag"] == value
        if kind == "tpr":
            assert stored["frames"] == []
        assert stored["source_provenance"] == {"origin": "original"}
        variant = "lite" if value == 1 else "heavy"
        manifest = model_manifest(variant)
        report_text = screen.report_view.toPlainText()
        assert f"MediaPipe Pose Landmarker {variant.capitalize()}" in report_text
        assert manifest["version"] in report_text
        with ZipFile(reference) as archive:
            saved_report = archive.read("report_00.md").decode()
            overlay = archive.read("overlay_00.mp4")
            lines = archive.read("lines_00.mp4")
            assert json.loads(archive.read("movement_00.v1.json"))["tag"] == value
            assert archive.read("overlay_00_lines.mp4") == lines
            assert "old" not in archive.read("overlay_00.notes.jsonl").decode()
            assert archive.read("video_00.mp4") == raw_video
            assert all(archive.read(name) == content for name, content in other.items())
            metadata = json.loads(archive.read("report_00.metadata.json"))
            assert len(metadata["frame_timestamps"]) == 12
            assert metadata["overlay_video"] == "overlay_00.mp4"
            assert metadata["lines_video"] == "lines_00.mp4"
        assert overlay and lines
        if previous_overlay is not None:
            assert overlay != previous_overlay
            assert lines != previous_lines
            assert saved_report != previous_report
        previous_overlay = overlay
        previous_lines = lines
        previous_report = saved_report
        for name, content in (("overlay", overlay), ("lines", lines)):
            output = tmp_path / f"{name}-{value}.mp4"
            output.write_bytes(content)
            capture = cv2.VideoCapture(str(output))
            try:
                decoded = 0
                first_frame = None
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    if first_frame is None:
                        first_frame = image
                    decoded += 1
                assert decoded == 12
                assert first_frame is not None
                if name in previous_pixels:
                    assert not np.array_equal(first_frame, previous_pixels[name])
                previous_pixels[name] = first_frame
            finally:
                capture.release()
        assert f"| Pose Model Version | `{manifest['version']}` |" in saved_report
        assert f"| Pose Model SHA-256 | `{manifest['sha256']}` |" in saved_report
        assert "# Original reference report" not in saved_report
        assert f"`0` pivots, `{value - 1}` steps" in saved_report
        screen._on_segment_changed(0)
        assert screen._coach_data["frames"]["0"]["run_tag"] == value
        assert manifest["version"] in screen.report_view.toPlainText()
    screen.load_archive(reference)
    assert screen._coach_data["frames"]["0"]["run_tag"] == 2
    assert screen._movement_data["tag"] == 2
    assert "Original reference report" not in screen.report_view.toPlainText()
    assert model_manifest("heavy")["version"] in screen.report_view.toPlainText()


@pytest.mark.parametrize("outcome", ["failed", "canceled"])
def test_overlay_regeneration_failure_preserves_entire_archive(
    screen, video, tmp_path, monkeypatch, qapp, outcome,
):
    archive = tmp_path / "practice.tp"
    report = tmp_path / "original.md"
    report.write_text("# Original")
    append_segment(archive, report, video, None, [i / 10 for i in range(12)])
    before = archive.read_bytes()
    result = SimpleNamespace(
        frame_log=[{"timestamp": i / 10} for i in range(12)], analysis_mode="movement",
        selected_out=1.2, title="Test", readiness=SimpleNamespace(to_dict=dict),
        source_sha256=file_sha256(video),
        coverage={}, soundtrack=SimpleNamespace(to_metadata_dict=dict),
        manifest=SimpleNamespace(capture_profile={}),
    )
    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.review.analyse_review_clip",
        lambda *args, **kwargs: ReviewAnalysis(
            {"frames": {}}, {"pivots": [], "steps": []}, result=result,
        ),
    )
    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.review.render_review_report", lambda *args: "# New",
    )
    started = threading.Event()

    def fail(_video, _folder, _analysis, _cfg, cancel_check):
        started.set()
        if outcome == "canceled":
            while not cancel_check():
                time.sleep(0.005)
        raise ValueError("overlay encoder failed")

    monkeypatch.setattr("tango_posture.desktop_qt.screens.review.render_review_overlays", fail)
    screen.load_archive(archive)
    screen.analytics_btn.click()
    assert started.wait(timeout=3)
    if outcome == "canceled":
        screen._cancel_posture_analysis()
    deadline = time.monotonic() + 5
    while screen._analysis_thread is not None and time.monotonic() < deadline:
        qapp.processEvents()
        screen._poll_posture_analysis()
        time.sleep(0.01)
    assert screen._analysis_thread is None
    assert ("canceled" if outcome == "canceled" else "overlay encoder failed") in (
        screen.analysis_status.text()
    )
    assert archive.read_bytes() == before


def test_long_recording_is_not_partially_reanalyzed(screen, video, monkeypatch):
    screen.load_archive(video)
    monkeypatch.setattr(type(screen._playback.reader), "duration", property(lambda self: 121))
    screen._start_posture_analysis()
    assert screen._analysis_thread is None
    assert "limited to 120 seconds" in screen.analysis_status.text()


@pytest.mark.parametrize("outcome", ["failed", "canceled"])
def test_unsuccessful_reference_reanalysis_preserves_previous_results(
    screen, video, tmp_path, monkeypatch, qapp, outcome,
):
    reference = tmp_path / "teacher.tpr"
    practice = tmp_path / "source.tp"
    report = tmp_path / "teacher.md"
    report.write_text("# Reference")
    append_segment(practice, report, video, None, [index / 10 for index in range(12)])
    promote_reference(practice, reference, "Teacher")
    store_analytics(reference, 0, {
        "coach_data": {"frames": {"0": {"head_angle": 7}}},
        "movement_data": {"steps": [], "pivots": []},
    })
    previous = read_analytics(reference, 0)
    with ZipFile(reference) as archive:
        previous_report = archive.read("report_00.md")
    started = threading.Event()

    def fail(*args, cancel_check, **kwargs):
        started.set()
        if outcome == "canceled":
            while not cancel_check():
                time.sleep(0.005)
            raise RuntimeError("canceled")
        raise ValueError("model unavailable")

    monkeypatch.setattr("tango_posture.desktop_qt.screens.review.analyse_review_clip", fail)
    screen.load_archive(reference)
    screen.analytics_btn.click()
    if outcome == "canceled":
        assert started.wait(timeout=3)
        screen._cancel_posture_analysis()
    deadline = time.monotonic() + 3
    while screen._analysis_thread is not None and time.monotonic() < deadline:
        qapp.processEvents()
        screen._poll_posture_analysis()
        time.sleep(0.01)
    assert screen._analysis_thread is None
    assert ("model unavailable" if outcome == "failed" else "canceled") in screen.analysis_status.text()
    assert read_analytics(reference, 0) == previous
    assert screen._coach_data["frames"]["0"]["head_angle"] == 7
    with ZipFile(reference) as archive:
        assert archive.read("report_00.md") == previous_report


def test_review_analysis_uses_in_memory_clip_analyzer_without_temp_db_or_report(
    video, tmp_path, monkeypatch
):
    observed = {}
    coach_data = {"frames": {}}
    manifest = SimpleNamespace(model_variant="lite", model_hash="abc")

    def fake_analyze_clip(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            coach_data=coach_data,
            movement_data={"steps": [{"frame": 1}]},
            manifest=manifest,
        )

    monkeypatch.setattr(
        "tango_posture.desktop_qt.review.analysis.analyze_clip",
        fake_analyze_clip,
    )
    data_dir = tmp_path / "data"
    result = analyse_review_clip(
        video,
        Config(data_dir=str(data_dir)),
        selected_out=1.0,
        progress_callback=lambda *_: None,
        cancel_check=lambda: False,
    )

    assert result.coach_data is coach_data
    assert result.movement_data == {"steps": [{"frame": 1}]}
    assert result.manifest is manifest
    assert observed["video_path"] == video
    assert observed["selected_in"] == 0.0
    assert observed["selected_out"] == 1.0
    assert observed["analysis_mode"] == "movement"
    assert observed["analyse_soundtrack"] is False
    assert "report_dir" not in observed
    assert "storage" not in observed
    assert not data_dir.exists()


def test_review_navigation_preserves_leased_video_and_blocks_cancelled_commit(
    screen, video, tmp_path, monkeypatch, qapp
):
    archive_path, _ = make_archive(tmp_path, video)
    started = threading.Event()
    proceed = threading.Event()
    worker_saw_readable_file = []

    def analyze(path, cfg, *, selected_out, progress_callback, cancel_check):
        started.set()
        proceed.wait(timeout=3.0)
        worker_saw_readable_file.append(path.is_file() and path.read_bytes() == video.read_bytes())
        return ReviewAnalysis({"frames": {"0": {"head_angle": 1.0}}}, {"pivots": [], "steps": []})

    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.review.analyse_review_clip", analyze
    )
    screen.load_archive(archive_path)
    original_extract_folder = screen._playback.archive.folder
    screen._start_posture_analysis()
    thread = screen._analysis_thread
    assert thread is not None

    deadline = time.monotonic() + 3
    while not started.is_set() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert started.is_set()

    # Simulate navigation away / release closing the playback archive while worker is still running
    screen._release()
    assert original_extract_folder.exists()
    proceed.set()
    thread.join(timeout=3.0)

    assert not original_extract_folder.exists()
    assert worker_saw_readable_file == [True]
    assert read_analytics(archive_path, 0) is None


def test_review_draws_steps_and_pivots_and_can_toggle_them(screen, video):
    screen.load_archive(video)
    screen.show_coach_cb.setChecked(False)
    screen._movement_data = {
        "width": 64, "height": 48,
        "steps": [{"frame": 0, "time": 0.0, "hx": 18, "hy": 42,
                   "tx": 22, "ty": 35, "side": "left"}],
        "pivots": [{"start_frame": 0, "end_frame": 5, "direction": "clockwise"}],
        "hip_frames": [{"frame": 0, "lx": 20, "ly": 20, "rx": 44, "ry": 20, "scale": 30}],
    }
    screen._show_frame(0)
    with_movement = screen.primary_stage._image.copy()
    screen.show_steps_cb.setChecked(False)
    screen.show_pivots_cb.setChecked(False)
    without_movement = screen.primary_stage._image.copy()
    assert with_movement != without_movement
    screen.show_steps_cb.setChecked(True)
    assert screen.primary_stage._image != without_movement
    screen.show_steps_cb.setChecked(False)
    screen.show_pivots_cb.setChecked(True)
    assert screen.primary_stage._image != without_movement


def test_lines_only_uses_floor_and_skeleton_without_text(screen, video, monkeypatch):
    screen.load_archive(video)
    screen.scrubber.setValue(8)
    frame = np.full((48, 64, 3), 120, np.uint8)
    floor = render_floor_guide(frame)
    screen._coach_data = {
        "width": 64, "height": 48,
        "frames": {"8": {"poses": [{
            "left_shoulder": [0.4, 0.25, 0, 1, 1],
            "left_hip": [0.4, 0.6, 0, 1, 1],
        }], "message": "LOWER YOUR SHOULDERS"}},
    }

    screen._update_calibration_comparison()

    def no_text(*args, **kwargs):
        pytest.fail("Lines-only replay must not draw text")

    monkeypatch.setattr(cv2, "putText", no_text)
    screen.lines_only_cb.setChecked(True)
    expected = render_coach_frame_overlay(
        floor, screen._coach_data, 8, Config(), lines_only=True
    )
    image = screen.primary_stage._image
    assert image.width() == 64
    assert image.pixelColor(5, 5).red() == int(expected[5, 5, 2])
    assert image.pixelColor(26, 20).green() > image.pixelColor(26, 20).red()
    assert not np.array_equal(expected, floor)


def test_archive_comparison_uses_raw_video_without_stream_selector(screen, video, tmp_path):
    path, _ = make_archive(tmp_path, video)
    screen.load_archive(path)
    screen._load_comparison(path)
    assert not hasattr(screen, "comparison_stream_combo")
    assert screen._comparison.reader.path == screen._comparison.variants["Raw video"].resolve()
    screen.lines_only_cb.setChecked(True)
    assert screen.stream_combo.currentText() == "Raw video"


@pytest.mark.parametrize("as_reference", [False, True])
def test_library_import_packages_matching_clips_and_rebased_analytics(
    video, tmp_path, monkeypatch, screen, as_reference,
):
    cfg = Config(data_dir=str(tmp_path))
    report = tmp_path / "reports" / "report_import.md"
    report.parent.mkdir()
    report.write_text("# Imported steps\n")
    write_metadata(report, f"import:{video.name}", video, None, [], extra={
        "clip_in": 0.2, "clip_out": 0.8, "is_imported_reference": True,
        "source_sha256": file_sha256(video),
    })
    coach = {"width": 64, "height": 48, "frames": {
        "2": {"head_angle": 5, "poses": []}, "7": {"head_angle": 7, "poses": []},
    }}
    movement = {
        "steps": [{"frame": 4, "time": 0.4, "hx": 18, "hy": 42,
                   "tx": 22, "ty": 35, "side": "left"}],
        "pivots": [{"start_frame": 1, "end_frame": 6, "direction": "clockwise"}],
        "hip_frames": [], "cut_frames": [3], "width": 64, "height": 48,
    }
    monkeypatch.setattr(
        'tango_posture.desktop_qt.library.archives.load_report_coach_data', lambda _: coach,
    )
    monkeypatch.setattr(
        'tango_posture.desktop_qt.library.archives.load_report_movement_data', lambda _: movement,
    )
    source_bytes = video.read_bytes()
    path = archive_report(report, cfg, lambda: False, as_reference=as_reference)
    assert path.suffix == (".tpr" if as_reference else ".tp")
    analytics = read_analytics(path, 0)
    assert set(analytics["coach_data"]["frames"]) == {"0", "5"}
    assert analytics["movement_data"]["steps"][0]["frame"] == 2
    assert analytics["movement_data"]["steps"][0]["time"] == pytest.approx(0.2)
    assert analytics["movement_data"]["pivots"][0]["start_frame"] == 0
    assert analytics["movement_data"]["cut_frames"] == [1]
    assert analytics["frame_timestamps"] == pytest.approx([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert video.read_bytes() == source_bytes
    assert report.is_file()
    screen.load_archive(path)
    assert screen.metric_values["head_angle"].text() == "+5.0°"
    for label in screen._playback.variants:
        assert screen._playback.select_variant(label).count == 6
        assert len(screen._playback.timestamps) == 6
    assert read_manifest(path)["segments"][0]["lines"] == "lines_00.mp4"
def test_review_posture_analysis_can_be_cancelled(screen, video, monkeypatch, qapp):
    started = threading.Event()

    def analyze(path, cfg, *, selected_out, progress_callback, cancel_check):
        started.set()
        while not cancel_check():
            time.sleep(0.005)
        raise RuntimeError("analysis canceled")

    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.review.analyse_review_clip", analyze
    )
    screen.load_archive(video)
    screen._start_posture_analysis()

    deadline = time.monotonic() + 3
    while not started.is_set() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert started.is_set()
    assert not screen.cancel_analysis_btn.isHidden()
    screen._cancel_posture_analysis()

    deadline = time.monotonic() + 3
    while screen._analysis_thread is not None and time.monotonic() < deadline:
        qapp.processEvents()
        screen._poll_posture_analysis()
        time.sleep(0.01)

    assert screen._analysis_thread is None
    assert screen.analysis_status.text() == "Posture analysis canceled."
    assert all(value.text() == "Not observed" for value in screen.metric_values.values())
    assert screen.cancel_analysis_btn.isHidden()


def test_local_video_decodes_seeks_and_plays_to_real_end(screen, video):
    now = [100.0]
    screen._clock = lambda: now[0]
    screen.load_archive(video)
    reader = screen._playback.reader
    assert reader is not None
    assert screen.duration_sec == pytest.approx(1.2)
    assert screen.scrubber.maximum() == 11
    assert screen.primary_stage._image is not None
    assert screen.primary_stage._image.pixelColor(20, 20).red() == pytest.approx(0, abs=3)
    first_image = screen.primary_stage._image
    screen.scrubber.setValue(8)
    assert reader.index == 8
    assert screen.current_time_sec == pytest.approx(0.8)
    assert screen.primary_stage._image.pixelColor(20, 20).red() == pytest.approx(120, abs=3)
    assert first_image.pixelColor(20, 20).red() == pytest.approx(0, abs=3)
    screen.scrubber.setValue(2)
    assert reader.index == 2
    assert screen.time_label.text() == "00:00.2 / 00:01.2"

    screen._toggle_playback()
    assert screen.timer.isActive()
    now[0] += 0.35
    screen._on_tick()
    assert reader.index == 5
    screen._toggle_playback()
    assert not screen.is_playing
    assert not screen.timer.isActive()
    now[0] += 20
    screen._on_tick()
    assert reader.index == 5

    screen._toggle_playback()
    screen.scrubber.setValue(1)  # Seeking while playing rebases the clock.
    now[0] += 0.35
    screen._on_tick()
    assert reader.index == 4
    now[0] += 2
    screen._on_tick()
    assert not screen.is_playing
    assert not screen.timer.isActive()
    assert reader.index == 11
    assert screen.current_time_sec == pytest.approx(1.2)
    assert screen.time_label.text() == "00:01.2 / 00:01.2"
    screen._toggle_playback()
    assert reader.index == 0
    assert screen.is_playing


def test_replay_speed_and_frame_step_controls(screen, video):
    now = [50.0]
    screen._clock = lambda: now[0]
    screen.load_archive(video)
    screen.speed_combo.setCurrentIndex(screen.speed_combo.findData(2.0))
    screen._toggle_playback()
    now[0] += 0.35
    screen._on_tick()
    reader = screen._playback.reader
    assert reader is not None
    assert reader.index == 7

    screen._step_frame(-1)
    assert not screen.is_playing
    assert reader.index == 6
    assert screen.current_time_sec == pytest.approx(0.6)
    screen._step_frame(1)
    assert reader.index == 7


def test_synchronized_side_by_side_comparison(screen, video, monkeypatch):
    screen.load_archive(video)
    screen.scrubber.setValue(8)
    monkeypatch.setattr(
        screen.bridge, "choose_library_archive",
        lambda *args: video,
    )
    screen.compare_btn.click()

    comparison = screen._comparison.reader
    assert comparison is not None
    assert screen.compare_cb.isChecked()
    assert not screen.comparison_stage.isHidden()
    assert comparison.index == 8

    screen.scrubber.setValue(3)
    assert comparison.index == 3
    screen.compare_cb.setChecked(False)
    assert screen.comparison_stage.isHidden()
    screen.compare_cb.setChecked(True)
    assert not screen.comparison_stage.isHidden()
    assert comparison.index == 3


def test_export_copies_selected_video_without_modifying_source(screen, video, tmp_path, monkeypatch):
    destination = tmp_path / "exported.avi"
    screen.load_archive(video)
    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.review.QFileDialog.getSaveFileName",
        lambda *args: (str(destination), "Video files"),
    )
    screen._export_video()
    assert destination.read_bytes() == video.read_bytes()
    assert video.is_file()


def test_standalone_import_report_relinks_by_hash_and_seeks(screen, video, tmp_path, monkeypatch):
    source = tmp_path / "source.avi"
    source.write_bytes(video.read_bytes())
    report = tmp_path / "report_import.md"
    report.write_text(
        "# Imported clip\n\nObservation at `00:00.8 (0.8s)`.\n", encoding="utf-8"
    )
    timestamps = [index / 10 for index in range(12)]
    write_metadata(
        report,
        "import:source.avi",
        source,
        None,
        timestamps,
        extra={
            "is_imported_reference": True,
            "source_sha256": file_sha256(source),
            "clip_in": 0.0,
            "clip_out": 1.2,
        },
    )
    moved = tmp_path / "moved.avi"
    source.rename(moved)
    messages = []
    screen.bridge.status_message.connect(messages.append)

    screen.load_archive(report)
    assert screen._playback.reader is None
    assert screen.relink_btn.isEnabled()
    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.review.QFileDialog.getOpenFileName",
        lambda *args: (str(moved), "Video files"),
    )
    screen._relink_report_video()

    reader = screen._playback.reader
    assert reader is not None
    assert screen._playback._report is not None
    assert screen._playback._report.source_sha256 == file_sha256(moved)
    screen.report_view.anchorClicked.emit(QUrl("seek:0.800000"))
    assert reader.index == 8
    assert any("verified SHA-256" in message for message in messages)


def test_calibration_archive_can_be_used_as_replay_comparison(video, tmp_path):
    path = tmp_path / "reference.tpc"
    report = "# Reference\n\nCalibration capture.\n"
    metadata = {
        "version": 1,
        "report_file": "report_00.md",
        "raw_video": "video_00.mp4",
        "overlay_video": None,
        "source_video": None,
        "frame_timestamps": [index / 10 for index in range(12)],
    }
    manifest = {
        "version": 1,
        "kind": "calibration",
        "name": "Reference",
        "reference": "reference.json",
        "segments": [
            {
                "index": 0,
                "video": "video_00.mp4",
                "overlay": None,
                "report": "report_00.md",
                "metadata": "report_00.metadata.json",
            }
        ],
    }
    with ZipFile(path, "w") as archive, video.open("rb") as stream:
        archive.writestr("video_00.mp4", stream.read())
        archive.writestr("reference.json", "{}")
        archive.writestr("report_00.md", report)
        archive.writestr("report_00.metadata.json", json.dumps(metadata))
        archive.writestr("manifest.json", json.dumps(manifest))

    playback = ReviewPlayback()
    try:
        playback.open(path)
        playback.select_segment(0)
        reader = playback.select_variant("Raw video")
        assert reader.count == 12
    finally:
        playback.close()


def test_archive_frame_timestamps_drive_nonuniform_playback_timeline(screen, video, tmp_path):
    path = tmp_path / "timed-practice.tp"
    report = tmp_path / "report.md"
    report.write_text(
        "# Timestamped recording\n\nEvent at `00:00.8 (0.8s)`.\n", encoding="utf-8"
    )
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    append_segment(path, report, video, None, timestamps)

    now = [100.0]
    screen._clock = lambda: now[0]
    screen.load_archive(path)
    assert screen.duration_sec == pytest.approx(1.4)
    assert screen._playback.timestamps == timestamps

    screen._toggle_playback()
    now[0] += 0.65
    screen._on_tick()
    reader = screen._playback.reader
    assert reader is not None
    assert reader.index == 5
    assert screen.current_time_sec == pytest.approx(0.5)

    now[0] += 0.15
    screen._on_tick()
    assert reader.index == 6
    assert screen.current_time_sec == pytest.approx(0.8)
    assert "00:00.8" in screen.report_view.toPlainText()
    screen.report_view.anchorClicked.emit(QUrl("seek:0.800000"))
    assert reader.index == 6
    screen.scrubber.setValue(8)
    assert screen.current_time_sec == pytest.approx(1.0)

    screen._toggle_playback()
    now[0] = 101.5
    screen._on_tick()
    assert reader.index == 11
    assert screen.current_time_sec == pytest.approx(1.4)
    assert not screen.is_playing


def test_archive_recordings_variants_and_reports_are_real_and_owned(screen, video, tmp_path):
    path, report = make_archive(tmp_path, video)
    report.write_text("# Second recording\n", encoding="utf-8")
    append_segment(path, report, video, None, [0.0, 0.1])
    screen.load_archive(path)
    archive = screen._playback.archive
    raw_reader = screen._playback.reader
    assert archive is not None
    assert raw_reader is not None
    folder = archive.folder
    assert folder.is_dir()
    assert screen.segment_combo.count() == 2
    assert [screen.stream_combo.itemText(i) for i in range(screen.stream_combo.count())] == [
        "Overlay video",
        "Raw video",
    ]
    assert "Stored report" in screen.report_view.toPlainText()
    assert "Actual recorded findings." in screen.report_view.toPlainText()
    assert screen.stream_combo.currentText() == "Raw video"
    screen.stream_combo.setCurrentText("Overlay video")
    assert screen.stream_combo.currentText() == "Raw video"
    assert raw_reader.path.name == "video_00.mp4"
    screen.segment_combo.setCurrentIndex(1)
    assert not raw_reader.cap.isOpened()
    assert screen.stream_combo.count() == 1
    assert screen.stream_combo.currentText() == "Raw video"
    assert "Unavailable in this recording: Overlay video." in screen.status_label.text()
    assert "Second recording" in screen.report_view.toPlainText()
    assert "frame timestamps do not match" in screen.status_label.text()
    second_reader = screen._playback.reader
    screen.load_archive(video)
    assert second_reader is not None and not second_reader.cap.isOpened()
    assert not folder.exists()
    assert screen._playback.archive is None
    assert screen.report_view.toPlainText() == ""


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "unsupported",
        "corrupt_video",
        "corrupt_archive",
        "empty_archive",
        "missing_member",
    ],
)
def test_open_failure_clears_previous_media_and_reports(screen, video, tmp_path, kind):
    archive_path, _ = make_archive(tmp_path, video)
    screen.load_archive(archive_path)
    archive = screen._playback.archive
    reader = screen._playback.reader
    assert archive is not None and reader is not None
    path = tmp_path / "bad.mp4"
    if kind == "unsupported":
        path = tmp_path / "notes.txt"
        path.write_text("not a supported replay file", encoding="utf-8")
    elif kind == "corrupt_video":
        path.write_bytes(b"not video")
    elif kind == "corrupt_archive":
        path = tmp_path / "bad.tp"
        path.write_bytes(b"not a zip")
    elif kind in ("empty_archive", "missing_member"):
        path = tmp_path / "bad.tp"
        segments = []
        if kind == "missing_member":
            segments = [
                {
                    "index": 0,
                    "video": "video_00.mp4",
                    "overlay": None,
                    "report": "report_00.md",
                    "metadata": "report_00.metadata.json",
                }
            ]
        with ZipFile(path, "w") as stored:
            stored.writestr("manifest.json", json.dumps({"version": 1, "segments": segments}))
    screen.load_archive(path)
    assert "Replay unavailable:" in screen.status_label.text()
    assert not reader.cap.isOpened()
    assert not archive.folder.exists()
    assert screen._playback.reader is None
    assert screen._playback.archive is None
    assert screen.primary_stage._image is None
    assert screen.duration_sec == 0
    assert not screen.play_btn.isEnabled()
    assert not screen.scrubber.isEnabled()
    assert screen.report_view.toPlainText() == ""


def test_report_only_recording_is_explicit_and_other_recordings_can_still_play(
    screen,
    video,
    tmp_path,
):
    path = tmp_path / "report-only.tp"
    report = tmp_path / "report.md"
    report.write_text("# Real report without media\n", encoding="utf-8")
    append_segment(path, report, None, None, [])
    append_segment(path, report, video, None, [])
    screen.load_archive(path)
    assert "no raw or overlay video available" in screen.status_label.text()
    assert not screen.play_btn.isEnabled()
    assert not screen.stream_combo.isEnabled()
    assert screen.primary_stage._image is None
    assert "Real report without media" in screen.report_view.toPlainText()
    screen.segment_combo.setCurrentIndex(1)
    assert screen.play_btn.isEnabled()
    assert screen._playback.reader is not None


def test_bad_overlay_is_not_replaced_with_fake_content_and_raw_can_be_selected(
    screen,
    video,
    tmp_path,
):
    path = tmp_path / "bad-overlay.tp"
    report = tmp_path / "report.md"
    report.write_text("# Real report", encoding="utf-8")
    overlay = tmp_path / "overlay.mp4"
    overlay.write_bytes(b"broken video")
    append_segment(path, report, video, overlay, [])
    screen.load_archive(path)
    assert screen.stream_combo.currentText() == "Raw video"
    screen.stream_combo.setCurrentText("Overlay video")
    assert screen.stream_combo.currentText() == "Raw video"
    assert screen.play_btn.isEnabled()
    assert screen.duration_sec == pytest.approx(1.2)
    assert screen._playback.reader.path.name == "video_00.mp4"


def test_archive_metadata_cannot_supply_external_unarchived_variants(screen, video, tmp_path):
    path = tmp_path / "external.tp"
    with ZipFile(path, "w") as stored:
        stored.writestr(
            "manifest.json",
            json.dumps(
                {
                    "version": 1,
                    "segments": [
                        {
                            "index": 0,
                            "video": None,
                            "overlay": None,
                            "report": "report_00.md",
                            "metadata": "report_00.metadata.json",
                        }
                    ],
                }
            ),
        )
        stored.writestr("report_00.md", "# Real report")
        stored.writestr(
            "report_00.metadata.json",
            json.dumps(
                {
                    "version": 1,
                    "report_file": "report_00.md",
                    "raw_video": str(video),
                    "overlay_video": str(video),
                    "frame_timestamps": [],
                }
            ),
        )
    screen.load_archive(path)
    assert "no raw or overlay video available" in screen.status_label.text()
    assert not screen.play_btn.isEnabled()
    assert screen.stream_combo.count() == 0


@pytest.mark.parametrize("operation", ["playback", "seek"])
def test_decode_failure_stops_and_releases(screen, video, monkeypatch, operation):
    now = [0.0]
    screen._clock = lambda: now[0]
    screen.load_archive(video)
    reader = screen._playback.reader
    assert reader is not None

    def broken_frame(index):
        raise ValueError(f"Cannot decode video frame {index}")

    monkeypatch.setattr(reader, "frame", broken_frame)
    if operation == "playback":
        screen._toggle_playback()
        now[0] = 0.5
        screen._on_tick()
    else:
        screen.scrubber.setValue(5)
    assert "Cannot decode video frame 5" in screen.status_label.text()
    assert not screen.timer.isActive()
    assert not screen.is_playing
    assert not reader.cap.isOpened()
    assert screen.primary_stage._image is None
    assert not screen.play_btn.isEnabled()


def test_navigation_parent_close_and_destruction_release_resources(qapp, video, tmp_path):
    stack = QStackedWidget()
    screen = ReviewScreen(ServiceBridge(Config()), Router())
    stack.addWidget(screen)
    stack.addWidget(QWidget())
    stack.show()
    qapp.processEvents()
    path, _ = make_archive(tmp_path, video)
    screen.load_archive(path)
    archive = screen._playback.archive
    reader = screen._playback.reader
    screen._toggle_playback()
    stack.setCurrentIndex(1)
    assert archive is not None and reader is not None
    assert not reader.cap.isOpened()
    assert not archive.folder.exists()
    assert not screen.timer.isActive()

    stack.setCurrentIndex(0)
    qapp.processEvents()
    assert screen.current_archive_path == path
    assert screen._playback.reader is not None
    assert screen._playback.archive is not None
    screen.load_archive(path)
    archive = screen._playback.archive
    reader = screen._playback.reader
    stack.close()
    assert archive is not None and reader is not None
    assert not reader.cap.isOpened()
    assert not archive.folder.exists()

    # An unshown widget can be destroyed without receiving a hide/close event.
    playback = ReviewPlayback()
    unshown = ReviewScreen(ServiceBridge(Config()), Router(), playback=playback)
    unshown.load_archive(path)
    archive = playback.archive
    reader = playback.reader
    unshown.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert archive is not None and reader is not None
    assert not reader.cap.isOpened()
    assert not archive.folder.exists()
    playback.close()  # Cleanup is idempotent.


def test_review_sorts_calibrations_by_name_and_date_with_formatted_timestamps(qapp, tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    cal_dir = tmp_path / "calibrations"
    base_range = ApprovedRange(
        lower=-10.0,
        upper=10.0,
        unit="degrees",
        view="all",
        passages=[Passage(start=0.0, end=2.0)],
        median=0.0,
        p10=-5.0,
        p90=5.0,
        valid_duration=2.0,
        selected_duration=2.0,
        samples=20,
        inside_fraction=1.0,
    )
    entries = [
        ("z-id", "Zulu Walk", "2026-11-10T09:15:00", "zulu.tpc"),
        ("a-id", "Alpha Embrace", "2026-11-14T20:05:00", "alpha.tpc"),
        ("m-id", "Medio Pivot", "2026-11-12T14:30:00", "medio.tpc"),
    ]
    for cal_id, name, created_at, filename in entries:
        rp = RangeProfile(
            id=cal_id,
            name=name,
            created_at_utc=created_at,
            context="Solo walking",
            model_hash=MANIFEST["sha256"],
            capture_profile=profile(cfg, 64, 48, "video"),
            source={"archive": "ref.tpr"},
            metrics={"head_pitch": base_range},
        )
        save_range_profile(cal_dir / filename, rp)

    widget = ReviewScreen(ServiceBridge(cfg), Router())
    try:
        # Default sort: Name (A-Z) with dd/mm/yyyy HH:mm label
        labels = [widget.reference_combo.itemText(i) for i in range(1, widget.reference_combo.count())]
        assert labels == [
            "Alpha Embrace (14/11/2026 20:05)",
            "Medio Pivot (12/11/2026 14:30)",
            "Zulu Walk (10/11/2026 09:15)",
        ]

        widget.reference_sort_combo.setCurrentIndex(
            widget.reference_sort_combo.findData("name_desc")
        )
        labels = [widget.reference_combo.itemText(i) for i in range(1, widget.reference_combo.count())]
        assert labels == [
            "Zulu Walk (10/11/2026 09:15)",
            "Medio Pivot (12/11/2026 14:30)",
            "Alpha Embrace (14/11/2026 20:05)",
        ]

        widget.reference_sort_combo.setCurrentIndex(
            widget.reference_sort_combo.findData("date_asc")
        )
        labels = [widget.reference_combo.itemText(i) for i in range(1, widget.reference_combo.count())]
        assert labels == [
            "Zulu Walk (10/11/2026 09:15)",
            "Medio Pivot (12/11/2026 14:30)",
            "Alpha Embrace (14/11/2026 20:05)",
        ]

        widget.reference_sort_combo.setCurrentIndex(
            widget.reference_sort_combo.findData("date_desc")
        )
        labels = [widget.reference_combo.itemText(i) for i in range(1, widget.reference_combo.count())]
        assert labels == [
            "Alpha Embrace (14/11/2026 20:05)",
            "Medio Pivot (12/11/2026 14:30)",
            "Zulu Walk (10/11/2026 09:15)",
        ]
    finally:
        widget.close()

