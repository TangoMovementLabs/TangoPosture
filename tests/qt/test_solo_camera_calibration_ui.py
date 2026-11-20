"""Qt UI tests for Solo Camera Calibration navigation, camera ownership, and staging."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from tango_posture.calibration.archives import save_range_profile
from tango_posture.calibration.baseline import profile as build_capture_profile
from tango_posture.calibration.models import ApprovedRange, Passage, RangeProfile
from tango_posture.config import Config
from tango_posture.desktop_qt.calibration.guided import GuidedCalibration
from tango_posture.desktop_qt.router import Route, Router
from tango_posture.desktop_qt.screens.home import HomeScreen
from tango_posture.desktop_qt.screens.references import ReferencesScreen
from tango_posture.desktop_qt.service_bridge import ServiceBridge
from tango_posture.desktop_qt.shell import MainWindow
from tango_posture.models import MANIFEST


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(["-platform", "offscreen"])


def test_calibration_menu_and_home_entry_point(qapp, tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    window = MainWindow(cfg)
    assert Route.REFERENCES in window.nav_buttons
    assert window.nav_buttons[Route.REFERENCES].text() == "Self Calibration"
    assert Route.PRACTICE in window.nav_buttons
    assert window.nav_buttons[Route.PRACTICE].text() == "Self Practice"

    practice = window.screens[Route.PRACTICE]
    assert not hasattr(practice, "stop_btn")
    assert not hasattr(practice, "abort_btn")
    assert not hasattr(practice, "calibrate_btn")
    assert practice.video_path.isHidden()
    assert hasattr(practice, "manual_notes")
    assert hasattr(practice, "save_notes_btn")
    assert practice.record_btn.isEnabled()
    assert [practice.mode_combo.itemData(i) for i in range(practice.mode_combo.count())] == ["auto", "single", "couple"]
    assert hasattr(practice, "head_align_cb")
    assert hasattr(practice, "shoulders_cb")
    assert hasattr(practice, "role_cb")
    assert hasattr(practice, "embrace_cb")

    home: HomeScreen = window.screens[Route.HOME]  # type: ignore[assignment]
    ref_screen: ReferencesScreen = window.screens[Route.REFERENCES]  # type: ignore[assignment]
    assert hasattr(ref_screen, "new_solo_cal_btn")
    assert ref_screen.new_solo_cal_btn.text() == "New Calibration"
    assert hasattr(ref_screen, "guided_wizard")
    assert hasattr(ref_screen.guided_wizard, "filename_preview")
    assert hasattr(ref_screen.guided_wizard, "review_name_input")

    window.router.navigate(Route.HOME)
    qapp.processEvents()
    # Home button navigates to Self Calibration
    home_buttons = [b for b in home.findChildren(type(ref_screen.new_solo_cal_btn)) if b.text() == "Open Self Calibration"]
    assert len(home_buttons) == 1
    home_buttons[0].click()
    qapp.processEvents()
    assert window.stack.currentWidget() is ref_screen
    window.close()


def test_references_lists_range_profile_and_check_with_camera_prepares_wizard(qapp, tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    cal_dir = tmp_path / "calibrations"
    cal_dir.mkdir(parents=True)
    prof = RangeProfile(
        name="Studio webcam · Neutral · Side",
        context="Neutral posture",
        model_hash=MANIFEST["sha256"],
        capture_profile=build_capture_profile(cfg, 640, 480, "camera:0"),
        source={"archive_sha256": "a" * 64, "segment_index": 0},
        metrics={
            "head_pitch": ApprovedRange(
                lower=-5.0,
                upper=5.0,
                unit="degrees",
                view="side",
                passages=[Passage(start=0.0, end=2.0)],
                median=0.0,
                p10=-3.0,
                p90=3.0,
                valid_duration=2.0,
                selected_duration=2.0,
                samples=20,
                inside_fraction=1.0,
                camera_check_passed=True,
            )
        },
    )
    prof_path = cal_dir / "studio.tpc"
    save_range_profile(prof_path, prof)

    bridge = ServiceBridge(cfg)
    screen = ReferencesScreen(bridge, Router())
    assert screen.table.rowCount() == 1
    assert screen.table.item(0, 1).text() == "Neutral posture"
    assert screen.table.item(0, 2).text() == "Side"
    assert screen.table.item(0, 4).text() == "1/1"
    assert screen.check_camera_btn.isEnabled()

    wizard = GuidedCalibration(
        bridge,
        screen,
        target_profile=prof,
        target_profile_path=prof_path,
        camera_discoverer=lambda configured_index=0: [(0, "Camera 0")],
    )
    # Camera stays off until user clicks Start camera preview
    assert not wizard._camera_active
    assert wizard.check_boxes["head_pitch"].isChecked()
    assert not hasattr(wizard, "side_combo")
    assert not hasattr(wizard, "facing_combo")
    assert not hasattr(screen, "tabs")
    assert not screen.guided_wizard._camera_active

    # Simulated camera error shows actionable message and Retry camera
    wizard._on_worker_failed("Camera busy")
    assert not wizard.retry_camera_btn.isHidden()
    assert "Retry camera" in wizard.status_banner.text()

    # Simulated camera frame renders onto preview_stage
    import numpy as np

    from tango_posture.desktop_qt.practice.worker import QtPracticeDisplay

    wizard._display = QtPracticeDisplay(capture_evidence=True)
    wizard._display.frames.put(np.zeros((120, 160, 3), dtype=np.uint8))
    wizard._poll_worker()
    assert wizard.preview_stage._image is not None and not wizard.preview_stage._image.isNull()

    wizard.close()
    screen.close()


def test_guided_enter_review_announces_once_opens_editor_and_saves_single_tpc(
    qapp, tmp_path, monkeypatch
):
    from tempfile import TemporaryDirectory

    import cv2
    import numpy as np

    import tango_posture.desktop_qt.calibration.guided as guided_mod
    from tango_posture.archive import append_segment, practice_directory, practice_path
    from tango_posture.calibration.archives import read_range_profile
    from tango_posture.calibration.camera_session import AcceptedPassageRecord
    from tango_posture.desktop_qt.calibration.editor import (
        CalibrationEditor,
        open_profile_calibration_editor,
    )
    from tango_posture.desktop_qt.screens.library import LibraryScreen

    cfg = Config(data_dir=str(tmp_path))
    bridge = ServiceBridge(cfg)
    wizard = GuidedCalibration(bridge, embedded=True, profile_name="My studio")
    wizard.speak_cb.setChecked(True)
    wizard.show()
    qapp.processEvents()

    temp_dir = TemporaryDirectory(dir=tmp_path)
    temp_root = Path(temp_dir.name)
    staging_dir = temp_root / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    prac_dir = practice_directory(str(temp_root))
    prac_dir.mkdir(parents=True, exist_ok=True)

    video = temp_root / "video.mp4"
    report = temp_root / "report.md"
    report.write_text("# Guided evidence\n", encoding="utf-8")
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (640, 480))
    assert writer.isOpened()

    front_pose = {
        "nose": [0.50, 0.20, 0.0, 0.99, 0.99],
        "left_ear": [0.54, 0.20, 0.0, 0.99, 0.99],
        "right_ear": [0.46, 0.20, 0.0, 0.99, 0.99],
        "left_shoulder": [0.62, 0.32, 0.0, 0.99, 0.99],
        "right_shoulder": [0.38, 0.32, 0.0, 0.99, 0.99],
        "left_hip": [0.58, 0.58, 0.0, 0.99, 0.99],
        "right_hip": [0.42, 0.58, 0.0, 0.99, 0.99],
        "left_ankle": [0.58, 0.90, 0.0, 0.99, 0.99],
        "right_ankle": [0.42, 0.90, 0.0, 0.99, 0.99],
    }
    side_pose = {
        "nose": [0.53, 0.20, 0.0, 0.99, 0.99],
        "left_ear": [0.50, 0.20, 0.0, 0.99, 0.99],
        "right_ear": [0.495, 0.20, 0.0, 0.45, 0.45],
        "left_shoulder": [0.50, 0.32, 0.0, 0.99, 0.99],
        "right_shoulder": [0.50, 0.32, 0.0, 0.40, 0.40],
        "left_hip": [0.50, 0.58, 0.0, 0.99, 0.99],
        "right_hip": [0.50, 0.58, 0.0, 0.40, 0.40],
        "left_ankle": [0.50, 0.90, 0.0, 0.99, 0.99],
        "right_ankle": [0.50, 0.90, 0.0, 0.40, 0.40],
    }
    entries = []
    timestamps = [i / 10.0 for i in range(30)]
    for idx in range(30):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        pose = front_pose if idx < 15 else side_pose
        entries.append({"poses": [pose]})
    writer.release()

    tp_file = practice_path(prac_dir, "solo_calibration_evidence")
    append_segment(
        tp_file,
        report,
        video,
        None,
        timestamps,
        analytics={
            "model_hash": MANIFEST["sha256"],
            "capture_profile": build_capture_profile(cfg, 640, 480, "camera:0"),
            "frames": entries,
            "max_gap": cfg.max_gap,
        },
    )

    wizard._temp_dir = temp_dir
    wizard._staging_dir = staging_dir
    wizard._frozen_cfg = cfg
    wizard._session._accepted_passages = {
        "front_a": AcceptedPassageRecord(
            phase_id="front_a",
            plan_kind="quick",
            view="front",
            repetition=1,
            role="neutral",
            passage=Passage(start=0.0, end=1.4),
            eligible_metrics={"shoulder_clearance": 1.4},
            usable=True,
        ),
        "side_a": AcceptedPassageRecord(
            phase_id="side_a",
            plan_kind="quick",
            view="side",
            repetition=1,
            role="neutral",
            passage=Passage(start=1.5, end=2.9),
            eligible_metrics={"trunk_inclination": 1.4},
            usable=True,
            detected_side="left",
            detected_facing="right",
        ),
    }
    wizard._session.state = "Review"

    spoken: list[tuple[int, str]] = []
    monkeypatch.setattr(wizard._speech, "speak", lambda pid, text: spoken.append((pid, text)))

    from tango_posture.desktop_qt.calibration import editor as editor_mod

    rendered_side: list[tuple[dict[str, str], bool]] = []
    real_render_approved = editor_mod.render_approved_feedback

    def _spy_render_approved(image, observation, cfg_arg, targets, rows):
        out = real_render_approved(image, observation, cfg_arg, targets, rows)
        row_states = {str(r["metric"]): str(r["state"]) for r in rows}
        rendered_side.append((row_states, bool(np.any(out > 0))))
        return out

    monkeypatch.setattr(editor_mod, "render_approved_feedback", _spy_render_approved)

    opened_editors: list[CalibrationEditor] = []

    class AutoSaveCalibrationEditor(CalibrationEditor):
        def exec(self) -> int:
            opened_editors.append(self)
            assert self.name_input.text() == "My studio"
            # Switch to the side metric ("trunk_inclination") and seek to a side frame (index 15)
            for step_idx in range(self.step_combo.count()):
                self.step_combo.setCurrentIndex(step_idx)
                m_idx = self.metric_combo.findData("trunk_inclination")
                if m_idx >= 0:
                    self.metric_combo.setCurrentIndex(m_idx)
                    break
            assert self.metric_combo.currentData() == "trunk_inclination"
            assert self.chart.data is not None
            assert len([v for _, v in self.chart.data.points if v is not None]) >= 10
            assert self.distribution.pixmap() is not None and not self.distribution.pixmap().isNull()

            rendered_side.clear()
            self.scrubber.setValue(15)
            qapp.processEvents()
            assert rendered_side
            last_states, has_lines = rendered_side[-1]
            assert last_states.get("trunk_inclination") in ("WITHIN_REFERENCE", "DEVIATION")
            assert has_lines

            self.name_input.setText("My Studio Custom Name")
            self._save()
            return self.result()

    monkeypatch.setattr(guided_mod, "CalibrationEditor", AutoSaveCalibrationEditor)

    wizard._enter_review()
    # Simulating speech finishing must NOT loop or re-trigger _enter_review
    wizard._speech.speech_finished.emit(0, False)
    qapp.processEvents()
    wizard._speech.speech_finished.emit(0, False)
    qapp.processEvents()

    assert wizard.stack.currentIndex() == 2
    assert wizard.isVisible()
    assert spoken == [(0, "Self calibration completed.")]
    assert len(opened_editors) == 1
    assert set(opened_editors[0].approved) == {"shoulder_clearance", "trunk_inclination"}

    # Only ONE .tpc file should exist, and NO .tpr or .tp files should be written to data_dir
    tpc_files = list((tmp_path / "calibrations").glob("*.tpc"))
    tpr_files = list((tmp_path / "references").glob("*.tpr")) if (tmp_path / "references").is_dir() else []
    tp_files = list((tmp_path / "practice").glob("*.tp")) if (tmp_path / "practice").is_dir() else []
    assert len(tpc_files) == 1
    assert tpc_files[0].name == "My Studio Custom Name.tpc"
    assert tpr_files == []
    assert tp_files == []

    prof = read_range_profile(tpc_files[0])
    assert set(prof.metrics) == {"shoulder_clearance", "trunk_inclination"}
    assert "archive" not in prof.source
    assert prof.source.get("embedded_evidence") == "evidence.tpr"

    # Reopening via Edit calibration after staging temp_dir is deleted preserves full video & charts
    temp_dir.cleanup()
    edited_dialogs: list[CalibrationEditor] = []

    class EmbeddedEvidenceEditDialog(CalibrationEditor):
        def exec(self) -> int:
            edited_dialogs.append(self)
            assert not self._profile_only
            assert self._reader is not None
            assert self.chart.data is not None and len(self.chart.data.segments) >= 1
            assert self.distribution.pixmap() is not None and not self.distribution.pixmap().isNull()
            self.lower_spin.setValue(self.lower_spin.value() - 1.0)
            self._approve()
            self._save()
            return self.result()

    monkeypatch.setattr(
        "tango_posture.desktop_qt.calibration.editor.CalibrationEditor",
        EmbeddedEvidenceEditDialog,
    )
    saved_after_edit = open_profile_calibration_editor(tpc_files[0], cfg, wizard, bridge=bridge)
    assert saved_after_edit is not None
    assert saved_after_edit.resolve() == tpc_files[0].resolve()
    assert len(edited_dialogs) == 1

    # LibraryScreen "Edit calibration" button also opens cleanly without error
    library = LibraryScreen(bridge, Router())
    try:
        row = next(
            i for i in range(library.table.rowCount()) if library.table.item(i, 4).text() == ".tpc"
        )
        edit_btn = next(
            b
            for b in library.table.cellWidget(row, 3).findChildren(type(wizard.save_views_btn))
            if b.text() == "Edit calibration"
        )
        edit_btn.click()
        assert len(edited_dialogs) == 2
    finally:
        library.close()

    # Legacy .tpc without embedded evidence still opens cleanly in _profile_only mode with visuals
    legacy_path = tmp_path / "calibrations" / "Legacy.tpc"
    save_range_profile(legacy_path, prof, overwrite=True)
    legacy_editor = CalibrationEditor(legacy_path, cfg)
    try:
        assert legacy_editor._profile_only
        assert legacy_editor.chart.data is not None and len(legacy_editor.chart.data.segments) >= 1
        assert (
            legacy_editor.distribution.pixmap() is not None
            and not legacy_editor.distribution.pixmap().isNull()
        )
    finally:
        legacy_editor.reject()

    wizard.close()


def test_guided_waits_for_speech_and_repeats_instruction_after_five_seconds(
    qapp, tmp_path, monkeypatch
):
    import tango_posture.desktop_qt.calibration.guided as guided_mod
    from tango_posture.calibration.camera_session import build_quick_setup_phases
    from tango_posture.desktop_qt.practice.worker import QtPracticeDisplay

    now = [100.0]
    monkeypatch.setattr(guided_mod.time, "monotonic", lambda: now[0])

    cfg = Config(data_dir=str(tmp_path))
    bridge = ServiceBridge(cfg)
    wizard = GuidedCalibration(bridge, embedded=True, profile_name="Timing check")
    wizard.speak_cb.setChecked(True)
    wizard.show()
    qapp.processEvents()

    spoken: list[tuple[int, str]] = []
    speaking = [True]
    monkeypatch.setattr(wizard._speech, "is_speaking", lambda: speaking[0])

    def _fake_speak(pid: int, text: str) -> None:
        spoken.append((pid, text))
        speaking[0] = True

    monkeypatch.setattr(wizard._speech, "speak", _fake_speak)

    # Active preview before capture starts: auto-begin must wait while intro speech is speaking
    wizard._display = QtPracticeDisplay(capture_evidence=True)
    wizard._camera_active = True
    wizard._auto_begin_pending = True
    wizard._auto_begin_armed = False
    wizard._poll_worker()
    assert not wizard._auto_begin_armed

    # Enter capture page in "Wait for view and pose" while speaking: repeat timer must wait
    wizard._auto_begin_pending = False
    wizard.stack.setCurrentIndex(1)
    wizard._session.start_plan(build_quick_setup_phases(), now_monotonic=now[0], speak_enabled=False)
    wizard._repeat_wait_since = None
    now[0] += 6.0
    wizard._update_repeat_timer(now[0])
    assert spoken == []

    # Speech finishes -> wait timer starts from completion (now=106.0)
    speaking[0] = False
    wizard._on_speech_finished(1, False)
    assert wizard._repeat_wait_since == pytest.approx(106.0)

    # 3 seconds later (< 5.0s): no repeat yet, countdown shows in phase label
    now[0] = 109.0
    wizard._update_repeat_timer(now[0])
    assert spoken == []
    assert "Repeat in 2s" in wizard.phase_label.text()

    # 5 seconds after speech ended (111.1 >= 106.0 + 5.0): instruction repeats by voice
    now[0] = 111.1
    wizard._update_repeat_timer(now[0])
    assert len(spoken) == 1
    assert spoken[0][1] == wizard._session.current_instruction

    wizard.close()


def test_guided_four_phase_with_dropouts_and_opposite_sides_preserves_all_metrics_and_embedded_video(
    qapp, tmp_path, monkeypatch
):
    """End-to-end 4-phase Self Calibration with pose dropouts and opposite side holds:
    auto-approves all 8 metrics, plots all timeseries/distributions, embeds video in 1 .tpc,
    and preserves full video + charts after temp_dir cleanup."""
    from tempfile import TemporaryDirectory

    import cv2
    import numpy as np

    import tango_posture.desktop_qt.calibration.guided as guided_mod
    from tango_posture.archive import append_segment, practice_directory, practice_path
    from tango_posture.calibration.archives import has_embedded_evidence, read_range_profile
    from tango_posture.calibration.camera_session import AcceptedPassageRecord
    from tango_posture.calibration.models import METRIC_DETAILS
    from tango_posture.desktop_qt.calibration import editor as editor_mod
    from tango_posture.desktop_qt.calibration.editor import CalibrationEditor

    cfg = Config(data_dir=str(tmp_path))
    bridge = ServiceBridge(cfg)
    wizard = GuidedCalibration(bridge, embedded=True, profile_name="JR Self Cam")
    wizard.speak_cb.setChecked(False)
    wizard.show()
    qapp.processEvents()

    temp_dir = TemporaryDirectory(dir=tmp_path)
    temp_root = Path(temp_dir.name)
    staging_dir = temp_root / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    prac_dir = practice_directory(str(temp_root))
    prac_dir.mkdir(parents=True, exist_ok=True)

    video = temp_root / "jr.mp4"
    report = temp_root / "jr.md"
    report.write_text("# JR Self Cam evidence\n", encoding="utf-8")

    front_pose = {
        "nose": [0.50, 0.20, 0.0, 0.99, 0.99],
        "left_ear": [0.54, 0.20, 0.0, 0.99, 0.99],
        "right_ear": [0.46, 0.20, 0.0, 0.99, 0.99],
        "left_shoulder": [0.62, 0.32, 0.0, 0.99, 0.99],
        "right_shoulder": [0.38, 0.32, 0.0, 0.99, 0.99],
        "left_hip": [0.58, 0.58, 0.0, 0.99, 0.99],
        "right_hip": [0.42, 0.58, 0.0, 0.99, 0.99],
        "left_knee": [0.58, 0.74, 0.0, 0.99, 0.99],
        "right_knee": [0.42, 0.74, 0.0, 0.99, 0.99],
        "left_ankle": [0.58, 0.90, 0.0, 0.99, 0.99],
        "right_ankle": [0.42, 0.90, 0.0, 0.99, 0.99],
    }

    def _side_pose(near_side: str, facing: str) -> dict[str, list[float]]:
        far_side = "right" if near_side == "left" else "left"
        nose_x = 0.56 if facing == "right" else 0.44
        return {
            "nose": [nose_x, 0.20, 0.0, 0.99, 0.99],
            f"{near_side}_ear": [0.50, 0.20, 0.0, 0.99, 0.99],
            f"{far_side}_ear": [0.50, 0.20, 0.0, 0.10, 0.10],
            f"{near_side}_shoulder": [0.50, 0.32, 0.0, 0.99, 0.99],
            f"{far_side}_shoulder": [0.495, 0.32, 0.0, 0.40, 0.40],
            f"{near_side}_hip": [0.50, 0.58, 0.0, 0.99, 0.99],
            f"{far_side}_hip": [0.495, 0.58, 0.0, 0.40, 0.40],
            f"{near_side}_knee": [0.52, 0.74, 0.0, 0.99, 0.99],
            f"{far_side}_knee": [0.49, 0.74, 0.0, 0.40, 0.40],
            f"{near_side}_ankle": [0.50, 0.90, 0.0, 0.99, 0.99],
            f"{far_side}_ankle": [0.50, 0.90, 0.0, 0.40, 0.40],
        }

    # 70 frames @ 10fps (0.0..6.9s) with explicit dropout gaps between holds:
    # 0..14 (0.0..1.4s): front_a
    # 15..16 (1.5..1.6s): DROPOUT (poses=[])
    # 17..31 (1.7..3.1s): front_b
    # 32..33 (3.2..3.3s): DROPOUT (poses=[])
    # 34..48 (3.4..4.8s): side_a (right/right, occluded left ear/shoulder/hip)
    # 49..50 (4.9..5.0s): DROPOUT (poses=[])
    # 51..65 (5.1..6.5s): side_b (left/left, occluded right ear/shoulder/hip)
    # 66..69: side_b tail
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (640, 480))
    assert writer.isOpened()
    entries = []
    timestamps = [i / 10.0 for i in range(70)]
    for idx in range(70):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        if idx in (15, 16, 32, 33, 49, 50):
            entries.append({"poses": []})
        elif idx <= 31:
            entries.append({"poses": [front_pose]})
        elif idx <= 48:
            entries.append({"poses": [_side_pose("right", "right")]})
        else:
            entries.append({"poses": [_side_pose("left", "left")]})
    writer.release()

    tp_file = practice_path(prac_dir, "solo_calibration_evidence")
    append_segment(
        tp_file,
        report,
        video,
        None,
        timestamps,
        analytics={
            "model_hash": MANIFEST["sha256"],
            "capture_profile": build_capture_profile(cfg, 640, 480, "camera:0"),
            "frames": entries,
            "max_gap": cfg.max_gap,
        },
    )

    wizard._temp_dir = temp_dir
    wizard._staging_dir = staging_dir
    wizard._frozen_cfg = cfg
    front_m = {"shoulder_clearance": 1.4, "torso_axis_tilt": 1.4}
    side_m = {
        "head_pitch": 1.4,
        "head_forward": 1.4,
        "head_inclination": 1.4,
        "trunk_inclination": 1.4,
        "hip_offset": 1.4,
        "knee_flexion": 1.4,
    }
    wizard._session._accepted_passages = {
        "front_a": AcceptedPassageRecord(
            phase_id="front_a", plan_kind="quick", view="front", repetition=1, role="neutral",
            passage=Passage(start=0.0, end=1.4), eligible_metrics=front_m, usable=True,
            detected_side="left", detected_facing="right",
        ),
        "front_b": AcceptedPassageRecord(
            phase_id="front_b", plan_kind="quick", view="front", repetition=2, role="neutral",
            passage=Passage(start=1.7, end=3.1), eligible_metrics=front_m, usable=True,
            detected_side="left", detected_facing="right",
        ),
        "side_a": AcceptedPassageRecord(
            phase_id="side_a", plan_kind="quick", view="side", repetition=1, role="neutral",
            passage=Passage(start=3.4, end=4.8), eligible_metrics=side_m, usable=True,
            detected_side="right", detected_facing="right",
        ),
        "side_b": AcceptedPassageRecord(
            phase_id="side_b", plan_kind="quick", view="side", repetition=2, role="neutral",
            passage=Passage(start=5.1, end=6.5), eligible_metrics=side_m, usable=True,
            detected_side="left", detected_facing="left",
        ),
    }
    wizard._session.state = "Review"

    rendered_overlays: list[tuple[dict[str, str], bool]] = []
    real_render_approved = editor_mod.render_approved_feedback

    def _spy_render(image, observation, cfg_arg, targets, rows):
        out = real_render_approved(image, observation, cfg_arg, targets, rows)
        rendered_overlays.append(({str(r["metric"]): str(r["state"]) for r in rows}, bool(np.any(out > 0))))
        return out

    monkeypatch.setattr(editor_mod, "render_approved_feedback", _spy_render)

    class InspectAndSaveEditor(CalibrationEditor):
        def exec(self) -> int:
            assert set(self.approved) == set(METRIC_DETAILS)
            for step_idx in range(self.step_combo.count()):
                self.step_combo.setCurrentIndex(step_idx)
                for m_idx in range(self.metric_combo.count()):
                    self.metric_combo.setCurrentIndex(m_idx)
                    assert self.chart.data is not None
                    valid_pts = [v for _, v in self.chart.data.points if v is not None]
                    assert len(valid_pts) >= 10
                    assert self.chart.states
                    assert self.distribution.pixmap() is not None and not self.distribution.pixmap().isNull()

            # Verify both side_a (idx 38) and side_b (idx 58) evaluate all side metrics & draw lines
            for frame_idx in (38, 58):
                rendered_overlays.clear()
                self.scrubber.setValue(frame_idx)
                qapp.processEvents()
                assert rendered_overlays
                states_map, has_lines = rendered_overlays[-1]
                for side_key in side_m:
                    assert states_map.get(side_key) in ("WITHIN_REFERENCE", "DEVIATION", "PENDING")
                assert has_lines

            self._save()
            return self.result()

    monkeypatch.setattr(guided_mod, "CalibrationEditor", InspectAndSaveEditor)
    wizard._enter_review()
    qapp.processEvents()

    tpc_files = list((tmp_path / "calibrations").glob("*.tpc"))
    assert len(tpc_files) == 1
    assert tpc_files[0].name == "JR Self Cam.tpc"
    assert not (tmp_path / "references").exists() or not list((tmp_path / "references").iterdir())
    assert has_embedded_evidence(tpc_files[0])
    assert "JR Self Cam.tpc" in wizard.review_summary.text()

    prof = read_range_profile(tpc_files[0])
    assert set(prof.metrics) == set(METRIC_DETAILS)

    # Delete staging temp_dir, reopen JR Self Cam.tpc, and verify full video + charts remain intact
    temp_dir.cleanup()
    reopened = CalibrationEditor(tpc_files[0], cfg)
    try:
        assert not reopened._profile_only
        assert reopened._reader is not None
        assert set(reopened.approved) == set(METRIC_DETAILS)
        for step_idx in range(reopened.step_combo.count()):
            reopened.step_combo.setCurrentIndex(step_idx)
            for m_idx in range(reopened.metric_combo.count()):
                reopened.metric_combo.setCurrentIndex(m_idx)
                assert reopened.chart.data is not None
                assert len([v for _, v in reopened.chart.data.points if v is not None]) >= 10
                assert reopened.distribution.pixmap() is not None and not reopened.distribution.pixmap().isNull()
        reopened.lower_spin.setValue(reopened.lower_spin.value() - 0.5)
        reopened._approve()
        reopened._save()
        assert reopened.saved_path is not None
        assert has_embedded_evidence(reopened.saved_path)
    finally:
        if reopened.isVisible():
            reopened.reject()

    wizard.close()


def test_speech_manager_queues_and_pads_tail_silence_without_cutting_off(qapp, tmp_path, monkeypatch):
    import threading
    import time
    import wave

    import tango_posture.desktop_qt.practice.worker as worker_mod
    from tango_posture.desktop_qt.practice.worker import SpeechManager, _pad_wav_silence

    # 1. Verify _pad_wav_silence appends exact trailing silence frames
    raw_wav = tmp_path / "raw.wav"
    padded_wav = tmp_path / "padded.wav"
    with wave.open(str(raw_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x01\x02" * 100)
    assert _pad_wav_silence(raw_wav, padded_wav, 0.40)
    with wave.open(str(padded_wav), "rb") as r:
        expected_pad = round(22050 * 0.40)
        assert r.getnframes() == 100 + expected_pad
        all_bytes = r.readframes(r.getnframes())
        assert all_bytes[:200] == b"\x01\x02" * 100
        assert all_bytes[200:] == b"\x00" * (expected_pad * 2)

    # 2. Verify SpeechManager serializes queued utterances via say->WAV->afplay without killing active speech
    monkeypatch.setattr(SpeechManager, "is_available", staticmethod(lambda: True))
    orig_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: True if str(self) == "/usr/bin/afplay" else orig_is_file(self),
    )

    first_play_started = threading.Event()
    release_first_play = threading.Event()
    commands_run: list[list[str]] = []
    terminated_cmds: list[list[str]] = []
    padded_frame_counts: list[int] = []

    class FakeProc:
        def __init__(self, cmd: list[str]) -> None:
            self.cmd = list(cmd)
            self._cancelled = threading.Event()
            commands_run.append(self.cmd)
            if self.cmd[0] == "/usr/bin/say" and "-o" in self.cmd:
                out_path = Path(self.cmd[self.cmd.index("-o") + 1])
                with wave.open(str(out_path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(22050)
                    w.writeframes(b"\x10\x00" * 50)

        def wait(self, timeout: float | None = None) -> int:
            if self.cmd[0] == "/usr/bin/afplay":
                with wave.open(self.cmd[1], "rb") as r:
                    padded_frame_counts.append(r.getnframes())
                if len(padded_frame_counts) == 1:
                    first_play_started.set()
                    while not release_first_play.is_set() and not self._cancelled.is_set():
                        time.sleep(0.01)
            return -15 if self._cancelled.is_set() else 0

        def terminate(self) -> None:
            terminated_cmds.append(self.cmd)
            self._cancelled.set()

        def kill(self) -> None:
            self._cancelled.set()

    monkeypatch.setattr(worker_mod.subprocess, "Popen", lambda cmd, **kw: FakeProc(cmd))

    mgr = SpeechManager(timeout_s=5.0, tail_silence_s=0.40, post_speech_delay_s=0.0)
    finished: list[tuple[int, bool]] = []
    mgr.speech_finished.connect(lambda pid, failed: finished.append((pid, failed)))

    mgr.speak(1, "First instruction")
    assert first_play_started.wait(timeout=2.0)
    assert mgr.is_speaking()

    # Queue second phrase while first is still playing -> first must NOT be terminated
    mgr.speak(2, "Second instruction")
    assert terminated_cmds == []
    assert mgr.is_speaking()

    release_first_play.set()
    deadline = time.monotonic() + 2.0
    while (mgr.is_speaking() or len(finished) < 2) and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert not mgr.is_speaking()
    assert terminated_cmds == []
    assert finished == [(1, False), (2, False)]
    assert len(padded_frame_counts) == 2
    assert all(cnt == 50 + round(22050 * 0.40) for cnt in padded_frame_counts)

    # 3. Verify cancel_all stops active utterance, clears pending queue, suppresses signals, and resets state
    block_play = threading.Event()
    play3_started = threading.Event()
    finished.clear()
    terminated_cmds.clear()

    class BlockingFakeProc(FakeProc):
        def wait(self, timeout: float | None = None) -> int:
            if self.cmd[0] == "/usr/bin/afplay":
                play3_started.set()
                while not block_play.is_set() and not self._cancelled.is_set():
                    time.sleep(0.01)
            return -15 if self._cancelled.is_set() else 0

    monkeypatch.setattr(worker_mod.subprocess, "Popen", lambda cmd, **kw: BlockingFakeProc(cmd))
    mgr.speak(3, "Third instruction")
    assert play3_started.wait(timeout=2.0)
    mgr.speak(4, "Queued fourth instruction")
    mgr.cancel_all()
    assert not mgr.is_speaking()
    assert any(cmd[0] == "/usr/bin/afplay" for cmd in terminated_cmds)
    time.sleep(0.05)
    qapp.processEvents()
    assert finished == []

    # Subsequent speak() after cancel_all() works normally
    block_play.set()
    mgr.speak(5, "Recovered instruction")
    deadline = time.monotonic() + 2.0
    while (mgr.is_speaking() or len(finished) < 1) and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert finished == [(5, False)]
    assert not mgr.is_speaking()


def test_guided_speech_transitions_do_not_cancel_completion_or_checks(qapp, tmp_path, monkeypatch):
    import tango_posture.desktop_qt.calibration.guided as guided_mod
    from tango_posture.calibration.camera_session import AcceptedPassageRecord

    cfg = Config(data_dir=str(tmp_path))
    bridge = ServiceBridge(cfg)
    wizard = GuidedCalibration(bridge, embedded=True, profile_name="Speech flow")
    wizard.speak_cb.setChecked(True)

    spoken: list[tuple[int, str]] = []
    cancel_calls: list[bool] = []
    monkeypatch.setattr(wizard._speech, "speak", lambda pid, text: spoken.append((pid, text)))
    orig_cancel = wizard._speech.cancel_all
    monkeypatch.setattr(
        wizard._speech,
        "cancel_all",
        lambda: (cancel_calls.append(True), orig_cancel())[-1],
    )
    monkeypatch.setattr(wizard, "_review_in_editor", lambda: None)

    # _enter_review must NOT cancel_all() before speaking "Self calibration completed."
    wizard._session._accepted_passages = {
        "front_a": AcceptedPassageRecord(
            phase_id="front_a",
            plan_kind="quick",
            view="front",
            repetition=1,
            role="neutral",
            passage=Passage(start=0.0, end=2.0),
            eligible_metrics={"shoulder_clearance": 2.0},
            usable=True,
        )
    }
    wizard._enter_review()
    qapp.processEvents()
    assert cancel_calls == []
    assert spoken == [(0, "Self calibration completed.")]

    # _run_selected_checks and _run_walking_plan must pass announce=False to _start_preview
    preview_announce_args: list[bool] = []

    def _fake_start_preview(*, auto_begin: bool = False, announce: bool = True) -> None:
        preview_announce_args.append(announce)
        if announce and wizard.speak_cb.isChecked():
            wizard._speech.speak(0, "Camera preview started.")

    monkeypatch.setattr(wizard, "_start_preview", _fake_start_preview)
    monkeypatch.setattr(guided_mod, "build_check_phases", lambda ids: [])
    monkeypatch.setattr(guided_mod, "build_walking_phases", list)

    wizard.check_boxes["head_pitch"].setChecked(True)
    spoken.clear()
    wizard._run_selected_checks()
    wizard._run_walking_plan()
    assert preview_announce_args == [False, False]
    assert not any("Camera preview started" in txt for _, txt in spoken)

    wizard.close()
