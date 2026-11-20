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

    # Reopening via Edit calibration works cleanly in profile-only mode without a .tpr file
    edited_dialogs: list[CalibrationEditor] = []

    class ProfileOnlyEditDialog(CalibrationEditor):
        def exec(self) -> int:
            edited_dialogs.append(self)
            assert self._profile_only
            assert self.chart.data is not None and len(self.chart.data.segments) >= 1
            assert self.distribution.pixmap() is not None and not self.distribution.pixmap().isNull()
            self.lower_spin.setValue(self.lower_spin.value() - 1.0)
            self._approve()
            self._save()
            return self.result()

    monkeypatch.setattr(
        "tango_posture.desktop_qt.calibration.editor.CalibrationEditor",
        ProfileOnlyEditDialog,
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

    wizard.close()
    temp_dir.cleanup()


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
