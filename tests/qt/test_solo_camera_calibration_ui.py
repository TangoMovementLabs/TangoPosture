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


def test_guided_enter_review_announces_saves_and_stays_open_for_review(qapp, tmp_path, monkeypatch):
    from tempfile import TemporaryDirectory

    import cv2
    import numpy as np

    from tango_posture.archive import append_segment, practice_directory, practice_path
    from tango_posture.calibration.archives import read_range_profile
    from tango_posture.calibration.camera_session import AcceptedPassageRecord

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

    wizard._enter_review()
    qapp.processEvents()

    assert wizard.stack.currentIndex() == 2
    assert wizard.isVisible()
    assert spoken == [(0, "Self calibration completed. Save it and review it.")]
    assert "Self calibration completed. Save it and review it." in wizard.review_summary.text()
    assert "Self calibration completed. Saved:" in wizard.status_banner.text()
    assert len(wizard.saved_paths) == 2
    for saved in wizard.saved_paths:
        assert saved.is_file()
        assert saved.suffix == ".tpc"
        assert saved.parent == tmp_path / "calibrations"
        prof = read_range_profile(saved)
        assert prof.metrics

    listed_paths = {ref.path.resolve() for ref in bridge.list_references() if ref.is_archive}
    assert {p.resolve() for p in wizard.saved_paths} <= listed_paths

    # Clicking Save ready profiles again succeeds cleanly with overwrite=True and keeps embedded view visible
    wizard.save_views_btn.click()
    qapp.processEvents()
    assert wizard.isVisible()
    assert len(wizard.saved_paths) == 2

    wizard.close()
    temp_dir.cleanup()
