"""Qt UI tests for Solo Camera Calibration navigation, camera ownership, and staging."""

from __future__ import annotations

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

    home: HomeScreen = window.screens[Route.HOME]  # type: ignore[assignment]
    ref_screen: ReferencesScreen = window.screens[Route.REFERENCES]  # type: ignore[assignment]
    assert hasattr(ref_screen, "new_solo_cal_btn")
    assert ref_screen.new_solo_cal_btn.text() == "New solo camera calibration…"
    assert hasattr(ref_screen, "check_camera_btn")
    assert ref_screen.check_camera_btn.text() == "Check with camera…"

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
    assert wizard.side_combo.currentText() == "left"
    assert wizard.facing_combo.currentText() == "right"

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
