"""Qt UI telemetry tests verifying menus, submenus, tabs, buttons, dialogs, and routing."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QMenu,
    QMenuBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tango_posture import troubleshoot
from tango_posture.config import Config
from tango_posture.desktop_qt.router import Route
from tango_posture.desktop_qt.shell import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    return app


@pytest.fixture(autouse=True)
def _cleanup_troubleshoot():
    yield
    troubleshoot.shutdown_troubleshoot()


def test_qt_ui_telemetry_logs_menus_submenus_tabs_buttons_and_dialogs(
    qapp: QApplication, tmp_path: Path
) -> None:
    log_path = troubleshoot.enable_troubleshoot(
        tmp_path / "logs",
        source="qt-ui-test",
        heartbeat_interval_s=0.0,
        include_profiler=False,
    )

    root = QWidget()
    root.setObjectName("TestRoot")
    root.setWindowTitle("Test UI Telemetry Window")
    layout = QVBoxLayout(root)

    menubar = QMenuBar(root)
    layout.setMenuBar(menubar)
    file_menu = menubar.addMenu("File")
    recent_submenu = QMenu("Recent", file_menu)
    file_menu.addMenu(recent_submenu)
    open_clip_action = recent_submenu.addAction("clip_01.mp4")

    tabs = QTabWidget(root)
    tabs.setObjectName("MainTabs")
    tabs.addTab(QWidget(), "General")
    tabs.addTab(QWidget(), "Calibration")
    layout.addWidget(tabs)

    action_btn = QPushButton("Analyze Clip", root)
    action_btn.setObjectName("analyzeBtn")
    layout.addWidget(action_btn)

    toggle_cb = QCheckBox("Mirror preview", root)
    toggle_cb.setObjectName("mirrorCb")
    layout.addWidget(toggle_cb)

    combo = QComboBox(root)
    combo.setObjectName("modelCombo")
    combo.addItems(["lite", "full", "heavy"])
    layout.addWidget(combo)

    dialog = QDialog(root)
    dialog.setObjectName("SampleDialog")
    dialog.setWindowTitle("Export Summary")

    root.show()
    qapp.processEvents()

    # Open submenu and trigger nested menu action.
    recent_submenu.show()
    qapp.processEvents()
    open_clip_action.trigger()
    recent_submenu.hide()

    # Switch tab submenu, click button, toggle checkbox, select combo, open/close dialog.
    tabs.setCurrentIndex(1)
    action_btn.click()
    toggle_cb.click()
    combo.activated.emit(2)
    dialog.show()
    qapp.processEvents()
    dialog.close()
    root.close()
    qapp.processEvents()

    troubleshoot.shutdown_troubleshoot()
    content = log_path.read_text(encoding="utf-8")

    assert "| ui.init | Installed Qt UI event filter on QApplication" in content
    assert "| ui.window | Window opened: QWidget#TestRoot ('Test UI Telemetry Window')" in content
    assert "| ui.submenu | Submenu opened: path=File > Recent" in content
    assert "| ui.menu.action | Triggered: File > Recent > 'clip_01.mp4'" in content
    assert "| ui.tab | Tab/Submenu selected: index=1 label='Calibration'" in content
    assert "| ui.button | Clicked: QWidget#TestRoot > QPushButton#analyzeBtn ('Analyze Clip')" in content
    assert "QCheckBox#mirrorCb ('Mirror preview') checked=True" in content
    assert "| ui.combo | Combo selected: index=2 text='heavy'" in content
    assert "| ui.window | Dialog opened: QWidget#TestRoot > QDialog#SampleDialog ('Export Summary')" in content
    assert "| ui.window | Dialog closed: QWidget#TestRoot > QDialog#SampleDialog ('Export Summary')" in content


def test_mainwindow_navigation_and_subtabs_logged(qapp: QApplication, tmp_path: Path) -> None:
    log_path = troubleshoot.enable_troubleshoot(
        tmp_path / "logs",
        source="qt-shell-test",
        heartbeat_interval_s=0.0,
        include_profiler=False,
    )

    window = MainWindow(Config(data_dir=str(tmp_path / "data")))
    window.show()
    qapp.processEvents()

    window.nav_buttons[Route.SETTINGS].click()
    qapp.processEvents()
    settings_screen = window.screens[Route.SETTINGS]
    settings_screen.tabs.setCurrentIndex(1)
    qapp.processEvents()

    window.nav_buttons[Route.HELP].click()
    qapp.processEvents()
    window.about_btn.click()
    qapp.processEvents()
    assert window.router.go_back()
    qapp.processEvents()

    if window.about_dialog is not None:
        window.about_dialog.close()
    window.close()
    qapp.processEvents()

    troubleshoot.shutdown_troubleshoot()
    content = log_path.read_text(encoding="utf-8")

    assert "| ui.route | Initial menu entered: 'home'" in content
    assert "('Settings') checked=True" in content
    assert "| ui.route | Menu entered: 'settings' from='home'" in content
    assert "| ui.screen | Activated screen=SettingsScreen route='settings'" in content
    assert "| ui.tab | Tab/Submenu selected: index=1" in content
    assert "| ui.route | Menu entered: 'help' from='settings'" in content
    assert "| ui.menu | Entered About dialog" in content
    assert "| ui.window | Dialog opened:" in content
    assert "| ui.route | Menu back: 'settings' from='help'" in content
