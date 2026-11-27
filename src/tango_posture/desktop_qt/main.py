"""Application launcher and entry point for the PySide6 Qt desktop interface."""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib.resources import files as package_files
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication

from ..config import Config
from ..troubleshoot import current_log_path, is_troubleshoot_enabled, log_qt_environment
from .practice.worker import CameraScanThread, discover_cameras
from .screens.splash import SplashScreen
from .shell import MainWindow

SPLASH_DURATION_MS = 3000


def _load_logo() -> QPixmap:
    data: bytes | None
    try:
        resource = package_files("tango_posture").joinpath("TangoPosture_logo_under_1MB.png")
        data = resource.read_bytes()
    except (OSError, ValueError, FileNotFoundError):
        path = Path(__file__).resolve().parents[3] / "assets" / "TangoPosture_logo_under_1MB.png"
        data = path.read_bytes() if path.is_file() else None

    pixmap = QPixmap()
    if data:
        pixmap.loadFromData(data)
    return pixmap


def _format_camera_scan_summary(cameras: list[tuple[int, str]]) -> str:
    if not cameras:
        return "No cameras detected. Starting TangoPosture..."
    if len(cameras) == 1:
        return f"Camera ready: {cameras[0][1]}. Starting TangoPosture..."
    return f"Found {len(cameras)} cameras. Starting TangoPosture..."


def show_startup_splash(
    window: MainWindow,
    logo: QPixmap,
    *,
    camera_discoverer: Callable[..., list[tuple[int, str]]] = discover_cameras,
    splash: SplashScreen | None = None,
    scan_thread: CameraScanThread | None = None,
) -> SplashScreen:
    if splash is None:
        splash = SplashScreen(logo, status_message="Scanning cameras...")
        splash.show()

    delay_done = False
    scan_done = False
    finished = False
    active_scan = scan_thread

    def maybe_finish() -> None:
        nonlocal finished
        if finished or not (delay_done and scan_done):
            return
        finished = True
        window.show()
        window.raise_()
        splash.close()

    def on_delay_done() -> None:
        nonlocal delay_done
        delay_done = True
        maybe_finish()

    def on_scan_done() -> None:
        nonlocal scan_done
        if scan_done or active_scan is None:
            return
        discovered = list(active_scan.discovered_cameras)
        window.bridge.set_discovered_cameras(discovered)
        splash.set_status(_format_camera_scan_summary(discovered))
        scan_done = True
        active_scan.deleteLater()
        maybe_finish()

    if active_scan is None:
        active_scan = CameraScanThread(
            configured_index=window.bridge.config.camera,
            discoverer=camera_discoverer,
            parent=window,
        )
        active_scan.progress.connect(splash.set_status)
        active_scan.finished.connect(on_scan_done)
        splash._scan_thread = active_scan
        active_scan.start()
    else:
        splash._scan_thread = active_scan
        active_scan.finished.connect(on_scan_done)
        if active_scan.isFinished():
            on_scan_done()

    QTimer.singleShot(SPLASH_DURATION_MS, on_delay_done)
    return splash


def launch_qt(cfg: Config) -> int:
    """Launch the modern PySide6 Qt desktop interface."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    elif not isinstance(app, QApplication):
        raise RuntimeError("A non-GUI Qt application is already running")

    app.setApplicationName("TangoPosture Studio")
    app.setOrganizationName("TangoPosture")
    if is_troubleshoot_enabled():
        log_qt_environment(app)

    logo = _load_logo()
    if not logo.isNull():
        pixmap = logo
        app.setWindowIcon(QIcon(pixmap))

    splash = SplashScreen(logo, status_message="Scanning cameras...")
    splash.show()
    app.processEvents()

    scan_thread = CameraScanThread(
        configured_index=cfg.camera,
        discoverer=discover_cameras,
        parent=app,
    )
    scan_thread.progress.connect(splash.set_status)
    scan_thread.start()

    window = MainWindow(cfg)
    log_path = current_log_path()
    if log_path is not None:
        window.bridge.status_message.emit(f"Troubleshoot log: {log_path}")
    show_startup_splash(window, logo, splash=splash, scan_thread=scan_thread)
    return app.exec()
