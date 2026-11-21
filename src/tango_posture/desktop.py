"""Entry point for the packaged desktop application."""

import os
import sys
from importlib.resources import files
from pathlib import Path


def data_directory() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "TangoPosture"
    if sys.platform == "win32":
        return Path(os.environ["LOCALAPPDATA"]) / "TangoPosture"
    if sys.platform.startswith("linux"):
        xdg_data = os.environ.get("XDG_DATA_HOME", "").strip()
        base = Path(xdg_data) if xdg_data else (Path.home() / ".local" / "share")
        return base / "TangoPosture"
    raise RuntimeError(f"Unsupported desktop platform: {sys.platform}")


def bundled_model() -> Path:
    return Path(str(files("tango_posture").joinpath("pose_landmarker_lite.task")))


def smoke_test() -> None:
    print("Importing packaged dependencies", flush=True)
    from importlib.metadata import version

    import cv2
    import numpy as np
    import sounddevice
    from PySide6.QtWidgets import QApplication

    from .config import Config
    from .desktop_qt.shell import MainWindow
    from .models import verify
    from .pose.mediapipe_backend import MediaPipeEstimator

    print("Checking Qt desktop", flush=True)
    app = QApplication.instance()
    if app is None:
        app = QApplication(["TangoPosture", "-platform", "offscreen"])
    elif not isinstance(app, QApplication):
        raise RuntimeError("A non-GUI Qt application is already running")
    window = MainWindow(Config())
    try:
        app.processEvents()
    finally:
        window.close()
    version("tango-posture")
    print("Verifying packaged resources and pose model", flush=True)
    for name in ("TangoPosture_logo_under_1MB.png", "TANGO_POSTURE_MANUAL.md"):
        if not files("tango_posture").joinpath(name).read_bytes():
            raise ValueError(f"Empty packaged resource: {name}")
    model = bundled_model()
    verify(model)
    print("Creating MediaPipe estimator", flush=True)
    estimator = MediaPipeEstimator(model)
    try:
        print("Running blank-frame inference", flush=True)
        estimator.estimate(np.zeros((64, 64, 3), dtype=np.uint8), 0.0)
    finally:
        print("Closing MediaPipe estimator", flush=True)
        estimator.close()
    print(f"Desktop smoke test passed (OpenCV {cv2.__version__}, sounddevice {sounddevice.__version__})")


def _parse_desktop_args(argv: list[str]) -> tuple[bool, Path | None]:
    """Accept only troubleshoot flags in packaged desktop mode; reject unexpected arguments."""
    troubleshoot = os.environ.get("TANGOPOSTURE_TROUBLESHOOT", "").strip() == "1"
    log_dir: Path | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--troubleshoot", "troubleshoot"):
            troubleshoot = True
            i += 1
        elif arg == "--log-dir":
            if i + 1 >= len(argv):
                raise ValueError("The desktop application does not accept command-line arguments")
            log_dir = Path(argv[i + 1])
            i += 2
        elif arg.startswith("--log-dir="):
            log_dir = Path(arg.split("=", 1)[1])
            i += 1
        else:
            raise ValueError("The desktop application does not accept command-line arguments")
    if log_dir is not None and not troubleshoot:
        raise ValueError("The desktop application does not accept command-line arguments")
    return troubleshoot, log_dir


def main() -> int:
    if sys.argv[1:] == ["--smoke-test"]:
        smoke_test()
        return 0
    troubleshoot, log_dir = _parse_desktop_args(sys.argv[1:])

    from .config import load, startup_config_path
    from .desktop_qt import launch_qt
    from .troubleshoot import enable_troubleshoot, log_config

    directory = data_directory()
    directory.mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    if troubleshoot:
        log_path = enable_troubleshoot(log_dir=log_dir or (directory / "logs"), source="desktop")
        print(f"Troubleshoot logging enabled: {log_path}", file=sys.stderr, flush=True)
    cfg = load(startup_config_path(None), {"model_path": str(bundled_model())})
    if troubleshoot:
        log_config(cfg, source="desktop")
    return launch_qt(cfg)
