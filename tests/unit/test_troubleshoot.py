"""Unit and integration tests for low-level --troubleshoot telemetry logging."""

from __future__ import annotations

import builtins
import logging
import sys
import threading
import warnings
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
import pytest
import sounddevice as sd

from tango_posture import cli, desktop, desktop_qt, troubleshoot
from tango_posture.storage import Storage


@pytest.fixture(autouse=True)
def _ensure_troubleshoot_shutdown():
    """Ensure every test leaves troubleshooting disabled and hooks restored."""
    yield
    troubleshoot.shutdown_troubleshoot()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_enable_troubleshoot_creates_unique_log_with_specs_packages_and_hooks(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"

    path1 = troubleshoot.enable_troubleshoot(
        log_dir,
        source="pytest-run-1",
        heartbeat_interval_s=0.0,
        include_profiler=False,
    )
    assert path1.parent == log_dir
    assert path1.is_file()
    assert path1.name.startswith("troubleshoot_")
    assert path1.suffix == ".log"
    assert troubleshoot.is_troubleshoot_enabled()
    assert troubleshoot.current_log_path() == path1

    # Generate debug log, warning, dynamic module import, and thread exception.
    logging.getLogger("tango_posture.test_probe").debug("probe-debug-payload=%d", 42)
    warnings.warn("probe-warning-message", UserWarning, stacklevel=1)

    sys.modules.pop("colorsys", None)
    builtins.__import__("colorsys")

    def _boom() -> None:
        raise RuntimeError("probe-thread-failure")

    worker = threading.Thread(target=_boom, name="probe-worker")
    worker.start()
    worker.join(timeout=2)

    troubleshoot.log_event("custom.probe", "custom-probe-line")
    troubleshoot.shutdown_troubleshoot()

    content = path1.read_text(encoding="utf-8")
    assert "TangoPosture Troubleshoot Session Started" in content
    assert "| system.specs |" in content
    assert "| python.packages |" in content
    assert "pytest==" in content
    assert "numpy==" in content
    assert "opencv-python==" in content
    assert "mediapipe==" in content
    assert "py.tango_posture.test_probe:" in content
    assert "probe-debug-payload=42" in content
    assert "probe-warning-message" in content
    assert "| python.import | loaded module=colorsys" in content
    assert "probe-thread-failure" in content
    assert "| INFO     | " in content
    assert "| custom.probe | custom-probe-line" in content
    assert "Session closing | total_duration_s=" in content

    # Second run creates a second unique log file.
    path2 = troubleshoot.enable_troubleshoot(
        log_dir,
        source="pytest-run-2",
        heartbeat_interval_s=0.0,
        include_profiler=False,
    )
    troubleshoot.shutdown_troubleshoot()
    assert path2 != path1
    assert path2.is_file()
    assert len(list(log_dir.glob("troubleshoot_*.log"))) == 2


def test_shutdown_restores_global_hooks(tmp_path: Path) -> None:
    orig_import = builtins.__import__
    orig_excepthook = sys.excepthook
    orig_threading_hook = threading.excepthook
    orig_vc = cv2.VideoCapture
    orig_vw = cv2.VideoWriter
    orig_mp_create = mp.tasks.vision.PoseLandmarker.create_from_options
    orig_sd_stream = sd.InputStream

    troubleshoot.enable_troubleshoot(
        tmp_path / "logs",
        source="restore-test",
        heartbeat_interval_s=0.0,
        include_profiler=False,
    )
    assert cv2.VideoCapture is not orig_vc
    assert mp.tasks.vision.PoseLandmarker.create_from_options is not orig_mp_create

    troubleshoot.shutdown_troubleshoot()

    assert not troubleshoot.is_troubleshoot_enabled()
    assert troubleshoot.current_log_path() is None
    assert builtins.__import__ is orig_import
    assert sys.excepthook is orig_excepthook
    assert threading.excepthook is orig_threading_hook
    assert cv2.VideoCapture is orig_vc
    assert cv2.VideoWriter is orig_vw
    assert mp.tasks.vision.PoseLandmarker.create_from_options is orig_mp_create
    assert sd.InputStream is orig_sd_stream


def test_opencv_capture_and_writer_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame_img = np.full((8, 12, 3), 120, dtype=np.uint8)

    class _FakeCap:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.props: dict[int, float] = {
                cv2.CAP_PROP_FRAME_WIDTH: 12.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 8.0,
                cv2.CAP_PROP_FPS: 30.0,
            }
            self.reads = 0

        def isOpened(self) -> bool:
            return True

        def open(self, *_args: object, **_kwargs: object) -> bool:
            return True

        def set(self, prop_id: int, value: float) -> bool:
            self.props[prop_id] = value
            return True

        def get(self, prop_id: int) -> float:
            return self.props.get(prop_id, 0.0)

        def read(self) -> tuple[bool, np.ndarray | None]:
            self.reads += 1
            if self.reads <= 2:
                return True, frame_img
            return False, None

        def grab(self) -> bool:
            return True

        def retrieve(self) -> tuple[bool, np.ndarray]:
            return True, frame_img

        def release(self) -> None:
            pass

    class _FakeWriter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.frames = 0

        def isOpened(self) -> bool:
            return True

        def write(self, _img: object) -> None:
            self.frames += 1

        def release(self) -> None:
            pass

    monkeypatch.setattr(cv2, "VideoCapture", _FakeCap)
    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)

    log_path = troubleshoot.enable_troubleshoot(
        tmp_path / "logs",
        source="opencv-test",
        heartbeat_interval_s=0.0,
        include_profiler=False,
    )

    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
    assert cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280
    ok1, img1 = cap.read()
    assert ok1 and img1 is not None
    ok2, _img2 = cap.read()
    assert ok2
    ok3, _img3 = cap.read()
    assert not ok3
    cap.release()

    writer = cv2.VideoWriter(str(tmp_path / "out.mp4"), 0, 30.0, (12, 8))
    assert writer.isOpened()
    writer.write(frame_img)
    writer.release()

    troubleshoot.shutdown_troubleshoot()

    content = log_path.read_text(encoding="utf-8")
    assert "| cv2.build_info |" in content
    assert "| camera.cv2.cap#1 | VideoCapture.open source=0" in content
    assert "CAP_PROP_FRAME_WIDTH" in content
    assert "read seq=1 ok=True" in content
    assert "sample_mean=" in content
    assert "read seq=3 ok=False" in content
    assert "release source=0" in content
    assert "frames_ok=2 frames_fail=1" in content
    assert "| video.cv2.writer#1 | VideoWriter.open" in content
    assert "write frame=1" in content
    assert "release file=" in content
    assert "frames=1" in content


def test_mediapipe_pose_landmarker_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_file = tmp_path / "fake.task"
    model_file.write_bytes(b"mediapipe-model-bytes")

    lm = SimpleNamespace(x=0.5, y=0.25, z=-0.1, visibility=0.95)
    fake_pose = [lm] * 33
    fake_result = SimpleNamespace(pose_landmarks=[fake_pose], pose_world_landmarks=[fake_pose])

    class _FakeLandmarker:
        def detect_for_video(self, _img: object, _ts_ms: int) -> object:
            return fake_result

        def detect(self, _img: object) -> object:
            return fake_result

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        mp.tasks.vision.PoseLandmarker,
        "create_from_options",
        staticmethod(lambda _opts: _FakeLandmarker()),
    )

    log_path = troubleshoot.enable_troubleshoot(
        tmp_path / "logs",
        source="mediapipe-test",
        heartbeat_interval_s=0.0,
        include_profiler=False,
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_file)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=2,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    res1 = landmarker.detect_for_video(object(), 33)
    res2 = landmarker.detect(object())
    assert res1 is fake_result
    assert res2 is fake_result
    landmarker.close()

    troubleshoot.shutdown_troubleshoot()

    content = log_path.read_text(encoding="utf-8")
    assert "| mediapipe.model#1 | PoseLandmarker.create_from_options" in content
    assert f"model={str(model_file)!r}" in content
    assert "size_bytes=21" in content
    assert "detect_for_video seq=1 ts_ms=33" in content
    assert "poses=1 world_poses=1" in content
    assert "nose=(0.500,0.250,v=0.95)" in content
    assert "calls=2 poses_total=2 empty_calls=0" in content


def test_sounddevice_input_stream_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeStream:
        def __init__(self, **kwargs: object) -> None:
            self.callback = kwargs.get("callback")

        def start(self) -> None:
            if callable(self.callback):
                self.callback(np.zeros((16, 1), dtype=np.float32), 16, None, "input-overflow")

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(sd, "InputStream", _FakeStream)

    log_path = troubleshoot.enable_troubleshoot(
        tmp_path / "logs",
        source="sounddevice-test",
        heartbeat_interval_s=0.0,
        include_profiler=False,
    )

    stream = sd.InputStream(samplerate=44100, channels=1, blocksize=1024, callback=lambda *_: None)
    stream.start()
    stream.stop()
    stream.close()

    troubleshoot.shutdown_troubleshoot()

    content = log_path.read_text(encoding="utf-8")
    assert "| audio.sd.init | sounddevice version=" in content
    assert "| audio.sd.stream#1 | InputStream.open kwargs={'samplerate': 44100" in content
    assert "callback status='input-overflow'" in content
    assert "InputStream.close() callbacks=1 status_errors=1" in content


def test_cli_troubleshoot_flag_and_normalize_argv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli._normalize_argv(["--troubleshoot"]) == ["--troubleshoot", "gui"]
    assert cli._normalize_argv(["--troubleshoot", "--log-dir", "mylogs"]) == [
        "--troubleshoot",
        "--log-dir",
        "mylogs",
        "gui",
    ]
    assert cli._normalize_argv(["sessions", "list", "--troubleshoot"]) == [
        "sessions",
        "list",
        "--troubleshoot",
    ]

    log_dir = tmp_path / "cli_logs"
    data_dir = tmp_path / "data"
    Storage(data_dir).close()

    cli.main(
        [
            "--troubleshoot",
            "--log-dir",
            str(log_dir),
            "--data-dir",
            str(data_dir),
            "sessions",
            "list",
        ]
    )
    err = capsys.readouterr().err
    assert "Troubleshoot logging enabled:" in err

    troubleshoot.shutdown_troubleshoot()
    files = list(log_dir.glob("troubleshoot_*.log"))
    assert len(files) == 1
    log_text = files[0].read_text(encoding="utf-8")
    assert "| config.cli:sessions | Effective configuration:" in log_text


def test_desktop_troubleshoot_flag_and_subcommand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    launched: list[object] = []
    monkeypatch.chdir(Path.cwd())
    monkeypatch.setattr(desktop, "data_directory", lambda: tmp_path / "app_data")
    monkeypatch.setattr(desktop, "bundled_model", lambda: tmp_path / "lite.task")
    monkeypatch.setattr(desktop_qt, "launch_qt", lambda cfg: launched.append(cfg))
    monkeypatch.setattr("tango_posture.desktop_qt.main.launch_qt", lambda cfg: launched.append(cfg))

    log_dir = tmp_path / "desktop_logs"
    monkeypatch.setattr(
        sys, "argv", ["TangoPosture", "troubleshoot", "--log-dir", str(log_dir)]
    )
    desktop.main()

    err = capsys.readouterr().err
    assert "Troubleshoot logging enabled:" in err
    assert len(launched) == 1

    troubleshoot.shutdown_troubleshoot()
    files = list(log_dir.glob("troubleshoot_*.log"))
    assert len(files) == 1
    log_text = files[0].read_text(encoding="utf-8")
    assert "| config.desktop | Effective configuration:" in log_text

    # Unexpected flags remain strictly rejected.
    monkeypatch.setattr(sys, "argv", ["TangoPosture", "--troubleshoot", "--unexpected"])
    with pytest.raises(ValueError, match="does not accept command-line arguments"):
        desktop.main()
