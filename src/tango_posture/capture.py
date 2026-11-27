"""Sequential file decode and a latest-frame-only webcam producer."""

import ctypes
import ctypes.util
import logging
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .config import Config

_logger = logging.getLogger(__name__)


def _mac_camera_status() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc") or "libobjc.dylib")
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        msg_str = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
        )(("objc_msgSend", objc))
        media = msg_str(
            objc.objc_getClass(b"NSString"),
            objc.sel_registerName(b"stringWithUTF8String:"),
            b"vide",
        )
        msg_status = ctypes.CFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )(("objc_msgSend", objc))
        return int(
            msg_status(
                objc.objc_getClass(b"AVCaptureDevice"),
                objc.sel_registerName(b"authorizationStatusForMediaType:"),
                media,
            )
        )
    except (OSError, AttributeError):
        return None


@dataclass
class Captured:
    image: Any
    timestamp: float
    captured_at: float


class Capture:
    def __init__(self, cfg: Config, video: str | None):
        if (
            video is not None
            and not Path(video).is_file()
            and (Path("videos") / Path(video).name).is_file()
        ):
            video = str(Path("videos") / Path(video).name)
        self.video = video
        source = video if video is not None else cfg.camera
        _logger.debug("Opening capture source=%r (requested=%dx%d)", source, cfg.width, cfg.height)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened() and video is None:
            tcc_status = _mac_camera_status()
            _logger.warning(
                "Initial webcam open failed (camera=%s tcc_status=%s); polling up to 30s",
                cfg.camera,
                tcc_status,
            )
            deadline = time.monotonic() + 30
            while _mac_camera_status() == 0 and time.monotonic() < deadline:
                time.sleep(0.25)
            final_tcc = _mac_camera_status()
            _logger.debug("Post-wait macOS camera TCC status=%s", final_tcc)
            if final_tcc == 3:
                self.cap.release()
                self.cap = cv2.VideoCapture(cfg.camera)
        if not self.cap.isOpened():
            _logger.error("Capture open failed source=%r", source)
            self.cap.release()
            if video:
                if not Path(video).is_file():
                    raise ValueError(
                        f"Cannot read video file: '{video}' does not exist "
                        f"(record one first with --camera 0 --record-video videos/{Path(video).name})"
                    )
                raise ValueError(f"Cannot read video file: '{video}'")
            raise ValueError("Camera unavailable: check device index and OS camera permissions")
        if not video:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        _logger.info(
            "Capture opened source=%r fps=%.3f frame_count=%.0f width=%.0f height=%.0f",
            source,
            self.fps,
            self.frame_count,
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        self.index = 0
        self.last = -1.0
        self.fallback = False
        self.fallback_offset: float | None = None
        self.dropped = 0
        self.origin = time.monotonic()
        self.queue: queue.Queue[Captured] = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.failed = threading.Event()
        self.thread: threading.Thread | None = None
        if not video:
            self.thread = threading.Thread(target=self._produce, daemon=True)
            self.thread.start()

    def _produce(self) -> None:
        _logger.debug("Webcam producer thread started")
        while not self.stop.is_set():
            ok, image = self.cap.read()
            now = time.monotonic()
            if not ok:
                _logger.error("Webcam producer cap.read() returned ok=False")
                self.failed.set()
                break
            frame = Captured(image, now - self.origin, now)
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                    if self.dropped == 1 or self.dropped % 25 == 0:
                        _logger.debug("Webcam producer dropped stale frame count=%d", self.dropped)
                except queue.Empty:
                    pass
                self.queue.put_nowait(frame)

    def read(self) -> Captured | None:
        if not self.video:
            if self.failed.is_set():
                raise OSError("Camera disconnected")
            try:
                frame = self.queue.get(timeout=0.03)
                if time.monotonic() - frame.captured_at > 0.5:
                    _logger.error("Camera stalled: frame age %.3fs > 0.5s", time.monotonic() - frame.captured_at)
                    raise OSError("Camera stalled: no fresh frame")
                return frame
            except queue.Empty:
                if time.monotonic() - self.origin - max(self.last, 0) > 3:
                    _logger.error("Camera stalled or permissions denied after 3s")
                    raise OSError("Camera stalled or permissions denied")
                return None
        ok, image = self.cap.read()
        if not ok:
            if self.frame_count > 0 and self.index < self.frame_count - 1:
                _logger.error(
                    "Video decode EOF early index=%d frame_count=%.0f",
                    self.index,
                    self.frame_count,
                )
                raise OSError("Video decoding stopped before declared EOF")
            return None
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if self.fallback or not math.isfinite(timestamp) or timestamp <= self.last:
            if not math.isfinite(self.fps) or self.fps <= 0:
                raise ValueError(
                    "Video has neither reliable presentation timestamps nor declared FPS"
                )
            if self.fallback_offset is None:
                self.fallback_offset = max(0, self.last + 1 / self.fps - self.index / self.fps)
                _logger.warning(
                    "Activating video timestamp fallback at index=%d raw_ts=%.4f last=%.4f offset=%.4f fps=%.3f",
                    self.index,
                    timestamp,
                    self.last,
                    self.fallback_offset,
                    self.fps,
                )
            self.fallback = True
            timestamp = self.index / self.fps + self.fallback_offset
        self.last = timestamp
        self.index += 1
        return Captured(image, timestamp, time.monotonic())

    def close(self) -> None:
        _logger.debug(
            "Closing capture source=%r frames=%d dropped=%d fallback=%s",
            self.video if self.video is not None else "webcam",
            self.index,
            self.dropped,
            self.fallback,
        )
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
        self.cap.release()
        if self.thread:
            self.thread.join(timeout=1)
