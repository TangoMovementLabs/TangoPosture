"""Qt worker adapter for the existing local practice pipeline."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
import wave
from collections import deque
from collections.abc import Callable
from pathlib import Path
from queue import SimpleQueue
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import cv2
from PySide6.QtCore import QObject, QThread, Signal

if TYPE_CHECKING:
    from PySide6.QtWidgets import QComboBox

from ...app import Application
from ...calibration.baseline import load_calibration_file
from ...config import Config
from ...domain import FrameObservation
from ...practice import GuiDisplay
from ...storage import Storage

PRACTICE_ERRORS = (OSError, ValueError, RuntimeError, ImportError, sqlite3.Error, cv2.error)


def _pad_wav_silence(src_wav: Path, dst_wav: Path, silence_s: float) -> bool:
    """Append trailing PCM silence to a WAV file so CoreAudio drains the phrase cleanly."""
    try:
        with wave.open(str(src_wav), "rb") as reader:
            params = reader.getparams()
            if (
                params.nframes <= 0
                or params.framerate <= 0
                or params.sampwidth <= 0
                or params.nchannels <= 0
            ):
                return False
            frames = reader.readframes(params.nframes)
        pad_frames = max(0, round(params.framerate * silence_s))
        silence = b"\x00" * (pad_frames * params.sampwidth * params.nchannels)
        with wave.open(str(dst_wav), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(frames)
            if silence:
                writer.writeframes(silence)
        return True
    except (wave.Error, OSError):
        return False


class SpeechManager(QObject):
    """Serialized, queued, cancellable speech runner scoped to prompt IDs."""

    speech_finished = Signal(int, bool)  # prompt_id, failed

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        timeout_s: float = 30.0,
        tail_silence_s: float = 0.40,
        post_speech_delay_s: float = 0.20,
    ) -> None:
        super().__init__(parent)
        self._timeout_s = timeout_s
        self._tail_silence_s = max(0.0, tail_silence_s)
        self._post_speech_delay_s = max(0.0, post_speech_delay_s)
        self._lock = threading.Lock()
        self._active_proc: subprocess.Popen[bytes] | None = None
        self._generation = 0
        self._speaking = False
        self._queue: deque[tuple[int, str, int]] = deque()
        self._worker_running = False
        self._gen_event = threading.Event()

    @staticmethod
    def is_available() -> bool:
        return sys.platform == "darwin" and Path("/usr/bin/say").is_file()

    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking

    def speak(self, prompt_id: int, text: str) -> None:
        if not text.strip() or not self.is_available():
            self.speech_finished.emit(prompt_id, not self.is_available())
            return
        with self._lock:
            self._queue.append((prompt_id, text, self._generation))
            self._speaking = True
            should_start = not self._worker_running
            if should_start:
                self._worker_running = True
        if should_start:
            threading.Thread(target=self._drain_queue, daemon=True, name="solo-cal-speech").start()

    def _run_proc(self, cmd: list[str], gen: int) -> tuple[int | None, bool]:
        proc: subprocess.Popen[bytes] | None = None
        try:
            with self._lock:
                if gen != self._generation:
                    return None, True
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._lock:
                if gen != self._generation:
                    try:
                        proc.terminate()
                        proc.wait(timeout=0.3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=0.3)
                    except OSError:
                        pass
                    return None, True
                self._active_proc = proc
            rc = proc.wait(timeout=self._timeout_s)
            with self._lock:
                cancelled = gen != self._generation
            return rc, cancelled
        except subprocess.TimeoutExpired:
            if proc is not None:
                try:
                    proc.kill()
                    proc.wait(timeout=0.3)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            with self._lock:
                cancelled = gen != self._generation
            return None, cancelled
        except OSError:
            with self._lock:
                cancelled = gen != self._generation
            return None, cancelled
        finally:
            with self._lock:
                if self._active_proc is proc:
                    self._active_proc = None

    def _run_utterance(self, text: str, gen: int, cancel_event: threading.Event) -> bool:
        if Path("/usr/bin/afplay").is_file() and self._tail_silence_s > 0:
            with TemporaryDirectory(prefix="tango-say-", ignore_cleanup_errors=True) as td:
                td_path = Path(td)
                raw_wav = td_path / "speech.wav"
                padded_wav = td_path / "speech_padded.wav"
                rc, cancelled = self._run_proc(
                    [
                        "/usr/bin/say",
                        "-o",
                        str(raw_wav),
                        "--file-format=WAVE",
                        "--data-format=LEI16@22050",
                        text,
                    ],
                    gen,
                )
                if cancelled:
                    return False
                if rc == 0 and _pad_wav_silence(raw_wav, padded_wav, self._tail_silence_s):
                    play_rc, play_cancelled = self._run_proc(["/usr/bin/afplay", str(padded_wav)], gen)
                    if play_cancelled:
                        return False
                    if play_rc == 0:
                        if self._post_speech_delay_s > 0:
                            cancel_event.wait(self._post_speech_delay_s)
                        return False

                raw_aiff = td_path / "speech.aiff"
                rc_aiff, cancelled_aiff = self._run_proc(
                    ["/usr/bin/say", "-o", str(raw_aiff), text],
                    gen,
                )
                if cancelled_aiff:
                    return False
                if rc_aiff == 0 and raw_aiff.is_file() and raw_aiff.stat().st_size > 4096:
                    play_rc, play_cancelled = self._run_proc(["/usr/bin/afplay", str(raw_aiff)], gen)
                    if play_cancelled:
                        return False
                    if play_rc == 0:
                        wait_s = max(self._post_speech_delay_s, self._tail_silence_s)
                        if wait_s > 0:
                            cancel_event.wait(wait_s)
                        return False

        rc_direct, cancelled_direct = self._run_proc(["/usr/bin/say", text], gen)
        if cancelled_direct:
            return False
        if rc_direct is None or rc_direct != 0:
            return True
        wait_s = max(self._post_speech_delay_s, self._tail_silence_s)
        if wait_s > 0:
            cancel_event.wait(wait_s)
        return False

    def _drain_queue(self) -> None:
        while True:
            with self._lock:
                while self._queue and self._queue[0][2] != self._generation:
                    self._queue.popleft()
                if not self._queue:
                    self._speaking = False
                    self._worker_running = False
                    return
                prompt_id, text, gen = self._queue.popleft()
                cancel_event = self._gen_event
            failed = self._run_utterance(text, gen, cancel_event)
            with self._lock:
                still_current = gen == self._generation
                while self._queue and self._queue[0][2] != self._generation:
                    self._queue.popleft()
                should_exit = not self._queue
                if should_exit:
                    if still_current:
                        self._speaking = False
                    self._worker_running = False
            if still_current:
                self.speech_finished.emit(prompt_id, failed)
            if should_exit:
                return

    def cancel_all(self) -> None:
        with self._lock:
            self._generation += 1
            self._queue.clear()
            self._speaking = False
            old_event = self._gen_event
            self._gen_event = threading.Event()
            proc = self._active_proc
            self._active_proc = None
        old_event.set()
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=0.3)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            except OSError:
                pass


def discover_cameras(max_indices: int = 5, *, configured_index: int = 0) -> list[tuple[int, str]]:
    """Probe available camera indices cleanly, always preserving configured_index as a fallback."""
    found: list[tuple[int, str]] = []
    seen: set[int] = set()
    indices = [configured_index] + [i for i in range(max_indices) if i != configured_index]
    for idx in indices:
        if idx < 0 or idx in seen:
            continue
        cap = None
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                details = f"{w}x{h}" if w > 0 and h > 0 else "connected"
                if fps > 0:
                    details += f" @ {fps:.0f} fps"
                found.append((idx, f"Camera {idx} ({details})"))
                seen.add(idx)
        except cv2.error:
            pass
        finally:
            if cap is not None:
                cap.release()
    if configured_index not in seen:
        found.insert(0, (configured_index, f"Camera {configured_index}"))
    return sorted(found, key=lambda item: item[0])


def populate_camera_combo(
    combo: QComboBox,
    discoverer: Callable[..., list[tuple[int, str]]] = discover_cameras,
    *,
    fallback_index: int = 0,
) -> list[tuple[int, str]]:
    """Discover cameras and refresh a QComboBox while preserving the selected index."""
    current_data = combo.currentData()
    current = fallback_index if current_data is None else int(current_data)
    discovered = discoverer(configured_index=current)
    combo.blockSignals(True)
    try:
        combo.clear()
        for idx, label in discovered:
            combo.addItem(label, idx)
        pos = combo.findData(current)
        if pos >= 0:
            combo.setCurrentIndex(pos)
    finally:
        combo.blockSignals(False)
    return discovered


class QtPracticeDisplay(GuiDisplay):
    """Thread-safe mailbox and explicit calibration handoff for Qt."""

    requires_reference_acceptance = True

    def __init__(self, *, capture_evidence: bool = False) -> None:
        super().__init__()
        self.calibration_candidates: SimpleQueue[str] = SimpleQueue()
        self.calibration_archives: SimpleQueue[Path] = SimpleQueue()
        self.capture_evidence = capture_evidence
        self.observations: SimpleQueue[FrameObservation] = SimpleQueue()
        self.prompts: SimpleQueue[str] = SimpleQueue()
        self.stop = False
        self.show_head_alignment: bool = True
        self.show_shoulders_height: bool = True
        self.show_role_detection: bool = True
        self.show_embrace_type: bool = True
        self.suppress_alerts: bool = False
        self.disable_settings_coaching: bool = False

    def observe(self, frame: FrameObservation) -> None:
        if self.capture_evidence:
            self.observations.put(frame)

    def candidate_ready(self, name: str) -> None:
        self.calibration_candidates.put(name)

    def calibration_saved(self, path: Path) -> None:
        self.calibration_archives.put(path.resolve())


class QtPracticeWorker(QObject):
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        cfg: Config,
        display: QtPracticeDisplay,
        *,
        reference_id: str | None = None,
        reference_path: Path | None = None,
        video: Path | None = None,
        practice_name: str | None = None,
        calibration_only: bool = False,
        reference_name: str = "My reference",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.display = display
        self.reference_id = reference_id
        self.reference_path = reference_path
        self.video = video
        self.practice_name = practice_name
        self.calibration_only = calibration_only
        self.reference_name = reference_name

    def run(self) -> None:
        storage: Storage | None = None
        try:
            storage = Storage(Path(self.cfg.data_dir))
            reference = None
            cfg = self.cfg
            if self.reference_id:
                try:
                    reference = (
                        load_calibration_file(str(self.reference_path), self.reference_path.parent)
                        if self.reference_path is not None and self.reference_path.is_file()
                        else storage.calibration(self.reference_id)
                    )
                except ValueError:
                    if self.reference_path is None:
                        raise
                    reference = load_calibration_file(
                        str(self.reference_path), self.reference_path.parent
                    )
                    if reference is None:
                        raise ValueError(f"Reference not found: {self.reference_path.name}")
                if reference is None:
                    raise ValueError("Selected calibration could not be loaded")
                cfg = self._config_for_reference(cfg, reference.profile)

            Application(
                cfg,
                storage,
                str(self.video) if self.video is not None else None,
                reference=reference,
                display=self.display,
                practice_name=self.practice_name,
                calibration_only=self.calibration_only,
                name=self.reference_name,
            ).run()
        except PRACTICE_ERRORS as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            if storage is not None:
                storage.close()
            self.finished.emit()

    @staticmethod
    def _config_for_reference(cfg: Config, profile: dict[str, object]) -> Config:
        updates: dict[str, object] = {
            "side": profile["side"],
            "facing": profile["facing"],
            "width": profile["width"],
            "height": profile["height"],
            "capture_profile": profile["capture_profile"],
        }
        source = profile["source"]
        if isinstance(source, str) and source.startswith("camera:"):
            camera_index = source.partition(":")[2]
            try:
                updates["camera"] = int(camera_index)
            except ValueError as error:
                raise ValueError("Reference contains an invalid camera source") from error
        return Config.model_validate({**cfg.model_dump(), **updates})


class PracticeThread(QThread):
    def __init__(self, worker: QtPracticeWorker, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.worker = worker

    def run(self) -> None:
        self.worker.run()
