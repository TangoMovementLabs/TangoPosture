"""Solo camera self-calibration wizard with readiness-driven capture and staging."""

from __future__ import annotations

import queue
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...archive import calibration_directory, calibration_path
from ...calibration.archives import save_range_profile
from ...calibration.camera_session import (
    CHECK_SPECS,
    SoloCameraSession,
    build_check_phases,
    build_quick_setup_phases,
    build_walking_phases,
    check_full_body_framing,
    evaluate_metric_repeat_check,
)
from ...calibration.evidence import cached_evidence
from ...calibration.measurements import measure_ranges
from ...calibration.models import METRIC_DETAILS, ApprovedRange, Passage, RangeProfile
from ...calibration.statistics import summarize_examples
from ...config import Config
from ...domain import FrameObservation
from ...references.archives import promote_reference, reference_directory, reference_path
from ..components.video_stage import VideoStage
from ..practice.worker import (
    PracticeThread,
    QtPracticeDisplay,
    QtPracticeWorker,
    SpeechManager,
    discover_cameras,
)
from ..service_bridge import ServiceBridge, numpy_to_qimage
from .editor import CalibrationEditor


class GuidedCalibration(QDialog):
    """Guided solo camera calibration wizard with readiness gates and atomic profile saving."""

    def __init__(
        self,
        bridge: ServiceBridge,
        parent: QWidget | None = None,
        *,
        profile_name: str = "Studio webcam",
        target_profile: RangeProfile | None = None,
        target_profile_path: Path | None = None,
        camera_discoverer: Callable[..., list[tuple[int, str]]] = discover_cameras,
        embedded: bool = False,
    ) -> None:
        super().__init__(parent)
        self.bridge = bridge
        self._cfg = bridge.config.model_copy(deep=True)
        self._target_profile = target_profile
        self._target_profile_path = target_profile_path
        self._camera_discoverer = camera_discoverer
        self._embedded = embedded
        self._detected_side = self._cfg.side
        self._detected_facing = self._cfg.facing
        self._last_spoken_hint = ""
        self.setWindowTitle(
            f"Check with camera · {target_profile.name}"
            if target_profile is not None
            else "Solo camera calibration"
        )
        self.resize(980, 720)

        self.saved_path: Path | None = None
        self.saved_paths: list[Path] = []
        self.saved_reference_path: Path | None = None

        self._temp_dir: TemporaryDirectory[str] | None = None
        self._staging_dir: Path | None = None
        self._staged_tpr: Path | None = None
        self._published_tpr: Path | None = None

        self._display: QtPracticeDisplay | None = None
        self._worker: QtPracticeWorker | None = None
        self._thread: PracticeThread | None = None
        self._camera_active = False
        self._paused = False
        self._frozen_cfg: Config | None = None
        self._collected_frames: list[FrameObservation] = []
        self._camera_meta: dict[str, Any] = {}

        self._speech = SpeechManager(self)
        self._speech.speech_finished.connect(self._on_speech_finished)
        self._session = SoloCameraSession(
            self._cfg,
            now_monotonic=time.monotonic(),
            auto_detect_side_facing=True,
        )

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(40)
        self._poll_timer.timeout.connect(self._poll_worker)

        self._setup_ui(profile_name)
        self._apply_target_profile_defaults()

    def _setup_ui(self, profile_name: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        self.status_banner = QLabel(
            "Camera is off. Click Start camera preview to check framing before recording."
        )
        self.status_banner.setWordWrap(True)
        root.addWidget(self.status_banner)

        self.preview_stage = VideoStage("Camera preview is off. Start preview to verify your setup.")
        self.preview_stage.setMinimumHeight(320)
        root.addWidget(self.preview_stage, 1)

        self.orientation_label = QLabel("")
        root.addWidget(self.orientation_label)

        self.stack = QStackedWidget()
        root.addWidget(self.stack)

        # Page 0: Camera Setup
        setup_page = QWidget()
        setup_layout = QVBoxLayout(setup_page)
        form = QFormLayout()

        self.name_input = QLineEdit(profile_name)
        form.addRow("Profile name:", self.name_input)

        cam_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.addItem(f"Camera {self._cfg.camera}", self._cfg.camera)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        cam_row.addWidget(self.camera_combo, 1)
        self.scan_cameras_btn = QPushButton("Scan cameras")
        self.scan_cameras_btn.clicked.connect(self._scan_cameras)
        cam_row.addWidget(self.scan_cameras_btn)
        self.start_preview_btn = QPushButton("Start camera preview")
        self.start_preview_btn.clicked.connect(self._start_preview)
        self.preview = self.start_preview_btn
        cam_row.addWidget(self.start_preview_btn)
        self.retry_camera_btn = QPushButton("Retry camera")
        self.retry_camera_btn.setVisible(False)
        self.retry_camera_btn.clicked.connect(self._start_preview)
        cam_row.addWidget(self.retry_camera_btn)
        form.addRow("Camera:", cam_row)

        toggles_row = QHBoxLayout()
        self.mirror_cb = QCheckBox("Mirror preview (display only)")
        self.mirror_cb.setChecked(self._cfg.mirror)
        self.mirror_cb.toggled.connect(self._update_orientation_indicator)
        toggles_row.addWidget(self.mirror_cb)

        self.speak_cb = QCheckBox("Speak instructions")
        self.speak_cb.setChecked(self._cfg.solo_camera.speak_instructions)
        toggles_row.addWidget(self.speak_cb)
        toggles_row.addStretch(1)
        form.addRow("Options:", toggles_row)

        setup_layout.addLayout(form)

        setup_note = QLabel(
            "Place the camera level at practice height and step back until your head and shoes fit. "
            "The app will guide you by voice and detect your side/facing automatically. "
            "Begin records local video and pose landmarks for review; microphone and beat tracking stay off."
        )
        setup_note.setWordWrap(True)
        setup_layout.addWidget(setup_note)

        setup_btns = QHBoxLayout()
        self.begin_btn = QPushButton("Begin")
        self.begin_btn.setProperty("class", "accent")
        self.begin_btn.setEnabled(False)
        self.begin_btn.clicked.connect(self._begin_capture)
        self.start = self.begin_btn
        setup_btns.addWidget(self.begin_btn)
        setup_cancel = QPushButton("Cancel")
        setup_cancel.clicked.connect(self.reject)
        setup_cancel.setVisible(not self._embedded)
        setup_btns.addWidget(setup_cancel)
        setup_btns.addStretch(1)
        setup_layout.addLayout(setup_btns)
        self.stack.addWidget(setup_page)

        # Page 1: Guided Capture
        capture_page = QWidget()
        cap_layout = QVBoxLayout(capture_page)
        self.phase_label = QLabel("Ready")
        self.phase_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        cap_layout.addWidget(self.phase_label)

        self.instruction_label = QLabel("")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("font-size: 15px;")
        cap_layout.addWidget(self.instruction_label)

        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        cap_layout.addWidget(self.hint_label)

        cap_btns = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        cap_btns.addWidget(self.pause_btn)

        self.repeat_btn = QPushButton("Repeat instruction")
        self.repeat_btn.clicked.connect(self._repeat_instruction)
        cap_btns.addWidget(self.repeat_btn)

        self.retry_hold_btn = QPushButton("Retry hold")
        self.retry_hold_btn.clicked.connect(self._retry_hold)
        cap_btns.addWidget(self.retry_hold_btn)

        self.skip_view_btn = QPushButton("Skip view")
        self.skip_view_btn.clicked.connect(self._skip_view)
        cap_btns.addWidget(self.skip_view_btn)

        self.start_take_btn = QPushButton("Start take")
        self.start_take_btn.setVisible(False)
        self.start_take_btn.clicked.connect(self._start_walking_take)
        cap_btns.addWidget(self.start_take_btn)

        self.end_take_btn = QPushButton("End take")
        self.end_take_btn.setVisible(False)
        self.end_take_btn.clicked.connect(self._end_walking_take)
        cap_btns.addWidget(self.end_take_btn)

        self.walk_returned_btn = QPushButton("Ready for pass 2")
        self.walk_returned_btn.setVisible(False)
        self.walk_returned_btn.clicked.connect(self._confirm_walk_return)
        cap_btns.addWidget(self.walk_returned_btn)

        cap_cancel = QPushButton("Cancel")
        cap_cancel.clicked.connect(self.reject)
        cap_btns.addWidget(cap_cancel)
        cap_btns.addStretch(1)
        cap_layout.addLayout(cap_btns)
        self.stack.addWidget(capture_page)

        # Page 2: Checks, Walking & Review Summary
        review_page = QWidget()
        rev_layout = QVBoxLayout(review_page)
        self.review_summary = QLabel("")
        self.review_summary.setWordWrap(True)
        rev_layout.addWidget(self.review_summary)

        checks_box = QGroupBox("Optional individual measurement checks & solo walking")
        checks_layout = QVBoxLayout(checks_box)
        self.check_boxes: dict[str, QCheckBox] = {}
        cb_row = QHBoxLayout()
        for cid, spec in CHECK_SPECS.items():
            cb = QCheckBox(spec.title)
            self.check_boxes[cid] = cb
            cb_row.addWidget(cb)
        checks_layout.addLayout(cb_row)

        extra_btns = QHBoxLayout()
        self.run_checks_btn = QPushButton("Check individual measurements")
        self.run_checks_btn.clicked.connect(self._run_selected_checks)
        extra_btns.addWidget(self.run_checks_btn)

        self.record_walking_btn = QPushButton("Record solo walking")
        self.record_walking_btn.clicked.connect(self._run_walking_plan)
        extra_btns.addWidget(self.record_walking_btn)
        extra_btns.addStretch(1)
        checks_layout.addLayout(extra_btns)
        rev_layout.addWidget(checks_box)

        save_btns = QHBoxLayout()
        self.open_editor_btn = QPushButton("Review ranges in editor…")
        self.open_editor_btn.setProperty("class", "accent")
        self.open_editor_btn.clicked.connect(self._review_in_editor)
        save_btns.addWidget(self.open_editor_btn)

        self.save_views_btn = QPushButton("Save ready profiles")
        self.save_views_btn.clicked.connect(self._save_ready_profiles)
        save_btns.addWidget(self.save_views_btn)

        self.keep_evidence_btn = QPushButton("Keep evidence only")
        self.keep_evidence_btn.clicked.connect(self._keep_evidence_only)
        save_btns.addWidget(self.keep_evidence_btn)

        rev_cancel = QPushButton("Close")
        rev_cancel.clicked.connect(self.reject)
        rev_cancel.setVisible(not self._embedded)
        save_btns.addWidget(rev_cancel)
        save_btns.addStretch(1)
        rev_layout.addLayout(save_btns)
        self.stack.addWidget(review_page)

        self._update_orientation_indicator()

    def _apply_target_profile_defaults(self) -> None:
        if self._target_profile is None:
            return
        prof = self._target_profile
        self.name_input.setText(f"{prof.name} camera check")
        cp = prof.capture_profile
        if cp.get("side") == "left":
            self._detected_side = "left"
        elif cp.get("side") == "right":
            self._detected_side = "right"
        if cp.get("facing") == "left":
            self._detected_facing = "left"
        elif cp.get("facing") == "right":
            self._detected_facing = "right"
        src = str(cp.get("source", ""))
        if src.startswith("camera:"):
            try:
                cam_idx = int(src.partition(":")[2])
                pos = self.camera_combo.findData(cam_idx)
                if pos >= 0:
                    self.camera_combo.setCurrentIndex(pos)
            except ValueError:
                pass
        for cid, spec in CHECK_SPECS.items():
            if any(m in prof.metrics for m in spec.metrics):
                self.check_boxes[cid].setChecked(True)

    def _update_orientation_indicator(self) -> None:
        mirror_txt = " · Preview mirrored (display only)" if self.mirror_cb.isChecked() else ""
        res = ""
        if self._camera_meta:
            w = self._camera_meta.get("width", 0)
            h = self._camera_meta.get("height", 0)
            fps = self._camera_meta.get("fps", 0.0)
            if w and h:
                res = f" · {w}×{h}" + (f" @ {fps:.0f} fps" if fps else "")
        self.orientation_label.setText(
            f"Auto-detected orientation: {self._detected_side} side, facing {self._detected_facing}{mirror_txt}{res}"
        )

    def _selected_setup_config(self) -> Config:
        cam_idx = int(self.camera_combo.currentData() or 0)
        return Config.model_validate({
            **self._cfg.model_dump(),
            "camera": cam_idx,
            "side": self._detected_side,
            "facing": self._detected_facing,
            "mirror": False,  # stored landmarks and footage stay unmirrored
            "beat_tracking": False,
        })

    def _scan_cameras(self) -> None:
        current = int(self.camera_combo.currentData() or self._cfg.camera)
        discovered = self._camera_discoverer(configured_index=current)
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for idx, label in discovered:
            self.camera_combo.addItem(label, idx)
        pos = self.camera_combo.findData(current)
        if pos >= 0:
            self.camera_combo.setCurrentIndex(pos)
        self.camera_combo.blockSignals(False)

    def _on_camera_changed(self) -> None:
        if self._camera_active:
            self._release_camera()
            self._start_preview()

    def _start_preview(self) -> None:
        self._release_camera()
        self._temp_dir = TemporaryDirectory(prefix="tango-solo-cal-")
        temp_root = Path(self._temp_dir.name)
        self._staging_dir = temp_root / "staging"
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        cfg = self._selected_setup_config()
        cfg = Config.model_validate({**cfg.model_dump(), "data_dir": str(temp_root)})

        self._display = QtPracticeDisplay(capture_evidence=True)
        self._display.disable_settings_coaching = True
        self._worker = QtPracticeWorker(
            cfg,
            self._display,
            practice_name="solo_calibration_evidence",
            calibration_only=False,
        )
        self._thread = PracticeThread(self._worker, self)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._camera_active = True
        self.retry_camera_btn.setVisible(False)
        self.start_preview_btn.setEnabled(False)
        self.status_banner.setText("Starting camera preview…")
        self._thread.start()
        self._poll_timer.start()
        if self.speak_cb.isChecked():
            self._speech.speak(
                0,
                "Camera preview started. Step back until your head and shoes fit in the frame, then press Begin.",
            )

    def _on_worker_failed(self, message: str) -> None:
        self._release_camera()
        self.retry_camera_btn.setVisible(True)
        self.start_preview_btn.setEnabled(True)
        self.begin_btn.setEnabled(False)
        if "busy" in message.lower() or "open" in message.lower() or "camera" in message.lower():
            self.status_banner.setText(
                f"Camera unavailable ({message}). Close any other preview using this camera and click Retry camera."
            )
        else:
            self.status_banner.setText(f"Camera error: {message}")
        self._session.set_camera_error(message)

    def _on_worker_finished(self) -> None:
        self._camera_active = False
        self.start_preview_btn.setEnabled(True)

    def _poll_worker(self) -> None:
        if self._display is None:
            return
        latest_frame = None
        while True:
            try:
                latest_frame = self._display.frames.get_nowait()
            except queue.Empty:
                break
        if latest_frame is not None:
            display_img = (
                cv2.flip(latest_frame, 1) if self.mirror_cb.isChecked() else latest_frame
            )
            self.preview_stage.set_frame(numpy_to_qimage(display_img))

        while not self._display.observations.empty():
            obs = self._display.observations.get_nowait()
            self._collected_frames.append(obs)
            self._camera_meta = {"width": obs.width, "height": obs.height, "fps": self._camera_meta.get("fps", 0.0)}
            self._update_orientation_indicator()
            if self.stack.currentIndex() == 0:
                ok, reason = check_full_body_framing(obs, self._selected_setup_config())
                self.begin_btn.setEnabled(ok)
                self.status_banner.setText(reason)
            elif self.stack.currentIndex() == 1 and not self._paused:
                prev_accepted = len(self._session.accepted_passages)
                self._session.process_frame(
                    obs,
                    now_monotonic=time.monotonic(),
                    speak_enabled=self.speak_cb.isChecked(),
                )
                if len(self._session.accepted_passages) > prev_accepted:
                    self._announce_captured_and_prompt()
                self._sync_capture_ui()
        if self.stack.currentIndex() == 1 and not self._paused and self._session.check_watchdog(time.monotonic()):
            self._on_worker_failed(self._session.error_message)

    def _begin_capture(self) -> None:
        if not self._camera_active:
            self.status_banner.setText("Start camera preview before beginning capture.")
            return
        if self._display is not None:
            self._display.recording.set()
        self._frozen_cfg = self._selected_setup_config()
        self.camera_combo.setEnabled(False)
        self._last_spoken_hint = ""
        self._session = SoloCameraSession(
            self._frozen_cfg,
            now_monotonic=time.monotonic(),
            auto_detect_side_facing=True,
        )
        phases = build_quick_setup_phases()
        self.stack.setCurrentIndex(1)
        prompt_id = self._session.start_plan(
            phases,
            now_monotonic=time.monotonic(),
            speak_enabled=self.speak_cb.isChecked(),
        )
        self._maybe_speak(prompt_id)
        self._sync_capture_ui()

    def _announce_captured_and_prompt(self) -> None:
        if self.speak_cb.isChecked():
            if self._session.state == "Prompt":
                self._speech.speak(
                    self._session.active_prompt_id,
                    f"Captured. {self._session.current_instruction}",
                )
            elif self._session.state == "Review":
                self._speech.speak(0, "Captured. Your standing examples are ready.")

    def _maybe_speak(self, prompt_id: int) -> None:
        if self.speak_cb.isChecked() and self._session.state == "Prompt":
            self._speech.speak(prompt_id, self._session.current_instruction)

    def _on_speech_finished(self, prompt_id: int, failed: bool) -> None:
        self._session.on_speech_finished(prompt_id, now_monotonic=time.monotonic(), failed=failed)
        self._sync_capture_ui()

    def _sync_capture_ui(self) -> None:
        phase = self._session.current_phase
        self.phase_label.setText(
            f"State: {self._session.state}"
            + (f" · {phase.view.capitalize()} rep {phase.repetition}" if phase else "")
        )
        self.instruction_label.setText(self._session.current_instruction)
        self.hint_label.setText(self._session.hint_message)
        self._detected_side = self._session.cfg.side
        self._detected_facing = self._session.cfg.facing
        if (
            self.speak_cb.isChecked()
            and self._session.hint_message
            and self._session.hint_message != self._last_spoken_hint
            and self._session.state == "Wait for view and pose"
            and "waiting" in self._session.hint_message.lower()
        ):
            self._last_spoken_hint = self._session.hint_message
            self._speech.speak(0, self._session.hint_message)

        is_walking = phase is not None and not phase.requires_stillness
        self.start_take_btn.setVisible(is_walking and self._session.state == "Wait for take")
        self.end_take_btn.setVisible(is_walking and self._session.state == "Capture take")

        if self._session.state == "Review":
            self._enter_review()

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.pause_btn.setText("Resume" if self._paused else "Pause")
        if self._paused:
            self._speech.cancel_all()

    def _repeat_instruction(self) -> None:
        self._speech.cancel_all()
        prompt_id = self._session.retry_phase(
            now_monotonic=time.monotonic(),
            speak_enabled=self.speak_cb.isChecked(),
        )
        self._maybe_speak(prompt_id)
        self._sync_capture_ui()

    def _retry_hold(self) -> None:
        self._repeat_instruction()

    def _skip_view(self) -> None:
        self._speech.cancel_all()
        prompt_id = self._session.skip_view(
            now_monotonic=time.monotonic(),
            speak_enabled=self.speak_cb.isChecked(),
        )
        self._maybe_speak(prompt_id)
        self._sync_capture_ui()

    def _start_walking_take(self) -> None:
        self._session.start_walking_take(now_monotonic=time.monotonic())
        self._sync_capture_ui()

    def _end_walking_take(self) -> None:
        self._session.end_walking_take()
        self._sync_capture_ui()

    def _confirm_walk_return(self) -> None:
        self.walk_returned_btn.setVisible(False)

    def _enter_review(self) -> None:
        self._speech.cancel_all()
        self._release_camera()
        self.stack.setCurrentIndex(2)
        records = self._session.accepted_passages
        if not records:
            self.review_summary.setText("No usable holds were captured yet.")
            return
        lines = [f"Captured {len(records)} accepted passage(s):"]
        for rec in records.values():
            metrics_txt = ", ".join(f"{k} ({v:.2f}s)" for k, v in rec.eligible_metrics.items())
            lines.append(
                f"• {rec.phase_id} [{rec.view}]: {rec.passage.start:.2f}s–{rec.passage.end:.2f}s · {metrics_txt}"
            )
            for mk, reason in rec.unavailable_reasons.items():
                lines.append(f"  - {mk} unavailable: {reason}")
        self.review_summary.setText("\n".join(lines))

    def _run_selected_checks(self) -> None:
        selected = [cid for cid, cb in self.check_boxes.items() if cb.isChecked()]
        if not selected:
            QMessageBox.information(self, "Select checks", "Select at least one measurement check.")
            return
        self._start_preview()
        if self._display is not None:
            self._display.recording.set()
        self.stack.setCurrentIndex(1)
        prompt_id = self._session.start_plan(
            build_check_phases(selected),
            now_monotonic=time.monotonic(),
            speak_enabled=self.speak_cb.isChecked(),
        )
        self._maybe_speak(prompt_id)
        self._sync_capture_ui()

    def _run_walking_plan(self) -> None:
        self._start_preview()
        if self._display is not None:
            self._display.recording.set()
        self.stack.setCurrentIndex(1)
        prompt_id = self._session.start_plan(
            build_walking_phases(),
            now_monotonic=time.monotonic(),
            speak_enabled=self.speak_cb.isChecked(),
        )
        self._maybe_speak(prompt_id)
        self._sync_capture_ui()

    def _stage_and_publish_evidence(self) -> Path:
        """Promote temporary practice capture into staging, verify it, and publish to references/."""
        if self._published_tpr is not None and self._published_tpr.is_file():
            return self._published_tpr
        if self._temp_dir is None:
            raise ValueError("No captured camera session is available to stage.")
        temp_root = Path(self._temp_dir.name)
        practice_files = sorted((temp_root / "practices").glob("*.tp"))
        if not practice_files:
            raise ValueError("No recorded practice archive was found in temporary storage.")
        src_tp = practice_files[-1]
        base_name = self.name_input.text().strip() or "Solo camera evidence"
        staged = self._staging_dir / f"{base_name}.tpr" if self._staging_dir else temp_root / f"{base_name}.tpr"
        if staged.exists():
            staged.unlink()
        promote_reference(src_tp, staged, base_name)
        # Validate ledger passages exist inside evidence timestamps
        ev = cached_evidence(staged)
        last_ts = ev.frames[-1].timestamp
        for rec in self._session.accepted_passages.values():
            if rec.passage.start >= last_ts:
                raise ValueError(f"Passage {rec.phase_id} is outside promoted evidence timestamps.")
        self._staged_tpr = staged
        ref_dir = reference_directory(self._cfg.data_dir)
        ref_dir.mkdir(parents=True, exist_ok=True)
        dest = reference_path(ref_dir, f"{base_name} {int(time.time())}")
        shutil.copy2(staged, dest)
        self._published_tpr = dest
        return dest

    def _ledger_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "phase_id": rec.phase_id,
                "plan_kind": rec.plan_kind,
                "view": rec.view,
                "repetition": rec.repetition,
                "role": rec.role,
                "start": rec.passage.start,
                "end": rec.passage.end,
                "check_id": rec.check_id,
                "eligible_metrics": rec.eligible_metrics,
                "unavailable_reasons": rec.unavailable_reasons,
            }
            for rec in self._session.accepted_passages.values()
        ]

    def _build_metric_drafts(self, view: str, *, context: str) -> tuple[dict[str, list[Passage]], dict[str, str], dict[str, str]]:
        passages_by_metric: dict[str, list[Passage]] = {}
        views_by_metric: dict[str, str] = {}
        reasons: dict[str, str] = {}
        role_target = "walking" if context == "Solo walking" else "neutral"
        for rec in self._session.accepted_passages.values():
            if rec.view != view or rec.role != role_target:
                continue
            for mk in rec.eligible_metrics:
                passages_by_metric.setdefault(mk, []).append(rec.passage)
                views_by_metric[mk] = view
            for mk, reason in rec.unavailable_reasons.items():
                reasons.setdefault(mk, reason)
        return passages_by_metric, views_by_metric, reasons

    def _review_in_editor(self) -> None:
        try:
            tpr = self._stage_and_publish_evidence()
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Evidence staging failed", str(error))
            return
        cfg = self._frozen_cfg or self._selected_setup_config()
        views_present = {rec.view for rec in self._session.accepted_passages.values()}
        default_view = "side" if "side" in views_present else next(iter(views_present), "side")
        metric_passages, metric_views, reasons = self._build_metric_drafts(
            default_view, context="Neutral posture"
        )
        base_name = self.name_input.text().strip() or "Studio webcam"
        editor = CalibrationEditor(
            tpr,
            cfg,
            self,
            bridge=self.bridge,
            base_profile=self._target_profile,
            metric_passages=metric_passages,
            metric_views=metric_views,
            unavailable_reasons=reasons,
            ledger=self._ledger_payload(),
            default_context="Neutral posture",
            default_name=f"{base_name} · Neutral · {default_view.capitalize()}",
        )
        if editor.exec() == QDialog.DialogCode.Accepted and editor.saved_path is not None:
            self.saved_path = editor.saved_path
            self.saved_paths.append(editor.saved_path)
            self.bridge.calibrations_changed.emit()
            self.accept()

    def _save_ready_profiles(self) -> None:
        try:
            tpr = self._stage_and_publish_evidence()
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Evidence staging failed", str(error))
            return
        cfg = self._frozen_cfg or self._selected_setup_config()
        ev = cached_evidence(tpr)
        cal_dir = calibration_directory(self._cfg.data_dir)
        cal_dir.mkdir(parents=True, exist_ok=True)
        base_name = self.name_input.text().strip() or "Studio webcam"
        max_gap = self._session.effective_max_gap()

        # Build samples per view
        raw_samples: dict[str, list[tuple[float, float | None]]] = {k: [] for k in METRIC_DETAILS}
        for fr in ev.frames:
            obs = measure_ranges(fr, cfg)
            for mk in METRIC_DETAILS:
                q = obs.quality.get(mk)
                val = obs.values.get(mk)
                raw_samples[mk].append((fr.timestamp, val if (q and q.valid) else None))

        targets: list[tuple[str, str]] = []
        if any(r.view == "front" and r.role == "neutral" for r in self._session.accepted_passages.values()):
            targets.append(("Neutral posture", "front"))
        if any(r.view == "side" and r.role == "neutral" for r in self._session.accepted_passages.values()):
            targets.append(("Neutral posture", "side"))
        if any(r.role == "walking" for r in self._session.accepted_passages.values()):
            targets.append(("Solo walking", "side"))

        if not targets:
            QMessageBox.information(self, "Nothing to save", "No usable holds or walking passes were captured.")
            return

        written: list[Path] = []
        failed_names: list[str] = []
        for context, view in targets:
            passages_by_metric, _, _ = self._build_metric_drafts(view, context=context)
            approved: dict[str, ApprovedRange] = {}
            for mk, passages in passages_by_metric.items():
                try:
                    stats = summarize_examples(
                        raw_samples[mk],
                        passages,
                        max_gap,
                        suggest_lower_pct=cfg.calibration_suggest_lower_pct,
                        suggest_upper_pct=cfg.calibration_suggest_upper_pct,
                        min_usable_samples=cfg.calibration_min_usable_samples,
                    )
                except ValueError:
                    continue
                check_ok = False
                if context != "Solo walking":
                    for cid, spec in CHECK_SPECS.items():
                        if mk in spec.metrics:
                            holds_4 = [
                                rec.passage
                                for pid in (f"{cid}_neutral_a", f"{cid}_changed_a", f"{cid}_neutral_b", f"{cid}_changed_b")
                                if (rec := self._session.accepted_passages.get(pid)) is not None
                            ]
                            if len(holds_4) == 4:
                                ok, _ = evaluate_metric_repeat_check(
                                    raw_samples[mk], holds_4, max_gap, expected_sign=spec.expected_sign
                                )
                                check_ok = ok
                approved[mk] = ApprovedRange(
                    lower=float(stats["lower"]),
                    upper=float(stats["upper"]),
                    unit=METRIC_DETAILS[mk][2],
                    view=view,  # type: ignore[arg-type]
                    passages=passages,
                    median=float(stats["median"]),
                    p10=float(stats["p10"]),
                    p90=float(stats["p90"]),
                    valid_duration=float(stats["valid_duration"]),
                    selected_duration=float(stats["selected_duration"]),
                    samples=int(stats["samples"]),
                    inside_fraction=float(stats["inside_fraction"]),
                    camera_check_passed=check_ok,
                )
            if not approved:
                continue
            prof_name = f"{base_name} · {'Solo walking' if context == 'Solo walking' else 'Neutral'} · {view.capitalize()}"
            prof_path = calibration_path(cal_dir, prof_name)
            profile_obj = RangeProfile(
                name=prof_name,
                context=context,
                model_hash=ev.model_hash,
                capture_profile=ev.capture_profile,
                source=ev.source,
                metrics=approved,
                camera_checks={"solo_camera_ledger": self._ledger_payload()},
            )
            try:
                save_range_profile(prof_path, profile_obj, overwrite=False)
                written.append(prof_path)
            except (OSError, ValueError, FileExistsError) as err:
                failed_names.append(f"{prof_name}: {err}")

        if written:
            self.saved_paths.extend(written)
            self.saved_path = written[-1]
            self.bridge.calibrations_changed.emit()
        if failed_names:
            QMessageBox.warning(
                self,
                "Partial save",
                "Some profiles could not be saved:\n" + "\n".join(failed_names),
            )
            return
        if written:
            self.accept()

    def _keep_evidence_only(self) -> None:
        try:
            tpr = self._stage_and_publish_evidence()
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Could not save evidence", str(error))
            return
        self.saved_reference_path = tpr
        self.status_banner.setText(f"Evidence saved to {tpr.name} without creating a profile.")
        self.accept()

    def _release_camera(self) -> None:
        self._poll_timer.stop()
        self._camera_active = False
        if self._display is not None:
            if self._display.recording.is_set():
                self._display.request("e")
            self._display.request("q")
            self._display.stop = True
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
        self._thread = None
        self._worker = None

    def closeEvent(self, event: QCloseEvent) -> None:
        self._speech.cancel_all()
        self._release_camera()
        super().closeEvent(event)

    def reject(self) -> None:
        self._speech.cancel_all()
        self._release_camera()
        super().reject()
