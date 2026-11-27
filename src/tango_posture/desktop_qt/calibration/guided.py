"""Solo camera self-calibration wizard with readiness-driven capture and staging."""

from __future__ import annotations

import math
import queue
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...archive import calibration_directory, calibration_path, practice_directory
from ...calibration.archives import EMBEDDED_EVIDENCE_MEMBER, save_range_profile
from ...calibration.camera_session import (
    CHECK_SPECS,
    SoloCameraSession,
    build_check_phases,
    build_quick_setup_phases,
    build_walking_phases,
    check_full_body_framing,
    evaluate_metric_repeat_check,
    ledger_metric_samples,
)
from ...calibration.evidence import cached_evidence
from ...calibration.measurements import metric_samples
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
    apply_discovered_cameras,
    discover_cameras,
    populate_camera_combo,
)
from ..service_bridge import ServiceBridge, numpy_to_qimage
from ..theme import COLOR_BASE, COLOR_BORDER, COLOR_TEXT_MUTED, COLOR_TEXT_PRIMARY
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
        self._review_entered = False
        self._editor_open = False
        self._repeat_wait_since: float | None = None
        self._repeat_interval_s = 5.0
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
        self.bridge.cameras_discovered.connect(self._on_cameras_discovered)

    def _setup_ui(self, profile_name: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0 if self._embedded else 16, 0 if self._embedded else 16, 0 if self._embedded else 16, 0 if self._embedded else 16)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # Left column: Live Camera Stage
        left_card = QFrame()
        left_card.setProperty("class", "card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        self.status_banner = QLabel(
            "Camera is off. Click Start camera preview to check framing before recording."
        )
        self.status_banner.setWordWrap(True)
        self.status_banner.setStyleSheet(
            f"background-color: {COLOR_BASE}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 6px; padding: 8px 10px; color: {COLOR_TEXT_PRIMARY}; font-size: 12px;"
        )
        left_layout.addWidget(self.status_banner)

        self.preview_stage = VideoStage("Camera preview is off. Start preview to verify your setup.")
        self.preview_stage.setMinimumSize(460, 340)
        left_layout.addWidget(self.preview_stage, 1)

        self.orientation_label = QLabel("")
        self.orientation_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        self.orientation_label.setWordWrap(True)
        left_layout.addWidget(self.orientation_label)
        splitter.addWidget(left_card)

        # Right column: Guided Controls Card
        right_card = QFrame()
        right_card.setProperty("class", "card")
        right_card.setMinimumWidth(360)
        right_card.setMaximumWidth(480)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        self.stack = QStackedWidget()
        right_layout.addWidget(self.stack, 1)
        splitter.addWidget(right_card)
        splitter.setSizes([640, 400])

        # Page 0: Camera Setup
        setup_page = QWidget()
        setup_layout = QVBoxLayout(setup_page)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.setSpacing(12)

        setup_title = QLabel("1. Camera & File Setup")
        setup_title.setStyleSheet("font-size: 15px; font-weight: bold;")
        setup_layout.addWidget(setup_title)

        form = QFormLayout()
        form.setSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.name_input = QLineEdit(profile_name)
        self.name_input.textChanged.connect(self._update_filename_preview)
        form.addRow("Calibration name:", self.name_input)

        self.filename_preview = QLabel("")
        self.filename_preview.setWordWrap(True)
        self.filename_preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.filename_preview.setStyleSheet(
            f"background-color: {COLOR_BASE}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 6px; padding: 8px; color: {COLOR_TEXT_PRIMARY}; font-size: 12px;"
        )
        form.addRow("Output files:", self.filename_preview)

        self.camera_combo = QComboBox()
        self.camera_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        if self.bridge.discovered_cameras:
            apply_discovered_cameras(
                self.camera_combo,
                self.bridge.discovered_cameras,
                fallback_index=self._cfg.camera,
            )
        else:
            self.camera_combo.addItem(f"Camera {self._cfg.camera}", self._cfg.camera)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        form.addRow("Camera:", self.camera_combo)
        setup_layout.addLayout(form)

        cam_btns = QHBoxLayout()
        self.scan_cameras_btn = QPushButton("Scan cameras")
        self.scan_cameras_btn.clicked.connect(self._scan_cameras)
        cam_btns.addWidget(self.scan_cameras_btn)
        self.start_preview_btn = QPushButton("Start camera preview")
        self.start_preview_btn.setProperty("class", "primary")
        self.start_preview_btn.clicked.connect(self._start_preview)
        self.preview = self.start_preview_btn
        cam_btns.addWidget(self.start_preview_btn, 1)
        self.retry_camera_btn = QPushButton("Retry camera")
        self.retry_camera_btn.setVisible(False)
        self.retry_camera_btn.clicked.connect(self._start_preview)
        cam_btns.addWidget(self.retry_camera_btn)
        setup_layout.addLayout(cam_btns)

        opts_box = QGroupBox("Options")
        opts_layout = QVBoxLayout(opts_box)
        opts_layout.setSpacing(6)
        self.mirror_cb = QCheckBox("Mirror preview (display only)")
        self.mirror_cb.setChecked(self._cfg.mirror)
        self.mirror_cb.toggled.connect(self._update_orientation_indicator)
        opts_layout.addWidget(self.mirror_cb)

        self.speak_cb = QCheckBox("Speak instructions")
        self.speak_cb.setChecked(self._cfg.solo_camera.speak_instructions)
        opts_layout.addWidget(self.speak_cb)

        self.auto_start_cb = QCheckBox("Auto-start when landmarks settle")
        self.auto_start_cb.setChecked(True)
        opts_layout.addWidget(self.auto_start_cb)
        setup_layout.addWidget(opts_box)

        setup_note = QLabel(
            "Place the camera level at practice height and step back until your head and shoes fit. "
            "The app will guide you by voice and detect your side/facing automatically."
        )
        setup_note.setWordWrap(True)
        setup_note.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        setup_layout.addWidget(setup_note)
        setup_layout.addStretch(1)

        setup_btns = QHBoxLayout()
        self.begin_btn = QPushButton("Begin")
        self.begin_btn.setProperty("class", "accent")
        self.begin_btn.setEnabled(False)
        self.begin_btn.clicked.connect(self._begin_capture)
        self.start = self.begin_btn
        setup_btns.addWidget(self.begin_btn, 1)
        setup_cancel = QPushButton("Cancel")
        setup_cancel.clicked.connect(self.reject)
        setup_cancel.setVisible(not self._embedded)
        setup_btns.addWidget(setup_cancel)
        setup_layout.addLayout(setup_btns)
        self.stack.addWidget(setup_page)

        # Page 1: Guided Capture
        capture_page = QWidget()
        cap_layout = QVBoxLayout(capture_page)
        cap_layout.setContentsMargins(0, 0, 0, 0)
        cap_layout.setSpacing(12)

        self.phase_label = QLabel("Ready")
        self.phase_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        cap_layout.addWidget(self.phase_label)

        self.instruction_label = QLabel("")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet(
            f"background-color: {COLOR_BASE}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 6px; padding: 12px; color: {COLOR_TEXT_PRIMARY}; font-size: 14px;"
        )
        cap_layout.addWidget(self.instruction_label)

        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
        cap_layout.addWidget(self.hint_label)
        cap_layout.addStretch(1)

        cap_grid = QGridLayout()
        cap_grid.setSpacing(8)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        cap_grid.addWidget(self.pause_btn, 0, 0)

        self.repeat_btn = QPushButton("Repeat instruction")
        self.repeat_btn.clicked.connect(self._repeat_instruction)
        cap_grid.addWidget(self.repeat_btn, 0, 1)

        self.retry_hold_btn = QPushButton("Retry hold")
        self.retry_hold_btn.clicked.connect(self._retry_hold)
        cap_grid.addWidget(self.retry_hold_btn, 1, 0)

        self.skip_view_btn = QPushButton("Skip view")
        self.skip_view_btn.clicked.connect(self._skip_view)
        cap_grid.addWidget(self.skip_view_btn, 1, 1)

        self.start_take_btn = QPushButton("Start take")
        self.start_take_btn.setVisible(False)
        self.start_take_btn.clicked.connect(self._start_walking_take)
        cap_grid.addWidget(self.start_take_btn, 2, 0)

        self.end_take_btn = QPushButton("End take")
        self.end_take_btn.setVisible(False)
        self.end_take_btn.clicked.connect(self._end_walking_take)
        cap_grid.addWidget(self.end_take_btn, 2, 1)

        self.walk_returned_btn = QPushButton("Ready for pass 2")
        self.walk_returned_btn.setVisible(False)
        self.walk_returned_btn.clicked.connect(self._confirm_walk_return)
        cap_grid.addWidget(self.walk_returned_btn, 3, 0, 1, 2)

        cap_cancel = QPushButton("Cancel")
        cap_cancel.clicked.connect(self.reject)
        cap_grid.addWidget(cap_cancel, 4, 0, 1, 2)
        cap_layout.addLayout(cap_grid)
        self.stack.addWidget(capture_page)

        # Page 2: Checks, Walking & Review Summary
        review_scroll = QScrollArea()
        review_scroll.setWidgetResizable(True)
        review_scroll.setFrameShape(QFrame.Shape.NoFrame)
        review_page = QWidget()
        rev_layout = QVBoxLayout(review_page)
        rev_layout.setContentsMargins(0, 0, 4, 0)
        rev_layout.setSpacing(10)

        rev_title = QLabel("Review & Save")
        rev_title.setStyleSheet("font-size: 15px; font-weight: bold;")
        rev_layout.addWidget(rev_title)

        self.review_summary = QLabel("")
        self.review_summary.setWordWrap(True)
        self.review_summary.setStyleSheet(
            f"background-color: {COLOR_BASE}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 6px; padding: 10px; color: {COLOR_TEXT_PRIMARY}; font-size: 12px;"
        )
        rev_layout.addWidget(self.review_summary)

        rev_form = QFormLayout()
        rev_form.setSpacing(8)
        self.review_name_input = QLineEdit(profile_name)
        self.review_name_input.textChanged.connect(self.name_input.setText)
        self.name_input.textChanged.connect(self.review_name_input.setText)
        rev_form.addRow("Calibration name:", self.review_name_input)

        self.review_filename_preview = QLabel("")
        self.review_filename_preview.setWordWrap(True)
        self.review_filename_preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.review_filename_preview.setStyleSheet(
            f"background-color: {COLOR_BASE}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 6px; padding: 8px; color: {COLOR_TEXT_PRIMARY}; font-size: 12px;"
        )
        rev_form.addRow("Output files:", self.review_filename_preview)
        rev_layout.addLayout(rev_form)

        checks_box = QGroupBox("Optional checks & solo walking")
        checks_layout = QVBoxLayout(checks_box)
        self.check_boxes: dict[str, QCheckBox] = {}
        cb_grid = QGridLayout()
        for idx, (cid, spec) in enumerate(CHECK_SPECS.items()):
            cb = QCheckBox(spec.title)
            self.check_boxes[cid] = cb
            cb_grid.addWidget(cb, idx // 2, idx % 2)
        checks_layout.addLayout(cb_grid)

        extra_btns = QHBoxLayout()
        self.run_checks_btn = QPushButton("Check measurements")
        self.run_checks_btn.clicked.connect(self._run_selected_checks)
        extra_btns.addWidget(self.run_checks_btn)

        self.record_walking_btn = QPushButton("Record solo walking")
        self.record_walking_btn.clicked.connect(self._run_walking_plan)
        extra_btns.addWidget(self.record_walking_btn)
        checks_layout.addLayout(extra_btns)
        rev_layout.addWidget(checks_box)

        save_btns = QVBoxLayout()
        save_btns.setSpacing(6)
        self.open_editor_btn = QPushButton("Review ranges in editor…")
        self.open_editor_btn.setProperty("class", "accent")
        self.open_editor_btn.clicked.connect(self._review_in_editor)
        save_btns.addWidget(self.open_editor_btn)

        self.save_views_btn = QPushButton("Save ready profiles")
        self.save_views_btn.setProperty("class", "primary")
        self.save_views_btn.clicked.connect(self._save_ready_profiles)
        save_btns.addWidget(self.save_views_btn)

        self.keep_evidence_btn = QPushButton("Keep evidence only")
        self.keep_evidence_btn.clicked.connect(self._keep_evidence_only)
        save_btns.addWidget(self.keep_evidence_btn)

        rev_cancel = QPushButton("Close")
        rev_cancel.clicked.connect(self.reject)
        rev_cancel.setVisible(not self._embedded)
        save_btns.addWidget(rev_cancel)
        rev_layout.addLayout(save_btns)
        rev_layout.addStretch(1)
        review_scroll.setWidget(review_page)
        self.stack.addWidget(review_scroll)

        self._update_orientation_indicator()
        self._update_filename_preview()

    def _update_filename_preview(self) -> None:
        base = self.name_input.text().strip() or "Studio webcam"
        preview_txt = f"• {base}.tpc"
        self.filename_preview.setText(preview_txt)
        if hasattr(self, "review_filename_preview"):
            self.review_filename_preview.setText(preview_txt)

    def start_new_calibration(self) -> None:
        """Reset state and start a complete new guided camera calibration."""
        self._speech.cancel_all()
        self._release_camera()
        self._paused = False
        self._review_entered = False
        self._editor_open = False
        self._repeat_wait_since = None
        self._repeat_interval_s = 5.0
        self._collected_frames.clear()
        self._staged_tpr = None
        self._published_tpr = None
        self.saved_path = None
        self.saved_paths.clear()
        self.camera_combo.setEnabled(True)
        self.stack.setCurrentIndex(0)
        self._start_preview(auto_begin=True)

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
        discovered = populate_camera_combo(
            self.camera_combo,
            self._camera_discoverer,
            fallback_index=self._cfg.camera,
        )
        self.bridge.set_discovered_cameras(discovered)

    def _on_cameras_discovered(self, cameras: object) -> None:
        if self._camera_active or not isinstance(cameras, list):
            return
        apply_discovered_cameras(
            self.camera_combo,
            cameras,
            fallback_index=self._cfg.camera,
        )

    def _on_camera_changed(self) -> None:
        if self._camera_active:
            self._release_camera()
            self._start_preview()

    def _start_preview(self, *, auto_begin: bool = False, announce: bool = True) -> None:
        self._speech.cancel_all()
        self._release_camera()
        self._auto_begin_pending = auto_begin
        self._preview_stable_since: float | None = None
        self._staged_tpr = None
        self._published_tpr = None
        self._temp_dir = TemporaryDirectory(prefix="tango-solo-cal-", ignore_cleanup_errors=True)
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
        if announce and self.speak_cb.isChecked():
            preview_prompt = (
                "Camera preview started. Step back until your head and shoes fit in the frame."
                if auto_begin or self.auto_start_cb.isChecked()
                else "Camera preview started. Step back until your head and shoes fit in the frame, then press Begin."
            )
            self._speech.speak(0, preview_prompt)

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
                if ok and (getattr(self, "_auto_begin_pending", False) or self.auto_start_cb.isChecked()):
                    now_m = time.monotonic()
                    if self._speech.is_speaking():
                        self._preview_stable_since = None
                    else:
                        prev_stable = getattr(self, "_preview_stable_since", None)
                        if prev_stable is None:
                            self._preview_stable_since = now_m
                        elif now_m - prev_stable >= self._cfg.solo_camera.stable_time_s:
                            self._auto_begin_pending = False
                            self._begin_capture()
                            return
                else:
                    self._preview_stable_since = None
            elif self.stack.currentIndex() == 1 and not self._paused:
                prev_accepted = len(self._session.accepted_passages)
                self._session.process_frame(
                    obs,
                    now_monotonic=time.monotonic(),
                    speak_enabled=self.speak_cb.isChecked(),
                )
                if len(self._session.accepted_passages) > prev_accepted:
                    self._repeat_wait_since = None
                    self._announce_captured_and_prompt()
                self._sync_capture_ui()
        if self.stack.currentIndex() == 1 and not self._paused:
            self._update_repeat_timer(time.monotonic())
            if self._session.check_watchdog(time.monotonic()):
                self._on_worker_failed(self._session.error_message)

    def _begin_capture(self) -> None:
        if not self._camera_active:
            self.status_banner.setText("Start camera preview before beginning capture.")
            return
        if self._display is not None:
            self._display.recording.set()
        self._review_entered = False
        self._editor_open = False
        self._repeat_wait_since = None
        self._repeat_interval_s = 5.0
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
        self._last_spoken_hint = ""
        if self.speak_cb.isChecked() and self._session.state == "Prompt":
            self._speech.speak(
                self._session.active_prompt_id,
                f"Captured. {self._session.current_instruction}",
            )

    def _maybe_speak(self, prompt_id: int) -> None:
        if self.speak_cb.isChecked() and self._session.state == "Prompt":
            self._speech.speak(prompt_id, self._session.current_instruction)

    def _on_speech_finished(self, prompt_id: int, failed: bool) -> None:
        if self._review_entered or self.stack.currentIndex() != 1:
            return
        self._session.on_speech_finished(prompt_id, now_monotonic=time.monotonic(), failed=failed)
        if self._session.state in ("Wait for view and pose", "Wait for take"):
            self._repeat_wait_since = time.monotonic()
        self._sync_capture_ui()

    def _update_repeat_timer(self, now_m: float) -> None:
        if self._session.state not in ("Wait for view and pose", "Wait for take"):
            self._repeat_wait_since = None
            return
        if self._speech.is_speaking():
            self._repeat_wait_since = now_m
            return
        if self._repeat_wait_since is None:
            self._repeat_wait_since = now_m
            return
        elapsed = now_m - self._repeat_wait_since
        if elapsed >= self._repeat_interval_s:
            self._repeat_wait_since = now_m
            if self.speak_cb.isChecked() and self._session.current_instruction:
                self._speech.speak(0, self._session.current_instruction)
        self._sync_capture_ui()

    def _sync_capture_ui(self) -> None:
        phase = self._session.current_phase
        phase_text = (
            f"State: {self._session.state}"
            + (f" · {phase.view.capitalize()} rep {phase.repetition}" if phase else "")
        )
        if (
            self._session.state in ("Wait for view and pose", "Wait for take")
            and self.speak_cb.isChecked()
            and not self._speech.is_speaking()
        ):
            if self._repeat_wait_since is None:
                self._repeat_wait_since = time.monotonic()
            remaining = max(0, math.ceil(self._repeat_interval_s - (time.monotonic() - self._repeat_wait_since)))
            if remaining > 0:
                phase_text += f" · Repeat in {remaining}s"
        elif self._session.state not in ("Wait for view and pose", "Wait for take"):
            self._repeat_wait_since = None
        self.phase_label.setText(phase_text)
        self.instruction_label.setText(self._session.current_instruction)
        self.hint_label.setText(self._session.hint_message)
        orient_changed = (
            self._detected_side != self._session.cfg.side
            or self._detected_facing != self._session.cfg.facing
        )
        self._detected_side = self._session.cfg.side
        self._detected_facing = self._session.cfg.facing
        if orient_changed:
            self._update_orientation_indicator()
        if self._frozen_cfg is not None and (
            self._frozen_cfg.side != self._session.cfg.side
            or self._frozen_cfg.facing != self._session.cfg.facing
        ):
            self._frozen_cfg = self._frozen_cfg.model_copy(
                update={"side": self._session.cfg.side, "facing": self._session.cfg.facing}
            )
        if (
            self.speak_cb.isChecked()
            and not self._speech.is_speaking()
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
        else:
            self._repeat_wait_since = time.monotonic()

    def _repeat_instruction(self) -> None:
        self._speech.cancel_all()
        self._last_spoken_hint = ""
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
        self._last_spoken_hint = ""
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
        if self._review_entered:
            return
        self._review_entered = True
        self._release_camera()
        self.stack.setCurrentIndex(2)
        self.status_banner.setText("Self calibration completed. Review and save your calibration.")
        records = self._session.accepted_passages
        if not records:
            self.review_summary.setText("No usable holds were captured yet.")
            return
        lines = [
            "Self calibration completed. Review ranges, edit the name if needed, and save your .tpc file."
        ]
        lines.append(f"Captured {len(records)} accepted passage(s):")
        for rec in records.values():
            metrics_txt = ", ".join(f"{k} ({v:.2f}s)" for k, v in rec.eligible_metrics.items())
            lines.append(
                f"• {rec.phase_id} [{rec.view}]: {rec.passage.start:.2f}s–{rec.passage.end:.2f}s · {metrics_txt}"
            )
            for mk, reason in rec.unavailable_reasons.items():
                lines.append(f"  - {mk} unavailable: {reason}")
        self.review_summary.setText("\n".join(lines))
        if self.speak_cb.isChecked():
            self._speech.speak(0, "Self calibration completed.")
        QTimer.singleShot(0, self._review_in_editor)

    def _run_selected_checks(self) -> None:
        selected = [cid for cid, cb in self.check_boxes.items() if cb.isChecked()]
        if not selected:
            QMessageBox.information(self, "Select checks", "Select at least one measurement check.")
            return
        self._review_entered = False
        self._editor_open = False
        self._repeat_wait_since = None
        self._repeat_interval_s = 5.0
        self._last_spoken_hint = ""
        self._start_preview(announce=False)
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
        self._review_entered = False
        self._editor_open = False
        self._repeat_wait_since = None
        self._repeat_interval_s = 5.0
        self._last_spoken_hint = ""
        self._start_preview(announce=False)
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

    def _stage_evidence(self) -> Path:
        """Promote temporary practice capture into temporary staging only (never writes to references/)."""
        if self._staged_tpr is not None and self._staged_tpr.is_file():
            return self._staged_tpr
        if self._temp_dir is None:
            raise ValueError("No captured camera session is available to stage.")
        temp_root = Path(self._temp_dir.name)
        prac_dir = practice_directory(str(temp_root))
        practice_files = sorted(prac_dir.glob("*.tp")) if prac_dir.is_dir() else []
        if not practice_files and (temp_root / "practices").is_dir():
            practice_files = sorted((temp_root / "practices").glob("*.tp"))
        if not practice_files:
            practice_files = sorted(temp_root.rglob("*.tp"))
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
            if rec.passage.start > last_ts + 1.0:
                raise ValueError(f"Passage {rec.phase_id} is outside promoted evidence timestamps.")
        self._staged_tpr = staged
        return staged

    def _stage_and_publish_evidence(self) -> Path:
        """Promote temporary practice capture into staging and publish to references/."""
        if self._published_tpr is not None and self._published_tpr.is_file():
            return self._published_tpr
        staged = self._stage_evidence()
        base_name = self.name_input.text().strip() or "Solo camera evidence"
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
                "detected_side": rec.detected_side,
                "detected_facing": rec.detected_facing,
            }
            for rec in self._session.accepted_passages.values()
        ]

    def _build_metric_drafts(
        self, view: str | None, *, context: str
    ) -> tuple[dict[str, list[Passage]], dict[str, str], dict[str, str]]:
        passages_by_metric: dict[str, list[Passage]] = {}
        views_by_metric: dict[str, str] = {}
        reasons: dict[str, str] = {}
        role_target = "walking" if context == "Solo walking" else "neutral"
        for rec in self._session.accepted_passages.values():
            if rec.role != role_target:
                continue
            if view is not None and rec.view != view:
                continue
            for mk in rec.eligible_metrics:
                passages_by_metric.setdefault(mk, []).append(rec.passage)
                views_by_metric[mk] = rec.view
            for mk, reason in rec.unavailable_reasons.items():
                reasons.setdefault(mk, reason)
        return passages_by_metric, views_by_metric, reasons

    def _review_in_editor(self) -> None:
        if self._editor_open or not self._session.accepted_passages:
            return
        self._editor_open = True
        try:
            try:
                tpr = self._stage_evidence()
            except (OSError, ValueError, RuntimeError) as error:
                QMessageBox.warning(self, "Evidence staging failed", str(error))
                return
            cfg = self._frozen_cfg or self._selected_setup_config()
            has_neutral = any(r.role == "neutral" for r in self._session.accepted_passages.values())
            context = "Neutral posture" if has_neutral else "Solo walking"
            metric_passages, metric_views, reasons = self._build_metric_drafts(
                None, context=context
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
                default_context=context,
                default_name=base_name,
                omit_source_archive=True,
                prefer_cfg_orientation=True,
                auto_approve_drafts=True,
            )
            if editor.exec() == QDialog.DialogCode.Accepted and editor.saved_path is not None:
                saved_name = editor.name_input.text().strip()
                if saved_name:
                    self.name_input.setText(saved_name)
                self.saved_path = editor.saved_path
                self.saved_paths = [editor.saved_path]
                self.status_banner.setText(f"Saved calibration: {editor.saved_path.name}")
                self.review_summary.setText(
                    f"Saved calibration: {editor.saved_path.name}\n\n"
                    "You can reopen it anytime from Library -> Edit calibration."
                )
                self.bridge.calibrations_changed.emit()
                if self._embedded:
                    self.accepted.emit()
                else:
                    self.accept()
        finally:
            self._editor_open = False

    def _save_ready_profiles(self, _checked: bool = False, *, auto_triggered: bool = False) -> None:
        try:
            tpr = self._stage_evidence()
            ev = cached_evidence(tpr)
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            if not auto_triggered:
                QMessageBox.warning(self, "Evidence staging failed", str(error))
            return
        cfg = self._frozen_cfg or self._selected_setup_config()
        cal_dir = calibration_directory(self._cfg.data_dir)
        cal_dir.mkdir(parents=True, exist_ok=True)
        base_name = self.name_input.text().strip() or "Studio webcam"
        max_gap = self._session.effective_max_gap()

        # Build samples per view
        passages_list = list(self._session.accepted_passages.values())
        ledger = self._ledger_payload()
        front_samples = metric_samples(ev.frames, cfg, view="front")
        side_samples = ledger_metric_samples(ev.frames, cfg, ledger, view="side")

        has_neutral = any(r.role == "neutral" for r in passages_list)
        has_walking = any(r.role == "walking" for r in passages_list)
        if not (has_neutral or has_walking):
            if not auto_triggered:
                QMessageBox.information(self, "Nothing to save", "No usable holds or walking passes were captured.")
            return
        context = "Neutral posture" if has_neutral else "Solo walking"
        passages_by_metric, views_by_metric, _ = self._build_metric_drafts(None, context=context)
        approved: dict[str, ApprovedRange] = {}
        for mk, passages in passages_by_metric.items():
            view = views_by_metric.get(mk, "side")
            samples_for_mk = side_samples[mk] if view == "side" else front_samples[mk]
            try:
                stats = summarize_examples(
                    samples_for_mk,
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
                                samples_for_mk, holds_4, max_gap, expected_sign=spec.expected_sign
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
            if not auto_triggered:
                QMessageBox.information(self, "Nothing to save", "No usable holds or walking passes were captured.")
            return

        prof_path = calibration_path(cal_dir, base_name)
        capture_profile = dict(ev.capture_profile)
        capture_profile["side"] = cfg.side
        capture_profile["facing"] = cfg.facing
        profile_obj = RangeProfile(
            name=base_name,
            context=context,
            model_hash=ev.model_hash,
            capture_profile=capture_profile,
            source={"dancer_mode": "single", "embedded_evidence": EMBEDDED_EVIDENCE_MEMBER},
            metrics=approved,
            camera_checks={"solo_camera_ledger": ledger},
        )
        try:
            save_range_profile(
                prof_path,
                profile_obj,
                overwrite=True,
                embedded_evidence=tpr,
            )
        except (OSError, ValueError, FileExistsError) as err:
            if not auto_triggered:
                QMessageBox.warning(self, "Could not save calibration", f"{base_name}: {err}")
            return

        self.saved_paths = [prof_path]
        self.saved_path = prof_path
        self.bridge.calibrations_changed.emit()
        self.status_banner.setText(f"Saved calibration: {prof_path.name}")
        if not auto_triggered and not self._embedded:
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
            self._thread.wait(10000)
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
