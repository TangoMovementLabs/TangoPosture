"""Live practice and recording workspace screen."""

from __future__ import annotations

import queue
import re
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...archive import (
    _archive_lock,
    _rewrite_members,
    practice_directory,
    practice_path,
    read_analytics,
    store_analytics,
)
from ...config import Config
from ...core.entities import ReferenceSummary
from ...core.session_state import (
    SessionState,
    abort_unsaved,
    accept_reference,
    choose_source,
    close,
    pause_scoring,
    ready,
    resume_scoring,
    start_recording,
    start_scoring,
    stop_and_save,
    stop_scoring,
)
from ...core.types import SourceKind
from ...couple import classify_opening_window
from ...domain import FrameObservation
from ..components.beat_strip import BeatStrip
from ..components.video_stage import VideoStage
from ..practice.worker import (
    PracticeThread,
    QtPracticeDisplay,
    QtPracticeWorker,
    apply_discovered_cameras,
    discover_cameras,
    populate_camera_combo,
)
from ..router import Router
from ..service_bridge import ServiceBridge, numpy_to_qimage
from ..theme import (
    COLOR_BASE,
    COLOR_BORDER,
    COLOR_BURGUNDY,
    COLOR_COPPER,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
)


class PracticeScreen(QWidget):
    stopped = Signal()

    def __init__(
        self,
        bridge: ServiceBridge,
        router: Router,
        parent: QWidget | None = None,
        *,
        camera_discoverer: Callable[..., list[tuple[int, str]]] = discover_cameras,
    ):
        super().__init__(parent)
        self.bridge = bridge
        self.router = router
        self._camera_discoverer = camera_discoverer
        self._last_bridge_camera = self.bridge.config.camera
        self.state = SessionState()
        self._references: dict[str, ReferenceSummary] = {}
        self._active_reference: ReferenceSummary | None = None
        self._calibrating = False
        self._calibration_candidate = False
        self._calibration_saved = False
        self._thread: QThread | None = None
        self._worker: QtPracticeWorker | None = None
        self._display: QtPracticeDisplay | None = None
        self._save_pending = False
        self._worker_error: str | None = None
        self._last_practice_path: Path | None = None
        self._recorded_frames: list[FrameObservation] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        screen_title = QLabel("Self Practice")
        screen_title.setProperty("class", "title")
        screen_sub = QLabel(
            "Live camera pose detection using the Lite model and .tp practice recording."
        )
        screen_sub.setProperty("class", "subtitle")
        title_col.addWidget(screen_title)
        title_col.addWidget(screen_sub)
        header.addLayout(title_col, 1)
        layout.addLayout(header)

        setup_card = QFrame()
        setup_card.setProperty("class", "card")
        setup_layout = QVBoxLayout(setup_card)
        setup_layout.setContentsMargins(14, 10, 14, 10)
        setup_layout.setSpacing(8)

        primary_row = QHBoxLayout()
        primary_row.setSpacing(10)

        primary_row.addWidget(QLabel("Practice name:"))
        self.session_title = QLineEdit(
            f"Practice_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        self.session_title.setAccessibleName("Practice archive name")
        primary_row.addWidget(self.session_title, 1)

        primary_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Auto-detect", "auto")
        self.mode_combo.addItem("Single dancer", "single")
        self.mode_combo.addItem("Couple", "couple")
        mode_idx = self.mode_combo.findData(self.bridge.config.dancer_analysis.default_dancer_mode)
        if mode_idx >= 0:
            self.mode_combo.setCurrentIndex(mode_idx)
        primary_row.addWidget(self.mode_combo)

        self.preview_btn = QPushButton("Start Preview")
        self.preview_btn.setProperty("class", "primary")
        self.preview_btn.clicked.connect(self._toggle_preview)
        primary_row.addWidget(self.preview_btn)

        self.record_btn = QPushButton("Start Record (R)")
        self.record_btn.setProperty("class", "accent")
        self.record_btn.clicked.connect(self._toggle_recording)
        primary_row.addWidget(self.record_btn)
        setup_layout.addLayout(primary_row)

        camera_row = QHBoxLayout()
        camera_row.setSpacing(10)
        camera_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setAccessibleName("Practice camera")
        self.camera_combo.setMinimumWidth(220)
        self.camera_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        if self.bridge.discovered_cameras:
            apply_discovered_cameras(
                self.camera_combo,
                self.bridge.discovered_cameras,
                fallback_index=self.bridge.config.camera,
            )
        else:
            self.camera_combo.addItem(
                f"Camera {self.bridge.config.camera}", self.bridge.config.camera
            )
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        camera_row.addWidget(self.camera_combo)

        self.scan_cameras_btn = QPushButton("Scan cameras")
        self.scan_cameras_btn.clicked.connect(self._scan_cameras)
        camera_row.addWidget(self.scan_cameras_btn)
        camera_row.addStretch()
        setup_layout.addLayout(camera_row)

        # Keep hidden compatibility widgets without cluttering the UI
        self.source_combo = QComboBox()
        self.source_combo.addItem("Camera", "camera")
        self.source_combo.setVisible(False)
        self.video_path = QLineEdit()
        self.video_path.setVisible(False)
        self.browse_video_btn = QPushButton("Browse...")
        self.browse_video_btn.setVisible(False)
        self.reference_combo = QComboBox()
        self.reference_combo.setVisible(False)
        self.confirm_camera_cb = QCheckBox()
        self.confirm_camera_cb.setVisible(False)
        self.reference_name = QLineEdit("My calibration")
        self.reference_name.setVisible(False)
        layout.addWidget(setup_card)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(10)

        self.stage = VideoStage(
            placeholder_text="Click Start Preview to open the camera with Lite pose detection."
        )
        center_layout.addWidget(self.stage, 1)

        controls_card = QFrame()
        controls_card.setProperty("class", "card")
        controls_outer = QVBoxLayout(controls_card)
        controls_outer.setContentsMargins(14, 10, 14, 10)
        controls_outer.setSpacing(6)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(14)

        self.auto_song_cb = QCheckBox("Automatically record songs")
        self.auto_song_cb.toggled.connect(self._set_auto_recording)
        controls_layout.addWidget(self.auto_song_cb)
        self.skeleton_btn = QCheckBox("Skeleton (V)")
        self.skeleton_btn.setChecked(True)
        self.skeleton_btn.toggled.connect(lambda checked: self._request_key("v"))
        controls_layout.addWidget(self.skeleton_btn)
        self.axis_btn = QCheckBox("Vertical axis (Y)")
        self.axis_btn.setChecked(True)
        self.axis_btn.toggled.connect(lambda checked: self._request_key("y"))
        controls_layout.addWidget(self.axis_btn)
        controls_layout.addStretch()

        self.status_label = QLabel("Camera is off. Click Start Record (R) or Start Preview.")
        self.status_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
        controls_layout.addWidget(self.status_label)

        self.close_btn = QPushButton("Close Preview")
        self.close_btn.clicked.connect(self._close_preview)
        controls_layout.addWidget(self.close_btn)
        controls_outer.addLayout(controls_layout)

        overlay_opts = QHBoxLayout()
        overlay_opts.setSpacing(14)
        self.head_align_cb = QCheckBox("Head horizontal alignment")
        self.head_align_cb.setChecked(True)
        self.head_align_cb.toggled.connect(self._sync_display_options)
        overlay_opts.addWidget(self.head_align_cb)

        self.shoulders_cb = QCheckBox("Shoulders height")
        self.shoulders_cb.setChecked(True)
        self.shoulders_cb.toggled.connect(self._sync_display_options)
        overlay_opts.addWidget(self.shoulders_cb)

        self.role_cb = QCheckBox("Role detection")
        self.role_cb.setChecked(True)
        self.role_cb.toggled.connect(self._sync_display_options)
        overlay_opts.addWidget(self.role_cb)

        self.embrace_cb = QCheckBox("Embrace type (Open / Close)")
        self.embrace_cb.setChecked(True)
        self.embrace_cb.toggled.connect(self._sync_display_options)
        overlay_opts.addWidget(self.embrace_cb)
        overlay_opts.addStretch()
        controls_outer.addLayout(overlay_opts)

        center_layout.addWidget(controls_card)
        splitter.addWidget(center)

        inspector = QFrame()
        inspector.setProperty("class", "card")
        inspector.setMinimumWidth(290)
        inspector.setMaximumWidth(370)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(16, 14, 16, 14)
        inspector_layout.setSpacing(12)

        title = QLabel("Session Status")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        subtitle = QLabel("Lite model pose tracking & recording status (no posture alerts).")
        subtitle.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        subtitle.setWordWrap(True)
        self.beat_strip = BeatStrip()
        self.notes_box = QLabel("Start Preview to begin pose detection. Press Record (R) to save a .tp file.")
        self.notes_box.setWordWrap(True)
        self.notes_box.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.notes_box.setStyleSheet(
            f"background-color: {COLOR_BASE}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 6px; padding: 10px; color: {COLOR_TEXT_PRIMARY}; font-size: 12px;"
        )
        inspector_layout.addWidget(title)
        inspector_layout.addWidget(subtitle)
        inspector_layout.addWidget(self.beat_strip)
        inspector_layout.addWidget(self.notes_box)

        notes_header = QLabel("Practice Notes")
        notes_header.setStyleSheet("font-size: 14px; font-weight: bold;")
        inspector_layout.addWidget(notes_header)
        self.manual_notes = QPlainTextEdit()
        self.manual_notes.setPlaceholderText("Add your own notes for this practice session…")
        self.manual_notes.setStyleSheet(
            f"background-color: {COLOR_BASE}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 6px; padding: 8px; color: {COLOR_TEXT_PRIMARY}; font-size: 12px;"
        )
        inspector_layout.addWidget(self.manual_notes, 1)

        self.save_notes_btn = QPushButton("Save Notes to .tp")
        self.save_notes_btn.clicked.connect(self._save_manual_notes)
        inspector_layout.addWidget(self.save_notes_btn)

        splitter.addWidget(inspector)
        splitter.setSizes([740, 320])

        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(33)
        self._frame_timer.timeout.connect(self._drain_display)
        self._shortcuts: list[QShortcut] = []
        for key, action in (
            ("R", self._toggle_recording),
            ("V", self.skeleton_btn.toggle),
            ("Y", self.axis_btn.toggle),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda action=action: self._run_shortcut(action))
            self._shortcuts.append(shortcut)
        self.bridge.config_changed.connect(self._on_config_changed)
        self.bridge.cameras_discovered.connect(self._on_cameras_discovered)
        self.refresh()
        self._update_source_controls()
        self._sync_controls()

    @property
    def worker_running(self) -> bool:
        return self._thread is not None

    def refresh(self) -> None:
        selected_id = self.reference_combo.currentData()
        references = self.bridge.list_references()
        self._references = {reference.id: reference for reference in references}
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        self.reference_combo.addItem("No reference (record only)", None)
        for reference in references:
            self.reference_combo.addItem(
                f"{reference.name} - {reference.side} side / {reference.facing} facing",
                reference.id,
            )
        if selected_id in self._references:
            self.reference_combo.setCurrentIndex(
                self.reference_combo.findData(selected_id)
            )
        self.reference_combo.blockSignals(False)
        self._reference_changed()

    def begin_calibration(self) -> None:
        if self.worker_running:
            self.bridge.status_message.emit(
                "Close the active preview before starting a new calibration."
            )
            return
        if self.source_combo.currentData() != "camera":
            self.bridge.status_message.emit(
                "New camera calibration requires the camera source. Select Camera first."
            )
            return
        name = self.reference_name.text().strip()
        if not name:
            self.status_label.setText("Enter a name for the new reference.")
            return
        from ..calibration.guided import GuidedCalibration

        dialog = GuidedCalibration(self.bridge, self, profile_name=name)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh()
            self.bridge.calibrations_changed.emit()
            self.status_label.setText(f"Approved calibration saved: {dialog.saved_path}")

    def stop_for_navigation(self) -> None:
        if self.worker_running and self._display is not None:
            self.status_label.setText(
                "Closing the practice preview before leaving; an active recording will be saved."
            )
            self._display.request("q")

    def _selected_reference(self) -> ReferenceSummary | None:
        identifier = self.reference_combo.currentData()
        return self._references.get(identifier) if isinstance(identifier, str) else None

    def _reference_changed(self, *_args: object) -> None:
        self.confirm_camera_cb.setChecked(False)
        self._sync_controls()

    def _update_source_controls(self, *_args: object) -> None:
        is_video = self.source_combo.currentData() == "video"
        self.video_path.setEnabled(is_video and not self.worker_running)
        self.browse_video_btn.setEnabled(is_video and not self.worker_running)

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a practice video",
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;All files (*)",
        )
        if path:
            self.video_path.setText(path)

    def _toggle_preview(self) -> None:
        if self.worker_running:
            self._close_preview()
        else:
            self._start_preview()

    def _start_preview(self, *, auto_record: bool = False) -> None:
        name = self.session_title.text().strip()
        try:
            practice_path(practice_directory(self.bridge.config.data_dir), name)
        except ValueError as error:
            self.status_label.setText(str(error))
            return

        self._launch_worker(
            reference=None,
            video=None,
            practice_name=name,
            auto_record=auto_record,
        )

    def _selected_camera(self) -> int:
        data = self.camera_combo.currentData()
        return self.bridge.config.camera if data is None else int(data)

    def _selected_practice_config(self) -> Config:
        cam_idx = self._selected_camera()
        base_cfg = self.bridge.config
        if cam_idx == base_cfg.camera:
            return base_cfg
        return Config.model_validate({
            **base_cfg.model_dump(exclude={"axis_curve"}),
            "camera": cam_idx,
        })

    def _scan_cameras(self) -> None:
        if self.worker_running:
            return
        discovered = populate_camera_combo(
            self.camera_combo,
            self._camera_discoverer,
            fallback_index=self.bridge.config.camera,
        )
        self.bridge.set_discovered_cameras(discovered)
        if not discovered:
            self.status_label.setText("No cameras detected.")
        elif len(discovered) == 1:
            self.status_label.setText(
                f"Camera scan complete: {self.camera_combo.currentText()}."
            )
        else:
            self.status_label.setText(
                f"Found {len(discovered)} cameras. Selected: {self.camera_combo.currentText()}."
            )

    def _on_cameras_discovered(self, cameras: object) -> None:
        if self.worker_running or not isinstance(cameras, list):
            return
        apply_discovered_cameras(
            self.camera_combo,
            cameras,
            fallback_index=self.bridge.config.camera,
        )

    def _on_camera_changed(self) -> None:
        if not self.worker_running and self.camera_combo.currentText():
            self.status_label.setText(f"Selected {self.camera_combo.currentText()}.")

    def _on_config_changed(self, cfg: object = None) -> None:
        active_cfg = cfg if isinstance(cfg, Config) else self.bridge.config
        if active_cfg.camera == self._last_bridge_camera:
            return
        self._last_bridge_camera = active_cfg.camera
        self.camera_combo.blockSignals(True)
        try:
            pos = self.camera_combo.findData(active_cfg.camera)
            if pos < 0:
                self.camera_combo.addItem(f"Camera {active_cfg.camera}", active_cfg.camera)
                pos = self.camera_combo.findData(active_cfg.camera)
            if pos >= 0:
                self.camera_combo.setCurrentIndex(pos)
        finally:
            self.camera_combo.blockSignals(False)

    def _launch_worker(
        self,
        *,
        reference: ReferenceSummary | None = None,
        video: Path | None = None,
        practice_name: str | None = None,
        calibration_only: bool = False,
        reference_name: str = "My reference",
        auto_record: bool = False,
    ) -> None:
        if self.worker_running:
            return
        display = QtPracticeDisplay(capture_evidence=True)
        display.suppress_alerts = True
        if self.auto_song_cb.isChecked() and not calibration_only:
            display.auto_recording.set()
        if auto_record and not calibration_only:
            display.recording.set()
        worker = QtPracticeWorker(
            self._selected_practice_config(),
            display,
            reference_id=reference.id if reference else None,
            reference_path=reference.path if reference else None,
            video=video,
            practice_name=practice_name,
            calibration_only=calibration_only,
            reference_name=reference_name,
        )
        worker.setParent(self)
        thread = PracticeThread(worker, self)
        worker.failed.connect(self._worker_failed)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._worker_finished)
        self._display = display
        self._sync_display_options()
        self._recorded_frames.clear()
        self._worker = worker
        self._thread = thread
        self._active_reference = reference
        self._calibrating = calibration_only
        self._calibration_candidate = False
        self._calibration_saved = False
        self._save_pending = False
        self._worker_error = None
        if self.state.phase.value == "preview":
            self.state = close(self.state)
        source = SourceKind.VIDEO if video is not None else SourceKind.CAMERA
        self.state = ready(choose_source(SessionState(), source))
        if auto_record and not calibration_only:
            self.state = start_recording(self.state)
        if reference:
            self.state = accept_reference(self.state, reference.id)
        self.stage.clear_frame()
        self.notes_box.setText(
            "Starting camera and pose processing..." if not calibration_only
            else "Starting camera calibration..."
        )
        self.status_label.setText(
            "Starting local practice pipeline. No network service is used."
        )
        self._sync_controls()
        self._frame_timer.start()
        thread.start()

    def _calibration_action(self) -> None:
        if not self.worker_running:
            self.begin_calibration()
        elif self._calibrating and self._display is not None:
            self._display.request("c")
            self.status_label.setText(
                "Calibration started. Hold a steady side-on position until the capture completes."
            )

    def _accept_calibration(self) -> None:
        if self._display is None or not self._calibration_candidate:
            return
        self._display.request("a")
        self._calibration_candidate = False
        self.status_label.setText("Saving the accepted reference archive...")

    def _start_scoring(self) -> None:
        if self._display is None or self._calibrating:
            return
        if self._active_reference is None:
            self.status_label.setText(
                "Select a saved reference, confirm camera placement, and restart preview before scoring."
            )
            return
        if not self._active_reference.scoring_ready:
            self.status_label.setText("Comparison only: complete Library > Check camera before scoring.")
            return
        if self.state.scoring.value != "off":
            return
        self._display.request("s")
        self.state = start_scoring(self.state)
        self.status_label.setText("Scoring started against the selected reference.")
        self._sync_controls()

    def _toggle_pause(self) -> None:
        if self._display is None:
            return
        if self.state.scoring.value == "active":
            self._display.request("p")
            self.state = pause_scoring(self.state)
            self.status_label.setText("Scoring paused; video preview and recording continue.")
        elif self.state.scoring.value == "paused":
            self._display.request("p")
            self.state = resume_scoring(self.state)
            self.status_label.setText("Scoring resumed.")
        self._sync_controls()

    def _stop_scoring(self) -> None:
        if self._display is None or self.state.scoring.value == "off":
            return
        self._display.request("t")
        self.state = stop_scoring(self.state)
        self.status_label.setText("Scoring stopped; any active video recording continues.")
        self._sync_controls()

    def _toggle_recording(self) -> None:
        if self._calibrating:
            return
        if not self.worker_running:
            self._start_preview(auto_record=True)
            return
        if self._display is None:
            return
        if self.state.recording:
            self._stop_and_save()
            return
        self._recorded_frames.clear()
        self._display.recording.set()
        self.state = start_recording(self.state)
        self.status_label.setText("Recording started...")
        self._sync_controls()

    def _stop_and_save(self) -> None:
        if self._display is None or not self.state.recording:
            return
        self._save_pending = True
        self._display.request("e")
        self.status_label.setText("Saving recording segment to the practice archive...")
        self._sync_controls()

    def _abort_recording(self) -> None:
        if self._display is None or not self.state.recording:
            return
        answer = QMessageBox.question(
            self,
            "Abort current recording?",
            "Discard the unsaved current segment? Previously saved segments will be preserved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer is QMessageBox.StandardButton.Yes:
            self._display.request("x")
            self.status_label.setText(
                "Aborting current recording; previously saved archive segments are preserved."
            )
            self._sync_controls()

    def _close_preview(self) -> None:
        if self._display is None:
            return
        if self.state.recording:
            answer = QMessageBox.question(
                self,
                "Save and close preview?",
                "The active recording will be saved before the camera closes.",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer is not QMessageBox.StandardButton.Save:
                return
        self._display.request("q")
        self.status_label.setText("Closing preview and saving any active recording...")
        self._sync_controls()

    def _set_auto_recording(self, enabled: bool) -> None:
        if self._display is not None:
            if enabled:
                self._display.auto_recording.set()
            else:
                self._display.auto_recording.clear()

    def _sync_display_options(self) -> None:
        if self._display is None:
            return
        self._display.show_head_alignment = self.head_align_cb.isChecked()
        self._display.show_shoulders_height = self.shoulders_cb.isChecked()
        self._display.show_role_detection = self.role_cb.isChecked()
        self._display.show_embrace_type = self.embrace_cb.isChecked()

    def _request_key(self, key: str) -> None:
        if self._display is not None and not self._calibrating:
            self._display.request(key)

    @staticmethod
    def _run_shortcut(action: Callable[[], None]) -> None:
        focused = QApplication.focusWidget()
        while focused is not None:
            if isinstance(focused, QLineEdit):
                return
            focused = focused.parentWidget()
        action()

    def _drain_display(self) -> None:
        display = self._display
        if display is None:
            return
        try:
            frame = display.frames.get_nowait()
        except queue.Empty:
            frame = None
        if frame is not None:
            try:
                self.stage.set_frame(numpy_to_qimage(frame))
            except (ValueError, RuntimeError) as error:
                self.status_label.setText(f"Could not display camera frame: {error}")
                display.request("q")

        if isinstance(display.notes, str) and display.notes:
            filtered = "\n".join(
                line for line in display.notes.splitlines()
                if "alert" not in line.lower() and "deviation" not in line.lower()
            )
            self.notes_box.setText(filtered or "Pose detection active (Lite model)")
            beat = re.search(r"Beat (\d+)/8 \((\d+(?:\.\d+)?) BPM\)", display.notes)
            if beat:
                self.beat_strip.update_beat(float(beat.group(2)), int(beat.group(1)), True)
            elif "Waiting for song to begin" in display.notes:
                self.beat_strip.update_beat(120.0, 1, False)

        while True:
            try:
                state = display.recording_states.get_nowait()
            except queue.Empty:
                break
            if state == "recording":
                if not self.state.recording:
                    self.state = start_recording(
                        self.state, max_auto=self.bridge.config.max_recordings
                    )
                self.stage.set_badge("RECORDING", COLOR_BURGUNDY)
            elif state == "ready":
                if self.state.recording:
                    self._save_pending = True
                self.stage.set_badge("SAVING", COLOR_COPPER)
            elif state == "idle":
                if self.state.recording:
                    self.state = abort_unsaved(self.state)
                self._save_pending = False
                self.stage.clear_badge()
                self.status_label.setText("No recording was active; preview remains open.")
                self._sync_controls()
            elif state == "limit":
                if self.state.recording:
                    self.state = abort_unsaved(self.state)
                self._save_pending = False
                self.status_label.setText("The configured recording limit has been reached.")
                self._sync_controls()

        while True:
            try:
                obs = display.observations.get_nowait()
            except queue.Empty:
                break
            if self.state.recording:
                self._recorded_frames.append(obs)

        while True:
            try:
                path, index = display.saved_segments.get_nowait()
            except queue.Empty:
                break
            if self.state.recording:
                self.state = stop_and_save(self.state)
            self._save_pending = False
            self._last_practice_path = path
            resolved_mode = self._resolve_and_store_dancer_mode(path, index)
            if self.manual_notes.toPlainText().strip():
                self._save_manual_notes()
            self.status_label.setText(
                f"Recording {index:02d} saved to {path.name} ({resolved_mode})."
            )
            self.stage.clear_badge()
            self._sync_controls()

        while True:
            try:
                path = display.reports.get_nowait()
            except queue.Empty:
                break
            self.bridge.status_message.emit(f"Saved practice archive: {path.name}")

        while True:
            try:
                candidate_name = display.calibration_candidates.get_nowait()
            except queue.Empty:
                break
            self._calibration_candidate = True
            self.status_label.setText(
                f"Calibration '{candidate_name}' is ready. Accept it to save the reference."
            )

        while True:
            try:
                path = display.calibration_archives.get_nowait()
            except queue.Empty:
                break
            self._calibration_saved = True
            self.status_label.setText(f"Reference saved: {path.name}. Closing camera...")
            display.request("q")

    def _worker_failed(self, message: str) -> None:
        self._worker_error = message
        self.status_label.setText(f"Practice unavailable: {message}")
        self.bridge.status_message.emit(f"Practice unavailable: {message}")

    def _worker_finished(self) -> None:
        self._drain_display()
        display = self._display
        was_aborted = display is not None and display.aborted.is_set()
        if display is not None:
            if display.aborted.is_set() and self.state.recording:
                self.state = abort_unsaved(self.state)
                self._save_pending = False
                self.status_label.setText(
                    "Current recording discarded. Previously saved segments are preserved."
                )
            elif self.state.recording:
                self.state = abort_unsaved(self.state)
                self._save_pending = False
            if self.state.scoring.value != "off":
                self.state = stop_scoring(self.state)
            if self.state.phase.value == "preview":
                self.state = close(self.state)
        self._frame_timer.stop()
        self._display = None
        self._worker = None
        thread = self._thread
        self._thread = None
        self._active_reference = None
        self._calibrating = False
        self._calibration_candidate = False
        if self._calibration_saved:
            self.refresh()
            self.bridge.status_message.emit("Calibration saved and camera released.")
        elif self._worker_error:
            self.status_label.setText(
                f"Practice ended with error: {self._worker_error}. Previously saved segments are preserved."
            )
        elif was_aborted:
            self.status_label.setText(
                "Current recording discarded. Previously saved segments are preserved."
            )
        elif not self.status_label.text().startswith("Practice unavailable:"):
            self.status_label.setText("Practice preview closed.")
        self._worker_error = None
        self.stage.clear_badge()
        self._sync_controls()
        self.stopped.emit()
        if thread is not None:
            thread.deleteLater()

    def _resolve_and_store_dancer_mode(self, path: Path, index: int) -> str:
        requested = str(self.mode_combo.currentData() or "auto")
        if requested in ("single", "couple"):
            resolved = requested
        elif len(self._recorded_frames) >= 2:
            decision = classify_opening_window(
                list(enumerate(self._recorded_frames)),
                self.bridge.config,
                clip_in=self._recorded_frames[0].timestamp,
                clip_out=self._recorded_frames[-1].timestamp,
            )
            resolved = "couple" if decision.resolved_mode == "couple" else "single"
        else:
            resolved = "single"
        self._recorded_frames.clear()
        try:
            from ...archive import PracticeArchive

            with _archive_lock(path):
                archive = PracticeArchive(path)
                try:
                    existing = read_analytics(archive.folder, index) or {}
                    updated = {
                        **existing,
                        "dancer_mode": resolved,
                        "requested_dancer_mode": requested,
                    }
                finally:
                    archive.close()
            store_analytics(path, index, updated)
        except (OSError, ValueError, KeyError):
            pass
        return resolved

    def _save_manual_notes(self) -> None:
        notes_text = self.manual_notes.toPlainText().strip()
        if not notes_text:
            self.status_label.setText("Enter notes before saving.")
            return
        if self._last_practice_path is None or not self._last_practice_path.is_file():
            self.status_label.setText("Notes kept; they will be saved with your next .tp recording.")
            return
        with _archive_lock(self._last_practice_path):
            _rewrite_members(self._last_practice_path, {"notes.md": notes_text.encode("utf-8")})
        self.status_label.setText(f"Practice notes saved to {self._last_practice_path.name}.")

    def _sync_controls(self) -> None:
        running = self.worker_running
        calibrating = running and self._calibrating
        self.preview_btn.setText("Close Preview" if running and not calibrating else "Start Preview")
        self.preview_btn.setEnabled(not calibrating)
        self.session_title.setEnabled(not running)
        self.mode_combo.setEnabled(not running)
        self.camera_combo.setEnabled(not running)
        self.scan_cameras_btn.setEnabled(not running)
        self.auto_song_cb.setEnabled(running and not calibrating)
        active = running and not calibrating
        self.record_btn.setEnabled(not self._save_pending and not calibrating)
        self.record_btn.setText(
            "Saving..." if self._save_pending else
            "Stop Record (R)" if self.state.recording else "Start Record (R)"
        )
        self.close_btn.setEnabled(active)
        self.skeleton_btn.setEnabled(active)
        self.axis_btn.setEnabled(active)
