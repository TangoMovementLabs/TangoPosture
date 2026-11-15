"""Review and replay workspace screen."""

from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Event, Thread
from typing import Any
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QHideEvent, QImage, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ...archive import (
    file_sha256,
    read_analytics,
    read_manifest,
    read_movement_analysis,
    store_analytics,
)
from ...calibration.baseline import load_calibration_file
from ...calibration.models import METRIC_DETAILS, ApprovedRange, profile_from_calibration
from ...movement import render_floor_guide, render_movement_overlay, validate_movement_analysis
from ...references.coach import (
    compare_coach_data,
    interpolate_coach_data,
    render_coach_frame_overlay,
    render_coach_head_alignment,
    render_coach_head_posture,
    render_coach_torso_axis,
)
from ...references.companions import load_report_coach_data, load_report_movement_data
from ...references.constants import MAX_IMPORT_RANGE_SECONDS
from ...replay import VideoReader, time_label
from ...reports import latest_analysis_model_section
from ..components.video_stage import VideoStage
from ..review.analysis import (
    ReviewAnalysis,
    analyse_review_clip,
    render_review_overlays,
    render_review_report,
)
from ..review.playback import VIDEO_SUFFIXES, ReviewPlayback
from ..router import Route, Router
from ..service_bridge import ServiceBridge, created_at_timestamp, format_created_at
from ..theme import (
    COLOR_BASE,
    COLOR_BORDER,
    COLOR_COPPER,
    COLOR_DANGER,
    COLOR_ELEVATED,
    COLOR_SUCCESS,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_WARNING,
)

REPLAY_ERRORS = (OSError, ValueError, RuntimeError, BadZipFile, cv2.error)

_TONE_COLORS = {
    "muted": COLOR_TEXT_MUTED,
    "danger": COLOR_DANGER,
    "warning": COLOR_WARNING,
    "success": COLOR_SUCCESS,
}
_STATE_TONES = {
    "WITHIN_REFERENCE": ("success", "Within approved range"),
    "PENDING": ("warning", "Leaving approved range"),
    "DEVIATION": ("danger", "Outside approved range"),
    "UNKNOWN": ("muted", "Not measurable on this frame"),
}
_DISPLAY_METRICS = (
    ("head_angle", "head_pitch"),
    ("shoulder_elev_pct", "shoulder_clearance"),
    ("axis_curve_smoothed", "hip_offset"),
    ("torso_axis_tilt", "torso_axis_tilt"),
    ("knee_bend_deg", "knee_flexion"),
)
_METRIC_TITLES = (
    ("head_angle", "Head alignment"),
    ("shoulder_elev_pct", "Shoulder height"),
    ("axis_curve_smoothed", "Torso / hip alignment"),
    ("torso_axis_tilt", "Torso axis"),
    ("knee_bend_deg", "Knee bend"),
)


def _section_label(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setStyleSheet(
        f"color: {COLOR_COPPER}; font-size: 10px; font-weight: 700; letter-spacing: 1px;"
    )
    return label


def _caption_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
    return label


def _rgba(color: str, alpha: float) -> str:
    red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _format_measure(value: float, unit: str) -> str:
    if unit == "degrees":
        return f"{value:.1f}°"
    if unit in {"axis_lengths", "torso_lengths"}:
        return f"{value * 100:+.1f}%"
    return f"{value:.2f}"


class ReviewScreen(QWidget):
    def __init__(
        self,
        bridge: ServiceBridge,
        router: Router,
        parent: QWidget | None = None,
        *,
        playback: ReviewPlayback | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(parent)
        self.bridge = bridge
        self.router = router
        self.current_archive_path: Path | None = None
        self._reopen_path: Path | None = None
        self.is_playing = False
        self.current_time_sec = 0.0
        self.duration_sec = 0.0
        self._playback = playback if playback is not None else ReviewPlayback()
        self._comparison = ReviewPlayback()
        self._clock = clock
        self._started_at = 0.0
        self._playback_rate = 1.0
        self._coach_data: dict[str, Any] | None = None
        self._compared_coach_data: dict[str, Any] | None = None
        self._calibration_error = ""
        self._calibration_targets: dict[str, ApprovedRange] = {}
        self._calibration_name = ""
        self._movement_data: dict[str, Any] | None = None
        self._analysis_events: queue.SimpleQueue[tuple[int, str, object]] = queue.SimpleQueue()
        self._analysis_cancel: Event | None = None
        self._analysis_thread: Thread | None = None
        self._analysis_commit_lock = threading.Lock()
        self._analysis_job_id = 0
        self._active_analysis_job: int | None = None
        self.destroyed.connect(self._playback.close)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)

        # Center Video & Transport
        video_col = QWidget()
        vc_layout = QVBoxLayout(video_col)
        vc_layout.setContentsMargins(0, 0, 0, 0)
        vc_layout.setSpacing(10)

        # Top Bar
        top_bar = QFrame()
        top_bar.setProperty("class", "card")
        tb_layout = QHBoxLayout(top_bar)
        tb_layout.setContentsMargins(12, 8, 12, 8)
        tb_layout.setSpacing(10)

        self.stream_combo = QComboBox()
        self.stream_combo.currentTextChanged.connect(self._on_stream_changed)
        self.stream_combo.setEnabled(False)
        self.segment_combo = QComboBox()
        self.segment_combo.currentIndexChanged.connect(self._on_segment_changed)
        self.segment_combo.setEnabled(False)

        open_btn = QPushButton("Open Library Archive…")
        open_btn.clicked.connect(self._open_file)
        self.relink_btn = QPushButton("Relink Report Video…")
        self.relink_btn.setEnabled(False)
        self.relink_btn.clicked.connect(self._relink_report_video)
        self.export_btn = QPushButton("Export Video…")
        self.export_btn.clicked.connect(self._export_video)
        self.export_btn.setEnabled(False)

        self.analytics_btn = QPushButton("Analyze posture && movement")
        self.analytics_btn.setEnabled(False)
        self.analytics_btn.setToolTip(
            "Analyze the raw video in the background to add synchronized skeleton and posture insights."
        )
        self.analytics_btn.clicked.connect(self._start_posture_analysis)

        self.compare_cb = QCheckBox("Compare reference")
        self.compare_cb.setEnabled(False)
        self.compare_cb.setToolTip("Show the selected reference recording beside this practice.")
        self.compare_cb.toggled.connect(self._set_comparison_visible)
        self.compare_btn = QPushButton("Open Comparison…")
        self.compare_btn.clicked.connect(self._open_comparison)
        self.compare_btn.setEnabled(False)

        tb_layout.addWidget(QLabel("Recording:"))
        tb_layout.addWidget(self.segment_combo)
        tb_layout.addWidget(QLabel("Stream:"))
        tb_layout.addWidget(self.stream_combo)
        tb_layout.addWidget(open_btn)
        tb_layout.addWidget(self.relink_btn)
        tb_layout.addWidget(self.export_btn)
        tb_layout.addWidget(self.compare_btn)
        tb_layout.addWidget(self.compare_cb)
        tb_layout.addStretch()
        vc_layout.addWidget(top_bar)
        self.status_label = QLabel(
            "Open a local video or .tp practice archive. Video only (no audio)."
        )
        self.status_label.setWordWrap(True)
        vc_layout.addWidget(self.status_label)

        # Video Stage
        self.stage_container = QWidget()
        self.stage_layout = QHBoxLayout(self.stage_container)
        self.stage_layout.setContentsMargins(0, 0, 0, 0)

        self.primary_stage = VideoStage("Select a practice archive or open a video to replay.")
        self.stage_layout.addWidget(self.primary_stage, 1)
        self.comparison_stage = VideoStage("Open a reference video or calibration archive to compare.")
        self.comparison_stage.setVisible(False)
        self.stage_layout.addWidget(self.comparison_stage, 1)
        vc_layout.addWidget(self.stage_container, 1)

        # Transport Bar
        transport_card = QFrame()
        transport_card.setProperty("class", "card")
        tr_layout = QHBoxLayout(transport_card)
        tr_layout.setContentsMargins(12, 8, 12, 8)
        tr_layout.setSpacing(12)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setEnabled(False)

        self.step_back_btn = QPushButton("-1 frame")
        self.step_back_btn.clicked.connect(lambda: self._step_frame(-1))
        self.step_back_btn.setEnabled(False)
        self.step_forward_btn = QPushButton("+1 frame")
        self.step_forward_btn.clicked.connect(lambda: self._step_frame(1))
        self.step_forward_btn.setEnabled(False)

        self.time_label = QLabel("00:00.0 / 00:00.0")
        self.time_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")

        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.setEnabled(False)
        self.scrubber.valueChanged.connect(self._on_scrubber_changed)

        tr_layout.addWidget(self.play_btn)
        tr_layout.addWidget(self.step_back_btn)
        tr_layout.addWidget(self.step_forward_btn)
        tr_layout.addWidget(self.scrubber, 1)
        tr_layout.addWidget(self.time_label)
        tr_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        for rate in (0.25, 0.5, 1.0, 1.5, 2.0):
            self.speed_combo.addItem(f"{rate:g}x", rate)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._set_playback_speed)
        tr_layout.addWidget(self.speed_combo)
        vc_layout.addWidget(transport_card)

        self.splitter.addWidget(video_col)

        # Right Inspector Panel
        inspector = QFrame()
        inspector.setProperty("class", "card")
        inspector.setMinimumWidth(300)
        inspector.setMaximumWidth(420)
        self.inspector = inspector
        insp_layout = QVBoxLayout(inspector)
        insp_layout.setContentsMargins(14, 12, 14, 12)

        tabs = QTabWidget()
        self.inspector_tabs = tabs
        # Tab 1: Frame Insights
        insights_tab = QWidget()
        it_layout = QVBoxLayout(insights_tab)
        it_layout.setContentsMargins(4, 12, 4, 12)
        it_layout.setSpacing(8)

        # Analysis
        it_layout.addWidget(_section_label("Analysis"))
        self.analytics_btn.setProperty("class", "primary")
        self.cancel_analysis_btn = QPushButton("Cancel analysis")
        self.cancel_analysis_btn.setEnabled(False)
        self.cancel_analysis_btn.setVisible(False)
        self.cancel_analysis_btn.setToolTip("Stop the running analysis; the archive stays unchanged.")
        self.cancel_analysis_btn.clicked.connect(self._cancel_posture_analysis)
        it_layout.addWidget(self.analytics_btn)
        it_layout.addWidget(self.cancel_analysis_btn)
        self.analysis_status = _caption_label(
            "Analyze a raw video to add synchronized posture insights. Analysis runs in the background."
        )
        it_layout.addWidget(self.analysis_status)
        it_layout.addSpacing(6)

        # Calibration profile
        cal_header = QHBoxLayout()
        cal_header.addWidget(_section_label("Calibration profile"))
        cal_header.addStretch()
        self.reference_sort_combo = QComboBox()
        self.reference_sort_combo.addItem("Name A–Z", "name_asc")
        self.reference_sort_combo.addItem("Name Z–A", "name_desc")
        self.reference_sort_combo.addItem("Newest first", "date_desc")
        self.reference_sort_combo.addItem("Oldest first", "date_asc")
        self.reference_sort_combo.setToolTip("Sort calibration profiles")
        self.reference_sort_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.reference_sort_combo.setMinimumContentsLength(8)
        self.reference_sort_combo.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.reference_sort_combo.currentIndexChanged.connect(lambda _: self.refresh())
        cal_header.addWidget(self.reference_sort_combo)
        it_layout.addLayout(cal_header)
        self.reference_combo = QComboBox()
        self.reference_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.reference_combo.setMinimumContentsLength(12)
        self.reference_combo.currentIndexChanged.connect(self._calibration_changed)
        it_layout.addWidget(self.reference_combo)
        it_layout.addSpacing(6)

        # Overlay layers
        it_layout.addWidget(_section_label("Overlay layers"))
        self.show_coach_cb = QCheckBox("Skeleton")
        self.show_coach_cb.setToolTip(
            "Show the tracked skeleton. With a calibration selected, approved metrics and alerts are drawn too."
        )
        self.show_torso_axis_cb = QCheckBox("Torso axis")
        self.show_torso_axis_cb.setToolTip(
            "Show the pink shoulder-to-hip midpoint line and its angle from image vertical. "
            "This is a measurement, not an alert."
        )
        self.show_head_alignment_cb = QCheckBox("Head alignment")
        self.show_head_alignment_cb.setToolTip(
            "Show the head alignment ear-to-ear reference line and tilt angle."
        )
        self.show_head_posture_cb = QCheckBox("Head posture")
        self.show_head_posture_cb.setToolTip(
            "Show the head posture reference line and angle from vertical."
        )
        self.show_pivots_cb = QCheckBox("Pivot rings")
        self.show_steps_cb = QCheckBox("Step footprints")
        self.lines_only_cb = QCheckBox("Floor lines only")
        self.lines_only_cb.setToolTip("Hide the video and show only the drawn lines on the floor.")
        for checkbox in (self.show_coach_cb, self.show_pivots_cb, self.show_steps_cb):
            checkbox.setChecked(True)
        layers = QGridLayout()
        layers.setHorizontalSpacing(12)
        layers.setVerticalSpacing(6)
        for position, checkbox in enumerate((
            self.show_coach_cb,
            self.show_torso_axis_cb,
            self.show_head_alignment_cb,
            self.show_head_posture_cb,
            self.show_pivots_cb,
            self.show_steps_cb,
            self.lines_only_cb,
        )):
            checkbox.toggled.connect(self._refresh_overlay)
            layers.addWidget(checkbox, position // 2, position % 2)
        it_layout.addLayout(layers)
        self.axis_status = _caption_label("")
        self.axis_status.setVisible(False)
        it_layout.addWidget(self.axis_status)
        it_layout.addSpacing(6)

        # Frame status and metrics
        it_layout.addWidget(_section_label("This frame"))
        self.alerts_text = QLabel("Posture alerts will appear here when a frame is analyzed.")
        self.alerts_text.setWordWrap(True)
        self._set_alert(self.alerts_text.text(), "muted")
        it_layout.addWidget(self.alerts_text)

        metrics_card = QFrame()
        metrics_card.setObjectName("metricsCard")
        metrics_card.setStyleSheet(
            f"QFrame#metricsCard {{ background-color: {COLOR_BASE}; "
            f"border: 1px solid {COLOR_BORDER}; border-radius: 6px; }}"
        )
        metrics_grid = QGridLayout(metrics_card)
        metrics_grid.setContentsMargins(10, 8, 10, 8)
        metrics_grid.setHorizontalSpacing(8)
        metrics_grid.setVerticalSpacing(2)
        metrics_grid.setColumnStretch(1, 1)
        self.metric_values: dict[str, QLabel] = {}
        self.metric_states: dict[str, QLabel] = {}
        self.metric_ranges: dict[str, QLabel] = {}
        for row_index, (key, title) in enumerate(_METRIC_TITLES):
            state_dot = QLabel("●")
            state_dot.setFixedWidth(12)
            name_label = QLabel(title)
            name_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-weight: 500;")
            name_label.setWordWrap(True)
            range_label = QLabel()
            range_label.setWordWrap(True)
            range_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 10px;")
            range_label.setVisible(False)
            value_label = QLabel("Not observed")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if row_index:
                divider = QFrame()
                divider.setFixedHeight(1)
                divider.setStyleSheet(f"background-color: {COLOR_BORDER};")
                metrics_grid.addWidget(divider, row_index * 3 - 1, 0, 1, 3)
            base = row_index * 3
            metrics_grid.addWidget(state_dot, base, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
            metrics_grid.addWidget(name_label, base, 1)
            metrics_grid.addWidget(range_label, base + 1, 1)
            metrics_grid.addWidget(value_label, base, 2, 2, 1, Qt.AlignmentFlag.AlignVCenter)
            self.metric_values[key] = value_label
            self.metric_states[key] = state_dot
            self.metric_ranges[key] = range_label
            self._set_metric_state(key, None)
        it_layout.addWidget(metrics_card)

        self.insights_text = _caption_label(
            "The original video remains unchanged. Analysis follows the same posture thresholds as live practice."
        )
        it_layout.addWidget(self.insights_text)
        it_layout.addStretch()

        # Tab 2: Report Markdown
        report_tab = QWidget()
        rt_layout = QVBoxLayout(report_tab)
        rt_layout.setContentsMargins(8, 10, 8, 10)
        self.report_view = QTextBrowser()
        self.report_view.setReadOnly(True)
        self.report_view.setPlaceholderText("No archived report is loaded.")
        self.report_view.setOpenLinks(False)
        self.report_view.setOpenExternalLinks(False)
        self.report_view.anchorClicked.connect(self._on_report_link)
        report_actions = QHBoxLayout()
        self.coach_btn = QPushButton("Open Report in AI Coach")
        self.coach_btn.setEnabled(False)
        self.coach_btn.clicked.connect(self._open_report_in_coach)
        report_actions.addWidget(self.coach_btn)
        report_actions.addStretch()
        rt_layout.addLayout(report_actions)
        rt_layout.addWidget(self.report_view)

        insights_scroll = QScrollArea()
        insights_scroll.setWidgetResizable(True)
        insights_scroll.setFrameShape(QFrame.Shape.NoFrame)
        insights_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        insights_scroll.setWidget(insights_tab)
        tabs.addTab(insights_scroll, "Insights")
        tabs.addTab(report_tab, "Report")
        insp_layout.addWidget(tabs)

        self.splitter.addWidget(inspector)
        self.splitter.setSizes([720, 340])

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self._analysis_timer = QTimer(self)
        self._analysis_timer.setInterval(100)
        self._analysis_timer.timeout.connect(self._poll_posture_analysis)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._release)
        self.refresh()
        self.bridge.config_changed.connect(lambda _cfg: self.refresh())

    def has_active_jobs(self) -> bool:
        return self._analysis_thread is not None and self._analysis_thread.is_alive()

    def refresh(self) -> None:
        selected = self.reference_combo.currentData()
        sort_mode = self.reference_sort_combo.currentData()
        references = [
            reference
            for reference in self.bridge.list_references()
            if reference.path.suffix.lower() == ".tpc" and reference.path.is_file()
        ]
        if sort_mode == "name_asc":
            references.sort(key=lambda ref: (ref.name.casefold(), str(ref.path)))
        elif sort_mode == "name_desc":
            references.sort(key=lambda ref: (ref.name.casefold(), str(ref.path)), reverse=True)
        elif sort_mode == "date_asc":
            references.sort(key=lambda ref: created_at_timestamp(ref.created_at, ref.path))
        elif sort_mode == "date_desc":
            references.sort(key=lambda ref: -created_at_timestamp(ref.created_at, ref.path))

        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        self.reference_combo.addItem("No calibration (visuals only)", None)
        for reference in references:
            date_label = format_created_at(reference.created_at, reference.path)
            label = f"{reference.name} ({date_label})" if date_label != "Recent" else reference.name
            self.reference_combo.addItem(label, str(reference.path))
            self.reference_combo.setItemData(
                self.reference_combo.count() - 1,
                f"{reference.name} · Created {date_label}",
                Qt.ItemDataRole.ToolTipRole,
            )
        index = self.reference_combo.findData(selected)
        if selected is not None and index < 0:
            self.reference_combo.addItem(f"Unavailable calibration: {Path(selected).name}", selected)
            index = self.reference_combo.count() - 1
        self.reference_combo.setCurrentIndex(max(0, index))
        self.reference_combo.blockSignals(False)
        calibration_changed = self.reference_combo.currentData() != selected
        dynamic_stream = self.stream_combo.currentText() in (
            "Raw video", "Local video", "Source video"
        )
        self._calibration_changed(refresh_overlay=calibration_changed or dynamic_stream)

    def _update_calibration_comparison(self) -> None:
        self._compared_coach_data = None
        self._calibration_error = ""
        self._calibration_targets = {}
        self._calibration_name = ""
        selected = self.reference_combo.currentData()
        if self._coach_data is None:
            return
        try:
            data = self._coach_data
            if isinstance(selected, str):
                path = Path(selected)
                reference = load_calibration_file(str(path), path.parent)
                if reference is None:
                    raise ValueError(f"Calibration not found: {path.name}")
                data = compare_coach_data(data, self.bridge.config, reference)
                range_profile = profile_from_calibration(reference)
                if range_profile is not None:
                    self._calibration_targets = dict(range_profile.metrics)
                    self._calibration_name = range_profile.name
            self._compared_coach_data = interpolate_coach_data(
                data, self.bridge.config,
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            self._calibration_error = f"Calibration comparison unavailable: {error}"
            self._compared_coach_data = self._coach_data
            self.analysis_status.setText(self._calibration_error)
            self.bridge.status_message.emit(self._calibration_error)

    def _calibration_changed(self, *_args: object, refresh_overlay: bool = True) -> None:
        self._update_calibration_comparison()
        if not self._calibration_error and self._coach_data is not None:
            self.analysis_status.setText(
                f"Comparing cached measurements with {self.reference_combo.currentText()}."
                if self.reference_combo.currentData() is not None else
                "No calibration selected; showing tracking and movement without posture alerts."
            )
        if refresh_overlay:
            self._refresh_overlay()

    def _display_coach_data(self) -> dict[str, Any] | None:
        return self._compared_coach_data

    def _using_range_targets(self) -> bool:
        return self.reference_combo.currentData() is not None

    def load_archive(
        self, path: Path, *, segment_index: int = 0, timestamp: float | None = None
    ) -> None:
        """Open either a local video or a validated practice archive."""
        self._reopen_path = None
        self._release()
        try:
            self._playback.open(path)
            self.current_archive_path = path
            if self._playback.archive is not None:
                self.segment_combo.blockSignals(True)
                self.segment_combo.addItems(
                    [f"Recording {index:02d}" for index in range(self._playback.segment_count)]
                )
                self.segment_combo.blockSignals(False)
                self.segment_combo.setEnabled(True)
                self._on_segment_changed(0)
                if 0 <= segment_index < self.segment_combo.count():
                    self.segment_combo.setCurrentIndex(segment_index)
            elif self._playback._report is not None:
                self.report_view.setMarkdown(
                    self._link_report_timestamps(self._playback.report)
                )
                self._coach_data = load_report_coach_data(self._playback._report)
                self._update_calibration_comparison()
                self._movement_data = load_report_movement_data(self._playback._report)
                self.coach_btn.setEnabled(True)
                self._set_relink_availability()
                if self._playback.variants:
                    self._populate_streams()
                else:
                    self.status_label.setText(
                        f"{path.name} — report loaded; no linked video is available. "
                        f"{self._playback.warning}"
                    )
            else:
                self._populate_streams()
            self._update_analysis_controls()
            if timestamp is not None:
                self._show_frame(self._playback.frame_for_timestamp(timestamp))
            self.compare_btn.setEnabled(self._playback.reader is not None)
        except REPLAY_ERRORS as error:
            self._release()
            self._playback_error(error)

    def _open_file(self) -> None:
        path = self.bridge.choose_library_archive(
            self, "Open Library archive", self.reference_sort_combo.currentData()
        )
        if path is not None:
            self.load_archive(path)

    def _set_relink_availability(self) -> None:
        report = self._playback._report
        archive_backed = self.current_archive_path is not None and (
            self.current_archive_path.suffix.lower() in (".tp", ".tpc", ".tpr")
        )
        self.relink_btn.setEnabled(
            report is not None
            and not archive_backed
            and any(not video.is_file() for video in report.videos.values())
        )

    def _relink_report_video(self) -> None:
        report = self._playback._report
        path = self.current_archive_path
        if report is None or path is None or path.suffix.lower() != ".md":
            self.bridge.status_message.emit("Open a standalone report before relinking its video.")
            return
        video, _ = QFileDialog.getOpenFileName(
            self,
            "Relink report to a local video",
            "",
            f"Video files ({' '.join('*' + suffix for suffix in VIDEO_SUFFIXES)})",
        )
        if not video:
            return
        from ...references.comparison import relink_report_video

        try:
            outcome = relink_report_video(report, Path(video))
        except (OSError, ValueError, RuntimeError) as error:
            self.bridge.status_message.emit(f"Could not relink report video: {error}")
            return
        self.load_archive(path)
        self.bridge.status_message.emit(outcome.message)

    def _is_dynamic_stream(self) -> bool:
        return self.stream_combo.currentText() in ("Raw video", "Local video", "Source video")

    def _export_primary_size(self, export_reader: VideoReader) -> tuple[int, int]:
        default_size = (export_reader.width, export_reader.height)
        if not self._is_dynamic_stream():
            return default_size
        coach_data = self._display_coach_data()
        lines_only = (
            self.lines_only_cb.isChecked()
            or not self._using_range_targets()
            or bool(self._calibration_error)
        )
        if coach_data is None or not self.show_coach_cb.isChecked() or lines_only:
            return default_size
        frames = coach_data.get("frames")
        if not isinstance(frames, dict) or not frames:
            return default_size
        sample_index: int | None = None
        for key, entry in frames.items():
            try:
                candidate = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= candidate < export_reader.count:
                if sample_index is None:
                    sample_index = candidate
                if isinstance(entry, dict) and (
                    entry.get("poses") or entry.get("measurement_poses")
                ):
                    sample_index = candidate
                    break
        if sample_index is None:
            return default_size
        fps = export_reader.fps if export_reader.fps > 0 else 30.0
        sample_time = self._playback.timestamp_for_frame(sample_index)
        sample_frame = self._render_primary_frame(
            export_reader.frame(sample_index),
            sample_index,
            sample_time,
            fps=fps,
            update_status=False,
        )
        return (sample_frame.shape[1], sample_frame.shape[0])

    def _export_video(self) -> None:
        reader = self._playback.reader
        if reader is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export selected video stream",
            reader.path.name,
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;All files (*)",
        )
        if not path:
            return
        destination = Path(path).expanduser().resolve()
        comparison_reader = self._comparison.reader if self.compare_cb.isChecked() else None
        source_paths = {reader.path.resolve()}
        if comparison_reader is not None:
            source_paths.add(comparison_reader.path.resolve())
        if destination in source_paths:
            self.bridge.status_message.emit("The selected video is already at that location.")
            return

        export_reader: VideoReader | None = None
        comp_reader: VideoReader | None = None
        writer: cv2.VideoWriter | None = None
        temp_path: Path | None = None
        try:
            export_reader = VideoReader(reader.path)
            if export_reader.count <= 0:
                raise ValueError("Selected video has no decodable frames.")
            if comparison_reader is not None:
                comp_reader = VideoReader(comparison_reader.path)

            primary_size = self._export_primary_size(export_reader)
            primary_w, primary_h = primary_size
            comp_size: tuple[int, int] | None = None
            out_size = primary_size
            if comp_reader is not None and comp_reader.count > 0:
                comp_w = max(
                    1,
                    round(comp_reader.width * (primary_h / max(1, comp_reader.height))),
                )
                comp_size = (comp_w, primary_h)
                out_size = (primary_w + comp_w, primary_h)

            destination.parent.mkdir(parents=True, exist_ok=True)
            suffix = destination.suffix.lower() or ".mp4"
            with NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.stem}_",
                suffix=suffix,
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
            fourcc = cv2.VideoWriter.fourcc(*("MJPG" if suffix == ".avi" else "mp4v"))
            fps = export_reader.fps if export_reader.fps > 0 else 30.0
            writer = cv2.VideoWriter(str(temp_path), fourcc, fps, out_size)
            if not writer.isOpened():
                raise OSError(f"Cannot create video writer for {destination}")

            for frame_index in range(export_reader.count):
                frame_time_sec = self._playback.timestamp_for_frame(frame_index)
                rendered = self._render_primary_frame(
                    export_reader.frame(frame_index),
                    frame_index,
                    frame_time_sec,
                    fps=fps,
                    update_status=False,
                )
                if (rendered.shape[1], rendered.shape[0]) != primary_size:
                    rendered = cv2.resize(rendered, primary_size)
                if comp_reader is not None and comp_size is not None:
                    comp_index = min(
                        max(self._comparison.frame_for_time(frame_time_sec), 0),
                        comp_reader.count - 1,
                    )
                    comp_frame = comp_reader.frame(comp_index)
                    if (comp_frame.shape[1], comp_frame.shape[0]) != comp_size:
                        comp_frame = cv2.resize(comp_frame, comp_size)
                    rendered = np.hstack((rendered, comp_frame))
                rendered = self._render_metrics_box(rendered, frame_index)
                writer.write(rendered)

            writer.release()
            writer = None
            temp_path.replace(destination)
            temp_path = None
        except REPLAY_ERRORS as error:
            self.bridge.status_message.emit(f"Could not export video: {error}")
            return
        finally:
            if writer is not None:
                writer.release()
            if export_reader is not None:
                export_reader.close()
            if comp_reader is not None:
                comp_reader.close()
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        self.bridge.status_message.emit(f"Exported video to {destination}")

    def _on_segment_changed(self, index: int) -> None:
        self._cancel_posture_analysis()
        self._active_analysis_job = None
        self._coach_data = None
        self._movement_data = None
        self._reset_transport()
        self.report_view.clear()
        self.stream_combo.blockSignals(True)
        self.stream_combo.clear()
        self.stream_combo.blockSignals(False)
        self.stream_combo.setEnabled(False)
        try:
            self._playback.select_segment(index)
            archive = self._playback.archive
            if archive is not None:
                analytics = read_analytics(archive.path, index)
                if analytics is not None:
                    self._coach_data = analytics.get("coach_data")
                    self._movement_data = analytics.get("movement_data")
                movement = read_movement_analysis(archive.path, index)
                if movement is not None and self._movement_data is None:
                    self._movement_data = validate_movement_analysis(
                        movement, expected_segment_index=index
                    )
            self._update_calibration_comparison()
            self._refresh_archive_report()
            self.coach_btn.setEnabled(self._playback._report is not None)
            self._set_relink_availability()
            self._populate_streams()
            self._update_analysis_controls()
        except REPLAY_ERRORS as error:
            self._playback_error(error)

    def _populate_streams(self) -> None:
        self.stream_combo.blockSignals(True)
        self.stream_combo.clear()
        self.stream_combo.addItems(list(self._playback.variants))
        for raw in ("Raw video", "Local video", "Source video"):
            if raw in self._playback.variants:
                self.stream_combo.setCurrentText(raw)
                break
        self.stream_combo.blockSignals(False)
        self.stream_combo.setEnabled(bool(self._playback.variants))
        if not self._playback.variants:
            raise ValueError("This recording has no raw or overlay video available")
        self._on_stream_changed(self.stream_combo.currentText())

    def _refresh_archive_report(self) -> None:
        archive = self._playback.archive
        index = self._playback.selected_segment_index
        if archive is None or index is None:
            return
        try:
            with ZipFile(archive.path) as stored:
                report = stored.read(archive.manifest["segments"][index]["report"]).decode("utf-8")
        except (KeyError, UnicodeError) as error:
            raise ValueError(f"Cannot read archived analysis report: {error}") from error
        self._playback.report = report
        if self._playback._report is not None:
            self._playback._report.markdown = report
        self.report_view.setMarkdown(self._link_report_timestamps(report))

    def _on_stream_changed(self, label: str) -> None:
        self._reset_transport()
        if label not in ("Raw video", "Local video", "Source video"):
            self.lines_only_cb.blockSignals(True)
            self.lines_only_cb.setChecked(False)
            self.lines_only_cb.blockSignals(False)
        try:
            reader = self._playback.select_variant(label)
            self.duration_sec = self._playback.duration()
            self.scrubber.setRange(0, reader.count - 1)
            self._show_frame(0)
            self._set_timer_interval(reader.fps)
            self.play_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            self.step_back_btn.setEnabled(True)
            self.step_forward_btn.setEnabled(True)
            self.scrubber.setEnabled(True)
            self.compare_btn.setEnabled(True)
            self._update_analysis_controls()
            name = self.current_archive_path.name if self.current_archive_path else ""
            unavailable = [
                stream
                for stream in ("Raw video", "Overlay video")
                if self._playback.archive is not None and stream not in self._playback.variants
            ]
            self.status_label.setText(
                f"{name} — {label}. Video only (no audio)."
                + (
                    f"\nUnavailable in this recording: {', '.join(unavailable)}."
                    if unavailable
                    else ""
                )
                + (
                    f"\nMetadata warning: {self._playback.warning}"
                    if self._playback.warning
                    else ""
                )
            )
        except REPLAY_ERRORS as error:
            self._playback_error(error)

    def _pause(self) -> None:
        self.is_playing = False
        self.timer.stop()
        self.play_btn.setText("▶ Play")

    def _toggle_playback(self) -> None:
        if self.is_playing:
            self._on_tick()
            self._pause()
            return
        if self._playback.reader is None:
            return
        if self.current_time_sec >= self.duration_sec:
            self._on_scrubber_changed(0)
            if self._playback.reader is None:
                return
        self._started_at = self._clock() - self.current_time_sec / self._playback_rate
        self.is_playing = True
        self.play_btn.setText("⏸ Pause")
        self.timer.start()

    def _on_tick(self) -> None:
        reader = self._playback.reader
        if not self.is_playing or reader is None:
            return
        elapsed = max(0.0, self._clock() - self._started_at) * self._playback_rate
        index = self._playback.frame_for_time(elapsed)
        try:
            if index != reader.index:
                self._show_frame(index)
            if elapsed >= self.duration_sec:
                self._pause()
                self.current_time_sec = self.duration_sec
                self._update_time_label()
        except REPLAY_ERRORS as error:
            self._playback_error(error)

    def _on_scrubber_changed(self, value: int) -> None:
        if self._playback.reader is None:
            return
        try:
            self._show_frame(value)
            self._started_at = self._clock() - self.current_time_sec / self._playback_rate
        except REPLAY_ERRORS as error:
            self._playback_error(error)

    def _step_frame(self, offset: int) -> None:
        reader = self._playback.reader
        if reader is None:
            return
        if self.is_playing:
            self._on_tick()
        if reader is not self._playback.reader:
            return
        self._pause()
        try:
            self._show_frame(reader.index + offset)
            self._started_at = self._clock() - self.current_time_sec / self._playback_rate
        except REPLAY_ERRORS as error:
            self._playback_error(error)

    def _set_playback_speed(self, index: int) -> None:
        if self.is_playing:
            self._on_tick()
        self._playback_rate = float(self.speed_combo.itemData(index) or 1.0)
        reader = self._playback.reader
        if reader is not None:
            self._set_timer_interval(reader.fps)
            if self.is_playing:
                self._started_at = self._clock() - self.current_time_sec / self._playback_rate

    def _set_timer_interval(self, fps: float) -> None:
        self.timer.setInterval(max(1, min(100, round(1000 / (fps * self._playback_rate)))))

    def _on_report_link(self, url: QUrl) -> None:
        if url.scheme() != "seek":
            return
        try:
            seconds = float(url.path())
            frame = self._playback.frame_for_report_time(seconds)
            self._pause()
            self._show_frame(frame)
        except REPLAY_ERRORS as error:
            self.status_label.setText(f"Could not seek to report timestamp: {error}")

    def _open_report_in_coach(self) -> None:
        path = self.current_archive_path
        index = self._playback.selected_segment_index
        if path is None:
            self.bridge.status_message.emit("Open a saved report first.")
            return
        if path.suffix.lower() == ".md":
            self.router.navigate(Route.COACH, path)
        elif index is not None and path.suffix.lower() in (".tp", ".tpc", ".tpr"):
            self.router.navigate(Route.COACH, {"archive_path": path, "segment_index": index})
        else:
            self.bridge.status_message.emit("Open a saved report first.")

    def _open_comparison(self) -> None:
        path = self.bridge.choose_library_archive(
            self, "Compare Library archive", self.reference_sort_combo.currentData()
        )
        if path is None:
            return
        self._load_comparison(path)

    def _load_comparison(self, path: Path) -> None:
        self._comparison.close()
        try:
            self._comparison.open(path)
            if self._comparison.archive is not None:
                index = self._playback.selected_segment_index or 0
                self._comparison.select_segment(min(index, self._comparison.segment_count - 1))
            label = "Raw video" if "Raw video" in self._comparison.variants else next(
                iter(self._comparison.variants), ""
            )
            self._comparison.select_variant(label)
            self.compare_cb.setEnabled(True)
            self.compare_cb.setChecked(True)
            self._sync_comparison_frame()
            self.compare_cb.setToolTip(f"Comparing with {Path(path).name}.")
        except REPLAY_ERRORS as error:
            self._comparison.close()
            self.comparison_stage.clear_frame()
            self.compare_cb.setChecked(False)
            self.compare_cb.setEnabled(False)
            self.status_label.setText(f"Comparison unavailable: {error}")

    def _set_comparison_visible(self, visible: bool) -> None:
        self.comparison_stage.setVisible(visible and self._comparison.reader is not None)
        if visible:
            self._sync_comparison_frame()

    def _sync_comparison_frame(self) -> None:
        reader = self._comparison.reader
        if reader is None:
            self.comparison_stage.clear_frame()
            return
        try:
            frame = reader.frame(self._comparison.frame_for_time(self.current_time_sec))
            height, width = frame.shape[:2]
            image = QImage(
                frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888
            ).copy()
            if image.isNull():
                raise ValueError("Cannot display decoded comparison frame")
            self.comparison_stage.set_frame(image)
        except REPLAY_ERRORS as error:
            self._comparison.close()
            self.comparison_stage.clear_frame()
            self.compare_cb.setChecked(False)
            self.compare_cb.setEnabled(False)
            self.status_label.setText(f"Comparison unavailable: {error}")

    @staticmethod
    def _link_report_timestamps(markdown: str) -> str:
        time_pattern = re.compile(r"`(\d{2,}:\d{2}\.\d)(?: \([^)]+\))?`")

        def link_time(match: re.Match[str]) -> str:
            label = match.group(1)
            minutes, seconds = label.split(":", 1)
            value = int(minutes) * 60 + float(seconds)
            return f"[{label}](seek:{value:.6f})"

        return time_pattern.sub(link_time, markdown)

    def _render_primary_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        frame_time_sec: float,
        *,
        fps: float,
        update_status: bool = False,
    ) -> np.ndarray:
        if not self._is_dynamic_stream():
            if update_status:
                self.axis_status.clear()
                self.axis_status.setVisible(False)
            return frame

        rendered = frame.copy()
        if self.lines_only_cb.isChecked():
            rendered = render_floor_guide(rendered)
        coach_data = self._display_coach_data()
        if coach_data is not None and self.show_coach_cb.isChecked():
            rendered = render_coach_frame_overlay(
                rendered.copy(),
                coach_data,
                frame_index,
                self.bridge.config,
                verbose_skeleton=True,
                show_vertical_axis=False,
                show_footer=False,
                lines_only=(
                    self.lines_only_cb.isChecked()
                    or not self._using_range_targets()
                    or bool(self._calibration_error)
                ),
            )

        status_messages: list[str] = []
        measurement_layers = (
            (self.show_torso_axis_cb, "Axis", render_coach_torso_axis),
            (self.show_head_alignment_cb, "Head alignment", render_coach_head_alignment),
            (self.show_head_posture_cb, "Head posture", render_coach_head_posture),
        )
        frames = coach_data.get("frames") if coach_data is not None else None
        entry = frames.get(str(frame_index)) if isinstance(frames, dict) else None
        for checkbox, label, renderer in measurement_layers:
            if not checkbox.isChecked():
                continue
            if (
                coach_data is None
                or not isinstance(entry, dict)
                or "measurement_poses" not in entry
            ):
                message = (
                    f"Raw {label.lower()} landmarks are unavailable. "
                    "Analyze the source video again."
                )
                if message not in status_messages:
                    status_messages.append(message)
                continue
            try:
                rendered = renderer(
                    rendered,
                    coach_data,
                    frame_index,
                    self.bridge.config,
                    show_labels=not self.lines_only_cb.isChecked(),
                )
            except (TypeError, ValueError) as error:
                message = f"{label} unavailable: {error}"
                if message not in status_messages:
                    status_messages.append(message)

        if update_status:
            self.axis_status.setText("\n".join(status_messages))
            self.axis_status.setVisible(bool(status_messages))

        return render_movement_overlay(
            rendered,
            self._movement_data,
            frame_index,
            fps,
            frame_timestamp=frame_time_sec,
            show_pivots=self.show_pivots_cb.isChecked(),
            show_steps=self.show_steps_cb.isChecked(),
            step_history_seconds=self.bridge.config.replay_step_history_seconds,
        )

    def _show_frame(self, index: int) -> None:
        reader = self._playback.reader
        if reader is None:
            return
        frame = reader.frame(index)
        self._update_frame_insights(reader.index)
        self.current_time_sec = self._playback.timestamp_for_frame(reader.index)
        frame = self._render_primary_frame(
            frame,
            reader.index,
            self.current_time_sec,
            fps=reader.fps,
            update_status=True,
        )
        height, width = frame.shape[:2]
        image = QImage(
            frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888
        ).copy()
        if image.isNull():
            raise ValueError("Cannot display decoded video frame")
        self.primary_stage.set_frame(image)
        self._sync_comparison_frame()
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(reader.index)
        self.scrubber.blockSignals(False)
        self._update_time_label()

    def _refresh_overlay(self) -> None:
        reader = self._playback.reader
        if reader is None:
            return
        if self.stream_combo.currentText() not in (
            "Raw video", "Local video", "Source video"
        ):
            for label in ("Raw video", "Local video", "Source video"):
                if label in self._playback.variants:
                    self.stream_combo.setCurrentText(label)
                    return
            self.lines_only_cb.blockSignals(True)
            self.lines_only_cb.setChecked(False)
            self.lines_only_cb.blockSignals(False)
            self.analysis_status.setText(
                "Tracking and calibration comparison require a raw/source video; "
                "baked overlay pixels cannot be removed."
            )
            self._reset_transport()
            self.primary_stage.clear_frame()
            return
        try:
            self._show_frame(reader.index)
        except REPLAY_ERRORS as error:
            self._playback_error(error)

    def _analysis_video_path(self) -> Path | None:
        for label in ("Raw video", "Local video", "Source video"):
            path = self._playback.variants.get(label)
            if path is not None and path.is_file():
                return path
        return None

    def _update_analysis_controls(self) -> None:
        thread = self._analysis_thread
        running = thread is not None and thread.is_alive()
        self.analytics_btn.setEnabled(self._analysis_video_path() is not None and not running)
        self.cancel_analysis_btn.setEnabled(
            self._analysis_cancel is not None and not self._analysis_cancel.is_set()
        )
        self.cancel_analysis_btn.setVisible(running or self.cancel_analysis_btn.isEnabled())

    def _start_posture_analysis(self) -> None:
        if self._analysis_thread is not None:
            self._poll_posture_analysis()
        reader = self._playback.reader
        video_path = self._analysis_video_path()
        if reader is None or video_path is None:
            self.analysis_status.setText("Open a raw or local video before analyzing posture.")
            return
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            return
        selected_out = reader.duration
        if selected_out <= 0:
            self.analysis_status.setText("This video has no analyzable duration.")
            return
        if selected_out > MAX_IMPORT_RANGE_SECONDS + 1e-6:
            self.analysis_status.setText(
                f"Full-recording analysis is limited to {MAX_IMPORT_RANGE_SECONDS:.0f} seconds. "
                "Choose a shorter recording; the archive was not changed."
            )
            return

        self._analysis_job_id += 1
        job_id = self._analysis_job_id
        cancel_event = threading.Event()
        self._analysis_cancel = cancel_event
        self._active_analysis_job = job_id
        self.analysis_status.setText("Preparing posture analysis…")
        self.cancel_analysis_btn.setEnabled(True)
        self.cancel_analysis_btn.setVisible(True)
        self.analytics_btn.setEnabled(False)
        self._analysis_timer.start()
        cfg = self.bridge.config.model_copy(deep=True)
        archive_path = self.current_archive_path
        segment_index = self._playback.selected_segment_index

        archive_lease = (
            self._playback.archive.acquire()
            if self._playback.archive is not None
            else None
        )

        def report_progress(current: int, total: int, message: str) -> None:
            self._analysis_events.put((job_id, "progress", (current, total, message)))

        def work() -> None:
            try:
                video_sha256 = file_sha256(video_path)
                coach_data = analyse_review_clip(
                    video_path,
                    cfg,
                    selected_out=selected_out,
                    progress_callback=report_progress,
                    cancel_check=cancel_event.is_set,
                )
                if (
                    not cancel_event.is_set()
                    and archive_path is not None
                    and archive_path.suffix.lower() in (".tp", ".tpr")
                    and segment_index is not None
                ):
                    if coach_data.result is None:
                        raise ValueError("Analysis did not produce the data required for a report")
                    result = coach_data.result
                    if result.source_sha256 != video_sha256:
                        raise ValueError("Source video changed during analysis; archive was not changed")
                    timestamps = [float(frame["timestamp"]) for frame in result.frame_log]
                    coach_data.movement_data.update(
                        segment_index=segment_index, source_member=video_path.name,
                        source_sha256=video_sha256,
                    )
                    previous = read_analytics(archive_path, segment_index) or {}
                    provenance = previous.get("source_provenance") or {}
                    if not isinstance(provenance, dict):
                        raise ValueError("Invalid archived source provenance")
                    manifest = read_manifest(archive_path)
                    provenance = {**provenance, "title": manifest.get("name")}
                    report = render_review_report(result, cfg, segment_index, provenance)
                    report_progress(90, 100, "Rendering updated overlay videos")
                    with TemporaryDirectory(prefix="tango-review-analysis-") as temporary:
                        overlay, lines = render_review_overlays(
                            video_path, Path(temporary), coach_data, cfg, cancel_event.is_set,
                        )
                        if file_sha256(video_path) != video_sha256:
                            raise ValueError("Source video changed during overlay rendering")
                        payload = {
                            "coach_data": coach_data.coach_data,
                            "movement_data": coach_data.movement_data, "max_gap": cfg.max_gap,
                            "model_hash": coach_data.coach_data.get("model_hash"),
                            "capture_profile": coach_data.coach_data.get("capture_profile"),
                            "frame_timestamps": timestamps,
                            "frames": [],
                        }
                        with self._analysis_commit_lock:
                            cancelled = cancel_event.is_set() or self._active_analysis_job != job_id
                            if not cancelled:
                                store_analytics(
                                    archive_path, segment_index, payload,
                                    expected_video_sha256=video_sha256, report_text=report,
                                    overlay_path=overlay, lines_path=lines, frame_log=result.frame_log,
                                    metadata_updates={
                                        "frame_timestamps": timestamps,
                                        "analysis_mode": result.analysis_mode,
                                        "is_imported_reference": True,
                                        "source": f"import:{video_path.name}",
                                        "source_sha256": video_sha256,
                                        "title": provenance.get("title") or result.title,
                                        "clip_in": 0.0, "clip_out": result.selected_out,
                                        "readiness": result.readiness.to_dict(),
                                        "reference_summary": None,
                                        "coverage": {
                                            key: value.to_dict()
                                            for key, value in result.coverage.items()
                                        },
                                        "soundtrack": result.soundtrack.to_metadata_dict(),
                                        "capture_profile": result.manifest.capture_profile,
                                    },
                                )
                            event = "cancelled" if cancelled else "done"
                            self._analysis_events.put((job_id, event, coach_data))
                        return
                with self._analysis_commit_lock:
                    cancelled = cancel_event.is_set() or self._active_analysis_job != job_id
                    event = "cancelled" if cancelled else "done"
                    self._analysis_events.put((job_id, event, coach_data))
            except Exception as error:  # noqa: BLE001 - surface background analysis errors in the UI
                with self._analysis_commit_lock:
                    event = (
                        "cancelled"
                        if cancel_event.is_set() or self._active_analysis_job != job_id
                        else "error"
                    )
                    self._analysis_events.put(
                        (job_id, event, f"{type(error).__name__}: {error}")
                    )
            finally:
                if archive_lease is not None:
                    archive_lease.close()

        self._analysis_thread = Thread(
            target=work,
            name="qt-review-posture-analysis",
            daemon=True,
        )
        self._analysis_thread.start()

    def _cancel_posture_analysis(self) -> None:
        with self._analysis_commit_lock:
            cancel_event = self._analysis_cancel
            if cancel_event is None or cancel_event.is_set():
                return
            cancel_event.set()
        self.analysis_status.setText("Canceling posture analysis…")
        self.cancel_analysis_btn.setEnabled(False)

    def _poll_posture_analysis(self) -> None:
        thread = self._analysis_thread
        finished = thread is not None and not thread.is_alive()
        while True:
            try:
                job_id, event, payload = self._analysis_events.get_nowait()
            except queue.Empty:
                break
            if job_id != self._active_analysis_job:
                continue
            if event == "progress" and isinstance(payload, tuple):
                current, total, message = payload
                self.analysis_status.setText(f"{message} ({current}/{total}%)")
            elif event == "done" and isinstance(payload, ReviewAnalysis):
                self._coach_data = payload.coach_data
                self._movement_data = payload.movement_data
                self._update_calibration_comparison()
                try:
                    if self._playback.archive is not None:
                        path = self._playback.archive.path
                        index = self._playback.selected_segment_index or 0
                        frame = self._playback.reader.index if self._playback.reader else 0
                        selected_stream = self.stream_combo.currentText()
                        self._analysis_cancel = None
                        self._active_analysis_job = None
                        self.load_archive(path, segment_index=index)
                        if (
                            selected_stream in self._playback.variants
                            and self.stream_combo.currentText() != selected_stream
                        ):
                            self.stream_combo.setCurrentText(selected_stream)
                        if self._playback.reader is not None:
                            self._show_frame(min(frame, self._playback.reader.count - 1))
                    else:
                        original = self._playback.report
                        section = latest_analysis_model_section(
                            payload.coach_data.get("model_hash"), has_original_report=bool(original),
                        )
                        section += "\n\nThis Review analysis is not saved with the standalone source."
                        self.report_view.setMarkdown(self._link_report_timestamps(
                            f"{section}\n\n{original}" if original else section
                        ))
                except (OSError, ValueError, UnicodeError, KeyError, BadZipFile) as error:
                    self.bridge.status_message.emit(f"Could not refresh the analysis report: {error}")
                frames = payload.coach_data.get("frames")
                frame_count = len(frames) if isinstance(frames, dict) else 0
                saved_note = (
                    "Previous analysis, report and overlay videos were replaced and saved. "
                    if self.current_archive_path is not None
                    and self.current_archive_path.suffix.lower() in (".tp", ".tpr") else ""
                )
                self.analysis_status.setText(
                    f"Analysis ready: {frame_count} frames, "
                    f"{len(payload.movement_data.get('pivots', []))} pivots, "
                    f"{len(payload.movement_data.get('steps', []))} steps. "
                    f"{saved_note}The review overlay and insights follow the current playback frame."
                )
                if self._calibration_error:
                    self.analysis_status.setText(self._calibration_error)
                reader = self._playback.reader
                if reader is not None:
                    self._show_frame(reader.index)
            elif event == "cancelled":
                self.analysis_status.setText("Posture analysis canceled.")
            elif event == "error":
                self.analysis_status.setText(f"Posture analysis failed: {payload}")
                self.bridge.status_message.emit(f"Posture analysis failed: {payload}")

        if thread is not None and finished:
            thread.join()
            self._analysis_thread = None
            self._analysis_cancel = None
            self._active_analysis_job = None
            self._analysis_timer.stop()
            self._update_analysis_controls()

    def _set_alert(self, text: str, tone: str) -> None:
        color = _TONE_COLORS[tone]
        background = COLOR_ELEVATED if tone == "muted" else _rgba(color, 0.14)
        text_color = COLOR_TEXT_MUTED if tone == "muted" else COLOR_TEXT_PRIMARY
        self.alerts_text.setText(text)
        self.alerts_text.setStyleSheet(
            f"background-color: {background}; border-left: 3px solid {color}; "
            f"border-radius: 4px; padding: 8px 10px; color: {text_color}; font-weight: 600;"
        )

    def _set_metric_state(self, key: str, state: str | None, detail: str = "") -> None:
        tone, label = _STATE_TONES.get(state or "", ("muted", "No calibration comparison"))
        color = _TONE_COLORS[tone] if state else COLOR_BORDER
        dot = self.metric_states[key]
        dot.setStyleSheet(f"color: {color}; font-size: 12px;")
        tooltip = f"{label}\n{detail}" if detail else label
        dot.setToolTip(tooltip)
        self.metric_values[key].setToolTip(tooltip)
        value_color = COLOR_TEXT_MUTED if state in (None, "UNKNOWN") else _TONE_COLORS[tone]
        if state == "WITHIN_REFERENCE" or state is None:
            value_color = COLOR_TEXT_PRIMARY
        self.metric_values[key].setStyleSheet(
            f"color: {value_color}; font-weight: 600;"
        )

    def _reset_metrics(self, text: str = "Not observed") -> None:
        for key, value in self.metric_values.items():
            value.setText(text)
            self.metric_ranges[key].setVisible(False)
            self._set_metric_state(key, None)

    def _frame_insights_snapshot(self, frame_index: int) -> dict[str, Any]:
        values = {key: "Not observed" for key, _ in _METRIC_TITLES}
        ranges: dict[str, str | None] = {key: None for key, _ in _METRIC_TITLES}
        states: dict[str, tuple[str | None, str, bool]] = {
            key: (None, "", False) for key, _ in _METRIC_TITLES
        }
        idle_note = (
            "Analyze the raw video to track the skeleton and measure posture. "
            "Existing full-lines recordings retain their original overlay."
        )
        if self._calibration_error:
            return {
                "values": values,
                "ranges": ranges,
                "states": states,
                "alert": (self._calibration_error, "warning"),
                "details": "Posture comparison is unavailable.",
            }
        data = self._display_coach_data()
        if data is None:
            return {
                "values": values,
                "ranges": ranges,
                "states": states,
                "alert": ("No synchronized posture analysis is loaded.", "muted"),
                "details": idle_note,
            }

        frames = data.get("frames")
        entry = frames.get(str(frame_index)) if isinstance(frames, dict) else None
        if not isinstance(entry, dict):
            return {
                "values": values,
                "ranges": ranges,
                "states": states,
                "alert": ("No pose data is available for this frame.", "muted"),
                "details": idle_note,
            }

        def metric(name: str, suffix: str, scale: float = 1.0, signed: bool = False) -> str:
            value = entry.get(name)
            if not isinstance(value, int | float) or isinstance(value, bool):
                return "Not observed"
            number = float(value) * scale
            format_spec = "+.1f" if signed else ".1f"
            return f"{number:{format_spec}}{suffix}"

        values = {
            "head_angle": metric("head_angle", "°", signed=True),
            "shoulder_elev_pct": metric("shoulder_elev_pct", "%", signed=True),
            "axis_curve_smoothed": metric("axis_curve_smoothed", "%", scale=100.0, signed=True),
            "torso_axis_tilt": metric("torso_axis_tilt", "°"),
            "knee_bend_deg": metric("knee_bend_deg", "°"),
        }

        if not self._using_range_targets():
            return {
                "values": values,
                "ranges": ranges,
                "states": states,
                "alert": ("Select a calibration profile to see posture alerts.", "muted"),
                "details": "Tracking and movement are shown without posture coaching.",
            }

        rows = {row["metric"]: row for row in entry.get("calibration_rows", [])}
        for display_key, key in _DISPLAY_METRICS:
            row = rows.get(key)
            target = self._calibration_targets.get(key)
            if row is None:
                values[display_key] = "Not in profile"
                states[display_key] = (
                    None,
                    "This calibration did not approve this metric.",
                    True,
                )
                continue
            unit = str(row.get("unit", ""))
            measured_value = row.get("smoothed")
            values[display_key] = (
                _format_measure(float(measured_value), unit)
                if isinstance(measured_value, int | float) and not isinstance(measured_value, bool)
                else "Not measured"
            )
            if target is not None:
                ranges[display_key] = (
                    f"Approved {_format_measure(target.lower, unit)} to "
                    f"{_format_measure(target.upper, unit)}"
                )
            states[display_key] = (str(row.get("state")), str(row.get("reason") or ""), False)

        message = entry.get("message")
        poses = entry.get("measurement_poses", entry.get("poses"))
        unknown = [row for row in rows.values() if row.get("state") == "UNKNOWN"]
        if isinstance(message, str) and message.strip():
            alert = (
                "\n".join(f"• {part.strip()}" for part in message.split(";") if part.strip()),
                "danger",
            )
        elif not poses:
            alert = ("No dancer detected; posture alerts are unavailable.", "warning")
        elif unknown:
            alert = (
                "Some approved-range measurements are unavailable (tracking/view).",
                "warning",
            )
        else:
            alert = ("No active posture alert on this frame.", "success")

        details = [f"Compared with {self._calibration_name or self.reference_combo.currentText()}."]
        reasons: dict[str, list[str]] = {}
        for row in unknown:
            label = METRIC_DETAILS.get(row["metric"], ("", row["metric"]))[1]
            reasons.setdefault(str(row.get("reason") or "Not visible"), []).append(label)
        for reason, labels in reasons.items():
            details.append(f"Not measured — {', '.join(labels)}: {reason}.")

        return {
            "values": values,
            "ranges": ranges,
            "states": states,
            "alert": alert,
            "details": "\n".join(details),
        }

    def _frame_metrics_lines(self, frame_index: int) -> list[str]:
        values = self._frame_insights_snapshot(frame_index)["values"]
        return [
            f"{title}: {str(values.get(key, 'Not observed')).replace('°', ' deg')}"
            for key, title in _METRIC_TITLES
        ]

    def _render_metrics_box(self, image: np.ndarray, frame_index: int) -> np.ndarray:
        output = image.copy()
        height, width = output.shape[:2]
        if height <= 0 or width <= 0:
            return output
        lines = self._frame_metrics_lines(frame_index)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = min(0.65, max(0.22, min(width, height) / 1400.0))
        thickness = 1
        sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        text_w = max((size[0] for size in sizes), default=1)
        text_h = max((size[1] for size in sizes), default=5)
        pad_x = max(2, round(10 * scale))
        pad_y = max(2, round(8 * scale))
        line_gap = max(2, round(6 * scale))
        line_step = text_h + line_gap
        natural_w = text_w + 2 * pad_x
        natural_h = len(lines) * text_h + max(0, len(lines) - 1) * line_gap + 2 * pad_y
        box_w = min(natural_w, max(8, int(width * 0.48)))
        box_h = min(natural_h, max(8, int(height * 0.48)))
        patch = np.full((box_h, box_w, 3), 255, dtype=np.uint8)
        for index, line in enumerate(lines):
            y = pad_y + text_h + index * line_step
            if y - text_h >= box_h - 1:
                break
            cv2.putText(patch, line, (pad_x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        margin = max(2, min(12, width // 40, height // 40))
        x0 = max(0, width - margin - box_w)
        y0 = max(0, height - margin - box_h)
        output[y0 : y0 + box_h, x0 : x0 + box_w] = patch
        return output

    def _update_frame_insights(self, frame_index: int) -> None:
        self._reset_metrics()
        snapshot = self._frame_insights_snapshot(frame_index)
        for key, value_text in snapshot["values"].items():
            self.metric_values[key].setText(value_text)
        for key, range_text in snapshot["ranges"].items():
            range_label = self.metric_ranges[key]
            if range_text is not None:
                range_label.setText(range_text)
                range_label.setVisible(True)
        for key, (state, detail, muted_value) in snapshot["states"].items():
            self._set_metric_state(key, state, detail)
            if muted_value:
                self.metric_values[key].setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
        alert_text, alert_tone = snapshot["alert"]
        self._set_alert(alert_text, alert_tone)
        self.insights_text.setText(snapshot["details"])

    def _update_time_label(self) -> None:
        self.time_label.setText(
            f"{time_label(self.current_time_sec)} / {time_label(self.duration_sec)}"
        )

    def _reset_transport(self) -> None:
        self._pause()
        self.current_time_sec = self.duration_sec = 0.0
        self.play_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.step_back_btn.setEnabled(False)
        self.step_forward_btn.setEnabled(False)
        self.scrubber.setEnabled(False)
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, 0)
        self.scrubber.blockSignals(False)
        self.primary_stage.clear_frame()
        self.primary_stage.clear_badge()
        self._update_time_label()

    def _playback_error(self, error: Exception) -> None:
        self._reset_transport()
        self._playback.close_reader()
        self._comparison.close()
        self.compare_cb.setChecked(False)
        self.compare_cb.setEnabled(False)
        self.comparison_stage.clear_frame()
        self.comparison_stage.setVisible(False)
        self.status_label.setText(f"Replay unavailable: {error}")

    def _release(self) -> None:
        self._cancel_posture_analysis()
        self._active_analysis_job = None
        self._coach_data = None
        self._compared_coach_data = None
        self._calibration_error = ""
        self._movement_data = None
        self._reopen_path = None
        self._reset_transport()
        self._playback.close()
        self._comparison.close()
        self.current_archive_path = None
        self.compare_btn.setEnabled(False)
        self.compare_cb.setChecked(False)
        self.compare_cb.setEnabled(False)
        self.relink_btn.setEnabled(False)
        self.comparison_stage.clear_frame()
        self.comparison_stage.setVisible(False)
        for combo in (self.stream_combo, self.segment_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)
            combo.setEnabled(False)
        self.report_view.clear()
        self.coach_btn.setEnabled(False)
        self._update_frame_insights(0)
        self.analysis_status.setText(
            "Analyze a raw video to add synchronized posture insights. Analysis runs in the background."
        )
        self._update_analysis_controls()
        self.status_label.setText("Playback closed. Open a video or .tp archive to replay.")

    def hideEvent(self, event: QHideEvent) -> None:
        path = self.current_archive_path
        self._release()
        self._reopen_path = path
        super().hideEvent(event)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        path, self._reopen_path = self._reopen_path, None
        if path is not None:
            self.load_archive(path)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._release()
        super().closeEvent(event)
