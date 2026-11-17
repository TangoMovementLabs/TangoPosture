"""Review and replay workspace screen."""

from __future__ import annotations

import copy
import inspect
import queue
import re
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Event, Thread
from typing import Any
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QHideEvent, QImage, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
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
    set_active_dancer_run,
    store_analytics,
    update_couple_annotations,
)
from ...calibration.baseline import load_calibration_file
from ...calibration.measurements import measure_ranges
from ...calibration.models import METRIC_DETAILS, ApprovedRange, profile_from_calibration
from ...calibration.normalization import HoldSourceIdentity, resolve_raw_media_sha256
from ...couple.measurements import rebuild_couple_result_with_annotations
from ...couple.models import (
    CoupleAnalysisRequest,
    CoupleAnnotationItem,
    CoupleAnnotations,
    PlannedTransitionMark,
)
from ...couple.overlay import render_couple_frame_overlay, smooth_couple_frame_overlays
from ...movement import render_floor_guide, render_movement_overlay, validate_movement_analysis
from ...pose.recovery import recover_clip_tracking_range
from ...references.coach import (
    _coach_observation,
    compare_coach_data,
    interpolate_coach_data,
    render_coach_frame_overlay,
    render_coach_head_alignment,
    render_coach_head_posture,
    render_coach_torso_axis,
)
from ...references.companions import (
    load_couple_companion,
    load_report_coach_data,
    load_report_movement_data,
    save_coach_companion,
    save_couple_companion,
)
from ...references.constants import MAX_IMPORT_RANGE_SECONDS
from ...references.results import ClipAnalysisResult
from ...replay import VideoReader, time_label
from ...reports import latest_analysis_model_section
from ..calibration import editor as calibration_editor
from ..components.analysis_options import NeedsSelectionDialog
from ..components.video_stage import VideoStage
from ..library.archives import promote_segment_to_tpr, save_clip_result_as_tpr
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


def _format_signed_adjustment(value: float, unit: str) -> str:
    sign = "+" if value >= 0 else "−"
    mag = abs(value)
    if unit == "degrees":
        return f"{sign}{mag:.1f}°"
    if unit in {"axis_lengths", "torso_lengths"}:
        return f"{sign}{mag * 100:.1f}%"
    return f"{sign}{mag:.2f}"


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
        self._calibration_id = ""
        self._bypass_normalization_for_current = False
        self._anchor_start_sec: float | None = None
        self._anchor_end_sec: float | None = None
        self._movement_data: dict[str, Any] | None = None
        self._dancer_analysis: dict[str, Any] | None = None
        self._couple_result: dict[str, Any] | None = None
        self._active_dancer_view_mode: str = "single"
        self._couple_run_id: str | None = None
        self._couple_companion_path: Path | None = None
        self._annotation_undo_stack: list[dict[str, Any]] = []
        self._pending_dancer_request: CoupleAnalysisRequest | None = None
        self._active_run_dancer_request: CoupleAnalysisRequest | None = None
        self._analysis_events: queue.SimpleQueue[tuple[int, str, object]] = queue.SimpleQueue()
        self._analysis_cancel: Event | None = None
        self._analysis_thread: Thread | None = None
        self._analysis_commit_lock = threading.Lock()
        self._analysis_job_id = 0
        self._active_analysis_job: int | None = None
        self._last_cfg_recovery_auto = bool(self.bridge.config.replay_recovery_auto)
        self._last_cfg_show_recovered = bool(self.bridge.config.replay_show_recovered_joints)
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
        self.mark_hold_start_btn = QPushButton("Mark hold start")
        self.mark_hold_start_btn.setToolTip(
            "Save the current playhead time as a hold start anchor for Measurement normalization in Settings."
        )
        self.mark_hold_start_btn.clicked.connect(lambda: self._mark_hold_anchor(is_start=True))
        tr_layout.addWidget(self.mark_hold_start_btn)
        self.mark_hold_end_btn = QPushButton("Mark hold end")
        self.mark_hold_end_btn.setToolTip(
            "Save the current playhead time as a hold end anchor for Measurement normalization in Settings."
        )
        self.mark_hold_end_btn.clicked.connect(lambda: self._mark_hold_anchor(is_start=False))
        tr_layout.addWidget(self.mark_hold_end_btn)
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
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Dancer mode:"))
        self.dancer_mode_combo = QComboBox()
        self.dancer_mode_combo.setObjectName("reviewDancerModeCombo")
        self.dancer_mode_combo.addItem("Auto", "auto")
        self.dancer_mode_combo.addItem("Single", "single")
        self.dancer_mode_combo.addItem("Couple", "couple")
        idx_mode = self.dancer_mode_combo.findData(
            str(self.bridge.config.dancer_analysis.default_dancer_mode)
        )
        if idx_mode >= 0:
            self.dancer_mode_combo.setCurrentIndex(idx_mode)
        self.dancer_mode_combo.currentIndexChanged.connect(
            lambda _: self._update_saved_vs_next_label()
        )
        mode_row.addWidget(self.dancer_mode_combo, 1)
        it_layout.addLayout(mode_row)

        view_run_row = QHBoxLayout()
        view_run_row.addWidget(QLabel("View run:"))
        self.view_run_combo = QComboBox()
        self.view_run_combo.setObjectName("reviewViewRunCombo")
        self.view_run_combo.setEnabled(False)
        self.view_run_combo.currentIndexChanged.connect(self._on_view_run_changed)
        view_run_row.addWidget(self.view_run_combo, 1)
        it_layout.addLayout(view_run_row)

        self.saved_vs_next_label = _caption_label("Saved result: None · Next run: Auto")
        self.saved_vs_next_label.setObjectName("reviewSavedVsNextLabel")
        it_layout.addWidget(self.saved_vs_next_label)

        self.analytics_btn.setProperty("class", "primary")
        self.cancel_analysis_btn = QPushButton("Cancel analysis")
        self.cancel_analysis_btn.setEnabled(False)
        self.cancel_analysis_btn.setVisible(False)
        self.cancel_analysis_btn.setToolTip("Stop the running analysis; the archive stays unchanged.")
        self.cancel_analysis_btn.clicked.connect(self._cancel_posture_analysis)
        it_layout.addWidget(self.analytics_btn)
        self.recover_gaps_cb = QCheckBox("Recover short tracking gaps")
        self.recover_gaps_cb.setChecked(bool(self.bridge.config.replay_recovery_auto))
        self.recover_gaps_cb.setToolTip(
            "When analyzing, run display-only backward pose recovery on short lower-limb dropouts."
        )
        it_layout.addWidget(self.recover_gaps_cb)
        self.recover_tracking_btn = QPushButton("Recover tracking…")
        self.recover_tracking_btn.setEnabled(False)
        self.recover_tracking_btn.setToolTip(
            "Run display-only backward lower-limb tracking recovery on a chosen gap or pivot."
        )
        self.recover_tracking_btn.clicked.connect(self._open_targeted_recovery_dialog)
        it_layout.addWidget(self.recover_tracking_btn)
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
        self.reference_combo.currentIndexChanged.connect(self._on_reference_combo_changed)
        it_layout.addWidget(self.reference_combo)
        self.edit_calibration_btn = QPushButton("Edit calibration…")
        self.edit_calibration_btn.setToolTip(
            "Open the calibration editor to view or edit the currently selected .tpc calibration file."
        )
        self.edit_calibration_btn.setEnabled(False)
        self.edit_calibration_btn.clicked.connect(self._edit_current_calibration)
        it_layout.addWidget(self.edit_calibration_btn)
        self.normalization_settings_btn = QPushButton("Measurement normalization settings…")
        self.normalization_settings_btn.setToolTip(
            "Configure optional offsets mapping this recording onto the selected calibration's scale."
        )
        self.normalization_settings_btn.clicked.connect(self._open_normalization_settings)
        it_layout.addWidget(self.normalization_settings_btn)
        self.normalization_recheck_bar = QFrame()
        self.normalization_recheck_bar.setProperty("class", "card")
        self.normalization_recheck_bar.setStyleSheet(
            f"border: 1px solid {COLOR_WARNING}; border-radius: 4px;"
        )
        recheck_layout = QVBoxLayout(self.normalization_recheck_bar)
        recheck_layout.setContentsMargins(8, 6, 8, 6)
        recheck_layout.setSpacing(6)
        self.recheck_status_lbl = QLabel("Needs recheck: stored normalization is stale.")
        self.recheck_status_lbl.setWordWrap(True)
        self.recheck_status_lbl.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 11px; font-weight: 600;")
        recheck_layout.addWidget(self.recheck_status_lbl)
        recheck_actions = QHBoxLayout()
        recheck_actions.setContentsMargins(0, 0, 0, 0)
        recheck_actions.setSpacing(6)
        self.recheck_open_settings_btn = QPushButton("Open Settings")
        self.recheck_open_settings_btn.clicked.connect(self._open_normalization_settings)
        recheck_actions.addWidget(self.recheck_open_settings_btn)
        self.recheck_use_original_btn = QPushButton("Use original scale for this comparison")
        self.recheck_use_original_btn.clicked.connect(self._use_original_scale_for_current)
        recheck_actions.addWidget(self.recheck_use_original_btn)
        recheck_layout.addLayout(recheck_actions)
        self.normalization_recheck_bar.setVisible(False)
        it_layout.addWidget(self.normalization_recheck_bar)
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
        self.show_dancer_a_cb = QCheckBox("Dancer A axis")
        self.show_dancer_b_cb = QCheckBox("Dancer B axis")
        self.show_couple_axis_cb = QCheckBox("Couple midpoint axis")
        self.show_couple_steps_cb = QCheckBox("Couple steps & turns")
        for checkbox in (
            self.show_coach_cb,
            self.show_pivots_cb,
            self.show_steps_cb,
            self.show_dancer_a_cb,
            self.show_dancer_b_cb,
            self.show_couple_axis_cb,
            self.show_couple_steps_cb,
        ):
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
            self.show_dancer_a_cb,
            self.show_dancer_b_cb,
            self.show_couple_axis_cb,
            self.show_couple_steps_cb,
        )):
            checkbox.toggled.connect(self._refresh_overlay)
            layers.addWidget(checkbox, position // 2, position % 2)
        it_layout.addLayout(layers)
        self.axis_status = _caption_label("")
        self.axis_status.setVisible(False)
        it_layout.addWidget(self.axis_status)
        it_layout.addSpacing(4)
        it_layout.addWidget(_section_label("Tracking evidence"))
        evidence_row = QHBoxLayout()
        evidence_row.setSpacing(10)
        self.show_estimates_cb = QCheckBox("Show estimates")
        self.show_estimates_cb.setChecked(bool(self.bridge.config.replay_show_recovered_joints))
        self.show_estimates_cb.setToolTip(
            "Show display-only backward-pass and pivot-interpolated lower-limb joint estimates."
        )
        self.show_estimates_cb.toggled.connect(self._on_tracking_display_toggled)
        self.forward_only_cb = QCheckBox("Forward only")
        self.forward_only_cb.setChecked(False)
        self.forward_only_cb.setToolTip(
            "Show strictly forward pose observations and hide all recovered and interpolated joints."
        )
        self.forward_only_cb.toggled.connect(self._on_tracking_display_toggled)
        evidence_row.addWidget(self.show_estimates_cb)
        evidence_row.addWidget(self.forward_only_cb)
        evidence_row.addStretch()
        it_layout.addLayout(evidence_row)
        self.tracking_legend_label = _caption_label(
            "Solid: forward · Dashed: backward pass · Dotted: interpolated"
        )
        it_layout.addWidget(self.tracking_legend_label)
        self.tracking_evidence_status = _caption_label("Open a recording to inspect tracking evidence.")
        it_layout.addWidget(self.tracking_evidence_status)
        self.show_on_raw_btn = QPushButton("Show on raw video")
        self.show_on_raw_btn.setVisible(False)
        self.show_on_raw_btn.clicked.connect(self._switch_to_raw_video)
        it_layout.addWidget(self.show_on_raw_btn)
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
        self.metric_name_labels: dict[str, QLabel] = {}
        self.metric_values: dict[str, QLabel] = {}
        self.metric_states: dict[str, QLabel] = {}
        self.metric_ranges: dict[str, QLabel] = {}
        for row_index, (key, title) in enumerate(_METRIC_TITLES):
            state_dot = QLabel("●")
            state_dot.setFixedWidth(12)
            name_label = QLabel(title)
            self.metric_name_labels[key] = name_label
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
        self.report_normalization_notice = QLabel(
            "Note: This archived Markdown report reflects the original unadjusted analysis scale. "
            "Normalized Review adjustments apply during interactive Review playback."
        )
        self.report_normalization_notice.setWordWrap(True)
        self.report_normalization_notice.setStyleSheet(
            f"color: {COLOR_COPPER}; font-size: 11px; padding: 4px 6px;"
        )
        self.report_normalization_notice.setVisible(False)
        rt_layout.addWidget(self.report_normalization_notice)
        rt_layout.addWidget(self.report_view)

        insights_scroll = QScrollArea()
        insights_scroll.setWidgetResizable(True)
        insights_scroll.setFrameShape(QFrame.Shape.NoFrame)
        insights_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        insights_scroll.setWidget(insights_tab)
        couple_scroll = self._build_couple_evidence_tab()
        self.inspector_tabs = tabs
        tabs.addTab(insights_scroll, "Insights")
        tabs.addTab(report_tab, "Report")
        tabs.addTab(couple_scroll, "Couple evidence")
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
        self.bridge.calibrations_changed.connect(self.refresh)

    def has_active_jobs(self) -> bool:
        return self._analysis_thread is not None and self._analysis_thread.is_alive()

    def refresh(self) -> None:
        self._invalidate_smoothed_couple_overlays()
        cfg_recovery_auto = bool(self.bridge.config.replay_recovery_auto)
        if cfg_recovery_auto != self._last_cfg_recovery_auto:
            self._last_cfg_recovery_auto = cfg_recovery_auto
            self.recover_gaps_cb.setChecked(cfg_recovery_auto)
        cfg_show_recovered = bool(self.bridge.config.replay_show_recovered_joints)
        if cfg_show_recovered != self._last_cfg_show_recovered:
            self._last_cfg_show_recovered = cfg_show_recovered
            self.show_estimates_cb.blockSignals(True)
            self.show_estimates_cb.setChecked(cfg_show_recovered)
            self.show_estimates_cb.blockSignals(False)
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
        self._sync_edit_calibration_button()
        calibration_changed = self.reference_combo.currentData() != selected
        dynamic_stream = self._is_dynamic_stream()
        self._calibration_changed(refresh_overlay=calibration_changed or dynamic_stream)

    def _sync_edit_calibration_button(self) -> None:
        self.edit_calibration_btn.setEnabled(
            calibration_editor.selected_tpc_calibration_path(self.reference_combo.currentData())
            is not None
        )

    def _edit_current_calibration(self) -> None:
        cal_path = calibration_editor.selected_tpc_calibration_path(self.reference_combo.currentData())
        if cal_path is None:
            self.bridge.status_message.emit("Select a .tpc calibration profile first.")
            return
        try:
            saved = calibration_editor.open_profile_calibration_editor(
                cal_path, self.bridge.config, self, bridge=self.bridge
            )
            if saved is not None:
                self.refresh()
                idx = self.reference_combo.findData(str(saved))
                if idx < 0:
                    idx = self.reference_combo.findData(str(saved.resolve()))
                if idx >= 0:
                    self.reference_combo.setCurrentIndex(idx)
                self._calibration_changed()
                self.bridge.status_message.emit(f"Approved calibration saved: {saved.name}")
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            self.bridge.status_message.emit(f"Cannot edit calibration: {error}")

    def _on_reference_combo_changed(self, *_args: object) -> None:
        self._bypass_normalization_for_current = False
        self._sync_edit_calibration_button()
        self._calibration_changed()

    def _practice_identity_for_comparison(self, data: Mapping[str, Any]) -> HoldSourceIdentity | None:
        if self.current_archive_path is None:
            return None
        seg_idx = int(self._playback.selected_segment_index or 0)
        raw_sha = resolve_raw_media_sha256(
            self.current_archive_path, seg_idx, {"coach_data": data},
        )
        return HoldSourceIdentity(
            archive_path=str(self.current_archive_path),
            archive_name=self.current_archive_path.name,
            segment_index=seg_idx,
            raw_media_sha256=raw_sha,
            media_sha256=raw_sha,
        )

    def _is_normalization_applied(self) -> bool:
        norm = (self._compared_coach_data or {}).get("normalization")
        return isinstance(norm, Mapping) and norm.get("status") == "applied"

    def _sync_normalization_notices(self) -> None:
        norm = (self._compared_coach_data or {}).get("normalization")
        if isinstance(norm, Mapping) and norm.get("status") == "needs_recheck" and not self._bypass_normalization_for_current:
            reason = str(norm.get("reason") or "Stored offset requires recheck before applying.")
            self.recheck_status_lbl.setText(f"Needs recheck: {reason} Not applied.")
            self.normalization_recheck_bar.setVisible(True)
        else:
            self.normalization_recheck_bar.setVisible(False)
        self.report_normalization_notice.setVisible(self._is_normalization_applied())

    def _mark_hold_anchor(self, *, is_start: bool) -> None:
        ts = float(self.current_time_sec)
        if is_start:
            self._anchor_start_sec = ts
            label = "start"
        else:
            self._anchor_end_sec = ts
            label = "end"
        self._publish_review_context()
        self.bridge.status_message.emit(f"Hold {label} anchor marked at {ts:.2f}s.")

    def _open_normalization_settings(self) -> None:
        self._publish_review_context()
        self.router.navigate(Route.SETTINGS, dict(self.bridge.review_context))

    def _use_original_scale_for_current(self) -> None:
        self._bypass_normalization_for_current = True
        self._calibration_changed()

    def _extract_frame_raw_values(self, frame_index: int) -> dict[str, float | None]:
        values: dict[str, float | None] = {}
        data = self._compared_coach_data or self._coach_data
        if not isinstance(data, Mapping):
            return values
        frames = data.get("frames")
        entry = frames.get(str(frame_index)) if isinstance(frames, Mapping) else None
        if not isinstance(entry, Mapping):
            return values
        for row in entry.get("calibration_rows", []):
            if isinstance(row, Mapping) and "metric" in row:
                raw_val = row.get("observed_raw")
                if raw_val is None and row.get("normalization_status") in (None, "unadjusted", "direct_comparison"):
                    raw_val = row.get("raw")
                if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
                    values[str(row["metric"])] = float(raw_val)
        if values:
            return values
        width, height = data.get("width"), data.get("height")
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            return values
        try:
            obs = _coach_observation(dict(entry), width, height, require_measurements=False)
            measured = measure_ranges(obs, self.bridge.config)
            for key, quality in measured.quality.items():
                val = measured.values.get(key)
                if quality.valid and isinstance(val, (int, float)):
                    values[key] = float(val)
        except (OSError, ValueError, TypeError):
            pass
        return values

    def _publish_review_context(self, frame_index: int | None = None) -> None:
        if frame_index is None:
            reader = self._playback.reader
            frame_index = reader.index if reader is not None else 0
        cal_path = self.reference_combo.currentData()
        self.bridge.review_context = {
            "archive_path": self.current_archive_path,
            "segment_index": int(self._playback.selected_segment_index or 0),
            "calibration_id": self._calibration_id or None,
            "calibration_path": cal_path,
            "playhead_seconds": float(self.current_time_sec),
            "anchor_start": self._anchor_start_sec,
            "anchor_end": self._anchor_end_sec,
            "frame_raw_values": self._extract_frame_raw_values(frame_index),
        }

    def _update_calibration_comparison(self) -> None:
        self._compared_coach_data = None
        self._calibration_error = ""
        self._calibration_targets = {}
        self._calibration_name = ""
        self._calibration_id = ""
        selected = self.reference_combo.currentData()
        if self._coach_data is None:
            self._sync_normalization_notices()
            self._publish_review_context()
            return
        try:
            data = self._coach_data
            if isinstance(selected, str):
                path = Path(selected)
                reference = load_calibration_file(str(path), path.parent)
                if reference is None:
                    raise ValueError(f"Calibration not found: {path.name}")
                practice_identity = self._practice_identity_for_comparison(data)
                data = compare_coach_data(
                    data,
                    self.bridge.config,
                    reference,
                    practice_identity=practice_identity,
                    bypass_normalization=self._bypass_normalization_for_current,
                )
                range_profile = profile_from_calibration(reference)
                if range_profile is not None:
                    self._calibration_targets = dict(range_profile.metrics)
                    self._calibration_name = range_profile.name
                    self._calibration_id = range_profile.id
            self._compared_coach_data = interpolate_coach_data(
                data,
                self.bridge.config,
                pose_recovery=data.get("pose_recovery"),
                movement_data=getattr(self, "_movement_data", None),
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            self._calibration_error = f"Calibration comparison unavailable: {error}"
            self._compared_coach_data = interpolate_coach_data(
                self._coach_data,
                self.bridge.config,
                pose_recovery=self._coach_data.get("pose_recovery"),
                movement_data=getattr(self, "_movement_data", None),
            )
            self.analysis_status.setText(self._calibration_error)
            self.bridge.status_message.emit(self._calibration_error)
        self._sync_normalization_notices()
        self._publish_review_context()

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
        self,
        path: Path,
        *,
        segment_index: int = 0,
        timestamp: float | None = None,
        run_id: str | None = None,
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
                comp_couple = load_couple_companion(
                    path.parent / f"{path.stem}.couple.v1.json"
                )
                if comp_couple is None and isinstance(self._coach_data, dict):
                    raw_c = self._coach_data.get("couple_analysis")
                    comp_couple = raw_c if isinstance(raw_c, dict) else None
                if comp_couple is not None:
                    self._couple_companion_path = path.parent / f"{path.stem}.couple.v1.json"
                    self._couple_result = comp_couple
                    self._couple_run_id = str(comp_couple.get("run_id") or "")
                    self._dancer_analysis = {
                        "schema_version": 1,
                        "active_mode": "couple",
                        "active_run_id": self._couple_run_id,
                        "runs": {
                            "couple": {
                                "run_id": self._couple_run_id,
                                "resolved_mode": "couple",
                                "couple_result": comp_couple,
                            }
                        },
                    }
                    self._sync_dancer_analysis_state(preferred_run_id=run_id)
                else:
                    self._sync_dancer_analysis_state(preferred_run_id=run_id)
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
            if run_id:
                self._select_run_by_id(run_id)
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
        return self._playback._selected_variant in ("Raw video", "Local video", "Source video")

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
        self._dancer_analysis = None
        self._couple_result = None
        self._couple_run_id = None
        self._annotation_undo_stack.clear()
        self._reset_transport()
        self.report_view.clear()
        try:
            self._playback.select_segment(index)
            archive = self._playback.archive
            if archive is not None:
                analytics = read_analytics(archive.path, index)
                if analytics is not None:
                    self._coach_data = analytics.get("coach_data")
                    self._movement_data = analytics.get("movement_data")
                    raw_da = analytics.get("dancer_analysis")
                    if isinstance(raw_da, dict):
                        self._dancer_analysis = raw_da
                movement = read_movement_analysis(archive.path, index)
                if movement is not None and self._movement_data is None:
                    self._movement_data = validate_movement_analysis(
                        movement, expected_segment_index=index
                    )
            self._sync_dancer_analysis_state()
            self._update_calibration_comparison()
            self._refresh_archive_report()
            self.coach_btn.setEnabled(self._playback._report is not None)
            self._set_relink_availability()
            self._populate_streams()
            self._update_analysis_controls()
        except REPLAY_ERRORS as error:
            self._playback_error(error)

    def _populate_streams(self) -> None:
        for raw in ("Raw video", "Local video", "Source video"):
            if raw in self._playback.variants:
                self._on_stream_changed(raw)
                return
        raise ValueError("This recording has no raw video available")

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
            self.status_label.setText(
                f"{name} — {label}. Video only (no audio)."
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
        normalized = self._is_normalization_applied()
        if path.suffix.lower() == ".md":
            if normalized:
                self.router.navigate(
                    Route.COACH, {"report_path": path, "normalized_review": True},
                )
            else:
                self.router.navigate(Route.COACH, path)
        elif index is not None and path.suffix.lower() in (".tp", ".tpc", ".tpr"):
            payload: dict[str, Any] = {"archive_path": path, "segment_index": index}
            if normalized:
                payload["normalized_review"] = True
            self.router.navigate(Route.COACH, payload)
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

    def _invalidate_smoothed_couple_overlays(self) -> None:
        self._smoothed_couple_overlays_cache: tuple[int, tuple[float, float], list[dict[str, Any]]] | None = None

    def _resolve_couple_frame_overlay(self, frame_index: int) -> dict[str, Any] | None:
        if not isinstance(self._couple_result, dict):
            return None
        overlay_root = self._couple_result.get("overlay")
        if isinstance(overlay_root, dict):
            by_frame = overlay_root.get("by_frame")
            if isinstance(by_frame, dict):
                cand = by_frame.get(str(frame_index))
                if isinstance(cand, dict):
                    return cand
        raw_overlays = self._couple_result.get("frame_overlays")
        if isinstance(raw_overlays, list) and raw_overlays:
            cfg = self.bridge.config
            key_sig = (
                id(self._couple_result),
                (
                    max(float(cfg.overlay_smoothing), 0.14),
                    float(cfg.dancer_analysis.display_interpolation_max_gap_s),
                ),
            )
            cache = getattr(self, "_smoothed_couple_overlays_cache", None)
            if cache is None or cache[:2] != key_sig:
                smoothed = smooth_couple_frame_overlays(
                    [item for item in raw_overlays if isinstance(item, dict)],
                    tau_s=key_sig[1][0],
                    max_gap_s=key_sig[1][1],
                )
                cache = (key_sig[0], key_sig[1], smoothed)
                self._smoothed_couple_overlays_cache = cache
            frame_overlays = cache[2]
            for item in frame_overlays:
                if item.get("frame_index") == frame_index:
                    return item
            if 0 <= frame_index < len(frame_overlays):
                return frame_overlays[frame_index]
        return None

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
        if self._active_dancer_view_mode == "couple" and isinstance(self._couple_result, dict):
            if update_status:
                self.axis_status.clear()
                self.axis_status.setVisible(False)
            frame_overlay = self._resolve_couple_frame_overlay(frame_index)
            ann = self._couple_result.get("annotations") or {}
            clip_in = float(self._couple_result.get("clip_in") or 0.0)
            events = [
                ev
                for ev in (self._couple_result.get("events") or [])
                if isinstance(ev, dict)
                and not ev.get("deleted")
                and (
                    abs(float(ev.get("timestamp", -999.0)) - frame_time_sec) <= 0.35
                    or (
                        clip_in > 0.0
                        and abs(float(ev.get("timestamp", -999.0)) - (frame_time_sec + clip_in)) <= 0.35
                    )
                )
            ]
            return render_couple_frame_overlay(
                rendered,
                frame_overlay if isinstance(frame_overlay, dict) else None,
                show_dancer_a=self.show_dancer_a_cb.isChecked(),
                show_dancer_b=self.show_dancer_b_cb.isChecked(),
                show_couple_axis=self.show_couple_axis_cb.isChecked(),
                show_steps_turns=self.show_couple_steps_cb.isChecked(),
                dancer_a_name=str(ann.get("dancer_a_name") or "A"),
                dancer_b_name=str(ann.get("dancer_b_name") or "B"),
                active_events=events,
            )
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
                show_estimates=self.show_estimates_cb.isChecked(),
                forward_only=self.forward_only_cb.isChecked(),
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
        self.current_time_sec = self._playback.timestamp_for_frame(reader.index)
        self._update_frame_insights(reader.index)
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
        try:
            self._show_frame(reader.index)
        except REPLAY_ERRORS as error:
            self._playback_error(error)

    def _on_tracking_display_toggled(self) -> None:
        reader = self._playback.reader
        self._update_tracking_evidence_ui(reader.index if reader is not None else None)
        self._refresh_overlay()

    def _switch_to_raw_video(self) -> None:
        for raw in ("Raw video", "Local video", "Source video"):
            if raw in self._playback.variants:
                self._on_stream_changed(raw)
                return
        self.tracking_evidence_status.setText("Raw video needed for tracking recovery.")

    def _update_tracking_evidence_ui(self, frame_index: int | None = None) -> None:
        reader = self._playback.reader
        has_raw = self._analysis_video_path() is not None
        if reader is not None and not self._is_dynamic_stream():
            self.show_estimates_cb.setEnabled(False)
            self.forward_only_cb.setEnabled(False)
            if has_raw:
                self.show_on_raw_btn.setVisible(True)
                self.show_on_raw_btn.setEnabled(True)
                self.tracking_evidence_status.setText(
                    "Viewing baked overlay stream. Click 'Show on raw video' to toggle live tracking evidence."
                )
            else:
                self.show_on_raw_btn.setVisible(False)
                self.tracking_evidence_status.setText("Raw video needed for tracking recovery.")
            return

        self.show_on_raw_btn.setVisible(False)
        self.show_estimates_cb.setEnabled(True)
        self.forward_only_cb.setEnabled(True)
        if reader is not None and not has_raw:
            self.tracking_evidence_status.setText("Raw video needed for tracking recovery.")
            return

        data = self._display_coach_data() or self._coach_data
        if data is None:
            self.tracking_evidence_status.setText(
                "Analyze a recording or open an analyzed archive to inspect tracking evidence."
            )
            return
        if self.forward_only_cb.isChecked():
            self.tracking_evidence_status.setText(
                "Forward only: showing raw forward pose observations."
            )
            return

        frames = data.get("frames")
        idx = frame_index if frame_index is not None else (reader.index if reader is not None else 0)
        entry = frames.get(str(idx)) if isinstance(frames, dict) else None
        if isinstance(entry, Mapping):
            summary = str(entry.get("display_status_summary") or "").strip()
            raw_origins = entry.get("display_joint_origins")
            origins: Mapping[str, Any] = (
                raw_origins if isinstance(raw_origins, Mapping) else {}
            )
            raw_diagnostics = entry.get("display_diagnostics")
            diagnostics: Mapping[str, Any] = (
                raw_diagnostics if isinstance(raw_diagnostics, Mapping) else {}
            )
            if not self.show_estimates_cb.isChecked():
                if summary:
                    self.tracking_evidence_status.setText(f"Estimates hidden · {summary}")
                else:
                    self.tracking_evidence_status.setText(
                        "Estimates hidden (showing forward observations only)."
                    )
                return
            if summary:
                self.tracking_evidence_status.setText(summary)
                return
            rec_origins = {v for v in origins.values() if v in ("backward", "interpolated")}
            if "backward" in rec_origins and "interpolated" in rec_origins:
                self.tracking_evidence_status.setText(
                    "Recovered for display (backward pass + interpolation)."
                )
            elif "backward" in rec_origins:
                self.tracking_evidence_status.setText("Recovered for display (backward pass).")
            elif "interpolated" in rec_origins:
                self.tracking_evidence_status.setText("Recovered for display (interpolated).")
            elif diagnostics:
                reason = str(next(iter(diagnostics.values())))
                self.tracking_evidence_status.setText(f"Forward tracking ({reason}).")
            else:
                self.tracking_evidence_status.setText("Forward tracking.")
            return

        recovery_meta = data.get("pose_recovery")
        if isinstance(recovery_meta, Mapping) and recovery_meta.get("status_message"):
            self.tracking_evidence_status.setText(str(recovery_meta["status_message"]))
        else:
            self.tracking_evidence_status.setText("No pose data on this frame.")

    def _analysis_video_path(self) -> Path | None:
        for label in ("Raw video", "Local video", "Source video"):
            path = self._playback.variants.get(label)
            if path is not None and path.is_file():
                return path
        return None

    def _update_analysis_controls(self) -> None:
        thread = self._analysis_thread
        running = thread is not None and thread.is_alive()
        has_raw = self._analysis_video_path() is not None
        self.analytics_btn.setEnabled(has_raw and not running)
        self.recover_gaps_cb.setEnabled(has_raw and not running)
        self.recover_tracking_btn.setEnabled(has_raw and self._coach_data is not None and not running)
        if self._playback.reader is not None and not has_raw:
            self.recover_tracking_btn.setToolTip("Raw video needed for tracking recovery.")
            self.recover_gaps_cb.setToolTip("Raw video needed for tracking recovery.")
        else:
            self.recover_tracking_btn.setToolTip(
                "Run display-only backward lower-limb tracking recovery on a chosen gap or pivot."
            )
            self.recover_gaps_cb.setToolTip(
                "When analyzing, run display-only backward pose recovery on short lower-limb dropouts."
            )
        self.cancel_analysis_btn.setEnabled(
            self._analysis_cancel is not None and not self._analysis_cancel.is_set()
        )
        self.cancel_analysis_btn.setVisible(running or self.cancel_analysis_btn.isEnabled())
        reader = self._playback.reader
        self._update_tracking_evidence_ui(reader.index if reader is not None else None)

    def _start_posture_analysis(
        self, *, override_request: CoupleAnalysisRequest | None = None
    ) -> None:
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

        req_obj = (
            override_request
            if override_request is not None
            else self._build_review_dancer_request()
        )
        self._active_run_dancer_request = req_obj

        self._analysis_job_id += 1
        job_id = self._analysis_job_id
        cancel_event = threading.Event()
        self._analysis_cancel = cancel_event
        self._active_analysis_job = job_id
        self.analysis_status.setText("Preparing posture & dancer analysis…")
        self.cancel_analysis_btn.setEnabled(True)
        self.cancel_analysis_btn.setVisible(True)
        self.analytics_btn.setEnabled(False)
        self._analysis_timer.start()
        cfg = self.bridge.config.model_copy(deep=True)
        cfg.replay_recovery_auto = self.recover_gaps_cb.isChecked()
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
                sig_params = inspect.signature(analyse_review_clip).parameters
                call_kwargs: dict[str, Any] = {
                    "selected_out": selected_out,
                    "progress_callback": report_progress,
                    "cancel_check": cancel_event.is_set,
                }
                if "dancer_request" in sig_params:
                    call_kwargs["dancer_request"] = req_obj.to_dict()
                coach_data = analyse_review_clip(
                    video_path,
                    cfg,
                    **call_kwargs,
                )
                if getattr(coach_data, "selection_required", False):
                    with self._analysis_commit_lock:
                        cancelled = cancel_event.is_set() or self._active_analysis_job != job_id
                        event = "cancelled" if cancelled else "selection_required"
                        self._analysis_events.put((job_id, event, coach_data))
                    return
                if (
                    not cancel_event.is_set()
                    and archive_path is not None
                    and archive_path.suffix.lower() in (".tp", ".tpr")
                    and segment_index is not None
                ):
                    result = getattr(coach_data, "result", None)
                    if result is None:
                        raise ValueError("Analysis did not produce the data required for a report")
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
                        couple_result = getattr(result, "couple_result", None)
                        requested_mode = getattr(result, "requested_dancer_mode", "auto")
                        resolved_mode = getattr(result, "resolved_dancer_mode", "single")
                        auto_decision = getattr(result, "auto_decision", None)
                        run_id = (
                            str(couple_result.get("run_id") or "")
                            if isinstance(couple_result, dict)
                            else f"single-{segment_index}-{uuid4().hex[:8]}"
                        )
                        dancer_run: dict[str, Any] = {
                            "run_id": run_id,
                            "requested_mode": requested_mode,
                            "resolved_mode": resolved_mode,
                            "auto_decision": auto_decision,
                        }
                        if resolved_mode == "couple" and couple_result is not None:
                            dancer_run["couple_result"] = couple_result
                            prev_coach = previous.get("coach_data")
                            if isinstance(prev_coach, dict) and any(
                                isinstance(v, dict) and (v.get("poses") or v.get("measurement_poses"))
                                for v in (prev_coach.get("frames") or {}).values()
                            ):
                                for preserve_key in ("coach_data", "movement_data", "session", "session_events", "frames", "frame_timestamps"):
                                    if preserve_key in previous:
                                        payload[preserve_key] = previous[preserve_key]
                        else:
                            dancer_run["coach_data"] = coach_data.coach_data
                            dancer_run["movement_data"] = coach_data.movement_data
                            dancer_run["session"] = getattr(result, "session_summary", {})
                            dancer_run["frames"] = getattr(result, "scored_frame_log", [])
                        with self._analysis_commit_lock:
                            cancelled = cancel_event.is_set() or self._active_analysis_job != job_id
                            if not cancelled:
                                store_analytics(
                                    archive_path, segment_index, payload,
                                    expected_video_sha256=video_sha256, report_text=report,
                                    overlay_path=overlay, lines_path=lines, frame_log=result.frame_log,
                                    dancer_analysis_run=dancer_run,
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
                                        "requested_dancer_mode": requested_mode,
                                        "resolved_dancer_mode": resolved_mode,
                                    },
                                )
                                if (
                                    archive_path.suffix.lower() == ".tp"
                                    and isinstance(result, ClipAnalysisResult)
                                ):
                                    with suppress(Exception):
                                        promoted_tpr = promote_segment_to_tpr(
                                            archive_path,
                                            segment_index,
                                            cfg,
                                            preferred_name=str(manifest.get("name") or archive_path.stem),
                                        )
                                        coach_data = replace(coach_data, saved_archive_path=promoted_tpr)
                            event = "cancelled" if cancelled else "done"
                            self._analysis_events.put((job_id, event, coach_data))
                        return
                result = getattr(coach_data, "result", None)
                if not cancel_event.is_set() and isinstance(result, ClipAnalysisResult):
                    report_progress(95, 100, "Saving .tpr reference archive")
                    saved_tpr = save_clip_result_as_tpr(
                        result,
                        cfg,
                        cancel_event.is_set,
                        preferred_name=video_path.stem,
                    )
                    coach_data = replace(coach_data, saved_archive_path=saved_tpr)
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

    def _open_targeted_recovery_dialog(self) -> None:
        reader = self._playback.reader
        video_path = self._analysis_video_path()
        if reader is None or video_path is None:
            self.analysis_status.setText("Raw video needed for tracking recovery.")
            return
        if self._coach_data is None:
            self.analysis_status.setText("Analyze the video before running targeted tracking recovery.")
            return
        duration = max(0.1, float(self.duration_sec or reader.duration or 0.1))
        playhead = min(max(0.0, float(self.current_time_sec)), duration)
        default_start = max(0.0, playhead - 1.0)
        default_end = min(duration, default_start + 2.0)
        default_start = max(0.0, default_end - 2.0)

        dialog = QDialog(self)
        dialog.setWindowTitle("Recover lower-limb tracking")
        form = QFormLayout(dialog)
        preset_combo = QComboBox()
        preset_combo.addItem(
            f"Playhead window ({time_label(default_start)}–{time_label(default_end)})",
            (default_start, default_end),
        )
        preselect_idx = 0

        frames_map = self._coach_data.get("frames") if isinstance(self._coach_data, Mapping) else {}
        first_source_ts = None
        if isinstance(frames_map, Mapping) and frames_map:
            first_key = min(frames_map, key=lambda k: int(k) if str(k).isdigit() else 0)
            first_entry = frames_map.get(first_key)
            if isinstance(first_entry, Mapping) and isinstance(first_entry.get("timestamp"), int | float):
                first_source_ts = float(first_entry["timestamp"])

        def _to_clip_time(ts_val: float) -> float:
            if first_source_ts is not None and ts_val > duration + 1e-3:
                return max(0.0, min(duration, ts_val - first_source_ts))
            return max(0.0, min(duration, ts_val))

        pivots = (
            self._movement_data.get("pivots")
            if isinstance(self._movement_data, Mapping)
            else None
        ) or []
        for p_idx, piv in enumerate(pivots, start=1):
            if not isinstance(piv, Mapping):
                continue
            ps = _to_clip_time(float(piv.get("start_time", 0.0)))
            pe = _to_clip_time(float(piv.get("end_time", ps + 0.5)))
            if pe <= ps:
                pe = min(duration, ps + 0.25)
            preset_combo.addItem(
                f"Pivot #{p_idx}: {piv.get('direction', '')} ({time_label(ps)}–{time_label(pe)})",
                (ps, pe),
            )
            if ps - 0.1 <= playhead <= pe + 0.1 and preselect_idx == 0:
                preselect_idx = preset_combo.count() - 1

        existing_rec = self._coach_data.get("pose_recovery") if isinstance(self._coach_data, Mapping) else None
        for win in (existing_rec.get("windows") if isinstance(existing_rec, Mapping) else None) or []:
            if not isinstance(win, Mapping):
                continue
            ws = _to_clip_time(float(win.get("gap_start_time", 0.0)))
            we = _to_clip_time(float(win.get("gap_end_time", ws + 0.2)))
            if we <= ws:
                we = min(duration, ws + 0.25)
            tag = "budget-skipped" if win.get("budget_skipped") else (win.get("skip_reason") or "gap")
            preset_combo.addItem(
                f"Flagged {tag} ({time_label(ws)}–{time_label(we)})",
                (ws, we),
            )
            if ws - 0.1 <= playhead <= we + 0.1 and preselect_idx == 0:
                preselect_idx = preset_combo.count() - 1

        start_spin = QDoubleSpinBox()
        end_spin = QDoubleSpinBox()
        for spin in (start_spin, end_spin):
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setRange(0.0, duration)

        def _apply_preset(index: int) -> None:
            data = preset_combo.itemData(index)
            if isinstance(data, tuple) and len(data) == 2:
                start_spin.setValue(float(data[0]))
                end_spin.setValue(float(data[1]))

        preset_combo.currentIndexChanged.connect(_apply_preset)
        preset_combo.setCurrentIndex(preselect_idx)
        _apply_preset(preselect_idx)

        form.addRow("Target window:", preset_combo)
        form.addRow("Start (clip s):", start_spin)
        form.addRow("End (clip s):", end_spin)
        form.addRow(
            QLabel("Display-only backward recovery; measured posture, steps, and pivots stay unchanged.")
        )
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Recover")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.run_targeted_recovery(start_spin.value(), end_spin.value())

    def run_targeted_recovery(
        self,
        start_time: float,
        end_time: float,
        *,
        estimator_factory: Callable[[], Any] | None = None,
    ) -> bool:
        """Run display-only targeted recovery on [start_time, end_time] and persist."""
        if self._analysis_thread is not None:
            self._poll_posture_analysis()
        video_path = self._analysis_video_path()
        if self._playback.reader is None or video_path is None:
            self.analysis_status.setText("Raw video needed for tracking recovery.")
            return False
        if self._coach_data is None:
            self.analysis_status.setText("Analyze the recording before running targeted recovery.")
            return False
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            return False

        t0 = float(min(start_time, end_time))
        t1 = float(max(start_time, end_time))
        if t1 <= t0:
            t1 = t0 + 0.25

        self._analysis_job_id += 1
        job_id = self._analysis_job_id
        cancel_event = threading.Event()
        self._analysis_cancel = cancel_event
        self._active_analysis_job = job_id
        self.analysis_status.setText("Recovering tracking in selected window…")
        self.cancel_analysis_btn.setEnabled(True)
        self.cancel_analysis_btn.setVisible(True)
        self.analytics_btn.setEnabled(False)
        self.recover_gaps_cb.setEnabled(False)
        self.recover_tracking_btn.setEnabled(False)
        self._analysis_timer.start()

        cfg = self.bridge.config.model_copy(deep=True)
        coach_snapshot = dict(self._coach_data)
        movement_snapshot = dict(self._movement_data) if isinstance(self._movement_data, Mapping) else None
        archive_path = self.current_archive_path
        segment_index = self._playback.selected_segment_index
        report = self._playback._report
        companion_path: Path | None = None
        if report is not None and (archive_path is None or archive_path.suffix.lower() == ".md"):
            cand = (
                report.path.parent / report.coach_overlay_file
                if report.coach_overlay_file
                else report.path.parent / f"{report.path.stem}.coach.v1.json"
            )
            if cand.is_file():
                companion_path = cand
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
                merged_recovery = recover_clip_tracking_range(
                    video_path=video_path,
                    coach_data=coach_snapshot,
                    cfg=cfg,
                    start_time=t0,
                    end_time=t1,
                    movement_data=movement_snapshot,
                    estimator_factory=estimator_factory,
                    cancel_check=cancel_event.is_set,
                    progress_callback=report_progress,
                )
                updated_coach = {**coach_snapshot, "pose_recovery": merged_recovery}
                updated_coach = interpolate_coach_data(
                    updated_coach,
                    cfg,
                    pose_recovery=merged_recovery,
                    movement_data=movement_snapshot,
                )
                if (
                    not cancel_event.is_set()
                    and archive_path is not None
                    and archive_path.suffix.lower() in (".tp", ".tpr")
                    and segment_index is not None
                ):
                    if file_sha256(video_path) != video_sha256:
                        raise ValueError("Source video changed during recovery; archive was not changed")
                    with self._analysis_commit_lock:
                        cancelled = cancel_event.is_set() or self._active_analysis_job != job_id
                        if not cancelled:
                            store_analytics(
                                archive_path,
                                segment_index,
                                {"coach_data": updated_coach},
                                expected_video_sha256=video_sha256,
                            )
                        event = "cancelled" if cancelled else "recovery_done"
                        self._analysis_events.put((job_id, event, (updated_coach, merged_recovery)))
                    return
                if not cancel_event.is_set() and companion_path is not None:
                    with self._analysis_commit_lock:
                        cancelled = cancel_event.is_set() or self._active_analysis_job != job_id
                        if not cancelled:
                            save_coach_companion(companion_path, updated_coach)
                        event = "cancelled" if cancelled else "recovery_done"
                        self._analysis_events.put((job_id, event, (updated_coach, merged_recovery)))
                    return
                with self._analysis_commit_lock:
                    cancelled = cancel_event.is_set() or self._active_analysis_job != job_id
                    event = "cancelled" if cancelled else "recovery_done"
                    self._analysis_events.put((job_id, event, (updated_coach, merged_recovery)))
            except Exception as error:  # noqa: BLE001 - surface background recovery errors in the UI
                with self._analysis_commit_lock:
                    event = (
                        "cancelled"
                        if cancel_event.is_set() or self._active_analysis_job != job_id
                        else "error"
                    )
                    self._analysis_events.put((job_id, event, f"{type(error).__name__}: {error}"))
            finally:
                if archive_lease is not None:
                    archive_lease.close()

        self._analysis_thread = Thread(
            target=work,
            name="qt-review-targeted-recovery",
            daemon=True,
        )
        self._analysis_thread.start()
        return True

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
        pending_override_req: CoupleAnalysisRequest | None = None
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
            elif event == "selection_required" and isinstance(payload, ReviewAnalysis):
                dlg = NeedsSelectionDialog(
                    payload.auto_decision or {},
                    video_path=self._analysis_video_path(),
                    parent=self,
                )
                if dlg.exec() == NeedsSelectionDialog.DialogCode.Accepted:
                    pending_override_req = dlg.apply_to_request(self._active_run_dancer_request)
                else:
                    self.analysis_status.setText(
                        "Needs selection cancelled; saved archive was left unchanged."
                    )
            elif event == "done" and isinstance(payload, ReviewAnalysis):
                self._coach_data = payload.coach_data
                self._movement_data = payload.movement_data
                if isinstance(payload.couple_result, dict):
                    self._couple_result = payload.couple_result
                    self._couple_run_id = str(payload.couple_result.get("run_id") or "")
                    self._active_dancer_view_mode = "couple"
                else:
                    self._couple_result = None
                    self._active_dancer_view_mode = "single"
                self._update_calibration_comparison()
                try:
                    if self._playback.archive is not None:
                        path = self._playback.archive.path
                        index = self._playback.selected_segment_index or 0
                        frame = self._playback.reader.index if self._playback.reader else 0
                        self._analysis_cancel = None
                        self._active_analysis_job = None
                        self.load_archive(path, segment_index=index)
                        if self._playback.reader is not None:
                            self._show_frame(min(frame, self._playback.reader.count - 1))
                    elif payload.saved_archive_path is not None:
                        frame = self._playback.reader.index if self._playback.reader else 0
                        self._analysis_cancel = None
                        self._active_analysis_job = None
                        self.load_archive(payload.saved_archive_path, segment_index=0)
                        if self._playback.reader is not None:
                            self._show_frame(min(frame, self._playback.reader.count - 1))
                    else:
                        self._sync_dancer_analysis_state()
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
                saved_note = ""
                if (
                    self.current_archive_path is not None
                    and self.current_archive_path.suffix.lower() in (".tp", ".tpr")
                ):
                    if (
                        payload.saved_archive_path is not None
                        and payload.saved_archive_path != self.current_archive_path
                    ):
                        saved_note = (
                            "Previous analysis, report and overlay videos were replaced and saved, "
                            f"and reference {payload.saved_archive_path.name} was created. "
                        )
                    else:
                        saved_note = "Previous analysis, report and overlay videos were replaced and saved. "
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
            elif event == "recovery_done" and isinstance(payload, tuple) and len(payload) == 2:
                updated_coach, merged_recovery = payload
                self._coach_data = updated_coach
                try:
                    if self._playback.archive is not None:
                        path = self._playback.archive.path
                        index = self._playback.selected_segment_index or 0
                        frame = self._playback.reader.index if self._playback.reader else 0
                        self._analysis_cancel = None
                        self._active_analysis_job = None
                        self.load_archive(path, segment_index=index)
                        if self._playback.reader is not None:
                            self._show_frame(min(frame, self._playback.reader.count - 1))
                    else:
                        self._update_calibration_comparison()
                        if self._playback.reader is not None:
                            self._show_frame(self._playback.reader.index)
                except (OSError, ValueError, UnicodeError, KeyError, BadZipFile) as error:
                    self.bridge.status_message.emit(f"Could not refresh archive after recovery: {error}")
                msg = (
                    str(merged_recovery.get("status_message") or "").strip()
                    if isinstance(merged_recovery, Mapping)
                    else ""
                )
                self.analysis_status.setText(msg or "Targeted tracking recovery updated.")
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
            if pending_override_req is not None:
                self._start_posture_analysis(override_request=pending_override_req)

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

    def _couple_insights_snapshot(self, frame_index: int) -> dict[str, Any]:
        assert isinstance(self._couple_result, dict)
        ann = self._couple_result.get("annotations") or {}
        a_name = str(ann.get("dancer_a_name") or "Dancer A")
        b_name = str(ann.get("dancer_b_name") or "Dancer B")
        titles = {
            "head_angle": f"{a_name} torso tilt",
            "shoulder_elev_pct": f"{b_name} torso tilt",
            "axis_curve_smoothed": "Couple midpoint tilt",
            "torso_axis_tilt": "Torso separation",
            "knee_bend_deg": "Pair axis status",
        }
        values: dict[str, str] = {key: "Unavailable" for key in titles}
        ranges: dict[str, str | None] = {key: None for key in titles}
        states: dict[str, tuple[str | None, str, bool]] = {
            key: ("UNKNOWN", "Not measured on this frame", True) for key in titles
        }
        samples = self._couple_result.get("samples") or []
        sample: dict[str, Any] | None = None
        if isinstance(samples, list) and samples:
            for item in samples:
                if isinstance(item, dict) and item.get("frame_index") == frame_index:
                    sample = item
                    break
            if sample is None and 0 <= frame_index < len(samples) and isinstance(samples[frame_index], dict):
                sample = samples[frame_index]

        frame_overlay = self._resolve_couple_frame_overlay(frame_index)
        ta = sample.get("torso_tilt_a_deg") if isinstance(sample, dict) else None
        tb = sample.get("torso_tilt_b_deg") if isinstance(sample, dict) else None
        tm = sample.get("couple_midpoint_tilt_deg") if isinstance(sample, dict) else None
        sep = sample.get("torso_separation_deg") if isinstance(sample, dict) else None
        if ta is None and isinstance(frame_overlay, dict):
            ta = frame_overlay.get("tilt_a_deg")
        if tb is None and isinstance(frame_overlay, dict):
            tb = frame_overlay.get("tilt_b_deg")
        if tm is None and isinstance(frame_overlay, dict):
            tm = frame_overlay.get("couple_midpoint_tilt_deg")
        if sep is None and isinstance(frame_overlay, dict):
            sep = frame_overlay.get("torso_separation_deg")

        def _fmt_deg(val: Any) -> str | None:
            return f"{float(val):+.1f}°" if isinstance(val, int | float) and not isinstance(val, bool) else None

        if (txt := _fmt_deg(ta)) is not None:
            values["head_angle"] = txt
            ranges["head_angle"] = "Measured individual torso axis"
            states["head_angle"] = ("WITHIN_REFERENCE", f"{a_name} signed torso tilt", False)
        if (txt := _fmt_deg(tb)) is not None:
            values["shoulder_elev_pct"] = txt
            ranges["shoulder_elev_pct"] = "Measured individual torso axis"
            states["shoulder_elev_pct"] = ("WITHIN_REFERENCE", f"{b_name} signed torso tilt", False)
        if (txt := _fmt_deg(tm)) is not None:
            values["axis_curve_smoothed"] = txt
            ranges["axis_curve_smoothed"] = "Measured couple midpoint axis"
            states["axis_curve_smoothed"] = ("WITHIN_REFERENCE", "Couple midpoint tilt", False)
        if isinstance(sep, int | float) and not isinstance(sep, bool):
            values["torso_axis_tilt"] = f"{float(sep):.1f}°"
            ranges["torso_axis_tilt"] = "Absolute angle between A & B"
            states["torso_axis_tilt"] = ("WITHIN_REFERENCE", "Torso axis separation", False)

        pair_ok = isinstance(tm, int | float) and not isinstance(tm, bool)
        if pair_ok:
            values["knee_bend_deg"] = "Measured"
            ranges["knee_bend_deg"] = "Both dancers tracked"
            states["knee_bend_deg"] = ("WITHIN_REFERENCE", "Pair axes measured", False)
            alert: tuple[str, str] = ("Couple axes measured on this frame.", "success")
        elif isinstance(ta, int | float) or isinstance(tb, int | float):
            values["knee_bend_deg"] = "Single dancer only"
            ranges["knee_bend_deg"] = "Need both dancers for pair axis"
            states["knee_bend_deg"] = ("PENDING", "Second dancer unavailable on this frame", True)
            alert = (
                "Pair axis unavailable on this frame — showing detected individual dancer axis only.",
                "warning",
            )
        else:
            values["knee_bend_deg"] = "Unavailable"
            alert = ("Neither dancer torso axis is measurable on this frame.", "muted")

        summaries = self._couple_result.get("summaries") or {}
        coord = summaries.get("coordination") or {}
        posture = summaries.get("posture") or {}
        detail_lines: list[str] = ["Couple Axis & Coordination Engine (isolated from Single posture coaching)."]
        for p_key, p_lbl in (
            ("torso_tilt_a", f"{a_name} axis"),
            ("torso_tilt_b", f"{b_name} axis"),
            ("couple_midpoint_tilt", "Couple midpoint"),
            ("torso_separation", "Separation"),
        ):
            ps = posture.get(p_key) or {}
            if isinstance(ps, dict) and ps.get("median_deg") is not None:
                detail_lines.append(
                    f"{p_lbl}: median {float(ps['median_deg']):+.1f}° · coverage {float(ps.get('coverage') or 0.0) * 100:.0f}%"
                )
        for c_key, c_lbl in (
            ("start_lag_ms", "Start lag (B−A)"),
            ("settle_lag_ms", "Settle lag (B−A)"),
        ):
            cs = coord.get(c_key) or {}
            if isinstance(cs, dict) and cs.get("value") is not None:
                detail_lines.append(f"{c_lbl}: {float(cs['value']):+.0f} ms")
        return {
            "titles": titles,
            "values": values,
            "ranges": ranges,
            "states": states,
            "alert": alert,
            "details": "\n".join(detail_lines),
        }

    def _frame_insights_snapshot(self, frame_index: int) -> dict[str, Any]:
        if self._active_dancer_view_mode == "couple" and isinstance(self._couple_result, dict):
            return self._couple_insights_snapshot(frame_index)
        titles = dict(_METRIC_TITLES)
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
                "titles": titles,
                "values": values,
                "ranges": ranges,
                "states": states,
                "alert": (self._calibration_error, "warning"),
                "details": "Posture comparison is unavailable.",
            }
        data = self._display_coach_data()
        if data is None:
            return {
                "titles": titles,
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
                "titles": titles,
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
                "titles": titles,
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
        norm_meta = data.get("normalization")
        norm_status = norm_meta.get("status") if isinstance(norm_meta, Mapping) else None
        has_applied_row = any(
            row.get("normalization_status") in ("applied", "unavailable")
            for row in rows.values()
        )
        if norm_status == "needs_recheck" and not self._bypass_normalization_for_current:
            reason_txt = str((norm_meta or {}).get("reason") or "Recheck required in Settings.")
            details.append(
                f"Needs recheck — {reason_txt} · Not applied (showing original calibration scale)."
            )
        elif norm_status != "applied" and not has_applied_row:
            details.append("Original calibration scale · no adjustment applied.")
        else:
            for row in rows.values():
                row_status = row.get("normalization_status")
                label = METRIC_DETAILS.get(row["metric"], ("", row["metric"]))[1]
                unit = str(row.get("unit", ""))
                if row_status == "applied" and row.get("show_calculation_in_review", True):
                    raw_val = row.get("observed_raw")
                    off_val = row.get("normalization_offset")
                    ref_val = row.get("reference_scale_raw")
                    smooth_val = row.get("smoothed")
                    raw_txt = _format_measure(float(raw_val), unit) if isinstance(raw_val, (int, float)) else "—"
                    adj_txt = _format_signed_adjustment(float(off_val), unit) if isinstance(off_val, (int, float)) else "—"
                    ref_txt = _format_measure(float(ref_val), unit) if isinstance(ref_val, (int, float)) else "—"
                    smooth_txt = _format_measure(float(smooth_val), unit) if isinstance(smooth_val, (int, float)) else "—"
                    if row.get("normalization_method") == "manual":
                        verify_tag = "Manual · unverified"
                    elif isinstance(row.get("verification_error"), (int, float)):
                        verify_tag = f"Verified · check error {float(row['verification_error']):.2f}"
                    else:
                        verify_tag = "Verified on check hold"
                    details.append(
                        f"{label}: Raw {raw_txt} · Adjustment {adj_txt} · Reference scale {ref_txt} "
                        f"· Used for assessment (smoothed) {smooth_txt} ({verify_tag})."
                    )
                elif row_status == "unavailable":
                    raw_val = row.get("observed_raw")
                    raw_txt = _format_measure(float(raw_val), unit) if isinstance(raw_val, (int, float)) else "—"
                    details.append(
                        f"{label}: Raw {raw_txt} · Unavailable · Not applied (outside physical domain)."
                    )
                elif row_status == "needs_recheck":
                    details.append(
                        f"{label}: Needs recheck · Not applied ({row.get('normalization_detail') or 'Recheck required'})."
                    )
                elif row_status == "direct_comparison":
                    details.append(f"{label}: Direct comparison.")

        reasons: dict[str, list[str]] = {}
        for row in unknown:
            label = METRIC_DETAILS.get(row["metric"], ("", row["metric"]))[1]
            reasons.setdefault(str(row.get("reason") or "Not visible"), []).append(label)
        for reason, labels in reasons.items():
            details.append(f"Not measured — {', '.join(labels)}: {reason}.")

        return {
            "titles": titles,
            "values": values,
            "ranges": ranges,
            "states": states,
            "alert": alert,
            "details": "\n".join(details),
        }

    def _frame_metrics_lines(self, frame_index: int) -> list[str]:
        snapshot = self._frame_insights_snapshot(frame_index)
        values = snapshot["values"]
        titles = snapshot.get("titles") or dict(_METRIC_TITLES)
        return [
            f"{titles.get(key, title)}: {str(values.get(key, 'Not observed')).replace('°', ' deg')}"
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
        titles = snapshot.get("titles")
        if isinstance(titles, Mapping):
            for key, label in self.metric_name_labels.items():
                if key in titles:
                    label.setText(str(titles[key]))
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
        self._update_tracking_evidence_ui(frame_index)
        self._publish_review_context(frame_index)

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
        self._calibration_id = ""
        self._bypass_normalization_for_current = False
        self._anchor_start_sec = None
        self._anchor_end_sec = None
        self._movement_data = None
        self._dancer_analysis = None
        self._couple_result = None
        self._active_dancer_view_mode = "single"
        self._couple_run_id = None
        self._couple_companion_path = None
        self._annotation_undo_stack.clear()
        if hasattr(self, "view_run_combo"):
            self._sync_dancer_analysis_state()
        if hasattr(self, "normalization_recheck_bar"):
            self._sync_normalization_notices()
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
        self.segment_combo.blockSignals(True)
        self.segment_combo.clear()
        self.segment_combo.blockSignals(False)
        self.segment_combo.setEnabled(False)
        self.report_view.clear()
        self.coach_btn.setEnabled(False)
        self._update_frame_insights(0)
        self.analysis_status.setText(
            "Analyze a raw video to add synchronized posture insights. Analysis runs in the background."
        )
        self._update_analysis_controls()
        self.status_label.setText("Playback closed. Open a video or .tp archive to replay.")

    def _build_couple_evidence_tab(self) -> QScrollArea:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(_section_label("Couple Decision & Provenance"))
        self.couple_provenance_label = _caption_label(
            "No Couple run loaded for this recording."
        )
        layout.addWidget(self.couple_provenance_label)

        self.couple_markers_summary = _caption_label("Timeline markers: none")
        self.couple_markers_summary.setObjectName("coupleTimelineMarkersLabel")
        layout.addWidget(self.couple_markers_summary)

        layout.addWidget(_section_label("Dancer Roles"))
        names_form = QFormLayout()
        names_form.setContentsMargins(0, 0, 0, 0)
        self.dancer_a_name_edit = QLineEdit("Dancer A")
        self.dancer_b_name_edit = QLineEdit("Dancer B")
        names_form.addRow("Dancer A role:", self.dancer_a_name_edit)
        names_form.addRow("Dancer B role:", self.dancer_b_name_edit)
        layout.addLayout(names_form)

        names_btn_row = QHBoxLayout()
        self.apply_names_btn = QPushButton("Apply Labels")
        self.apply_names_btn.clicked.connect(self._apply_couple_names)
        names_btn_row.addWidget(self.apply_names_btn)
        names_btn_row.addStretch()
        layout.addLayout(names_btn_row)

        layout.addWidget(_section_label("Planned Transitions & Events"))
        plan_grid = QGridLayout()
        plan_grid.setSpacing(6)
        mark_rev_btn = QPushButton("Planned Reversal @ frame")
        mark_pause_btn = QPushButton("Planned Pause")
        mark_settle_btn = QPushButton("Settling Step")
        mark_rev_btn.clicked.connect(lambda: self._mark_planned_transition("planned_reversal"))
        mark_pause_btn.clicked.connect(lambda: self._mark_planned_transition("planned_pause"))
        mark_settle_btn.clicked.connect(lambda: self._mark_planned_transition("settling_step"))
        plan_grid.addWidget(mark_rev_btn, 0, 0, 1, 2)
        plan_grid.addWidget(mark_pause_btn, 1, 0)
        plan_grid.addWidget(mark_settle_btn, 1, 1)
        layout.addLayout(plan_grid)

        ev_edit_form = QFormLayout()
        ev_edit_form.setContentsMargins(0, 0, 0, 0)
        self.couple_event_type_combo = QComboBox()
        for et in (
            "start",
            "slow",
            "settle",
            "contact",
            "musical_target",
            "speed_cue",
            "turn_start",
            "turn_end",
            "exit",
            "embrace_opening",
        ):
            self.couple_event_type_combo.addItem(et, et)
        self.couple_event_subject_combo = QComboBox()
        for subj in ("pair", "A", "B"):
            self.couple_event_subject_combo.addItem(subj, subj)
        self.couple_pivot_behavior_combo = QComboBox()
        for pb in ("appears_planted", "rotates_in_place", "repositions", "unknown"):
            self.couple_pivot_behavior_combo.addItem(pb.replace("_", " "), pb)
        ev_edit_form.addRow("Event type:", self.couple_event_type_combo)
        ev_edit_form.addRow("Subject:", self.couple_event_subject_combo)
        ev_edit_form.addRow("Pivot foot:", self.couple_pivot_behavior_combo)
        layout.addLayout(ev_edit_form)

        ev_btns = QGridLayout()
        ev_btns.setSpacing(6)
        self.add_couple_event_btn = QPushButton("Add @ frame")
        self.confirm_couple_event_btn = QPushButton("Confirm")
        self.update_couple_event_btn = QPushButton("Update")
        self.delete_couple_event_btn = QPushButton("Delete")
        self.add_couple_event_btn.clicked.connect(self._add_couple_event)
        self.confirm_couple_event_btn.clicked.connect(self._confirm_couple_candidate)
        self.update_couple_event_btn.clicked.connect(self._update_couple_event)
        self.delete_couple_event_btn.clicked.connect(self._delete_couple_event)
        ev_btns.addWidget(self.add_couple_event_btn, 0, 0)
        ev_btns.addWidget(self.confirm_couple_event_btn, 0, 1)
        ev_btns.addWidget(self.update_couple_event_btn, 1, 0)
        ev_btns.addWidget(self.delete_couple_event_btn, 1, 1)
        layout.addLayout(ev_btns)

        self.couple_events_table = QTableWidget(0, 5)
        self.couple_events_table.setObjectName("coupleEventsTable")
        self.couple_events_table.setHorizontalHeaderLabels(
            ["Time (s)", "Event", "Subject", "Status", "Detail"]
        )
        hdr = self.couple_events_table.horizontalHeader()
        hdr.setMinimumSectionSize(58)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.couple_events_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.couple_events_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.couple_events_table.setMinimumHeight(240)
        self.couple_events_table.cellClicked.connect(self._on_couple_event_clicked)
        layout.addWidget(self.couple_events_table, 1)

        save_row = QHBoxLayout()
        self.couple_undo_btn = QPushButton("Undo")
        self.couple_undo_btn.setEnabled(False)
        self.couple_undo_btn.clicked.connect(self._undo_couple_annotations)
        self.couple_save_annotations_btn = QPushButton("Save Annotations")
        self.couple_save_annotations_btn.setObjectName("saveCoupleAnnotationsBtn")
        self.couple_save_annotations_btn.setProperty("class", "primary")
        self.couple_save_annotations_btn.clicked.connect(self._save_couple_annotations)
        save_row.addWidget(self.couple_undo_btn)
        save_row.addWidget(self.couple_save_annotations_btn)
        layout.addLayout(save_row)

        self.couple_evidence_status = _caption_label("")
        layout.addWidget(self.couple_evidence_status)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(tab)
        return scroll

    def _build_review_dancer_request(self) -> CoupleAnalysisRequest:
        raw_mode = str(self.dancer_mode_combo.currentData() or "auto")
        mode = raw_mode if raw_mode in ("auto", "single", "couple") else "auto"
        ann = self._current_annotations_model()
        return CoupleAnalysisRequest(
            requested_mode=mode,  # type: ignore[arg-type]
            forced_override=False,
            annotations=ann,
            auto_turn_suggestions=bool(self.bridge.config.dancer_analysis.auto_turn_suggestions),
        )

    def _update_saved_vs_next_label(self) -> None:
        if not hasattr(self, "saved_vs_next_label"):
            return
        next_label = str(self.dancer_mode_combo.currentText() or "Auto")
        saved_label = "None"
        if isinstance(self._dancer_analysis, dict) and isinstance(
            self._dancer_analysis.get("runs"), dict
        ):
            runs = self._dancer_analysis["runs"]
            modes = [m.capitalize() for m in ("single", "couple") if m in runs]
            if modes:
                active = str(self._dancer_analysis.get("active_mode") or modes[0]).capitalize()
                saved_label = f"{' + '.join(modes)} (viewing {self._active_dancer_view_mode.capitalize() or active})"
        elif self._coach_data is not None or self._movement_data is not None:
            saved_label = "Single (legacy)"
        self.saved_vs_next_label.setText(f"Saved result: {saved_label} · Next run: {next_label}")

    def _sync_mode_specific_controls(self) -> None:
        if not hasattr(self, "show_coach_cb"):
            return
        is_couple = self._active_dancer_view_mode == "couple" and isinstance(self._couple_result, dict)
        for single_cb in (
            self.show_coach_cb,
            self.show_torso_axis_cb,
            self.show_head_alignment_cb,
            self.show_head_posture_cb,
            self.show_pivots_cb,
            self.show_steps_cb,
        ):
            single_cb.setVisible(not is_couple)
        for couple_cb in (
            self.show_dancer_a_cb,
            self.show_dancer_b_cb,
            self.show_couple_axis_cb,
            self.show_couple_steps_cb,
        ):
            couple_cb.setVisible(is_couple)

    def _sync_dancer_analysis_state(self, *, preferred_run_id: str | None = None) -> None:
        self.view_run_combo.blockSignals(True)
        self.view_run_combo.clear()
        da = self._dancer_analysis
        runs = da.get("runs") if isinstance(da, dict) and isinstance(da.get("runs"), dict) else {}
        if runs:
            for m_key in ("couple", "single"):
                entry = runs.get(m_key)
                if isinstance(entry, dict):
                    rid = str(entry.get("run_id") or m_key)
                    self.view_run_combo.addItem(f"{m_key.capitalize()} run ({rid[:12]})", m_key)
            self.view_run_combo.setEnabled(self.view_run_combo.count() > 1)
            active_mode = str(da.get("active_mode") or "single") if isinstance(da, dict) else "single"
            if preferred_run_id:
                for m_key, entry in runs.items():
                    if isinstance(entry, dict) and str(entry.get("run_id") or "") == preferred_run_id:
                        active_mode = m_key
                        break
            idx = self.view_run_combo.findData(active_mode)
            if idx >= 0:
                self.view_run_combo.setCurrentIndex(idx)
            self._apply_run_selection(active_mode)
        else:
            self.view_run_combo.setEnabled(False)
            self._active_dancer_view_mode = "couple" if self._couple_result is not None else "single"
        self.view_run_combo.blockSignals(False)
        self._sync_mode_specific_controls()
        self._update_saved_vs_next_label()
        self._refresh_couple_evidence_tab()

    def _apply_run_selection(self, mode_key: str) -> None:
        da = self._dancer_analysis
        raw_runs = da.get("runs") if isinstance(da, dict) else None
        runs: dict[str, Any] = raw_runs if isinstance(raw_runs, dict) else {}
        entry = runs.get(mode_key)
        self._active_dancer_view_mode = mode_key if isinstance(entry, dict) else "single"
        if mode_key == "couple" and isinstance(entry, dict) and isinstance(entry.get("couple_result"), dict):
            self._couple_result = copy.deepcopy(entry["couple_result"])
            self._couple_run_id = str(
                entry.get("run_id") or self._couple_result.get("run_id") or ""
            )
        elif mode_key == "single" and isinstance(entry, dict):
            if isinstance(entry.get("coach_data"), dict):
                self._coach_data = entry["coach_data"]
            if isinstance(entry.get("movement_data"), dict):
                self._movement_data = entry["movement_data"]
            self._couple_result = None
            self._couple_run_id = str(entry.get("run_id") or "")
        self._sync_mode_specific_controls()

    def _select_run_by_id(self, run_id: str) -> None:
        da = self._dancer_analysis
        raw_runs = da.get("runs") if isinstance(da, dict) else None
        runs: dict[str, Any] = raw_runs if isinstance(raw_runs, dict) else {}
        for m_key, entry in runs.items():
            if not isinstance(entry, dict):
                continue
            cand_rid = str(entry.get("run_id") or "")
            couple_rid = (
                str((entry.get("couple_result") or {}).get("run_id") or "")
                if isinstance(entry.get("couple_result"), dict)
                else ""
            )
            if run_id in (cand_rid, couple_rid):
                self.view_run_combo.blockSignals(True)
                idx = self.view_run_combo.findData(m_key)
                if idx >= 0:
                    self.view_run_combo.setCurrentIndex(idx)
                self.view_run_combo.blockSignals(False)
                self._apply_run_selection(m_key)
                self._update_calibration_comparison()
                self._refresh_couple_evidence_tab()
                self._refresh_overlay()
                return
        self.analysis_status.setText(
            f"Run '{run_id}' was not found or was superseded; archive unchanged."
        )
        self.status_label.setText(
            f"Requested run '{run_id}' is unavailable on this recording."
        )

    def _on_view_run_changed(self) -> None:
        mode_key = str(self.view_run_combo.currentData() or "")
        if not mode_key:
            return
        self._apply_run_selection(mode_key)
        archive = self._playback.archive
        index = self._playback.selected_segment_index
        if archive is not None and index is not None and self._dancer_analysis is not None:
            try:
                updated = set_active_dancer_run(archive.path, index, mode_key)
                if isinstance(updated.get("dancer_analysis"), dict):
                    self._dancer_analysis = updated["dancer_analysis"]
            except (OSError, ValueError):
                pass
        self._update_calibration_comparison()
        self._update_saved_vs_next_label()
        self._refresh_couple_evidence_tab()
        self._refresh_overlay()

    def _refresh_couple_evidence_tab(self) -> None:
        if not hasattr(self, "couple_events_table"):
            return
        self.couple_events_table.setRowCount(0)
        res = self._couple_result
        if not isinstance(res, dict):
            self.couple_provenance_label.setText("No Couple run active for this recording.")
            self.couple_markers_summary.setText("Timeline markers: none")
            self.couple_undo_btn.setEnabled(False)
            self.couple_save_annotations_btn.setEnabled(False)
            return

        dec = res.get("auto_decision") or {}
        if isinstance(dec, dict) and dec:
            self.couple_provenance_label.setText(
                f"Resolved: {dec.get('resolved_mode', 'couple')} "
                f"(requested: {dec.get('requested_mode', 'auto')}) · "
                f"{dec.get('explanation', '')}"
            )
        else:
            self.couple_provenance_label.setText("Couple run active.")

        ann = res.get("annotations") or {}
        self.dancer_a_name_edit.setText(str(ann.get("dancer_a_name") or "Dancer A"))
        self.dancer_b_name_edit.setText(str(ann.get("dancer_b_name") or "Dancer B"))

        events = [ev for ev in (res.get("events") or []) if isinstance(ev, dict)]
        planned = list(ann.get("planned_transitions") or [])
        marker_parts: list[str] = []
        for ev in events:
            if ev.get("deleted"):
                continue
            r = self.couple_events_table.rowCount()
            self.couple_events_table.insertRow(r)
            ts = float(ev.get("timestamp", 0.0))
            etype = str(ev.get("event_type", ""))
            subj = str(ev.get("subject", "pair"))
            status = str(ev.get("status", "measured"))
            detail = str(ev.get("pivot_behavior") or ev.get("label") or "")
            t_item = QTableWidgetItem(f"{ts:.2f}")
            t_item.setData(Qt.ItemDataRole.UserRole, ev)
            self.couple_events_table.setItem(r, 0, t_item)
            self.couple_events_table.setItem(r, 1, QTableWidgetItem(etype))
            self.couple_events_table.setItem(r, 2, QTableWidgetItem(subj))
            self.couple_events_table.setItem(r, 3, QTableWidgetItem(status))
            self.couple_events_table.setItem(r, 4, QTableWidgetItem(detail))
            if len(marker_parts) < 8:
                marker_parts.append(f"{etype}@{ts:.1f}s")
        for pm in planned:
            if isinstance(pm, dict):
                marker_parts.append(f"{pm.get('kind', 'planned')}@{float(pm.get('timestamp', 0.0)):.1f}s")

        self.couple_markers_summary.setText(
            "Timeline markers: " + (", ".join(marker_parts) if marker_parts else "none")
        )
        self.couple_undo_btn.setEnabled(bool(self._annotation_undo_stack))
        self.couple_save_annotations_btn.setEnabled(True)

    def _current_annotations_model(self) -> CoupleAnnotations:
        if isinstance(self._couple_result, dict) and isinstance(
            self._couple_result.get("annotations"), dict
        ):
            return CoupleAnnotations.from_dict(self._couple_result["annotations"])
        return CoupleAnnotations()

    def _push_annotation_undo(self) -> None:
        if isinstance(self._couple_result, dict):
            self._annotation_undo_stack.append(
                copy.deepcopy(self._couple_result.get("annotations") or {})
            )
            self.couple_undo_btn.setEnabled(True)

    def _rebuild_local_couple_preview(self, annotations: CoupleAnnotations) -> None:
        if not isinstance(self._couple_result, dict):
            return
        self._couple_result = rebuild_couple_result_with_annotations(
            self._couple_result,
            annotations.to_dict(),
            self.bridge.config,
        )
        self._refresh_couple_evidence_tab()
        self._refresh_overlay()

    def _apply_couple_names(self) -> None:
        if not isinstance(self._couple_result, dict):
            return
        self._push_annotation_undo()
        ann = self._current_annotations_model()
        ann.dancer_a_name = self.dancer_a_name_edit.text().strip() or "Dancer A"
        ann.dancer_b_name = self.dancer_b_name_edit.text().strip() or "Dancer B"
        self._rebuild_local_couple_preview(ann)
        self.couple_evidence_status.setText("Updated dancer labels; click Save Annotations to persist.")

    def _mark_planned_transition(self, kind: str) -> None:
        if not isinstance(self._couple_result, dict):
            return
        self._push_annotation_undo()
        ann = self._current_annotations_model()
        ts = float(self.current_time_sec)
        ann.planned_transitions.append(
            PlannedTransitionMark(
                mark_id=f"plan-{uuid4().hex[:8]}",
                timestamp=ts,
                window_end=ts + 0.25,
                kind=kind,  # type: ignore[arg-type]
                note=f"Marked at {ts:.2f}s",
            )
        )
        self._rebuild_local_couple_preview(ann)
        self.couple_evidence_status.setText(f"Marked {kind} at {ts:.2f}s; click Save Annotations to persist.")

    def _add_couple_event(self) -> None:
        if not isinstance(self._couple_result, dict):
            return
        self._push_annotation_undo()
        ann = self._current_annotations_model()
        etype = str(self.couple_event_type_combo.currentData() or "turn_start")
        subj = str(self.couple_event_subject_combo.currentData() or "pair")
        pb = str(self.couple_pivot_behavior_combo.currentData() or "unknown")
        ts = float(self.current_time_sec)
        ann.items.append(
            CoupleAnnotationItem(
                annotation_id=f"ann-{uuid4().hex[:8]}",
                event_type=etype,  # type: ignore[arg-type]
                timestamp=ts,
                subject=subj,  # type: ignore[arg-type]
                status="user_confirmed",
                pivot_behavior=pb if etype in ("turn_start", "turn_end", "exit") else None,  # type: ignore[arg-type]
            )
        )
        self._rebuild_local_couple_preview(ann)
        self.couple_evidence_status.setText(f"Added {etype} ({subj}) at {ts:.2f}s.")

    def _selected_couple_event(self) -> dict[str, Any] | None:
        row = self.couple_events_table.currentRow()
        if row < 0:
            return None
        item = self.couple_events_table.item(row, 0)
        if item is None:
            return None
        ev = item.data(Qt.ItemDataRole.UserRole)
        return ev if isinstance(ev, dict) else None

    def _confirm_couple_candidate(self) -> None:
        ev = self._selected_couple_event()
        if ev is None or not isinstance(self._couple_result, dict):
            return
        self._push_annotation_undo()
        ann = self._current_annotations_model()
        ev_id = str(ev.get("event_id") or "")
        ann_id = str(ev.get("annotation_id") or "")
        existing = next(
            (i for i in ann.items if i.annotation_id == ann_id or (ev_id and i.source_candidate_id == ev_id)),
            None,
        )
        if existing is not None:
            existing.status = "user_confirmed"
            existing.deleted = False
        else:
            ann.items.append(
                CoupleAnnotationItem(
                    annotation_id=f"conf-{uuid4().hex[:8]}",
                    event_type=ev["event_type"],
                    timestamp=float(ev.get("timestamp", 0.0)),
                    end_timestamp=ev.get("end_timestamp"),
                    subject=ev.get("subject", "pair"),
                    status="user_confirmed",
                    source_candidate_id=ev_id or None,
                    pivot_behavior=ev.get("pivot_behavior"),
                    label=str(ev.get("label") or ""),
                )
            )
        self._rebuild_local_couple_preview(ann)

    def _update_couple_event(self) -> None:
        ev = self._selected_couple_event()
        if ev is None or not isinstance(self._couple_result, dict):
            return
        self._push_annotation_undo()
        ann = self._current_annotations_model()
        etype = str(self.couple_event_type_combo.currentData() or ev["event_type"])
        subj = str(self.couple_event_subject_combo.currentData() or ev.get("subject", "pair"))
        pb = str(self.couple_pivot_behavior_combo.currentData() or "unknown")
        ev_id = str(ev.get("event_id") or "")
        ann_id = str(ev.get("annotation_id") or "")
        existing = next(
            (i for i in ann.items if i.annotation_id == ann_id or (ev_id and i.source_candidate_id == ev_id)),
            None,
        )
        if existing is not None:
            existing.event_type = etype  # type: ignore[assignment]
            existing.subject = subj  # type: ignore[assignment]
            existing.status = "user_confirmed"
            existing.pivot_behavior = pb if etype in ("turn_start", "turn_end", "exit") else None  # type: ignore[assignment]
        else:
            ann.items.append(
                CoupleAnnotationItem(
                    annotation_id=f"edit-{uuid4().hex[:8]}",
                    event_type=etype,  # type: ignore[arg-type]
                    timestamp=float(ev.get("timestamp", 0.0)),
                    end_timestamp=ev.get("end_timestamp"),
                    subject=subj,  # type: ignore[arg-type]
                    status="user_confirmed",
                    source_candidate_id=ev_id or None,
                    pivot_behavior=pb if etype in ("turn_start", "turn_end", "exit") else None,  # type: ignore[arg-type]
                )
            )
        self._rebuild_local_couple_preview(ann)

    def _delete_couple_event(self) -> None:
        ev = self._selected_couple_event()
        if ev is None or not isinstance(self._couple_result, dict):
            return
        self._push_annotation_undo()
        ann = self._current_annotations_model()
        ev_id = str(ev.get("event_id") or "")
        ann_id = str(ev.get("annotation_id") or "")
        existing = next(
            (i for i in ann.items if i.annotation_id == ann_id or (ev_id and i.source_candidate_id == ev_id)),
            None,
        )
        if existing is not None:
            existing.deleted = True
        else:
            ann.items.append(
                CoupleAnnotationItem(
                    annotation_id=f"del-{uuid4().hex[:8]}",
                    event_type=ev["event_type"],
                    timestamp=float(ev.get("timestamp", 0.0)),
                    subject=ev.get("subject", "pair"),
                    status="user_confirmed",
                    source_candidate_id=ev_id or None,
                    deleted=True,
                )
            )
        self._rebuild_local_couple_preview(ann)

    def _undo_couple_annotations(self) -> None:
        if not self._annotation_undo_stack or not isinstance(self._couple_result, dict):
            return
        prev = self._annotation_undo_stack.pop()
        self._rebuild_local_couple_preview(CoupleAnnotations.from_dict(prev))
        self.couple_undo_btn.setEnabled(bool(self._annotation_undo_stack))

    def _save_couple_annotations(self) -> None:
        if not isinstance(self._couple_result, dict):
            return
        annotations_dict = self._current_annotations_model().to_dict()
        run_id = str(self._couple_run_id or self._couple_result.get("run_id") or "")
        archive = self._playback.archive
        seg_idx = self._playback.selected_segment_index
        if archive is not None and seg_idx is not None:
            try:
                expected_sha = str(self._couple_result.get("video_sha256") or "") or None
                rebuilt = update_couple_annotations(
                    archive.path,
                    seg_idx,
                    run_id=run_id,
                    annotations=annotations_dict,
                    cfg=self.bridge.config,
                    expected_video_sha256=expected_sha,
                )
                self._couple_result = rebuilt
                analytics = read_analytics(archive.path, seg_idx)
                if isinstance(analytics, dict) and isinstance(analytics.get("dancer_analysis"), dict):
                    self._dancer_analysis = analytics["dancer_analysis"]
                self._refresh_couple_evidence_tab()
                self._refresh_overlay()
                self.couple_evidence_status.setText("Saved annotations atomically into archive.")
                return
            except (OSError, ValueError, RuntimeError) as exc:
                self.couple_evidence_status.setText(f"Could not save annotations: {exc}")
                return
        if self._couple_companion_path is not None:
            rebuilt = rebuild_couple_result_with_annotations(
                self._couple_result, annotations_dict, self.bridge.config
            )
            save_couple_companion(self._couple_companion_path, rebuilt)
            self._couple_result = rebuilt
            self._refresh_couple_evidence_tab()
            self._refresh_overlay()
            self.couple_evidence_status.setText("Saved annotations into Couple companion file.")

    def _on_couple_event_clicked(self, row: int, _col: int) -> None:
        item = self.couple_events_table.item(row, 0)
        if item is None:
            return
        ev = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(ev, dict) and "timestamp" in ev and self._playback.reader is not None:
            ts = float(ev["timestamp"])
            self._show_frame(self._playback.frame_for_timestamp(ts))

    def hideEvent(self, event: QHideEvent) -> None:
        if self.has_active_jobs():
            self._pause()
            super().hideEvent(event)
            return
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
