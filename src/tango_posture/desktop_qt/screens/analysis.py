"""Dedicated Analysis screen for running Auto, Single, and Couple analysis."""

from __future__ import annotations

import shutil
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...archive import extract_segment_video, read_analytics, read_manifest, store_analytics
from ...config import Config
from ...couple.models import CoupleAnalysisRequest, CoupleAnnotations, DancerAnchor
from ...references.errors import ImportCancelledError
from ...references.pipeline import analyze_clip
from ...references.results import ClipAnalysisResult
from ...references.video import probe_import_video
from ...replay import VideoReader
from ..components.analysis_options import AnalysisOptionsWidget, NeedsSelectionDialog
from ..library.archives import promote_segment_to_tpr, save_clip_result_as_tpr
from ..review.analysis import render_review_report
from ..router import Route, Router
from ..service_bridge import ServiceBridge


class _AnalysisEvents(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(str, object)
    failed = Signal(str, str)
    cancelled = Signal(str)


class _InteractivePreviewLabel(QLabel):
    """Frame preview label that translates mouse clicks into canonical [0, 1] coordinates."""

    clicked_norm = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 220)
        self._display_rect_size = (0, 0)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pm = self.pixmap()
            if pm is not None and not pm.isNull():
                pw, ph = pm.width(), pm.height()
                ox = max(0.0, (self.width() - pw) * 0.5)
                oy = max(0.0, (self.height() - ph) * 0.5)
                x = (event.position().x() - ox) / max(1.0, float(pw))
                y = (event.position().y() - oy) / max(1.0, float(ph))
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    self.clicked_norm.emit(float(x), float(y))
        super().mousePressEvent(event)


class AnalysisScreen(QWidget):
    """Run Auto, Single, or Couple analysis on archive recordings or local video clips."""

    def __init__(
        self,
        bridge: ServiceBridge,
        router: Router,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.bridge = bridge
        self.router = router

        self._events = _AnalysisEvents(self)
        self._events.progress.connect(self._on_progress)
        self._events.completed.connect(self._on_completed)
        self._events.failed.connect(self._on_failed)
        self._events.cancelled.connect(self._on_cancelled)

        self._job_id: str | None = None
        self._cancel_event: Event | None = None
        self._worker_thread: Thread | None = None
        self._temp_dir: TemporaryDirectory[str] | None = None

        self._last_result: ClipAnalysisResult | None = None
        self._last_archive_path: Path | None = None
        self._last_segment_index: int = 0
        self._last_video_path: Path | None = None
        self._last_run_id: str | None = None
        self._active_request: CoupleAnalysisRequest | None = None
        self._active_target_info: dict[str, Any] = {}

        self._anchors: dict[str, DancerAnchor] = {}
        self._travel_points: list[tuple[float, float]] = []
        self._mark_mode: str = "none"
        self._preview_frame_bgr: np.ndarray | None = None
        self._preview_timestamp: float = 0.0

        self.destroyed.connect(lambda: self._shutdown())
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        # Header
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Video Analysis")
        title.setProperty("class", "title")
        subtitle = QLabel(
            "Run Auto, Single, or Couple movement & posture analysis on archive recordings or local video clips."
        )
        subtitle.setProperty("class", "subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.refresh_btn = QPushButton("Refresh sources")
        self.refresh_btn.clicked.connect(self.refresh)
        header.addWidget(self.refresh_btn)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # Left pane: source, preview, anchors, options & run controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(12)
        left_scroll.setWidget(left_container)

        source_card = QFrame()
        source_card.setProperty("class", "card")
        src_layout = QVBoxLayout(source_card)
        src_layout.setContentsMargins(14, 14, 14, 14)
        src_layout.setSpacing(8)
        src_title = QLabel("1. Source video or archive recording")
        src_title.setProperty("class", "section")
        src_layout.addWidget(src_title)

        mode_row = QHBoxLayout()
        self.source_group = QButtonGroup(self)
        self.archive_radio = QRadioButton("Archive recording")
        self.local_radio = QRadioButton("Local video clip")
        self.archive_radio.setChecked(True)
        self.source_group.addButton(self.archive_radio)
        self.source_group.addButton(self.local_radio)
        mode_row.addWidget(self.archive_radio)
        mode_row.addWidget(self.local_radio)
        mode_row.addStretch()
        src_layout.addLayout(mode_row)

        self.archive_form = QWidget()
        af_layout = QFormLayout(self.archive_form)
        af_layout.setContentsMargins(0, 0, 0, 0)
        self.archive_combo = QComboBox()
        self.archive_combo.setObjectName("analysisArchiveCombo")
        self.segment_combo = QComboBox()
        self.segment_combo.setObjectName("analysisSegmentCombo")
        af_layout.addRow("Archive:", self.archive_combo)
        af_layout.addRow("Recording:", self.segment_combo)
        src_layout.addWidget(self.archive_form)

        self.local_form = QWidget()
        lf_layout = QFormLayout(self.local_form)
        lf_layout.setContentsMargins(0, 0, 0, 0)
        file_row = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setObjectName("analysisLocalVideoEdit")
        self.video_path_edit.setPlaceholderText("Choose a local .mp4 / .mov / .mkv file…")
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse_video)
        file_row.addWidget(self.video_path_edit, 1)
        file_row.addWidget(self.browse_btn)
        lf_layout.addRow("Video file:", file_row)

        range_row = QHBoxLayout()
        self.clip_in_spin = QDoubleSpinBox()
        self.clip_in_spin.setRange(0.0, 36000.0)
        self.clip_in_spin.setDecimals(2)
        self.clip_out_spin = QDoubleSpinBox()
        self.clip_out_spin.setRange(0.0, 36000.0)
        self.clip_out_spin.setDecimals(2)
        range_row.addWidget(QLabel("Start (s):"))
        range_row.addWidget(self.clip_in_spin)
        range_row.addWidget(QLabel("End (s):"))
        range_row.addWidget(self.clip_out_spin)
        lf_layout.addRow("Clip range:", range_row)
        src_layout.addWidget(self.local_form)
        self.local_form.hide()
        left_layout.addWidget(source_card)

        # Preview + optional anchor / travel direction marking
        preview_card = QFrame()
        preview_card.setProperty("class", "card")
        pv_layout = QVBoxLayout(preview_card)
        pv_layout.setContentsMargins(14, 14, 14, 14)
        pv_layout.setSpacing(8)
        pv_title = QLabel("2. Preview frame, anchors & travel axis (optional)")
        pv_title.setProperty("class", "section")
        pv_layout.addWidget(pv_title)

        self.preview_label = _InteractivePreviewLabel()
        self.preview_label.setText("Select an archive recording or local video to preview.")
        self.preview_label.clicked_norm.connect(self._on_preview_clicked)
        pv_layout.addWidget(self.preview_label)

        mark_row = QHBoxLayout()
        self.mark_a_btn = QPushButton("Mark Dancer A")
        self.mark_a_btn.setCheckable(True)
        self.mark_b_btn = QPushButton("Mark Dancer B")
        self.mark_b_btn.setCheckable(True)
        self.mark_axis_btn = QPushButton("Mark Travel Axis")
        self.mark_axis_btn.setCheckable(True)
        self.clear_marks_btn = QPushButton("Clear marks")
        self.mark_a_btn.clicked.connect(lambda: self._set_mark_mode("A"))
        self.mark_b_btn.clicked.connect(lambda: self._set_mark_mode("B"))
        self.mark_axis_btn.clicked.connect(lambda: self._set_mark_mode("axis"))
        self.clear_marks_btn.clicked.connect(self._clear_marks)
        mark_row.addWidget(self.mark_a_btn)
        mark_row.addWidget(self.mark_b_btn)
        mark_row.addWidget(self.mark_axis_btn)
        mark_row.addWidget(self.clear_marks_btn)
        pv_layout.addLayout(mark_row)

        self.marks_status_label = QLabel("No anchors or travel axis marks set.")
        self.marks_status_label.setProperty("class", "muted")
        self.marks_status_label.setWordWrap(True)
        pv_layout.addWidget(self.marks_status_label)
        left_layout.addWidget(preview_card)

        # Shared options card
        options_card = QFrame()
        options_card.setProperty("class", "card")
        opt_layout = QVBoxLayout(options_card)
        opt_layout.setContentsMargins(14, 14, 14, 14)
        opt_layout.setSpacing(8)
        opt_title = QLabel("3. Dancer analysis options")
        opt_title.setProperty("class", "section")
        opt_layout.addWidget(opt_title)
        self.options_widget = AnalysisOptionsWidget(self.bridge.config, show_recover_gaps=True)
        opt_layout.addWidget(self.options_widget)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setObjectName("runAnalysisButton")
        self.run_btn.setProperty("class", "primary")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.cancel_btn)
        opt_layout.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        opt_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to analyze.")
        self.status_label.setObjectName("analysisStatusLabel")
        self.status_label.setWordWrap(True)
        opt_layout.addWidget(self.status_label)
        left_layout.addWidget(options_card)
        left_layout.addStretch()

        # Right pane: provenance & summary evidence
        right_card = QFrame()
        right_card.setProperty("class", "card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(10)

        prov_header = QHBoxLayout()
        prov_title = QLabel("Decision Provenance & Summary")
        prov_title.setProperty("class", "section")
        prov_header.addWidget(prov_title)
        prov_header.addStretch()
        self.resolved_badge = QLabel("Not run")
        self.resolved_badge.setObjectName("analysisResolvedBadge")
        self.resolved_badge.setProperty("class", "badge")
        prov_header.addWidget(self.resolved_badge)
        right_layout.addLayout(prov_header)

        self.explanation_label = QLabel(
            "Run analysis to inspect opening-window Auto evidence, tracking coverage, "
            "and Single or Couple measurements."
        )
        self.explanation_label.setWordWrap(True)
        right_layout.addWidget(self.explanation_label)

        self.scene_change_label = QLabel("")
        self.scene_change_label.setProperty("class", "muted")
        self.scene_change_label.setWordWrap(True)
        self.scene_change_label.hide()
        right_layout.addWidget(self.scene_change_label)

        self.provenance_table = QTableWidget(0, 2)
        self.provenance_table.setHorizontalHeaderLabels(["Opening-window check", "Value"])
        self.provenance_table.horizontalHeader().setStretchLastSection(True)
        self.provenance_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.provenance_table.setMaximumHeight(230)
        self.provenance_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.provenance_table)

        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Metric / Evidence", "Value", "Status"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.results_table, 1)

        nav_row = QHBoxLayout()
        self.open_review_btn = QPushButton("Open in Review")
        self.open_analytics_btn = QPushButton("Open in Analytics")
        self.open_review_btn.setEnabled(False)
        self.open_analytics_btn.setEnabled(False)
        self.open_review_btn.clicked.connect(self._open_in_review)
        self.open_analytics_btn.clicked.connect(self._open_in_analytics)
        nav_row.addStretch()
        nav_row.addWidget(self.open_review_btn)
        nav_row.addWidget(self.open_analytics_btn)
        right_layout.addLayout(nav_row)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_card)
        splitter.setSizes([520, 480])

        self.archive_radio.toggled.connect(self._on_source_kind_toggled)
        self.archive_combo.currentIndexChanged.connect(self._on_archive_changed)
        self.segment_combo.currentIndexChanged.connect(self._on_segment_changed)

        self.refresh()

    def has_active_jobs(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def _refresh_archive_combo(self, select_path: Path | None = None) -> None:
        current_path = select_path if isinstance(select_path, Path) else self.archive_combo.currentData()
        self.archive_combo.blockSignals(True)
        self.archive_combo.clear()
        items = [
            item for item in self.bridge.list_library_archives() if item.get("segments_count", 0) > 0
        ]
        for item in items:
            path = Path(item["path"])
            label = f"{item['name']} ({item.get('segments_count', 0)} segs)"
            self.archive_combo.addItem(label, path)
        if isinstance(current_path, Path):
            idx = self._find_archive_combo_index(current_path)
            if idx >= 0:
                self.archive_combo.setCurrentIndex(idx)
        self.archive_combo.blockSignals(False)

    def refresh(self) -> None:
        self.options_widget.set_config(self.bridge.config)
        self._refresh_archive_combo()
        self._on_archive_changed()

    def _find_archive_combo_index(self, target: Path) -> int:
        resolved = target.expanduser().resolve()
        for i in range(self.archive_combo.count()):
            item_path = self.archive_combo.itemData(i)
            if isinstance(item_path, Path) and item_path.expanduser().resolve() == resolved:
                return i
        return -1

    def load_source(self, payload: object) -> None:
        """Accept an archive Path or dict payload from Library or Review navigation."""
        if isinstance(payload, str) and payload.strip():
            payload = Path(payload)
        if isinstance(payload, Path):
            if payload.suffix.lower() in (".tp", ".tpr"):
                self.archive_radio.setChecked(True)
                self.refresh()
                idx = self._find_archive_combo_index(payload)
                if idx >= 0:
                    self.archive_combo.setCurrentIndex(idx)
            else:
                self.local_radio.setChecked(True)
                self._set_local_video(payload)
        elif isinstance(payload, dict):
            arch = payload.get("archive_path")
            seg = payload.get("segment_index", 0)
            vpath = payload.get("video_path")
            if isinstance(arch, str) and arch.strip():
                arch = Path(arch)
            if isinstance(arch, Path):
                self.archive_radio.setChecked(True)
                self.refresh()
                idx = self._find_archive_combo_index(arch)
                if idx >= 0:
                    self.archive_combo.setCurrentIndex(idx)
                if isinstance(seg, int) and 0 <= seg < self.segment_combo.count():
                    self.segment_combo.setCurrentIndex(seg)
            elif isinstance(vpath, (Path, str)):
                self.local_radio.setChecked(True)
                self._set_local_video(Path(vpath))

    def _on_source_kind_toggled(self, archive_checked: bool) -> None:
        self.archive_form.setVisible(archive_checked)
        self.local_form.setVisible(not archive_checked)
        if archive_checked:
            self._on_segment_changed()
        elif self.video_path_edit.text().strip():
            self._load_video_preview(Path(self.video_path_edit.text().strip()))

    def _on_archive_changed(self) -> None:
        self.segment_combo.blockSignals(True)
        self.segment_combo.clear()
        path = self.archive_combo.currentData()
        if isinstance(path, Path) and path.is_file():
            try:
                manifest = read_manifest(path)
                for idx, seg in enumerate(manifest.get("segments", [])):
                    ts_list = seg.get("timestamps") or []
                    dur = (float(ts_list[-1]) - float(ts_list[0])) if len(ts_list) >= 2 else 0.0
                    self.segment_combo.addItem(f"Segment {idx + 1} ({dur:.1f} s)", idx)
            except (OSError, ValueError, KeyError) as exc:
                self.status_label.setText(f"Could not read archive manifest: {exc}")
        self.segment_combo.blockSignals(False)
        self._on_segment_changed()

    def _on_segment_changed(self) -> None:
        if not self.archive_radio.isChecked():
            return
        path = self.archive_combo.currentData()
        seg_idx = self.segment_combo.currentData()
        if not isinstance(path, Path) or not isinstance(seg_idx, int) or not path.is_file():
            self._preview_frame_bgr = None
            self.preview_label.setText("No archive recording available.")
            self.open_review_btn.setEnabled(False)
            self.open_analytics_btn.setEnabled(False)
            return
        self._last_archive_path = path
        self._last_segment_index = seg_idx
        self.open_review_btn.setEnabled(True)
        analytics = read_analytics(path, seg_idx)
        self.open_analytics_btn.setEnabled(analytics is not None)
        if isinstance(analytics, dict):
            self._show_saved_archive_analytics(analytics)
        try:
            with TemporaryDirectory(prefix="tango-analysis-preview-") as tmp:
                vid_path = Path(tmp) / "preview.mp4"
                extract_segment_video(path, seg_idx, vid_path)
                self._load_video_preview(vid_path)
        except (OSError, ValueError, RuntimeError):
            self._preview_frame_bgr = None
            self.preview_label.setText(f"Archive segment {seg_idx + 1} ready to analyze.")

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a local video file",
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;All files (*)",
        )
        if path:
            self._set_local_video(Path(path))

    def _set_local_video(self, path: Path) -> None:
        self.video_path_edit.setText(str(path))
        try:
            probe = probe_import_video(path)
            self.clip_in_spin.setRange(0.0, probe.duration_seconds)
            self.clip_out_spin.setRange(0.0, probe.duration_seconds)
            self.clip_in_spin.setValue(0.0)
            self.clip_out_spin.setValue(probe.duration_seconds)
            self._preview_timestamp = 0.0
            self._load_video_preview(probe.path)
        except (OSError, ValueError, RuntimeError) as exc:
            self.status_label.setText(f"Could not inspect video: {exc}")

    def _load_video_preview(self, video_path: Path) -> None:
        try:
            reader = VideoReader(video_path)
            try:
                frame = reader.frame(0)
                if frame is not None:
                    self._preview_frame_bgr = frame.copy()
                    self._render_preview_pixmap()
                else:
                    self._preview_frame_bgr = None
                    self.preview_label.setText("Video preview unavailable.")
            finally:
                reader.close()
        except (OSError, ValueError, RuntimeError):
            self._preview_frame_bgr = None

    def _render_preview_pixmap(self) -> None:
        frame = self._preview_frame_bgr
        if frame is None or frame.size == 0:
            return
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        for did, anchor in self._anchors.items():
            if anchor.hip_xy is not None:
                px = max(0, min(w - 1, round(anchor.hip_xy[0] * w)))
                py = max(0, min(h - 1, round(anchor.hip_xy[1] * h)))
                color = (255, 170, 55) if did == "A" else (90, 110, 255)
                cv2.circle(canvas, (px, py), 10, color, 2, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    did,
                    (px + 12, max(16, py)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        if len(self._travel_points) >= 1:
            pts_px = [
                (max(0, min(w - 1, round(p[0] * w))), max(0, min(h - 1, round(p[1] * h))))
                for p in self._travel_points
            ]
            for px, py in pts_px:
                cv2.circle(canvas, (px, py), 6, (230, 210, 80), -1, cv2.LINE_AA)
            if len(pts_px) >= 2:
                cv2.arrowedLine(
                    canvas, pts_px[0], pts_px[-1], (230, 210, 80), 2, cv2.LINE_AA, tipLength=0.2
                )
        max_w = max(320, self.preview_label.width() - 8)
        max_h = 260
        scale = min(max_w / max(1, w), max_h / max(1, h))
        dw = max(1, round(w * scale))
        dh = max(1, round(h * scale))
        resized = cv2.resize(canvas, (dw, dh), interpolation=cv2.INTER_AREA)
        qimg = QImage(resized.data, dw, dh, dw * 3, QImage.Format.Format_BGR888).copy()
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    def _set_mark_mode(self, mode: str) -> None:
        self._mark_mode = mode if self._mark_mode != mode else "none"
        self.mark_a_btn.setChecked(self._mark_mode == "A")
        self.mark_b_btn.setChecked(self._mark_mode == "B")
        self.mark_axis_btn.setChecked(self._mark_mode == "axis")

    def _clear_marks(self) -> None:
        self._anchors.clear()
        self._travel_points.clear()
        self._set_mark_mode("none")
        self.marks_status_label.setText("No anchors or travel axis marks set.")
        self._render_preview_pixmap()

    def _on_preview_clicked(self, x: float, y: float) -> None:
        if self._mark_mode in ("A", "B"):
            self._anchors[self._mark_mode] = DancerAnchor(
                dancer_id=self._mark_mode,  # type: ignore[arg-type]
                timestamp=self._preview_timestamp,
                frame_index=0,
                hip_xy=(x, y),
            )
            self._set_mark_mode("none")
        elif self._mark_mode == "axis":
            if len(self._travel_points) >= 2:
                self._travel_points.clear()
            self._travel_points.append((x, y))
            if len(self._travel_points) >= 2:
                self._set_mark_mode("none")
        parts = [f"Dancer {did} ({a.hip_xy[0]:.2f}, {a.hip_xy[1]:.2f})" for did, a in sorted(self._anchors.items()) if a.hip_xy]
        if self._travel_points:
            parts.append(f"Travel axis points: {len(self._travel_points)}/2")
        self.marks_status_label.setText("; ".join(parts) if parts else "No anchors or travel axis marks set.")
        self._render_preview_pixmap()

    def _build_current_annotations(self) -> CoupleAnnotations:
        return CoupleAnnotations(
            anchors=list(self._anchors.values()),
            travel_direction_points=list(self._travel_points),
        )

    def _on_run_clicked(self) -> None:
        request = self.options_widget.build_request(annotations=self._build_current_annotations())
        self._start_analysis_job(request)

    def _start_analysis_job(self, request: CoupleAnalysisRequest) -> None:
        if self.has_active_jobs():
            self.status_label.setText("An analysis job is already running.")
            return
        self._cleanup_temp()
        cfg: Config = self.bridge.config.model_copy(deep=True)
        recover = bool(self.options_widget.recover_gaps_cb.isChecked())

        target_info: dict[str, Any] = {"cfg": cfg, "recover": recover}
        if self.archive_radio.isChecked():
            arch = self.archive_combo.currentData()
            seg = self.segment_combo.currentData()
            if not isinstance(arch, Path) or not isinstance(seg, int) or not arch.is_file():
                self.status_label.setText("Choose a valid archive recording first.")
                return
            self._temp_dir = TemporaryDirectory(prefix="tango-analysis-run-")
            vid_dest = Path(self._temp_dir.name) / f"seg_{seg:02d}.mp4"
            try:
                vid_sha = extract_segment_video(arch, seg, vid_dest)
            except (OSError, ValueError, RuntimeError) as exc:
                self._cleanup_temp()
                self.status_label.setText(f"Could not extract segment video: {exc}")
                return
            target_info.update({
                "kind": "archive",
                "archive_path": arch,
                "segment_index": seg,
                "video_path": vid_dest,
                "video_sha256": vid_sha,
                "clip_in": 0.0,
                "clip_out": None,
                "title": f"{arch.stem} · Segment {seg + 1}",
            })
        else:
            raw_path = self.video_path_edit.text().strip()
            if not raw_path or not Path(raw_path).is_file():
                self.status_label.setText("Choose an existing local video file first.")
                return
            clip_in = float(self.clip_in_spin.value())
            clip_out = float(self.clip_out_spin.value())
            if clip_out <= clip_in:
                self.status_label.setText("Clip end must be later than clip start.")
                return
            target_info.update({
                "kind": "local",
                "archive_path": None,
                "segment_index": 0,
                "video_path": Path(raw_path),
                "video_sha256": None,
                "clip_in": clip_in,
                "clip_out": clip_out,
                "title": Path(raw_path).stem,
            })

        self._active_request = request
        self._active_target_info = target_info
        self._last_video_path = target_info["video_path"]
        job_id = uuid4().hex
        self._job_id = job_id
        cancel_ev = Event()
        self._cancel_event = cancel_ev
        self._set_busy(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_label.setText("Starting analysis…")

        def _worker() -> None:
            try:
                res = analyze_clip(
                    video_path=target_info["video_path"],
                    cfg=cfg,
                    selected_in=float(target_info["clip_in"]),
                    selected_out=target_info["clip_out"],
                    title=str(target_info["title"]),
                    analysis_mode="movement",
                    analyse_soundtrack=bool(request.analyze_soundtrack),
                    recover_tracking=recover,
                    dancer_request=request,
                    progress_callback=lambda c, t, m: self._events.progress.emit(job_id, c, t, m),
                    cancel_check=cancel_ev.is_set,
                )
                if cancel_ev.is_set():
                    self._events.cancelled.emit(job_id)
                else:
                    if not res.selection_required and target_info.get("kind") == "local":
                        self._events.progress.emit(job_id, 95, 100, "Saving .tpr reference archive…")
                        saved_tpr = save_clip_result_as_tpr(
                            res,
                            cfg,
                            cancel_ev.is_set,
                            preferred_name=str(target_info["title"]),
                        )
                        target_info["saved_tpr_path"] = saved_tpr
                    self._events.completed.emit(job_id, res)
            except ImportCancelledError:
                self._events.cancelled.emit(job_id)
            except Exception as exc:  # noqa: BLE001 - worker reports failure safely to UI
                self._events.failed.emit(job_id, str(exc))

        t = Thread(target=_worker, name="qt-analysis-worker", daemon=True)
        self._worker_thread = t
        t.start()

    def _on_cancel_clicked(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.cancel_btn.setEnabled(False)
            self.status_label.setText("Cancellation requested…")

    def _on_progress(self, job_id: str, current: int, total: int, message: str) -> None:
        if job_id != self._job_id:
            return
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        if message:
            self.status_label.setText(message)

    def _on_completed(self, job_id: str, result: object) -> None:
        if job_id != self._job_id or not isinstance(result, ClipAnalysisResult):
            return
        self._set_busy(False)
        self.progress_bar.hide()

        if result.selection_required:
            decision = dict(result.auto_decision or {})
            video_path = self._active_target_info.get("video_path")
            self._populate_decision_view(decision, None)
            dlg = NeedsSelectionDialog(
                decision,
                video_path=video_path if isinstance(video_path, Path) else None,
                parent=self,
            )
            if dlg.exec() == NeedsSelectionDialog.DialogCode.Accepted:
                override_req = dlg.apply_to_request(self._active_request)
                if override_req is not None:
                    self._start_analysis_job(override_req)
                    return
            self._cleanup_temp()
            self.status_label.setText(
                "Analysis paused at Needs selection; archive was left untouched."
            )
            return

        self._last_result = result
        target_kind = self._active_target_info.get("kind")
        archive_path = self._active_target_info.get("archive_path")
        segment_index = int(self._active_target_info.get("segment_index", 0))
        video_sha = self._active_target_info.get("video_sha256")
        cfg: Config = self._active_target_info.get("cfg", self.bridge.config)

        run_id = (
            str(result.couple_result.get("run_id") or "")
            if isinstance(result.couple_result, dict)
            else f"single-{segment_index}-{uuid4().hex[:8]}"
        )
        self._last_run_id = run_id

        if target_kind == "archive" and isinstance(archive_path, Path):
            dancer_run: dict[str, Any] = {
                "run_id": run_id,
                "requested_mode": result.requested_dancer_mode,
                "resolved_mode": result.resolved_dancer_mode,
                "auto_decision": result.auto_decision,
            }
            if result.resolved_dancer_mode == "couple" and result.couple_result is not None:
                dancer_run["couple_result"] = result.couple_result
            else:
                dancer_run["coach_data"] = result.coach_data
                dancer_run["movement_data"] = result.movement_data
                dancer_run["session"] = result.session_summary
                dancer_run["frames"] = result.scored_frame_log
            payload: dict[str, Any] = {
                "model_hash": result.manifest.model_hash,
                "capture_profile": result.manifest.capture_profile,
                "max_gap": cfg.max_gap,
                "frame_timestamps": [float(r["timestamp"]) for r in result.frame_log],
                "frames": result.scored_frame_log,
                "coach_data": result.coach_data,
                "movement_data": result.movement_data,
                "session": result.session_summary,
                "session_events": result.session_events or [],
                "source_provenance": {
                    "sha256": result.source_sha256,
                    "clip_in": result.selected_in,
                    "clip_out": result.selected_out,
                },
            }
            if result.resolved_dancer_mode == "couple":
                prev_analytics = read_analytics(archive_path, segment_index) or {}
                prev_coach = prev_analytics.get("coach_data")
                if isinstance(prev_coach, dict) and any(
                    isinstance(v, dict) and (v.get("poses") or v.get("measurement_poses"))
                    for v in (prev_coach.get("frames") or {}).values()
                ):
                    for preserve_key in ("coach_data", "movement_data", "session", "session_events", "frames", "frame_timestamps"):
                        if preserve_key in prev_analytics:
                            payload[preserve_key] = prev_analytics[preserve_key]
            report_text = render_review_report(
                result, cfg, segment_index, {"title": archive_path.stem}
            )
            metadata_updates: dict[str, Any] = {
                "frame_timestamps": [float(r["timestamp"]) for r in result.frame_log],
                "analysis_mode": result.analysis_mode,
                "is_imported_reference": archive_path.suffix.lower() == ".tpr",
                "clip_in": result.selected_in,
                "clip_out": result.selected_out,
                "capture_profile": result.manifest.capture_profile,
                "requested_dancer_mode": result.requested_dancer_mode,
                "resolved_dancer_mode": result.resolved_dancer_mode,
            }
            try:
                store_analytics(
                    archive_path,
                    segment_index,
                    payload,
                    expected_video_sha256=str(video_sha) if video_sha else None,
                    report_text=report_text,
                    metadata_updates=metadata_updates,
                    dancer_analysis_run=dancer_run,
                )
                promoted_tpr: Path | None = None
                if archive_path.suffix.lower() == ".tp":
                    with suppress(Exception):
                        promoted_tpr = promote_segment_to_tpr(
                            archive_path,
                            segment_index,
                            cfg,
                            preferred_name=archive_path.stem,
                        )
                self._last_archive_path = archive_path
                self._last_segment_index = segment_index
                self._refresh_archive_combo(select_path=promoted_tpr or archive_path)
                self.open_review_btn.setEnabled(True)
                self.open_analytics_btn.setEnabled(True)
                if promoted_tpr is not None:
                    self.status_label.setText(
                        f"Saved {result.resolved_dancer_mode.capitalize()} analysis into "
                        f"{archive_path.name} and {promoted_tpr.name}."
                    )
                else:
                    self.status_label.setText(
                        f"Saved {result.resolved_dancer_mode.capitalize()} analysis into {archive_path.name}."
                    )
            except (OSError, ValueError, RuntimeError) as exc:
                self.status_label.setText(f"Analysis finished but archive save failed: {exc}")
        else:
            saved_tpr = self._active_target_info.get("saved_tpr_path")
            if isinstance(saved_tpr, Path):
                self._last_archive_path = saved_tpr
                self._last_segment_index = 0
                self._refresh_archive_combo(select_path=saved_tpr)
                self.open_review_btn.setEnabled(True)
                self.open_analytics_btn.setEnabled(True)
                self.status_label.setText(
                    f"Saved {result.resolved_dancer_mode.capitalize()} analysis into {saved_tpr.name}."
                )
            else:
                self.status_label.setText(
                    f"Completed {result.resolved_dancer_mode.capitalize()} analysis on local clip."
                )

        self._cleanup_temp()
        self._populate_decision_view(result.auto_decision, result.couple_result)

    def _on_failed(self, job_id: str, message: str) -> None:
        if job_id != self._job_id:
            return
        self._set_busy(False)
        self.progress_bar.hide()
        self._cleanup_temp()
        self.status_label.setText(f"Analysis failed: {message}")

    def _on_cancelled(self, job_id: str) -> None:
        if job_id != self._job_id:
            return
        self._set_busy(False)
        self.progress_bar.hide()
        self._cleanup_temp()
        self.status_label.setText("Analysis cancelled; saved data unchanged.")

    def _show_saved_archive_analytics(self, analytics: dict[str, Any]) -> None:
        da = analytics.get("dancer_analysis")
        if not isinstance(da, dict):
            return
        active_mode = str(da.get("active_mode") or "single")
        entry = (da.get("runs") or {}).get(active_mode)
        if isinstance(entry, dict):
            self._last_run_id = str(entry.get("run_id") or "")
            self._populate_decision_view(entry.get("auto_decision"), entry.get("couple_result"))

    def _populate_decision_view(
        self,
        decision: dict[str, Any] | None,
        couple_result: dict[str, Any] | None,
    ) -> None:
        self.provenance_table.setRowCount(0)
        self.results_table.setRowCount(0)
        self.scene_change_label.hide()

        if decision:
            req = str(decision.get("requested_mode", "auto")).capitalize()
            res = str(decision.get("resolved_mode", "single")).replace("_", " ").capitalize()
            badge_text = f"{req} → {res}" if req.lower() == "auto" else res
            self.resolved_badge.setText(badge_text)
            self.explanation_label.setText(str(decision.get("explanation") or ""))
            rows = [
                (
                    "Opening window",
                    (
                        f"{float(decision.get('effective_window_in', 0.0)):.1f}s – "
                        f"{float(decision.get('effective_window_out', 0.0)):.1f}s "
                        f"({float(decision.get('effective_duration_s', 0.0)):.1f}s)"
                    ),
                ),
                (
                    "Primary eligible coverage",
                    (
                        f"{float(decision.get('primary_coverage', 0.0)) * 100:.1f}% "
                        f"({float(decision.get('primary_eligible_duration_s', 0.0)):.2f}s)"
                    ),
                ),
                (
                    "Second coexistence duration",
                    f"{float(decision.get('second_coexistence_duration_s', 0.0)):.2f}s",
                ),
                (
                    "Joint paired coverage",
                    f"{float(decision.get('second_joint_coverage', 0.0)) * 100:.1f}%",
                ),
                (
                    "Max continuous paired span",
                    f"{float(decision.get('max_continuous_paired_span_s', 0.0)):.2f}s",
                ),
                (
                    "Max co-movement span",
                    f"{float(decision.get('max_co_movement_span_s', 0.0)):.2f}s",
                ),
            ]
            for label, val in rows:
                r = self.provenance_table.rowCount()
                self.provenance_table.insertRow(r)
                self.provenance_table.setItem(r, 0, QTableWidgetItem(label))
                self.provenance_table.setItem(r, 1, QTableWidgetItem(val))

        if isinstance(couple_result, dict):
            scene = couple_result.get("post_window_scene_change")
            if isinstance(scene, dict) and scene.get("detected"):
                self.scene_change_label.setText(
                    f"Post-window scene note: {scene.get('message', '')}"
                )
                self.scene_change_label.show()
            summary = couple_result.get("summary") or {}
            for key, item in summary.items():
                if not isinstance(item, dict):
                    continue
                metric_val = item.get("value")
                unit = str(item.get("unit") or "")
                val_text = (
                    f"{metric_val:.3f} {unit}".strip()
                    if isinstance(metric_val, (int, float))
                    else "Unavailable"
                )
                r = self.results_table.rowCount()
                self.results_table.insertRow(r)
                self.results_table.setItem(r, 0, QTableWidgetItem(key.replace("_", " ")))
                self.results_table.setItem(r, 1, QTableWidgetItem(val_text))
                self.results_table.setItem(r, 2, QTableWidgetItem(str(item.get("status") or "")))

    def _open_in_review(self) -> None:
        if isinstance(self._last_archive_path, Path):
            payload: dict[str, Any] = {
                "archive_path": self._last_archive_path,
                "segment_index": self._last_segment_index,
            }
            if self._last_run_id:
                payload["run_id"] = self._last_run_id
            self.router.navigate(Route.REVIEW, payload)

    def _open_in_analytics(self) -> None:
        if isinstance(self._last_archive_path, Path):
            self.router.navigate(
                Route.ANALYTICS,
                {"archive_path": self._last_archive_path, "segment_index": self._last_segment_index},
            )

    def _set_busy(self, busy: bool) -> None:
        self.run_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.archive_combo.setEnabled(not busy)
        self.segment_combo.setEnabled(not busy)
        self.refresh_btn.setEnabled(not busy)

    def _cleanup_temp(self) -> None:
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir.name, ignore_errors=True)
            self._temp_dir = None

    def _shutdown(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
        self._cleanup_temp()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._shutdown()
        super().closeEvent(event)
