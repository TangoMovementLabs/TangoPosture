"""Library screen listing all saved practices and archives."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

import cv2
from PySide6.QtCore import QFile, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent
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
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...archive import PracticeArchive, _archive_lock, read_analytics, rename_archive
from ...calibration.archives import read_range_profile
from ...calibration.models import METRIC_DETAILS, RangeProfile
from ...couple.models import CoupleAnalysisRequest
from ...references.archives import promote_reference, reference_directory, reference_path
from ...references.constants import MAX_IMPORT_RANGE_SECONDS
from ...references.results import ImportedClipResult
from ...references.video import probe_import_video
from ...references.worker import ReferenceImportWorker
from ...replay import VideoReader
from ...review import report_directory
from ..components.analysis_options import AnalysisOptionsWidget, NeedsSelectionDialog
from ..library.archives import archive_report
from ..router import Route, Router
from ..service_bridge import (
    ServiceBridge,
    format_created_at,
    sort_named_dated_items,
)


def _format_metric_unit(unit: str) -> str:
    return {
        "degrees": "°",
        "torso_lengths": "torso len",
        "axis_lengths": "axis len",
        "normalized_clearance": "clearance",
    }.get(unit, unit)


class ProfileDetailsDialog(QDialog):
    """Structured modal dialog displaying a saved calibration profile's metadata and ranges."""

    def __init__(
        self,
        profile: RangeProfile,
        path: Path,
        parent: QWidget | None = None,
        *,
        on_use_defaults: Callable[[], None] | None = None,
        on_edit_calibration: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.profile = profile
        self.path = path
        self.setWindowTitle(f"Calibration Profile — {profile.name}")
        self.resize(780, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Summary Card
        summary_card = QFrame()
        summary_card.setProperty("class", "card")
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(16, 14, 16, 14)
        summary_layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        title_label = QLabel(profile.name)
        title_label.setProperty("class", "title")
        header_row.addWidget(title_label)
        header_row.addStretch()

        context_badge = QLabel(profile.context)
        context_badge.setProperty("class", "badge")
        version_badge = QLabel(profile.measurement_version)
        version_badge.setProperty("class", "badge")
        header_row.addWidget(context_badge)
        header_row.addWidget(version_badge)
        summary_layout.addLayout(header_row)

        capture = profile.capture_profile
        side = str(capture.get("side", "left")).capitalize()
        facing = str(capture.get("facing", "right")).capitalize()
        width = capture.get("width", "?")
        height = capture.get("height", "?")
        source_name = str(
            profile.source.get("archive")
            or capture.get("source")
            or path.name
        )
        created_text = format_created_at(profile.created_at_utc, path)

        meta_grid = QGridLayout()
        meta_grid.setHorizontalSpacing(18)
        meta_grid.setVerticalSpacing(6)
        metadata_items = [
            ("Date Created:", created_text),
            ("Capture Orientation:", f"{side} side · {facing} facing"),
            ("Resolution:", f"{width}×{height}"),
            ("Source:", source_name),
        ]
        for idx, (label_text, value_text) in enumerate(metadata_items):
            row = idx // 2
            col = (idx % 2) * 2
            key_lbl = QLabel(label_text)
            key_lbl.setProperty("class", "subtitle")
            val_lbl = QLabel(value_text)
            val_lbl.setWordWrap(True)
            meta_grid.addWidget(key_lbl, row, col)
            meta_grid.addWidget(val_lbl, row, col + 1)
        meta_grid.setColumnStretch(1, 1)
        meta_grid.setColumnStretch(3, 1)
        summary_layout.addLayout(meta_grid)
        layout.addWidget(summary_card)

        # Approved Posture Ranges Card
        ranges_card = QFrame()
        ranges_card.setProperty("class", "card")
        ranges_layout = QVBoxLayout(ranges_card)
        ranges_layout.setContentsMargins(16, 14, 16, 14)
        ranges_layout.setSpacing(10)

        ranges_header = QHBoxLayout()
        ranges_title = QLabel("Approved Posture Ranges")
        ranges_title.setStyleSheet("font-size: 14px; font-weight: 600;")
        count_badge = QLabel(f"{len(profile.metrics)} metrics")
        count_badge.setProperty("class", "badge")
        ranges_header.addWidget(ranges_title)
        ranges_header.addStretch()
        ranges_header.addWidget(count_badge)
        ranges_layout.addLayout(ranges_header)

        self.metrics_table = QTableWidget(len(profile.metrics), 6)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Body Part", "Measurement", "Approved Range", "Median (P10–P90)", "View", "Coverage"]
        )
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header_view = self.metrics_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for col in (2, 3, 4, 5):
            header_view.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        for row_idx, (key, value) in enumerate(profile.metrics.items()):
            body_part, label, default_unit = METRIC_DETAILS.get(key, ("Posture", key, value.unit))
            unit_suffix = _format_metric_unit(value.unit or default_unit)
            sep = "" if unit_suffix == "°" else " "
            range_text = f"{value.lower:.4g}{sep}{unit_suffix} … {value.upper:.4g}{sep}{unit_suffix}"
            stats_text = (
                f"{value.median:.4g}{sep}{unit_suffix} "
                f"({value.p10:.4g}–{value.p90:.4g})"
            )
            coverage_text = (
                f"{value.valid_duration:.1f}/{value.selected_duration:.1f}s "
                f"({value.samples} samples)"
            )
            row_values = (
                body_part,
                label,
                range_text,
                stats_text,
                value.view.capitalize(),
                coverage_text,
            )
            for col_idx, cell_text in enumerate(row_values):
                item = QTableWidgetItem(cell_text)
                if col_idx in (2, 3, 5):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                elif col_idx == 4:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.metrics_table.setItem(row_idx, col_idx, item)
        self.metrics_table.resizeRowsToContents()
        ranges_layout.addWidget(self.metrics_table)
        layout.addWidget(ranges_card, 1)

        # Footer
        footer = QHBoxLayout()
        footer_note = QLabel(
            "Scoring and Review comparisons use these saved ranges directly; "
            "the original reference video is not required."
        )
        footer_note.setProperty("class", "subtitle")
        footer_note.setWordWrap(True)
        footer.addWidget(footer_note, 1)

        if on_edit_calibration is not None:
            self.edit_calibration_btn = QPushButton("Edit calibration…")
            self.edit_calibration_btn.setProperty("class", "accent")

            def _edit() -> None:
                self.accept()
                on_edit_calibration()

            self.edit_calibration_btn.clicked.connect(_edit)
            footer.addWidget(self.edit_calibration_btn)
        if on_use_defaults is not None:
            defaults_btn = QPushButton("Use as Settings defaults")
            defaults_btn.clicked.connect(on_use_defaults)
            footer.addWidget(defaults_btn)
        close_btn = QPushButton("Close")
        close_btn.setProperty("class", "primary")
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        layout.addLayout(footer)


class _ImportEvents(QObject):
    worker_event = Signal(object)


class LibraryScreen(QWidget):
    def __init__(self, bridge: ServiceBridge, router: Router, parent: QWidget | None = None):
        super().__init__(parent)
        self.bridge = bridge
        self.router = router
        self.practices: list[dict[str, Any]] = []
        self._import_events = _ImportEvents(self)
        self._import_events.worker_event.connect(self._handle_import_event)
        self._import_worker = ReferenceImportWorker(self._import_events.worker_event.emit)
        self._import_job_id: str | None = None
        self._import_cfg = bridge.config.model_copy(deep=True)
        self._import_as_reference = True
        self._last_import_params: dict[str, Any] = {}
        self._package_cancel = Event()
        self._package_thread: Thread | None = None
        self._package_events: Queue[tuple[Any, ...]] = Queue()
        self._profile_dialog: ProfileDetailsDialog | None = None
        self._package_timer = QTimer(self)
        self._package_timer.setInterval(100)
        self._package_timer.timeout.connect(self._poll_package_events)
        self.destroyed.connect(lambda: self._shutdown_import())
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_import)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Header
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Library")
        title.setProperty("class", "title")
        subtitle = QLabel(
            "Practice sessions, replayable references and approved calibration profiles."
        )
        subtitle.setProperty("class", "subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()

        self.import_btn = QPushButton("Import & Analyze Video…")
        self.import_btn.clicked.connect(self._analyze_import_video)
        self.open_btn = QPushButton("Import Local Video…")
        self.open_btn.clicked.connect(self._import_video)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        header.addWidget(self.import_btn)
        self.open_btn.hide()
        header.addWidget(self.refresh_btn)
        layout.addLayout(header)

        # Search Bar
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search practices by name…")
        self.search_input.textChanged.connect(self._filter_table)
        search_row.addWidget(self.search_input)
        search_row.addWidget(QLabel("File type:"))
        self.type_filter = QComboBox()
        self.type_filter.addItem("All types", "")
        self.type_filter.addItem("References", ".tpr")
        self.type_filter.addItem("Calibrations", ".tpc")
        self.type_filter.addItem("Practices", ".tp")
        self.type_filter.currentIndexChanged.connect(
            lambda _: self._filter_table(self.search_input.text())
        )
        search_row.addWidget(self.type_filter)
        search_row.addWidget(QLabel("Sort by:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Date created (newest)", "date_desc")
        self.sort_combo.addItem("Date created (oldest)", "date_asc")
        self.sort_combo.addItem("Name (A–Z)", "name_asc")
        self.sort_combo.addItem("Name (Z–A)", "name_desc")
        self.sort_combo.currentIndexChanged.connect(
            lambda _: self._filter_table(self.search_input.text())
        )
        search_row.addWidget(self.sort_combo)
        self.import_reference_cb = QCheckBox("Import as reference")
        self.import_reference_cb.setChecked(True)
        search_row.addWidget(self.import_reference_cb)
        layout.addLayout(search_row)

        # Table
        table_frame = QFrame()
        table_frame.setProperty("class", "card")
        tf_layout = QVBoxLayout(table_frame)
        tf_layout.setContentsMargins(12, 12, 12, 12)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Name", "Recordings", "Date / Created", "Actions", "Type"])
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionsClickable(True)
        header_view.setSortIndicatorShown(True)
        header_view.setSortIndicator(2, Qt.SortOrder.DescendingOrder)
        header_view.sectionClicked.connect(self._on_header_clicked)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tf_layout.addWidget(self.table)
        layout.addWidget(table_frame, 1)

        import_status = QHBoxLayout()
        self.import_status = QLabel(
            ".tp practice · .tpr replayable reference · .tpc approved calibration profile. "
            "Legacy .tpc recordings remain readable."
        )
        self.import_status.setWordWrap(True)
        self.import_progress = QProgressBar()
        self.import_progress.setRange(0, 100)
        self.import_progress.setValue(0)
        self.import_progress.setMaximumWidth(180)
        self.import_progress.hide()
        self.cancel_import_btn = QPushButton("Cancel Import")
        self.cancel_import_btn.setEnabled(False)
        self.cancel_import_btn.clicked.connect(self._cancel_import)
        import_status.addWidget(self.import_status, 1)
        import_status.addWidget(self.import_progress)
        import_status.addWidget(self.cancel_import_btn)
        layout.addLayout(import_status)

        self.refresh()

    def has_active_jobs(self) -> bool:
        return (
            self._import_job_id is not None or self._import_worker.is_running()
            or (self._package_thread is not None and self._package_thread.is_alive())
        )

    def refresh(self) -> None:
        self.practices = self.bridge.list_library_archives()
        self._filter_table(self.search_input.text())

    def _on_header_clicked(self, logical_index: int) -> None:
        current = self.sort_combo.currentData()
        if logical_index == 0:
            target = "name_desc" if current == "name_asc" else "name_asc"
        elif logical_index == 2:
            target = "date_asc" if current == "date_desc" else "date_desc"
        else:
            return
        idx = self.sort_combo.findData(target)
        if idx >= 0:
            self.sort_combo.setCurrentIndex(idx)

    def _filter_table(self, query: str) -> None:
        q = query.strip().lower()
        suffix = self.type_filter.currentData()
        sort_mode = self.sort_combo.currentData()
        self.practices = sort_named_dated_items(self.practices, sort_mode)
        filtered = [
            p for p in self.practices
            if q in p["name"].lower() and (not suffix or p["path"].suffix.lower() == suffix)
        ]

        header_view = self.table.horizontalHeader()
        if sort_mode == "name_asc":
            header_view.setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        elif sort_mode == "name_desc":
            header_view.setSortIndicator(0, Qt.SortOrder.DescendingOrder)
        elif sort_mode == "date_asc":
            header_view.setSortIndicator(2, Qt.SortOrder.AscendingOrder)
        else:
            header_view.setSortIndicator(2, Qt.SortOrder.DescendingOrder)

        self.table.setRowCount(0)
        for row_idx, p in enumerate(filtered):
            self.table.insertRow(row_idx)
            self.table.setItem(row_idx, 0, QTableWidgetItem(p["name"]))
            segments = "Report" if p.get("kind") == "report" else str(p["segments_count"])
            self.table.setItem(row_idx, 1, QTableWidgetItem(segments))
            created_display = format_created_at(p.get("created_at", ""), p.get("path"))
            self.table.setItem(row_idx, 2, QTableWidgetItem(created_display))
            profile_only = p["segments_count"] == 0
            type_label = p["path"].suffix
            mode_badge = "" if profile_only else self._archive_mode_badge(p["path"])
            if mode_badge:
                type_label = f"{type_label} · {mode_badge}"
            self.table.setItem(row_idx, 4, QTableWidgetItem(type_label))

            btn = QPushButton("Profile details" if profile_only else "Open in Review")
            btn.clicked.connect(
                lambda _, path=p["path"], profile_only=profile_only:
                self._profile_details(path) if profile_only else self.router.navigate(Route.REVIEW, path)
            )
            rename_btn = QPushButton("Rename")
            rename_btn.clicked.connect(
                lambda _, path=p["path"], name=p["name"]: self._rename(path, name)
            )
            delete_btn = QPushButton("Delete")
            delete_btn.clicked.connect(lambda _, path=p["path"]: self._delete(path))
            actions = QWidget()
            actions_layout = QHBoxLayout(actions)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            for button in (btn, rename_btn, delete_btn):
                actions_layout.addWidget(button)
            if not profile_only and p["path"].suffix.lower() in (".tp", ".tpr"):
                analyze_btn = QPushButton("Analyze")
                analyze_btn.clicked.connect(
                    lambda _, path=p["path"]: self.router.navigate(Route.ANALYSIS, path)
                )
                actions_layout.addWidget(analyze_btn)
            if p["path"].suffix.lower() == ".tpr":
                calibrate_btn = QPushButton("Create calibration")
                calibrate_btn.clicked.connect(lambda _, path=p["path"]: self._calibrate_reference(path))
                actions_layout.addWidget(calibrate_btn)
            elif p["path"].suffix.lower() == ".tp":
                promote_btn = QPushButton("Promote to Reference")
                promote_btn.clicked.connect(lambda _, path=p["path"]: self._promote(path))
                actions_layout.addWidget(promote_btn)
            elif profile_only:
                edit_btn = QPushButton("Edit calibration")
                edit_btn.clicked.connect(lambda _, path=p["path"]: self._edit_calibration(path))
                actions_layout.addWidget(edit_btn)
                defaults_btn = QPushButton("Use as Settings defaults")
                defaults_btn.clicked.connect(lambda _, path=p["path"]: self._use_defaults(path))
                actions_layout.addWidget(defaults_btn)
            self.table.setCellWidget(row_idx, 3, actions)

    def _edit_calibration(self, path: Path) -> None:
        from ..calibration.editor import open_profile_calibration_editor

        try:
            saved = open_profile_calibration_editor(
                path, self.bridge.config, self, bridge=self.bridge
            )
            if saved is not None:
                self.refresh()
                self.import_status.setText(f"Approved calibration saved: {saved}")
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            self.import_status.setText(f"Cannot edit calibration: {error}")

    def _calibrate_reference(self, path: Path) -> None:
        from ..calibration.editor import CalibrationEditor

        try:
            editor = CalibrationEditor(path, self.bridge.config, self, bridge=self.bridge)
            if editor.exec() == QDialog.DialogCode.Accepted:
                self.refresh()
                self.import_status.setText(f"Approved calibration saved: {editor.saved_path}")
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            self.import_status.setText(f"Cannot create calibration: {error}")

    def _use_defaults(self, path: Path) -> None:
        try:
            self.bridge.use_calibration_defaults(read_range_profile(path), self)
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.import_status.setText(f"Cannot use calibration defaults: {error}")

    def _profile_details(self, path: Path) -> None:
        try:
            profile = read_range_profile(path)
            dialog = ProfileDetailsDialog(
                profile,
                path,
                self,
                on_use_defaults=lambda: self._use_defaults(path),
                on_edit_calibration=lambda: self._edit_calibration(path),
            )
            self._profile_dialog = dialog
            dialog.exec()
        except (OSError, ValueError, KeyError) as error:
            self.import_status.setText(f"Cannot read profile: {error}")

    def _promote(self, path: Path) -> None:
        try:
            replay = PracticeArchive(path)
            try:
                dialog = QDialog(self)
                dialog.setWindowTitle("Promote recording / excerpt to Reference")
                form = QFormLayout(dialog)
                name_input = QLineEdit(f"{path.stem} reference")
                recording = QComboBox()
                recording.addItems([str(i) for i in range(len(replay.manifest["segments"]))])
                start, end = QDoubleSpinBox(), QDoubleSpinBox()
                rates: dict[str, float] = {}
                def select_recording() -> None:
                    segment = replay.manifest["segments"][recording.currentIndex()]
                    if not segment.get("video"):
                        raise ValueError("This recording has no original video")
                    reader = VideoReader(replay.folder / segment["video"])
                    try:
                        rates["fps"], rates["count"] = reader.fps, reader.count
                        for field in (start, end):
                            field.setRange(0, reader.duration)
                            field.setDecimals(3)
                        start.setValue(0)
                        end.setValue(reader.duration)
                    finally:
                        reader.close()
                recording.currentIndexChanged.connect(select_recording)
                select_recording()
                form.addRow("Reference name:", name_input)
                form.addRow("Recording index:", recording)
                form.addRow("Start seconds:", start)
                form.addRow("End seconds:", end)
                buttons = QDialogButtonBox(
                    QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
                )
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                form.addRow(buttons)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                destination = reference_path(reference_directory(self.bridge.config.data_dir),
                                             name_input.text().strip())
                first = round(start.value() * rates["fps"])
                last = round(end.value() * rates["fps"])
                if last <= first:
                    raise ValueError("Choose a nonempty excerpt")
                index = recording.currentIndex()
                stop = None if last == rates["count"] else last
                name = name_input.text().strip()
            finally:
                replay.close()
            self._package_cancel.clear()
            self._import_job_id = "promotion"
            self._set_import_busy(True)
            self.import_status.setText("Creating a separate reference; practice original retained…")
            def work() -> None:
                try:
                    result = promote_reference(
                        path, destination, name, index,
                        start_frame=first, end_frame=stop, cancel_check=self._package_cancel.is_set,
                    )
                    self._package_events.put(("archive_done", "promotion", result))
                except Exception as error:  # noqa: BLE001 - worker boundary reports failures
                    self._package_events.put(("import_error", "promotion", str(error)))
            self._package_thread = Thread(target=work, name="qt-reference-promotion", daemon=True)
            self._package_thread.start()
            self._package_timer.start()
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            self.import_status.setText(f"Cannot promote reference: {error}")

    def _rename(self, path: Path, current_name: str) -> None:
        name, accepted = QInputDialog.getText(self, "Rename Library entry", "Name:", text=current_name)
        if not accepted:
            return
        try:
            rename_archive(path, name)
        except (OSError, ValueError) as error:
            self.import_status.setText(f"Could not rename archive: {error}")
            return
        self.refresh()
        self.import_status.setText("Library name saved inside the archive.")

    def _delete(self, path: Path) -> None:
        decision = QMessageBox.question(
            self, "Move archive to Trash?",
            f"Move {path.name} and all its recordings and analytics to Trash?\n"
            "The original imported video and legacy reports will not be deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if decision != QMessageBox.StandardButton.Yes:
            return
        archive = QFile(str(path))
        try:
            with _archive_lock(path):
                success = archive.moveToTrash()
        except OSError as error:
            self.import_status.setText(f"Could not move {path.name} to Trash: {error}")
            return
        if not success:
            self.import_status.setText(
                f"Could not move {path.name} to Trash: {archive.errorString()}. Nothing was deleted."
            )
            return
        self.refresh()
        self.import_status.setText(f"{path.name} moved to Trash.")

    def _shutdown_import(self) -> None:
        self._package_cancel.set()
        self._import_worker.close()
        if self._package_thread is not None:
            self._package_thread.join(timeout=1.5)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._package_timer.stop()
        self._shutdown_import()
        super().closeEvent(event)

    def _poll_package_events(self) -> None:
        while True:
            try:
                event = self._package_events.get_nowait()
            except Empty:
                break
            self._handle_import_event(event)

    def _import_video(self) -> None:
        self._analyze_import_video()

    def _analyze_import_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a local video to analyze",
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;All files (*)",
        )
        if not path:
            return
        try:
            probe = probe_import_video(Path(path))
        except (cv2.error, OSError, ValueError, RuntimeError) as error:
            self.import_status.setText(f"Could not inspect video: {error}")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Import and analyze local video")
        form = QFormLayout(dialog)
        title_input = QLineEdit(probe.path.stem)
        start_input = QDoubleSpinBox()
        end_input = QDoubleSpinBox()
        for field in (start_input, end_input):
            field.setDecimals(2)
            field.setSingleStep(0.5)
            field.setRange(0.0, probe.duration_seconds)
        end_input.setValue(min(probe.duration_seconds, MAX_IMPORT_RANGE_SECONDS))
        start_input.valueChanged.connect(
            lambda value: end_input.setRange(
                value, min(probe.duration_seconds, value + MAX_IMPORT_RANGE_SECONDS)
            )
        )
        form.addRow("Title:", title_input)
        form.addRow("Clip start (seconds):", start_input)
        form.addRow("Clip end (seconds):", end_input)
        options_widget = AnalysisOptionsWidget(
            self.bridge.config, show_recover_gaps=True, compact=True, parent=dialog
        )
        form.addRow(options_widget)
        form.addRow(
            QLabel(
                f"Maximum clip length {MAX_IMPORT_RANGE_SECONDS:.0f} seconds. "
                "The source video is not modified."
            )
        )
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Analyze")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected_in = start_input.value()
        selected_out = end_input.value()
        if selected_out <= selected_in:
            self.import_status.setText("Clip end must be later than clip start.")
            return
        if selected_out - selected_in > MAX_IMPORT_RANGE_SECONDS:
            self.import_status.setText(
                f"Choose a clip no longer than {MAX_IMPORT_RANGE_SECONDS:.0f} seconds."
            )
            return
        request = options_widget.build_request()
        self._start_import(
            probe.path,
            selected_in,
            selected_out,
            title_input.text().strip() or probe.path.stem,
            recover_tracking=options_widget.recover_gaps_cb.isChecked(),
            dancer_request=request.to_dict(),
            analyse_soundtrack=bool(request.analyze_soundtrack),
        )

    def _archive_mode_badge(self, archive_path: Path) -> str:
        try:
            analytics = read_analytics(archive_path, 0)
        except (OSError, ValueError):
            return ""
        if not isinstance(analytics, dict):
            return ""
        da = analytics.get("dancer_analysis")
        if isinstance(da, dict) and isinstance(da.get("runs"), dict):
            has_s = "single" in da["runs"]
            has_c = "couple" in da["runs"]
            if has_s and has_c:
                return "Single + Couple"
            if has_c:
                return "Couple"
            if has_s:
                return "Single"
        return "Single"

    def _start_import(
        self,
        path: Path | str,
        selected_in: float,
        selected_out: float | None,
        title: str,
        *,
        recover_tracking: bool | None = None,
        dancer_request: dict[str, Any] | None = None,
        analyse_soundtrack: bool = False,
    ) -> None:
        self._package_cancel.clear()
        self._import_cfg = self.bridge.config.model_copy(deep=True)
        self._import_as_reference = self.import_reference_cb.isChecked()
        self._last_import_params = {
            "path": path,
            "selected_in": selected_in,
            "selected_out": selected_out,
            "title": title,
            "recover_tracking": recover_tracking,
            "dancer_request": dancer_request,
            "analyse_soundtrack": analyse_soundtrack,
        }
        try:
            self._import_job_id = self._import_worker.start(
                video_path=path,
                cfg=self._import_cfg,
                report_dir=report_directory(self._import_cfg.data_dir),
                selected_in=selected_in,
                selected_out=selected_out,
                title=title,
                analysis_mode="movement",
                analyse_soundtrack=bool(analyse_soundtrack),
                recover_tracking=(
                    bool(self._import_cfg.replay_recovery_auto)
                    if recover_tracking is None
                    else bool(recover_tracking)
                ),
                dancer_request=dancer_request,
            )
        except (OSError, RuntimeError, ValueError) as error:
            self.import_status.setText(f"Could not start import: {error}")
            self._finish_import()
            return
        self._set_import_busy(True)
        self.import_status.setText("Analyzing local video…")
        self.import_progress.setRange(0, 0)
        self.import_progress.show()
        self.cancel_import_btn.setEnabled(True)

    def _cancel_import(self) -> None:
        if self._import_job_id is None:
            return
        self._import_worker.cancel()
        self._package_cancel.set()
        self.cancel_import_btn.setEnabled(False)
        self.import_status.setText("Cancellation requested; waiting for the current frame to finish.")

    def _handle_import_event(self, event: object) -> None:
        if not isinstance(event, tuple) or len(event) < 2:
            self.import_status.setText("Import worker returned an invalid status event.")
            self._finish_import()
            return
        kind, job_id, *values = event
        if job_id != self._import_job_id:
            return
        if kind == "import_progress" and len(values) == 3:
            current, total, message = values
            if isinstance(current, int) and isinstance(total, int) and total > 0:
                self.import_progress.setRange(0, total)
                self.import_progress.setValue(current)
            if isinstance(message, str):
                self.import_status.setText(message)
        elif kind == "import_done" and len(values) == 1 and isinstance(
            values[0], ImportedClipResult
        ):
            result = values[0]
            if result.selection_required:
                self._finish_import()
                dlg = NeedsSelectionDialog(
                    result.auto_decision or {},
                    video_path=result.video_path,
                    parent=self,
                )
                if dlg.exec() == NeedsSelectionDialog.DialogCode.Accepted:
                    base_req = CoupleAnalysisRequest.from_dict(
                        self._last_import_params.get("dancer_request")
                    )
                    override_req = dlg.apply_to_request(base_req)
                    if override_req is not None:
                        self._start_import(
                            self._last_import_params["path"],
                            float(self._last_import_params["selected_in"]),
                            self._last_import_params["selected_out"],
                            str(self._last_import_params["title"]),
                            recover_tracking=self._last_import_params.get("recover_tracking"),
                            dancer_request=override_req.to_dict(),
                            analyse_soundtrack=bool(
                                self._last_import_params.get("analyse_soundtrack", False)
                            ),
                        )
                        return
                self.import_status.setText(
                    "Import paused at Needs selection; no files were written."
                )
                return
            job = job_id
            cfg = self._import_cfg
            as_reference = self._import_as_reference
            events = self._package_events
            self.import_status.setText(
                f"Saving indexed raw, overlay, lines and analytics in a "
                f"{'.tpr' if as_reference else '.tp'} archive…"
            )

            def package() -> None:
                try:
                    path = archive_report(
                        result.report_path, cfg, self._package_cancel.is_set,
                        as_reference=as_reference,
                    )
                    events.put(("archive_done", job, path))
                except Exception as error:  # noqa: BLE001 - worker boundary reports failures
                    kind = "import_cancelled" if self._package_cancel.is_set() else "import_error"
                    events.put((kind, job, str(error)))

            self._package_thread = Thread(target=package, name="qt-library-packaging", daemon=True)
            self._package_thread.start()
            self._package_timer.start()
        elif kind == "archive_done" and values and isinstance(values[0], Path):
            self._finish_import()
            self.import_status.setText("Portable Library archive saved with indexed analytics.")
            self.bridge.status_message.emit("Local video analysis completed.")
            self.refresh()
            self.router.navigate(Route.REVIEW, values[0])
        elif kind == "import_cancelled":
            self.import_status.setText("Import cancelled. You can select another video.")
            self._finish_import()
        elif kind == "import_error" and values:
            self.import_status.setText(f"Import failed: {values[0]}")
            self._finish_import()
        else:
            self.import_status.setText(f"Import worker returned an unexpected event: {kind!r}.")
            self._finish_import()

    def _finish_import(self) -> None:
        self._package_timer.stop()
        self._import_job_id = None
        self.import_progress.hide()
        self.import_progress.setRange(0, 100)
        self.import_progress.setValue(0)
        self.cancel_import_btn.setEnabled(False)
        self._set_import_busy(False)

    def _set_import_busy(self, busy: bool) -> None:
        self.table.setEnabled(not busy)
        self.import_btn.setEnabled(not busy)
        self.open_btn.setEnabled(not busy)
        self.refresh_btn.setEnabled(not busy)
        self.import_reference_cb.setEnabled(not busy)
