"""Archive-contained analytics workspace with calibration comparison and chart export."""

from __future__ import annotations

import csv
import json
import sqlite3
from itertools import pairwise
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...archive import read_manifest
from ...calibration.baseline import load_calibration_file
from ...calibration.models import METRIC_DETAILS, RangeProfile
from ...domain import Calibration
from ..analytics.data import (
    FRAME_METRICS,
    ComparisonResult,
    MetricComparison,
    SessionData,
    compare_archive_with_calibration,
    finite_number,
    load_archive_dataset,
)
from ..components.metric_chart import MetricChart
from ..router import Route, Router
from ..service_bridge import (
    ServiceBridge,
    created_at_timestamp,
    format_created_at,
)
from ..theme import COLOR_BASE, COLOR_BORDER, COLOR_TEXT_MUTED, COLOR_TEXT_PRIMARY

ALL_CHART_METRICS = (*METRIC_DETAILS, *FRAME_METRICS)


def _seconds(value: object) -> str:
    number = finite_number(value)
    return f"{number:.2f} s" if number is not None and number >= 0 else "Unknown"


def _metric_title(metric: str) -> str:
    if metric in METRIC_DETAILS:
        body_part, label, unit = METRIC_DETAILS[metric]
        return f"{body_part} — {label} ({unit.replace('_', ' ')})"
    label, unit, _ = FRAME_METRICS[metric]
    return f"{label} ({unit})"


def _metric_unit(metric: str) -> str:
    if metric in METRIC_DETAILS:
        return METRIC_DETAILS[metric][2].replace("_", " ")
    return FRAME_METRICS[metric][1]


def _default_calibration_directory(bridge: ServiceBridge) -> Path:
    data_dir = Path(bridge.config.data_dir)
    return Path("calibrations") if bridge.config.data_dir == "data" else data_dir / "calibrations"


def _find_combo_data(combo: QComboBox, target: object) -> int:
    normalized = list(target) if isinstance(target, tuple) else target
    for idx in range(combo.count()):
        item = combo.itemData(idx)
        candidate = list(item) if isinstance(item, tuple) else item
        if candidate == normalized:
            return idx
    return -1


class AnalyticsScreen(QWidget):
    def __init__(self, bridge: ServiceBridge, router: Router, parent: QWidget | None = None):
        super().__init__(parent)
        self.bridge = bridge
        self.router = router
        self.dataset: SessionData | None = None
        self.recordings: list[tuple[Path, int, list[float]]] = []
        self._current_source: tuple[Path, int] | None = None
        self._external_archives: list[Path] = []
        self._external_calibrations: dict[str, Calibration] = {}
        self._comparison: ComparisonResult | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        title = QLabel("Archived Recording Analytics")
        title.setProperty("class", "title")
        layout.addWidget(title)
        subtitle = QLabel(
            "Select a .tpr reference or .tp practice archive to plot all posture metrics, "
            "compare against a calibration reference profile, export charts to files, or click a "
            "chart to seek that frame in Review."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Archive:"))
        self.session_combo = QComboBox()
        self.session_combo.setAccessibleName("Library archive and recording analytics")
        self.session_combo.currentIndexChanged.connect(self._select_session)
        toolbar.addWidget(self.session_combo, 1)
        self.open_archive_btn = QPushButton("Open .tp / .tpr…")
        self.open_archive_btn.clicked.connect(self._browse_archive)
        toolbar.addWidget(self.open_archive_btn)
        refresh_btn = QPushButton("Refresh archives")
        refresh_btn.clicked.connect(self.refresh)
        toolbar.addWidget(refresh_btn)
        self.align_btn = QPushButton("Align Time Series")
        self.align_btn.setEnabled(False)
        self.align_btn.setToolTip(
            "Unavailable: no compatible paired time series or calculated alignment is persisted. "
            "Session reference deltas are not temporal alignment."
        )
        toolbar.addWidget(self.align_btn)
        layout.addLayout(toolbar)

        ref_toolbar = QHBoxLayout()
        ref_toolbar.addWidget(QLabel("Calibration reference:"))
        self.calibration_combo = QComboBox()
        self.calibration_combo.setAccessibleName("Calibration line reference")
        self.calibration_combo.currentIndexChanged.connect(self._apply_calibration_reference)
        ref_toolbar.addWidget(self.calibration_combo, 1)
        self.open_calibration_btn = QPushButton("Open .tpc…")
        self.open_calibration_btn.clicked.connect(self._browse_calibration)
        ref_toolbar.addWidget(self.open_calibration_btn)
        self.export_charts_btn = QPushButton("Export All Charts…")
        self.export_charts_btn.setEnabled(False)
        self.export_charts_btn.clicked.connect(lambda: self.export_all_charts())
        ref_toolbar.addWidget(self.export_charts_btn)
        layout.addLayout(ref_toolbar)

        self.state_label = QLabel()
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)
        alignment_note = QLabel(
            "Time alignment unavailable; no offset or scale has been calculated."
        )
        layout.addWidget(alignment_note)
        event_label = QLabel("Posture deviation events · double-click to seek in Review")
        layout.addWidget(event_label)
        self.event_table = QTableWidget(0, 5)
        self.event_table.setHorizontalHeaderLabels(
            ["Metric", "Onset", "Detected", "End", "Peak deviation"]
        )
        self.event_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.event_table.setMaximumHeight(155)
        self.event_table.cellDoubleClicked.connect(self._seek_event)
        layout.addWidget(self.event_table)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        cards = QVBoxLayout(content)
        self.charts: dict[str, MetricChart] = {}
        self.coverage_labels: dict[str, QLabel] = {}
        self.export_buttons: dict[str, QPushButton] = {}
        for metric in ALL_CHART_METRICS:
            card = QFrame()
            card.setProperty("class", "card")
            card_layout = QVBoxLayout(card)
            header_row = QHBoxLayout()
            label = QLabel(
                f"{METRIC_DETAILS[metric][0]} — {METRIC_DETAILS[metric][1]} "
                f"({METRIC_DETAILS[metric][2].replace('_', ' ')})"
                if metric in METRIC_DETAILS
                else f"{FRAME_METRICS[metric][0]} — measurement ({FRAME_METRICS[metric][1]})"
            )
            label.setStyleSheet("font-weight: 600;")
            header_row.addWidget(label, 1)
            export_btn = QPushButton("Export Chart…")
            export_btn.setEnabled(False)
            export_btn.clicked.connect(
                lambda _checked=False, key=metric: self.export_chart(key)
            )
            header_row.addWidget(export_btn)
            card_layout.addLayout(header_row)
            coverage = QLabel()
            coverage.setWordWrap(True)
            card_layout.addWidget(coverage)
            chart = MetricChart()
            chart.seek_requested.connect(self._seek_to_recording)
            card_layout.addWidget(chart)
            cards.addWidget(card)
            self.charts[metric] = chart
            self.coverage_labels[metric] = coverage
            self.export_buttons[metric] = export_btn
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        self.bridge.config_changed.connect(lambda _cfg: self.refresh())
        self.refresh()

    def _clear(self, message: str) -> None:
        self.dataset = None
        self.recordings = []
        self._current_source = None
        self._comparison = None
        self.state_label.setText(message)
        self.event_table.setRowCount(0)
        self.export_charts_btn.setEnabled(False)
        for metric in self.charts:
            self.charts[metric].set_data(None)
            self.coverage_labels[metric].setText("Coverage: Unknown · Events: Unknown")
            self.export_buttons[metric].setEnabled(False)

    def _populate_calibrations(self, preserve: object) -> None:
        self.calibration_combo.blockSignals(True)
        self.calibration_combo.clear()
        self.calibration_combo.addItem("No calibration reference", None)
        self.calibration_combo.addItem("Settings default ranges", "__defaults__")
        seen_paths: set[Path] = set()
        for reference in self.bridge.list_references():
            resolved = reference.path.resolve()
            if not resolved.is_file():
                continue
            seen_paths.add(resolved)
            date_label = format_created_at(reference.created_at, reference.path)
            suffix_tag = ".tpc" if reference.is_archive else "legacy"
            label = (
                f"{reference.name} ({suffix_tag} · {date_label})"
                if date_label != "Recent"
                else f"{reference.name} ({suffix_tag})"
            )
            self.calibration_combo.addItem(label, str(resolved))
        for path_str, cal in self._external_calibrations.items():
            resolved = Path(path_str).resolve()
            if resolved in seen_paths or not resolved.is_file():
                continue
            self.calibration_combo.addItem(f"{cal.name or resolved.name} (external)", path_str)
        index = _find_combo_data(self.calibration_combo, preserve) if preserve is not None else 0
        self.calibration_combo.setCurrentIndex(max(0, index))
        self.calibration_combo.blockSignals(False)

    def refresh(self) -> None:
        selected = self.session_combo.currentData()
        selected_calibration = self.calibration_combo.currentData()
        self._populate_calibrations(selected_calibration)

        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        self.session_combo.addItem("Select a Library recording or reference…", None)
        message = (
            "Select a Library .tp practice or .tpr reference recording to inspect metrics "
            "and compare against a calibration."
        )
        try:
            archives = [
                archive
                for archive in self.bridge.list_library_archives()
                if archive["path"].suffix.lower() in (".tp", ".tpr")
            ]
            archives.sort(
                key=lambda item: -created_at_timestamp(item.get("created_at", ""), item.get("path"))
            )
            listed_paths: set[Path] = set()
            for archive in archives:
                archive_path = archive["path"].resolve()
                listed_paths.add(archive_path)
                manifest = read_manifest(archive_path)
                kind_tag = " (Reference .tpr)" if archive_path.suffix.lower() == ".tpr" else ""
                date_label = format_created_at(archive.get("created_at", ""), archive_path)
                for index, _ in enumerate(manifest["segments"]):
                    label = f"{archive['name']}{kind_tag} · Recording {index:02d}"
                    self.session_combo.addItem(label, (archive_path, index))
                    self.session_combo.setItemData(
                        self.session_combo.count() - 1,
                        f"{archive_path.name} · Created {date_label}",
                        Qt.ItemDataRole.ToolTipRole,
                    )
            for ext_path in self._external_archives:
                resolved = ext_path.resolve()
                if resolved in listed_paths or not resolved.is_file():
                    continue
                manifest = read_manifest(resolved)
                display_name = str(manifest.get("name") or resolved.stem)
                kind_tag = " (Reference .tpr)" if resolved.suffix.lower() == ".tpr" else ""
                for index, _ in enumerate(manifest["segments"]):
                    self.session_combo.addItem(
                        f"{display_name}{kind_tag} · Recording {index:02d} (opened)",
                        (resolved, index),
                    )
            if self.session_combo.count() == 1:
                message = (
                    "No Library practice archives (.tp) or reference (.tpr) recordings. "
                    "Import or save a .tp or .tpr recording first."
                )
        except (OSError, sqlite3.Error, ValueError, BadZipFile) as exc:
            message = f"Analytics unavailable: {exc}"
        index = _find_combo_data(self.session_combo, selected) if selected else 0
        self.session_combo.setCurrentIndex(max(0, index))
        self.session_combo.blockSignals(False)
        self._clear(message)
        if self.session_combo.currentData():
            self._select_session()

    def load_archive(
        self,
        archive_path: Path,
        segment_index: int = 0,
        *,
        calibration_path: Path | None = None,
    ) -> None:
        """Open a specific .tp or .tpr archive (and optional calibration) in Analytics."""
        resolved = Path(archive_path).expanduser().resolve()
        if resolved.suffix.lower() not in (".tp", ".tpr") or not resolved.is_file():
            self.bridge.status_message.emit(f"Archive not found or unsupported: {resolved}")
            return
        if resolved not in self._external_archives:
            self._external_archives.append(resolved)
        if calibration_path is not None:
            self._select_calibration_path(calibration_path)
        self.refresh()
        target_index = _find_combo_data(self.session_combo, (resolved, segment_index))
        if target_index >= 0:
            self.session_combo.setCurrentIndex(target_index)

    def _browse_archive(self) -> None:
        start_dir = str(Path(self.bridge.config.data_dir))
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Open practice (.tp) or reference (.tpr) archive",
            start_dir,
            "TangoPosture archives (*.tp *.tpr);;All files (*)",
        )
        if chosen:
            self.load_archive(Path(chosen), 0)

    def _select_calibration_path(self, cal_path: Path) -> bool:
        resolved = Path(cal_path).expanduser().resolve()
        cal = load_calibration_file(str(resolved), _default_calibration_directory(self.bridge))
        if cal is None:
            self.bridge.status_message.emit(f"Could not read calibration file: {resolved.name}")
            return False
        self._external_calibrations[str(resolved)] = cal
        self._populate_calibrations(str(resolved))
        return True

    def _browse_calibration(self) -> None:
        start_dir = str(_default_calibration_directory(self.bridge))
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Select calibration profile (.tpc or legacy JSON)",
            start_dir,
            "Calibration profiles (*.tpc *.json);;All files (*)",
        )
        if chosen and self._select_calibration_path(Path(chosen)):
            self._apply_calibration_reference()

    def _resolve_selected_calibration(self) -> RangeProfile | Calibration | str | None:
        selected = self.calibration_combo.currentData()
        if selected is None or selected == "__defaults__":
            return selected
        if isinstance(selected, str):
            cached = self._external_calibrations.get(selected)
            if cached is not None:
                return cached
            path = Path(selected)
            cal = load_calibration_file(str(path), _default_calibration_directory(self.bridge))
            if cal is None:
                raise ValueError(f"Selected calibration is unavailable: {path.name}")
            return cal
        return None

    def _select_session(self) -> None:
        selected = self.session_combo.currentData()
        self._clear("Select a Library recording or reference to inspect analytics.")
        if not selected:
            return
        path, index = selected
        try:
            dataset = load_archive_dataset(path, index, self.bridge.config)
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, BadZipFile) as exc:
            self._clear(f"No compatible dataset: {exc}")
            return
        self.dataset = dataset
        self._current_source = (path, index)
        try:
            manifest = read_manifest(path)
            with ZipFile(path) as archive:
                metadata = json.loads(archive.read(manifest["segments"][index]["metadata"]))
            if not isinstance(metadata, dict):
                raise TypeError("Invalid archived recording metadata")
            timestamps = metadata.get("frame_timestamps")
            if (
                not isinstance(timestamps, list)
                or any(
                    finite_number(value) is None or value < 0 for value in timestamps
                )
                or any(earlier >= later for earlier, later in pairwise(timestamps))
            ):
                raise ValueError("Invalid archived recording frame timeline")
            self.recordings = [(path, index, timestamps)]
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, BadZipFile) as exc:
            self.recordings = []
            self.bridge.status_message.emit(f"Session video navigation unavailable: {exc}")
        self._apply_calibration_reference()

    def _apply_calibration_reference(self) -> None:
        dataset = self.dataset
        source = self._current_source
        if dataset is None or source is None:
            return
        path, index = source
        cal_error = ""
        try:
            calibration = self._resolve_selected_calibration()
            comparison = compare_archive_with_calibration(
                path, index, dataset, self.bridge.config, calibration
            )
        except (OSError, ValueError, TypeError, KeyError, BadZipFile) as exc:
            cal_error = str(exc)
            comparison = compare_archive_with_calibration(
                path, index, dataset, self.bridge.config, None
            )
        self._comparison = comparison

        time_end = max(
            (t for item in comparison.metrics.values() for t, _ in item.data.points),
            default=0.0,
        )
        note = (
            f"Lines break at null/unknown samples and gaps > {dataset.max_gap:g} s."
            if dataset.max_gap is not None
            else "Tracking gap limit unknown: samples are not connected."
        )
        archive_kind = "Reference (.tpr)" if path.suffix.lower() == ".tpr" else "Practice (.tp)"
        cal_summary = (
            f"Calibration error: {cal_error}"
            if cal_error
            else f"Calibration: {comparison.calibration_name}"
        )
        extra_note = f" {comparison.note}" if comparison.note else ""
        self.state_label.setText(
            f"{archive_kind} · Session {dataset.identifier} · {dataset.status} · "
            f"Active duration: {_seconds(dataset.active_duration)} · {cal_summary}. "
            f"{note}{extra_note}"
        )

        self.event_table.setRowCount(len(comparison.events))
        for row, event in enumerate(comparison.events):
            values = (
                event.metric,
                _seconds(event.onset - dataset.origin_timestamp),
                _seconds(event.detection - dataset.origin_timestamp),
                _seconds(event.end - dataset.origin_timestamp) if event.end is not None else "Open",
                f"{event.peak_deviation:+.3g}" if event.peak_deviation is not None else "Unknown",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, event.onset)
                self.event_table.setItem(row, column, item)

        calibration_selected = (
            self.calibration_combo.currentData() is not None and not cal_error
        )
        for metric in self.charts:
            self.charts[metric].set_data(None)
            self.export_buttons[metric].setEnabled(False)
            self.coverage_labels[metric].setText(
                "Unavailable: no approved / stored observations for this metric."
            )

        any_chart_enabled = False
        for metric, comp_item in comparison.metrics.items():
            if metric not in self.charts:
                continue
            data = comp_item.data
            chart = self.charts[metric]
            chart.set_data(
                data,
                time_end,
                dataset.origin_timestamp,
                bounds=comp_item.bounds,
                reference_line=comp_item.reference_line,
                states=comp_item.states,
            )
            enabled = chart.isEnabled()
            self.export_buttons[metric].setEnabled(enabled)
            any_chart_enabled = any_chart_enabled or enabled
            self.coverage_labels[metric].setText(
                self._format_coverage_label(metric, comp_item, calibration_selected)
            )
        self.export_charts_btn.setEnabled(any_chart_enabled)

    def _format_coverage_label(
        self,
        metric: str,
        item: MetricComparison,
        calibration_selected: bool,
    ) -> str:
        data = item.data
        ratio = finite_number(data.summary.get("valid_coverage"))
        coverage = f"{ratio:.1%}" if ratio is not None and 0 <= ratio <= 1 else "Unknown"
        valid = _seconds(data.summary.get("valid_duration"))
        unknown = _seconds(data.summary.get("unknown_duration"))
        available = sum(value is not None for _, value in data.points)
        total = len(data.points)

        if not calibration_selected:
            if data.summary.get("source") == "observations":
                return (
                    f"Valid coverage: {coverage} · Valid: {valid} · Unknown: {unknown} · "
                    f"Measured samples: {available}/{total}"
                )
            if metric in METRIC_DETAILS:
                return (
                    f"Valid coverage: {coverage} · Valid: {valid} · Unknown: {unknown} · "
                    f"Delta samples: {available}/{total} · "
                    f"Persisted events: {data.event_count} ({data.open_event_count} open)"
                )
            return (
                f"Stored measurement samples: {available}/{total}. "
                "Missing observations remain unknown; these are not posture scores."
            )

        parts: list[str] = []
        if item.bounds is not None:
            parts.append(f"Range: {item.bounds[0]:.3g} … {item.bounds[1]:.3g} {_metric_unit(metric)}")
        elif item.calibration_label:
            parts.append(item.calibration_label)
        if item.reference_line is not None:
            parts.append(f"Ref: {item.reference_line:.3g}")
        within_ratio = finite_number(data.summary.get("within_reference_ratio"))
        if within_ratio is not None and 0 <= within_ratio <= 1:
            parts.append(f"Within range: {within_ratio:.1%}")
        parts.append(f"Valid coverage: {coverage}")
        parts.append(f"Samples: {available}/{total}")
        parts.append(f"Events: {data.event_count} ({data.open_event_count} open)")
        if item.note:
            parts.append(item.note)
        return " · ".join(parts)

    def _chart_export_subtitle(self, metric: str) -> str:
        comp = self._comparison
        source_name = self._current_source[0].name if self._current_source else "archive"
        parts = [f"Archive: {source_name}"]
        if comp is not None:
            parts.append(f"Calibration: {comp.calibration_name}")
            item = comp.metrics.get(metric)
            if item is not None:
                if item.bounds is not None:
                    parts.append(
                        f"Range: {item.bounds[0]:.3g}..{item.bounds[1]:.3g} {_metric_unit(metric)}"
                    )
                if item.reference_line is not None:
                    parts.append(f"Ref: {item.reference_line:.3g}")
                within = finite_number(item.data.summary.get("within_reference_ratio"))
                if within is not None:
                    parts.append(f"Within: {within:.1%}")
        return " · ".join(parts)

    def _active_metrics(self) -> list[str]:
        return [metric for metric in ALL_CHART_METRICS if self.charts[metric].isEnabled()]

    def export_chart(self, metric: str, destination: Path | None = None) -> Path | None:
        """Export a single metric chart to PNG, SVG, CSV, or JSON."""
        chart = self.charts.get(metric)
        if chart is None or not chart.isEnabled():
            self.bridge.status_message.emit(f"Chart {metric} has no active data to export.")
            return None
        source_stem = self._current_source[0].stem if self._current_source else "analytics"
        if destination is None:
            chosen, _ = QFileDialog.getSaveFileName(
                self,
                f"Export {_metric_title(metric)}",
                f"{source_stem}_{metric}.png",
                "PNG Image (*.png);;SVG Vector (*.svg);;CSV Table (*.csv);;JSON Data (*.json)",
            )
            if not chosen:
                return None
            destination = Path(chosen)
        destination = Path(destination)
        title = _metric_title(metric)
        subtitle = self._chart_export_subtitle(metric)
        if destination.suffix.lower() == ".json":
            destination.parent.mkdir(parents=True, exist_ok=True)
            item = self._comparison.metrics.get(metric) if self._comparison else None
            payload = {
                "metric": metric,
                "title": title,
                "unit": _metric_unit(metric),
                "archive": str(self._current_source[0].name) if self._current_source else "",
                "segment_index": self._current_source[1] if self._current_source else 0,
                "calibration": self._comparison.calibration_name if self._comparison else "",
                "bounds": list(chart.bounds) if chart.bounds is not None else None,
                "reference_line": chart.reference_line,
                "summary": item.data.summary if item is not None else {},
                "samples": [
                    {
                        "timestamp_s": ts,
                        "value": val,
                        "state": chart.states.get(ts, "UNKNOWN" if val is None else ""),
                    }
                    for ts, val in (chart.data.points if chart.data else [])
                ],
            }
            destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            chart.export_to_file(destination, title=title, subtitle=subtitle)
        self.bridge.status_message.emit(f"Exported {metric} chart to {destination.name}")
        return destination

    def export_all_charts(self, destination: Path | None = None) -> Path | None:
        """Export all active metric charts to a single composite PNG/SVG or combined CSV/JSON."""
        active = self._active_metrics()
        if not active:
            self.bridge.status_message.emit("No active metric charts to export.")
            return None
        source_stem = self._current_source[0].stem if self._current_source else "analytics"
        if destination is None:
            chosen, _ = QFileDialog.getSaveFileName(
                self,
                "Export all posture metric charts",
                f"{source_stem}_all_metrics.png",
                "PNG Composite Image (*.png);;SVG Vector Document (*.svg);;"
                "CSV Combined Table (*.csv);;JSON Export (*.json)",
            )
            if not chosen:
                return None
            destination = Path(chosen)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = destination.suffix.lower()
        comp = self._comparison
        header_title = (
            f"TangoPosture Analytics — {self._current_source[0].name} "
            f"(Recording {self._current_source[1]:02d})"
            if self._current_source
            else "TangoPosture Analytics"
        )
        header_sub = (
            f"Calibration reference: {comp.calibration_name}"
            + (f" · {comp.note}" if comp.note else "")
            if comp is not None
            else ""
        )

        if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
            width = 960
            header_h = 68
            chart_h = 240
            total_h = header_h + chart_h * len(active)
            composite = QImage(width, total_h, QImage.Format.Format_RGB32)
            composite.fill(QColor(COLOR_BASE))
            painter = QPainter(composite)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(0, 0, width, header_h, QColor(COLOR_BASE))
            title_font = QFont(painter.font())
            title_font.setPointSize(13)
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.setPen(QColor(COLOR_TEXT_PRIMARY))
            painter.drawText(16, 28, header_title)
            if header_sub:
                sub_font = QFont(painter.font())
                sub_font.setPointSize(9)
                sub_font.setBold(False)
                painter.setFont(sub_font)
                painter.setPen(QColor(COLOR_TEXT_MUTED))
                painter.drawText(16, 48, header_sub)
            painter.setPen(QColor(COLOR_BORDER))
            painter.drawLine(0, header_h - 1, width, header_h - 1)
            for idx, metric in enumerate(active):
                tile = self.charts[metric].render_to_image(
                    width=width,
                    height=chart_h,
                    title=_metric_title(metric),
                    subtitle=self._chart_export_subtitle(metric),
                )
                painter.drawImage(0, header_h + idx * chart_h, tile)
            painter.end()
            if not composite.save(str(destination)):
                raise OSError(f"Could not save composite chart image to {destination}")
        elif suffix == ".svg":
            from xml.sax.saxutils import escape

            width = 960
            header_h = 64
            chart_h = 240
            total_h = header_h + chart_h * len(active)
            svg_parts = [
                '<?xml version="1.0" encoding="UTF-8"?>',
                (
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h}" '
                    f'viewBox="0 0 {width} {total_h}">'
                ),
                f'<rect x="0" y="0" width="{width}" height="{total_h}" fill="{COLOR_BASE}"/>',
                (
                    f'<text x="16" y="28" fill="{COLOR_TEXT_PRIMARY}" font-family="sans-serif" '
                    f'font-size="15" font-weight="bold">{escape(header_title)}</text>'
                ),
                (
                    f'<text x="16" y="48" fill="{COLOR_TEXT_MUTED}" font-family="sans-serif" '
                    f'font-size="11">{escape(header_sub)}</text>'
                ),
            ]
            for idx, metric in enumerate(active):
                svg_parts.append(
                    self.charts[metric].svg_fragment(
                        0.0,
                        float(header_h + idx * chart_h),
                        width=width,
                        height=chart_h,
                        title=_metric_title(metric),
                        subtitle=self._chart_export_subtitle(metric),
                    )
                )
            svg_parts.append("</svg>\n")
            destination.write_text("\n".join(svg_parts), encoding="utf-8")
        elif suffix == ".csv":
            with destination.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "metric",
                    "title",
                    "unit",
                    "timestamp_s",
                    "value",
                    "state",
                    "min_bound",
                    "max_bound",
                    "reference_line",
                ])
                for metric in active:
                    chart = self.charts[metric]
                    min_b = f"{chart.bounds[0]:.6g}" if chart.bounds is not None else ""
                    max_b = f"{chart.bounds[1]:.6g}" if chart.bounds is not None else ""
                    ref_v = f"{chart.reference_line:.6g}" if chart.reference_line is not None else ""
                    m_title = _metric_title(metric)
                    m_unit = _metric_unit(metric)
                    for ts, val in (chart.data.points if chart.data else []):
                        writer.writerow([
                            metric,
                            m_title,
                            m_unit,
                            f"{ts:.6f}",
                            "" if val is None else f"{val:.6g}",
                            chart.states.get(ts, "UNKNOWN" if val is None else ""),
                            min_b,
                            max_b,
                            ref_v,
                        ])
        elif suffix == ".json":
            dataset = self.dataset
            origin = dataset.origin_timestamp if dataset is not None else 0.0
            payload: dict[str, Any] = {
                "archive": str(self._current_source[0].name) if self._current_source else "",
                "segment_index": self._current_source[1] if self._current_source else 0,
                "active_duration_s": dataset.active_duration if dataset is not None else None,
                "calibration": comp.calibration_name if comp is not None else "",
                "calibration_note": comp.note if comp is not None else "",
                "events": [
                    {
                        "metric": ev.metric,
                        "onset_s": ev.onset - origin,
                        "detection_s": ev.detection - origin,
                        "end_s": ev.end - origin if ev.end is not None else None,
                        "peak_deviation": ev.peak_deviation,
                        "end_reason": ev.end_reason,
                    }
                    for ev in (comp.events if comp is not None else [])
                ],
                "metrics": {},
            }
            for metric in active:
                chart = self.charts[metric]
                item = comp.metrics.get(metric) if comp is not None else None
                payload["metrics"][metric] = {
                    "title": _metric_title(metric),
                    "unit": _metric_unit(metric),
                    "bounds": list(chart.bounds) if chart.bounds is not None else None,
                    "reference_line": chart.reference_line,
                    "summary": item.data.summary if item is not None else {},
                    "samples": [
                        {
                            "timestamp_s": ts,
                            "value": val,
                            "state": chart.states.get(ts, "UNKNOWN" if val is None else ""),
                        }
                        for ts, val in (chart.data.points if chart.data else [])
                    ],
                }
            destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported multi-chart export format: {suffix or '(none)'}")

        self.bridge.status_message.emit(f"Exported {len(active)} charts to {destination.name}")
        return destination

    def _seek_to_recording(self, timestamp: float) -> None:
        match = next(
            (
                (path, index)
                for path, index, timestamps in self.recordings
                if timestamps and timestamps[0] <= timestamp <= timestamps[-1]
            ),
            None,
        )
        if match is None:
            self.bridge.status_message.emit(
                "No archived video frame covers this chart timestamp."
            )
            return
        path, index = match
        self.router.navigate(
            Route.REVIEW,
            {"archive_path": path, "segment_index": index, "timestamp": timestamp},
        )

    def _seek_event(self, row: int, _column: int) -> None:
        item = self.event_table.item(row, 0)
        if item is None:
            return
        timestamp = finite_number(item.data(Qt.ItemDataRole.UserRole))
        if timestamp is not None:
            self._seek_to_recording(timestamp)
