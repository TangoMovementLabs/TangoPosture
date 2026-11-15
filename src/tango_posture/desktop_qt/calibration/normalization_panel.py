"""Qt settings panel for per-metric Review measurement normalization."""

from __future__ import annotations

import math
from pathlib import Path
import threading
from typing import Any, Callable, Mapping

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...archive import read_analytics
from ...calibration.archives import read_range_profile
from ...calibration.models import ApprovedRange, METRIC_DETAILS, Passage, RangeProfile
from ...calibration.normalization import (
    HoldSourceIdentity,
    HoldWindow,
    HoldWindowStats,
    MatchedHoldResult,
    MeasurementNormalizationConfig,
    MetricNormalizationRule,
    NormalizationPairRule,
    apply_metric_offset,
    build_source_identity,
    evaluate_matched_holds,
    evaluate_pair_applicability,
    find_matching_pair,
    in_physical_domain,
    make_pair_rule_id,
    profile_content_digest,
    resolve_profile_reference_source,
    resolve_raw_media_sha256,
    select_archive_frames,
    utc_timestamp_now,
)
from ...config import Config
from ..service_bridge import ArchiveKind, ServiceBridge
from ..theme import (
    COLOR_BORDER,
    COLOR_COPPER,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_TEXT_MUTED,
    COLOR_WARNING,
)


class _HoldComputeSignals(QObject):
    """Bridge signals for asynchronous matched-hold calculations."""

    finished = Signal(int, object, object, object, str)


class NullableDoubleEdit(QLineEdit):
    """Line edit that represents blank text as `None` and validates finite floats."""

    valueChanged = Signal()

    def __init__(
        self,
        *,
        placeholder: str = "",
        minimum: float = -1e6,
        maximum: float = 1e6,
        decimals: int = 4,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._minimum = minimum
        self._maximum = maximum
        self.setPlaceholderText(placeholder)
        validator = QDoubleValidator(minimum, maximum, decimals, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(validator)
        self.textChanged.connect(lambda _text: self.valueChanged.emit())

    def value(self) -> float | None:
        raw = self.text().strip().replace(",", ".")
        if not raw or raw in ("-", "+", ".", "-.", "+."):
            return None
        try:
            numeric = float(raw)
        except ValueError:
            return None
        if not math.isfinite(numeric):
            return None
        if numeric < self._minimum or numeric > self._maximum:
            return None
        return numeric

    def setValue(self, value: float | None) -> None:
        blocker = self.signalsBlocked()
        self.blockSignals(True)
        try:
            if value is None or not math.isfinite(float(value)):
                self.setText("")
            else:
                self.setText(f"{float(value):.4g}")
        finally:
            self.blockSignals(blocker)
        self.valueChanged.emit()


def _format_metric_value(value: float, unit: str) -> str:
    if unit == "degrees":
        return f"{value:.2f}°"
    return f"{value:.3f} {unit}"


def _format_signed_offset(value: float, unit: str) -> str:
    sign = "+" if value >= 0 else "−"
    mag = abs(value)
    if unit == "degrees":
        return f"{sign}{mag:.2f}°"
    return f"{sign}{mag:.3f} {unit}"


def _make_window_spin(maximum: float = 86400.0) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, maximum)
    spin.setDecimals(2)
    spin.setSingleStep(0.25)
    spin.setSuffix(" s")
    spin.setMaximumWidth(115)
    return spin


class MetricNormalizationRowWidget(QFrame):
    """Per-metric editor card covering mode, manual/matched holds, and preview."""

    changed = Signal()
    compute_requested = Signal(str)

    def __init__(
        self,
        metric: str,
        target: ApprovedRange,
        *,
        playhead_provider: Callable[[], float | None],
        review_frame_provider: Callable[[str], float | None],
        default_tolerance_provider: Callable[[], float],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.metric = metric
        self.target = target
        self._playhead_provider = playhead_provider
        self._review_frame_provider = review_frame_provider
        self._default_tolerance_provider = default_tolerance_provider
        self._verified = False
        self._verification_error: float | None = None
        self._status_message = ""
        self._fit_ref_stats: HoldWindowStats | None = None
        self._fit_prac_stats: HoldWindowStats | None = None
        self._check_ref_stats: HoldWindowStats | None = None
        self._check_prac_stats: HoldWindowStats | None = None
        self._dirty = False

        _, label, unit = METRIC_DETAILS[metric]
        self._unit = unit
        self.setProperty("class", "card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Header row
        header = QHBoxLayout()
        header.setSpacing(8)
        title_lbl = QLabel(f"<b>{label}</b>")
        header.addWidget(title_lbl)
        bounds_lbl = QLabel(
            f"Approved reference range: {target.lower:.3g} .. {target.upper:.3g} {unit}"
        )
        bounds_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
        header.addWidget(bounds_lbl)
        header.addStretch(1)
        self.status_badge = QLabel("Direct comparison")
        self.status_badge.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 12px; font-weight: 600;"
        )
        header.addWidget(self.status_badge)
        layout.addLayout(header)

        # Mode and method row
        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        self.direct_radio = QRadioButton("Direct comparison")
        self.adjust_radio = QRadioButton("Use adjustment")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.direct_radio)
        self.mode_group.addButton(self.adjust_radio)
        self.direct_radio.setChecked(True)
        mode_row.addWidget(self.direct_radio)
        mode_row.addWidget(self.adjust_radio)

        self.method_combo = QComboBox()
        self.method_combo.addItem("Matched holds", "matched_holds")
        self.method_combo.addItem("Manual offset", "manual")
        mode_row.addWidget(QLabel("Method:"))
        mode_row.addWidget(self.method_combo)

        self.show_calc_cb = QCheckBox("Show calculation in Review")
        self.show_calc_cb.setChecked(True)
        mode_row.addWidget(self.show_calc_cb)
        mode_row.addStretch(1)

        self.reset_btn = QPushButton("Reset adjustment")
        self.reset_btn.clicked.connect(self.reset_rule)
        mode_row.addWidget(self.reset_btn)
        layout.addLayout(mode_row)

        # Manual container
        self.manual_container = QWidget()
        manual_layout = QHBoxLayout(self.manual_container)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(8)
        manual_layout.addWidget(QLabel("Signed offset:"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-360.0, 360.0)
        self.offset_spin.setDecimals(4)
        self.offset_spin.setSingleStep(0.1)
        self.offset_spin.setSuffix("°" if unit == "degrees" else f" {unit}")
        manual_layout.addWidget(self.offset_spin)
        self.manual_note_lbl = QLabel("Manual · unverified (cannot verify baseline neutrality without matched holds)")
        self.manual_note_lbl.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 12px;")
        manual_layout.addWidget(self.manual_note_lbl, 1)
        layout.addWidget(self.manual_container)

        # Matched holds container
        self.matched_container = QFrame()
        self.matched_container.setStyleSheet(f"border: 1px solid {COLOR_BORDER}; border-radius: 4px;")
        matched_layout = QVBoxLayout(self.matched_container)
        matched_layout.setContentsMargins(10, 8, 10, 8)
        matched_layout.setSpacing(6)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.addWidget(QLabel("<b>Hold window</b>"), 0, 0)
        grid.addWidget(QLabel("<b>Start</b>"), 0, 1)
        grid.addWidget(QLabel("<b>End</b>"), 0, 2)
        grid.addWidget(QLabel("<b>Playhead</b>"), 0, 3)

        self.fit_ref_start = _make_window_spin()
        self.fit_ref_end = _make_window_spin()
        self.fit_prac_start = _make_window_spin()
        self.fit_prac_end = _make_window_spin()
        self.check_ref_start = _make_window_spin()
        self.check_ref_end = _make_window_spin()
        self.check_prac_start = _make_window_spin()
        self.check_prac_end = _make_window_spin()

        self.fit_prac_set_start_btn = QPushButton("Set start from playhead")
        self.fit_prac_set_end_btn = QPushButton("Set end from playhead")
        self.check_prac_set_start_btn = QPushButton("Set start from playhead")
        self.check_prac_set_end_btn = QPushButton("Set end from playhead")

        self.fit_prac_set_start_btn.clicked.connect(
            lambda: self._apply_playhead(self.fit_prac_start, self.fit_prac_end, is_start=True)
        )
        self.fit_prac_set_end_btn.clicked.connect(
            lambda: self._apply_playhead(self.fit_prac_start, self.fit_prac_end, is_start=False)
        )
        self.check_prac_set_start_btn.clicked.connect(
            lambda: self._apply_playhead(self.check_prac_start, self.check_prac_end, is_start=True)
        )
        self.check_prac_set_end_btn.clicked.connect(
            lambda: self._apply_playhead(self.check_prac_start, self.check_prac_end, is_start=False)
        )

        grid.addWidget(QLabel("Reference Fit hold:"), 1, 0)
        grid.addWidget(self.fit_ref_start, 1, 1)
        grid.addWidget(self.fit_ref_end, 1, 2)

        grid.addWidget(QLabel("Practice Fit hold:"), 2, 0)
        grid.addWidget(self.fit_prac_start, 2, 1)
        grid.addWidget(self.fit_prac_end, 2, 2)
        fit_btn_box = QHBoxLayout()
        fit_btn_box.setContentsMargins(0, 0, 0, 0)
        fit_btn_box.setSpacing(4)
        fit_btn_box.addWidget(self.fit_prac_set_start_btn)
        fit_btn_box.addWidget(self.fit_prac_set_end_btn)
        grid.addLayout(fit_btn_box, 2, 3)

        grid.addWidget(QLabel("Reference Check hold:"), 3, 0)
        grid.addWidget(self.check_ref_start, 3, 1)
        grid.addWidget(self.check_ref_end, 3, 2)

        grid.addWidget(QLabel("Practice Check hold:"), 4, 0)
        grid.addWidget(self.check_prac_start, 4, 1)
        grid.addWidget(self.check_prac_end, 4, 2)
        chk_btn_box = QHBoxLayout()
        chk_btn_box.setContentsMargins(0, 0, 0, 0)
        chk_btn_box.setSpacing(4)
        chk_btn_box.addWidget(self.check_prac_set_start_btn)
        chk_btn_box.addWidget(self.check_prac_set_end_btn)
        grid.addLayout(chk_btn_box, 4, 3)

        matched_layout.addLayout(grid)

        tol_row = QHBoxLayout()
        tol_row.setSpacing(8)
        tol_row.addWidget(QLabel("Maximum verification error:"))
        self.max_error_edit = NullableDoubleEdit(
            placeholder="Use default", minimum=0.0001, maximum=180.0, decimals=4,
        )
        self.max_error_edit.setMaximumWidth(120)
        tol_row.addWidget(self.max_error_edit)
        self.compute_btn = QPushButton("Compute && verify holds")
        self.compute_btn.clicked.connect(lambda: self.compute_requested.emit(self.metric))
        tol_row.addWidget(self.compute_btn)
        tol_row.addStretch(1)
        matched_layout.addLayout(tol_row)

        self.hold_summary_lbl = QLabel("Hold quality: not yet computed")
        self.hold_summary_lbl.setWordWrap(True)
        self.hold_summary_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
        matched_layout.addWidget(self.hold_summary_lbl)
        layout.addWidget(self.matched_container)

        # Preview row
        preview_row = QHBoxLayout()
        preview_row.setSpacing(8)
        preview_row.addWidget(QLabel("Raw measurement (Preview input):"))
        self.preview_input = NullableDoubleEdit(placeholder="blank")
        self.preview_input.setMaximumWidth(110)
        preview_row.addWidget(self.preview_input)
        self.use_review_frame_btn = QPushButton("Use current Review frame")
        self.use_review_frame_btn.clicked.connect(self._apply_current_review_frame)
        preview_row.addWidget(self.use_review_frame_btn)
        self.preview_output_lbl = QLabel("Value on reference scale: —")
        self.preview_output_lbl.setWordWrap(True)
        preview_row.addWidget(self.preview_output_lbl, 1)
        layout.addLayout(preview_row)

        # Signals
        self.direct_radio.toggled.connect(self._on_widget_edited)
        self.adjust_radio.toggled.connect(self._on_widget_edited)
        self.method_combo.currentIndexChanged.connect(self._on_widget_edited)
        self.offset_spin.valueChanged.connect(self._on_manual_offset_edited)
        self.show_calc_cb.toggled.connect(self._on_widget_edited)
        self.max_error_edit.valueChanged.connect(self._on_matched_param_edited)
        for spin in (
            self.fit_ref_start,
            self.fit_ref_end,
            self.fit_prac_start,
            self.fit_prac_end,
            self.check_ref_start,
            self.check_ref_end,
            self.check_prac_start,
            self.check_prac_end,
        ):
            spin.valueChanged.connect(self._on_matched_param_edited)
        self.preview_input.valueChanged.connect(self.update_preview)

        self._sync_visibility()
        self.update_preview()

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_clean(self) -> None:
        self._dirty = False

    def _apply_playhead(self, start_spin: QDoubleSpinBox, end_spin: QDoubleSpinBox, *, is_start: bool) -> None:
        ts = self._playhead_provider()
        if ts is None or not math.isfinite(ts):
            return
        self._dirty = True
        if is_start:
            start_spin.setValue(max(0.0, float(ts)))
            if end_spin.value() <= start_spin.value():
                end_spin.setValue(start_spin.value() + 2.0)
        else:
            end_spin.setValue(max(0.0, float(ts)))

    def _apply_current_review_frame(self) -> None:
        val = self._review_frame_provider(self.metric)
        self.preview_input.setValue(val)
        self.update_preview()

    def _on_widget_edited(self) -> None:
        self._dirty = True
        self._sync_visibility()
        self.update_preview()
        self.changed.emit()

    def _on_manual_offset_edited(self) -> None:
        self._dirty = True
        if self.method_combo.currentData() == "manual":
            self._verified = False
            self._verification_error = None
            self._status_message = "Manual · unverified"
        self.update_preview()
        self._update_status_badge()
        self.changed.emit()

    def _on_matched_param_edited(self) -> None:
        self._dirty = True
        if self.method_combo.currentData() == "matched_holds":
            self._verified = False
            self._status_message = "Hold windows or tolerance changed; recompute required"
        self.update_preview()
        self._update_status_badge()
        self.changed.emit()

    def _sync_visibility(self) -> None:
        adjust_on = self.adjust_radio.isChecked()
        method = str(self.method_combo.currentData() or "manual")
        self.method_combo.setEnabled(adjust_on)
        self.show_calc_cb.setEnabled(adjust_on)
        self.manual_container.setVisible(adjust_on and method == "manual")
        self.matched_container.setVisible(adjust_on and method == "matched_holds")
        self._update_status_badge()

    def _update_status_badge(self) -> None:
        if not self.adjust_radio.isChecked():
            self.status_badge.setText("Direct comparison")
            self.status_badge.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
            return
        method = str(self.method_combo.currentData() or "manual")
        if method == "manual":
            self.status_badge.setText("Manual · unverified")
            self.status_badge.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 12px; font-weight: 600;")
            return
        if self._verified and self._verification_error is not None:
            tol = self.max_error_edit.value() or self._default_tolerance_provider()
            if self._verification_error <= tol + 1e-9:
                self.status_badge.setText(f"Verified on check hold (error {self._verification_error:.2f})")
                self.status_badge.setStyleSheet(f"color: {COLOR_SUCCESS}; font-size: 12px; font-weight: 600;")
                return
        self.status_badge.setText("Needs recheck")
        self.status_badge.setStyleSheet(f"color: {COLOR_DANGER}; font-size: 12px; font-weight: 600;")

    def reset_rule(self) -> None:
        """Reset this metric rule back to Direct comparison with zero offset."""
        self._dirty = True
        blocker = self.blockSignals(True)
        try:
            self.direct_radio.setChecked(True)
            self.method_combo.setCurrentIndex(0)
            self.offset_spin.setValue(0.0)
            self.show_calc_cb.setChecked(True)
            self.max_error_edit.setValue(None)
            for spin in (
                self.fit_ref_start,
                self.fit_ref_end,
                self.fit_prac_start,
                self.fit_prac_end,
                self.check_ref_start,
                self.check_ref_end,
                self.check_prac_start,
                self.check_prac_end,
            ):
                spin.setValue(0.0)
            self._verified = False
            self._verification_error = None
            self._status_message = ""
            self._fit_ref_stats = None
            self._fit_prac_stats = None
            self._check_ref_stats = None
            self._check_prac_stats = None
            self.hold_summary_lbl.setText("Hold quality: reset to Direct comparison")
            self._sync_visibility()
        finally:
            self.blockSignals(blocker)
        self.update_preview()
        self.changed.emit()

    def _window_from_spins(self, start_spin: QDoubleSpinBox, end_spin: QDoubleSpinBox) -> HoldWindow | None:
        s = float(start_spin.value())
        e = float(end_spin.value())
        if e > s:
            return HoldWindow(start=s, end=e)
        return None

    def to_rule(self) -> MetricNormalizationRule:
        """Return a Pydantic `MetricNormalizationRule` matching the row's widgets."""
        method = str(self.method_combo.currentData() or "manual")
        return MetricNormalizationRule(
            enabled=self.adjust_radio.isChecked(),
            method="matched_holds" if method == "matched_holds" else "manual",
            offset=float(self.offset_spin.value()),
            show_calculation_in_review=self.show_calc_cb.isChecked(),
            max_verification_error=self.max_error_edit.value(),
            fit_reference_window=self._window_from_spins(self.fit_ref_start, self.fit_ref_end),
            fit_practice_window=self._window_from_spins(self.fit_prac_start, self.fit_prac_end),
            check_reference_window=self._window_from_spins(self.check_ref_start, self.check_ref_end),
            check_practice_window=self._window_from_spins(self.check_prac_start, self.check_prac_end),
            fit_reference_stats=self._fit_ref_stats,
            fit_practice_stats=self._fit_prac_stats,
            check_reference_stats=self._check_ref_stats,
            check_practice_stats=self._check_prac_stats,
            verification_error=self._verification_error,
            verified=self._verified,
            status_message=self._status_message,
        )

    def set_rule(self, rule: MetricNormalizationRule) -> None:
        """Populate UI widgets from `rule` without marking dirty."""
        widgets = (
            self.direct_radio,
            self.adjust_radio,
            self.method_combo,
            self.offset_spin,
            self.show_calc_cb,
            self.max_error_edit,
            self.fit_ref_start,
            self.fit_ref_end,
            self.fit_prac_start,
            self.fit_prac_end,
            self.check_ref_start,
            self.check_ref_end,
            self.check_prac_start,
            self.check_prac_end,
        )
        blocked = [w.blockSignals(True) for w in widgets]
        try:
            self.adjust_radio.setChecked(bool(rule.enabled))
            self.direct_radio.setChecked(not rule.enabled)
            idx = self.method_combo.findData(rule.method)
            self.method_combo.setCurrentIndex(max(0, idx))
            self.offset_spin.setValue(float(rule.offset))
            self.show_calc_cb.setChecked(bool(rule.show_calculation_in_review))
            self.max_error_edit.setValue(rule.max_verification_error)

            def _set_win(s_spin: QDoubleSpinBox, e_spin: QDoubleSpinBox, win: HoldWindow | None) -> None:
                if win is None:
                    s_spin.setValue(0.0)
                    e_spin.setValue(0.0)
                else:
                    s_spin.setValue(float(win.start))
                    e_spin.setValue(float(win.end))

            _set_win(self.fit_ref_start, self.fit_ref_end, rule.fit_reference_window)
            _set_win(self.fit_prac_start, self.fit_prac_end, rule.fit_practice_window)
            _set_win(self.check_ref_start, self.check_ref_end, rule.check_reference_window)
            _set_win(self.check_prac_start, self.check_prac_end, rule.check_practice_window)
            self._fit_ref_stats = rule.fit_reference_stats
            self._fit_prac_stats = rule.fit_practice_stats
            self._check_ref_stats = rule.check_reference_stats
            self._check_prac_stats = rule.check_practice_stats
            self._verified = bool(rule.verified)
            self._verification_error = rule.verification_error
            self._status_message = rule.status_message
            self._refresh_hold_summary()
            self._sync_visibility()
        finally:
            for w, state in zip(widgets, blocked, strict=True):
                w.blockSignals(state)
        self._dirty = False
        self.update_preview()

    def _refresh_hold_summary(self) -> None:
        parts: list[str] = []
        for name, stats in (
            ("Fit Ref", self._fit_ref_stats),
            ("Fit Prac", self._fit_prac_stats),
            ("Check Ref", self._check_ref_stats),
            ("Check Prac", self._check_prac_stats),
        ):
            if stats is not None:
                parts.append(
                    f"{name}: {stats.median:.2f} ({stats.valid_frames}/{stats.total_frames}, {stats.coverage_ratio * 100:.0f}%)"
                )
        if parts:
            header = f"Offset {_format_signed_offset(float(self.offset_spin.value()), self._unit)}"
            if self._status_message:
                header += f" · {self._status_message}"
            self.hold_summary_lbl.setText(f"{header}\n" + " | ".join(parts))
        elif self._status_message:
            self.hold_summary_lbl.setText(self._status_message)
        else:
            self.hold_summary_lbl.setText("Hold quality: configure Fit and Check windows and compute")

    def update_preview(self) -> None:
        """Compute live one-value preview (`Value on reference scale`) from `preview_input`."""
        raw = self.preview_input.value()
        if raw is None:
            self.preview_output_lbl.setText("Value on reference scale: — (blank preview input)")
            self.preview_output_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
            return
        if not self.adjust_radio.isChecked():
            inside = self.target.lower <= raw <= self.target.upper
            zone = "inside range" if inside else "outside range"
            self.preview_output_lbl.setText(
                f"Direct comparison: {_format_metric_value(raw, self._unit)} ({zone})"
            )
            self.preview_output_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
            return
        offset = float(self.offset_spin.value())
        method = str(self.method_combo.currentData() or "manual")
        if method == "matched_holds" and not self._verified:
            self.preview_output_lbl.setText("Needs recheck: compute and pass Check hold verification first")
            self.preview_output_lbl.setStyleSheet(f"color: {COLOR_WARNING};")
            return
        adjusted = apply_metric_offset(self.metric, raw, offset)
        if not in_physical_domain(self.metric, adjusted):
            self.preview_output_lbl.setText(
                f"Adjustment unavailable: {_format_metric_value(raw, self._unit)} {_format_signed_offset(offset, self._unit)} "
                f"falls outside valid physical domain for {METRIC_DETAILS[self.metric][1]}"
            )
            self.preview_output_lbl.setStyleSheet(f"color: {COLOR_DANGER}; font-weight: 600;")
            return
        inside = self.target.lower <= adjusted <= self.target.upper
        zone = (
            f"inside {self.target.lower:.3g}..{self.target.upper:.3g}"
            if inside
            else f"outside {self.target.lower:.3g}..{self.target.upper:.3g}"
        )
        suffix = " (Manual · unverified)" if method == "manual" else " (Verified)"
        self.preview_output_lbl.setText(
            f"Value on reference scale: {_format_metric_value(adjusted, self._unit)} ({zone}){suffix}"
        )
        self.preview_output_lbl.setStyleSheet(
            f"color: {COLOR_SUCCESS if inside else COLOR_COPPER}; font-weight: 600;"
        )


class MeasurementNormalizationPanel(QFrame):
    """Settings card managing global options, pair selection, and per-metric rules."""

    config_modified = Signal()

    def __init__(self, bridge: ServiceBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bridge = bridge
        self.setProperty("class", "card")

        self._pairs: dict[str, NormalizationPairRule] = {}
        self._profiles_by_id: dict[str, tuple[Path, RangeProfile]] = {}
        self._practices_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        self._ref_source_overrides: dict[str, Path] = {}
        self._review_context: dict[str, Any] = {}
        self._metric_rows: dict[str, MetricNormalizationRowWidget] = {}
        self._active_request_id = 0
        self._cancel_event = threading.Event()
        self._compute_thread: threading.Thread | None = None
        self._signals = _HoldComputeSignals(self)
        self._signals.finished.connect(
            self._on_compute_finished, Qt.ConnectionType.QueuedConnection
        )
        self._loading = False
        self._current_pair_key: tuple[str, tuple[str, int]] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # Master enable & clinical scope banner
        top_row = QHBoxLayout()
        self.enable_cb = QCheckBox("Enable measurement normalization")
        self.enable_cb.toggled.connect(self._on_top_option_changed)
        top_row.addWidget(self.enable_cb)
        top_row.addStretch(1)
        root.addLayout(top_row)

        disclaimer = QLabel(
            "Offsets adjust Review metrics onto an approved calibration's reference scale only to remove "
            "verified camera, costume, or anatomy neutral differences. Never normalize away student posture habits."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
        root.addWidget(disclaimer)

        # Hold quality global controls
        quality_box = QFrame()
        quality_box.setStyleSheet(f"border: 1px solid {COLOR_BORDER}; border-radius: 4px;")
        quality_form = QGridLayout(quality_box)
        quality_form.setContentsMargins(10, 8, 10, 8)
        quality_form.setHorizontalSpacing(12)
        quality_form.addWidget(QLabel("<b>Hold quality defaults</b>"), 0, 0, 1, 6)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 500)
        self.min_samples_spin.setValue(8)
        self.min_samples_spin.valueChanged.connect(self._on_top_option_changed)
        quality_form.addWidget(QLabel("Minimum valid frames:"), 1, 0)
        quality_form.addWidget(self.min_samples_spin, 1, 1)

        self.min_coverage_spin = QDoubleSpinBox()
        self.min_coverage_spin.setRange(0.0, 1.0)
        self.min_coverage_spin.setDecimals(2)
        self.min_coverage_spin.setSingleStep(0.05)
        self.min_coverage_spin.setValue(0.60)
        self.min_coverage_spin.valueChanged.connect(self._on_top_option_changed)
        quality_form.addWidget(QLabel("Minimum hold coverage:"), 1, 2)
        quality_form.addWidget(self.min_coverage_spin, 1, 3)

        self.default_max_error_spin = QDoubleSpinBox()
        self.default_max_error_spin.setRange(0.001, 180.0)
        self.default_max_error_spin.setDecimals(3)
        self.default_max_error_spin.setSingleStep(0.1)
        self.default_max_error_spin.setValue(1.0)
        self.default_max_error_spin.valueChanged.connect(self._on_tolerance_changed)
        quality_form.addWidget(QLabel("Default max check error:"), 1, 4)
        quality_form.addWidget(self.default_max_error_spin, 1, 5)
        root.addWidget(quality_box)

        # Calibration + Recording selector form
        pair_form = QFormLayout()
        pair_form.setSpacing(6)
        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._on_selection_changed)
        pair_form.addRow("Calibration profile (.tpc):", self.profile_combo)

        self.practice_combo = QComboBox()
        self.practice_combo.currentIndexChanged.connect(self._on_selection_changed)
        pair_form.addRow("Recording to normalize:", self.practice_combo)
        root.addLayout(pair_form)

        pair_actions = QHBoxLayout()
        pair_actions.setSpacing(8)
        self.pair_enabled_cb = QCheckBox("Apply this adjustment in Review")
        self.pair_enabled_cb.setChecked(True)
        self.pair_enabled_cb.toggled.connect(self._on_pair_option_changed)
        pair_actions.addWidget(self.pair_enabled_cb)

        self.locate_reference_btn = QPushButton("Locate reference source…")
        self.locate_reference_btn.clicked.connect(self._locate_reference_source)
        pair_actions.addWidget(self.locate_reference_btn)

        self.compute_all_btn = QPushButton("Compute && verify holds")
        self.compute_all_btn.clicked.connect(lambda: self.compute_matched_holds_async(None))
        pair_actions.addWidget(self.compute_all_btn)

        self.remove_pair_btn = QPushButton("Remove comparison")
        self.remove_pair_btn.clicked.connect(self._remove_current_pair)
        pair_actions.addWidget(self.remove_pair_btn)
        pair_actions.addStretch(1)
        root.addLayout(pair_actions)

        self.pair_status_lbl = QLabel("Select a calibration profile and recording to configure metric normalization.")
        self.pair_status_lbl.setWordWrap(True)
        self.pair_status_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
        root.addWidget(self.pair_status_lbl)

        self.metrics_host = QWidget()
        self.metrics_layout = QVBoxLayout(self.metrics_host)
        self.metrics_layout.setContentsMargins(0, 0, 0, 0)
        self.metrics_layout.setSpacing(8)
        root.addWidget(self.metrics_host)

    @property
    def metric_rows(self) -> dict[str, MetricNormalizationRowWidget]:
        return dict(self._metric_rows)

    def _on_top_option_changed(self) -> None:
        if self._loading:
            return
        self.config_modified.emit()

    def _on_tolerance_changed(self) -> None:
        if self._loading:
            return
        for row in self._metric_rows.values():
            row._update_status_badge()
            row.update_preview()
        self.config_modified.emit()

    def _on_pair_option_changed(self) -> None:
        if self._loading:
            return
        self._commit_visible_pair(force_create=True)
        self.config_modified.emit()

    def _current_review_context(self) -> dict[str, Any]:
        merged = dict(getattr(self.bridge, "review_context", {}) or {})
        merged.update(self._review_context)
        return merged

    def _current_playhead(self) -> float | None:
        val = self._current_review_context().get("playhead_seconds")
        return float(val) if isinstance(val, (int, float)) and math.isfinite(val) else None

    def _current_frame_raw(self, metric: str) -> float | None:
        vals = self._current_review_context().get("frame_raw_values")
        if isinstance(vals, Mapping):
            raw = vals.get(metric)
            if isinstance(raw, (int, float)) and math.isfinite(raw):
                return float(raw)
        return None

    def refresh_library_choices(self) -> None:
        """Scan Library `.tpc` profiles and `.tp`/`.tpr` recordings without losing active selection."""
        current_profile_id = self.profile_combo.currentData()
        current_practice_key = self.practice_combo.currentData()

        self._profiles_by_id.clear()
        for ref in self.bridge.list_references():
            if not ref.is_archive or not ref.path.is_file():
                continue
            try:
                profile = read_range_profile(ref.path)
            except (OSError, ValueError):
                continue
            self._profiles_by_id[profile.id] = (ref.path, profile)

        self._practices_by_key.clear()
        for item in self.bridge.list_library_archives():
            kind = item.get("kind")
            if kind not in (ArchiveKind.PRACTICE, ArchiveKind.REFERENCE):
                continue
            archive_path = Path(item["path"])
            seg_count = int(item.get("segments_count") or 0)
            name = str(item.get("name") or archive_path.stem)
            for seg_idx in range(max(1, seg_count)):
                key = (str(archive_path.resolve()), seg_idx)
                label = f"{name}" if seg_count <= 1 else f"{name} · Segment {seg_idx + 1}"
                if kind == ArchiveKind.REFERENCE:
                    label += " (Reference clip)"
                self._practices_by_key[key] = {
                    "path": archive_path.resolve(),
                    "segment_index": seg_idx,
                    "label": label,
                    "name": name,
                }

        blocked_prof = self.profile_combo.blockSignals(True)
        blocked_prac = self.practice_combo.blockSignals(True)
        try:
            self.profile_combo.clear()
            for cal_id, (_path, profile) in self._profiles_by_id.items():
                self.profile_combo.addItem(
                    f"{profile.name} ({profile.capture_profile.get('side', '?')})",
                    cal_id,
                )
            if current_profile_id is not None:
                idx = self.profile_combo.findData(current_profile_id)
                if idx >= 0:
                    self.profile_combo.setCurrentIndex(idx)

            self.practice_combo.clear()
            for key, info in self._practices_by_key.items():
                self.practice_combo.addItem(info["label"], key)
            if current_practice_key is not None:
                idx = self.practice_combo.findData(current_practice_key)
                if idx >= 0:
                    self.practice_combo.setCurrentIndex(idx)
        finally:
            self.profile_combo.blockSignals(blocked_prof)
            self.practice_combo.blockSignals(blocked_prac)

    def set_config(self, cfg: Config) -> None:
        """Populate panel widgets and stored pairs from `cfg.measurement_normalization`."""
        self._loading = True
        try:
            norm = cfg.measurement_normalization
            self._pairs = {pair.id: pair.model_copy(deep=True) for pair in norm.pairs}
            self.enable_cb.setChecked(bool(norm.enabled))
            self.min_samples_spin.setValue(int(norm.min_hold_samples))
            self.min_coverage_spin.setValue(float(norm.min_hold_coverage))
            self.default_max_error_spin.setValue(float(norm.default_max_verification_error))
            self.refresh_library_choices()
            if self.profile_combo.currentIndex() < 0 and self.profile_combo.count() > 0:
                self.profile_combo.setCurrentIndex(0)
            if self.practice_combo.currentIndex() < 0 and self.practice_combo.count() > 0:
                self.practice_combo.setCurrentIndex(0)
            self._load_selected_pair()
        finally:
            self._loading = False

    def select_context(self, payload: Mapping[str, Any]) -> None:
        """Focus normalization controls and preselect profile/recording from Review payload."""
        self._review_context = dict(payload)
        self.refresh_library_choices()
        cal_id = payload.get("calibration_id")
        cal_path = payload.get("calibration_path")
        archive_path = payload.get("archive_path")
        segment_index = int(payload.get("segment_index") or 0)

        self._loading = True
        try:
            if isinstance(cal_id, str) and cal_id:
                idx = self.profile_combo.findData(cal_id)
                if idx >= 0:
                    self.profile_combo.setCurrentIndex(idx)
            elif isinstance(cal_path, (Path, str)):
                target_path = Path(cal_path).resolve()
                for idx in range(self.profile_combo.count()):
                    pid = self.profile_combo.itemData(idx)
                    if pid in self._profiles_by_id and self._profiles_by_id[pid][0].resolve() == target_path:
                        self.profile_combo.setCurrentIndex(idx)
                        break

            if isinstance(archive_path, (Path, str)):
                key = (str(Path(archive_path).resolve()), segment_index)
                idx = self.practice_combo.findData(key)
                if idx >= 0:
                    self.practice_combo.setCurrentIndex(idx)
            self._load_selected_pair()
        finally:
            self._loading = False

    def _on_selection_changed(self) -> None:
        if self._loading:
            return
        self._commit_visible_pair(force_create=False)
        self._active_request_id += 1
        self._cancel_event.set()
        self._load_selected_pair()

    def _selected_profile_and_practice(
        self,
    ) -> tuple[str, Path, RangeProfile, Path, int, str] | None:
        cal_id = self.profile_combo.currentData()
        prac_key = self.practice_combo.currentData()
        if not isinstance(cal_id, str) or cal_id not in self._profiles_by_id:
            return None
        if not isinstance(prac_key, tuple) or prac_key not in self._practices_by_key:
            return None
        cal_path, profile = self._profiles_by_id[cal_id]
        prac_info = self._practices_by_key[prac_key]
        return cal_id, cal_path, profile, prac_info["path"], prac_info["segment_index"], prac_info["label"]

    def _find_working_pair(self, cal_id: str, prac_path: Path, segment_index: int) -> NormalizationPairRule | None:
        raw_sha = resolve_raw_media_sha256(prac_path, segment_index)
        for pair in self._pairs.values():
            if pair.calibration_id != cal_id:
                continue
            if raw_sha and pair.practice_source.raw_media_sha256:
                if pair.practice_source.raw_media_sha256 == raw_sha:
                    return pair
            elif pair.practice_source.segment_index == segment_index and pair.practice_source.archive_path:
                try:
                    if Path(pair.practice_source.archive_path).resolve() == prac_path.resolve():
                        return pair
                except OSError:
                    continue
        return None

    def _clear_metric_rows(self) -> None:
        self._metric_rows.clear()
        while self.metrics_layout.count():
            item = self.metrics_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_current_practice_identity(
        self, prac_path: Path, segment_index: int, cfg: Config, *, capture_override: Mapping[str, Any] | None = None,
    ) -> HoldSourceIdentity:
        raw_sha = resolve_raw_media_sha256(prac_path, segment_index)
        analytics = read_analytics(prac_path, segment_index) or {}
        try:
            obs, meta = select_archive_frames(prac_path, segment_index, cfg, prefer_coach_frames=True)
            return build_source_identity(
                archive_path=prac_path,
                segment_index=segment_index,
                cfg=cfg,
                observations=obs,
                capture_profile=meta.get("capture_profile") or capture_override or {},
                raw_media_sha256=meta.get("raw_media_sha256") or raw_sha,
            )
        except (OSError, ValueError, TypeError):
            capture = analytics.get("capture_profile") or capture_override or {}
            return HoldSourceIdentity(
                archive_path=str(prac_path),
                archive_name=prac_path.name,
                segment_index=segment_index,
                raw_media_sha256=raw_sha,
                media_sha256=raw_sha,
                capture_identity={k: v for k, v in capture.items() if v is not None},
            )

    def _load_selected_pair(self) -> None:
        self._clear_metric_rows()
        sel = self._selected_profile_and_practice()
        if sel is None:
            self._current_pair_key = None
            self.pair_status_lbl.setText("Add or select both a calibration profile (.tpc) and a recording in Library.")
            return

        cal_id, cal_path, profile, prac_path, seg_idx, prac_label = sel
        self._current_pair_key = (cal_id, (str(prac_path.resolve()), seg_idx))
        existing = self._find_working_pair(cal_id, prac_path, seg_idx)

        cfg = self._current_effective_cfg()
        comp_cfg = cfg.model_copy(update={
            "side": profile.capture_profile.get("side", cfg.side),
            "facing": profile.capture_profile.get("facing", cfg.facing),
        })
        if existing is not None:
            self.pair_enabled_cb.setChecked(bool(existing.enabled))
            temp_cfg = comp_cfg.model_copy(update={
                "measurement_normalization": comp_cfg.measurement_normalization.model_copy(update={
                    "enabled": True,
                    "pairs": [existing],
                })
            })
            prac_id = self._build_current_practice_identity(prac_path, seg_idx, comp_cfg)
            applicability = evaluate_pair_applicability(temp_cfg, profile, prac_id)
            if applicability.status == "needs_recheck":
                self.pair_status_lbl.setText(f"Needs recheck: {applicability.reason}")
                self.pair_status_lbl.setStyleSheet(f"color: {COLOR_DANGER}; font-size: 12px; font-weight: 600;")
            else:
                self.pair_status_lbl.setText(f"Configured comparison: {profile.name} <-> {prac_label}")
                self.pair_status_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
        else:
            self.pair_enabled_cb.setChecked(True)
            self.pair_status_lbl.setText(f"Direct comparison (no adjustments saved yet for {profile.name} <-> {prac_label})")
            self.pair_status_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")

        for key, target in profile.metrics.items():
            row = MetricNormalizationRowWidget(
                key,
                target,
                playhead_provider=self._current_playhead,
                review_frame_provider=self._current_frame_raw,
                default_tolerance_provider=lambda: float(self.default_max_error_spin.value()),
                parent=self.metrics_host,
            )
            if existing is not None and key in existing.metrics:
                row.set_rule(existing.metrics[key])
            else:
                row.set_rule(MetricNormalizationRule())
            row.changed.connect(self._on_row_changed)
            row.compute_requested.connect(lambda m: self.compute_matched_holds_async(m))
            self._metric_rows[key] = row
            self.metrics_layout.addWidget(row)

    def _current_effective_cfg(self) -> Config:
        norm = MeasurementNormalizationConfig(
            enabled=self.enable_cb.isChecked(),
            min_hold_samples=int(self.min_samples_spin.value()),
            min_hold_coverage=float(self.min_coverage_spin.value()),
            default_max_verification_error=float(self.default_max_error_spin.value()),
            pairs=list(self._pairs.values()),
        )
        return self.bridge.config.model_copy(update={"measurement_normalization": norm})

    def _on_row_changed(self) -> None:
        if self._loading:
            return
        self._commit_visible_pair(force_create=False)
        self.config_modified.emit()

    def _commit_visible_pair(self, *, force_create: bool) -> NormalizationPairRule | None:
        sel = self._selected_profile_and_practice()
        if sel is None or not self._metric_rows:
            return None
        cal_id, cal_path, profile, prac_path, seg_idx, prac_label = sel
        existing = self._find_working_pair(cal_id, prac_path, seg_idx)

        rules = {key: row.to_rule() for key, row in self._metric_rows.items()}
        any_dirty = any(row.is_dirty for row in self._metric_rows.values())
        any_active_or_custom = any(
            r.enabled or abs(r.offset) > 0.0 or r.fit_reference_window is not None or r.verified
            for r in rules.values()
        )
        if existing is None and not force_create and not (any_dirty and any_active_or_custom):
            return None

        cfg = self._current_effective_cfg()
        comp_cfg = cfg.model_copy(update={
            "side": profile.capture_profile.get("side", cfg.side),
            "facing": profile.capture_profile.get("facing", cfg.facing),
        })
        prac_id = self._build_current_practice_identity(prac_path, seg_idx, comp_cfg)
        ref_identity = existing.reference_source if existing is not None else HoldSourceIdentity()
        override = self._ref_source_overrides.get(cal_id)
        if override is not None or not ref_identity.frames_fingerprint:
            try:
                ref_path = resolve_profile_reference_source(profile, cal_path, comp_cfg, override_path=override)
                ref_obs, ref_meta = select_archive_frames(ref_path, 0, comp_cfg, prefer_coach_frames=False)
                ref_identity = build_source_identity(
                    archive_path=ref_path,
                    segment_index=0,
                    cfg=comp_cfg,
                    observations=ref_obs,
                    capture_profile=ref_meta.get("capture_profile") or profile.capture_profile,
                    raw_media_sha256=ref_meta.get("raw_media_sha256"),
                )
            except (OSError, ValueError):
                pass

        pair_id = existing.id if existing is not None else make_pair_rule_id(
            cal_id, prac_id.raw_media_sha256, f"{prac_path}:{seg_idx}",
        )
        now = utc_timestamp_now()
        pair = NormalizationPairRule(
            id=pair_id,
            name=f"{profile.name} · {prac_label}",
            enabled=self.pair_enabled_cb.isChecked(),
            calibration_id=cal_id,
            calibration_name=profile.name,
            profile_content_digest=profile_content_digest(profile),
            reference_source=ref_identity,
            practice_source=prac_id,
            metrics=rules,
            created_at_utc=existing.created_at_utc or now if existing else now,
            updated_at_utc=now,
        )
        self._pairs[pair.id] = pair
        for row in self._metric_rows.values():
            row.mark_clean()
        return pair

    def _remove_current_pair(self) -> None:
        sel = self._selected_profile_and_practice()
        if sel is None:
            return
        cal_id, _cal_path, _profile, prac_path, seg_idx, _prac_label = sel
        existing = self._find_working_pair(cal_id, prac_path, seg_idx)
        if existing is not None and existing.id in self._pairs:
            del self._pairs[existing.id]
        self._load_selected_pair()
        self.config_modified.emit()

    def _locate_reference_source(self) -> None:
        sel = self._selected_profile_and_practice()
        if sel is None:
            return
        cal_id, cal_path, profile, _prac_path, _seg_idx, _prac_label = sel
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Locate Source Reference (.tpr)", "", "Tango Reference (*.tpr)",
        )
        if not chosen:
            return
        override = Path(chosen)
        cfg = self._current_effective_cfg()
        comp_cfg = cfg.model_copy(update={
            "side": profile.capture_profile.get("side", cfg.side),
            "facing": profile.capture_profile.get("facing", cfg.facing),
        })
        try:
            resolve_profile_reference_source(profile, cal_path, comp_cfg, override_path=override)
        except (OSError, ValueError) as error:
            self.pair_status_lbl.setText(f"Could not use reference source: {error}")
            self.pair_status_lbl.setStyleSheet(f"color: {COLOR_DANGER}; font-size: 12px;")
            return
        self._ref_source_overrides[cal_id] = override
        self.pair_status_lbl.setText(f"Reference source located: {override.name}")
        self.pair_status_lbl.setStyleSheet(f"color: {COLOR_SUCCESS}; font-size: 12px;")

    def _load_frames_for_compute(
        self,
        cal_id: str,
        cal_path: Path,
        profile: RangeProfile,
        prac_path: Path,
        seg_idx: int,
        comp_cfg: Config,
        override_path: Path | None,
    ) -> tuple[Any, dict[str, Any], Any, dict[str, Any]]:
        ref_path = resolve_profile_reference_source(
            profile, cal_path, comp_cfg, override_path=override_path,
        )
        ref_obs, ref_meta = select_archive_frames(ref_path, 0, comp_cfg, prefer_coach_frames=False)
        prac_obs, prac_meta = select_archive_frames(prac_path, seg_idx, comp_cfg, prefer_coach_frames=True)
        return ref_obs, {**ref_meta, "path": ref_path}, prac_obs, {**prac_meta, "path": prac_path}

    def compute_matched_holds_async(self, target_metric: str | None = None) -> None:
        """Compute matched-hold offsets in a background thread with snapshot cancellation."""
        sel = self._selected_profile_and_practice()
        if sel is None:
            return
        cal_id, cal_path, profile, prac_path, seg_idx, prac_label = sel

        metrics_to_compute: list[tuple[str, MetricNormalizationRule]] = []
        for key, row in self._metric_rows.items():
            if target_metric is not None and key != target_metric:
                continue
            rule = row.to_rule()
            if target_metric is None and (not rule.enabled or rule.method != "matched_holds"):
                continue
            metrics_to_compute.append((key, rule))
        if not metrics_to_compute:
            self.pair_status_lbl.setText("Enable 'Use adjustment' with 'Matched holds' on at least one metric to compute.")
            return

        cfg = self._current_effective_cfg()
        comp_cfg = cfg.model_copy(update={
            "side": profile.capture_profile.get("side", cfg.side),
            "facing": profile.capture_profile.get("facing", cfg.facing),
        })
        override_path = self._ref_source_overrides.get(cal_id)

        self._active_request_id += 1
        req_id = self._active_request_id
        self._cancel_event.set()
        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        snapshot_key = (cal_id, str(prac_path.resolve()), seg_idx)

        self.pair_status_lbl.setText("Computing and verifying matched holds…")
        self.pair_status_lbl.setStyleSheet(f"color: {COLOR_COPPER}; font-size: 12px;")

        def _worker() -> None:
            try:
                if cancel_event.is_set():
                    return
                ref_obs, ref_meta, prac_obs, prac_meta = self._load_frames_for_compute(
                    cal_id, cal_path, profile, prac_path, seg_idx, comp_cfg, override_path,
                )
                if cancel_event.is_set():
                    return
                results: dict[str, MatchedHoldResult] = {}
                for key, rule in metrics_to_compute:
                    if cancel_event.is_set():
                        return
                    target = profile.metrics[key]
                    res = evaluate_matched_holds(
                        metric=key,
                        target=target,
                        rule=rule.model_copy(update={"enabled": True}),
                        reference_frames=ref_obs,
                        practice_frames=prac_obs,
                        cfg=comp_cfg,
                    )
                    results[key] = res
                ref_identity = build_source_identity(
                    archive_path=Path(ref_meta["path"]),
                    segment_index=0,
                    cfg=comp_cfg,
                    observations=ref_obs,
                    capture_profile=ref_meta.get("capture_profile") or profile.capture_profile,
                    raw_media_sha256=ref_meta.get("raw_media_sha256"),
                )
                prac_identity = build_source_identity(
                    archive_path=prac_path,
                    segment_index=seg_idx,
                    cfg=comp_cfg,
                    observations=prac_obs,
                    capture_profile=prac_meta.get("capture_profile") or {},
                    raw_media_sha256=prac_meta.get("raw_media_sha256"),
                )
                self._signals.finished.emit(
                    req_id,
                    {"key": snapshot_key, "cal_id": cal_id, "profile": profile, "prac_label": prac_label, "ref_identity": ref_identity, "prac_identity": prac_identity},
                    results,
                    {},
                    "",
                )
            except Exception as error:  # noqa: BLE001
                self._signals.finished.emit(req_id, {"key": snapshot_key}, {}, {}, str(error))

        thread = threading.Thread(target=_worker, daemon=True)
        self._compute_thread = thread
        thread.start()

    def _on_compute_finished(
        self,
        request_id: int,
        context: dict[str, Any],
        results: dict[str, MatchedHoldResult],
        _extra: dict[str, Any],
        error_msg: str,
    ) -> None:
        if request_id != self._active_request_id:
            return
        sel = self._selected_profile_and_practice()
        if sel is None:
            return
        cal_id, _cal_path, _profile, prac_path, seg_idx, _prac_label = sel
        if context.get("key") != (cal_id, str(prac_path.resolve()), seg_idx):
            return
        if error_msg:
            self.pair_status_lbl.setText(f"Could not compute holds: {error_msg}")
            self.pair_status_lbl.setStyleSheet(f"color: {COLOR_DANGER}; font-size: 12px;")
            return

        any_verified = False
        any_failed = False
        for key, res in results.items():
            row = self._metric_rows.get(key)
            if row is not None:
                row.set_rule(res.rule)
                row._dirty = True
            if res.eligible:
                any_verified = True
            else:
                any_failed = True

        pair = self._commit_visible_pair(force_create=True)
        if pair is not None and "ref_identity" in context and "prac_identity" in context:
            self._pairs[pair.id] = pair.model_copy(update={
                "reference_source": context["ref_identity"],
                "practice_source": context["prac_identity"],
            })

        if any_verified and not any_failed:
            self.pair_status_lbl.setText("Matched holds computed and verified on Check hold.")
            self.pair_status_lbl.setStyleSheet(f"color: {COLOR_SUCCESS}; font-size: 12px; font-weight: 600;")
        elif any_failed:
            self.pair_status_lbl.setText("Verification check failed or hold quality gate unmet (Needs recheck).")
            self.pair_status_lbl.setStyleSheet(f"color: {COLOR_DANGER}; font-size: 12px; font-weight: 600;")
        self.config_modified.emit()

    def measurement_normalization_config(self) -> MeasurementNormalizationConfig:
        """Collect validated `MeasurementNormalizationConfig` including any visible pair edits."""
        self._commit_visible_pair(force_create=False)
        return MeasurementNormalizationConfig(
            enabled=self.enable_cb.isChecked(),
            min_hold_samples=int(self.min_samples_spin.value()),
            min_hold_coverage=float(self.min_coverage_spin.value()),
            default_max_verification_error=float(self.default_max_error_spin.value()),
            pairs=[pair.model_copy(deep=True) for pair in self._pairs.values()],
        )
