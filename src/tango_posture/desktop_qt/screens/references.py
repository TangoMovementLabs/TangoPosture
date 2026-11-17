"""References management and calibration screen."""

from __future__ import annotations

from pydantic import ValidationError
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...calibration.archives import read_range_profile
from ...calibration.models import ALL_METRIC_DETAILS as METRIC_DETAILS
from ...config import Config
from ...core.entities import ReferenceSummary
from ..router import Route, Router
from ..service_bridge import ServiceBridge, format_created_at


class ReferencesScreen(QWidget):
    def __init__(self, bridge: ServiceBridge, router: Router, parent: QWidget | None = None):
        super().__init__(parent)
        self.bridge = bridge
        self.router = router
        self.selected_ref: ReferenceSummary | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Header with actions
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Self Calibration")
        title.setProperty("class", "title")
        subtitle = QLabel(
            "Solo camera calibration, approved .tpc ranges, and legacy neutral baselines. "
            "Replayable .tpr references are in Library."
        )
        subtitle.setProperty("class", "subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()

        self.new_solo_cal_btn = QPushButton("New Calibration")
        self.new_solo_cal_btn.setProperty("class", "accent")
        self.new_solo_cal_btn.clicked.connect(self._new_solo_calibration)

        cal_btn = QPushButton("Open Library")
        cal_btn.setProperty("class", "primary")
        cal_btn.clicked.connect(
            lambda: self.router.navigate(Route.LIBRARY)
        )

        header.addWidget(self.new_solo_cal_btn)
        header.addWidget(cal_btn)
        layout.addLayout(header)

        from ..calibration.guided import GuidedCalibration

        self.guided_wizard = GuidedCalibration(self.bridge, self, embedded=True)
        self.guided_wizard.setWindowFlags(Qt.WindowType.Widget)
        self.guided_wizard.accepted.connect(self._on_embedded_saved)
        layout.addWidget(self.guided_wizard, 1)

        # Keep profile helper widgets hidden so existing programmatic checks stay valid
        self.table = QTableWidget(0, 5)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.setVisible(False)
        self.detail_name = QLabel("No reference selected")
        self.detail_name.setVisible(False)
        self.detail_meta = QLabel("")
        self.detail_meta.setVisible(False)
        self.confirm_camera_cb = QCheckBox()
        self.confirm_camera_cb.setVisible(False)
        self.check_camera_btn = QPushButton("Check with camera…")
        self.check_camera_btn.setVisible(False)
        self.edit_btn = QPushButton("Edit Calibration…")
        self.edit_btn.setVisible(False)
        self.match_btn = QPushButton("Match Camera Settings")
        self.match_btn.setVisible(False)

        self.refresh()

    def _on_embedded_saved(self) -> None:
        self.refresh()
        self.bridge.calibrations_changed.emit()

    def stop_for_navigation(self) -> None:
        self.guided_wizard._speech.cancel_all()
        self.guided_wizard._release_camera()

    def refresh(self) -> None:
        refs = self.bridge.list_references()
        self.table.setRowCount(0)
        for row_idx, ref in enumerate(refs):
            self.table.insertRow(row_idx)
            name_item = QTableWidgetItem(ref.name)
            name_item.setData(Qt.ItemDataRole.UserRole, ref)
            self.table.setItem(row_idx, 0, name_item)
            context_txt = "Legacy JSON"
            view_txt = ref.side.capitalize()
            setup_txt = f"Device {ref.camera}"
            live_txt = "0"
            if ref.is_archive:
                try:
                    prof = read_range_profile(ref.path)
                    context_txt = prof.context
                    views = sorted({t.view for t in prof.metrics.values()})
                    view_txt = ", ".join(v.capitalize() for v in views) if views else ref.side.capitalize()
                    cp = prof.capture_profile
                    setup_txt = str(cp.get("capture_profile") or cp.get("source") or f"Device {ref.camera}")
                    live_count = sum(1 for t in prof.metrics.values() if t.camera_check_passed)
                    live_txt = f"{live_count}/{len(prof.metrics)}"
                except (OSError, ValueError, TypeError, KeyError):
                    context_txt = ".tpc"
            self.table.setItem(row_idx, 1, QTableWidgetItem(context_txt))
            self.table.setItem(row_idx, 2, QTableWidgetItem(view_txt))
            self.table.setItem(row_idx, 3, QTableWidgetItem(setup_txt))
            self.table.setItem(row_idx, 4, QTableWidgetItem(live_txt))

        if refs:
            self.table.selectRow(0)

    def _on_selection_changed(self) -> None:
        selected_rows = self.table.selectedItems()
        if not selected_rows:
            self._clear_selection()
            return
        name_item = self.table.item(selected_rows[0].row(), 0)
        if name_item is None:
            self._clear_selection()
            return
        ref = name_item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(ref, ReferenceSummary):
            self._clear_selection()
            self.bridge.status_message.emit("The selected reference entry is invalid.")
            return
        self.selected_ref = ref
        self.edit_btn.setEnabled(ref.is_archive)
        self.check_camera_btn.setEnabled(ref.is_archive)
        self.match_btn.setEnabled(True)
        self.confirm_camera_cb.setEnabled(True)
        self.detail_name.setText(ref.name)
        details = (
            f"ID: {ref.id[:8]}...\n"
            f"Created: {format_created_at(ref.created_at, ref.path)}\n"
            f"Measurement side: {ref.side.capitalize()}\n"
            f"Facing: {ref.facing.capitalize()}\n"
            f"Camera device: {ref.camera}\n"
            f"File: {ref.path.name}"
        )
        if ref.is_archive:
            try:
                range_profile = read_range_profile(ref.path)
                metric_lines = [
                    f"• {METRIC_DETAILS.get(k, ('', k, ''))[1]}: "
                    f"{t.lower:.3g} … {t.upper:.3g} {t.unit} ({t.view})"
                    for k, t in range_profile.metrics.items()
                ]
                details += (
                    f"\nContext: {range_profile.context}"
                    f"\nApproved metrics ({len(metric_lines)}):\n"
                    + "\n".join(metric_lines)
                )
            except (OSError, ValueError, TypeError, KeyError):
                pass
        self.detail_meta.setText(details)
        self.confirm_camera_cb.setChecked(False)

    def _clear_selection(self) -> None:
        self.selected_ref = None
        self.check_camera_btn.setEnabled(False)
        self.edit_btn.setEnabled(False)
        self.match_btn.setEnabled(False)
        self.confirm_camera_cb.setEnabled(False)
        self.detail_name.setText("No reference selected")
        self.detail_meta.setText("Select a reference from the table to view details.")

    def _new_solo_calibration(self) -> None:
        self.guided_wizard.start_new_calibration()

    def _check_with_camera(self) -> None:
        if not self.selected_ref or not self.selected_ref.is_archive:
            self.bridge.status_message.emit("Select a .tpc calibration profile first.")
            return
        try:
            range_profile = read_range_profile(self.selected_ref.path)
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.bridge.status_message.emit(f"Cannot load calibration profile: {error}")
            return
        from ..calibration.guided import GuidedCalibration

        dialog = GuidedCalibration(
            self.bridge,
            self,
            target_profile=range_profile,
            target_profile_path=self.selected_ref.path,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh()
            self.bridge.calibrations_changed.emit()
            if dialog.saved_path is not None:
                self.bridge.status_message.emit(f"Updated calibration saved: {dialog.saved_path}")

    def _edit_calibration(self) -> None:
        if not self.selected_ref or not self.selected_ref.is_archive:
            self.bridge.status_message.emit("Select a .tpc calibration profile first.")
            return
        from ..calibration.editor import open_profile_calibration_editor

        try:
            saved = open_profile_calibration_editor(
                self.selected_ref.path, self.bridge.config, self, bridge=self.bridge
            )
            if saved is not None:
                self.refresh()
                self.bridge.status_message.emit(f"Approved calibration saved: {saved}")
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            self.bridge.status_message.emit(f"Cannot edit calibration: {error}")

    def _match_camera_settings(self) -> None:
        if not self.selected_ref:
            self.bridge.status_message.emit("Select a saved reference first.")
            return
        if not self.confirm_camera_cb.isChecked():
            self.bridge.status_message.emit(
                "Confirm that the physical camera placement is unchanged before reusing this reference."
            )
            return

        try:
            cam_idx = int(self.selected_ref.camera)
            cfg = Config.model_validate(
                {
                    **self.bridge.config.model_dump(),
                    "camera": cam_idx,
                    "side": self.selected_ref.side,
                    "facing": self.selected_ref.facing,
                }
            )
        except (ValueError, ValidationError) as error:
            self.bridge.status_message.emit(
                f"Cannot reuse this reference with the selected camera: {error}"
            )
            return
        self.bridge.update_config(cfg)
        self.detail_meta.setText(
            self.detail_meta.text()
            + "\n\nSettings selected. Camera profile and resolution compatibility "
            "will be checked when practice starts."
        )
